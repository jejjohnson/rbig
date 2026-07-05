"""Sensitive-attribute removal at chosen strength (issue #131).

:class:`RBIGFairTransformer` removes information about a sensitive
attribute ``A`` from the features, with four strategies of increasing
strength:

- ``"projection"`` — remove the single A-correlated direction (linear;
  ``A`` needed only at fit).
- ``"subspace"`` — remove the top ``n_components`` A-predictive linear
  directions by iterated regression + null-space projection.  Design
  note (deviation from the original issue sketch, on purpose): the
  directions come from *linear* A-regressions rather than a polynomial
  one — a nonlinear regression has no single X-space direction to
  project out, and nonlinear leakage is exactly what ``"transport"``
  is for.
- ``"transport"`` — per-group flow into a pooled reference flow:
  ``x → h⁻¹(g_a(x))`` matches the full distribution across groups
  (``p(X_fair | A=0) ≈ p(X_fair | A=1)``), catching variance/shape
  leakage no projection can.
- ``"conditional"`` — transport within Y-strata: removes ``I(X; A | Y)``
  while preserving the Y-signal.

**Shared rotation frames are load-bearing for transport.** The group
flows ``g_a`` reuse the reference flow's fitted rotations and refit only
the per-layer marginals on the group's data.  With *independently*
fitted group flows the transport still matches distributions, but PCA
sign/order flips between the group and reference frames scramble
per-sample coordinates — task signal in dimensions where the groups
already agree gets destroyed, and intermediate ``alpha`` blends can
*amplify* A-predictability.  With shared frames, equal marginals give an
identity map layer by layer, so ``h⁻¹(g_a(x)) ≈ x`` wherever the group
already matches the pooled distribution (a regression test pins this).

Tail-extended marginals are load-bearing here: the cross-group inverse
``h⁻¹`` is evaluated at latent points mapped from regions sparse in the
pooled fit, where clipped inverses degenerate (boundary notebook 18).

The ``alpha`` knob blends ``α·X_fair + (1−α)·X``.  No free lunch: when
``I(A; Y) > 0``, removing A-information necessarily removes Y-signal.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.marginal import MarginalGaussianize
from rbig._src.model import AnnealedRBIG

_STRATEGIES = ("projection", "subspace", "transport", "conditional")
# Beyond this many distinct values, A is almost certainly continuous and
# per-group flows would be fitted on slivers.
_MAX_GROUPS = 20


class _SharedRotationGroupFlow:
    """Forward-only group flow in the reference flow's rotation frames.

    Per reference layer, a fresh tail-extended marginal Gaussianization
    is fitted on the group's data and composed with the reference
    layer's *already fitted* rotation.  Sharing frames keeps the
    group→reference transport pointwise coherent (module docstring).
    """

    def __init__(self, reference: AnnealedRBIG, X_group: np.ndarray):
        self.layers_: list[tuple[MarginalGaussianize, object]] = []
        Xt = X_group.copy()  # (n_g, d) working copy through the layers
        for layer in reference.layers_:
            marginal = MarginalGaussianize(tail="gaussian").fit(Xt)
            Xt = layer.rotation.transform(marginal.transform(Xt))
            self.layers_.append((marginal, layer.rotation))

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        for marginal, rotation in self.layers_:
            Xt = rotation.transform(marginal.transform(Xt))
        return Xt


class RBIGFairTransformer(TransformerMixin, BaseEstimator):
    """Remove sensitive-attribute information from features.

    Parameters
    ----------
    strategy : {"projection", "subspace", "transport", "conditional"}, default "transport"
        Removal strength (see module docstring).
    alpha : float, default 1.0
        Fairness–utility knob: output is ``α·X_fair + (1−α)·X``.
    sensitive_col : int or None, default None
        Pipeline mode: ``A`` is this column of ``X``, consumed at fit
        and transform and dropped from the output.  When ``None``, pass
        ``A`` as a keyword (metadata-routing mode); ``projection`` and
        ``subspace`` then need it only at fit.
    n_components : int, default 1
        Number of directions removed by ``strategy="subspace"``.
    min_samples_per_stratum : int or None, default None
        Guard for ``conditional``'s ``(a, y)`` cells; ``None`` means
        ``20 · n_features``.  Cells below it fall back to the global
        (unconditional) transport flows with a warning — a tiny flow is
        never fitted silently.
    n_layers : int, default 40
        Layer budget of every flow.
    tol : float or "auto", default 1e-5
        Convergence tolerance of every flow.
    random_state : int or None, optional
        Seed for the flows.

    Attributes
    ----------
    sensitive_dir_ : np.ndarray of shape (d_f,)
        (projection) The removed unit direction.
    sensitive_subspace_ : np.ndarray of shape (k, d_f)
        (subspace) Orthonormal removed directions.
    group_flows_ : dict
        (transport/conditional) Per-group forward flows keyed by group
        value, sharing the reference flow's rotation frames.
    reference_flow_ : AnnealedRBIG
        (transport/conditional) Pooled reference flow.
    stratum_flows_ : dict
        (conditional) ``(group, y)``-cell flows sharing the stratum
        reference's frames; absent cells fall back to ``group_flows_``.
    stratum_refs_ : dict
        (conditional) Per-Y-stratum pooled reference flows.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGFairTransformer, make_variance_leak
    >>> X, meta = make_variance_leak(n_samples=400, seed=0)
    >>> XA = np.column_stack([meta["A"], X])  # A as column 0
    >>> fair = RBIGFairTransformer(
    ...     strategy="transport", sensitive_col=0, n_layers=5, random_state=0
    ... ).fit(XA)
    >>> fair.transform(XA).shape  # A column consumed and dropped
    (400, 4)
    """

    def __init__(
        self,
        strategy: str = "transport",
        alpha: float = 1.0,
        sensitive_col: int | None = None,
        n_components: int = 1,
        min_samples_per_stratum: int | None = None,
        n_layers: int = 40,
        tol: float | str = 1e-5,
        random_state: int | None = None,
    ):
        self.strategy = strategy
        self.alpha = alpha
        self.sensitive_col = sensitive_col
        self.n_components = n_components
        self.min_samples_per_stratum = min_samples_per_stratum
        self.n_layers = n_layers
        self.tol = tol
        self.random_state = random_state

    # ── internals ────────────────────────────────────────────────────────────

    def _flow(self, seed_offset: int = 0) -> AnnealedRBIG:
        seed = None if self.random_state is None else self.random_state + seed_offset
        return AnnealedRBIG(
            n_layers=self.n_layers,
            tol=self.tol,
            random_state=seed,
            marginal_kwargs={"tail": "gaussian"},
        )

    def _split_sensitive(
        self, X: np.ndarray, A
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Resolve (features, A): extract/drop ``sensitive_col`` or use kwarg.

        Returns the feature block F (X without the sensitive column in
        ``sensitive_col`` mode) and the per-sample attribute values, or
        ``None`` when no attribute source is available.
        """
        if self.sensitive_col is not None:
            idx = int(self.sensitive_col)
            if not -X.shape[1] <= idx < X.shape[1]:
                raise ValueError(
                    f"sensitive_col={idx} out of range for {X.shape[1]} columns."
                )
            if X.shape[1] < 2:
                raise ValueError(
                    "X needs at least 2 columns when sensitive_col consumes one."
                )
            a = X[:, idx]
            feat = np.delete(X, idx % X.shape[1], axis=1)  # F: (n, d-1)
            return feat, a
        if A is None:
            return X, None
        a = np.asarray(A).ravel()
        if a.shape[0] != X.shape[0]:
            raise ValueError(f"A has {a.shape[0]} entries, X has {X.shape[0]} rows.")
        return X, a

    def _groups_of(self, a: np.ndarray, n_features: int) -> np.ndarray:
        groups = np.unique(a)
        if groups.size < 2:
            raise ValueError("The sensitive attribute has a single group.")
        if groups.size > _MAX_GROUPS:
            raise ValueError(
                f"A has {groups.size} distinct values — transport needs a "
                f"discrete sensitive attribute (<= {_MAX_GROUPS} groups)."
            )
        counts = np.array([(a == g).sum() for g in groups])
        if counts.min() < 10:
            raise ValueError(
                f"Every sensitive group needs at least 10 samples to fit a "
                f"flow; smallest group has {counts.min()}."
            )
        return groups

    @staticmethod
    def _lstsq_direction(f_centered: np.ndarray, a_centered: np.ndarray) -> np.ndarray:
        """Unit least-squares direction of A within F (zeros if independent)."""
        w, *_ = np.linalg.lstsq(f_centered, a_centered, rcond=None)
        norm = float(np.linalg.norm(w))
        return w / norm if norm > 1e-12 else np.zeros_like(w)

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y=None, *, A=None) -> RBIGFairTransformer:
        """Fit the removal machinery for the configured strategy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features (including the sensitive column in
            ``sensitive_col`` mode).
        y : np.ndarray of shape (n_samples,), optional
            Discrete task labels — required by ``strategy="conditional"``.
        A : np.ndarray of shape (n_samples,), optional
            Sensitive attribute (metadata-routing mode).

        Returns
        -------
        self : RBIGFairTransformer
            The fitted transformer.
        """
        # sensitive_col mode consumes one column, so at least 2 are needed.
        X = validate_data(
            self, X, ensure_min_features=2 if self.sensitive_col is not None else 1
        )
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; choose from {_STRATEGIES}."
            )
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")
        feat, a = self._split_sensitive(X, A)
        if a is None:
            raise ValueError(
                "No sensitive attribute: set sensitive_col or pass A to fit."
            )
        if self.strategy in ("projection", "subspace"):
            self._fit_linear(feat, a)
        else:
            self._fit_transport(feat, a, y)
        return self

    def _fit_linear(self, feat: np.ndarray, a: np.ndarray) -> None:
        """Fit the projection direction or the deflated subspace."""
        d_f = feat.shape[1]
        self.mean_ = feat.mean(axis=0)
        fc = feat - self.mean_  # centered features: (n, d_f)
        ac = np.asarray(a, dtype=float) - float(np.mean(a))
        if self.strategy == "projection":
            self.sensitive_dir_ = self._lstsq_direction(fc, ac)
            return
        k = int(self.n_components)
        if not 0 < k <= d_f:
            raise ValueError(f"n_components must be in [1, {d_f}], got {k}.")
        dirs: list[np.ndarray] = []
        f_work = fc.copy()
        for _ in range(k):
            w = self._lstsq_direction(f_work, ac)
            if not w.any():
                break  # remaining A-dependence is not linear
            # Deflate: remove the found direction, repeat.
            f_work = f_work - np.outer(f_work @ w, w)
            dirs.append(w)
        self.sensitive_subspace_ = np.array(dirs) if dirs else np.zeros((0, d_f))

    def _fit_transport(self, feat: np.ndarray, a: np.ndarray, y) -> None:
        """Fit the pooled reference, group flows, and optional Y-strata."""
        groups = self._groups_of(a, feat.shape[1])
        self.groups_ = groups
        self.reference_flow_ = self._flow(0).fit(feat)
        self.group_flows_ = {
            g: _SharedRotationGroupFlow(self.reference_flow_, feat[a == g])
            for g in groups
        }
        if self.strategy == "transport":
            return
        if y is None:
            raise ValueError("strategy='conditional' requires y at fit time.")
        y = np.asarray(y).ravel()
        if y.shape[0] != feat.shape[0]:
            raise ValueError(f"y has {y.shape[0]} entries, X has {feat.shape[0]} rows.")
        strata = np.unique(y)
        if strata.size > _MAX_GROUPS:
            raise ValueError(
                f"y has {strata.size} distinct values — conditional transport "
                f"needs discrete labels (<= {_MAX_GROUPS} strata)."
            )
        min_n = (
            20 * feat.shape[1]
            if self.min_samples_per_stratum is None
            else int(self.min_samples_per_stratum)
        )
        self.stratum_refs_ = {}
        self.stratum_flows_ = {}
        for i, s in enumerate(strata):
            s_mask = y == s
            cells_ok = True
            for g in groups:
                n_cell = int((s_mask & (a == g)).sum())
                if n_cell < max(min_n, 10):
                    warnings.warn(
                        f"Stratum (A={g!r}, y={s!r}) has {n_cell} samples "
                        f"(< {max(min_n, 10)}); it falls back to the global "
                        f"transport flows.",
                        UserWarning,
                        stacklevel=3,
                    )
                    cells_ok = False
            if not cells_ok:
                continue  # all cells of this stratum use the global fallback
            self.stratum_refs_[s] = self._flow(i + 1).fit(feat[s_mask])
            for g in groups:
                cell = s_mask & (a == g)
                self.stratum_flows_[(g, s)] = _SharedRotationGroupFlow(
                    self.stratum_refs_[s], feat[cell]
                )

    # ── transform ────────────────────────────────────────────────────────────

    def _remove_linear(self, feat: np.ndarray) -> np.ndarray:
        fc = feat - self.mean_
        if self.strategy == "projection":
            w = self.sensitive_dir_
            return feat - np.outer(fc @ w, w)
        sub = self.sensitive_subspace_  # (k, d_f), orthonormal rows
        if sub.shape[0] == 0:
            return feat
        return feat - (fc @ sub.T) @ sub

    def _transport(self, feat: np.ndarray, a: np.ndarray) -> np.ndarray:
        out = np.empty_like(feat)
        for g, flow in self.group_flows_.items():
            mask = a == g
            if not mask.any():
                continue
            # x -> g_a(x) -> h^{-1}(.): group law into the pooled reference.
            out[mask] = self.reference_flow_.inverse_transform(
                flow.transform(feat[mask])
            )
        unseen = ~np.isin(a, self.groups_)
        if unseen.any():
            raise ValueError(
                f"A contains groups unseen at fit: {np.unique(a[unseen])!r}."
            )
        return out

    def _conditional(self, feat: np.ndarray, a: np.ndarray, y: np.ndarray):
        out = np.empty_like(feat)
        done = np.zeros(feat.shape[0], dtype=bool)
        for (g, s), flow in self.stratum_flows_.items():
            mask = (a == g) & (y == s)
            if not mask.any():
                continue
            out[mask] = self.stratum_refs_[s].inverse_transform(
                flow.transform(feat[mask])
            )
            done[mask] = True
        if not done.all():
            # Merged cells and unseen strata use the global transport flows.
            out[~done] = self._transport(feat[~done], a[~done])
        return out

    def transform(self, X: np.ndarray, *, A=None, y=None) -> np.ndarray:
        """Apply the removal and the ``alpha`` blend.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (including the sensitive column in
            ``sensitive_col`` mode, which is consumed and dropped).
        A : np.ndarray of shape (n_samples,), optional
            Sensitive attribute — required by ``transport`` and
            ``conditional`` in metadata-routing mode.
        y : np.ndarray of shape (n_samples,), optional
            Labels (or pseudo-labels) for ``conditional``; without them
            it falls back to unconditional transport with a warning.

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, d_out)
            ``α·X_fair + (1−α)·X`` — ``d_out = n_features − 1`` in
            ``sensitive_col`` mode, ``n_features`` otherwise.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        feat, a = self._split_sensitive(X, A)

        if self.strategy in ("projection", "subspace"):
            fair = self._remove_linear(feat)
        else:
            if a is None:
                raise ValueError(
                    f"strategy={self.strategy!r} needs the sensitive attribute "
                    "at transform time: set sensitive_col or pass A."
                )
            if self.strategy == "transport":
                fair = self._transport(feat, a)
            elif y is None:
                warnings.warn(
                    "strategy='conditional' got no y at transform time; "
                    "falling back to unconditional transport.  Pass "
                    "pseudo-labels for the two-stage pattern.",
                    UserWarning,
                    stacklevel=2,
                )
                fair = self._transport(feat, a)
            else:
                y = np.asarray(y).ravel()
                if y.shape[0] != X.shape[0]:
                    raise ValueError(
                        f"y has {y.shape[0]} entries, X has {X.shape[0]} rows."
                    )
                fair = self._conditional(feat, a, y)
        # Fairness-utility blend; alpha = 1 is full removal.
        return self.alpha * fair + (1.0 - self.alpha) * feat

    def fit_transform(self, X: np.ndarray, y=None, *, A=None) -> np.ndarray:
        """Fit and transform in one step, forwarding ``A``/``y`` to both.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.
        y : np.ndarray of shape (n_samples,), optional
            Labels for ``strategy="conditional"``.
        A : np.ndarray of shape (n_samples,), optional
            Sensitive attribute (metadata-routing mode).

        Returns
        -------
        X_out : np.ndarray of shape (n_samples, d_out)
            The transformed training data.
        """
        return self.fit(X, y, A=A).transform(X, A=A, y=y)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Output feature names ``rbigfair0 ..`` (sensitive column dropped).

        Parameters
        ----------
        input_features : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        names : np.ndarray of shape (d_out,)
            Output feature names.
        """
        check_is_fitted(self)
        d_out = self.n_features_in_ - (1 if self.sensitive_col is not None else 0)
        return np.asarray([f"rbigfair{i}" for i in range(d_out)], dtype=object)
