"""Mutual-information feature selection on RBIG estimators (issue #128).

:class:`RBIGMISelector` is a ``SelectorMixin`` consumer of the
information-theory suite (:func:`~rbig._src.rbig_measures.estimate_mi` and
entropy combinations).  Four strategies of increasing fidelity and cost:

- ``"filter"`` — univariate relevance ``I(X_j; Y)``; O(d) MI estimates.
- ``"mrmr"`` — greedy relevance minus mean redundancy; O(d * k) estimates.
- ``"greedy"`` — forward selection on exact *conditional* MI
  ``I(X_j; Y | X_S) = I([X_j, X_S]; Y) - I(X_S; Y)``; captures synergy
  (XOR-style feature pairs) that univariate filters provably miss.
- ``"joint"`` — exhaustive ``argmax_S I(X_S; Y)``; guarded to
  ``d <= 20`` unless ``force=True`` (no pruning: MI is not sub-additive,
  synergistic groups can beat the sum of their univariate scores).

Discrete targets are dithered with seeded uniform jitter before MI
estimation (the marginal-layer recipe from the P0 batch), mirroring how
``mutual_info_classif`` jitters discrete inputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.rbig_measures import estimate_mi

_STRATEGIES = ("filter", "mrmr", "greedy", "joint")


class RBIGMISelector(SelectorMixin, BaseEstimator):
    """Feature selection by RBIG mutual information.

    Parameters
    ----------
    n_features_to_select : int or float, default 10
        Number of features to keep (capped at ``d``); a float in (0, 1)
        selects that fraction of features.
    strategy : {"filter", "mrmr", "greedy", "joint"}, default "greedy"
        Selection strategy (see module docstring).
    mi_threshold : float or None, default None
        After selection, drop chosen features whose univariate relevance
        ``I(X_j; Y)`` falls below this many nats.
    n_layers_rbig : int, default 25
        Layer budget of every internal flow fit.
    tol_rbig : float or "auto", default 1e-4
        Convergence tolerance of the internal flows.
    force : bool, default False
        Allow ``strategy="joint"`` above 20 features (exponential cost).
    random_state : int or None, optional
        Seed for the internal flows and the discrete-target dithering.

    Attributes
    ----------
    mi_scores_ : np.ndarray of shape (n_features,)
        Univariate relevance ``I(X_j; Y)`` in nats.
    selected_features_ : list of int
        Selected indices in selection order.
    support_ : np.ndarray of shape (n_features,)
        Boolean support mask.
    selection_path_ : list of tuple
        ``(feature_index, score_at_selection)`` per greedy/mrmr step
        (selection order and cumulative-relevance scree data).

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGMISelector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((400, 3))
    >>> y = X[:, 1] + 0.1 * rng.standard_normal(400)
    >>> sel = RBIGMISelector(
    ...     n_features_to_select=1, strategy="filter", n_layers_rbig=5
    ... ).fit(X, y)
    >>> int(sel.get_support(indices=True)[0])
    1
    """

    def __init__(
        self,
        n_features_to_select: float = 10,
        strategy: str = "greedy",
        mi_threshold: float | None = None,
        n_layers_rbig: int = 25,
        tol_rbig: float | str = 1e-4,
        force: bool = False,
        random_state: int | None = None,
    ):
        self.n_features_to_select = n_features_to_select
        self.strategy = strategy
        self.mi_threshold = mi_threshold
        self.n_layers_rbig = n_layers_rbig
        self.tol_rbig = tol_rbig
        self.force = force
        self.random_state = random_state

    # ── internals ────────────────────────────────────────────────────────────

    def _rbig_kwargs(self) -> dict[str, Any]:
        return {
            "n_layers": self.n_layers_rbig,
            "tol": self.tol_rbig,
            "patience": 5,
            "random_state": self.random_state,
        }

    def _prepare_y(self, y: np.ndarray) -> np.ndarray:
        """Column-shape y; dither discrete targets with seeded jitter.

        Non-numeric (categorical) labels are ordinal-encoded first, so
        string classes work exactly like integer classes.
        """
        y = np.asarray(y)
        if not np.issubdtype(y.dtype, np.number):
            _classes, y = np.unique(y, return_inverse=True)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        uniques = np.unique(y)
        if uniques.size < max(20, int(np.sqrt(y.size))):
            gaps = np.diff(uniques)
            scale = 0.1 * gaps.min() if gaps.size else 1e-3
            rng = np.random.default_rng(self.random_state)
            y = y + rng.uniform(-0.5, 0.5, size=y.shape) * scale
        return y

    def _joint_relevance(self, cols: tuple[int, ...]) -> float:
        """Cached ``I(X_cols; Y)`` via the TC-of-Gaussianized-blocks route.

        ``estimate_mi`` Gaussianizes each block *fully* before measuring
        the cross-block total correlation, so within-block dependence is
        removed and near-discrete (dithered) targets are handled far more
        robustly than an entropy-difference construction, whose per-fit
        biases do not cancel on spiky marginals.
        """
        key = tuple(sorted(cols))
        if key not in self._rel_cache:
            self._rel_cache[key] = float(
                estimate_mi(self._X[:, list(key)], self._y, **self._rbig_kwargs())
            )
        return self._rel_cache[key]

    def _pairwise_redundancy(self, i: int, j: int) -> float:
        key = (min(i, j), max(i, j))
        if key not in self._red_cache:
            self._red_cache[key] = float(
                estimate_mi(
                    self._X[:, [key[0]]], self._X[:, [key[1]]], **self._rbig_kwargs()
                )
            )
        return self._red_cache[key]

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> RBIGMISelector:
        """Estimate relevance and run the configured selection strategy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features.
        y : np.ndarray of shape (n_samples,)
            Target (continuous, or discrete — dithered internally).

        Returns
        -------
        self : RBIGMISelector
            The fitted selector.
        """
        X, y = validate_data(self, X, y)
        if self.strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; choose from {_STRATEGIES}."
            )
        d = X.shape[1]
        if isinstance(self.n_features_to_select, float):
            if not 0.0 < self.n_features_to_select <= 1.0:
                raise ValueError("Fractional n_features_to_select must be in (0, 1].")
            k = max(1, int(round(self.n_features_to_select * d)))
        else:
            k = int(self.n_features_to_select)
            if k < 1:
                raise ValueError(
                    f"n_features_to_select must be a positive integer or a "
                    f"fraction in (0, 1]; got {self.n_features_to_select!r}."
                )
            k = min(k, d)
        if self.strategy == "joint" and d > 20 and not self.force:
            raise ValueError(
                f"strategy='joint' is exhaustive and guarded to d <= 20 "
                f"(got d = {d}). Use strategy='greedy', or pass force=True "
                "if you accept the exponential cost."
            )

        self._X = X
        self._y = self._prepare_y(y)
        self._rel_cache: dict[tuple[int, ...], float] = {}
        self._red_cache: dict[tuple[int, int], float] = {}

        # Univariate relevance (used by every strategy and mi_threshold).
        self.mi_scores_ = np.array([self._joint_relevance((j,)) for j in range(d)])

        if self.strategy == "filter":
            order = np.argsort(self.mi_scores_)[::-1][:k]
            selected = list(map(int, order))
            path = [(j, float(self.mi_scores_[j])) for j in selected]
        elif self.strategy == "mrmr":
            selected, path = self._fit_mrmr(d, k)
        elif self.strategy == "greedy":
            selected, path = self._fit_greedy(d, k)
        else:  # joint
            selected, path = self._fit_joint(d, k)

        if self.mi_threshold is not None and self.strategy in ("filter", "mrmr"):
            # Honor the cutoff fully: if every candidate is weaker than the
            # requested minimum MI, the selection is empty and transform
            # returns an (n, 0) matrix — screening semantics, like
            # SelectKBest(k=0).  (Greedy/joint scores are group gains, so
            # the univariate threshold does not apply to them.)
            selected = [j for j in selected if self.mi_scores_[j] >= self.mi_threshold]

        self.selected_features_ = selected
        self.selection_path_ = path
        self.support_ = np.zeros(d, dtype=bool)
        self.support_[selected] = True
        del self._X, self._y
        return self

    def _fit_mrmr(self, d: int, k: int) -> tuple[list[int], list[tuple[int, float]]]:
        selected = [int(np.argmax(self.mi_scores_))]
        path = [(selected[0], float(self.mi_scores_[selected[0]]))]
        while len(selected) < k:
            best_j, best_score = -1, -np.inf
            for j in range(d):
                if j in selected:
                    continue
                redundancy = np.mean(
                    [self._pairwise_redundancy(j, s) for s in selected]
                )
                score = self.mi_scores_[j] - redundancy
                if score > best_score:
                    best_j, best_score = j, score
            selected.append(best_j)
            path.append((best_j, float(best_score)))
        return selected, path

    def _fit_greedy(self, d: int, k: int) -> tuple[list[int], list[tuple[int, float]]]:
        selected: list[int] = []
        path: list[tuple[int, float]] = []
        current = 0.0
        while len(selected) < k:
            best_j, best_gain, best_joint = -1, -np.inf, 0.0
            for j in range(d):
                if j in selected:
                    continue
                joint = self._joint_relevance(tuple([*selected, j]))
                gain = joint - current  # I(X_j; Y | X_S)
                if gain > best_gain:
                    best_j, best_gain, best_joint = j, gain, joint
            selected.append(best_j)
            current = best_joint
            path.append((best_j, float(current)))
        return selected, path

    def _fit_joint(self, d: int, k: int) -> tuple[list[int], list[tuple[int, float]]]:
        from itertools import combinations

        best_set, best_rel = None, -np.inf
        # No univariate-sum pruning: MI is not sub-additive (synergistic
        # groups can beat the sum of their univariate scores), so any such
        # bound could skip the true argmax.  The d <= 20 guard is the only
        # cost control; joint is exponential by design.
        for combo in combinations(range(d), k):
            rel = self._joint_relevance(combo)
            if rel > best_rel:
                best_set, best_rel = combo, rel
        selected = list(best_set)
        return selected, [(j, float(best_rel)) for j in selected]

    # ── SelectorMixin contract ───────────────────────────────────────────────

    def _get_support_mask(self) -> np.ndarray:
        check_is_fitted(self)
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags
