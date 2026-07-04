"""RBIG-way information theory measures via per-layer TC reduction.

This module implements the IT estimation approach from
Laparra et al. (2011, 2020) which uses the cumulative reduction in
total correlation across RBIG layers — no Jacobian estimation needed.

Two levels of abstraction are provided:

**Level 1** — pass pre-fitted ``AnnealedRBIG`` models:

    total_correlation_rbig_reduction, entropy_rbig_reduction,
    mutual_information_rbig_reduction, kl_divergence_rbig_reduction

**Level 0** — pass raw data (models are fitted internally):

    estimate_tc, estimate_entropy, estimate_mi, estimate_kld

References
----------
Laparra, V., Camps-Valls, G., & Malo, J. (2011).
    Iterative Gaussianization: From ICA to Random Rotations. IEEE TNNs.
Laparra, V., Johnson, J.E., et al. (2020).
    Information Theory Measures via Multidimensional Gaussianization.
    arXiv:2010.03807.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rbig._src.model import AnnealedRBIG

import numpy as np

from rbig._src.densities import kl_to_standard_normal, marginal_entropy

# ---------------------------------------------------------------------------
# Level 1 — pre-fitted models
# ---------------------------------------------------------------------------


def total_correlation_rbig_reduction(model: AnnealedRBIG) -> float:
    """Total correlation via per-layer TC reduction.

    TC(X) = tc_per_layer_[0] − tc_per_layer_[-1]

    Parameters
    ----------
    model : AnnealedRBIG
        Fitted RBIG model.

    Returns
    -------
    tc : float
        Estimated total correlation in nats.
    """
    return model.total_correlation_reduction()


def entropy_rbig_reduction(model: AnnealedRBIG, X: np.ndarray) -> float:
    """Entropy via marginal entropies minus RBIG-way TC.

    H(X) = Σ_d H(X_d) − TC(X)

    where H(X_d) are KDE-based marginal entropies and TC is obtained
    from the per-layer TC reduction of the fitted model.

    Parameters
    ----------
    model : AnnealedRBIG
        Fitted RBIG model (used only for TC estimation).
    X : np.ndarray of shape (n_samples, n_features)
        Data whose entropy is estimated.  Marginal entropies are
        computed from this array via KDE.

    Returns
    -------
    h : float
        Estimated differential entropy in nats.
    """
    h_marginals = marginal_entropy(X)  # shape (n_features,)
    tc = total_correlation_rbig_reduction(model)
    return float(np.sum(h_marginals) - tc)


def mutual_information_rbig_reduction(
    model_X: AnnealedRBIG,
    model_Y: AnnealedRBIG,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    model_XY: AnnealedRBIG | None = None,
    rbig_kwargs: dict[str, Any] | None = None,
) -> float:
    """Mutual information via RBIG-way TC reduction.

    MI(X; Y) = TC([G_X(X), G_Y(Y)])

    Gaussianize X and Y independently, concatenate the results, then
    measure the total correlation of the concatenated representation.

    Parameters
    ----------
    model_X : AnnealedRBIG
        Fitted on X.
    model_Y : AnnealedRBIG
        Fitted on Y.
    X : np.ndarray of shape (n_samples, d_x)
        Samples from the marginal of X.
    Y : np.ndarray of shape (n_samples, d_y)
        Samples from the marginal of Y (same n_samples as X).
    model_XY : AnnealedRBIG, optional
        Pre-fitted model on the concatenated Gaussianized data.  If
        ``None``, a new model is fitted internally using *rbig_kwargs*.
    rbig_kwargs : dict, optional
        Keyword arguments forwarded to ``AnnealedRBIG`` when fitting
        the joint model.  Ignored if *model_XY* is provided.

    Returns
    -------
    mi : float
        Estimated mutual information MI(X; Y) in nats.
    """
    from rbig._src.model import AnnealedRBIG

    Z_X = model_X.transform(X)
    Z_Y = model_Y.transform(Y)
    Z = np.hstack([Z_X, Z_Y])

    if model_XY is None:
        kwargs = rbig_kwargs or {}
        model_XY = AnnealedRBIG(**kwargs)
        model_XY.fit(Z)

    return model_XY.total_correlation_reduction()


def kl_divergence_rbig_reduction(
    model_Y: AnnealedRBIG,
    X: np.ndarray,
    *,
    rbig_kwargs: dict[str, Any] | None = None,
) -> float:
    """KL divergence via RBIG-way TC reduction.

    KLD(P_X ‖ P_Y) ≈ Σ_d D(Z_d ‖ N(0,1)) + TC(Z)

    where Z = G_Y(X) is the result of applying Y's Gaussianization
    transform to X's samples.  D(Z_d ‖ N(0,1)) is the per-marginal
    KL divergence to standard Gaussian and TC(Z) is estimated via a
    new RBIG fit on Z.

    If X ~ P_Y then Z ≈ N(0, I), so D = 0 and TC = 0, giving KLD ≈ 0.

    Parameters
    ----------
    model_Y : AnnealedRBIG
        Fitted on samples from Q (the reference distribution).
    X : np.ndarray of shape (n_samples, n_features)
        Samples from P (the distribution being compared).
    rbig_kwargs : dict, optional
        Keyword arguments forwarded to ``AnnealedRBIG`` for the TC
        estimation step on Z.

    Returns
    -------
    kld : float
        Estimated KL divergence KLD(P ‖ Q) in nats.  Non-negative
        when well-estimated.
    """
    from rbig._src.model import AnnealedRBIG

    Z = model_Y.transform(X)

    # Per-marginal KL divergence to N(0,1)
    kl_marginals = kl_to_standard_normal(Z)

    # TC of Z via RBIG
    kwargs = rbig_kwargs or {}
    model_Z = AnnealedRBIG(**kwargs)
    model_Z.fit(Z)
    tc_z = model_Z.total_correlation_reduction()

    return float(np.sum(kl_marginals) + tc_z)


# ---------------------------------------------------------------------------
# Level 0 — data only (fits models internally)
# ---------------------------------------------------------------------------

_DEFAULT_RBIG_KWARGS: dict[str, Any] = {
    "n_layers": 100,
    "rotation": "pca",
    "patience": 10,
}


def estimate_tc(X: np.ndarray, **rbig_kwargs: Any) -> float:
    """Estimate total correlation of X via RBIG-way TC reduction.

    TC(X) = Σₖ ΔTCₖ  (sum of per-layer TC drops)

    Fits an ``AnnealedRBIG`` model internally and returns the
    cumulative TC reduction.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.  Defaults: n_layers=100,
        rotation="pca", patience=10.

    Returns
    -------
    tc : float
        Estimated total correlation in nats.
    """
    from rbig._src.model import AnnealedRBIG

    kwargs = {**_DEFAULT_RBIG_KWARGS, **rbig_kwargs}
    model = AnnealedRBIG(**kwargs)
    model.fit(X)
    return total_correlation_rbig_reduction(model)


def estimate_entropy(X: np.ndarray, **rbig_kwargs: Any) -> float:
    """Estimate differential entropy of X via RBIG-way TC reduction.

    H(X) = Σ_d H(X_d) − TC(X)

    Fits an ``AnnealedRBIG`` model internally for the TC term;
    marginal entropies are estimated via KDE.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    h : float
        Estimated differential entropy in nats.
    """
    from rbig._src.model import AnnealedRBIG

    kwargs = {**_DEFAULT_RBIG_KWARGS, **rbig_kwargs}
    model = AnnealedRBIG(**kwargs)
    model.fit(X)
    return entropy_rbig_reduction(model, X)


def estimate_mi(
    X: np.ndarray,
    Y: np.ndarray,
    **rbig_kwargs: Any,
) -> float:
    """Estimate mutual information MI(X; Y) via RBIG-way TC reduction.

    MI(X; Y) = TC([G_X(X), G_Y(Y)])

    Fits separate ``AnnealedRBIG`` models on X and Y, then measures
    the TC of the concatenated Gaussianized representation.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d_x)
        Samples from the marginal of X.
    Y : np.ndarray of shape (n_samples, d_y)
        Samples from the marginal of Y.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG`` for all three model fits.

    Returns
    -------
    mi : float
        Estimated mutual information in nats.
    """
    from rbig._src.model import AnnealedRBIG

    kwargs = {**_DEFAULT_RBIG_KWARGS, **rbig_kwargs}
    model_X = AnnealedRBIG(**kwargs)
    model_Y = AnnealedRBIG(**kwargs)
    model_X.fit(X)
    model_Y.fit(Y)
    return mutual_information_rbig_reduction(model_X, model_Y, X, Y, rbig_kwargs=kwargs)


def estimate_kld(
    X: np.ndarray,
    Y: np.ndarray,
    **rbig_kwargs: Any,
) -> float:
    """Estimate KL divergence KLD(P_X ‖ P_Y) via RBIG-way TC reduction.

    KLD(P_X ‖ P_Y) ≈ Σ_d KL(P_{Z_d} ‖ N(0, 1)) + TC(Z),  where Z = G_Y(X)

    Fits an ``AnnealedRBIG`` model on Y, applies its transform to X,
    then estimates the sum of per-marginal KL divergences to N(0, 1)
    plus the total correlation of Z.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Samples from P (the distribution being compared).
    Y : np.ndarray of shape (n_samples, n_features)
        Samples from Q (the reference distribution).
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    kld : float
        Estimated KL divergence in nats.
    """
    from rbig._src.model import AnnealedRBIG

    kwargs = {**_DEFAULT_RBIG_KWARGS, **rbig_kwargs}
    model_Y = AnnealedRBIG(**kwargs)
    model_Y.fit(Y)
    return kl_divergence_rbig_reduction(model_Y, X, rbig_kwargs=kwargs)


# ---------------------------------------------------------------------------
# Extended measures (issue #124): CMI, DTC, O-information, interaction
# information, JSD, pairwise MI — all reduced to entropy combinations.
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float, hi: float, name: str, tol: float = 0.05) -> float:
    """Clamp an estimate to its theoretical range, warning on large cuts.

    Silent clamping is how estimator bias hides: when the raw value violates
    the bound by more than ``tol`` nats, a ``UserWarning`` reports it so the
    bias is visible while downstream code still receives a valid value.
    """
    clipped = float(min(max(value, lo), hi))
    if abs(clipped - value) > tol:
        import warnings

        warnings.warn(
            f"{name} estimate {value:.4f} clamped to {clipped:.4f} "
            f"(theoretical range [{lo}, {hi}]); the excess indicates "
            "estimator bias at this sample size.",
            UserWarning,
            stacklevel=3,
        )
    return clipped


def estimate_cmi(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, **rbig_kwargs: Any
) -> float:
    """Estimate conditional mutual information I(X; Y | Z).

    Uses the entropy decomposition (4 RBIG fits)::

        I(X; Y | Z) = H([X,Z]) + H([Y,Z]) - H([X,Y,Z]) - H(Z)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d_x)
        Samples of X.
    Y : np.ndarray of shape (n_samples, d_y)
        Samples of Y.
    Z : np.ndarray of shape (n_samples, d_z)
        Samples of the conditioning variable.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG`` for all four fits.

    Returns
    -------
    cmi : float
        Estimated conditional MI in nats (clamped to >= 0 with a warning
        when the raw estimate is materially negative).
    """
    h_xz = estimate_entropy(np.hstack([X, Z]), **rbig_kwargs)
    h_yz = estimate_entropy(np.hstack([Y, Z]), **rbig_kwargs)
    h_xyz = estimate_entropy(np.hstack([X, Y, Z]), **rbig_kwargs)
    h_z = estimate_entropy(Z, **rbig_kwargs)
    return _clamp(h_xz + h_yz - h_xyz - h_z, 0.0, np.inf, "CMI")


def estimate_dtc(X: np.ndarray, **rbig_kwargs: Any) -> float:
    """Estimate dual total correlation (binding information) of X.

    Uses the entropy decomposition (d + 1 RBIG fits)::

        DTC(X) = sum_i H(X_{-i}) - (d - 1) * H(X)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d)
        Data matrix (d >= 2).
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    dtc : float
        Estimated dual total correlation in nats (clamped to >= 0).
    """
    d = X.shape[1]
    if d < 2:
        raise ValueError("DTC requires at least 2 dimensions.")
    h_joint = estimate_entropy(X, **rbig_kwargs)
    h_minus = sum(
        estimate_entropy(np.delete(X, i, axis=1), **rbig_kwargs) for i in range(d)
    )
    return _clamp(h_minus - (d - 1) * h_joint, 0.0, np.inf, "DTC")


def estimate_o_information(X: np.ndarray, **rbig_kwargs: Any) -> float:
    """Estimate the O-information Omega(X) = TC(X) - DTC(X).

    Positive values indicate redundancy-dominated interdependence, negative
    values synergy-dominated (Rosas et al. 2019).  Roughly 2d + 1 RBIG fits.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d)
        Data matrix (d >= 3 for a meaningful sign).
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    omega : float
        Estimated O-information in nats (signed; not clamped).
    """
    return float(estimate_tc(X, **rbig_kwargs) - estimate_dtc(X, **rbig_kwargs))


def estimate_interaction_information(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, **rbig_kwargs: Any
) -> float:
    """Estimate interaction information II(X; Y; Z) = I(X;Y|Z) - I(X;Y).

    Signed: positive means Z enhances the X-Y dependence (synergy),
    negative means Z explains part of it (redundancy).  About 7 RBIG fits.

    Parameters
    ----------
    X, Y, Z : np.ndarray of shape (n_samples, d_*)
        Samples of the three variables.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    ii : float
        Estimated interaction information in nats (signed; not clamped).
    """
    # Raw (unclamped) CMI so the sign structure is preserved.
    h_xz = estimate_entropy(np.hstack([X, Z]), **rbig_kwargs)
    h_yz = estimate_entropy(np.hstack([Y, Z]), **rbig_kwargs)
    h_xyz = estimate_entropy(np.hstack([X, Y, Z]), **rbig_kwargs)
    h_z = estimate_entropy(Z, **rbig_kwargs)
    cmi_raw = h_xz + h_yz - h_xyz - h_z
    return float(cmi_raw - estimate_mi(X, Y, **rbig_kwargs))


def estimate_jsd(X: np.ndarray, Y: np.ndarray, **rbig_kwargs: Any) -> float:
    """Estimate the Jensen-Shannon divergence JSD(P || Q).

    Uses the mixture-entropy identity (3 RBIG fits)::

        JSD = H(M) - (H(P) + H(Q)) / 2,   M = (P + Q) / 2

    with the mixture represented by an equal-size balanced concatenation
    of the two samples.  Bounded in [0, ln 2]; tail-sensitive like every
    sample-based divergence — prefer tail-extended marginals (see the
    marginal ``tail`` parameter) when the supports differ.

    Parameters
    ----------
    X : np.ndarray of shape (n_p, d)
        Samples from P.
    Y : np.ndarray of shape (n_q, d)
        Samples from Q.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    jsd : float
        Estimated JSD in nats, clamped to [0, ln 2].
    """
    m = min(X.shape[0], Y.shape[0])
    mixture = np.vstack([X[:m], Y[:m]])
    h_m = estimate_entropy(mixture, **rbig_kwargs)
    h_p = estimate_entropy(X, **rbig_kwargs)
    h_q = estimate_entropy(Y, **rbig_kwargs)
    return _clamp(h_m - 0.5 * (h_p + h_q), 0.0, float(np.log(2.0)), "JSD")


def pairwise_mi_matrix(
    X: np.ndarray, normalized: bool = False, **rbig_kwargs: Any
) -> np.ndarray:
    """Pairwise mutual-information matrix of the columns of X.

    Diagonal entries are the marginal entropies; off-diagonal entries the
    pairwise MI ``I(X_i; X_j)``.  Cost is 3 RBIG fits per pair — use
    :class:`InformationMeasures` to amortize repeated queries.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, d)
        Data matrix.
    normalized : bool, default False
        Normalize off-diagonal entries to ``I / sqrt(H_i * H_j)``.
    **rbig_kwargs
        Forwarded to ``AnnealedRBIG``.

    Returns
    -------
    M : np.ndarray of shape (d, d)
        Symmetric MI matrix.
    """
    d = X.shape[1]
    M = np.zeros((d, d))
    for i in range(d):
        M[i, i] = estimate_entropy(X[:, [i]], **rbig_kwargs)
    for i in range(d):
        for j in range(i + 1, d):
            mi = estimate_mi(X[:, [i]], X[:, [j]], **rbig_kwargs)
            if normalized:
                denom = np.sqrt(max(M[i, i] * M[j, j], 1e-300))
                mi = mi / denom
            M[i, j] = M[j, i] = mi
    return M


class InformationMeasures:
    """Fit-caching interface to the RBIG information-theoretic measures.

    Every measure reduces to a combination of joint entropies over column
    subsets; this class caches each subset's entropy (one ``AnnealedRBIG``
    fit per distinct subset), so composite quantities and repeated queries
    do not refit.  On d=20 data the O-information alone needs ~2d+1 = 41
    fits — caching is what makes interactive use viable.

    Parameters
    ----------
    **rbig_kwargs
        Forwarded to every internal ``AnnealedRBIG`` fit (e.g.
        ``n_layers=40, tol=1e-5, random_state=0``).

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import InformationMeasures
    >>> X = np.random.default_rng(0).standard_normal((500, 3))
    >>> im = InformationMeasures(n_layers=10, random_state=0).fit(X)
    >>> float(im.tc()) >= 0.0
    True
    """

    def __init__(self, **rbig_kwargs: Any):
        self.rbig_kwargs = rbig_kwargs
        self._cache: dict[tuple[int, ...], float] = {}
        self._X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> InformationMeasures:
        """Store the data and clear the entropy cache."""
        self._X = np.asarray(X, dtype=float)
        self._cache = {}
        return self

    def _require_fit(self) -> np.ndarray:
        if self._X is None:
            raise ValueError("Call fit(X) before querying measures.")
        return self._X

    def entropy(self, cols: tuple[int, ...] | list[int] | None = None) -> float:
        """Joint entropy of the given columns (cached; 1 fit per subset)."""
        X = self._require_fit()
        key = tuple(sorted(range(X.shape[1]) if cols is None else cols))
        if key not in self._cache:
            self._cache[key] = estimate_entropy(X[:, list(key)], **self.rbig_kwargs)
        return self._cache[key]

    def mi(self, cols_a: list[int], cols_b: list[int]) -> float:
        """Mutual information between two column groups (<= 3 fits)."""
        joint = tuple(sorted([*cols_a, *cols_b]))
        return _clamp(
            self.entropy(cols_a) + self.entropy(cols_b) - self.entropy(joint),
            0.0,
            np.inf,
            "MI",
        )

    def cmi(self, cols_a: list[int], cols_b: list[int], cols_c: list[int]) -> float:
        """Conditional MI I(A; B | C) (<= 4 fits)."""
        return _clamp(
            self.entropy([*cols_a, *cols_c])
            + self.entropy([*cols_b, *cols_c])
            - self.entropy([*cols_a, *cols_b, *cols_c])
            - self.entropy(cols_c),
            0.0,
            np.inf,
            "CMI",
        )

    def tc(self) -> float:
        """Total correlation of all columns (<= d + 1 fits)."""
        X = self._require_fit()
        marginals = sum(self.entropy([i]) for i in range(X.shape[1]))
        return _clamp(marginals - self.entropy(), 0.0, np.inf, "TC")

    def dtc(self) -> float:
        """Dual total correlation of all columns (<= d + 1 fits)."""
        X = self._require_fit()
        d = X.shape[1]
        h_minus = sum(self.entropy([j for j in range(d) if j != i]) for i in range(d))
        return _clamp(h_minus - (d - 1) * self.entropy(), 0.0, np.inf, "DTC")

    def o_information(self) -> float:
        """O-information Omega = TC - DTC (signed; <= 2d + 1 fits)."""
        return float(self.tc() - self.dtc())

    def pairwise_mi_matrix(self, normalized: bool = False) -> np.ndarray:
        """Pairwise MI matrix over all columns (cached entropies)."""
        X = self._require_fit()
        d = X.shape[1]
        M = np.zeros((d, d))
        for i in range(d):
            M[i, i] = self.entropy([i])
        for i in range(d):
            for j in range(i + 1, d):
                mi = self.mi([i], [j])
                if normalized:
                    mi = mi / np.sqrt(max(M[i, i] * M[j, j], 1e-300))
                M[i, j] = M[j, i] = mi
        return M
