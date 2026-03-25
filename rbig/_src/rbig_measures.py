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

from typing import Any

import numpy as np

from rbig._src.densities import kl_to_standard_normal, marginal_entropy

# ---------------------------------------------------------------------------
# Level 1 — pre-fitted models
# ---------------------------------------------------------------------------


def total_correlation_rbig_reduction(model) -> float:
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


def entropy_rbig_reduction(model, X: np.ndarray) -> float:
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
    model_X,
    model_Y,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    model_XY=None,
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
    model_Y,
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

    KLD(P_X ‖ P_Y) ≈ Σ_d D(Z_d ‖ N(0,1)) + TC(Z)  where Z = G_Y(X)

    Fits an ``AnnealedRBIG`` model on Y, applies its transform to X,
    then estimates per-marginal KL to N(0,1) + TC of the result.

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
