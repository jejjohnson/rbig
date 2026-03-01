"""Information-theoretic metrics for RBIG."""

from __future__ import annotations

import numpy as np


def mutual_information_rbig(
    model_X: AnnealedRBIG,
    model_Y: AnnealedRBIG,
    model_XY: AnnealedRBIG,
) -> float:
    """Mutual information via RBIG: MI(X;Y) = H(X) + H(Y) - H(X,Y)."""
    hx = model_X.entropy()
    hy = model_Y.entropy()
    hxy = model_XY.entropy()
    return float(hx + hy - hxy)


def kl_divergence_rbig(
    model_P: AnnealedRBIG,
    X_Q: np.ndarray,
) -> float:
    """KL divergence KL(P||Q) via RBIG.

    Parameters
    ----------
    model_P : fitted AnnealedRBIG on samples from P
    X_Q : samples from Q to compare against
    """
    log_pq = model_P.score_samples(X_Q)
    hp = model_P.entropy()
    return float(-np.mean(log_pq) - hp)


def total_correlation_rbig(X: np.ndarray, n_layers: int = 50) -> float:
    """Estimate total correlation of X using RBIG.

    TC(X) = sum_i H(X_i) - H(X)
    """
    from rbig._src.densities import joint_entropy_gaussian, marginal_entropy

    marg_h = marginal_entropy(X)
    joint_h = joint_entropy_gaussian(X)
    return float(np.sum(marg_h) - joint_h)


def entropy_normal_approx(X: np.ndarray) -> float:
    """Entropy via Gaussian approximation H(X) ≈ 0.5*log|2πe Σ|."""
    from rbig._src.densities import joint_entropy_gaussian

    return joint_entropy_gaussian(X)


def negentropy(X: np.ndarray) -> np.ndarray:
    """Negentropy of each marginal: J(x) = H(Gauss) - H(x) >= 0."""
    _n, _d = X.shape
    gauss_h = 0.5 * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.var(X, axis=0))
    from rbig._src.densities import marginal_entropy

    marg_h = marginal_entropy(X)
    return gauss_h - marg_h
