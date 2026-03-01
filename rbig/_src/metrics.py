"""Information-theoretic metrics for RBIG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rbig._src.model import AnnealedRBIG


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


def total_correlation_rbig(X: np.ndarray) -> float:
    """Estimate total correlation of X.

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


def entropy_univariate(x: np.ndarray) -> float:
    """Univariate differential entropy via Vasicek spacing estimator."""
    n = len(x)
    m = max(1, int(np.floor(np.sqrt(n / 2))))
    x_sorted = np.sort(x)
    diffs = x_sorted[m:] - x_sorted[: n - m]
    h = np.mean(np.log(n / (2 * m) * diffs + 1e-300))
    return float(h)


def entropy_marginal(X: np.ndarray) -> np.ndarray:
    """Per-dimension marginal entropy using Vasicek spacing estimator."""
    n_features = X.shape[1]
    return np.array([entropy_univariate(X[:, i]) for i in range(n_features)])


def entropy_quantile_spacing(x: np.ndarray, n_quantiles: int = 100) -> float:
    """Entropy estimate via quantile spacings."""
    quantiles = np.linspace(0, 1, n_quantiles + 2)[1:-1]
    q = np.quantile(x, quantiles)
    diffs = np.diff(q)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 0.0
    return float(np.mean(np.log(diffs * n_quantiles + 1e-300)))


def entropy_rbig(model: AnnealedRBIG, X: np.ndarray) -> float:
    """Entropy estimated through a fitted RBIG model.

    H(X) = -E[log p(x)] ≈ -mean(score_samples(X))
    """
    log_probs = model.score_samples(X)
    return float(-np.mean(log_probs))


def negative_log_likelihood(model: AnnealedRBIG, X: np.ndarray) -> float:
    """Negative log-likelihood under the RBIG model."""
    log_probs = model.score_samples(X)
    return float(-np.mean(log_probs))


def information_summary(model: AnnealedRBIG, X: np.ndarray) -> dict:
    """Returns dict with entropy, TC, and MI estimates from RBIG model.

    Returns
    -------
    dict with keys: 'entropy', 'total_correlation', 'neg_log_likelihood'
    """
    from rbig._src.densities import marginal_entropy as _marginal_entropy

    h = entropy_rbig(model, X)
    marginal_h = _marginal_entropy(X)
    tc = float(np.sum(marginal_h) - h)
    return {
        "entropy": h,
        "total_correlation": tc,
        "neg_log_likelihood": negative_log_likelihood(model, X),
    }


def information_reduction(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """TC reduction between two representations.

    Returns TC(X_before) - TC(X_after)
    """
    from rbig._src.densities import (
        joint_entropy_gaussian,
        marginal_entropy as _marginal_entropy,
    )

    def _tc(X):
        return float(np.sum(_marginal_entropy(X)) - joint_entropy_gaussian(X))

    return _tc(X_before) - _tc(X_after)
