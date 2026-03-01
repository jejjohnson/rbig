"""Density estimation utilities for RBIG."""

from __future__ import annotations

import numpy as np
from scipy import stats


def marginal_entropy(X: np.ndarray, correction: bool = True) -> np.ndarray:
    """Estimate marginal entropy of each dimension using KDE.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    correction : bool, apply correction for finite sample

    Returns
    -------
    entropies : array of shape (n_features,)
    """
    _n_samples, n_features = X.shape
    entropies = np.zeros(n_features)
    for i in range(n_features):
        kde = stats.gaussian_kde(X[:, i])
        log_density = np.log(kde(X[:, i]) + 1e-300)
        entropies[i] = -np.mean(log_density)
    return entropies


def joint_entropy_gaussian(X: np.ndarray) -> float:
    """Entropy of multivariate Gaussian with same covariance as X.

    H(X) = 0.5 * log |2*pi*e*Sigma|
    """
    _n, d = X.shape
    cov = np.cov(X.T)
    if d == 1:
        cov = np.array([[cov]])
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        log_det = -np.inf
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)


def total_correlation(X: np.ndarray) -> float:
    """Estimate Total Correlation (multivariate MI) of X.

    TC = sum_i H(X_i) - H(X)

    Uses Gaussian entropy approximation for joint.
    """
    marg_h = marginal_entropy(X)
    joint_h = joint_entropy_gaussian(X)
    return float(np.sum(marg_h) - joint_h)


def gaussian_entropy(n_features: int, cov: np.ndarray | None = None) -> float:
    """Entropy of multivariate Gaussian distribution."""
    if cov is None:
        return 0.5 * n_features * (1 + np.log(2 * np.pi))
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return 0.5 * (n_features * (1 + np.log(2 * np.pi)) + log_det)


def entropy_reduction(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """Compute entropy reduction between two representations.

    TC(X_before) - TC(X_after)
    """
    tc_before = total_correlation(X_before)
    tc_after = total_correlation(X_after)
    return tc_before - tc_after
