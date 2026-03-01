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


from rbig._src.base import Bijector


class Tanh(Bijector):
    """Tanh bijector with log det Jacobian."""

    def fit(self, X: np.ndarray) -> Tanh:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """log |det J| = sum_i log(1 - tanh(x_i)^2)"""
        return np.sum(np.log(1 - np.tanh(X) ** 2 + 1e-300), axis=1)


class Exp(Bijector):
    """Exp bijector with log det Jacobian."""

    def fit(self, X: np.ndarray) -> Exp:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.exp(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(X, 1e-300))

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """log |det J| = sum_i x_i"""
        return np.sum(X, axis=1)


class Cube(Bijector):
    """Cube bijector (x^3) with log det Jacobian."""

    def fit(self, X: np.ndarray) -> Cube:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X**3

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return np.cbrt(X)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """log |det J| = sum_i log(3 * x_i^2)"""
        return np.sum(np.log(3 * X**2 + 1e-300), axis=1)


def check_density(model, X: np.ndarray, n_grid: int = 1000) -> float:
    """Utility to verify density approximately integrates to 1.

    Uses importance sampling: E_q[p(x)/q(x)] ≈ 1
    where q is N(0,1).
    """
    Z = stats.norm.rvs(size=(n_grid, X.shape[1]))
    log_p = model.score_samples(Z)
    log_q = np.sum(stats.norm.logpdf(Z), axis=1)
    log_ratio = log_p - log_q
    log_ratio = np.clip(log_ratio, -500, 500)
    return float(np.mean(np.exp(log_ratio)))
