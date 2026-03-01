"""Parametric marginal transforms."""

from __future__ import annotations

import numpy as np
from scipy import stats

from rbig._src.base import BaseTransform


class LogitTransform(BaseTransform):
    """Logit transform: maps [0,1] to (-inf, inf)."""

    def fit(self, X: np.ndarray) -> LogitTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.log(X / (1 - X))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.sum(-np.log(X) - np.log(1 - X), axis=1)


class BoxCoxTransform(BaseTransform):
    """Box-Cox power transform for each feature."""

    def __init__(self, method: str = "mle"):
        self.method = method

    def fit(self, X: np.ndarray) -> BoxCoxTransform:
        self.lambdas_ = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.all(xi > 0):
                _, lam = stats.boxcox(xi)
            else:
                lam = 0.0
            self.lambdas_[i] = lam
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            xi = X[:, i]
            lam = self.lambdas_[i]
            if np.all(xi > 0):
                Xt[:, i] = stats.boxcox(xi, lmbda=lam)
            else:
                Xt[:, i] = xi
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            lam = self.lambdas_[i]
            if np.abs(lam) < 1e-10:
                Xt[:, i] = np.exp(X[:, i])
            else:
                Xt[:, i] = np.power(np.maximum(lam * X[:, i] + 1, 0), 1 / lam)
        return Xt

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        log_jac = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            xi = X[:, i]
            lam = self.lambdas_[i]
            if np.abs(lam) < 1e-10:
                log_jac += -xi
            else:
                log_jac += (lam - 1) * np.log(np.maximum(xi, 1e-300))
        return log_jac


class QuantileTransform(BaseTransform):
    """Quantile transform to Gaussian using sklearn."""

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "normal"):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def fit(self, X: np.ndarray) -> QuantileTransform:
        from sklearn.preprocessing import QuantileTransformer

        n_quantiles = min(self.n_quantiles, X.shape[0])
        self.qt_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=self.output_distribution,
            random_state=0,
        )
        self.qt_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.inverse_transform(X)
