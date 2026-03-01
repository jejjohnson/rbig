"""Marginal Gaussianization transforms."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtri

from rbig._src.base import BaseTransform


class MarginalUniformize(BaseTransform):
    """Transform each marginal to uniform [0,1] using empirical CDF."""

    def __init__(self, bound_correct: bool = True, eps: float = 1e-6):
        self.bound_correct = bound_correct
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalUniformize:
        self.support_ = np.sort(X, axis=0)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            Xt[:, i] = self._uniformize(X[:, i], self.support_[:, i])
        if self.bound_correct:
            Xt = np.clip(Xt, self.eps, 1 - self.eps)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            Xt[:, i] = np.interp(
                X[:, i],
                np.linspace(0, 1, len(self.support_[:, i])),
                self.support_[:, i],
            )
        return Xt

    @staticmethod
    def _uniformize(x: np.ndarray, support: np.ndarray) -> np.ndarray:
        n = len(support)
        ranks = np.searchsorted(support, x, side="left")
        return (ranks + 0.5) / n


class MarginalGaussianize(BaseTransform):
    """Transform each marginal to standard Gaussian using empirical CDF + probit."""

    def __init__(self, bound_correct: bool = True, eps: float = 1e-6):
        self.bound_correct = bound_correct
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalGaussianize:
        self.support_ = np.sort(X, axis=0)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            u = MarginalUniformize._uniformize(X[:, i], self.support_[:, i])
            if self.bound_correct:
                u = np.clip(u, self.eps, 1 - self.eps)
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            u = stats.norm.cdf(X[:, i])
            Xt[:, i] = np.interp(
                u, np.linspace(0, 1, len(self.support_[:, i])), self.support_[:, i]
            )
        return Xt

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for Gaussianization transform."""
        Xt = self.transform(X)
        # log|det J| = sum of log|d(g(x))/dx| = sum of log(phi(g(x)) / f(x))
        # For empirical CDF: log p_uniform / log p_normal
        log_prob = np.sum(stats.norm.logpdf(Xt), axis=1)
        return log_prob


class MarginalKDEGaussianize(BaseTransform):
    """Transform each marginal to Gaussian using KDE-estimated CDF."""

    def __init__(self, bw_method: str | float | None = None, eps: float = 1e-6):
        self.bw_method = bw_method
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalKDEGaussianize:
        self.kdes_ = []
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            self.kdes_.append(stats.gaussian_kde(X[:, i], bw_method=self.bw_method))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            u = np.array(
                [self.kdes_[i].integrate_box_1d(-np.inf, xi) for xi in X[:, i]]
            )
            u = np.clip(u, self.eps, 1 - self.eps)
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        from scipy.optimize import brentq

        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            for j, xj in enumerate(X[:, i]):
                u = stats.norm.cdf(xj)
                try:
                    Xt[j, i] = brentq(
                        lambda x, u=u, i=i: (
                            self.kdes_[i].integrate_box_1d(-np.inf, x) - u
                        ),
                        -100,
                        100,
                    )
                except ValueError:
                    Xt[j, i] = 0.0
        return Xt
