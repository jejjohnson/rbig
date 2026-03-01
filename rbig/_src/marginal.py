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
        """Log |det J| for marginal Gaussianization.

        For g(x) = Phi^{-1}(F_n(x)):
            log|dg/dx| = log f_n(x_i) - log phi(g(x_i))

        where f_n is estimated from the spacing of the empirical support.
        """
        Xt = self.transform(X)
        log_jac = np.zeros(X.shape[0])
        n = self.support_.shape[0]
        for i in range(self.n_features_):
            support_i = self.support_[:, i]
            spacings = np.diff(support_i)
            pos_sp = spacings[spacings > 0]
            min_sp = pos_sp.min() if len(pos_sp) > 0 else 1e-10
            safe_sp = np.where(spacings > 0, spacings, min_sp)
            # Pad to n elements: first element uses the first spacing
            sp_full = np.concatenate([[safe_sp[0]], safe_sp])
            ranks = np.clip(np.searchsorted(support_i, X[:, i], side="left"), 0, n - 1)
            local_spacing = sp_full[ranks]
            log_f_i = -np.log(n * local_spacing)
            log_phi_gi = stats.norm.logpdf(Xt[:, i])
            log_jac += log_f_i - log_phi_gi
        return log_jac


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
