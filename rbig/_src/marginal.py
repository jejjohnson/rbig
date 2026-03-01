"""Marginal Gaussianization transforms."""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtri

from rbig._src.base import BaseTransform, Bijector


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


class QuantileGaussianizer(Bijector):
    """Gaussianize each marginal using sklearn QuantileTransformer."""

    def __init__(self, n_quantiles: int = 1000, random_state: int | None = 0):
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "QuantileGaussianizer":
        from sklearn.preprocessing import QuantileTransformer

        n_quantiles = min(self.n_quantiles, X.shape[0])
        self.qt_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self.random_state,
        )
        self.qt_.fit(X)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.qt_.inverse_transform(X)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Estimate log |det J| by finite differences on the quantile transform."""
        eps = 1e-5
        log_det = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            dummy_plus = X.copy()
            dummy_plus[:, i] = X[:, i] + eps
            dummy_minus = X.copy()
            dummy_minus[:, i] = X[:, i] - eps
            y_plus = self.qt_.transform(dummy_plus)[:, i]
            y_minus = self.qt_.transform(dummy_minus)[:, i]
            deriv = (y_plus - y_minus) / (2 * eps)
            log_det += np.log(np.maximum(np.abs(deriv), 1e-300))
        return log_det


class KDEGaussianizer(Bijector):
    """Gaussianize each marginal using KDE-estimated CDF."""

    def __init__(self, bw_method: str | float | None = None, eps: float = 1e-6):
        self.bw_method = bw_method
        self.eps = eps

    def fit(self, X: np.ndarray) -> "KDEGaussianizer":
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

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            pdf = self.kdes_[i](X[:, i])
            u = np.array(
                [self.kdes_[i].integrate_box_1d(-np.inf, xi) for xi in X[:, i]]
            )
            u = np.clip(u, self.eps, 1 - self.eps)
            g = ndtri(u)
            log_phi = stats.norm.logpdf(g)
            log_det += np.log(np.maximum(pdf, 1e-300)) - log_phi
        return log_det


class GMMGaussianizer(Bijector):
    """Gaussianize each marginal using a Gaussian Mixture Model CDF."""

    def __init__(self, n_components: int = 5, random_state: int | None = 0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GMMGaussianizer":
        from sklearn.mixture import GaussianMixture

        self.gmms_ = []
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            gmm = GaussianMixture(
                n_components=min(self.n_components, X.shape[0] // 5),
                random_state=self.random_state,
            )
            gmm.fit(X[:, i : i + 1])
            self.gmms_.append(gmm)
        return self

    def _cdf(self, gmm, x: np.ndarray) -> np.ndarray:
        """Compute CDF of GMM at points x (1D)."""
        weights = gmm.weights_
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_.ravel())
        cdf = np.zeros_like(x, dtype=float)
        for w, mu, sigma in zip(weights, means, stds):
            cdf += w * stats.norm.cdf(x, loc=mu, scale=sigma)
        return cdf

    def _pdf(self, gmm, x: np.ndarray) -> np.ndarray:
        """Compute PDF of GMM at points x (1D)."""
        weights = gmm.weights_
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_.ravel())
        pdf = np.zeros_like(x, dtype=float)
        for w, mu, sigma in zip(weights, means, stds):
            pdf += w * stats.norm.pdf(x, loc=mu, scale=sigma)
        return pdf

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            u = self._cdf(self.gmms_[i], X[:, i])
            u = np.clip(u, 1e-6, 1 - 1e-6)
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
                            self._cdf(self.gmms_[i], np.array([x]))[0] - u
                        ),
                        -50,
                        50,
                    )
                except ValueError:
                    Xt[j, i] = 0.0
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            u = self._cdf(self.gmms_[i], X[:, i])
            u = np.clip(u, 1e-6, 1 - 1e-6)
            g = ndtri(u)
            pdf = self._pdf(self.gmms_[i], X[:, i])
            log_phi = stats.norm.logpdf(g)
            log_det += np.log(np.maximum(pdf, 1e-300)) - log_phi
        return log_det


class SplineGaussianizer(Bijector):
    """Gaussianize each marginal using monotone splines (PchipInterpolator)."""

    def __init__(self, n_quantiles: int = 200, eps: float = 1e-6):
        self.n_quantiles = n_quantiles
        self.eps = eps

    def fit(self, X: np.ndarray) -> "SplineGaussianizer":
        from scipy.interpolate import PchipInterpolator

        self.splines_ = []
        self.inv_splines_ = []
        self.n_features_ = X.shape[1]
        n_q = min(self.n_quantiles, X.shape[0])
        quantiles = np.linspace(0, 1, n_q)
        for i in range(self.n_features_):
            xi_sorted = np.sort(X[:, i])
            x_q = np.quantile(xi_sorted, quantiles)
            g_q = ndtri(np.clip(quantiles, self.eps, 1 - self.eps))
            self.splines_.append(PchipInterpolator(x_q, g_q))
            self.inv_splines_.append(PchipInterpolator(g_q, x_q))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            Xt[:, i] = self.splines_[i](X[:, i])
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            Xt[:, i] = self.inv_splines_[i](X[:, i])
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            deriv = self.splines_[i](X[:, i], 1)  # first derivative
            log_det += np.log(np.maximum(np.abs(deriv), 1e-300))
        return log_det
