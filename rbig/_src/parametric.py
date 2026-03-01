"""Parametric marginal transforms for RBIG."""
from typing import Optional, Union

import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import QuantileTransformer

BOUNDS_THRESHOLD = 1e-7


# ---------------------------------------------------------------------------
# Base helper
# ---------------------------------------------------------------------------


def _clip_uniform(u: np.ndarray) -> np.ndarray:
    return np.clip(u, BOUNDS_THRESHOLD, 1.0 - BOUNDS_THRESHOLD)


# ---------------------------------------------------------------------------
# Parametric marginal classes
# ---------------------------------------------------------------------------


class HistogramUniformization:
    """Histogram-based CDF/PPF for marginal uniformization."""

    def __init__(self, bins: Union[str, int] = "auto", alpha: float = 1e-5):
        self.bins = bins
        self.alpha = alpha

    def fit(self, X: np.ndarray) -> "HistogramUniformization":
        X = np.asarray(X, dtype=float).ravel()
        hist = np.histogram(X, bins=self.bins)
        # add regularisation
        hpdf = hist[0].astype(float) + self.alpha
        estimator = stats.rv_histogram((hpdf, hist[1]))
        self._estimator = estimator
        self._support = (hist[1].min(), hist[1].max())
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(self._estimator.cdf(x))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self._estimator.pdf(x), 1e-12, None))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self._estimator.ppf(_clip_uniform(u))


class KDEUniformization:
    """Kernel Density Estimate-based CDF/PPF."""

    def __init__(self, bw_method: Optional[str] = "scott", n_quantiles: int = 500):
        self.bw_method = bw_method
        self.n_quantiles = n_quantiles

    def fit(self, X: np.ndarray) -> "KDEUniformization":
        X = np.asarray(X, dtype=float).ravel()
        kde = stats.gaussian_kde(X, bw_method=self.bw_method)
        self._kde = kde

        # Build support grid for CDF/PPF
        x_min = X.min() - 3 * X.std()
        x_max = X.max() + 3 * X.std()
        support = np.linspace(x_min, x_max, self.n_quantiles)
        pdf_vals = kde(support)
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        cdf_vals = np.cumsum(pdf_vals)
        cdf_vals = cdf_vals / cdf_vals[-1]
        cdf_vals = np.maximum.accumulate(cdf_vals)

        self._support = support
        self._cdf_fn = interp1d(
            support, cdf_vals, kind="linear", fill_value=(0.0, 1.0), bounds_error=False
        )
        self._ppf_fn = interp1d(
            cdf_vals, support, kind="linear", fill_value="extrapolate", bounds_error=False
        )
        self._pdf_vals = pdf_vals
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(self._cdf_fn(x))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(self._kde(x), 1e-12, None))

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self._ppf_fn(_clip_uniform(u))


class QuantileUniformization:
    """sklearn QuantileTransformer-based uniformization."""

    def __init__(self, n_quantiles: int = 1000, random_state: Optional[int] = None):
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "QuantileUniformization":
        X = np.asarray(X, dtype=float).ravel()[:, None]
        self._qt = QuantileTransformer(
            n_quantiles=min(self.n_quantiles, X.shape[0]),
            output_distribution="uniform",
            random_state=self.random_state,
        )
        self._qt.fit(X)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()[:, None]
        return _clip_uniform(self._qt.transform(x).ravel())

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        # Approximate logpdf: log of the derivative of the CDF
        x = np.asarray(x, dtype=float).ravel()
        eps = 1e-5
        f1 = self.cdf(x + eps)
        f0 = self.cdf(x - eps)
        pdf = np.clip((f1 - f0) / (2 * eps), 1e-12, None)
        return np.log(pdf)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).ravel()[:, None]
        return self._qt.inverse_transform(_clip_uniform(u)).ravel()


class GaussianMixtureMarginal:
    """Gaussian Mixture Model marginal density."""

    def __init__(self, n_components: int = 3, random_state: Optional[int] = None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> "GaussianMixtureMarginal":
        from sklearn.mixture import GaussianMixture
        X = np.asarray(X, dtype=float).ravel()[:, None]
        self._gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit(X)
        # Build CDF/PPF via numerical integration on a grid
        x_min = X.min() - 4 * X.std()
        x_max = X.max() + 4 * X.std()
        support = np.linspace(x_min, x_max, 2000)
        pdf_vals = np.exp(self._gmm.score_samples(support[:, None]))
        cdf_vals = np.cumsum(pdf_vals * (support[1] - support[0]))
        cdf_vals = cdf_vals / cdf_vals[-1]
        cdf_vals = np.maximum.accumulate(cdf_vals)
        self._support = support
        self._cdf_fn = interp1d(
            support, cdf_vals, kind="linear", fill_value=(0.0, 1.0), bounds_error=False
        )
        self._ppf_fn = interp1d(
            cdf_vals, support, kind="linear", fill_value="extrapolate", bounds_error=False
        )
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(self._cdf_fn(np.asarray(x, dtype=float)))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        return self._gmm.score_samples(x[:, None])

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return self._ppf_fn(_clip_uniform(np.asarray(u, dtype=float)))


class LogisticMarginal:
    """Logistic distribution marginal."""

    def fit(self, X: np.ndarray) -> "LogisticMarginal":
        X = np.asarray(X, dtype=float).ravel()
        loc, scale = stats.logistic.fit(X)
        self._loc = loc
        self._scale = scale
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(stats.logistic.cdf(x, loc=self._loc, scale=self._scale))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.logistic.logpdf(x, loc=self._loc, scale=self._scale)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return stats.logistic.ppf(_clip_uniform(u), loc=self._loc, scale=self._scale)


class UniformMarginal:
    """Uniform distribution marginal."""

    def fit(self, X: np.ndarray) -> "UniformMarginal":
        X = np.asarray(X, dtype=float).ravel()
        self._low = X.min()
        self._high = X.max()
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(stats.uniform.cdf(x, loc=self._low, scale=self._high - self._low))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.uniform.logpdf(x, loc=self._low, scale=self._high - self._low)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return stats.uniform.ppf(_clip_uniform(u), loc=self._low, scale=self._high - self._low)


class NormalMarginal:
    """Normal distribution marginal."""

    def fit(self, X: np.ndarray) -> "NormalMarginal":
        X = np.asarray(X, dtype=float).ravel()
        self._loc, self._scale = stats.norm.fit(X)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(stats.norm.cdf(x, loc=self._loc, scale=self._scale))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.logpdf(x, loc=self._loc, scale=self._scale)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return stats.norm.ppf(_clip_uniform(u), loc=self._loc, scale=self._scale)


class LaplaceMarginal:
    """Laplace distribution marginal."""

    def fit(self, X: np.ndarray) -> "LaplaceMarginal":
        X = np.asarray(X, dtype=float).ravel()
        self._loc, self._scale = stats.laplace.fit(X)
        return self

    def cdf(self, x: np.ndarray) -> np.ndarray:
        return _clip_uniform(stats.laplace.cdf(x, loc=self._loc, scale=self._scale))

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        return stats.laplace.logpdf(x, loc=self._loc, scale=self._scale)

    def ppf(self, u: np.ndarray) -> np.ndarray:
        return stats.laplace.ppf(_clip_uniform(u), loc=self._loc, scale=self._scale)


class ParametricMarginalGaussianizer:
    """Apply a parametric marginal uniformization, then Gaussian PPF.

    Parameters
    ----------
    marginal : object
        Fitted marginal with .cdf() and .ppf() methods.
    """

    def __init__(self, marginal):
        self.marginal = marginal

    def gaussianize(self, x: np.ndarray) -> np.ndarray:
        u = self.marginal.cdf(x)
        return stats.norm.ppf(_clip_uniform(u))

    def inverse_gaussianize(self, z: np.ndarray) -> np.ndarray:
        u = stats.norm.cdf(z)
        return self.marginal.ppf(_clip_uniform(u))


# ---------------------------------------------------------------------------
# Factory and functional API
# ---------------------------------------------------------------------------


def fit_parametric_marginal(X: np.ndarray, method: str = "histogram") -> object:
    """Factory: fit a parametric marginal to 1-D data.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples,)
    method : {'histogram', 'kde', 'quantile', 'normal', 'logistic', 'laplace', 'uniform'}

    Returns
    -------
    fitted marginal object
    """
    X = np.asarray(X, dtype=float).ravel()
    mapping = {
        "histogram": HistogramUniformization,
        "kde": KDEUniformization,
        "quantile": QuantileUniformization,
        "normal": NormalMarginal,
        "logistic": LogisticMarginal,
        "laplace": LaplaceMarginal,
        "uniform": UniformMarginal,
    }
    if method not in mapping:
        raise ValueError(f"Unknown method: {method!r}. Choose from {list(mapping)}")
    return mapping[method]().fit(X)


def parametric_gaussianize(X: np.ndarray, params) -> np.ndarray:
    """Gaussianize X using a fitted parametric marginal.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples,)
    params : fitted marginal object (has .cdf())

    Returns
    -------
    Z : np.ndarray, shape (n_samples,)
    """
    u = params.cdf(np.asarray(X, dtype=float))
    return stats.norm.ppf(_clip_uniform(u))


def parametric_gaussianize_inverse(Z: np.ndarray, params) -> np.ndarray:
    """Inverse Gaussianize Z using a fitted parametric marginal.

    Parameters
    ----------
    Z : np.ndarray, shape (n_samples,)
    params : fitted marginal object (has .ppf())

    Returns
    -------
    X : np.ndarray, shape (n_samples,)
    """
    u = stats.norm.cdf(np.asarray(Z, dtype=float))
    return params.ppf(_clip_uniform(u))
