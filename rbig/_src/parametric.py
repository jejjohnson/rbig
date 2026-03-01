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


# ---- Analytical IT formulas ----


def entropy_gaussian(cov: np.ndarray) -> float:
    """Analytical entropy of a Gaussian distribution.

    H(X) = 0.5 * log |2*pi*e*Sigma|
    """
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return -np.inf
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)


def total_correlation_gaussian(cov: np.ndarray) -> float:
    """Analytical total correlation of a Gaussian.

    TC = sum_i H(X_i) - H(X) = -0.5 * log |corr|
    where corr is the correlation matrix.
    """
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    marginal_vars = np.diag(cov)
    sum_marg_h = 0.5 * np.sum(np.log(2 * np.pi * np.e * marginal_vars))
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        return np.inf
    joint_h = 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)
    return float(sum_marg_h - joint_h)


def mutual_information_gaussian(
    cov_X: np.ndarray,
    cov_Y: np.ndarray,
    cov_XY: np.ndarray,
) -> float:
    """Analytical mutual information between two Gaussians.

    MI(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    hx = entropy_gaussian(cov_X)
    hy = entropy_gaussian(cov_Y)
    hxy = entropy_gaussian(cov_XY)
    return float(hx + hy - hxy)


def kl_divergence_gaussian(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mu1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    """Analytical KL divergence KL(P0||P1) between two Gaussians.

    KL = 0.5 * [tr(Sigma1^{-1} Sigma0) + (mu1-mu0)^T Sigma1^{-1} (mu1-mu0)
                - d + log(|Sigma1|/|Sigma0|)]
    """
    mu0 = np.atleast_1d(mu0)
    mu1 = np.atleast_1d(mu1)
    cov0 = np.atleast_2d(cov0)
    cov1 = np.atleast_2d(cov1)
    d = len(mu0)
    cov1_inv = np.linalg.inv(cov1)
    diff = mu1 - mu0
    _sign0, log_det0 = np.linalg.slogdet(cov0)
    _sign1, log_det1 = np.linalg.slogdet(cov1)
    trace_term = np.trace(cov1_inv @ cov0)
    quad_term = diff @ cov1_inv @ diff
    log_det_term = log_det1 - log_det0
    return float(0.5 * (trace_term + quad_term - d + log_det_term))


# ---- Distribution samplers ----


def gaussian(
    n_samples: int = 1000,
    loc: float = 0.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from univariate Gaussian."""
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=loc, scale=scale, size=n_samples)


def multivariate_gaussian(
    n_samples: int = 1000,
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    d: int = 2,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from multivariate Gaussian."""
    rng = np.random.default_rng(random_state)
    if mean is None:
        mean = np.zeros(d)
    if cov is None:
        cov = np.eye(len(mean))
    return rng.multivariate_normal(mean, cov, size=n_samples)


def uniform(
    n_samples: int = 1000,
    low: float = 0.0,
    high: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from uniform distribution."""
    rng = np.random.default_rng(random_state)
    return rng.uniform(low=low, high=high, size=n_samples)


def exponential(
    n_samples: int = 1000,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from exponential distribution."""
    rng = np.random.default_rng(random_state)
    return rng.exponential(scale=scale, size=n_samples)


def laplace(
    n_samples: int = 1000,
    loc: float = 0.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Laplace distribution."""
    rng = np.random.default_rng(random_state)
    return rng.laplace(loc=loc, scale=scale, size=n_samples)


def student_t(
    n_samples: int = 1000,
    df: float = 5.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Student-t distribution."""
    rng = np.random.default_rng(random_state)
    return rng.standard_t(df=df, size=n_samples)


def gamma(
    n_samples: int = 1000,
    shape: float = 2.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Gamma distribution."""
    rng = np.random.default_rng(random_state)
    return rng.gamma(shape=shape, scale=scale, size=n_samples)


def beta(
    n_samples: int = 1000,
    a: float = 2.0,
    b: float = 2.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Beta distribution."""
    rng = np.random.default_rng(random_state)
    return rng.beta(a=a, b=b, size=n_samples)


def lognormal(
    n_samples: int = 1000,
    mean: float = 0.0,
    sigma: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from log-normal distribution."""
    rng = np.random.default_rng(random_state)
    return rng.lognormal(mean=mean, sigma=sigma, size=n_samples)


def dirichlet(
    n_samples: int = 1000,
    alpha: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Dirichlet distribution."""
    rng = np.random.default_rng(random_state)
    if alpha is None:
        alpha = np.ones(3)
    return rng.dirichlet(alpha=alpha, size=n_samples)


def wishart(
    n_samples: int = 1000,
    df: int = 5,
    scale: np.ndarray | None = None,
    d: int = 3,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from Wishart distribution."""
    from scipy.stats import wishart as wishart_dist

    if scale is None:
        scale = np.eye(d)
    rng = np.random.default_rng(random_state)
    seed = int(rng.integers(0, 2**31))
    return wishart_dist.rvs(df=df, scale=scale, size=n_samples, random_state=seed)


def von_mises(
    n_samples: int = 1000,
    mu: float = 0.0,
    kappa: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from von Mises distribution."""
    rng = np.random.default_rng(random_state)
    return rng.vonmises(mu=mu, kappa=kappa, size=n_samples)
