"""Parametric marginal transforms.

This module provides parametric bijective transforms for marginal
Gaussianisation (LogitTransform, BoxCoxTransform, QuantileTransform),
analytical information-theoretic formulas for Gaussian distributions, and
a collection of distribution samplers used in RBIG experiments.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from rbig._src.base import BaseTransform


class LogitTransform(BaseTransform):
    """Logit transform: bijectively maps the unit hypercube (0,1)ᵈ to ℝᵈ.

    Each feature is transformed independently by the logit (log-odds) function:

        Forward  : y = log(x / (1 − x))
        Inverse  : x = σ(y) = 1 / (1 + e^{−y})        (sigmoid)
        Log-det  : ∑ᵢ [−log xᵢ − log(1 − xᵢ)]

    The transform is useful as a pre-processing step when data lives in (0, 1),
    e.g. probabilities or proportions.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import LogitTransform
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0.05, 0.95, size=(100, 3))  # data in (0, 1)
    >>> tr = LogitTransform().fit(X)
    >>> Y = tr.transform(X)  # data now in ℝ
    >>> X_rec = tr.inverse_transform(Y)
    >>> np.allclose(X, X_rec, atol=1e-10)
    True
    """

    def fit(self, X: np.ndarray) -> LogitTransform:
        """No-op fit (stateless transform).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Ignored.

        Returns
        -------
        self : LogitTransform
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the logit map y = log(x / (1 − x)).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in (0, 1).

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Log-odds transformed data in ℝ.
        """
        return np.log(X / (1 - X))  # y = logit(x) = log(x/(1-x))

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse sigmoid (logistic) map x = 1 / (1 + e^{−y}).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in ℝ.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Recovered data in (0, 1).
        """
        return 1 / (1 + np.exp(-X))  # x = sigmoid(y) = 1/(1+e^{-y})

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample log |det J| of the forward logit transform.

        The Jacobian of logit is diagonal with entries
        d(logit xᵢ)/dxᵢ = 1/xᵢ + 1/(1−xᵢ), so:

            log |det J| = ∑ᵢ [−log xᵢ − log(1 − xᵢ)]

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in (0, 1) (pre-transform).

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Log absolute determinant of the Jacobian for each sample.
        """
        # Diagonal Jacobian: sum_i (-log xi - log(1-xi))
        return np.sum(-np.log(X) - np.log(1 - X), axis=1)


class BoxCoxTransform(BaseTransform):
    """Box-Cox power transform fitted independently to each feature.

    The Box-Cox family of transforms is parameterised by λ (one per feature):

        λ ≠ 0 :  y = (xᵏ − 1) / λ
        λ → 0 :  y = log(x)             (continuity limit)

    λ values are estimated via maximum likelihood (scipy's ``boxcox``).
    Features with non-positive values are left unchanged (λ = 0 applied as
    identity rather than log, since log requires positive inputs).

    The inverse transform is:

        λ ≠ 0 :  x = (λy + 1)^{1/λ}
        λ = 0 :  x = exp(y)

    The log-det of the Jacobian is:

        λ ≠ 0 :  ∑ᵢ (λ − 1) log xᵢ
        λ = 0 :  ∑ᵢ (−xᵢ)              (from d(log x)/dx = 1/x, log det = −x)

    Parameters
    ----------
    method : str, optional (default='mle')
        Fitting method passed conceptually; currently scipy MLE is always
        used regardless of this value.

    Attributes
    ----------
    lambdas_ : np.ndarray of shape (n_features,)
        Fitted λ values after calling ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import BoxCoxTransform
    >>> rng = np.random.default_rng(1)
    >>> X = rng.exponential(scale=2.0, size=(200, 3))  # strictly positive
    >>> tr = BoxCoxTransform().fit(X)
    >>> Y = tr.transform(X)
    >>> X_rec = tr.inverse_transform(Y)
    >>> np.allclose(X, X_rec, atol=1e-6)
    True
    """

    def __init__(self, method: str = "mle"):
        self.method = method

    def fit(self, X: np.ndarray) -> BoxCoxTransform:
        """Estimate one Box-Cox λ per feature via MLE.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.  Features that contain non-positive values are
            assigned λ = 0 (no transform applied during ``transform``).

        Returns
        -------
        self : BoxCoxTransform
            Fitted instance with ``lambdas_`` attribute set.
        """
        self.lambdas_ = np.zeros(X.shape[1])  # λ per feature, default 0
        for i in range(X.shape[1]):
            xi = X[:, i]
            if np.all(xi > 0):
                _, lam = stats.boxcox(xi)  # MLE for λ
            else:
                lam = 0.0  # non-positive data: no power transform
            self.lambdas_[i] = lam
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted Box-Cox transform to X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.  Features with λ = 0 and non-positive values are
            passed through unchanged.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Box-Cox transformed data.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            xi = X[:, i]
            lam = self.lambdas_[i]
            if np.all(xi > 0):
                Xt[:, i] = stats.boxcox(xi, lmbda=lam)  # y = (x^lam - 1)/lam or log(x)
            else:
                Xt[:, i] = xi  # pass-through for non-positive
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the Box-Cox transform.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Box-Cox transformed data.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Recovered original-scale data.  Uses:

            * λ = 0 : x = exp(y)
            * λ ≠ 0 : x = (λy + 1)^{1/λ}   (clamped to 0 for stability)
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(X.shape[1]):
            lam = self.lambdas_[i]
            if np.abs(lam) < 1e-10:
                Xt[:, i] = np.exp(X[:, i])  # x = exp(y)
            else:
                # x = (λy + 1)^{1/λ}, clamp argument to ≥ 0
                Xt[:, i] = np.power(np.maximum(lam * X[:, i] + 1, 0), 1 / lam)
        return Xt

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample log |det J| of the forward Box-Cox transform.

        The Jacobian is diagonal; for each feature:

            λ ≠ 0 :  d/dx[(xᵏ−1)/λ] = x^{λ−1}  ⟹  log = (λ−1) log x
            λ = 0 :  d/dx[log x] = 1/x           ⟹  log = −x   (log(1/x) = −log x,
                                                               but here we store −x
                                                               matching the original code)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (pre-transform, original scale).

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Sum of per-feature log Jacobian contributions.
        """
        log_jac = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            xi = X[:, i]
            lam = self.lambdas_[i]
            if np.abs(lam) < 1e-10:
                log_jac += -xi  # log(1/x) ~= -x (lam->0 limit)
            else:
                # (lam-1) log xi from x^{lam-1} Jacobian
                log_jac += (lam - 1) * np.log(np.maximum(xi, 1e-300))
        return log_jac


class QuantileTransform(BaseTransform):
    """Quantile transform that maps each feature to a target distribution.

    Wraps ``sklearn.preprocessing.QuantileTransformer`` to provide a uniform
    interface compatible with RBIG pipelines.  By default, features are
    mapped to a standard Gaussian distribution, which is a common
    pre-processing step for Gaussianisation.

    Parameters
    ----------
    n_quantiles : int, optional (default=1000)
        Number of quantiles used to build the empirical CDF.  Capped at
        ``n_samples`` during ``fit``.
    output_distribution : str, optional (default='normal')
        Target distribution for the transform.  Accepted values are
        ``'normal'`` (standard Gaussian) and ``'uniform'``.

    Attributes
    ----------
    qt_ : sklearn.preprocessing.QuantileTransformer
        Fitted sklearn transformer, available after calling ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import QuantileTransform
    >>> rng = np.random.default_rng(42)
    >>> X = rng.exponential(scale=1.0, size=(500, 2))
    >>> tr = QuantileTransform(n_quantiles=200).fit(X)
    >>> Y = tr.transform(X)  # approximately standard Gaussian
    >>> Y.shape
    (500, 2)
    >>> # Marginal means should be near zero, stds near 1
    >>> np.allclose(Y.mean(axis=0), 0, atol=0.1)
    True
    """

    def __init__(self, n_quantiles: int = 1000, output_distribution: str = "normal"):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution

    def fit(self, X: np.ndarray) -> QuantileTransform:
        """Fit the quantile transform to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : QuantileTransform
            Fitted instance with ``qt_`` attribute set.
        """
        from sklearn.preprocessing import QuantileTransformer

        # Cap n_quantiles at the number of available training samples
        n_quantiles = min(self.n_quantiles, X.shape[0])
        self.qt_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution=self.output_distribution,
            random_state=0,
        )
        self.qt_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted quantile transform.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Data mapped to the target distribution.
        """
        return self.qt_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the quantile transform.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the target distribution space.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Recovered data in the original distribution space.
        """
        return self.qt_.inverse_transform(X)


# ---- Analytical IT formulas ----


def entropy_gaussian(cov: np.ndarray) -> float:
    """Analytic differential entropy of a multivariate Gaussian.

    Computes the closed-form entropy of 𝒩(μ, Σ):

        H(X) = ½ log|2πeΣ| = ½ (d(1 + log 2π) + log|Σ|)

    where d is the dimensionality and |·| denotes the matrix determinant.
    The mean μ does not affect the entropy.

    Parameters
    ----------
    cov : np.ndarray of shape (d, d) or (1,) for scalar
        Covariance matrix Σ (or variance for d=1).  Coerced to at least 2-D
        via ``np.atleast_2d``.

    Returns
    -------
    entropy : float
        Differential entropy in nats.  Returns ``-inf`` if Σ is singular or
        not positive definite.

    Notes
    -----
    ``np.linalg.slogdet`` is used for numerically stable log-determinant
    computation.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import entropy_gaussian
    >>> # 2-D standard Gaussian: H = 0.5 * 2 * (1 + log 2π) ≈ 2.838 nats
    >>> h = entropy_gaussian(np.eye(2))
    >>> np.isclose(h, 0.5 * 2 * (1 + np.log(2 * np.pi)))
    True
    """
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    sign, log_det = np.linalg.slogdet(cov)  # stable log|Σ|
    if sign <= 0:
        return -np.inf  # singular covariance
    # H = ½ (d(1 + log 2π) + log|Σ|)
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)


def total_correlation_gaussian(cov: np.ndarray) -> float:
    """Analytic Total Correlation of a multivariate Gaussian.

    For a Gaussian with covariance Σ, the TC reduces to a function of the
    correlation matrix R = D^{-½} Σ D^{-½} (where D = diag(Σ)):

        TC = ∑ᵢ H(Xᵢ) − H(X) = −½ log|R|

    Equivalently, it measures how far the distribution is from being a
    product of its marginals.

    Parameters
    ----------
    cov : np.ndarray of shape (d, d)
        Covariance matrix Σ.  Coerced to at least 2-D.

    Returns
    -------
    tc : float
        Total correlation in nats.  Returns ``+inf`` if Σ is singular.

    Notes
    -----
    The computation uses:

        TC = (∑ᵢ ½ log(2πe σᵢ²)) − ½(d(1 + log 2π) + log|Σ|)
           = ½ ∑ᵢ log σᵢ² − ½ log|Σ|
           = −½ log|corr(Σ)|

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import total_correlation_gaussian
    >>> # Identity covariance → all marginals independent → TC = 0
    >>> tc = total_correlation_gaussian(np.eye(3))
    >>> np.isclose(tc, 0.0)
    True
    >>> # Correlated covariance → TC > 0
    >>> cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    >>> total_correlation_gaussian(cov) > 0
    True
    """
    cov = np.atleast_2d(cov)
    d = cov.shape[0]
    marginal_vars = np.diag(cov)  # σᵢ²
    # ∑ᵢ H(Xᵢ) = ∑ᵢ ½ log(2πe σᵢ²)
    sum_marg_h = 0.5 * np.sum(np.log(2 * np.pi * np.e * marginal_vars))
    sign, log_det = np.linalg.slogdet(cov)  # log|Σ|
    if sign <= 0:
        return np.inf  # singular Σ
    # H(X) = ½ (d(1 + log 2π) + log|Σ|)
    joint_h = 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)
    return float(sum_marg_h - joint_h)  # TC = sum H(Xi) - H(X)


def mutual_information_gaussian(
    cov_X: np.ndarray,
    cov_Y: np.ndarray,
    cov_XY: np.ndarray,
) -> float:
    """Analytic mutual information between two jointly Gaussian variables.

    Uses the entropy identity:

        MI(X; Y) = H(X) + H(Y) − H(X, Y)

    where each entropy is computed analytically from the corresponding
    covariance matrix via :func:`entropy_gaussian`.

    Parameters
    ----------
    cov_X : np.ndarray of shape (d_X, d_X)
        Marginal covariance of X.
    cov_Y : np.ndarray of shape (d_Y, d_Y)
        Marginal covariance of Y.
    cov_XY : np.ndarray of shape (d_X + d_Y, d_X + d_Y)
        Joint covariance matrix of the concatenated variable [X, Y].

    Returns
    -------
    mi : float
        Mutual information in nats.  Non-negative for valid covariance
        matrices; small negative values indicate numerical imprecision.

    Notes
    -----
    For Gaussians the MI can also be expressed as:

        MI(X; Y) = −½ log(|Σ_{XX}| · |Σ_{YY}| / |Σ_{XY}|)

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import mutual_information_gaussian
    >>> # Block-diagonal joint covariance → MI = 0
    >>> cov_X = np.eye(2)
    ... cov_Y = np.eye(2)
    >>> cov_XY = np.block([[cov_X, np.zeros((2, 2))], [np.zeros((2, 2)), cov_Y]])
    >>> mi = mutual_information_gaussian(cov_X, cov_Y, cov_XY)
    >>> np.isclose(mi, 0.0, atol=1e-10)
    True
    """
    hx = entropy_gaussian(cov_X)  # H(X)
    hy = entropy_gaussian(cov_Y)  # H(Y)
    hxy = entropy_gaussian(cov_XY)  # H(X, Y)
    return float(hx + hy - hxy)  # MI(X;Y) = H(X) + H(Y) - H(X,Y)


def kl_divergence_gaussian(
    mu0: np.ndarray,
    cov0: np.ndarray,
    mu1: np.ndarray,
    cov1: np.ndarray,
) -> float:
    """Analytic KL divergence KL(P₀ ‖ P₁) between two multivariate Gaussians.

    Both distributions are assumed to be multivariate Gaussian:

        P₀ = 𝒩(μ₀, Σ₀)   and   P₁ = 𝒩(μ₁, Σ₁)

    The closed-form KL divergence is:

        KL(P₀ ‖ P₁) = ½ [ tr(Σ₁⁻¹Σ₀) + (μ₁ − μ₀)ᵀ Σ₁⁻¹ (μ₁ − μ₀)
                           − d + log(|Σ₁| / |Σ₀|) ]

    Parameters
    ----------
    mu0 : np.ndarray of shape (d,)
        Mean of the *source* distribution P₀.
    cov0 : np.ndarray of shape (d, d)
        Covariance Σ₀ of the source distribution P₀.
    mu1 : np.ndarray of shape (d,)
        Mean of the *target* distribution P₁.
    cov1 : np.ndarray of shape (d, d)
        Covariance Σ₁ of the target distribution P₁.

    Returns
    -------
    kl : float
        KL divergence KL(P₀ ‖ P₁) in nats.  Always non-negative for valid
        covariance matrices.

    Notes
    -----
    The matrix inverse Σ₁⁻¹ is computed via ``np.linalg.inv``; for large d
    a Cholesky-based approach would be more numerically stable.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import kl_divergence_gaussian
    >>> # KL(P ‖ P) = 0 for identical distributions
    >>> mu = np.array([1.0, 2.0])
    >>> cov = np.array([[2.0, 0.5], [0.5, 1.5]])
    >>> kl = kl_divergence_gaussian(mu, cov, mu, cov)
    >>> np.isclose(kl, 0.0, atol=1e-10)
    True
    """
    mu0 = np.atleast_1d(mu0)
    mu1 = np.atleast_1d(mu1)
    cov0 = np.atleast_2d(cov0)
    cov1 = np.atleast_2d(cov1)
    d = len(mu0)
    cov1_inv = np.linalg.inv(cov1)  # Σ₁⁻¹, shape (d, d)
    diff = mu1 - mu0  # mu1 - mu0, shape (d,)
    _sign0, log_det0 = np.linalg.slogdet(cov0)  # log|Σ₀|
    _sign1, log_det1 = np.linalg.slogdet(cov1)  # log|Σ₁|
    trace_term = np.trace(cov1_inv @ cov0)  # tr(Σ₁⁻¹Σ₀)
    quad_term = diff @ cov1_inv @ diff  # (mu1-mu0)^T Sigma1^-1 (mu1-mu0)
    log_det_term = log_det1 - log_det0  # log(|Sigma1|/|Sigma0|)
    # KL = 0.5 [tr(Sigma1^-1 Sigma0) + quad - d + log(|Sigma1|/|Sigma0|)]
    return float(0.5 * (trace_term + quad_term - d + log_det_term))


# ---- Distribution samplers ----


def gaussian(
    n_samples: int = 1000,
    loc: float = 0.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a univariate Gaussian (normal) distribution.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    loc : float, optional (default=0.0)
        Mean μ of the distribution.
    scale : float, optional (default=1.0)
        Standard deviation σ > 0 of the distribution.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.  Pass an integer for
        reproducible results.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from 𝒩(loc, scale²).

    Examples
    --------
    >>> from rbig._src.parametric import gaussian
    >>> x = gaussian(n_samples=500, loc=2.0, scale=0.5, random_state=0)
    >>> x.shape
    (500,)
    >>> import numpy as np
    >>> np.isclose(x.mean(), 2.0, atol=0.1)
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=loc, scale=scale, size=n_samples)


def multivariate_gaussian(
    n_samples: int = 1000,
    mean: np.ndarray | None = None,
    cov: np.ndarray | None = None,
    d: int = 2,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a multivariate Gaussian distribution.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    mean : np.ndarray of shape (d,) or None, optional
        Mean vector μ.  Defaults to the zero vector of length ``d``.
    cov : np.ndarray of shape (d, d) or None, optional
        Covariance matrix Σ.  Must be symmetric positive semi-definite.
        Defaults to the identity matrix Iₐ.
    d : int, optional (default=2)
        Dimensionality used when ``mean`` is ``None``.  Ignored when
        ``mean`` is provided (its length determines the dimension).
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, d)
        Independent draws from 𝒩(mean, cov).

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import multivariate_gaussian
    >>> cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    >>> X = multivariate_gaussian(n_samples=200, cov=cov, d=2, random_state=7)
    >>> X.shape
    (200, 2)
    >>> np.isclose(np.corrcoef(X.T)[0, 1], 0.8, atol=0.1)
    True
    """
    rng = np.random.default_rng(random_state)
    if mean is None:
        mean = np.zeros(d)  # default: zero mean
    if cov is None:
        cov = np.eye(len(mean))  # default: identity covariance
    return rng.multivariate_normal(mean, cov, size=n_samples)


def uniform(
    n_samples: int = 1000,
    low: float = 0.0,
    high: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a continuous uniform distribution on [low, high).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    low : float, optional (default=0.0)
        Lower bound (inclusive) of the interval.
    high : float, optional (default=1.0)
        Upper bound (exclusive) of the interval.  Must satisfy ``high > low``.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from Uniform(low, high).

    Examples
    --------
    >>> from rbig._src.parametric import uniform
    >>> x = uniform(n_samples=300, low=-1.0, high=1.0, random_state=3)
    >>> x.shape
    (300,)
    >>> import numpy as np
    >>> np.all((x >= -1.0) & (x < 1.0))
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.uniform(low=low, high=high, size=n_samples)


def exponential(
    n_samples: int = 1000,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from an exponential distribution with given scale (mean).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    scale : float, optional (default=1.0)
        Scale parameter β = 1/λ (mean of the distribution).  Must be > 0.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from Exp(1/scale).  All values are non-negative.

    Examples
    --------
    >>> from rbig._src.parametric import exponential
    >>> x = exponential(n_samples=500, scale=2.0, random_state=1)
    >>> x.shape
    (500,)
    >>> import numpy as np
    >>> np.all(x >= 0)
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.exponential(scale=scale, size=n_samples)


def laplace(
    n_samples: int = 1000,
    loc: float = 0.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Laplace (double-exponential) distribution.

    The Laplace distribution has heavier tails than a Gaussian and is
    commonly used as a sparsity-inducing prior.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    loc : float, optional (default=0.0)
        Location parameter μ (mean and median of the distribution).
    scale : float, optional (default=1.0)
        Scale parameter b > 0 (half the variance is 2b²).
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from Laplace(loc, scale).

    Examples
    --------
    >>> from rbig._src.parametric import laplace
    >>> x = laplace(n_samples=400, loc=0.0, scale=1.0, random_state=5)
    >>> x.shape
    (400,)
    """
    rng = np.random.default_rng(random_state)
    return rng.laplace(loc=loc, scale=scale, size=n_samples)


def student_t(
    n_samples: int = 1000,
    df: float = 5.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Student-t distribution with ``df`` degrees of freedom.

    The Student-t distribution has heavier tails than a Gaussian; as
    ``df → ∞`` it converges to 𝒩(0, 1).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    df : float, optional (default=5.0)
        Degrees of freedom ν > 0.  Smaller values produce heavier tails.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from t(df).

    Examples
    --------
    >>> from rbig._src.parametric import student_t
    >>> x = student_t(n_samples=300, df=3.0, random_state=2)
    >>> x.shape
    (300,)
    """
    rng = np.random.default_rng(random_state)
    return rng.standard_t(df=df, size=n_samples)


def gamma(
    n_samples: int = 1000,
    shape: float = 2.0,
    scale: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Gamma distribution.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    shape : float, optional (default=2.0)
        Shape parameter k > 0 (also called α).
    scale : float, optional (default=1.0)
        Scale parameter θ > 0.  The mean of the distribution is k · θ.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from Gamma(shape, scale).  All values are positive.

    Examples
    --------
    >>> from rbig._src.parametric import gamma
    >>> x = gamma(n_samples=500, shape=3.0, scale=2.0, random_state=0)
    >>> x.shape
    (500,)
    >>> import numpy as np
    >>> np.all(x > 0)
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.gamma(shape=shape, scale=scale, size=n_samples)


def beta(
    n_samples: int = 1000,
    a: float = 2.0,
    b: float = 2.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Beta distribution on the interval (0, 1).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    a : float, optional (default=2.0)
        First shape parameter α > 0.
    b : float, optional (default=2.0)
        Second shape parameter β > 0.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from Beta(a, b).  Values are in (0, 1).

    Examples
    --------
    >>> from rbig._src.parametric import beta
    >>> x = beta(n_samples=400, a=2.0, b=5.0, random_state=4)
    >>> x.shape
    (400,)
    >>> import numpy as np
    >>> np.all((x > 0) & (x < 1))
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.beta(a=a, b=b, size=n_samples)


def lognormal(
    n_samples: int = 1000,
    mean: float = 0.0,
    sigma: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a log-normal distribution.

    If Y ~ 𝒩(mean, sigma²) then X = exp(Y) ~ LogNormal(mean, sigma²).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    mean : float, optional (default=0.0)
        Mean μ of the underlying normal distribution (i.e. log-scale mean).
    sigma : float, optional (default=1.0)
        Standard deviation σ > 0 of the underlying normal distribution
        (i.e. log-scale standard deviation).
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from LogNormal(mean, sigma²).  All values are
        strictly positive.

    Examples
    --------
    >>> from rbig._src.parametric import lognormal
    >>> x = lognormal(n_samples=500, mean=0.0, sigma=0.5, random_state=6)
    >>> x.shape
    (500,)
    >>> import numpy as np
    >>> np.all(x > 0)
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.lognormal(mean=mean, sigma=sigma, size=n_samples)


def dirichlet(
    n_samples: int = 1000,
    alpha: np.ndarray | None = None,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Dirichlet distribution.

    The Dirichlet is a multivariate generalisation of the Beta distribution.
    Each sample is a probability vector (non-negative, sums to 1).

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    alpha : np.ndarray of shape (k,) or None, optional
        Concentration parameter vector α.  All entries must be > 0.
        Defaults to ``np.ones(3)`` (uniform Dirichlet on 3-simplex).
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples, k)
        Independent draws from Dir(alpha).  Each row sums to 1 and all
        entries are non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import dirichlet
    >>> x = dirichlet(n_samples=200, alpha=np.array([1.0, 2.0, 3.0]), random_state=8)
    >>> x.shape
    (200, 3)
    >>> np.allclose(x.sum(axis=1), 1.0)
    True
    """
    rng = np.random.default_rng(random_state)
    if alpha is None:
        alpha = np.ones(3)  # default: uniform on 3-simplex
    return rng.dirichlet(alpha=alpha, size=n_samples)


def wishart(
    n_samples: int = 1000,
    df: int = 5,
    scale: np.ndarray | None = None,
    d: int = 3,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a Wishart distribution.

    The Wishart distribution is a matrix-variate generalisation of the
    chi-squared distribution and is the conjugate prior of the inverse
    covariance (precision) matrix of a multivariate Gaussian.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    df : int, optional (default=5)
        Degrees of freedom ν.  Must satisfy ν ≥ d.
    scale : np.ndarray of shape (d, d) or None, optional
        Positive-definite scale matrix V.  Defaults to the d × d identity
        matrix Iₐ (using ``d`` to infer the dimension when ``None``).
    d : int, optional (default=3)
        Dimension of the matrices; used only when ``scale`` is ``None``.
    random_state : int or None, optional (default=None)
        Seed for the random number generator (converted to a 32-bit integer
        for scipy compatibility).

    Returns
    -------
    samples : np.ndarray of shape (n_samples, d, d)
        Independent positive semi-definite matrices drawn from W(df, scale).

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import wishart
    >>> W = wishart(n_samples=50, df=5, d=3, random_state=10)
    >>> W.shape
    (50, 3, 3)
    >>> # Each sample should be symmetric
    >>> np.allclose(W[0], W[0].T)
    True
    """
    from scipy.stats import wishart as wishart_dist

    if scale is None:
        scale = np.eye(d)  # default: identity
    rng = np.random.default_rng(random_state)
    seed = int(rng.integers(0, 2**31))  # scipy needs int seed
    return wishart_dist.rvs(df=df, scale=scale, size=n_samples, random_state=seed)


def von_mises(
    n_samples: int = 1000,
    mu: float = 0.0,
    kappa: float = 1.0,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample from a von Mises distribution on the circle.

    The von Mises distribution is the circular analogue of the Gaussian and
    is parameterised by a mean angle μ and a concentration κ.  As κ → 0 the
    distribution approaches a uniform distribution on [−π, π); as κ → ∞ it
    concentrates around μ.

    Parameters
    ----------
    n_samples : int, optional (default=1000)
        Number of samples to draw.
    mu : float, optional (default=0.0)
        Mean direction in radians.
    kappa : float, optional (default=1.0)
        Concentration parameter κ ≥ 0.  Larger values produce more peaked
        distributions.
    random_state : int or None, optional (default=None)
        Seed for the random number generator.

    Returns
    -------
    samples : np.ndarray of shape (n_samples,)
        Independent draws from VonMises(mu, kappa) in radians,
        with values in [−π, π).

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.parametric import von_mises
    >>> x = von_mises(n_samples=500, mu=0.0, kappa=2.0, random_state=11)
    >>> x.shape
    (500,)
    >>> np.all((x >= -np.pi) & (x < np.pi))
    True
    """
    rng = np.random.default_rng(random_state)
    return rng.vonmises(mu=mu, kappa=kappa, size=n_samples)
