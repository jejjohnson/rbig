"""Density estimation utilities for RBIG.

This module provides functions for estimating marginal and joint entropies,
total correlation, and a set of elementary bijectors (Tanh, Exp, Cube) used
as building blocks inside RBIG pipelines.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def marginal_entropy(X: np.ndarray, correction: bool = True) -> np.ndarray:
    """Estimate the marginal (per-dimension) differential entropy using KDE.

    For each dimension i, kernel density estimation (KDE) is used to obtain
    a smooth density estimate f̂(xᵢ), and the entropy is approximated as

        H(Xᵢ) = −𝔼[log f̂(xᵢ)] ≈ −(1/N) ∑ⱼ log f̂(xᵢⱼ)

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.  Each column is treated as an independent marginal.
    correction : bool, optional (default=True)
        Placeholder flag for an optional finite-sample bias correction.
        Currently unused in the computation.

    Returns
    -------
    entropies : np.ndarray of shape (n_features,)
        Estimated differential entropy (nats) for each feature dimension.

    Notes
    -----
    The KDE bandwidth is chosen by Scott's rule (scipy default).  A small
    constant (1e-300) is added inside the log to prevent ``−∞`` values when
    the density estimate is numerically zero.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import marginal_entropy
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> h = marginal_entropy(X)
    >>> h.shape
    (3,)
    >>> # Gaussian entropy is 0.5*(1 + ln 2π) ≈ 1.419 nats
    >>> np.allclose(h, 0.5 * (1 + np.log(2 * np.pi)), atol=0.15)
    True
    """
    _n_samples, n_features = X.shape
    entropies = np.zeros(n_features)
    for i in range(n_features):
        kde = stats.gaussian_kde(X[:, i])  # fit KDE for column i
        log_density = np.log(kde(X[:, i]) + 1e-300)  # evaluate log f̂(xᵢ)
        entropies[i] = -np.mean(log_density)  # H(Xi) ~= -E[log f_hat]
    return entropies


def joint_entropy_gaussian(X: np.ndarray) -> float:
    """Entropy of a multivariate Gaussian fitted to the covariance of X.

    Treats the data as if it were drawn from a multivariate Gaussian with the
    empirical covariance Σ = cov(X) and computes the analytic entropy:

        H(X) = ½ log|2πeΣ| = ½ (d(1 + log 2π) + log|Σ|)

    where d is the number of features and |·| denotes the matrix determinant.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix used to estimate the covariance.

    Returns
    -------
    entropy : float
        Differential entropy (nats) of the fitted Gaussian.  Returns ``-inf``
        if the covariance matrix is singular or not positive definite.

    Notes
    -----
    The log-determinant is computed via ``np.linalg.slogdet`` for numerical
    stability.  When the matrix is singular (sign ≤ 0), ``-inf`` is returned.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import joint_entropy_gaussian
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((1000, 2))
    >>> h = joint_entropy_gaussian(X)
    >>> # For N(0, I₂): H = ½(2(1+log 2π)) ≈ 2.838 nats
    >>> np.isfinite(h)
    True
    """
    _n, d = X.shape
    cov = np.cov(X.T)  # empirical covariance, shape (d, d)
    if d == 1:
        cov = np.array([[cov]])  # ensure 2-D for slogdet
    sign, log_det = np.linalg.slogdet(cov)  # stable log|Σ|
    if sign <= 0:
        log_det = -np.inf  # singular / not PD covariance
    # H = ½ (d(1 + log 2π) + log|Σ|)
    return 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)


def total_correlation(X: np.ndarray) -> float:
    """Estimate the Total Correlation (multivariate mutual information) of X.

    Total Correlation (TC) measures the statistical dependence among all
    dimensions simultaneously:

        TC(X) = ∑ᵢ H(Xᵢ) − H(X)

    Marginal entropies H(Xᵢ) are estimated via KDE (see `marginal_entropy`),
    while the joint entropy H(X) is approximated by fitting a multivariate
    Gaussian to the data (see `joint_entropy_gaussian`).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    tc : float
        Estimated total correlation in nats.  A value near zero indicates
        near-independence across dimensions.

    Notes
    -----
    Because the joint term uses a Gaussian approximation, TC may be slightly
    biased for non-Gaussian data.  For a fully non-parametric estimate use
    ``total_correlation_rbig`` from ``rbig._src.metrics``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import total_correlation
    >>> rng = np.random.default_rng(0)
    >>> X_indep = rng.standard_normal((500, 3))  # independent columns → TC ≈ 0
    >>> tc = total_correlation(X_indep)
    >>> tc >= -0.5  # should be close to 0
    True
    """
    marg_h = marginal_entropy(X)  # ∑ᵢ H(Xᵢ) via KDE, shape (n_features,)
    joint_h = joint_entropy_gaussian(X)  # H(X) via Gaussian approximation
    return float(np.sum(marg_h) - joint_h)


def gaussian_entropy(n_features: int, cov: np.ndarray | None = None) -> float:
    """Analytic entropy of a multivariate Gaussian distribution.

    Computes the differential entropy of 𝒩(μ, Σ):

        H = ½ log|2πeΣ| = ½ (d(1 + log 2π) + log|Σ|)

    where d = ``n_features`` and |·| denotes the determinant.

    Parameters
    ----------
    n_features : int
        Dimensionality d of the distribution.
    cov : np.ndarray of shape (n_features, n_features) or None
        Covariance matrix Σ.  If ``None``, the identity matrix is assumed
        (i.e. Σ = Iₐ), and the entropy reduces to
        ``½ · d · (1 + log 2π)``.

    Returns
    -------
    entropy : float
        Differential entropy in nats.  Returns ``-inf`` if the covariance
        matrix is singular or not positive definite.

    Notes
    -----
    ``np.linalg.slogdet`` is used for numerically stable log-determinant
    computation.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import gaussian_entropy
    >>> # Identity covariance: H = 0.5 * d * (1 + log 2π)
    >>> h = gaussian_entropy(2, cov=np.eye(2))
    >>> np.isclose(h, 0.5 * 2 * (1 + np.log(2 * np.pi)))
    True
    >>> # None ⟹ identity covariance assumed
    >>> gaussian_entropy(2) == gaussian_entropy(2, cov=np.eye(2))
    True
    """
    if cov is None:
        # Σ = I_d  ⟹  log|Σ| = 0
        return 0.5 * n_features * (1 + np.log(2 * np.pi))
    sign, log_det = np.linalg.slogdet(cov)  # stable log|Σ|
    if sign <= 0:
        return -np.inf  # singular covariance
    # H = ½ (d(1 + log 2π) + log|Σ|)
    return 0.5 * (n_features * (1 + np.log(2 * np.pi)) + log_det)


def entropy_reduction(X_before: np.ndarray, X_after: np.ndarray) -> float:
    """Compute the reduction in Total Correlation between two representations.

    Measures how much statistical dependence is removed by a transformation:

        ΔTC = TC(X_before) − TC(X_after)

    A positive value indicates that the transformation has reduced the total
    correlation (i.e. made the dimensions more independent).

    Parameters
    ----------
    X_before : np.ndarray of shape (n_samples, n_features)
        Data matrix before the transformation.
    X_after : np.ndarray of shape (n_samples, n_features)
        Data matrix after the transformation.  Must have the same number of
        samples and features as ``X_before``.

    Returns
    -------
    delta_tc : float
        Reduction in total correlation (nats).  Positive values indicate
        increased statistical independence after the transformation.

    Notes
    -----
    Both TC values are estimated using :func:`total_correlation`, which
    internally uses KDE for marginal entropies and a Gaussian approximation
    for the joint entropy.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import entropy_reduction
    >>> rng = np.random.default_rng(1)
    >>> # Correlated before, independent after whitening
    >>> A = np.array([[1.0, 0.9], [0.9, 1.0]])
    >>> X = rng.multivariate_normal([0, 0], A, size=500)
    >>> X_white = (X - X.mean(0)) / X.std(0)
    >>> entropy_reduction(X, X_white) >= 0
    True
    """
    tc_before = total_correlation(X_before)  # TC of original representation
    tc_after = total_correlation(X_after)  # TC of transformed representation
    return tc_before - tc_after  # DeltaTC = TC_before - TC_after


from rbig._src.base import Bijector


class Tanh(Bijector):
    """Elementwise hyperbolic-tangent bijector with analytic log-det Jacobian.

    Implements the smooth, bounded map:

        Forward  : y = tanh(x)
        Inverse  : x = arctanh(y)   (y clipped to (−1+ε, 1−ε) for stability)
        Log-det  : ∑ᵢ log(1 − tanh²(xᵢ)) = ∑ᵢ log sech²(xᵢ)

    Because tanh saturates near ±1, the inverse is numerically clipped to
    avoid divergence.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import Tanh
    >>> X = np.array([[0.0, 1.0], [-1.0, 0.5]])
    >>> bij = Tanh().fit(X)
    >>> Y = bij.transform(X)  # y = tanh(x)
    >>> X_rec = bij.inverse_transform(Y)  # x = arctanh(y)
    >>> np.allclose(X, X_rec, atol=1e-5)
    True
    """

    def fit(self, X: np.ndarray) -> Tanh:
        """No-op fit (stateless bijector).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Ignored.

        Returns
        -------
        self : Tanh
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward tanh map.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in ℝ.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Transformed data in (−1, 1).
        """
        return np.tanh(X)  # y = tanh(x), element-wise

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse arctanh map with clipping for stability.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in (−1, 1).  Values outside this range are clipped to
            (−1 + 1e-6, 1 − 1e-6).

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Reconstructed data in ℝ.
        """
        # Clip to open interval to prevent arctanh(±1) = ±∞
        return np.arctanh(np.clip(X, -1 + 1e-6, 1 - 1e-6))

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample log |det J| of the forward transform.

        The Jacobian of tanh is diagonal with entries
        d(tanh(xᵢ))/dxᵢ = 1 − tanh²(xᵢ) = sech²(xᵢ), so:

            log |det J| = ∑ᵢ log(1 − tanh²(xᵢ))

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (pre-transform).

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Log absolute determinant of the Jacobian for each sample.
        """
        # 1 - tanh^2(xi) is the diagonal Jacobian; sum over features
        return np.sum(np.log(1 - np.tanh(X) ** 2 + 1e-300), axis=1)


class Exp(Bijector):
    """Elementwise exponential bijector with analytic log-det Jacobian.

    Implements the map from ℝ to ℝ₊:

        Forward  : y = exp(x)
        Inverse  : x = log(y)   (y clipped to (ε, ∞) for stability)
        Log-det  : ∑ᵢ xᵢ          (since d(exp xᵢ)/dxᵢ = exp xᵢ, log = xᵢ)

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import Exp
    >>> X = np.array([[-1.0, 0.0], [1.0, 2.0]])
    >>> bij = Exp().fit(X)
    >>> Y = bij.transform(X)
    >>> np.allclose(bij.inverse_transform(Y), X, atol=1e-10)
    True
    """

    def fit(self, X: np.ndarray) -> Exp:
        """No-op fit (stateless bijector).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Ignored.

        Returns
        -------
        self : Exp
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward exponential map y = exp(x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in ℝ.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Transformed data in ℝ₊.
        """
        return np.exp(X)  # y = exp(x), element-wise

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse log map x = log(y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in ℝ₊.  Values below 1e-300 are clipped for stability.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Reconstructed data in ℝ.
        """
        return np.log(np.maximum(X, 1e-300))  # clip to prevent log(0)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample log |det J| of the forward transform.

        The Jacobian of exp is diagonal with entries exp(xᵢ), so:

            log |det J| = ∑ᵢ xᵢ

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (pre-transform).

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Log absolute determinant of the Jacobian for each sample.
        """
        return np.sum(X, axis=1)  # log |det J| = ∑ᵢ xᵢ


class Cube(Bijector):
    """Elementwise cube bijector (y = x³) with analytic log-det Jacobian.

    Implements the strictly monotone map:

        Forward  : y = x³
        Inverse  : x = y^{1/3}  (real cube root, handles negative values)
        Log-det  : ∑ᵢ log(3xᵢ²)

    Because the derivative vanishes at x = 0, a small constant (1e-300) is
    added inside the log to avoid ``−∞``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import Cube
    >>> X = np.array([[-2.0, 1.0], [0.5, -0.5]])
    >>> bij = Cube().fit(X)
    >>> Y = bij.transform(X)  # y = x³
    >>> X_rec = bij.inverse_transform(Y)
    >>> np.allclose(X, X_rec, atol=1e-10)
    True
    """

    def fit(self, X: np.ndarray) -> Cube:
        """No-op fit (stateless bijector).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Ignored.

        Returns
        -------
        self : Cube
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward cube map y = x³.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in ℝ.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Transformed data, element-wise cubed.
        """
        return X**3  # y = x³, element-wise

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse cube-root map x = y^{1/3}.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Cubed data in ℝ (may include negative values).

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Real cube root of each element.
        """
        return np.cbrt(X)  # real cube root, handles negative values

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample log |det J| of the forward transform.

        The Jacobian of x³ is diagonal with entries 3xᵢ², so:

            log |det J| = ∑ᵢ log(3xᵢ²)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data (pre-transform).

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Log absolute determinant of the Jacobian for each sample.
        """
        # 1e-300 guards against log(0) when x = 0
        return np.sum(np.log(3 * X**2 + 1e-300), axis=1)


def check_density(model, X: np.ndarray, n_grid: int = 1000) -> float:
    """Verify that the model density approximately integrates to one.

    Uses importance sampling with a standard-Gaussian proposal q = 𝒩(0, I):

        𝔼_q[p(x)/q(x)] = ∫ p(x) dx ≈ 1

    An estimate close to 1 confirms that the model is a properly normalised
    density; values far from 1 suggest numerical issues in the model.

    Parameters
    ----------
    model : fitted RBIG model
        Must expose a ``score_samples(Z)`` method that returns
        log p(z) for each row of Z.
    X : np.ndarray of shape (n_samples, n_features)
        Reference data used only to infer the feature dimensionality.
    n_grid : int, optional (default=1000)
        Number of Monte-Carlo samples drawn from the proposal q.

    Returns
    -------
    estimate : float
        Importance-weighted average 𝔼_q[p(x)/q(x)].  Should be close to 1
        for a well-calibrated model.

    Notes
    -----
    Log-ratios log(p/q) are clipped to [−500, 500] before exponentiation to
    prevent numerical overflow or underflow.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.densities import check_density
    >>> # A mock model that returns N(0,I) log-density
    >>> class GaussModel:
    ...     def score_samples(self, Z):
    ...         return -0.5 * np.sum(Z**2, axis=1) - Z.shape[1] * 0.5 * np.log(
    ...             2 * np.pi
    ...         )
    >>> X_ref = np.zeros((10, 2))
    >>> est = check_density(GaussModel(), X_ref, n_grid=5000)
    >>> np.isclose(est, 1.0, atol=0.15)
    True
    """
    # Sample from the proposal q = N(0, I_d)
    Z = stats.norm.rvs(size=(n_grid, X.shape[1]))  # shape (n_grid, d)
    log_p = model.score_samples(Z)  # log p(z), shape (n_grid,)
    log_q = np.sum(stats.norm.logpdf(Z), axis=1)  # log q(z) = ∑ᵢ log N(zᵢ)
    log_ratio = log_p - log_q  # log(p/q), shape (n_grid,)
    log_ratio = np.clip(log_ratio, -500, 500)  # numerical safety
    return float(np.mean(np.exp(log_ratio)))  # E_q[p/q] ~= integral p dx
