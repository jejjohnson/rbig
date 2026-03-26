"""Tests for new parametric IT formulas and distribution samplers."""

import numpy as np

from rbig import (
    beta,
    dirichlet,
    entropy_gaussian,
    exponential,
    gamma,
    gaussian,
    kl_divergence_gaussian,
    laplace,
    lognormal,
    multivariate_gaussian,
    mutual_information_gaussian,
    student_t,
    total_correlation_gaussian,
    uniform,
    von_mises,
    wishart,
)

# ---- Analytical IT formulas ----


def test_entropy_gaussian_scalar():
    cov = np.array([[1.0]])
    h = entropy_gaussian(cov)
    expected = 0.5 * (1 + np.log(2 * np.pi))
    assert abs(h - expected) < 1e-10


def test_entropy_gaussian_multivariate():
    cov = np.eye(3)
    h = entropy_gaussian(cov)
    expected = 0.5 * 3 * (1 + np.log(2 * np.pi))
    assert abs(h - expected) < 1e-10


def test_total_correlation_gaussian_identity():
    cov = np.eye(3)
    tc = total_correlation_gaussian(cov)
    # Independent Gaussians have TC = 0
    assert abs(tc) < 1e-10


def test_total_correlation_gaussian_correlated():
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    tc = total_correlation_gaussian(cov)
    assert tc > 0


def test_mutual_information_gaussian():
    cov_X = np.eye(2)
    cov_Y = np.eye(2)
    cov_XY = np.eye(4)
    mi = mutual_information_gaussian(cov_X, cov_Y, cov_XY)
    # Independent variables have MI = 0
    assert abs(mi) < 1e-10


def test_kl_divergence_gaussian_same():
    mu = np.zeros(2)
    cov = np.eye(2)
    kl = kl_divergence_gaussian(mu, cov, mu, cov)
    assert abs(kl) < 1e-10


def test_kl_divergence_gaussian_non_negative():
    mu0 = np.zeros(2)
    mu1 = np.array([1.0, 0.0])
    cov0 = np.eye(2)
    cov1 = np.eye(2)
    kl = kl_divergence_gaussian(mu0, cov0, mu1, cov1)
    assert kl >= 0


# ---- Distribution samplers ----


def test_gaussian_shape():
    x = gaussian(n_samples=100, random_state=42)
    assert x.shape == (100,)


def test_multivariate_gaussian_shape():
    x = multivariate_gaussian(n_samples=100, d=3, random_state=42)
    assert x.shape == (100, 3)


def test_uniform_shape():
    x = uniform(n_samples=100, random_state=42)
    assert x.shape == (100,)
    assert np.all(x >= 0)
    assert np.all(x <= 1)


def test_exponential_shape():
    x = exponential(n_samples=100, random_state=42)
    assert x.shape == (100,)
    assert np.all(x >= 0)


def test_laplace_shape():
    x = laplace(n_samples=100, random_state=42)
    assert x.shape == (100,)


def test_student_t_shape():
    x = student_t(n_samples=100, df=5.0, random_state=42)
    assert x.shape == (100,)


def test_gamma_shape():
    x = gamma(n_samples=100, random_state=42)
    assert x.shape == (100,)
    assert np.all(x >= 0)


def test_beta_shape():
    x = beta(n_samples=100, random_state=42)
    assert x.shape == (100,)
    assert np.all(x >= 0)
    assert np.all(x <= 1)


def test_lognormal_shape():
    x = lognormal(n_samples=100, random_state=42)
    assert x.shape == (100,)
    assert np.all(x > 0)


def test_dirichlet_shape():
    x = dirichlet(n_samples=100, alpha=np.array([1.0, 2.0, 3.0]), random_state=42)
    assert x.shape == (100, 3)
    np.testing.assert_allclose(np.sum(x, axis=1), 1.0, atol=1e-10)


def test_wishart_shape():
    x = wishart(n_samples=10, df=5, d=3, random_state=42)
    assert x.shape == (10, 3, 3)


def test_von_mises_shape():
    x = von_mises(n_samples=100, random_state=42)
    assert x.shape == (100,)


# ---- Transform coverage tests ----


def test_logit_transform_log_det_jacobian():
    """LogitTransform.log_det_jacobian returns correct shape for data in (0,1)."""
    from rbig._src.parametric import LogitTransform

    rng = np.random.default_rng(42)
    X = rng.uniform(0.01, 0.99, size=(50, 3))
    lt = LogitTransform().fit(X)
    ldj = lt.log_det_jacobian(X)
    assert ldj.shape == (50,)
    assert np.all(np.isfinite(ldj))
    # log_det should be positive (logit stretches near 0 and 1)
    assert np.all(ldj > 0)


def test_boxcox_fit_non_positive():
    """BoxCoxTransform with non-positive data sets lambda=0 for those features."""
    from rbig._src.parametric import BoxCoxTransform

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))  # contains negatives
    bc = BoxCoxTransform().fit(X)
    # Both features should get lambda=0 since data is not all positive
    np.testing.assert_allclose(bc.lambdas_, [0.0, 0.0])


def test_boxcox_inverse_transform():
    """BoxCoxTransform roundtrip: fit, transform, inverse_transform."""
    from rbig._src.parametric import BoxCoxTransform

    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal(size=(100, 2))) + 0.1  # positive data
    bc = BoxCoxTransform().fit(X)
    Xt = bc.transform(X)
    Xr = bc.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, X, atol=1e-5)


def test_boxcox_log_det_jacobian():
    """BoxCoxTransform.log_det_jacobian returns finite array of correct shape."""
    from rbig._src.parametric import BoxCoxTransform

    rng = np.random.default_rng(42)
    X = np.abs(rng.standard_normal(size=(80, 3))) + 0.1  # positive
    bc = BoxCoxTransform().fit(X)
    ldj = bc.log_det_jacobian(X)
    assert ldj.shape == (80,)
    assert np.all(np.isfinite(ldj))


def test_boxcox_log_det_jacobian_lambda_zero():
    """BoxCoxTransform.log_det_jacobian with lambda=0 (non-positive data)."""
    from rbig._src.parametric import BoxCoxTransform

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(60, 2))  # contains negatives
    bc = BoxCoxTransform().fit(X)
    # lambdas_ should be 0 → the lam=0 branch in log_det_jacobian
    ldj = bc.log_det_jacobian(X)
    assert ldj.shape == (60,)
    assert np.all(np.isfinite(ldj))


def test_quantile_transform_non_positive():
    """QuantileTransform handles data containing negatives."""
    from rbig._src.parametric import QuantileTransform

    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(200, 2))  # has negatives
    qt = QuantileTransform().fit(X)
    Xt = qt.transform(X)
    assert Xt.shape == X.shape
    assert np.all(np.isfinite(Xt))


def test_entropy_gaussian_singular():
    """entropy_gaussian returns -inf for a singular covariance matrix."""
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1 → singular
    h = entropy_gaussian(cov)
    assert h == -np.inf


def test_total_correlation_gaussian_singular():
    """total_correlation_gaussian returns +inf for singular covariance."""
    cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    tc = total_correlation_gaussian(cov)
    assert tc == np.inf


def test_multivariate_gaussian_defaults():
    """multivariate_gaussian with default mean/cov uses zero mean and identity."""
    x = multivariate_gaussian(n_samples=50, d=3, random_state=42)
    assert x.shape == (50, 3)


def test_dirichlet_default_alpha():
    """dirichlet with no alpha uses default uniform on 3-simplex."""
    x = dirichlet(n_samples=50, random_state=42)
    assert x.shape == (50, 3)
    np.testing.assert_allclose(np.sum(x, axis=1), 1.0, atol=1e-10)


def test_wishart_default_scale():
    """wishart with no scale uses identity matrix."""
    x = wishart(n_samples=10, df=5, random_state=42)
    assert x.shape == (10, 3, 3)
    # Each sample should be symmetric (positive semi-definite)
    np.testing.assert_allclose(x[0], x[0].T)
