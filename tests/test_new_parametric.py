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
