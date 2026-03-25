"""Integration tests: accuracy on known distributions.

These tests verify numerical correctness — not just shapes and types —
by comparing RBIG outputs against analytical values for distributions
where closed-form answers exist.

All tests in this file are marked ``@pytest.mark.integration`` and are
**excluded from the default pytest run** (see pyproject.toml). Run them
explicitly with::

    pytest -m integration          # only integration tests
    pytest -m ""                   # everything (unit + integration)

The bug that motivated these tests: MarginalGaussianize.log_det_jacobian
used a spacing-based density estimator with +0.58 nats/feature/layer bias,
causing entropy estimates to be off by 10-20x. Shape-only tests did not
catch this because the outputs had the right shape and type.
"""

import numpy as np
import pytest
from scipy import stats

from rbig import (
    AnnealedRBIG,
    MarginalGaussianize,
    MarginalUniformize,
    entropy_rbig,
    mutual_information_rbig,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Fixtures: large enough for statistical accuracy
# ---------------------------------------------------------------------------


@pytest.fixture
def gaussian_2d():
    """Standard 2D Gaussian, N=2000."""
    rng = np.random.RandomState(42)
    return rng.randn(2000, 2)


@pytest.fixture
def gaussian_2d_scaled():
    """2D Gaussian with known covariance, N=2000."""
    rng = np.random.RandomState(42)
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    return rng.multivariate_normal(np.zeros(2), cov, size=2000), cov


# ---------------------------------------------------------------------------
# Level 1: MarginalGaussianize log_det_jacobian
# ---------------------------------------------------------------------------


class TestMarginalGaussianizeLogDet:
    """log_det_jacobian must be unbiased for known distributions."""

    def test_identity_on_standard_normal(self, gaussian_2d):
        """For N(0,1) input, the transform is ~identity, so log|det J| ≈ 0."""
        mg = MarginalGaussianize()
        mg.fit(gaussian_2d)
        ldj = mg.log_det_jacobian(gaussian_2d)
        # Mean should be near zero; allow statistical noise
        assert abs(ldj.mean()) < 0.1, (
            f"log_det_jacobian mean = {ldj.mean():.4f}, expected ≈ 0 "
            f"for standard normal input"
        )

    def test_scaled_gaussian(self, gaussian_2d_scaled):
        """For N(0, sigma^2) input, log|det J| should average to log(sigma) per dim."""
        X, cov = gaussian_2d_scaled
        mg = MarginalGaussianize()
        mg.fit(X)
        ldj = mg.log_det_jacobian(X)
        # For the change-of-variables: if we Gaussianize N(0, sigma^2) to N(0,1),
        # the mean log p(x) should give us the correct entropy.
        # We check that entropy computed from this log_det is close to truth.
        Z = mg.transform(X)
        log_pz = np.sum(stats.norm.logpdf(Z), axis=1)
        h_estimated = -np.mean(log_pz + ldj)
        h_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))
        assert abs(h_estimated - h_true) < 0.15, (
            f"Entropy from log_det_jacobian = {h_estimated:.4f}, "
            f"true = {h_true:.4f}"
        )


# ---------------------------------------------------------------------------
# Level 2: MarginalUniformize inverse accuracy
# ---------------------------------------------------------------------------


class TestMarginalUniformizeInverse:
    def test_roundtrip_recovery(self, gaussian_2d):
        """Inverse transform should recover original data (within interpolation error)."""
        t = MarginalUniformize()
        t.fit(gaussian_2d)
        Xt = t.transform(gaussian_2d)
        Xr = t.inverse_transform(Xt)
        # Interpolation-based inverse has small errors, especially at boundaries
        # 99%+ of points recover within 0.01; tail points may have
        # larger errors due to empirical CDF interpolation at boundaries
        median_err = np.median(np.abs(Xr - gaussian_2d))
        assert median_err < 0.01, f"median roundtrip error = {median_err:.4e}"
        pct99_err = np.percentile(np.abs(Xr - gaussian_2d), 99)
        assert pct99_err < 0.1, f"99th percentile roundtrip error = {pct99_err:.4e}"


class TestMarginalGaussianizeInverse:
    def test_roundtrip_recovery(self, gaussian_2d):
        """Inverse transform should recover original data (within interpolation error)."""
        t = MarginalGaussianize()
        t.fit(gaussian_2d)
        Xt = t.transform(gaussian_2d)
        Xr = t.inverse_transform(Xt)
        # 99%+ of points recover within 0.01; tail points may have
        # larger errors due to empirical CDF interpolation at boundaries
        median_err = np.median(np.abs(Xr - gaussian_2d))
        assert median_err < 0.01, f"median roundtrip error = {median_err:.4e}"
        pct99_err = np.percentile(np.abs(Xr - gaussian_2d), 99)
        assert pct99_err < 0.1, f"99th percentile roundtrip error = {pct99_err:.4e}"


# ---------------------------------------------------------------------------
# Level 3: AnnealedRBIG score_samples / entropy
# ---------------------------------------------------------------------------


class TestRBIGScoreSamples:
    def test_score_samples_on_standard_normal(self, gaussian_2d):
        """Mean log-likelihood of N(0,I) data should match analytical value."""
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(gaussian_2d)
        mean_ll = model.score(gaussian_2d)
        # Analytical: E[log p(x)] = -H(X) = -D/2 * (1 + log(2*pi))
        D = gaussian_2d.shape[1]
        expected_ll = -D * 0.5 * (1 + np.log(2 * np.pi))
        assert (
            abs(mean_ll - expected_ll) < 0.1
        ), f"mean log-likelihood = {mean_ll:.4f}, expected = {expected_ll:.4f}"

    def test_score_matches_cached_path(self, gaussian_2d):
        """score_samples (full forward pass) should match score_samples_raw_ (cached)."""
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(gaussian_2d)
        ll_forward = model.score_samples(gaussian_2d)
        ll_cached = model.score_samples_raw_()
        np.testing.assert_allclose(ll_forward, ll_cached, atol=1e-6)


class TestRBIGEntropy:
    def test_standard_normal(self, gaussian_2d):
        """Entropy of N(0,I) should match analytical value."""
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(gaussian_2d)
        h = model.entropy()
        D = gaussian_2d.shape[1]
        h_true = D * 0.5 * (1 + np.log(2 * np.pi))
        assert abs(h - h_true) < 0.1, f"entropy = {h:.4f}, true = {h_true:.4f}"

    def test_scaled_gaussian(self, gaussian_2d_scaled):
        """Entropy of N(0, Sigma) should match analytical value."""
        X, cov = gaussian_2d_scaled
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(X)
        h = model.entropy()
        h_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))
        assert abs(h - h_true) < 0.2, f"entropy = {h:.4f}, true = {h_true:.4f}"

    def test_entropy_positive(self, gaussian_2d):
        """Differential entropy of a continuous distribution should be reasonable."""
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(gaussian_2d)
        h = model.entropy()
        # For 2D data, entropy should be in a reasonable range (0 to ~10 nats)
        assert h > 0, f"entropy = {h:.4f}, expected > 0 for 2D Gaussian"
        assert h < 10, f"entropy = {h:.4f}, suspiciously large for 2D data"


# ---------------------------------------------------------------------------
# Level 4: entropy_rbig functional API
# ---------------------------------------------------------------------------


class TestEntropyRBIGFunc:
    def test_matches_model_entropy(self, gaussian_2d):
        """entropy_rbig(model, X) should match model.entropy()."""
        model = AnnealedRBIG(n_layers=30, rotation="pca", patience=10, random_state=42)
        model.fit(gaussian_2d)
        h_method = model.entropy()
        h_func = entropy_rbig(model, gaussian_2d)
        assert abs(h_method - h_func) < 0.1, (
            f"model.entropy() = {h_method:.4f}, " f"entropy_rbig() = {h_func:.4f}"
        )


# ---------------------------------------------------------------------------
# Level 5: Mutual Information on independent data
# ---------------------------------------------------------------------------


class TestMutualInformationAccuracy:
    def test_independent_variables_low_mi(self):
        """MI between independent 1D variables should be near zero."""
        rng = np.random.RandomState(42)
        # Use 1D variables to avoid KDE curse-of-dimensionality bias
        X = rng.randn(1500, 1)
        Y = rng.randn(1500, 1)

        kwargs = dict(n_layers=50, rotation="pca", patience=10, random_state=42)
        model_x = AnnealedRBIG(**kwargs)
        model_y = AnnealedRBIG(**kwargs)
        model_xy = AnnealedRBIG(**kwargs)

        model_x.fit(X)
        model_y.fit(Y)
        model_xy.fit(np.hstack([X, Y]))

        mi = mutual_information_rbig(model_x, model_y, model_xy)
        assert (
            abs(mi) < 0.2
        ), f"MI between independent variables = {mi:.4f}, expected ≈ 0"
