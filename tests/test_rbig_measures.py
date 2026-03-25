"""Tests for RBIG-way information theory measures (per-layer TC reduction).

Unit tests run by default; integration tests require ``pytest -m integration``.
"""

import numpy as np
import pytest

from rbig import (
    AnnealedRBIG,
    entropy_rbig,
    entropy_rbig_reduction,
    estimate_entropy,
    estimate_kld,
    estimate_mi,
    estimate_tc,
    kl_divergence_rbig_reduction,
    mutual_information_rbig_reduction,
    total_correlation_rbig_reduction,
)

# ---------------------------------------------------------------------------
# Shared RBIG kwargs (fast settings for unit tests)
# ---------------------------------------------------------------------------
_FAST = dict(n_layers=30, rotation="pca", patience=5, random_state=42)


# ---------------------------------------------------------------------------
# Unit tests (fast, default run)
# ---------------------------------------------------------------------------


class TestTotalCorrelationReduction:
    def test_returns_float(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        tc = total_correlation_rbig_reduction(model)
        assert isinstance(tc, float)

    def test_model_method_matches(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        assert (
            total_correlation_rbig_reduction(model)
            == model.total_correlation_reduction()
        )

    def test_non_negative(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        assert model.total_correlation_reduction() >= -0.05


class TestEntropyRbigReduction:
    def test_returns_float(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        h = entropy_rbig_reduction(model, simple_2d)
        assert isinstance(h, float)

    def test_positive(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        h = entropy_rbig_reduction(model, simple_2d)
        assert h > 0


class TestMutualInformationRbigReduction:
    def test_returns_float(self, simple_2d):
        X = simple_2d[:, :1]
        Y = simple_2d[:, 1:]
        model_X = AnnealedRBIG(**_FAST)
        model_Y = AnnealedRBIG(**_FAST)
        model_X.fit(X)
        model_Y.fit(Y)
        mi = mutual_information_rbig_reduction(
            model_X,
            model_Y,
            X,
            Y,
            rbig_kwargs=_FAST,
        )
        assert isinstance(mi, float)


class TestKLDivergenceRbigReduction:
    def test_returns_float(self, simple_2d):
        model = AnnealedRBIG(**_FAST)
        model.fit(simple_2d)
        kld = kl_divergence_rbig_reduction(model, simple_2d, rbig_kwargs=_FAST)
        assert isinstance(kld, float)


class TestEstimateTC:
    def test_independent_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 2)  # independent columns
        tc = estimate_tc(X, **_FAST)
        assert abs(tc) < 0.5, f"TC of independent data = {tc:.4f}, expected ≈ 0"


class TestEstimateEntropy:
    def test_returns_float(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 2)
        h = estimate_entropy(X, **_FAST)
        assert isinstance(h, float)
        assert h > 0


class TestEstimateMI:
    def test_returns_float(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 1)
        Y = rng.randn(300, 1)
        mi = estimate_mi(X, Y, **_FAST)
        assert isinstance(mi, float)


class TestEstimateKLD:
    def test_same_distribution_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(300, 2)
        Y = rng.randn(300, 2)  # same distribution
        kld = estimate_kld(X, Y, **_FAST)
        assert abs(kld) < 1.0, f"KLD(P||P) = {kld:.4f}, expected ≈ 0"


# ---------------------------------------------------------------------------
# Integration tests (accuracy, slow)
# ---------------------------------------------------------------------------

pytestmark_integration = pytest.mark.integration


@pytest.fixture
def gaussian_2d_large():
    rng = np.random.RandomState(42)
    return rng.randn(2000, 2)


@pytest.fixture
def correlated_2d_large():
    rng = np.random.RandomState(42)
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    return rng.multivariate_normal(np.zeros(2), cov, size=2000), cov


_ACCURATE = dict(n_layers=50, rotation="pca", patience=10, random_state=42)


@pytest.mark.integration
class TestTCAccuracy:
    def test_independent_gaussian(self, gaussian_2d_large):
        tc = estimate_tc(gaussian_2d_large, **_ACCURATE)
        assert abs(tc) < 0.15, f"TC of independent Gaussian = {tc:.4f}, expected ≈ 0"

    def test_correlated_gaussian(self, correlated_2d_large):
        X, cov = correlated_2d_large
        tc = estimate_tc(X, **_ACCURATE)
        # Analytical TC for bivariate Gaussian:
        # TC = -0.5 * log(det(cov) / prod(diag(cov)))
        tc_true = -0.5 * np.log(np.linalg.det(cov) / np.prod(np.diag(cov)))
        assert abs(tc - tc_true) < 0.15, f"TC = {tc:.4f}, analytical = {tc_true:.4f}"


@pytest.mark.integration
class TestEntropyAccuracy:
    def test_standard_normal(self, gaussian_2d_large):
        h = estimate_entropy(gaussian_2d_large, **_ACCURATE)
        D = gaussian_2d_large.shape[1]
        h_true = D * 0.5 * (1 + np.log(2 * np.pi))
        assert abs(h - h_true) < 0.2, f"entropy = {h:.4f}, true = {h_true:.4f}"

    def test_correlated_gaussian(self, correlated_2d_large):
        X, cov = correlated_2d_large
        h = estimate_entropy(X, **_ACCURATE)
        h_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * cov))
        assert abs(h - h_true) < 0.25, f"entropy = {h:.4f}, true = {h_true:.4f}"


@pytest.mark.integration
class TestMIAccuracy:
    def test_independent_near_zero(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1500, 1)
        Y = rng.randn(1500, 1)
        mi = estimate_mi(X, Y, **_ACCURATE)
        assert abs(mi) < 0.2, f"MI(independent) = {mi:.4f}, expected ≈ 0"

    def test_correlated_gaussian(self):
        rng = np.random.RandomState(42)
        rho = 0.8
        cov = np.array([[1.0, rho], [rho, 1.0]])
        XY = rng.multivariate_normal(np.zeros(2), cov, size=2000)
        X, Y = XY[:, :1], XY[:, 1:]
        mi = estimate_mi(X, Y, **_ACCURATE)
        mi_true = -0.5 * np.log(1 - rho**2)
        assert abs(mi - mi_true) < 0.2, f"MI = {mi:.4f}, analytical = {mi_true:.4f}"


@pytest.mark.integration
class TestKLDAccuracy:
    def test_same_distribution(self):
        rng = np.random.RandomState(42)
        X = rng.randn(1500, 2)
        Y = rng.randn(1500, 2)
        kld = estimate_kld(X, Y, **_ACCURATE)
        assert abs(kld) < 0.3, f"KLD(P||P) = {kld:.4f}, expected ≈ 0"

    def test_shifted_mean(self):
        rng = np.random.RandomState(42)
        X = rng.randn(2000, 2) + np.array([1.0, 0.0])  # shifted
        Y = rng.randn(2000, 2)  # N(0, I)
        kld = estimate_kld(X, Y, **_ACCURATE)
        # KLD(N(mu, I) || N(0, I)) = 0.5 * ||mu||^2 = 0.5
        kld_true = 0.5
        assert abs(kld - kld_true) < 0.4, (
            f"KLD = {kld:.4f}, analytical = {kld_true:.4f}"
        )


@pytest.mark.integration
class TestCrossConsistency:
    def test_entropy_methods_agree(self, gaussian_2d_large):
        """RBIG-reduction entropy should roughly agree with change-of-vars entropy."""
        model = AnnealedRBIG(**_ACCURATE)
        model.fit(gaussian_2d_large)

        h_cov = entropy_rbig(model, gaussian_2d_large)  # change-of-variables
        h_red = entropy_rbig_reduction(model, gaussian_2d_large)  # RBIG-way

        assert abs(h_cov - h_red) < 0.3, (
            f"change-of-vars entropy = {h_cov:.4f}, "
            f"RBIG-reduction entropy = {h_red:.4f}"
        )
