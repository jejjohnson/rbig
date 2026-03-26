"""Tests for marginal Gaussianizers and migrated utilities."""

import numpy as np

from rbig import (
    GMMGaussianizer,
    KDEGaussianizer,
    MarginalUniformize,
    QuantileGaussianizer,
    SplineGaussianizer,
    make_cdf_monotonic,
)

# ── Gaussianizer tests ────────────────────────────────────────────────────


def test_quantile_gaussianizer_shape(simple_2d):
    t = QuantileGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_quantile_gaussianizer_gaussian(simple_2d):
    t = QuantileGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert abs(np.mean(Xt)) < 0.5


def test_quantile_gaussianizer_inverse(simple_2d):
    t = QuantileGaussianizer()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-5)


def test_quantile_gaussianizer_log_det(simple_2d):
    t = QuantileGaussianizer()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)


def test_kde_gaussianizer_shape(simple_2d):
    t = KDEGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_kde_gaussianizer_gaussian(simple_2d):
    t = KDEGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert abs(np.mean(Xt)) < 0.5


def test_kde_gaussianizer_log_det(simple_2d):
    t = KDEGaussianizer()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)


def test_gmm_gaussianizer_shape(simple_2d):
    t = GMMGaussianizer(n_components=3)
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_gmm_gaussianizer_gaussian(simple_2d):
    t = GMMGaussianizer(n_components=3)
    Xt = t.fit_transform(simple_2d)
    assert abs(np.mean(Xt)) < 0.5


def test_gmm_gaussianizer_log_det(simple_2d):
    t = GMMGaussianizer(n_components=3)
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)


def test_spline_gaussianizer_shape(simple_2d):
    t = SplineGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_spline_gaussianizer_gaussian(simple_2d):
    t = SplineGaussianizer()
    Xt = t.fit_transform(simple_2d)
    assert abs(np.mean(Xt)) < 0.5


def test_spline_gaussianizer_inverse(simple_2d):
    t = SplineGaussianizer()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    assert Xr.shape == simple_2d.shape


def test_spline_gaussianizer_log_det(simple_2d):
    t = SplineGaussianizer()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    assert np.all(np.isfinite(ldj))


# ── make_cdf_monotonic ────────────────────────────────────────────────────


class TestMakeCdfMonotonic:
    def test_sorted_input_unchanged(self):
        cdf = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
        result = make_cdf_monotonic(cdf)
        np.testing.assert_array_equal(result, cdf)

    def test_fixes_violations(self):
        cdf = np.array([0.1, 0.5, 0.3, 0.7, 0.6])
        result = make_cdf_monotonic(cdf)
        assert np.all(np.diff(result) >= 0)
        np.testing.assert_array_equal(result, [0.1, 0.5, 0.5, 0.7, 0.7])

    def test_preserves_shape_1d(self):
        cdf = np.array([0.5, 0.3, 0.8])
        result = make_cdf_monotonic(cdf)
        assert result.shape == cdf.shape

    def test_preserves_shape_2d(self):
        cdf = np.array([[0.5, 0.2], [0.3, 0.8], [0.9, 0.5]])
        result = make_cdf_monotonic(cdf)
        assert result.shape == cdf.shape
        assert np.all(np.diff(result, axis=0) >= 0)

    def test_constant_input(self):
        cdf = np.array([0.5, 0.5, 0.5])
        result = make_cdf_monotonic(cdf)
        np.testing.assert_array_equal(result, cdf)

    def test_single_element(self):
        cdf = np.array([0.42])
        result = make_cdf_monotonic(cdf)
        np.testing.assert_array_equal(result, cdf)


# ── MarginalUniformize histogram CDF ─────────────────────────────────────


class TestMarginalUniformizeHistogramCDF:
    def test_shape_preserved(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        uni = MarginalUniformize(pdf_extension=10.0).fit(X)
        U = uni.transform(X)
        assert U.shape == X.shape

    def test_output_in_bounds(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))
        eps = 1e-6
        uni = MarginalUniformize(pdf_extension=10.0, eps=eps).fit(X)
        U = uni.transform(X)
        assert U.min() >= eps
        assert U.max() <= 1 - eps

    def test_roundtrip_recovery(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 2))
        uni = MarginalUniformize(pdf_extension=10.0, pdf_resolution=2000).fit(X)
        U = uni.transform(X)
        Xr = uni.inverse_transform(U)
        np.testing.assert_allclose(Xr, X, atol=0.3)

    def test_default_behavior_unchanged(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))
        uni_default = MarginalUniformize().fit(X)
        uni_zero = MarginalUniformize(pdf_extension=0.0).fit(X)
        U1 = uni_default.transform(X)
        U2 = uni_zero.transform(X)
        np.testing.assert_array_equal(U1, U2)

    def test_stores_cdf_attributes(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 2))
        uni = MarginalUniformize(pdf_extension=10.0).fit(X)
        assert hasattr(uni, "cdf_support_")
        assert hasattr(uni, "cdf_values_")
        assert hasattr(uni, "pdf_support_")
        assert hasattr(uni, "pdf_values_")
        assert len(uni.cdf_support_) == 2
        assert len(uni.cdf_values_) == 2


# ── Coverage gap tests ────────────────────────────────────────────────────


def test_marginal_uniformize_constant_feature():
    """MarginalUniformize with one constant column doesn't crash."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    X[:, 1] = 5.0  # constant column
    uni = MarginalUniformize(pdf_extension=10.0)
    uni.fit(X)
    U = uni.transform(X)
    assert U.shape == X.shape
    assert np.all(np.isfinite(U))


def test_marginal_uniformize_inverse_no_extension():
    """MarginalUniformize(pdf_extension=0.0) roundtrip via inverse_transform."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))
    uni = MarginalUniformize(pdf_extension=0.0).fit(X)
    U = uni.transform(X)
    Xr = uni.inverse_transform(U)
    assert Xr.shape == X.shape
    assert np.all(np.isfinite(Xr))
    np.testing.assert_allclose(Xr, X, atol=0.5)


def test_kde_gaussianizer_inverse_transform():
    """KDEGaussianizer inverse_transform returns finite values with correct shape."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    t = KDEGaussianizer()
    t.fit(X)
    Xt = t.transform(X)
    Xr = t.inverse_transform(Xt)
    assert Xr.shape == X.shape
    assert np.all(np.isfinite(Xr))


def test_gmm_gaussianizer_inverse_transform():
    """GMMGaussianizer inverse_transform returns finite values with correct shape."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    t = GMMGaussianizer(n_components=3)
    t.fit(X)
    Xt = t.transform(X)
    Xr = t.inverse_transform(Xt)
    assert Xr.shape == X.shape
    assert np.all(np.isfinite(Xr))


def test_spline_gaussianizer_inverse_roundtrip():
    """SplineGaussianizer inverse_transform approximately recovers input."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 2))
    t = SplineGaussianizer()
    t.fit(X)
    Xt = t.transform(X)
    Xr = t.inverse_transform(Xt)
    assert Xr.shape == X.shape
    assert np.all(np.isfinite(Xr))
    np.testing.assert_allclose(Xr, X, atol=1e-3)
