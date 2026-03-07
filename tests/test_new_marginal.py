"""Tests for migrated marginal utilities: make_cdf_monotonic, MarginalUniformize histogram CDF."""

import numpy as np

from rbig._src.marginal import MarginalUniformize, make_cdf_monotonic


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
