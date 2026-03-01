"""Tests for new marginal Gaussianizer classes."""

import numpy as np

from rbig import (
    GMMGaussianizer,
    KDEGaussianizer,
    QuantileGaussianizer,
    SplineGaussianizer,
)


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
