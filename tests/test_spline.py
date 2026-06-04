"""Tests for the shared 1D rational-quadratic spline primitive."""

from __future__ import annotations

import numpy as np
import pytest

from rbig._src.spline import RQSpline


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_forward_inverse_roundtrip(rng):
    x = rng.standard_normal(2000)
    spline = RQSpline(n_knots=300).fit(x)
    z, _ = spline.forward(x)
    x_rec, _ = spline.inverse(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-5)


def test_roundtrip_skewed(rng):
    # Skewed, strictly positive data.
    x = rng.gamma(shape=2.0, scale=1.0, size=3000)
    spline = RQSpline(n_knots=500).fit(x)
    z, _ = spline.forward(x)
    x_rec, _ = spline.inverse(z)
    np.testing.assert_allclose(x_rec, x, atol=1e-4)


def test_forward_approximately_gaussian(rng):
    x = rng.exponential(scale=2.0, size=4000)
    spline = RQSpline(n_knots=500).fit(x)
    z, _ = spline.forward(x)
    # The Gaussianized output should be close to standard normal.
    assert abs(z.mean()) < 0.1
    assert abs(z.std() - 1.0) < 0.15


def test_monotonic_output(rng):
    x = rng.standard_normal(1000)
    spline = RQSpline(n_knots=200).fit(x)
    grid = np.linspace(x.min(), x.max(), 500)
    z, _ = spline.forward(grid)
    # Strictly increasing forward map.
    assert np.all(np.diff(z) > 0)


def test_log_det_matches_numerical_gradient(rng):
    x = rng.standard_normal(3000)
    spline = RQSpline(n_knots=400).fit(x)
    pts = np.linspace(-1.5, 1.5, 50)
    _, log_dz_dx = spline.forward(pts)
    h = 1e-6
    z_plus, _ = spline.forward(pts + h)
    z_minus, _ = spline.forward(pts - h)
    numerical = (z_plus - z_minus) / (2 * h)
    np.testing.assert_allclose(np.exp(log_dz_dx), numerical, rtol=1e-3, atol=1e-4)


def test_inverse_log_det_is_negative_forward(rng):
    x = rng.standard_normal(2000)
    spline = RQSpline(n_knots=300).fit(x)
    pts = np.linspace(-2.0, 2.0, 40)
    z, log_dz_dx = spline.forward(pts)
    _, log_dx_dz = spline.inverse(z)
    np.testing.assert_allclose(log_dx_dz, -log_dz_dx, atol=1e-6)


def test_tails_extrapolate_linearly(rng):
    x = rng.standard_normal(2000)
    spline = RQSpline(n_knots=200).fit(x)
    # Points well outside the fitted support use linear tails -> finite,
    # monotone, and exactly invertible.
    far = np.array([x.min() - 5.0, x.max() + 5.0])
    z, log_dz_dx = spline.forward(far)
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(log_dz_dx))
    x_rec, _ = spline.inverse(z)
    np.testing.assert_allclose(x_rec, far, atol=1e-8)


def test_constant_feature_is_identity():
    x = np.full(100, 3.0)
    spline = RQSpline().fit(x)
    z, log_dz_dx = spline.forward(x)
    np.testing.assert_allclose(z, 0.0, atol=1e-12)
    np.testing.assert_allclose(log_dz_dx, 0.0)
    x_rec, _ = spline.inverse(z)
    np.testing.assert_allclose(x_rec, x)


def test_n_knots_capped_to_samples(rng):
    x = rng.standard_normal(10)
    spline = RQSpline(n_knots=1000).fit(x)
    assert spline.x_knots_.size <= 10
