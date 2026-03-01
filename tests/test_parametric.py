"""Tests for parametric transforms."""

import numpy as np

from rbig import BoxCoxTransform, LogitTransform, QuantileTransform


def test_logit_transform_shape(uniform_2d):
    t = LogitTransform()
    Xt = t.fit_transform(uniform_2d)
    assert Xt.shape == uniform_2d.shape


def test_logit_transform_inverse(uniform_2d):
    t = LogitTransform()
    t.fit(uniform_2d)
    Xt = t.transform(uniform_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, uniform_2d, atol=1e-10)


def test_box_cox_transform_shape():
    rng = np.random.default_rng(42)
    X = rng.exponential(size=(100, 3)) + 0.1  # ensure positive
    t = BoxCoxTransform()
    Xt = t.fit_transform(X)
    assert Xt.shape == X.shape


def test_quantile_transform_shape(simple_2d):
    t = QuantileTransform()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_quantile_transform_inverse(simple_2d):
    t = QuantileTransform()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-6)
