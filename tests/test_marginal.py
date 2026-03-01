"""Tests for marginal transforms."""

import numpy as np

from rbig import MarginalGaussianize, MarginalKDEGaussianize, MarginalUniformize


def test_marginal_uniformize_shape(simple_2d):
    t = MarginalUniformize()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_marginal_uniformize_range(simple_2d):
    t = MarginalUniformize()
    Xt = t.fit_transform(simple_2d)
    assert np.all(Xt > 0)
    assert np.all(Xt < 1)


def test_marginal_gaussianize_shape(simple_2d):
    t = MarginalGaussianize()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_marginal_gaussianize_mean(simple_2d):
    t = MarginalGaussianize()
    Xt = t.fit_transform(simple_2d)
    # Transformed data should be approximately Gaussian (zero mean)
    assert abs(np.mean(Xt)) < 0.5


def test_marginal_gaussianize_inverse(simple_2d):
    t = MarginalGaussianize()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    # Inverse should recover original data approximately
    assert Xr.shape == simple_2d.shape


def test_marginal_kde_gaussianize_shape(simple_2d):
    t = MarginalKDEGaussianize()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_marginal_kde_gaussianize_normal(simple_2d):
    t = MarginalKDEGaussianize()
    Xt = t.fit_transform(simple_2d)
    # Transformed data should be approximately Gaussian
    assert abs(np.mean(Xt)) < 0.5
    assert abs(np.std(Xt) - 1.0) < 0.5
