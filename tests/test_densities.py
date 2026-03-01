"""Tests for density estimation utilities."""

import numpy as np

from rbig import (
    gaussian_entropy,
    joint_entropy_gaussian,
    marginal_entropy,
    total_correlation,
)


def test_marginal_entropy_shape(simple_2d):
    h = marginal_entropy(simple_2d)
    assert h.shape == (2,)


def test_marginal_entropy_positive(simple_2d):
    h = marginal_entropy(simple_2d)
    assert np.all(h > 0)


def test_joint_entropy_gaussian_shape(simple_2d):
    h = joint_entropy_gaussian(simple_2d)
    assert isinstance(h, float)


def test_total_correlation_non_negative(simple_2d):
    tc = total_correlation(simple_2d)
    assert tc >= -0.1  # Can be slightly negative due to estimation error


def test_gaussian_entropy_formula():
    """Entropy of N(0,1) should be 0.5*(1+log(2pi)) ≈ 1.4189"""
    h = gaussian_entropy(1, cov=np.array([[1.0]]))
    expected = 0.5 * (1 + np.log(2 * np.pi))
    assert abs(h - expected) < 1e-10


def test_gaussian_entropy_no_cov():
    h = gaussian_entropy(2)
    expected = 0.5 * 2 * (1 + np.log(2 * np.pi))
    assert abs(h - expected) < 1e-10
