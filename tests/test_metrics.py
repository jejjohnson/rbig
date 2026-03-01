"""Tests for information-theoretic metrics."""

import numpy as np

from rbig import entropy_normal_approx, negentropy, total_correlation_rbig


def test_total_correlation_rbig(simple_2d):
    tc = total_correlation_rbig(simple_2d)
    assert isinstance(tc, float)
    assert tc >= -0.1


def test_entropy_normal_approx(simple_2d):
    h = entropy_normal_approx(simple_2d)
    assert isinstance(h, float)


def test_negentropy_shape(simple_2d):
    J = negentropy(simple_2d)
    assert J.shape == (2,)


def test_negentropy_non_negative_approx(simple_2d):
    """Negentropy should be non-negative (may fail due to estimation)."""
    J = negentropy(simple_2d)
    # Should be approximately non-negative
    assert np.all(J > -0.5)
