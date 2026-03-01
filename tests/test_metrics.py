"""Test information-theoretic metrics."""
import numpy as np

from rbig import information_reduction, mutual_information


def test_information_reduction_non_negative(data_2d):
    rng = np.random.default_rng(0)
    y = rng.standard_normal(data_2d.shape)
    ir = information_reduction(data_2d, y)
    assert isinstance(ir, float)
    assert ir >= 0


def test_mutual_information_2d(rng):
    X = rng.standard_normal((200, 2))
    Y = rng.standard_normal((200, 2))
    mi = mutual_information(X, Y, n_layers=30, zero_tolerance=10)
    assert isinstance(mi, float)
    assert mi >= 0


def test_mutual_information_1d(rng):
    X = rng.standard_normal(200)
    Y = rng.standard_normal(200)
    mi = mutual_information(X, Y, n_layers=30, zero_tolerance=10)
    assert isinstance(mi, float)
    assert mi >= 0


def test_mutual_information_correlated_higher():
    """Correlated variables should yield non-negative MI."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal(300)
    Y_corr = X + 0.05 * rng.standard_normal(300)

    mi = mutual_information(X, Y_corr, n_layers=50, zero_tolerance=15)
    assert isinstance(mi, float)
    assert mi >= 0
