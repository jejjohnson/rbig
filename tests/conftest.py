"""Test configuration and shared fixtures."""

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_2d(rng):
    """Simple 2D Gaussian dataset."""
    n = 200
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    return rng.multivariate_normal(mean, cov, size=n)


@pytest.fixture
def simple_5d(rng):
    """Simple 5D dataset."""
    n = 300
    cov = np.eye(5)
    cov[0, 1] = cov[1, 0] = 0.3
    return rng.multivariate_normal(np.zeros(5), cov, size=n)


@pytest.fixture
def uniform_2d(rng):
    """2D uniform data."""
    return rng.uniform(0.01, 0.99, size=(200, 2))
