import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_2d(rng):
    return rng.standard_normal((100, 2))


@pytest.fixture
def sample_3d(rng):
    return rng.standard_normal((100, 3))


@pytest.fixture
def fitted_rbig(sample_2d):
    from rbig import RBIG
    model = RBIG(n_layers=10, zero_tolerance=5)
    model.fit(sample_2d)
    return model
