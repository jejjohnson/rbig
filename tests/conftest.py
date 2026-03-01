import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def data_2d(rng):
    return rng.standard_normal((200, 2))


@pytest.fixture
def data_nd(rng):
    return rng.standard_normal((500, 5))
