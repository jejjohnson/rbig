import numpy as np
import pytest
from rbig._src.metrics import information_reduction, neg_entropy_normal
from rbig._src.marginal import entropy_marginal


def test_information_reduction(rng):
    X = rng.standard_normal((200, 3))
    Y = rng.standard_normal((200, 3))
    I = information_reduction(X, Y)
    assert isinstance(I, float)


def test_neg_entropy_normal(rng):
    data = rng.standard_normal((200, 2))
    neg = neg_entropy_normal(data)
    assert neg.shape == (2,)
    assert np.abs(neg).max() < 2.0


def test_entropy_marginal(rng):
    data = rng.standard_normal((200, 3))
    H = entropy_marginal(data)
    assert H.shape == (3,)
