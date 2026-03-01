import numpy as np
import pytest
from rbig import RBIGKLD


def test_rbigkld_fit(rng):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2))
    model = RBIGKLD(n_layers=10, zero_tolerance=10)
    model.fit(X, Y)
    assert hasattr(model, 'kld')


def test_rbigkld_get_kld(rng):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2))
    model = RBIGKLD(n_layers=10, zero_tolerance=10)
    model.fit(X, Y)
    kld = model.get_kld()
    assert isinstance(kld, float)
