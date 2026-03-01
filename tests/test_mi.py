import numpy as np
import pytest
from rbig import RBIGMI


def test_rbigmi_fit(rng):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2))
    model = RBIGMI(n_layers=10, zero_tolerance=10)
    model.fit(X, Y)
    assert hasattr(model, 'rbig_model_XY')


def test_rbigmi_mutual_information(rng):
    X = rng.standard_normal((100, 2))
    Y = rng.standard_normal((100, 2))
    model = RBIGMI(n_layers=10, zero_tolerance=10)
    model.fit(X, Y)
    mi = model.mutual_information()
    assert isinstance(mi, float)
    assert mi >= 0
