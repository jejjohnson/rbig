"""Test base abstract classes."""
import numpy as np
import pytest

from rbig._src.base import Bijector, MarginalBijector, RBIGLayer, RotationTransform


def test_bijector_is_abstract():
    with pytest.raises(TypeError):
        Bijector()


def test_marginal_bijector_is_abstract():
    with pytest.raises(TypeError):
        MarginalBijector()


def test_rotation_transform_is_abstract():
    with pytest.raises(TypeError):
        RotationTransform()


def test_rbig_layer_is_abstract():
    with pytest.raises(TypeError):
        RBIGLayer()


def test_bijector_interface():
    class IdentityBijector(Bijector):
        def forward(self, x):
            return x

        def inverse(self, y):
            return y

        def log_abs_det_jacobian(self, x):
            return np.zeros(x.shape[0])

    b = IdentityBijector()
    x = np.ones((5, 2))
    y, ladj = b.forward_and_ladj(x)
    assert y.shape == x.shape
    assert ladj.shape == (5,)
