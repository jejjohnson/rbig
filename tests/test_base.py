"""Tests for base classes."""

import numpy as np
import pytest

from rbig import BaseTransform


class ConcreteTransform(BaseTransform):
    def fit(self, X):
        return self

    def transform(self, X):
        return X + 1.0

    def inverse_transform(self, X):
        return X - 1.0


def test_base_transform_fit_transform(simple_2d):
    t = ConcreteTransform()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape
    np.testing.assert_allclose(Xt, simple_2d + 1.0)


def test_base_transform_inverse(simple_2d):
    t = ConcreteTransform()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d)


def test_log_det_jacobian_not_implemented(simple_2d):
    t = ConcreteTransform()
    t.fit(simple_2d)
    with pytest.raises(NotImplementedError):
        t.log_det_jacobian(simple_2d)
