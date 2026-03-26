"""Tests for new Bijector base classes."""

import numpy as np
import pytest

from rbig import Bijector, CompositeBijector, MarginalBijector, RotationBijector


class ConcreteBijector(Bijector):
    def fit(self, X):
        return self

    def transform(self, X):
        return X + 1.0

    def inverse_transform(self, X):
        return X - 1.0

    def get_log_det_jacobian(self, X):
        return np.zeros(X.shape[0])


class ConcreteRotation(RotationBijector):
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


class ConcreteMarginal(MarginalBijector):
    def fit(self, X):
        return self

    def transform(self, X):
        return X * 2

    def inverse_transform(self, X):
        return X / 2

    def get_log_det_jacobian(self, X):
        return np.full(X.shape[0], np.log(2.0) * X.shape[1])


def test_bijector_fit_transform(simple_2d):
    b = ConcreteBijector()
    Xt = b.fit_transform(simple_2d)
    np.testing.assert_allclose(Xt, simple_2d + 1.0)


def test_bijector_log_det_jacobian(simple_2d):
    b = ConcreteBijector()
    b.fit(simple_2d)
    ldj = b.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)


def test_rotation_bijector_log_det_zero(simple_2d):
    r = ConcreteRotation()
    r.fit(simple_2d)
    ldj = r.get_log_det_jacobian(simple_2d)
    np.testing.assert_allclose(ldj, 0.0)


def test_marginal_bijector(simple_2d):
    m = ConcreteMarginal()
    Xt = m.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_composite_bijector_fit_transform(simple_2d):
    b1 = ConcreteBijector()
    b2 = ConcreteBijector()
    comp = CompositeBijector([b1, b2])
    Xt = comp.fit_transform(simple_2d)
    np.testing.assert_allclose(Xt, simple_2d + 2.0)


def test_composite_bijector_inverse(simple_2d):
    b1 = ConcreteBijector()
    b2 = ConcreteBijector()
    comp = CompositeBijector([b1, b2])
    comp.fit(simple_2d)
    Xt = comp.transform(simple_2d)
    Xr = comp.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d)


def test_bijector_abstract():
    with pytest.raises(TypeError):
        Bijector()


# ── Coverage gap tests ────────────────────────────────────────────────────


def test_composed_bijector_log_det_jacobian(simple_2d):
    """CompositeBijector.get_log_det_jacobian accumulates log-dets correctly."""
    from rbig import QuantileGaussianizer

    b1 = QuantileGaussianizer()
    b2 = QuantileGaussianizer()
    comp = CompositeBijector([b1, b2])
    comp.fit(simple_2d)
    ldj = comp.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    assert np.all(np.isfinite(ldj))
