"""Tests for new density bijector classes."""

import numpy as np

from rbig import Cube, Exp, Tanh


def test_tanh_shape(simple_2d):
    t = Tanh()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_tanh_range(simple_2d):
    t = Tanh()
    Xt = t.fit_transform(simple_2d)
    assert np.all(Xt > -1)
    assert np.all(Xt < 1)


def test_tanh_inverse(simple_2d):
    t = Tanh()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-6)


def test_tanh_log_det(simple_2d):
    t = Tanh()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    assert np.all(ldj <= 0)  # tanh is contractive


def test_exp_shape(simple_2d):
    t = Exp()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_exp_positive(simple_2d):
    t = Exp()
    Xt = t.fit_transform(simple_2d)
    assert np.all(Xt > 0)


def test_exp_inverse(simple_2d):
    t = Exp()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-10)


def test_exp_log_det(simple_2d):
    t = Exp()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
    np.testing.assert_allclose(ldj, np.sum(simple_2d, axis=1))


def test_cube_shape(simple_2d):
    t = Cube()
    Xt = t.fit_transform(simple_2d)
    assert Xt.shape == simple_2d.shape


def test_cube_inverse(simple_2d):
    t = Cube()
    t.fit(simple_2d)
    Xt = t.transform(simple_2d)
    Xr = t.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_2d, atol=1e-10)


def test_cube_log_det(simple_2d):
    t = Cube()
    t.fit(simple_2d)
    ldj = t.get_log_det_jacobian(simple_2d)
    assert ldj.shape == (simple_2d.shape[0],)
