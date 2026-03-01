"""Test rotation transforms."""
import numpy as np
import pytest

from rbig import PCARotation, RandomRotation, fit_rotation, apply_rotation, apply_rotation_inverse


def test_pca_rotation_shape(data_2d):
    rot = PCARotation().fit(data_2d)
    Y = rot.forward(data_2d)
    assert Y.shape == data_2d.shape


def test_pca_rotation_inverse(data_2d):
    rot = PCARotation().fit(data_2d)
    Y = rot.forward(data_2d)
    X_rec = rot.inverse(Y)
    np.testing.assert_allclose(X_rec, data_2d, atol=1e-10)


def test_random_rotation_shape(data_2d):
    rot = RandomRotation(random_state=42).fit(data_2d)
    Y = rot.forward(data_2d)
    assert Y.shape == data_2d.shape


def test_random_rotation_inverse(data_2d):
    rot = RandomRotation(random_state=42).fit(data_2d)
    Y = rot.forward(data_2d)
    X_rec = rot.inverse(Y)
    np.testing.assert_allclose(X_rec, data_2d, atol=1e-10)


def test_fit_rotation_pca(data_2d):
    R, meta = fit_rotation(data_2d, rotation_type="PCA")
    assert R.shape == (data_2d.shape[1], data_2d.shape[1])
    assert meta["type"] == "PCA"


def test_fit_rotation_random(data_2d):
    R, meta = fit_rotation(data_2d, rotation_type="random", random_state=0)
    assert R.shape == (data_2d.shape[1], data_2d.shape[1])


def test_apply_rotation_inverse(data_2d):
    R, _ = fit_rotation(data_2d, rotation_type="PCA")
    Y = apply_rotation(data_2d, R)
    X_rec = apply_rotation_inverse(Y, R)
    np.testing.assert_allclose(X_rec, data_2d, atol=1e-10)
