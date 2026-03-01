"""Tests for rotation transforms."""

import numpy as np

from rbig import ICARotation, PCARotation


def test_pca_rotation_shape(simple_5d):
    r = PCARotation()
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == simple_5d.shape


def test_pca_rotation_whitened(simple_5d):
    r = PCARotation(whiten=True)
    Xt = r.fit_transform(simple_5d)
    # Whitened data should have approximately identity covariance
    cov = np.cov(Xt.T)
    np.testing.assert_allclose(np.diag(cov), np.ones(5), atol=0.2)


def test_pca_rotation_inverse(simple_5d):
    r = PCARotation(whiten=False)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-10)


def test_pca_log_det_jacobian_shape(simple_5d):
    r = PCARotation()
    r.fit(simple_5d)
    ldj = r.log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)


def test_ica_rotation_shape(simple_5d):
    r = ICARotation(random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape[1] == simple_5d.shape[1]


def test_ica_rotation_log_det_jacobian(simple_5d):
    r = ICARotation(random_state=42)
    r.fit(simple_5d)
    ldj = r.log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)
