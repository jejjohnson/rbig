"""Tests for new rotation classes."""

import numpy as np

from rbig import (
    GaussianRandomProjection,
    OrthogonalDimensionalityReduction,
    PicardRotation,
    RandomOrthogonalProjection,
    RandomRotation,
)


def test_random_rotation_shape(simple_5d):
    r = RandomRotation(random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == simple_5d.shape


def test_random_rotation_orthogonal(simple_5d):
    r = RandomRotation(random_state=42)
    r.fit(simple_5d)
    Q = r.rotation_matrix_
    # Q^T Q should be identity for orthogonal matrix
    np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-10)


def test_random_rotation_inverse(simple_5d):
    r = RandomRotation(random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-10)


def test_random_rotation_log_det(simple_5d):
    r = RandomRotation(random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)


def test_random_orthogonal_projection_shape(simple_5d):
    r = RandomOrthogonalProjection(n_components=3, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == (simple_5d.shape[0], 3)


def test_random_orthogonal_projection_log_det(simple_5d):
    r = RandomOrthogonalProjection(n_components=3, random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)


def test_gaussian_random_projection_shape(simple_5d):
    r = GaussianRandomProjection(n_components=3, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == (simple_5d.shape[0], 3)


def test_gaussian_random_projection_log_det(simple_5d):
    r = GaussianRandomProjection(n_components=3, random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)


def test_orthogonal_dim_reduction_shape(simple_5d):
    r = OrthogonalDimensionalityReduction(n_components=3, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == (simple_5d.shape[0], 3)


def test_orthogonal_dim_reduction_log_det(simple_5d):
    r = OrthogonalDimensionalityReduction(n_components=3, random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)


def test_picard_rotation_shape(simple_5d):
    r = PicardRotation(random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape[1] == simple_5d.shape[1]


def test_picard_rotation_log_det(simple_5d):
    r = PicardRotation(random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)
