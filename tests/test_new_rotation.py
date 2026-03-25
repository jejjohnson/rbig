"""Tests for new rotation classes."""

import numpy as np
import pytest

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


def test_random_orthogonal_projection_shape_full(simple_5d):
    """Full (square) projection is bijective and works."""
    r = RandomOrthogonalProjection(n_components=5, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == simple_5d.shape


def test_random_orthogonal_projection_shape_reducing(simple_5d):
    """Dimensionality-reducing projection: transform works but inverse/ldj raise."""
    r = RandomOrthogonalProjection(n_components=3, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    assert Xt.shape == (simple_5d.shape[0], 3)
    with pytest.raises(NotImplementedError):
        r.inverse_transform(Xt)
    with pytest.raises(NotImplementedError):
        r.get_log_det_jacobian(simple_5d)


def test_random_orthogonal_projection_log_det_full(simple_5d):
    """Square projection has log det = 0."""
    r = RandomOrthogonalProjection(n_components=5, random_state=42)
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


def test_gaussian_random_projection_inverse(simple_5d):
    """Inverse transform via pseudoinverse should return correct shape."""
    r = GaussianRandomProjection(n_components=3, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    assert Xr.shape == simple_5d.shape


def test_orthogonal_dim_reduction_shape(simple_5d):
    """Dimensionality-reducing: transform works but inverse/ldj raise."""
    r = OrthogonalDimensionalityReduction(n_components=3, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    assert Xt.shape == (simple_5d.shape[0], 3)
    with pytest.raises(NotImplementedError):
        r.inverse_transform(Xt)
    with pytest.raises(NotImplementedError):
        r.get_log_det_jacobian(simple_5d)


def test_orthogonal_dim_reduction_full(simple_5d):
    """Full (square) rotation: log det = 0, inverse works."""
    r = OrthogonalDimensionalityReduction(n_components=5, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-10)


def test_picard_rotation_shape(simple_5d):
    r = PicardRotation(random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape[1] == simple_5d.shape[1]


def test_picard_rotation_log_det(simple_5d):
    r = PicardRotation(random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)


def test_picard_rotation_log_det_non_square(simple_5d):
    """log det jacobian raises when transform is non-square."""
    r = PicardRotation(n_components=3, random_state=42)
    r.fit(simple_5d)
    with pytest.raises(ValueError, match="square"):
        r.get_log_det_jacobian(simple_5d)
