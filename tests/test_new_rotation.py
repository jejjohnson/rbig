"""Tests for new rotation classes."""

import numpy as np
import pytest

from rbig import (
    GaussianRandomProjection,
    ICARotation,
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
    """log det jacobian raises when orthogonal=False and transform is non-square."""
    r = PicardRotation(n_components=3, random_state=42, orthogonal=False)
    r.fit(simple_5d)
    with pytest.raises(ValueError, match="square"):
        r.get_log_det_jacobian(simple_5d)


def test_picard_rotation_orthogonal_log_det_zero(simple_5d):
    """orthogonal=True with square transform gives log_det = 0."""
    r = PicardRotation(n_components=None, random_state=42, orthogonal=True)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    np.testing.assert_allclose(ldj, 0.0)


def test_ica_orthogonal_non_square_raises(simple_5d):
    """ICARotation(orthogonal=True, n_components=3) raises on non-square fit."""
    r = ICARotation(n_components=3, random_state=42, orthogonal=True)
    with pytest.raises(ValueError, match="orthogonal=True requires"):
        r.fit(simple_5d)


def test_picard_orthogonal_non_square_raises(simple_5d):
    """PicardRotation(orthogonal=True, n_components=3) raises on non-square fit."""
    r = PicardRotation(n_components=3, random_state=42, orthogonal=True)
    with pytest.raises(ValueError, match="orthogonal=True requires"):
        r.fit(simple_5d)


# ---- ICARotation non-orthogonal tests ----


def test_ica_rotation_non_orthogonal_transform(simple_5d):
    """ICARotation(orthogonal=False) fit_transform returns correct shape."""
    r = ICARotation(orthogonal=False, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == simple_5d.shape


def test_ica_rotation_non_orthogonal_inverse(simple_5d):
    """ICARotation(orthogonal=False) inverse_transform roundtrip."""
    r = ICARotation(orthogonal=False, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-5)


def test_ica_rotation_non_orthogonal_log_det(simple_5d):
    """ICARotation(orthogonal=False) log_det_jacobian returns finite array."""
    r = ICARotation(orthogonal=False, random_state=42)
    r.fit(simple_5d)
    ldj = r.log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)
    assert np.all(np.isfinite(ldj))


# ---- ICARotation FastICA fallback tests ----


def test_ica_rotation_fastica_fallback(simple_2d):
    """ICARotation works via FastICA when picard is not available."""
    import sys
    from unittest.mock import patch
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with patch.dict(sys.modules, {"picard": None}):
        r = ICARotation(random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            r.fit(simple_2d)
        Xt = r.transform(simple_2d)
        assert Xt.shape == simple_2d.shape
        assert r.K_ is None  # signals FastICA path


def test_ica_rotation_fastica_transform_inverse(simple_2d):
    """FastICA fallback: transform + inverse_transform roundtrip."""
    import sys
    from unittest.mock import patch
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with patch.dict(sys.modules, {"picard": None}):
        r = ICARotation(random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            r.fit(simple_2d)
        Xt = r.transform(simple_2d)
        Xr = r.inverse_transform(Xt)
        assert Xr.shape == simple_2d.shape
        np.testing.assert_allclose(Xr, simple_2d, atol=1e-3)


def test_ica_rotation_fastica_log_det(simple_2d):
    """FastICA fallback: log_det_jacobian returns finite values."""
    import sys
    from unittest.mock import patch
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with patch.dict(sys.modules, {"picard": None}):
        r = ICARotation(random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            r.fit(simple_2d)
        ldj = r.log_det_jacobian(simple_2d)
        assert ldj.shape == (simple_2d.shape[0],)
        assert np.all(np.isfinite(ldj))


# ---- RandomOrthogonalProjection square inverse ----


def test_random_orthogonal_projection_inverse_square(simple_5d):
    """RandomOrthogonalProjection(n_components=5) inverse_transform roundtrip."""
    r = RandomOrthogonalProjection(n_components=5, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-10)


# ---- PicardRotation non-orthogonal tests ----


def test_picard_rotation_non_orthogonal_transform(simple_5d):
    """PicardRotation(orthogonal=False) transform returns correct shape."""
    r = PicardRotation(orthogonal=False, random_state=42)
    Xt = r.fit_transform(simple_5d)
    assert Xt.shape == simple_5d.shape


def test_picard_rotation_non_orthogonal_inverse(simple_5d):
    """PicardRotation(orthogonal=False) inverse_transform roundtrip."""
    r = PicardRotation(orthogonal=False, random_state=42)
    r.fit(simple_5d)
    Xt = r.transform(simple_5d)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, simple_5d, atol=1e-5)


def test_picard_rotation_non_orthogonal_log_det(simple_5d):
    """PicardRotation(orthogonal=False) log_det returns non-zero finite values."""
    r = PicardRotation(orthogonal=False, n_components=None, random_state=42)
    r.fit(simple_5d)
    ldj = r.get_log_det_jacobian(simple_5d)
    assert ldj.shape == (simple_5d.shape[0],)
    assert np.all(np.isfinite(ldj))
