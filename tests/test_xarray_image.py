"""Tests for xarray image utilities."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig import matrix_to_xr_image, xr_image_to_matrix


def test_xr_image_to_matrix():
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (10, 10)),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    matrix, _coords = xr_image_to_matrix(da)
    assert matrix.ndim == 2
    assert matrix.shape[0] == 100


def test_matrix_to_xr_image():
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0, 1, (100, 1))
    coords = {"x": np.arange(10), "y": np.arange(10)}
    da = matrix_to_xr_image(matrix, coords, (10, 10))
    assert da.shape == (10, 10)


def test_xr_image_to_matrix_multifeature():
    """3D image (10,10,3) flattens to matrix shape (100,3)."""
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (10, 10, 3)),
        dims=["x", "y", "channel"],
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
            "channel": ["r", "g", "b"],
        },
    )
    matrix, coords = xr_image_to_matrix(da)
    assert matrix.ndim == 2
    assert matrix.shape == (100, 3)
    assert "channel" in coords


def test_matrix_to_xr_image_multifeature():
    """Roundtrip for multi-feature image preserves shape and dims."""
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (10, 10, 3)),
        dims=["x", "y", "channel"],
        coords={
            "x": np.arange(10),
            "y": np.arange(10),
            "channel": ["r", "g", "b"],
        },
    )
    matrix, coords = xr_image_to_matrix(da)
    da_back = matrix_to_xr_image(matrix, coords, (10, 10, 3))
    assert da_back.shape == (10, 10, 3)
    assert tuple(da_back.dims) == ("x", "y", "channel")
    np.testing.assert_allclose(da_back.values, da.values)


def test_xr_image_roundtrip_grayscale():
    """to_matrix then back preserves values for grayscale."""
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (10, 10)),
        dims=["x", "y"],
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    matrix, coords = xr_image_to_matrix(da)
    da_back = matrix_to_xr_image(matrix, coords, (10, 10))
    assert da_back.shape == da.shape
    np.testing.assert_allclose(da_back.values, da.values)


def test_xr_image_roundtrip_multifeature():
    """to_matrix then back preserves values for 3D multi-feature image."""
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (8, 8, 3)),
        dims=["x", "y", "channel"],
        coords={
            "x": np.arange(8),
            "y": np.arange(8),
            "channel": ["r", "g", "b"],
        },
    )
    matrix, coords = xr_image_to_matrix(da)
    da_back = matrix_to_xr_image(matrix, coords, (8, 8, 3))
    assert da_back.shape == da.shape
    assert tuple(da_back.dims) == tuple(da.dims)
    np.testing.assert_allclose(da_back.values, da.values)


def test_xr_apply_rbig_2d():
    """xr_apply_rbig on 2D input passes data directly to model.transform."""
    from rbig import AnnealedRBIG, xr_apply_rbig

    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, (50, 3))
    da = xr.DataArray(
        data,
        dims=["sample", "feature"],
        coords={"sample": np.arange(50), "feature": ["a", "b", "c"]},
        name="test2d",
    )
    model = AnnealedRBIG(n_layers=3, random_state=0)
    model.fit(data)
    result = xr_apply_rbig(da, model)
    assert result.shape == da.shape
    assert tuple(result.dims) == tuple(da.dims)
    assert result.name == "test2d"


def test_xr_apply_rbig_3d():
    """xr_apply_rbig on 3D input reshapes, transforms, and restores shape/dims/name."""
    from rbig import AnnealedRBIG, xr_apply_rbig

    rng = np.random.default_rng(42)
    data_3d = rng.uniform(0, 1, (8, 8, 3))
    da = xr.DataArray(
        data_3d,
        dims=["x", "y", "channel"],
        coords={
            "x": np.arange(8),
            "y": np.arange(8),
            "channel": ["r", "g", "b"],
        },
        name="image3d",
    )
    # Fit model on the reshaped 2D data
    data_2d = data_3d.reshape(-1, 3)
    model = AnnealedRBIG(n_layers=3, random_state=0)
    model.fit(data_2d)
    result = xr_apply_rbig(da, model)
    assert result.shape == (8, 8, 3)
    assert tuple(result.dims) == ("x", "y", "channel")
    assert result.name == "image3d"
    # Verify coords preserved
    np.testing.assert_array_equal(result.coords["x"].values, np.arange(8))
    np.testing.assert_array_equal(result.coords["y"].values, np.arange(8))
