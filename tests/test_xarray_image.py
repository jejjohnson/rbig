"""Tests for xarray image utilities."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig._src.xarray_image import matrix_to_xr_image, xr_image_to_matrix


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
