"""Tests for xarray spatiotemporal utilities."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig._src.xarray_st import xr_st_to_matrix


def test_xr_st_to_matrix_dataarray():
    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (5, 4, 4)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(5),
            "lat": np.arange(4),
            "lon": np.arange(4),
        },
    )
    matrix, meta = xr_st_to_matrix(da, time_dim="time", spatial_dims=("lat", "lon"))
    assert matrix.ndim == 2
    assert meta["type"] == "DataArray"


def test_xr_st_to_matrix_dataset():
    rng = np.random.default_rng(42)
    ds = xr.Dataset(
        {
            "u": xr.DataArray(
                rng.uniform(0, 1, (5, 4)),
                dims=["time", "lat"],
                coords={"time": np.arange(5), "lat": np.arange(4)},
            ),
            "v": xr.DataArray(
                rng.uniform(0, 1, (5, 4)),
                dims=["time", "lat"],
                coords={"time": np.arange(5), "lat": np.arange(4)},
            ),
        }
    )
    matrix, meta = xr_st_to_matrix(
        ds, feature_vars=["u", "v"], time_dim="time", spatial_dims=("lat",)
    )
    assert matrix.shape[1] == 2
    assert meta["type"] == "Dataset"
