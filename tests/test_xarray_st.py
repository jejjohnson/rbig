"""Tests for xarray spatiotemporal utilities."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig import xr_st_to_matrix


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


def test_xr_st_to_matrix_dataset_default_vars():
    """Dataset with feature_vars=None auto-detects all data_vars."""
    rng = np.random.default_rng(42)
    ds = xr.Dataset(
        {
            "temperature": xr.DataArray(
                rng.uniform(0, 1, (5, 4)),
                dims=["time", "lat"],
                coords={"time": np.arange(5), "lat": np.arange(4)},
            ),
            "pressure": xr.DataArray(
                rng.uniform(0, 1, (5, 4)),
                dims=["time", "lat"],
                coords={"time": np.arange(5), "lat": np.arange(4)},
            ),
            "humidity": xr.DataArray(
                rng.uniform(0, 1, (5, 4)),
                dims=["time", "lat"],
                coords={"time": np.arange(5), "lat": np.arange(4)},
            ),
        }
    )
    matrix, meta = xr_st_to_matrix(
        ds, feature_vars=None, time_dim="time", spatial_dims=("lat",)
    )
    assert matrix.shape[1] == 3
    assert meta["type"] == "Dataset"
    assert set(meta["feature_vars"]) == {"temperature", "pressure", "humidity"}


def test_matrix_to_xr_st_dataarray():
    """Roundtrip DataArray through xr_st_to_matrix and matrix_to_xr_st."""
    from rbig import matrix_to_xr_st

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
    da_back = matrix_to_xr_st(matrix, meta)
    assert da_back.shape == da.shape
    assert tuple(da_back.dims) == tuple(da.dims)
    np.testing.assert_allclose(da_back.values, da.values)


def test_matrix_to_xr_st_dataset():
    """Roundtrip Dataset through xr_st_to_matrix and matrix_to_xr_st."""
    from rbig import matrix_to_xr_st

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
    ds_back = matrix_to_xr_st(matrix, meta)
    assert isinstance(ds_back, xr.Dataset)
    assert "u" in ds_back.data_vars
    assert "v" in ds_back.data_vars
    assert ds_back["u"].shape == (5, 4)
    np.testing.assert_allclose(ds_back["u"].values, ds["u"].values)
    np.testing.assert_allclose(ds_back["v"].values, ds["v"].values)


def test_xr_rbig_fit_transform_dataarray():
    """Full pipeline with xr_rbig_fit_transform on a DataArray."""
    from rbig import AnnealedRBIG, xr_rbig_fit_transform

    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (20, 4, 5)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(20),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
    )
    model = AnnealedRBIG(n_layers=3, random_state=0)
    matrix, transformed = xr_rbig_fit_transform(da, model)
    assert matrix.ndim == 2
    assert matrix.shape == (20 * 4 * 5, 1)
    assert transformed.shape == da.shape


def test_xarray_rbig_mutual_information():
    """XarrayRBIG.mutual_information with two DataArrays returns a float."""
    from rbig import XarrayRBIG

    rng = np.random.default_rng(42)
    x = xr.DataArray(
        rng.uniform(0, 1, (50, 1)),
        dims=["time", "feature"],
        coords={"time": np.arange(50), "feature": ["a"]},
    )
    y = xr.DataArray(
        rng.uniform(0, 1, (50, 1)),
        dims=["time", "feature"],
        coords={"time": np.arange(50), "feature": ["b"]},
    )
    xrbig = XarrayRBIG(n_layers=3, random_state=0)
    mi = xrbig.mutual_information(x, y)
    assert isinstance(mi, float)


def test_xarray_rbig_transform_preserves_coords():
    """XarrayRBIG.transform re-attaches coords and name after transform."""
    from rbig import XarrayRBIG

    rng = np.random.default_rng(42)
    da = xr.DataArray(
        rng.uniform(0, 1, (20, 4, 5)),
        dims=["time", "lat", "lon"],
        coords={
            "time": np.arange(20),
            "lat": np.arange(4),
            "lon": np.arange(5),
        },
        name="temperature",
    )
    xrbig = XarrayRBIG(n_layers=3, random_state=0)
    xrbig.fit(da)
    result = xrbig.transform(da)
    assert result.shape == da.shape
    assert result.name == "temperature"
    np.testing.assert_array_equal(result.coords["time"].values, np.arange(20))
    np.testing.assert_array_equal(result.coords["lat"].values, np.arange(4))
    np.testing.assert_array_equal(result.coords["lon"].values, np.arange(5))
