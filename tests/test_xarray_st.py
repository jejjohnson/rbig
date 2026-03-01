"""Test xarray spatiotemporal support."""
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig._src.xarray_st import rbig_dataarray, rbig_dataset  # noqa: E402


@pytest.fixture
def simple_dataarray(rng):
    data = rng.standard_normal((50, 3))
    return xr.DataArray(data, dims=["time", "space"])


@pytest.fixture
def simple_dataset(rng):
    data_a = rng.standard_normal(50)
    data_b = rng.standard_normal(50)
    return xr.Dataset({"a": ("time", data_a), "b": ("time", data_b)})


def test_rbig_dataarray(simple_dataarray):
    model, da_gauss = rbig_dataarray(simple_dataarray, dim="time", n_layers=20, zero_tolerance=5)
    assert da_gauss.shape == simple_dataarray.shape
    assert np.all(np.isfinite(da_gauss.values))


def test_rbig_dataset(simple_dataset):
    model, ds_gauss = rbig_dataset(
        simple_dataset, features=["a", "b"], n_layers=20, zero_tolerance=5
    )
    assert "a" in ds_gauss
    assert "b" in ds_gauss
