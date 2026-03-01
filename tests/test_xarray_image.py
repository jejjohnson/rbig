"""Test xarray image support."""
import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from rbig._src.xarray_image import xarray_image_gaussianize, xarray_image_rbig


@pytest.fixture
def image_dataarray(rng):
    data = rng.standard_normal((16, 16))
    return xr.DataArray(data, dims=["x", "y"])


def test_xarray_image_gaussianize(image_dataarray):
    da_gauss = xarray_image_gaussianize(
        image_dataarray,
        patch_size=(4, 4),
        n_layers=20,
        zero_tolerance=10,
    )
    assert da_gauss.shape == image_dataarray.shape
    assert np.all(np.isfinite(da_gauss.values))


def test_xarray_image_rbig(image_dataarray):
    model, da_gauss = xarray_image_rbig(
        image_dataarray,
        patch_size=(4, 4),
        n_layers=20,
        zero_tolerance=10,
    )
    assert da_gauss.shape == image_dataarray.shape
    assert np.all(np.isfinite(da_gauss.values))
