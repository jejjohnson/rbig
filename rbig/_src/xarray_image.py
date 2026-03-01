"""xarray image support for RBIG."""
from typing import Tuple

import numpy as np

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False


def _require_xarray():
    if not _XARRAY_AVAILABLE:
        raise ImportError("xarray is required. Install with: pip install xarray")


def xarray_image_gaussianize(
    da,
    spatial_dims: Tuple[str, str] = ("x", "y"),
    patch_size: Tuple[int, int] = (8, 8),
    **rbig_kwargs,
):
    """Gaussianize an xr.DataArray image using patch-based RBIG.

    Parameters
    ----------
    da : xr.DataArray
        Image DataArray. Must have the two spatial dimensions.
    spatial_dims : tuple of str
        Names of the two spatial dimensions.
    patch_size : tuple of int
    **rbig_kwargs

    Returns
    -------
    da_gauss : xr.DataArray
        Gaussianized image DataArray (same shape).
    """
    _require_xarray()
    from rbig._src.image import image_gaussianize

    image = da.values
    gauss_image = image_gaussianize(image, patch_size=patch_size, **rbig_kwargs)
    return xr.DataArray(gauss_image, dims=da.dims, coords=da.coords, attrs=da.attrs)


def xarray_image_rbig(
    da,
    spatial_dims: Tuple[str, str] = ("x", "y"),
    patch_size: Tuple[int, int] = (8, 8),
    **rbig_kwargs,
):
    """Fit an ImageRBIG model to an xr.DataArray image.

    Parameters
    ----------
    da : xr.DataArray
    spatial_dims : tuple of str
    patch_size : tuple of int
    **rbig_kwargs

    Returns
    -------
    model : ImageRBIG
        Fitted model.
    da_gauss : xr.DataArray
        Gaussianized image DataArray.
    """
    _require_xarray()
    from rbig._src.image import ImageRBIG

    image = da.values
    model = ImageRBIG(patch_size=patch_size, **rbig_kwargs).fit(image)
    gauss_image = model.transform(image)
    da_gauss = xr.DataArray(gauss_image, dims=da.dims, coords=da.coords, attrs=da.attrs)
    return model, da_gauss
