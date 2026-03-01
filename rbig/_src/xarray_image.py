"""Xarray interface for image transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from rbig._src.model import AnnealedRBIG


def xr_image_to_matrix(
    da: xr.DataArray,
    spatial_dims: tuple[str, str] = ("x", "y"),
) -> tuple[np.ndarray, dict]:
    """Convert an xarray DataArray image to a 2D matrix.

    Parameters
    ----------
    da : xarray DataArray with spatial dimensions
    spatial_dims : names of spatial dimensions

    Returns
    -------
    matrix : array of shape (n_pixels, n_features)
    coords : dict with original coordinates for reconstruction
    """
    coords = {dim: da[dim].values for dim in da.dims}
    matrix = (
        da.values.reshape(-1, 1)
        if da.ndim == 2
        else da.values.reshape(-1, da.shape[-1])
    )
    return matrix, coords


def matrix_to_xr_image(
    matrix: np.ndarray,
    coords: dict,
    spatial_shape: tuple[int, ...],
    name: str = "data",
) -> xr.DataArray:
    """Reconstruct xarray DataArray from 2D matrix.

    Parameters
    ----------
    matrix : array of shape (n_pixels, n_features)
    coords : dict of coordinates from xr_image_to_matrix
    spatial_shape : shape of the spatial dimensions
    name : name for the DataArray

    Returns
    -------
    da : xarray DataArray
    """
    import xarray as xr

    arr = matrix.reshape(spatial_shape)
    dims = list(coords.keys())
    xr_coords = {k: v for k, v in coords.items()}
    return xr.DataArray(arr, coords=xr_coords, dims=dims, name=name)


def xr_apply_rbig(
    da: xr.DataArray,
    model: AnnealedRBIG,
    feature_dim: str = "variable",
) -> xr.DataArray:
    """Apply a fitted RBIG model to an xarray DataArray.

    Parameters
    ----------
    da : xarray DataArray with feature dimension
    model : fitted AnnealedRBIG model
    feature_dim : name of the feature dimension

    Returns
    -------
    transformed : xarray DataArray
    """
    import xarray as xr

    X = da.values
    original_shape = X.shape
    if X.ndim > 2:
        X_2d = X.reshape(-1, X.shape[-1])
    else:
        X_2d = X

    Xt = model.transform(X_2d)
    Xt = Xt.reshape(original_shape)

    return xr.DataArray(Xt, coords=da.coords, dims=da.dims, name=da.name)
