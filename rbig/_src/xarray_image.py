"""Xarray interface for image transforms.

This module provides utilities to convert between xarray DataArrays that
represent 2-D images (with named spatial dimensions) and the 2-D
``(n_pixels, n_features)`` matrix format expected by RBIG and other
array-based transforms.
"""

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
    """Flatten an xarray image DataArray into a 2-D pixel matrix.

    Reshapes a 2-D (or 2-D + feature) DataArray into a ``(n_pixels, n_features)``
    matrix where ``n_pixels = ∏ sizes(spatial_dims)``.  For a purely 2-D
    grayscale image, ``n_features = 1``; for an image with a trailing feature
    axis the last dimension is preserved.

    Parameters
    ----------
    da : xr.DataArray, shape ``(x, y)`` or ``(x, y, n_features)``
        Source image DataArray.  The first two axes should correspond to
        ``spatial_dims``; any additional trailing axis is treated as the
        feature axis.
    spatial_dims : tuple of str, default ``("x", "y")``
        Names of the two spatial dimensions in ``da``.  Used only to record
        coordinate information in the returned metadata dict.

    Returns
    -------
    matrix : np.ndarray, shape ``(n_pixels, n_features)``
        Row-major flattening of the spatial axes:

        .. math::

            (n_x,\\; n_y,\\; [n_f])
            \\longrightarrow
            (n_x \\cdot n_y,\\; n_f)

        where ``n_f = 1`` for grayscale and ``n_f = da.shape[-1]`` for
        multi-feature images.
    coords : dict
        Mapping from dimension name to coordinate array.  All dimensions of
        ``da`` are included.  Pass this dict to :func:`matrix_to_xr_image` to
        reconstruct the DataArray.

    Notes
    -----
    The operation is equivalent to:

    .. math::

        \\mathbf{M}[i, :] = \\operatorname{vec}(\\text{image pixel } i)

    where pixels are indexed in C (row-major) order.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> img = xr.DataArray(
    ...     rng.standard_normal((16, 16)),
    ...     dims=["x", "y"],
    ...     coords={"x": np.arange(16), "y": np.arange(16)},
    ... )
    >>> matrix, coords = xr_image_to_matrix(img)
    >>> matrix.shape  # 16*16 pixels, 1 grayscale feature
    (256, 1)
    >>> set(coords.keys()) == {"x", "y"}
    True
    """
    # Store per-dimension coordinates for exact reconstruction
    coords = {dim: da[dim].values for dim in da.dims}
    # Flatten spatial axes; keep trailing feature axis if present
    matrix = (
        da.values.reshape(-1, 1)  # (H*W, 1) for grayscale
        if da.ndim == 2
        else da.values.reshape(-1, da.shape[-1])  # (H*W, C) for multi-feature
    )
    return matrix, coords


def matrix_to_xr_image(
    matrix: np.ndarray,
    coords: dict,
    spatial_shape: tuple[int, ...],
    name: str = "data",
) -> xr.DataArray:
    """Reconstruct an xarray image DataArray from a 2-D pixel matrix.

    This is the inverse of :func:`xr_image_to_matrix`.

    Parameters
    ----------
    matrix : np.ndarray, shape ``(n_pixels, n_features)``
        Pixel matrix as produced (or transformed) by a RBIG model.
        ``n_pixels`` must equal ``prod(spatial_shape)``.
    coords : dict
        Coordinate dictionary returned by :func:`xr_image_to_matrix`,
        mapping dimension names to their coordinate arrays.
    spatial_shape : tuple of int
        Target spatial shape of the output array, e.g. ``(H, W)`` or
        ``(H, W, C)``.  The matrix is reshaped to this shape.
    name : str, default ``"data"``
        Name assigned to the reconstructed DataArray.

    Returns
    -------
    da : xr.DataArray, shape ``spatial_shape``
        Reconstructed image DataArray with the original dimension names and
        coordinate arrays.

    Notes
    -----
    The reshape inverts the flattening of :func:`xr_image_to_matrix`:

    .. math::

        (n_x \\cdot n_y,\\; n_f)
        \\longrightarrow
        (n_x,\\; n_y,\\; [n_f])

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> img = xr.DataArray(
    ...     rng.standard_normal((8, 8)),
    ...     dims=["x", "y"],
    ...     coords={"x": np.arange(8), "y": np.arange(8)},
    ... )
    >>> matrix, coords = xr_image_to_matrix(img)
    >>> img_back = matrix_to_xr_image(matrix, coords, spatial_shape=(8, 8))
    >>> img_back.shape
    (8, 8)
    >>> img_back.name
    'data'
    """
    import xarray as xr

    arr = matrix.reshape(spatial_shape)  # (H, W) or (H, W, C)
    dims = list(coords.keys())
    xr_coords = {k: v for k, v in coords.items()}
    return xr.DataArray(arr, coords=xr_coords, dims=dims, name=name)


def xr_apply_rbig(
    da: xr.DataArray,
    model: AnnealedRBIG,
    feature_dim: str = "variable",
) -> xr.DataArray:
    """Apply a fitted RBIG model to an xarray DataArray.

    Reshapes the DataArray into a 2-D matrix, runs the fitted ``model``'s
    ``transform``, and returns the result with the original shape, dimension
    names, coordinates, and DataArray name preserved.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.  If ``da.ndim > 2``, all axes except the last are
        merged into the sample axis; otherwise the array is used as-is.
    model : AnnealedRBIG
        A fitted RBIG model instance exposing a ``transform(X)`` method
        where ``X`` has shape ``(n_samples, n_features)``.
    feature_dim : str, default ``"variable"``
        Name of the feature dimension (informational; not used for reshaping,
        but available for extension to multi-feature DataArrays).

    Returns
    -------
    transformed : xr.DataArray
        Transformed DataArray with:

        * the same shape as ``da``
        * the same ``dims``, ``coords``, and ``name``
        * values replaced by the RBIG-transformed values

    Notes
    -----
    The pipeline for a 3-D input ``(d0, d1, n_features)`` is:

    .. math::

        (d_0,\\; d_1,\\; n_f)
        \\xrightarrow{\\text{reshape}}
        (d_0 \\cdot d_1,\\; n_f)
        \\xrightarrow{\\text{RBIG.transform}}
        (d_0 \\cdot d_1,\\; n_f)
        \\xrightarrow{\\text{reshape}}
        (d_0,\\; d_1,\\; n_f)

    For 2-D input the data is passed directly without reshaping.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> da = xr.DataArray(
    ...     rng.standard_normal((20, 3)),
    ...     dims=["sample", "feature"],
    ...     coords={"sample": np.arange(20), "feature": ["r", "g", "b"]},
    ...     name="pixels",
    ... )
    >>> # model = AnnealedRBIG(n_layers=5).fit(da.values)
    >>> # transformed = xr_apply_rbig(da, model)
    >>> # transformed.shape == da.shape
    """
    import xarray as xr

    X = da.values
    original_shape = X.shape
    # Collapse leading spatial axes into the sample axis if ndim > 2
    if X.ndim > 2:
        X_2d = X.reshape(-1, X.shape[-1])  # (n_pixels, n_features)
    else:
        X_2d = X  # already (n_samples, n_features)

    Xt = model.transform(X_2d)  # (n_samples, n_features) → transformed
    Xt = Xt.reshape(original_shape)  # restore original spatial structure

    return xr.DataArray(Xt, coords=da.coords, dims=da.dims, name=da.name)
