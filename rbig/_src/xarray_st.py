"""xarray spatiotemporal support for RBIG."""
from typing import List

import numpy as np

try:
    import xarray as xr
    _XARRAY_AVAILABLE = True
except ImportError:
    _XARRAY_AVAILABLE = False


def _require_xarray():
    if not _XARRAY_AVAILABLE:
        raise ImportError(
            "xarray is required for this function. Install it with: pip install xarray"
        )


def rbig_dataset(
    ds,
    features: List[str],
    n_layers: int = 1000,
    **rbig_kwargs,
):
    """Apply RBIG to a set of variables in an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
    features : list of str
        Variable names to use as features.
    n_layers : int
    **rbig_kwargs
        Passed to AnnealedRBIG.

    Returns
    -------
    model : AnnealedRBIG
        Fitted model.
    ds_gauss : xr.Dataset
        Gaussianized dataset.
    """
    _require_xarray()
    from rbig._src.model import AnnealedRBIG

    X = np.stack([ds[f].values.ravel() for f in features], axis=1)
    model = AnnealedRBIG(n_layers=n_layers, **rbig_kwargs).fit(X)
    Z = model.transform(X)

    ds_gauss = xr.Dataset(
        {f: (ds[f].dims, Z[:, i].reshape(ds[f].shape)) for i, f in enumerate(features)},
        coords=ds.coords,
    )
    return model, ds_gauss


def rbig_dataarray(
    da,
    dim: str = "time",
    n_layers: int = 1000,
    **rbig_kwargs,
):
    """Apply RBIG to an xr.DataArray along a given dimension.

    Parameters
    ----------
    da : xr.DataArray
    dim : str
        Dimension to treat as samples.
    n_layers : int
    **rbig_kwargs

    Returns
    -------
    model : AnnealedRBIG
    da_gauss : xr.DataArray
    """
    _require_xarray()
    from rbig._src.model import AnnealedRBIG

    # Move the sample dimension to axis 0, reshape the rest into features
    da_t = da.transpose(dim, ...)
    values = da_t.values
    n_samples = values.shape[0]
    orig_feature_shape = values.shape[1:]
    X = values.reshape(n_samples, -1)

    model = AnnealedRBIG(n_layers=n_layers, **rbig_kwargs).fit(X)
    Z = model.transform(X)
    Z_shaped = Z.reshape((n_samples,) + orig_feature_shape)

    da_gauss = xr.DataArray(Z_shaped, dims=da_t.dims, coords=da_t.coords)
    return model, da_gauss.transpose(*da.dims)


def total_correlation_dataset(
    ds,
    features: List[str],
    **rbig_kwargs,
) -> float:
    """Compute total correlation for features in an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
    features : list of str
    **rbig_kwargs

    Returns
    -------
    tc : float
    """
    _require_xarray()
    from rbig._src.metrics import total_correlation

    model, _ = rbig_dataset(ds, features, **rbig_kwargs)
    return total_correlation(model)


def entropy_dataset(
    ds,
    features: List[str],
    **rbig_kwargs,
) -> float:
    """Compute differential entropy for features in an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
    features : list of str
    **rbig_kwargs

    Returns
    -------
    h : float
    """
    _require_xarray()
    from rbig._src.metrics import entropy_rbig

    model, _ = rbig_dataset(ds, features, **rbig_kwargs)
    return entropy_rbig(model)


def mutual_information_dataset(
    ds,
    X_features: List[str],
    Y_features: List[str],
    **rbig_kwargs,
) -> float:
    """Compute mutual information between two sets of features in an xr.Dataset.

    Parameters
    ----------
    ds : xr.Dataset
    X_features : list of str
    Y_features : list of str
    **rbig_kwargs

    Returns
    -------
    mi : float
    """
    _require_xarray()
    from rbig._src.metrics import mutual_information

    X = np.stack([ds[f].values.ravel() for f in X_features], axis=1)
    Y = np.stack([ds[f].values.ravel() for f in Y_features], axis=1)
    return mutual_information(X, Y, **rbig_kwargs)
