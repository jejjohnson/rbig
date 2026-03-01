"""Xarray spatiotemporal interface for RBIG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import xarray as xr

    from rbig._src.model import AnnealedRBIG


def xr_st_to_matrix(
    ds: xr.Dataset | xr.DataArray,
    feature_vars: list[str] | None = None,
    time_dim: str = "time",
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
) -> tuple[np.ndarray, dict]:
    """Convert xarray spatiotemporal data to 2D matrix.

    Stacks spatial dimensions and uses variables as features.

    Parameters
    ----------
    ds : xarray Dataset or DataArray
    feature_vars : list of variable names (Dataset only)
    time_dim : name of time dimension
    spatial_dims : names of spatial dimensions

    Returns
    -------
    matrix : array of shape (n_time * n_space, n_features) or (n_samples, n_features)
    meta : dict with metadata for reconstruction
    """
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        arr = ds.values
        n_time = arr.shape[0] if time_dim in ds.dims else 1
        n_spatial = int(np.prod([ds.sizes[d] for d in spatial_dims if d in ds.dims]))
        matrix = (
            arr.reshape(n_time * n_spatial, -1) if arr.ndim > 1 else arr.reshape(-1, 1)
        )
        meta = {
            "type": "DataArray",
            "shape": arr.shape,
            "dims": ds.dims,
            "coords": {k: v.values for k, v in ds.coords.items()},
        }
    else:
        if feature_vars is None:
            feature_vars = list(ds.data_vars)
        arrays = [ds[v].values for v in feature_vars]
        n_time = ds.sizes.get(time_dim, 1)
        n_spatial = int(np.prod([ds.sizes[d] for d in spatial_dims if d in ds.sizes]))
        matrix = np.stack([a.reshape(n_time * n_spatial) for a in arrays], axis=-1)
        meta = {
            "type": "Dataset",
            "feature_vars": feature_vars,
            "shape": arrays[0].shape,
            "dims": list(ds.dims),
        }

    return matrix, meta


def matrix_to_xr_st(
    matrix: np.ndarray,
    meta: dict,
    time_dim: str = "time",
) -> xr.Dataset | xr.DataArray:
    """Reconstruct xarray object from 2D matrix.

    Parameters
    ----------
    matrix : array of shape (n_samples, n_features)
    meta : dict from xr_st_to_matrix
    time_dim : name of time dimension

    Returns
    -------
    reconstructed xarray object
    """
    import xarray as xr

    if meta["type"] == "DataArray":
        arr = matrix.reshape(meta["shape"])
        return xr.DataArray(arr, dims=meta["dims"], name="data")
    else:
        original_shape = meta["shape"]
        ds_dict = {}
        for i, var in enumerate(meta["feature_vars"]):
            ds_dict[var] = xr.DataArray(
                matrix[:, i].reshape(original_shape), dims=meta["dims"]
            )
        return xr.Dataset(ds_dict)


def xr_rbig_fit_transform(
    ds: xr.Dataset | xr.DataArray,
    model: AnnealedRBIG,
    feature_vars: list[str] | None = None,
    time_dim: str = "time",
    spatial_dims: tuple[str, ...] = ("lat", "lon"),
) -> tuple[np.ndarray, xr.Dataset | xr.DataArray]:
    """Fit RBIG and transform xarray spatiotemporal data.

    Returns
    -------
    matrix : the 2D matrix used for fitting
    transformed : xarray with transformed data
    """
    matrix, meta = xr_st_to_matrix(ds, feature_vars, time_dim, spatial_dims)
    Xt = model.fit_transform(matrix)
    reconstructed = matrix_to_xr_st(Xt, meta, time_dim)
    return matrix, reconstructed


class XarrayRBIG:
    """RBIG model for xarray datasets."""

    def __init__(
        self,
        n_layers: int = 100,
        strategy: list | None = None,
        tol: float = 1e-5,
        random_state: int | None = None,
        rbig_class=None,
        rbig_kwargs: dict | None = None,
    ):
        self.n_layers = n_layers
        self.strategy = strategy
        self.tol = tol
        self.random_state = random_state
        self.rbig_class = rbig_class
        self.rbig_kwargs = rbig_kwargs or {}

    def fit(self, X) -> dict:
        """Fit RBIG model to xarray data."""
        from rbig._src.metrics import information_summary
        from rbig._src.model import AnnealedRBIG

        rbig_cls = self.rbig_class if self.rbig_class is not None else AnnealedRBIG
        kwargs = {
            "n_layers": self.n_layers,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        if self.strategy is not None:
            kwargs["strategy"] = self.strategy
        kwargs.update(self.rbig_kwargs)

        matrix, self.meta_ = xr_st_to_matrix(X)
        self.model_ = rbig_cls(**kwargs)
        self.model_.fit(matrix)
        return information_summary(self.model_, matrix)

    def transform(self, X):
        """Gaussianize samples."""
        matrix, _ = xr_st_to_matrix(X)
        Xt = self.model_.transform(matrix)
        return matrix_to_xr_st(Xt, self.meta_)

    def score_samples(self, X):
        """Per-sample log p(x)."""
        matrix, _ = xr_st_to_matrix(X)
        return self.model_.score_samples(matrix)

    def mutual_information(self, X, Y) -> float:
        """Mutual information between two xarray variables."""
        from rbig._src.metrics import entropy_rbig
        from rbig._src.model import AnnealedRBIG

        rbig_cls = self.rbig_class if self.rbig_class is not None else AnnealedRBIG
        kwargs = {
            "n_layers": self.n_layers,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        kwargs.update(self.rbig_kwargs)

        X_mat, _ = xr_st_to_matrix(X)
        Y_mat, _ = xr_st_to_matrix(Y)
        XY_mat = np.hstack([X_mat, Y_mat])

        mx = rbig_cls(**kwargs).fit(X_mat)
        my = rbig_cls(**kwargs).fit(Y_mat)
        mxy = rbig_cls(**kwargs).fit(XY_mat)

        hx = entropy_rbig(mx, X_mat)
        hy = entropy_rbig(my, Y_mat)
        hxy = entropy_rbig(mxy, XY_mat)
        return float(hx + hy - hxy)
