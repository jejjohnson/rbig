"""Xarray spatiotemporal interface for RBIG.

This module bridges xarray Dataset / DataArray objects and the array-based
RBIG API.  Spatiotemporal data (e.g. climate fields with ``time``, ``lat``,
``lon`` dimensions) are reshaped into a 2-D ``(samples, features)`` matrix,
fed to RBIG, and then reconstructed back into the original xarray structure.
"""

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
    """Flatten an xarray spatiotemporal object into a 2-D sample matrix.

    Each observation is formed by stacking all spatial grid points for a
    single time step, giving ``n_samples = n_time * n_space`` rows.  For a
    :class:`xarray.Dataset`, each data variable becomes one feature column.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Source spatiotemporal data.  The object should have at least the
        ``time_dim`` and any of the ``spatial_dims`` as named dimensions.
    feature_vars : list of str or None, default None
        Variable names to include as feature columns (Dataset only).  If
        ``None``, all data variables are used.  Ignored for DataArray input.
    time_dim : str, default ``"time"``
        Name of the time dimension in ``ds``.
    spatial_dims : tuple of str, default ``("lat", "lon")``
        Names of the spatial dimensions.  All present spatial dimensions are
        included in the flattening.

    Returns
    -------
    matrix : np.ndarray, shape ``(n_time * n_space, n_features)``
        2-D array suitable for RBIG.  For a DataArray, ``n_features = 1``
        (or the size of any non-spatial, non-time trailing axis).  For a
        Dataset with ``n_vars`` variables, ``n_features = n_vars``.
    meta : dict
        Metadata required by :func:`matrix_to_xr_st` to reconstruct the
        original xarray object.  Keys depend on the input type:

        * ``"type"`` — ``"DataArray"`` or ``"Dataset"``
        * ``"shape"`` — original numpy array shape
        * ``"dims"`` — original dimension names
        * ``"coords"`` — coordinate arrays (DataArray only)
        * ``"feature_vars"`` — list of variable names (Dataset only)

    Notes
    -----
    The stacking operation is

    .. math::

        (n_{\\text{time}},\\, n_{\\text{lat}},\\, n_{\\text{lon}},\\, \\ldots)
        \\longrightarrow
        (n_{\\text{time}} \\times n_{\\text{space}},\\; n_{\\text{features}})

    where :math:`n_{\\text{space}} = \\prod_d n_d` over all spatial dimensions
    ``d`` present in ``ds``.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> da = xr.DataArray(
    ...     rng.standard_normal((10, 4, 5)),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> matrix, meta = xr_st_to_matrix(da)
    >>> matrix.shape  # 10*20 time-space samples, 1 feature
    (200, 1)
    >>> meta["type"]
    'DataArray'
    """
    import xarray as xr

    if isinstance(ds, xr.DataArray):
        arr = ds.values
        n_time = arr.shape[0] if time_dim in ds.dims else 1
        n_spatial = int(np.prod([ds.sizes[d] for d in spatial_dims if d in ds.dims]))
        # Collapse time and space into the sample axis
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
        # Each variable becomes one column; rows index (time * space)
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
    """Reconstruct an xarray object from a 2-D sample matrix.

    This is the inverse of :func:`xr_st_to_matrix`.

    Parameters
    ----------
    matrix : np.ndarray, shape ``(n_samples, n_features)``
        2-D array as produced (or transformed) by RBIG.
    meta : dict
        Metadata dict returned by :func:`xr_st_to_matrix`.  Must contain the
        ``"type"`` key (``"DataArray"`` or ``"Dataset"``) plus the
        corresponding shape and dimension information.
    time_dim : str, default ``"time"``
        Name of the time dimension (used for labelling, not reshaping).

    Returns
    -------
    out : xr.DataArray or xr.Dataset
        Reconstructed xarray object with the original shape and dimension
        names.  Coordinates are **not** re-attached (use
        :meth:`xr.DataArray.assign_coords` if needed).

    Notes
    -----
    The reshape inverts the stacking performed by :func:`xr_st_to_matrix`:

    .. math::

        (n_{\\text{time}} \\times n_{\\text{space}},\\; n_{\\text{features}})
        \\longrightarrow
        (n_{\\text{time}},\\, n_{\\text{lat}},\\, n_{\\text{lon}},\\, \\ldots)

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> da = xr.DataArray(
    ...     rng.standard_normal((10, 4, 5)),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> matrix, meta = xr_st_to_matrix(da)
    >>> da_back = matrix_to_xr_st(matrix, meta)
    >>> da_back.shape
    (10, 4, 5)
    """
    import xarray as xr

    if meta["type"] == "DataArray":
        # Restore the original multi-dimensional shape
        arr = matrix.reshape(meta["shape"])
        return xr.DataArray(arr, dims=meta["dims"], name="data")
    else:
        original_shape = meta["shape"]
        ds_dict = {}
        for i, var in enumerate(meta["feature_vars"]):
            # Each column of `matrix` corresponds to one data variable
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
    """Fit an RBIG model on xarray spatiotemporal data and return the transform.

    Converts ``ds`` to a 2-D matrix, calls ``model.fit_transform``, and
    reconstructs the result as an xarray object matching the original
    structure.

    Parameters
    ----------
    ds : xr.Dataset or xr.DataArray
        Input spatiotemporal data.
    model : AnnealedRBIG
        An unfitted RBIG model instance exposing ``fit_transform(X)``.
    feature_vars : list of str or None, default None
        Variable names to use as features (Dataset only).
    time_dim : str, default ``"time"``
        Name of the time dimension.
    spatial_dims : tuple of str, default ``("lat", "lon")``
        Names of the spatial dimensions.

    Returns
    -------
    matrix : np.ndarray, shape ``(n_time * n_space, n_features)``
        The 2-D matrix used for fitting (useful for diagnostics).
    transformed : xr.DataArray or xr.Dataset
        Gaussianised data reconstructed in the original xarray shape.

    Notes
    -----
    The pipeline is:

    .. math::

        \\text{ds}
        \\xrightarrow{\\text{xr\\_st\\_to\\_matrix}}
        \\mathbf{X}
        \\xrightarrow{\\text{RBIG fit\\_transform}}
        \\mathbf{X}_t
        \\xrightarrow{\\text{matrix\\_to\\_xr\\_st}}
        \\text{transformed}

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> da = xr.DataArray(
    ...     rng.standard_normal((20, 4, 5)),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> # model = AnnealedRBIG(n_layers=5)
    >>> # matrix, da_t = xr_rbig_fit_transform(da, model)
    """
    matrix, meta = xr_st_to_matrix(ds, feature_vars, time_dim, spatial_dims)
    Xt = model.fit_transform(matrix)
    reconstructed = matrix_to_xr_st(Xt, meta, time_dim)
    return matrix, reconstructed


class XarrayRBIG:
    """RBIG model with an xarray-aware interface.

    Wraps an :class:`~rbig._src.model.AnnealedRBIG` (or compatible class) so
    that it can be fitted and applied directly to :class:`xarray.DataArray` /
    :class:`xarray.Dataset` objects with spatiotemporal dimensions.  The
    underlying model operates on a 2-D ``(samples, features)`` matrix obtained
    via :func:`xr_st_to_matrix`.

    Parameters
    ----------
    n_layers : int, default 100
        Maximum number of RBIG layers.
    strategy : list or None, default None
        Rotation strategy list passed to the underlying RBIG model.  If
        ``None``, the default rotation of the model class is used.
    tol : float, default 1e-5
        Convergence tolerance for early stopping.
    random_state : int or None, default None
        Random seed for reproducibility.
    rbig_class : class or None, default None
        RBIG model class to instantiate.  Defaults to
        :class:`~rbig._src.model.AnnealedRBIG` when ``None``.
    rbig_kwargs : dict or None, default None
        Additional keyword arguments forwarded to ``rbig_class``.
    verbose : bool or int, default=False
        Controls progress bar display.  Passed through to the underlying
        RBIG model.

    Attributes
    ----------
    model_ : AnnealedRBIG
        The fitted underlying RBIG model.
    meta_ : dict
        xarray metadata captured during :meth:`fit`, used to reconstruct
        output arrays.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> rng = np.random.default_rng(0)
    >>> da = xr.DataArray(
    ...     rng.standard_normal((30, 4, 5)),
    ...     dims=["time", "lat", "lon"],
    ... )
    >>> xrbig = XarrayRBIG(n_layers=10, random_state=0)
    >>> # info = xrbig.fit(da)
    >>> # da_t = xrbig.transform(da)
    """

    def __init__(
        self,
        n_layers: int = 100,
        strategy: list | None = None,
        tol: float = 1e-5,
        random_state: int | None = None,
        rbig_class=None,
        rbig_kwargs: dict | None = None,
        verbose: bool | int = False,
    ):
        self.n_layers = n_layers
        self.strategy = strategy
        self.tol = tol
        self.random_state = random_state
        self.rbig_class = rbig_class
        self.rbig_kwargs = rbig_kwargs or {}
        self.verbose = verbose

    def fit(self, X) -> dict:
        """Fit the RBIG model to xarray data and return an information summary.

        Parameters
        ----------
        X : xr.DataArray or xr.Dataset
            Input spatiotemporal data.  Internally converted to a 2-D matrix
            via :func:`xr_st_to_matrix`.

        Returns
        -------
        info : dict
            Dictionary of RBIG information metrics (e.g. total correlation,
            entropy estimates) as returned by
            :func:`~rbig._src.metrics.information_summary`.
        """
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
        kwargs["verbose"] = self.verbose
        kwargs.update(self.rbig_kwargs)

        # Convert xarray → (n_samples, n_features) matrix and store metadata
        matrix, self.meta_ = xr_st_to_matrix(X)
        self.model_ = rbig_cls(**kwargs)
        self.model_.fit(matrix)
        return information_summary(self.model_, matrix)

    def transform(self, X):
        """Gaussianise samples and return an xarray object.

        Applies the fitted RBIG transform to ``X``, then reconstructs the
        original xarray structure.  Original coordinates and DataArray name
        are re-attached when possible.

        Parameters
        ----------
        X : xr.DataArray or xr.Dataset
            Data to transform.  Must have the same structure as the data
            passed to :meth:`fit`.

        Returns
        -------
        out : xr.DataArray or xr.Dataset
            Gaussianised data with the same shape and dimension names as ``X``.
        """
        matrix, _ = xr_st_to_matrix(X)
        Xt = self.model_.transform(matrix)
        out = matrix_to_xr_st(Xt, self.meta_)
        # Re-attach original xarray coordinates and name when available
        if hasattr(X, "assign_coords") and hasattr(X, "coords"):
            try:
                out = out.assign_coords(X.coords)
            except Exception:
                pass
        if hasattr(X, "name") and hasattr(out, "name") and X.name is not None:
            try:
                out.name = X.name
            except Exception:
                pass
        return out

    def score_samples(self, X):
        """Compute per-sample log-probability log p(x).

        Parameters
        ----------
        X : xr.DataArray or xr.Dataset
            Input data.

        Returns
        -------
        log_prob : np.ndarray, shape ``(n_samples,)``
            Log-probability of each sample under the fitted RBIG model.
        """
        matrix, _ = xr_st_to_matrix(X)
        return self.model_.score_samples(matrix)

    def mutual_information(self, X, Y) -> float:
        """Estimate mutual information between two xarray variables via RBIG.

        Fits independent RBIG models to ``X``, ``Y``, and their concatenation
        ``[X, Y]``, then computes:

        .. math::

            \\mathrm{MI}(X;\\,Y)
            = H(X) + H(Y) - H(X,\\,Y)

        where each differential entropy :math:`H` is estimated from the RBIG
        log-determinant accumulation.

        Parameters
        ----------
        X : xr.DataArray or xr.Dataset
            First variable.
        Y : xr.DataArray or xr.Dataset
            Second variable.  Must have the same number of samples as ``X``
            after flattening.

        Returns
        -------
        mi : float
            Estimated mutual information in nats.

        Notes
        -----
        All three RBIG models share the same ``n_layers``, ``tol``, and
        ``random_state`` settings as the parent :class:`XarrayRBIG` instance.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> rng = np.random.default_rng(0)
        >>> x = xr.DataArray(rng.standard_normal((50, 1)), dims=["time", "f"])
        >>> y = xr.DataArray(rng.standard_normal((50, 1)), dims=["time", "f"])
        >>> xrbig = XarrayRBIG(n_layers=5, random_state=0)
        >>> # mi = xrbig.mutual_information(x, y)
        """
        from rbig._src.metrics import entropy_rbig
        from rbig._src.model import AnnealedRBIG

        rbig_cls = self.rbig_class if self.rbig_class is not None else AnnealedRBIG
        kwargs = {
            "n_layers": self.n_layers,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        kwargs.update(self.rbig_kwargs)

        # Flatten both variables to 2-D matrices
        X_mat, _ = xr_st_to_matrix(X)
        Y_mat, _ = xr_st_to_matrix(Y)
        XY_mat = np.hstack([X_mat, Y_mat])  # joint representation (n, dx + dy)

        # Fit three separate RBIG models for H(X), H(Y), H(X,Y)
        mx = rbig_cls(**kwargs).fit(X_mat)
        my = rbig_cls(**kwargs).fit(Y_mat)
        mxy = rbig_cls(**kwargs).fit(XY_mat)

        hx = entropy_rbig(mx, X_mat)  # H(X)
        hy = entropy_rbig(my, Y_mat)  # H(Y)
        hxy = entropy_rbig(mxy, XY_mat)  # H(X, Y)
        # MI(X;Y) = H(X) + H(Y) - H(X,Y)
        return float(hx + hy - hxy)
