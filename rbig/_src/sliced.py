"""Sliced Iterative Gaussianization: SIGLayer, GIS, and SIG.

This module implements the Sliced Iterative Gaussianization family of
Dai & Seljak (2020).  The building block is :class:`SIGLayer`, which finds
``K`` orthonormal projection directions (via Stiefel-manifold optimization
or at random) and applies a 1D rational-quadratic spline Gaussianization
along each of them, leaving the orthogonal complement untouched.

Two scikit-learn estimators stack these layers greedily:

* :class:`GIS` -- *Gaussianization via Iterative Slicing*.  Maps arbitrary
  data to a standard Gaussian; oriented toward density estimation
  (``transform`` / ``score_samples``).
* :class:`SIG` -- *Sliced Iterative Gaussianization*.  The same underlying
  flow oriented toward generation (``sample`` / ``inverse_transform``),
  defaulting to a sliced-Wasserstein stopping signal.

Both compose with a shared whitening pre-transform and the shared
:class:`~rbig._src.convergence.StoppingCriterion`, and both define a valid
normalizing flow with a tractable log-determinant.

References
----------
Dai, B., & Seljak, U. (2020). Sliced Iterative Normalizing Flows.
https://arxiv.org/abs/2007.00674
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.convergence import StoppingCriterion
from rbig._src.rotation import PCARotation
from rbig._src.spline import RQSpline
from rbig._src.stiefel import (
    max_sliced_wasserstein_directions,
    random_orthogonal_directions,
)


class SIGLayer:
    """One sliced-Gaussianization layer: projection + per-direction splines.

    The layer finds ``K`` orthonormal directions ``A`` and Gaussianizes the
    source data along each via an independent :class:`RQSpline`.  The
    component orthogonal to ``A`` is passed through unchanged, so in the
    rotated basis the map is ``(g_1, ..., g_K, identity, ...)`` and its
    log-determinant is the sum of the per-direction spline log-derivatives.

    Parameters
    ----------
    n_directions : int, default 1
        Number of projection directions ``K``.
    n_knots : int, default 1000
        Knots per 1D spline (see :class:`RQSpline`).
    alpha : float, default 1e-3
        Lower clip on spline derivatives (regularizes extreme slopes).
    direction_method : {"stiefel", "random"}, default "stiefel"
        How projection directions are found.  ``"stiefel"`` maximizes the
        K-sliced Wasserstein distance between source and target; ``"random"``
        draws a random orthonormal frame.
    stiefel_lr : float, default 1.0
        Initial step size for the Stiefel line search.
    stiefel_max_iter : int, default 100
        Maximum Stiefel optimization iterations.
    kde_bandwidth : str, float, or None, default None
        Bandwidth passed to each spline's KDE.
    random_state : int or None, optional
        Seed for direction finding and the Gaussian reference.

    Attributes
    ----------
    A_ : np.ndarray of shape (n_features, n_directions)
        Fitted orthonormal projection directions.
    splines_ : list of RQSpline
        One fitted spline per direction.
    n_features_in_ : int
        Number of input features seen during :meth:`fit`.
    """

    def __init__(
        self,
        n_directions: int = 1,
        n_knots: int = 1000,
        alpha: float = 1e-3,
        direction_method: str = "stiefel",
        stiefel_lr: float = 1.0,
        stiefel_max_iter: int = 100,
        kde_bandwidth: str | float | None = None,
        random_state: int | None = None,
    ):
        self.n_directions = n_directions
        self.n_knots = n_knots
        self.alpha = alpha
        self.direction_method = direction_method
        self.stiefel_lr = stiefel_lr
        self.stiefel_max_iter = stiefel_max_iter
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state

    def fit(self, X: np.ndarray, X_target: np.ndarray | None = None) -> SIGLayer:
        """Find directions and fit a Gaussianizing spline along each.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Source data to Gaussianize.  The splines are fitted on the
            projections of ``X``.
        X_target : np.ndarray or None, optional
            Reference used only for direction finding with the ``"stiefel"``
            method.  When ``None`` a standard-Gaussian sample matching ``X``
            is drawn.

        Returns
        -------
        self : SIGLayer
            The fitted layer.
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        self.n_features_in_ = d
        k = max(1, min(self.n_directions, d))
        rng = np.random.default_rng(self.random_state)

        if self.direction_method == "random":
            self.A_ = random_orthogonal_directions(
                d, k, random_state=int(rng.integers(2**31))
            )
        elif self.direction_method == "stiefel":
            if X_target is None:
                X_target = rng.standard_normal((n, d))
            self.A_ = max_sliced_wasserstein_directions(
                X,
                X_target,
                k,
                max_iter=self.stiefel_max_iter,
                lr=self.stiefel_lr,
                random_state=int(rng.integers(2**31)),
            )
        else:
            raise ValueError(
                f"Unknown direction_method {self.direction_method!r}. "
                "Use 'stiefel' or 'random'."
            )

        X_proj = X @ self.A_  # (n, k)
        self.splines_ = [
            RQSpline(
                n_knots=self.n_knots,
                kde_bw=self.kde_bandwidth,
                min_derivative=self.alpha,
            ).fit(X_proj[:, j])
            for j in range(k)
        ]
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gaussianize along ``A`` and accumulate the log-determinant.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_features)
            Data with the ``K`` projected directions Gaussianized.
        log_det : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant of the layer.
        """
        X = np.asarray(X, dtype=float)
        X_proj = X @ self.A_
        residual = X - X_proj @ self.A_.T
        z_proj = np.empty_like(X_proj)
        log_det = np.zeros(X.shape[0])
        for j, spline in enumerate(self.splines_):
            z_proj[:, j], log_dz = spline.forward(X_proj[:, j])
            log_det += log_dz
        return residual + z_proj @ self.A_.T, log_det

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Invert the layer.

        Parameters
        ----------
        Z : np.ndarray of shape (n_samples, n_features)
            Data in the layer's output space.

        Returns
        -------
        X : np.ndarray of shape (n_samples, n_features)
            Recovered input-space data.
        """
        Z = np.asarray(Z, dtype=float)
        Z_proj = Z @ self.A_
        residual = Z - Z_proj @ self.A_.T
        x_proj = np.empty_like(Z_proj)
        for j, spline in enumerate(self.splines_):
            x_proj[:, j], _ = spline.inverse(Z_proj[:, j])
        return residual + x_proj @ self.A_.T


class _BaseSliced(TransformerMixin, BaseEstimator):
    """Shared machinery for the sliced-Gaussianization estimators.

    Subclasses (:class:`GIS`, :class:`SIG`) supply the parameter set via
    their own ``__init__`` and differ only in default configuration; the
    fit/transform/score logic lives here.
    """

    def _resolve_directions(self, d: int) -> int:
        """Resolve ``K`` per layer (default ``D // 2``, at least 1)."""
        if self.n_directions is None:
            return max(1, d // 2)
        return max(1, min(self.n_directions, d))

    def fit(self, X: np.ndarray, y=None) -> _BaseSliced:
        """Greedily stack sliced-Gaussianization layers.

        Whitens the data (optional), then repeatedly fits a
        :class:`SIGLayer` on the running representation, advancing it toward
        a standard Gaussian and monitoring a held-out
        :class:`StoppingCriterion` for early stopping.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for scikit-learn compatibility.

        Returns
        -------
        self : _BaseSliced
            The fitted model.
        """
        X = validate_data(self, X)
        n, d = X.shape
        if n < 2:
            raise ValueError(f"Need at least 2 samples, got {n}.")
        if self.whiten and n <= d:
            # PCA(whiten=True) keeps only min(n_samples, n_features) (and at
            # most n_samples - 1) components, so wide data yields a
            # rank-deficient, non-square whitening that breaks the bijection
            # (inverse_transform / sample would mismatch shapes).
            raise ValueError(
                f"whiten=True requires n_samples > n_features for a full-rank "
                f"square whitening; got n_samples={n}, n_features={d}. "
                f"Pass whiten=False for wide data."
            )
        self.n_features_in_ = d
        rng = np.random.default_rng(self.random_state)
        k = self._resolve_directions(d)

        # Optional whitening pre-transform.
        if self.whiten:
            self.whitener_ = PCARotation(whiten=True).fit(X)
            Xw = self.whitener_.transform(X)
        else:
            self.whitener_ = None
            Xw = X

        crit = StoppingCriterion(
            metric=self.stopping_metric,
            patience=self.patience,
            validation_fraction=self.validation_fraction,
            random_state=self.random_state,
        )
        X_train, X_val = crit.split(Xw, random_state=self.random_state)

        self.layers_: list[SIGLayer] = []
        log_det_val = np.zeros(X_val.shape[0])
        R_train = X_train.copy()
        R_val = X_val.copy()

        for _ in range(self.n_layers):
            target = (
                rng.standard_normal(R_train.shape)
                if self.direction_method == "stiefel"
                else None
            )
            layer = SIGLayer(
                n_directions=k,
                n_knots=self.n_knots,
                alpha=self.alpha,
                direction_method=self.direction_method,
                stiefel_lr=self.stiefel_lr,
                stiefel_max_iter=self.stiefel_max_iter,
                kde_bandwidth=self.kde_bandwidth,
                random_state=int(rng.integers(2**31)),
            )
            layer.fit(R_train, X_target=target)
            R_train, _ = layer.transform(R_train)
            R_val, ld_val = layer.transform(R_val)
            log_det_val += ld_val
            self.layers_.append(layer)

            if crit.update(R_val, log_det=log_det_val):
                break

        # Keep only the prefix up to the best validation iterate.  Any layers
        # appended after the metric peaked (during the patience window) only
        # make the selected stopping metric worse, so they are discarded.
        if crit.best_iter_ >= 0:
            self.layers_ = self.layers_[: crit.best_iter_ + 1]

        self.stopping_criterion_ = crit
        self.n_layers_ = len(self.layers_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map data to the (approximately) Gaussian latent space."""
        check_is_fitted(self)
        Xt = validate_data(self, X, reset=False)
        if self.whitener_ is not None:
            Xt = self.whitener_.transform(Xt)
        for layer in self.layers_:
            Xt, _ = layer.transform(Xt)
        return Xt

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map latent-space data back to the original input space."""
        check_is_fitted(self)
        Xt = np.asarray(Z, dtype=float)
        for layer in reversed(self.layers_):
            Xt = layer.inverse_transform(Xt)
        if self.whitener_ is not None:
            Xt = self.whitener_.inverse_transform(Xt)
        return Xt

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood via the change-of-variables formula."""
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        log_det = np.zeros(X.shape[0])
        if self.whitener_ is not None:
            log_det += self.whitener_.log_det_jacobian(X)
            Xt = self.whitener_.transform(X)
        else:
            Xt = X.copy()
        for layer in self.layers_:
            Xt, ld = layer.transform(Xt)
            log_det += ld
        log_pz = np.sum(stats.norm.logpdf(Xt), axis=1)
        return log_pz + log_det

    def score(self, X: np.ndarray, y=None) -> float:
        """Mean per-sample log-likelihood in nats."""
        return float(np.mean(self.score_samples(X)))

    def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        """Generate samples by inverting Gaussian noise through the flow."""
        check_is_fitted(self)
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal((n_samples, self.n_features_in_))
        return self.inverse_transform(Z)


class GIS(_BaseSliced):
    """Gaussianization via Iterative Slicing (Dai & Seljak 2020).

    Maps arbitrary distributions to a standard Gaussian via greedy layer
    stacking.  Each layer finds optimal projection directions and applies 1D
    spline Gaussianization along them.  Oriented toward density estimation:
    the headline methods are :meth:`transform` and :meth:`score_samples`.

    Parameters
    ----------
    n_layers : int, default 500
        Maximum number of layers (early-stopped via ``patience``).
    n_directions : int or None, default None
        Directions per layer ``K``.  ``None`` selects ``D // 2``.
    n_knots : int, default 1000
        Knots per 1D spline.
    alpha : float, default 1e-3
        Lower clip on spline derivatives.
    whiten : bool, default True
        Apply a PCA whitening pre-transform before stacking layers.
    direction_method : {"stiefel", "random"}, default "stiefel"
        Projection-direction strategy.
    stopping_metric : {"log_likelihood", "swd", "total_correlation"}, default "log_likelihood"
        Signal driving early stopping.
    validation_fraction : float, default 0.2
        Held-out fraction for the stopping criterion.
    patience : int, default 10
        Non-improving layers tolerated before stopping.
    stiefel_lr : float, default 1.0
        Initial Stiefel line-search step size.
    stiefel_max_iter : int, default 100
        Maximum Stiefel iterations per layer.
    kde_bandwidth : str, float, or None, default None
        Bandwidth for each spline's KDE.
    random_state : int or None, optional
        Seed for reproducibility.

    Attributes
    ----------
    layers_ : list of SIGLayer
        Fitted layers in forward (data -> Gaussian) order.
    whitener_ : PCARotation or None
        Fitted whitening pre-transform.
    stopping_criterion_ : StoppingCriterion
        The fitted stopping tracker (exposes ``history_``, ``best_iter_``).
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import GIS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((400, 3))
    >>> model = GIS(n_layers=5, random_state=0).fit(X)
    >>> Z = model.transform(X)
    >>> Z.shape
    (400, 3)
    """

    def __init__(
        self,
        n_layers: int = 500,
        n_directions: int | None = None,
        n_knots: int = 1000,
        alpha: float = 1e-3,
        whiten: bool = True,
        direction_method: str = "stiefel",
        stopping_metric: str = "log_likelihood",
        validation_fraction: float = 0.2,
        patience: int = 10,
        stiefel_lr: float = 1.0,
        stiefel_max_iter: int = 100,
        kde_bandwidth: str | float | None = None,
        random_state: int | None = None,
    ):
        self.n_layers = n_layers
        self.n_directions = n_directions
        self.n_knots = n_knots
        self.alpha = alpha
        self.whiten = whiten
        self.direction_method = direction_method
        self.stopping_metric = stopping_metric
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.stiefel_lr = stiefel_lr
        self.stiefel_max_iter = stiefel_max_iter
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state


class SIG(_BaseSliced):
    """Sliced Iterative Gaussianization (Dai & Seljak 2020).

    The generative-oriented sibling of :class:`GIS`.  It builds the same
    sliced-Gaussianization flow but is configured and documented for
    sampling: draw ``z ~ N(0, I)`` and push it through
    :meth:`inverse_transform` (or :meth:`sample`) to obtain data-like
    samples.  Defaults to a sliced-Wasserstein stopping signal, which tracks
    sample quality more directly than the validation log-likelihood.

    Parameters
    ----------
    n_layers : int, default 500
        Maximum number of layers (early-stopped via ``patience``).
    n_directions : int or None, default None
        Directions per layer ``K``.  ``None`` selects ``D // 2``.
    n_knots : int, default 1000
        Knots per 1D spline.
    alpha : float, default 1e-3
        Lower clip on spline derivatives.
    whiten : bool, default True
        Apply a PCA whitening pre-transform.
    direction_method : {"stiefel", "random"}, default "stiefel"
        Projection-direction strategy.
    stopping_metric : {"log_likelihood", "swd", "total_correlation"}, default "swd"
        Signal driving early stopping.
    validation_fraction : float, default 0.2
        Held-out fraction for the stopping criterion.
    patience : int, default 10
        Non-improving layers tolerated before stopping.
    stiefel_lr : float, default 1.0
        Initial Stiefel line-search step size.
    stiefel_max_iter : int, default 100
        Maximum Stiefel iterations per layer.
    kde_bandwidth : str, float, or None, default None
        Bandwidth for each spline's KDE.
    random_state : int or None, optional
        Seed for reproducibility.

    Attributes
    ----------
    layers_ : list of SIGLayer
        Fitted layers in forward (data -> Gaussian) order; sampling applies
        them in reverse.
    whitener_ : PCARotation or None
        Fitted whitening pre-transform.
    stopping_criterion_ : StoppingCriterion
        The fitted stopping tracker.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import SIG
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((400, 2))
    >>> model = SIG(n_layers=5, random_state=0).fit(X)
    >>> samples = model.sample(50, random_state=1)
    >>> samples.shape
    (50, 2)
    """

    def __init__(
        self,
        n_layers: int = 500,
        n_directions: int | None = None,
        n_knots: int = 1000,
        alpha: float = 1e-3,
        whiten: bool = True,
        direction_method: str = "stiefel",
        stopping_metric: str = "swd",
        validation_fraction: float = 0.2,
        patience: int = 10,
        stiefel_lr: float = 1.0,
        stiefel_max_iter: int = 100,
        kde_bandwidth: str | float | None = None,
        random_state: int | None = None,
    ):
        self.n_layers = n_layers
        self.n_directions = n_directions
        self.n_knots = n_knots
        self.alpha = alpha
        self.whiten = whiten
        self.direction_method = direction_method
        self.stopping_metric = stopping_metric
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.stiefel_lr = stiefel_lr
        self.stiefel_max_iter = stiefel_max_iter
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state
