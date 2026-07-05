"""Negentropy-based dimensionality reduction (issue #126).

:class:`RBIGReducer` keeps the axes that carry *non-Gaussian structure*
(negentropy) rather than merely high variance: PCA-whitened axes are ranked
by ``J_k = H_gauss - H(Z_k)`` and pruned below a threshold (or outside the
top ``n_components``).  Low-variance bimodal signal survives; high-variance
Gaussian noise is dropped — the two cases where variance criteria fail.

Design note (deviation from the original issue sketch, on purpose): the
ranking is computed in the *whitened rotation space*, not in the fully
converged RBIG latent space.  A converged flow Gaussianizes every marginal
by construction, so per-axis negentropy there is ~0 for all axes and cannot
rank anything — the same class of error as the draft design doc's
"Omega = 0 for Gaussians" claim.  Whitening removes the variance signal
(every axis has unit variance) so that negentropy is the *only* remaining
per-axis criterion.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.metrics import entropy_quantile_spacing
from rbig._src.rotation import PCARotation

# Differential entropy of N(0, 1); whitened axes have unit variance.
_H_GAUSS = 0.5 * np.log(2.0 * np.pi * np.e)


class RBIGReducer(TransformerMixin, BaseEstimator):
    """Dimensionality reduction by negentropy pruning of whitened axes.

    Where PCA keeps high-variance axes, ``RBIGReducer`` whitens (removing
    the variance signal entirely) and keeps the axes with the highest
    negentropy ``J_k = H_gauss - H(Z_k) >= 0`` — the per-axis measure of
    non-Gaussian structure (multi-modality, skew, heavy tails).

    Parameters
    ----------
    threshold : float or "auto", default 0.01
        Keep axes with ``J_k >= threshold`` (nats).  ``"auto"`` uses an
        adaptive noise floor: the median negentropy of the bottom half of
        axes plus 3x its spread.  Ignored when ``n_components`` is set.
    n_components : int or None, default None
        Keep exactly the top ``n_components`` axes by negentropy
        (overrides ``threshold``).
    n_quantiles : int, default 100
        Spacing-estimator resolution for the per-axis entropies.
    random_state : int or None, optional
        Present for API consistency; the reduction is deterministic.

    Attributes
    ----------
    rotation_ : PCARotation
        The fitted whitening rotation.
    negentropies_ : np.ndarray of shape (n_features,)
        Per-axis negentropy in the whitened space, in nats.
    keep_mask_ : np.ndarray of shape (n_features,)
        Boolean mask of the retained axes.
    explained_negentropy_ratio_ : float
        Fraction of total negentropy retained by the kept axes.
    d_out_ : int
        Number of retained axes.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGReducer, make_signal_plus_noise_dims
    >>> X, meta = make_signal_plus_noise_dims(n_samples=1000, seed=0)
    >>> red = RBIGReducer(n_components=2).fit(X)
    >>> red.transform(X).shape
    (1000, 2)
    """

    def __init__(
        self,
        threshold: float | str = 0.01,
        n_components: int | None = None,
        n_quantiles: int = 100,
        random_state: int | None = None,
    ):
        self.threshold = threshold
        self.n_components = n_components
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RBIGReducer:
        """Whiten, measure per-axis negentropy, and choose the kept axes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : RBIGReducer
            The fitted reducer.
        """
        X = validate_data(self, X)
        n, d = X.shape
        if n <= d:
            # PCA whitening keeps at most n_samples - 1 components; wide or
            # single-sample data would break the square inverse rotation.
            raise ValueError(
                f"RBIGReducer requires n_samples > n_features for a "
                f"full-rank whitening; got n_samples = {n}, n_features = {d}."
            )
        self.rotation_ = PCARotation(whiten=True).fit(X)
        Z = self.rotation_.transform(X)  # Z: (n, d), unit variance per axis

        neg = np.array(
            [
                max(
                    _H_GAUSS
                    + 0.5 * np.log(max(Z[:, k].var(), 1e-300))
                    - entropy_quantile_spacing(Z[:, k], n_quantiles=self.n_quantiles),
                    0.0,
                )
                for k in range(d)
            ]
        )
        self.negentropies_ = neg  # (d,)

        if self.n_components is not None:
            k = int(self.n_components)
            if not 0 < k <= d:
                raise ValueError(f"n_components must be in [1, {d}], got {k}.")
            keep = np.zeros(d, dtype=bool)
            keep[np.argsort(neg)[::-1][:k]] = True
        else:
            if self.threshold == "auto":
                # Adaptive noise floor from the least-structured half.
                bottom = np.sort(neg)[: max(d // 2, 1)]
                cut = float(np.median(bottom) + 3.0 * bottom.std())
            else:
                cut = float(self.threshold)
            keep = neg >= cut
            if not keep.any():
                # Never return an empty representation.
                keep[np.argmax(neg)] = True
        self.keep_mask_ = keep
        self.d_out_ = int(keep.sum())
        total = float(neg.sum())
        self.explained_negentropy_ratio_ = (
            float(neg[keep].sum() / total) if total > 0 else 1.0
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project to the whitened space and keep the selected axes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z_red : np.ndarray of shape (n_samples, d_out_)
            Retained whitened coordinates.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.rotation_.transform(X)[:, self.keep_mask_]

    def inverse_transform(self, Z_red: np.ndarray) -> np.ndarray:
        """Zero-fill the pruned axes and invert the whitening rotation.

        Zero is the mean of every whitened axis, so this is the
        maximum-entropy imputation: the reconstruction error is exactly the
        discarded structure, not an interpolation artifact.

        Parameters
        ----------
        Z_red : np.ndarray of shape (n_samples, d_out_)
            Reduced representation.

        Returns
        -------
        X_rec : np.ndarray of shape (n_samples, n_features)
            Reconstruction in the original space.
        """
        check_is_fitted(self)
        Z_red = np.asarray(Z_red, dtype=float)
        Z = np.zeros((Z_red.shape[0], self.keep_mask_.size))
        Z[:, self.keep_mask_] = Z_red
        return self.rotation_.inverse_transform(Z)

    def negentropy_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Sorted negentropies and their cumulative ratio (scree analog).

        Returns
        -------
        spectrum : np.ndarray of shape (n_features,)
            Per-axis negentropies, sorted descending.
        cumulative_ratio : np.ndarray of shape (n_features,)
            Cumulative share of total negentropy (monotone, ends at 1.0).
        """
        check_is_fitted(self)
        spectrum = np.sort(self.negentropies_)[::-1]
        total = spectrum.sum()
        cumulative = (
            np.cumsum(spectrum) / total if total > 0 else np.ones_like(spectrum)
        )
        return spectrum, cumulative

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared reconstruction error of ``inverse(transform(X))``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        mse : float
            Mean squared error over all entries.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return float(np.mean((self.inverse_transform(self.transform(X)) - X) ** 2))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Feature names of the reduced output: ``rbigreducer0 ..``.

        Parameters
        ----------
        input_features : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        names : np.ndarray of shape (d_out_,)
            Output feature names.
        """
        check_is_fitted(self)
        return np.asarray(
            [f"rbigreducer{i}" for i in range(self.d_out_)], dtype=object
        )
