"""Log-likelihood anomaly scoring on the RBIG density (issue #125).

:class:`RBIGOutlierDetector` is a thin ``OutlierMixin`` consumer of
:class:`~rbig._src.model.AnnealedRBIG`: the anomaly score of a point is its
exact log-density under the fitted flow, and the decision threshold is the
``contamination`` quantile of the training scores.

The marginals are deliberately the plain empirical (clipping) ones:
out-of-range extremes map to the most extreme rank, which is exactly the
wanted behavior for anomaly *ranking* — in contrast to the tail-extended
marginals recommended for likelihood/sampling workflows (see the boundary
notebooks 07/18: past the support the clipped log-density plunges and then
flattens, so ranking survives while absolute calibration does not).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.model import AnnealedRBIG


class RBIGOutlierDetector(OutlierMixin, BaseEstimator):
    """Density-based outlier detection via the RBIG normalizing flow.

    Global and calibrated: ``score_samples(x)`` is the exact model
    log-density ``log p(x)`` (higher = more normal), and ``predict``
    flags the ``contamination`` fraction with the lowest training density
    as outliers — the complement to depth-based (IsolationForest) and
    local (LOF) detectors.

    Parameters
    ----------
    n_layers : int, default 50
        Maximum RBIG layers (early-stopped via ``patience``).
    tol : float or "auto", default 1e-5
        Total-correlation convergence tolerance of the underlying flow.
    contamination : float, default 0.05
        Expected fraction of outliers; sets the decision threshold at the
        corresponding quantile of the training scores.
    patience : int, default 10
        Non-improving layers tolerated before the flow stops early.
    rotation : str, default "pca"
        Rotation strategy of the underlying flow.
    random_state : int or None, optional
        Seed for the underlying flow.

    Attributes
    ----------
    model_ : AnnealedRBIG
        The fitted density model.
    train_scores_ : np.ndarray of shape (n_samples,)
        Training log-densities.
    offset_ : float
        Decision threshold: ``decision_function = score_samples - offset_``
        with the sklearn convention that non-negative means inlier.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGOutlierDetector
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 2))
    >>> det = RBIGOutlierDetector(n_layers=5, random_state=0).fit(X)
    >>> labels = det.predict(np.vstack([X[:5], [[8.0, 8.0]]]))
    >>> int(labels[-1])
    -1
    """

    def __init__(
        self,
        n_layers: int = 50,
        tol: float | str = 1e-5,
        contamination: float = 0.05,
        patience: int = 10,
        rotation: str = "pca",
        random_state: int | None = None,
    ):
        self.n_layers = n_layers
        self.tol = tol
        self.contamination = contamination
        self.patience = patience
        self.rotation = rotation
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RBIGOutlierDetector:
        """Fit the density model and calibrate the contamination threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (assumed predominantly inliers).
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : RBIGOutlierDetector
            The fitted detector.
        """
        X = validate_data(self, X)
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError(
                f"contamination must be in (0, 0.5], got {self.contamination}."
            )
        if X.shape[1] > 30:
            import warnings

            warnings.warn(
                "Log-likelihood scores concentrate in high dimensions "
                "(d > 30); consider reducing dimensionality first, e.g. "
                "with RBIGReducer.",
                UserWarning,
                stacklevel=2,
            )
        self.model_ = AnnealedRBIG(
            n_layers=self.n_layers,
            tol=self.tol,
            patience=self.patience,
            rotation=self.rotation,
            random_state=self.random_state,
        ).fit(X)
        self.train_scores_ = self.model_.score_samples(X)
        self.offset_ = float(
            np.percentile(self.train_scores_, 100.0 * self.contamination)
        )
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Log-density of each sample under the flow (higher = more normal).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Per-sample ``log p(x)`` in nats.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.model_.score_samples(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Shifted scores: non-negative means inlier (sklearn convention).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        decision : np.ndarray of shape (n_samples,)
            ``score_samples(X) - offset_``.
        """
        return self.score_samples(X) - self.offset_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Label each sample +1 (inlier) or -1 (outlier).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            +1 for inliers, -1 for outliers.
        """
        return np.where(self.decision_function(X) >= 0.0, 1, -1)
