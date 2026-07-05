"""Gaussianize-then-cluster (issue #127).

:class:`RBIGKMeans` runs k-means (or a Gaussian mixture) in the RBIG
latent space, where Euclidean geometry is calibrated: elongated,
unequal-variance, and curved clusters become round-ish Gaussian blobs.
Centroids are mapped back to data space through the exact inverse flow
for interpretability.

The layer budget deliberately defaults to a *small* ``n_layers_rbig=10``:
multi-modality **is** non-Gaussianity, so a fully converged flow absorbs
the between-cluster structure itself (each marginal Gaussianization step
compresses the gaps between modes).  A shallow flow calibrates the
within-cluster geometry while leaving enough between-cluster separation
to cluster on — the over-Gaussianization regression test pins this
trade-off.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.model import AnnealedRBIG

_METHODS = ("kmeans", "gmm")


class RBIGKMeans(
    ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator
):
    """K-means / Gaussian-mixture clustering in the RBIG latent space.

    Fits an :class:`~rbig._src.model.AnnealedRBIG` flow, clusters the
    Gaussianized coordinates ``Z = f(X)``, and exposes the centroids in
    both spaces (``centroids_z_`` and ``centroids_x_ = f⁻¹(centroids_z_)``).

    Parameters
    ----------
    n_clusters : int, default 3
        Number of clusters.
    n_layers_rbig : int, default 10
        Layer budget of the flow.  Deliberately small — see the module
        docstring; deeper flows Gaussianize away the mode structure the
        clusterer needs.
    tol_rbig : float or "auto", default 1e-5
        Convergence tolerance of the flow.
    method : {"kmeans", "gmm"}, default "kmeans"
        Z-space clusterer: :class:`~sklearn.cluster.KMeans` or
        :class:`~sklearn.mixture.GaussianMixture` (full covariances).
    n_init : int, default 10
        Restarts of the inner clusterer.
    random_state : int or None, optional
        Seed shared by the flow and the inner clusterer.

    Attributes
    ----------
    model_ : AnnealedRBIG
        The fitted flow.
    inner_ : KMeans or GaussianMixture
        The fitted Z-space clusterer.
    labels_ : np.ndarray of shape (n_samples,)
        Training cluster assignments.
    centroids_z_ : np.ndarray of shape (n_clusters, n_features)
        Cluster centers in the Gaussianized space.
    centroids_x_ : np.ndarray of shape (n_clusters, n_features)
        Cluster centers mapped back to data space (interpretable).

    Notes
    -----
    Model selection (choosing ``n_clusters``) should use silhouette
    scores on the **Z-space** coordinates (``transform(X)``), where the
    Euclidean metric the silhouette assumes is calibrated — not on raw X.

    For ``method="kmeans"``, ``predict_proba`` uses a shared-variance
    isotropic Gaussian per cluster with uniform weights, so its argmax
    coincides exactly with the nearest-centroid ``predict``.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig import RBIGKMeans
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack(
    ...     [
    ...         rng.standard_normal((100, 2)) * [3.0, 0.3] + [0, +2],
    ...         rng.standard_normal((100, 2)) * [3.0, 0.3] + [0, -2],
    ...     ]
    ... )
    >>> km = RBIGKMeans(n_clusters=2, n_layers_rbig=3, random_state=0).fit(X)
    >>> km.labels_.shape, km.centroids_x_.shape
    ((200,), (2, 2))
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_layers_rbig: int = 10,
        tol_rbig: float | str = 1e-5,
        method: str = "kmeans",
        n_init: int = 10,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.n_layers_rbig = n_layers_rbig
        self.tol_rbig = tol_rbig
        self.method = method
        self.n_init = n_init
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RBIGKMeans:
        """Fit the flow, cluster in Z-space, and map centroids back.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        self : RBIGKMeans
            The fitted clusterer.
        """
        X = validate_data(self, X)
        if self.method not in _METHODS:
            raise ValueError(f"Unknown method {self.method!r}; choose from {_METHODS}.")
        if X.shape[0] < self.n_clusters:
            raise ValueError(
                f"n_samples={X.shape[0]} should be >= n_clusters={self.n_clusters}."
            )
        self.model_ = AnnealedRBIG(
            n_layers=self.n_layers_rbig,
            tol=self.tol_rbig,
            random_state=self.random_state,
        ).fit(X)
        Z = self.model_.transform(X)  # Z: (n, d), calibrated geometry

        if self.method == "kmeans":
            from sklearn.cluster import KMeans

            self.inner_ = KMeans(
                n_clusters=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(Z)
            self.labels_ = self.inner_.labels_
            self.centroids_z_ = self.inner_.cluster_centers_
            # Shared isotropic variance for the probabilistic view; the
            # shared scale + uniform weights keep argmax(predict_proba)
            # identical to the nearest-centroid predict.
            sq_dist = ((Z - self.centroids_z_[self.labels_]) ** 2).sum(axis=1)
            self._sigma2 = float(max(sq_dist.mean() / Z.shape[1], 1e-12))
        else:  # gmm
            from sklearn.mixture import GaussianMixture

            self.inner_ = GaussianMixture(
                n_components=self.n_clusters,
                n_init=self.n_init,
                random_state=self.random_state,
            ).fit(Z)
            self.labels_ = self.inner_.predict(Z)
            self.centroids_z_ = self.inner_.means_

        self.centroids_x_ = self.model_.inverse_transform(self.centroids_z_)
        self._n_features_out = X.shape[1]  # transform outputs Z, same width
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Gaussianized coordinates ``Z = f(X)`` (for plotting/silhouette).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Latent coordinates.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.model_.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each sample to its Z-space cluster.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster indices in ``[0, n_clusters)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return self.inner_.predict(self.model_.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Per-cluster membership probabilities in Z-space.

        For ``method="gmm"`` these are the mixture responsibilities; for
        ``method="kmeans"`` a shared-variance isotropic Gaussian is
        placed on each centroid (uniform weights), so the argmax matches
        ``predict`` exactly.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_clusters)
            Rows sum to 1.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        Z = self.model_.transform(X)
        if self.method == "gmm":
            return self.inner_.predict_proba(Z)
        # log N(z; c_k, sigma^2 I) up to a shared constant: -||z-c_k||^2/(2s2)
        sq = ((Z[:, None, :] - self.centroids_z_[None, :, :]) ** 2).sum(axis=2)
        logits = -0.5 * sq / self._sigma2  # (n, k)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Z-space log-likelihood of each sample under its assigned cluster.

        The value is ``log π_k + log N(z; μ_k, Σ_k)`` for the assigned
        component ``k`` — a *latent-space* quantity for ranking samples
        within/between clusters, not a data-space density (use
        ``AnnealedRBIG.score_samples`` for that).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Query points.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Assigned-component log-likelihoods in nats.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        Z = self.model_.transform(X)
        d = Z.shape[1]
        if self.method == "gmm":
            from scipy.stats import multivariate_normal

            labels = self.inner_.predict(Z)
            log_w = np.log(self.inner_.weights_)
            out = np.empty(Z.shape[0])
            for k in range(self.n_clusters):
                mask = labels == k
                if not mask.any():
                    continue
                out[mask] = log_w[k] + multivariate_normal.logpdf(
                    Z[mask], mean=self.inner_.means_[k], cov=self.inner_.covariances_[k]
                )
            return out
        labels = self.inner_.predict(Z)
        sq = ((Z - self.centroids_z_[labels]) ** 2).sum(axis=1)
        log_norm = -0.5 * d * np.log(2.0 * np.pi * self._sigma2)
        return np.log(1.0 / self.n_clusters) + log_norm - 0.5 * sq / self._sigma2
