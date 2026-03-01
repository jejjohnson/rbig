"""Rotation transforms for RBIG."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from rbig._src.base import BaseTransform


class PCARotation(BaseTransform):
    """PCA-based rotation for whitening."""

    def __init__(self, n_components: int | None = None, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: np.ndarray) -> PCARotation:
        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca_.inverse_transform(X)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute Jacobian determinant (constant for linear transforms)."""
        if self.whiten:
            log_det = -0.5 * np.sum(np.log(self.pca_.explained_variance_))
        else:
            log_det = 0.0
        return np.full(X.shape[0], log_det)


class ICARotation(BaseTransform):
    """ICA-based rotation using Picard algorithm."""

    def __init__(
        self, n_components: int | None = None, random_state: int | None = None
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> ICARotation:
        try:
            from picard import picard

            n = X.shape[1] if self.n_components is None else self.n_components
            K, W, _ = picard(
                X.T,
                n_components=n,
                random_state=self.random_state,
                max_iter=500,
                tol=1e-5,
            )
            self.K_ = K
            self.W_ = W
            self.n_features_in_ = X.shape[1]
        except ImportError:
            from sklearn.decomposition import FastICA

            self.ica_ = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=500,
            )
            self.ica_.fit(X)
            self.K_ = None
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.K_ is None:
            return self.ica_.transform(X)
        Xw = X @ self.K_.T
        return Xw @ self.W_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.K_ is None:
            return self.ica_.inverse_transform(X)
        Xw = X @ np.linalg.pinv(self.W_).T
        return Xw @ np.linalg.pinv(self.K_).T

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute Jacobian determinant (constant for linear transforms)."""
        if self.K_ is None:
            W = self.ica_.components_
            log_det = np.log(np.abs(np.linalg.det(W)))
        else:
            WK = self.W_ @ self.K_
            log_det = np.log(np.abs(np.linalg.det(WK)))
        return np.full(X.shape[0], log_det)


from rbig._src.base import RotationBijector


class RandomRotation(RotationBijector):
    """Random orthogonal rotation via QR decomposition (Haar measure)."""

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> RandomRotation:
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        A = rng.standard_normal((n_features, n_features))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix_ = Q
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.rotation_matrix_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.rotation_matrix_  # Q^{-1} = Q^T


class RandomOrthogonalProjection(RotationBijector):
    """Semi-orthogonal projection from D to K dimensions via QR."""

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> RandomOrthogonalProjection:
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        A = rng.standard_normal((D, K))
        Q, _ = np.linalg.qr(A)
        self.projection_matrix_ = Q[:, :K]  # (D, K)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.projection_matrix_  # (N, K)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.projection_matrix_.T  # approximate inverse (N, D)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class GaussianRandomProjection(RotationBijector):
    """Johnson-Lindenstrauss style random projection."""

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> GaussianRandomProjection:
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        self.matrix_ = rng.standard_normal((D, K)) / np.sqrt(K)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X @ self.matrix_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X @ np.linalg.pinv(self.matrix_).T

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class OrthogonalDimensionalityReduction(RotationBijector):
    """Full rotation + dimension splitting with rejection likelihood."""

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> OrthogonalDimensionalityReduction:
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix_ = Q  # (D, D)
        self.n_components_ = K
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xr = X @ self.rotation_matrix_.T  # (N, D)
        return Xr[:, : self.n_components_]  # (N, K)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        D = self.rotation_matrix_.shape[0]
        N = X.shape[0]
        Xfull = np.zeros((N, D))
        Xfull[:, : self.n_components_] = X
        return Xfull @ self.rotation_matrix_  # approximate

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class PicardRotation(RotationBijector):
    """ICA rotation via the Picard algorithm."""

    def __init__(
        self,
        n_components: int | None = None,
        extended: bool = False,
        random_state: int | None = None,
        max_iter: int = 500,
        tol: float = 1e-5,
    ):
        self.n_components = n_components
        self.extended = extended
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray) -> PicardRotation:
        try:
            from picard import picard

            n = X.shape[1] if self.n_components is None else self.n_components
            K, W, _ = picard(
                X.T,
                n_components=n,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
                extended=self.extended,
            )
            self.K_ = K
            self.W_ = W
            self.use_picard_ = True
        except (ImportError, TypeError):
            from sklearn.decomposition import FastICA

            self.ica_ = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
            self.ica_.fit(X)
            self.K_ = None
            self.use_picard_ = False
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.use_picard_:
            return self.ica_.transform(X)
        return (X @ self.K_.T) @ self.W_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if not self.use_picard_:
            return self.ica_.inverse_transform(X)
        return (X @ np.linalg.pinv(self.W_).T) @ np.linalg.pinv(self.K_).T

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        if not self.use_picard_:
            W = self.ica_.components_
            log_det = np.log(np.abs(np.linalg.det(W)))
        else:
            WK = self.W_ @ self.K_
            log_det = np.log(np.abs(np.linalg.det(WK)))
        return np.full(X.shape[0], log_det)
