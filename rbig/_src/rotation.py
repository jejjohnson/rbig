"""PCA and random rotation transforms for RBIG."""
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import ortho_group

from rbig._src.base import RotationTransform


def fit_rotation(
    X: np.ndarray,
    rotation_type: str = "PCA",
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, dict]:
    """Fit a rotation matrix to data X.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    rotation_type : {'PCA', 'random'}
    random_state : int or None

    Returns
    -------
    rotation_matrix : np.ndarray, shape (n_features, n_features)
    metadata : dict
    """
    if rotation_type == "PCA":
        rot = PCARotation()
        rot.fit(X)
        return rot.rotation_matrix_, {"type": "PCA", "explained_variance": rot.explained_variance_}
    elif rotation_type == "random":
        rot = RandomRotation(random_state=random_state)
        rot.fit(X)
        return rot.rotation_matrix_, {"type": "random"}
    else:
        raise ValueError(f"Unknown rotation_type: {rotation_type!r}")


def apply_rotation(X: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply rotation: Y = X @ R."""
    return X @ rotation_matrix


def apply_rotation_inverse(Y: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Apply inverse rotation: X = Y @ R.T."""
    return Y @ rotation_matrix.T


class PCARotation(RotationTransform):
    """PCA-based rotation transform."""

    def fit(self, x: np.ndarray) -> "PCARotation":
        pca = PCA(whiten=False)
        pca.fit(x)
        self.rotation_matrix_ = pca.components_.T
        self.explained_variance_ = pca.explained_variance_ratio_
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        return apply_rotation(x, self.rotation_matrix_)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return apply_rotation_inverse(y, self.rotation_matrix_)


class RandomRotation(RotationTransform):
    """Random orthogonal rotation transform."""

    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state

    def fit(self, x: np.ndarray) -> "RandomRotation":
        n_features = x.shape[1]
        self.rotation_matrix_ = ortho_group.rvs(n_features, random_state=self.random_state)
        return self

    def forward(self, x: np.ndarray) -> np.ndarray:
        return apply_rotation(x, self.rotation_matrix_)

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return apply_rotation_inverse(y, self.rotation_matrix_)
