import numpy as np
from scipy.stats import ortho_group
from sklearn.decomposition import PCA


def pca_rotation(X, random_state=None, **kwargs):
    """Apply PCA rotation to X. Returns (X_rotated, rotation_matrix)."""
    pca_model = PCA(random_state=random_state, **kwargs)
    X_rotated = pca_model.fit_transform(X)
    return X_rotated, pca_model.components_.T


def random_rotation(X, n_dimensions=None):
    """Apply random orthogonal rotation to X. Returns (X_rotated, rotation_matrix)."""
    if n_dimensions is None:
        n_dimensions = X.shape[1]
    R = ortho_group.rvs(n_dimensions)
    return np.dot(X, R), R


def apply_rotation(X, rotation_matrix):
    """Apply rotation matrix to X."""
    return np.dot(X, rotation_matrix)


def apply_rotation_inverse(X, rotation_matrix):
    """Apply inverse rotation (transpose) to X."""
    return np.dot(X, rotation_matrix.T)
