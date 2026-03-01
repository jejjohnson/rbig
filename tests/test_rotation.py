import numpy as np
import pytest
from rbig._src.rotation import pca_rotation, random_rotation, apply_rotation, apply_rotation_inverse


def test_pca_rotation(sample_2d):
    X_rot, R = pca_rotation(sample_2d)
    assert X_rot.shape == sample_2d.shape
    assert R.shape == (2, 2)


def test_random_rotation(sample_2d):
    X_rot, R = random_rotation(sample_2d, n_dimensions=2)
    assert X_rot.shape == sample_2d.shape


def test_rotation_roundtrip(sample_2d):
    X_rot, R = pca_rotation(sample_2d)
    X_rec = apply_rotation_inverse(X_rot, R)
    # PCA centers data, so we compare against the centered input
    X_centered = sample_2d - sample_2d.mean(axis=0)
    np.testing.assert_allclose(X_rec, X_centered, atol=1e-10)
