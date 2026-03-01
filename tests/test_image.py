"""Tests for image utilities."""

import numpy as np
import pytest

from rbig._src.image import extract_patches, matrix_to_patches, patches_to_matrix


def test_extract_patches_2d():
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 1, size=(32, 32))
    patches = extract_patches(image, patch_size=8, step=8)
    assert patches.ndim == 3
    assert patches.shape[1] == 8
    assert patches.shape[2] == 8


def test_extract_patches_3d():
    rng = np.random.default_rng(42)
    image = rng.uniform(0, 1, size=(32, 32, 3))
    patches = extract_patches(image, patch_size=8, step=8)
    assert patches.ndim == 4
    assert patches.shape[1] == 8
    assert patches.shape[2] == 8
    assert patches.shape[3] == 3


def test_patches_to_matrix():
    rng = np.random.default_rng(42)
    patches = rng.uniform(0, 1, size=(10, 8, 8))
    matrix = patches_to_matrix(patches)
    assert matrix.shape == (10, 64)


def test_matrix_to_patches():
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0, 1, size=(10, 64))
    patches = matrix_to_patches(matrix, (8, 8))
    assert patches.shape == (10, 8, 8)


def test_wavelet_transform():
    pytest.importorskip("pywt")
    from rbig._src.image import WaveletTransform

    rng = np.random.default_rng(42)
    # Create a set of simple images (8x8)
    images = rng.uniform(0, 1, size=(10, 8, 8))
    t = WaveletTransform(wavelet="haar", level=1)
    t.fit(images)
    Xt = t.transform(images)
    assert Xt.ndim == 2
    assert Xt.shape[0] == 10
