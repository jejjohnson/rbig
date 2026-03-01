"""Test image Gaussianization."""
import numpy as np
import pytest

from rbig import (
    ImageRBIG,
    extract_patches_2d,
    image_gaussianize,
    reconstruct_from_patches_2d,
)


@pytest.fixture
def small_image(rng):
    return rng.standard_normal((16, 16)).astype(float)


def test_extract_patches_shape(small_image):
    patches = extract_patches_2d(small_image, patch_size=(4, 4), stride=4)
    # 16/4 * 16/4 = 16 patches
    assert patches.shape == (16, 16)


def test_reconstruct_from_patches(small_image):
    patch_size = (4, 4)
    stride = 4
    patches = extract_patches_2d(small_image, patch_size=patch_size, stride=stride)
    rec = reconstruct_from_patches_2d(patches, small_image.shape, patch_size, stride=stride)
    np.testing.assert_allclose(rec, small_image, atol=1e-10)


def test_image_rbig_fit_transform(small_image):
    model = ImageRBIG(patch_size=(4, 4), n_layers=30, zero_tolerance=10)
    model.fit(small_image)
    gauss = model.transform(small_image)
    assert gauss.shape == small_image.shape
    assert np.all(np.isfinite(gauss))


def test_image_rbig_inverse(small_image):
    model = ImageRBIG(patch_size=(4, 4), n_layers=30, zero_tolerance=10)
    model.fit(small_image)
    gauss = model.transform(small_image)
    rec = model.inverse_transform(gauss)
    assert rec.shape == small_image.shape


def test_image_gaussianize(small_image):
    result = image_gaussianize(small_image, patch_size=(4, 4), n_layers=20, zero_tolerance=10)
    assert result.shape == small_image.shape
    assert np.all(np.isfinite(result))
