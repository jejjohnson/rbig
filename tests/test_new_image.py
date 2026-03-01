"""Tests for new image classes."""

import numpy as np
import pytest

from rbig import (
    DCTRotation,
    HartleyRotation,
    ImageRBIG,
    RandomChannelRotation,
)


@pytest.fixture
def flat_images():
    """Flat (N, C*H*W) image data."""
    rng = np.random.default_rng(42)
    N, C, H, W = 20, 1, 8, 8
    return rng.normal(size=(N, C * H * W)), C, H, W


@pytest.fixture
def rgb_flat_images():
    """Flat RGB (N, C*H*W) image data."""
    rng = np.random.default_rng(42)
    N, C, H, W = 20, 3, 8, 8
    return rng.normal(size=(N, C * H * W)), C, H, W


def test_dct_rotation_shape(flat_images):
    X, C, H, W = flat_images
    r = DCTRotation(C=C, H=H, W=W)
    Xt = r.fit_transform(X)
    assert Xt.shape == X.shape


def test_dct_rotation_inverse(flat_images):
    X, C, H, W = flat_images
    r = DCTRotation(C=C, H=H, W=W)
    r.fit(X)
    Xt = r.transform(X)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, X, atol=1e-10)


def test_dct_rotation_log_det(flat_images):
    X, C, H, W = flat_images
    r = DCTRotation(C=C, H=H, W=W)
    r.fit(X)
    ldj = r.get_log_det_jacobian(X)
    np.testing.assert_allclose(ldj, 0.0)


def test_hartley_rotation_shape(flat_images):
    X, C, H, W = flat_images
    r = HartleyRotation(C=C, H=H, W=W)
    Xt = r.fit_transform(X)
    assert Xt.shape == X.shape


def test_hartley_rotation_log_det(flat_images):
    X, C, H, W = flat_images
    r = HartleyRotation(C=C, H=H, W=W)
    r.fit(X)
    ldj = r.get_log_det_jacobian(X)
    np.testing.assert_allclose(ldj, 0.0)


def test_random_channel_rotation_shape(rgb_flat_images):
    X, C, H, W = rgb_flat_images
    r = RandomChannelRotation(C=C, H=H, W=W, random_state=42)
    Xt = r.fit_transform(X)
    assert Xt.shape == X.shape


def test_random_channel_rotation_inverse(rgb_flat_images):
    X, C, H, W = rgb_flat_images
    r = RandomChannelRotation(C=C, H=H, W=W, random_state=42)
    r.fit(X)
    Xt = r.transform(X)
    Xr = r.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, X, atol=1e-10)


def test_random_channel_rotation_log_det(rgb_flat_images):
    X, C, H, W = rgb_flat_images
    r = RandomChannelRotation(C=C, H=H, W=W, random_state=42)
    r.fit(X)
    ldj = r.get_log_det_jacobian(X)
    np.testing.assert_allclose(ldj, 0.0)


def test_image_rbig_fit_transform(flat_images):
    X, C, H, W = flat_images
    model = ImageRBIG(n_layers=2, C=C, H=H, W=W, strategy="dct", random_state=42)
    model.fit(X)
    Xt = model.transform(X)
    assert Xt.shape == X.shape


def test_image_rbig_inverse(flat_images):
    X, C, H, W = flat_images
    model = ImageRBIG(n_layers=2, C=C, H=H, W=W, strategy="dct", random_state=42)
    model.fit(X)
    Xt = model.transform(X)
    Xr = model.inverse_transform(Xt)
    assert Xr.shape == X.shape


def test_ortho_wavelet_shape():
    pytest.importorskip("pywt")
    from rbig import OrthogonalWaveletLayer

    rng = np.random.default_rng(42)
    N, C, H, W = 10, 1, 8, 8
    X = rng.normal(size=(N, C * H * W))
    layer = OrthogonalWaveletLayer(C=C, H=H, W=W)
    Xt = layer.fit_transform(X)
    # Output should be (N, 4C * H/2 * W/2)
    assert Xt.shape[0] == N
    assert Xt.shape[1] == 4 * C * (H // 2) * (W // 2)


def test_ortho_wavelet_inverse():
    pytest.importorskip("pywt")
    from rbig import OrthogonalWaveletLayer

    rng = np.random.default_rng(42)
    N, C, H, W = 10, 1, 8, 8
    X = rng.normal(size=(N, C * H * W))
    layer = OrthogonalWaveletLayer(C=C, H=H, W=W)
    layer.fit(X)
    Xt = layer.transform(X)
    Xr = layer.inverse_transform(Xt)
    np.testing.assert_allclose(Xr, X, atol=1e-10)
