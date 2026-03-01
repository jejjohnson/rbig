"""Image-related transforms for RBIG."""

from __future__ import annotations

import numpy as np

from rbig._src.base import BaseTransform


def patches_to_matrix(
    patches: np.ndarray,
) -> np.ndarray:
    """Convert image patches to 2D matrix.

    Parameters
    ----------
    patches : array of shape (n_patches, patch_h, patch_w) or (n_patches, patch_h, patch_w, channels)

    Returns
    -------
    matrix : array of shape (n_patches, n_features)
    """
    return patches.reshape(patches.shape[0], -1)


def matrix_to_patches(
    matrix: np.ndarray,
    patch_shape: tuple[int, ...],
) -> np.ndarray:
    """Convert 2D matrix back to image patches.

    Parameters
    ----------
    matrix : array of shape (n_patches, n_features)
    patch_shape : shape of each patch (h, w) or (h, w, c)

    Returns
    -------
    patches : array of shape (n_patches, *patch_shape)
    """
    return matrix.reshape(matrix.shape[0], *patch_shape)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 8,
    step: int = 1,
) -> np.ndarray:
    """Extract patches from a 2D image.

    Parameters
    ----------
    image : array of shape (H, W) or (H, W, C)
    patch_size : size of square patches
    step : step between patches

    Returns
    -------
    patches : array of shape (n_patches, patch_size, patch_size) or (n_patches, patch_size, patch_size, C)
    """
    if image.ndim == 2:
        H, W = image.shape
        patches = []
        for i in range(0, H - patch_size + 1, step):
            for j in range(0, W - patch_size + 1, step):
                patches.append(image[i : i + patch_size, j : j + patch_size])
        return np.array(patches)
    elif image.ndim == 3:
        H, W, _C = image.shape
        patches = []
        for i in range(0, H - patch_size + 1, step):
            for j in range(0, W - patch_size + 1, step):
                patches.append(image[i : i + patch_size, j : j + patch_size, :])
        return np.array(patches)
    else:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")


class WaveletTransform(BaseTransform):
    """Wavelet transform for image compression/decomposition.

    Requires PyWavelets (pywt).
    """

    def __init__(
        self, wavelet: str = "haar", level: int = 1, mode: str = "periodization"
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def fit(self, X: np.ndarray) -> WaveletTransform:
        import pywt

        self.pywt_ = pywt
        self.original_shape_ = X.shape
        test = X[0]
        coeffs = pywt.wavedec2(test, self.wavelet, level=self.level, mode=self.mode)
        self.coeff_slices_ = None
        arr, self.coeff_slices_ = pywt.coeffs_to_array(coeffs)
        self.coeff_shape_ = arr.shape
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import pywt

        result = []
        for xi in X:
            coeffs = pywt.wavedec2(xi, self.wavelet, level=self.level, mode=self.mode)
            arr, _ = pywt.coeffs_to_array(coeffs)
            result.append(arr.ravel())
        return np.array(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        import pywt

        result = []
        for xi in X:
            arr = xi.reshape(self.coeff_shape_)
            coeffs = pywt.array_to_coeffs(
                arr, self.coeff_slices_, output_format="wavedec2"
            )
            img = pywt.waverec2(coeffs, self.wavelet, mode=self.mode)
            result.append(img)
        return np.array(result)
