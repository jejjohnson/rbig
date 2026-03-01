"""Image Gaussianization via patch-based RBIG."""
from typing import Optional, Tuple

import numpy as np


def extract_patches_2d(
    image: np.ndarray,
    patch_size: Tuple[int, int],
    stride: int = 1,
) -> np.ndarray:
    """Extract 2-D patches from a grayscale or multi-channel image.

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, C)
    patch_size : (ph, pw)
    stride : int

    Returns
    -------
    patches : np.ndarray, shape (n_patches, ph * pw * C)
    """
    if image.ndim == 2:
        image = image[:, :, None]
    H, W, C = image.shape
    ph, pw = patch_size

    patches = []
    for r in range(0, H - ph + 1, stride):
        for c in range(0, W - pw + 1, stride):
            patch = image[r : r + ph, c : c + pw, :]
            patches.append(patch.ravel())

    return np.stack(patches, axis=0)


def reconstruct_from_patches_2d(
    patches: np.ndarray,
    image_shape: Tuple[int, ...],
    patch_size: Tuple[int, int],
    stride: int = 1,
) -> np.ndarray:
    """Reconstruct an image from patches using averaging.

    Parameters
    ----------
    patches : np.ndarray, shape (n_patches, ph * pw * C)
    image_shape : (H, W) or (H, W, C)
    patch_size : (ph, pw)
    stride : int

    Returns
    -------
    image : np.ndarray, same shape as image_shape
    """
    if len(image_shape) == 2:
        H, W = image_shape
        C = 1
        squeeze = True
    else:
        H, W, C = image_shape
        squeeze = False

    ph, pw = patch_size
    output = np.zeros((H, W, C), dtype=float)
    weight = np.zeros((H, W, C), dtype=float)

    idx = 0
    for r in range(0, H - ph + 1, stride):
        for c in range(0, W - pw + 1, stride):
            patch = patches[idx].reshape(ph, pw, C)
            output[r : r + ph, c : c + pw, :] += patch
            weight[r : r + ph, c : c + pw, :] += 1.0
            idx += 1

    weight = np.where(weight == 0, 1, weight)
    result = output / weight

    if squeeze:
        return result[:, :, 0]
    return result


def image_gaussianize(
    image: np.ndarray,
    patch_size: Tuple[int, int] = (8, 8),
    **rbig_kwargs,
) -> np.ndarray:
    """Gaussianize an image via patch-based RBIG.

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, C)
    patch_size : (ph, pw)
    **rbig_kwargs
        Passed to AnnealedRBIG.

    Returns
    -------
    gauss_image : np.ndarray, same shape as image
    """
    model = ImageRBIG(patch_size=patch_size, **rbig_kwargs)
    return model.fit(image).transform(image)


class ImageRBIG:
    """RBIG for images using patch extraction.

    Parameters
    ----------
    patch_size : tuple of int, default=(8, 8)
    stride : int, default=1
    **rbig_kwargs
        Passed to AnnealedRBIG.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int] = (8, 8),
        stride: int = 1,
        **rbig_kwargs,
    ):
        self.patch_size = patch_size
        self.stride = stride
        self.rbig_kwargs = rbig_kwargs

    def fit(self, image: np.ndarray) -> "ImageRBIG":
        """Fit RBIG on image patches.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)
        """
        from rbig._src.model import AnnealedRBIG

        self._image_shape = image.shape
        patches = extract_patches_2d(image, self.patch_size, stride=self.stride)
        self._model = AnnealedRBIG(**self.rbig_kwargs).fit(patches)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image to Gaussian domain.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        gauss_image : np.ndarray, same shape as image
        """
        patches = extract_patches_2d(image, self.patch_size, stride=self.stride)
        gauss_patches = self._model.transform(patches)
        return reconstruct_from_patches_2d(
            gauss_patches, self._image_shape, self.patch_size, stride=self.stride
        )

    def inverse_transform(self, image: np.ndarray) -> np.ndarray:
        """Inverse transform image from Gaussian domain.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        orig_image : np.ndarray, same shape as image
        """
        patches = extract_patches_2d(image, self.patch_size, stride=self.stride)
        orig_patches = self._model.inverse_transform(patches)
        return reconstruct_from_patches_2d(
            orig_patches, self._image_shape, self.patch_size, stride=self.stride
        )
