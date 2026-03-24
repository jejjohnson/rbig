"""Image-related transforms for RBIG.

This module provides patch extraction utilities and bijective image transforms
(wavelet, Hartley, DCT, random channel rotation) that serve as the rotation
step inside RBIG applied to image data.  All bijectors follow the convention
``(N, C·H·W)`` for the flattened representation and ``(N, C, H, W)`` for the
tensor representation internally.
"""

from __future__ import annotations

import numpy as np

from rbig._src.base import BaseTransform


def patches_to_matrix(
    patches: np.ndarray,
) -> np.ndarray:
    """Flatten image patches into a 2-D feature matrix.

    Each patch is ravelled along its spatial (and optionally channel) axes so
    that a collection of patches becomes a conventional ``(samples, features)``
    matrix suitable for statistical modelling.

    Parameters
    ----------
    patches : np.ndarray, shape ``(n_patches, h, w)`` or \
``(n_patches, h, w, C)``
        Stack of image patches.  The first axis indexes patches; all remaining
        axes are flattened into the feature axis.

    Returns
    -------
    matrix : np.ndarray, shape ``(n_patches, h*w)`` or \
``(n_patches, h*w*C)``
        Row-major flattening of every patch:
        ``n_features = h * w`` (grayscale) or ``h * w * C`` (colour).

    Notes
    -----
    The reshape is equivalent to

    .. math::

        \\mathbf{M}[i, :] = \\operatorname{vec}(\\mathbf{P}[i])

    where :math:`\\operatorname{vec}` stacks columns in row-major (C) order.

    Examples
    --------
    >>> import numpy as np
    >>> patches = np.random.default_rng(0).standard_normal((100, 8, 8))
    >>> mat = patches_to_matrix(patches)
    >>> mat.shape
    (100, 64)
    """
    # Collapse spatial (and channel) dims: (n_patches, h, w[, C]) → (n_patches, h*w[*C])
    return patches.reshape(patches.shape[0], -1)


def matrix_to_patches(
    matrix: np.ndarray,
    patch_shape: tuple[int, ...],
) -> np.ndarray:
    """Reshape a 2-D feature matrix back into a stack of image patches.

    This is the inverse of :func:`patches_to_matrix`.

    Parameters
    ----------
    matrix : np.ndarray, shape ``(n_patches, n_features)``
        Flattened patch matrix where ``n_features = prod(patch_shape)``.
    patch_shape : tuple of int
        Target shape for each individual patch, e.g. ``(h, w)`` or
        ``(h, w, C)``.  The product of all elements must equal
        ``matrix.shape[1]``.

    Returns
    -------
    patches : np.ndarray, shape ``(n_patches, *patch_shape)``
        Stack of patches with the original spatial structure restored.

    Notes
    -----
    The reshape satisfies

    .. math::

        \\mathbf{P}[i] = \\operatorname{unvec}(\\mathbf{M}[i, :],\\, h, w)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> mat = rng.standard_normal((100, 64))
    >>> patches = matrix_to_patches(mat, patch_shape=(8, 8))
    >>> patches.shape
    (100, 8, 8)
    """
    # Restore spatial structure: (n_patches, n_features) → (n_patches, *patch_shape)
    return matrix.reshape(matrix.shape[0], *patch_shape)


def extract_patches(
    image: np.ndarray,
    patch_size: int = 8,
    step: int = 1,
) -> np.ndarray:
    """Extract overlapping square patches from a 2-D or 3-D image.

    Uses a sliding-window approach with configurable stride.  For a grayscale
    image of shape ``(H, W)`` the number of patches along each axis is
    ``⌊(H - patch_size) / step⌋ + 1``.

    Parameters
    ----------
    image : np.ndarray, shape ``(H, W)`` or ``(H, W, C)``
        Input image.  Grayscale images are 2-D; colour/multi-band images carry
        a trailing channel axis ``C``.
    patch_size : int, default 8
        Side length (in pixels) of each square patch.  Both height and width
        of every extracted patch equal ``patch_size``.
    step : int, default 1
        Stride between consecutive patch origins along each spatial axis.
        ``step=1`` gives maximally overlapping patches; ``step=patch_size``
        gives non-overlapping (tiling) patches.

    Returns
    -------
    patches : np.ndarray
        - shape ``(n_patches, patch_size, patch_size)`` for 2-D input
        - shape ``(n_patches, patch_size, patch_size, C)`` for 3-D input

        where ``n_patches = n_row_patches * n_col_patches`` and
        ``n_row_patches = ⌊(H - patch_size) / step⌋ + 1``.

    Raises
    ------
    ValueError
        If ``image.ndim`` is not 2 or 3.

    Notes
    -----
    The patch at row ``i``, column ``j`` covers pixels

    .. math::

        \\text{patch}_{i,j} = \\text{image}[i \\cdot s : i \\cdot s + p,\\;
                                              j \\cdot s : j \\cdot s + p]

    where :math:`s` = ``step`` and :math:`p` = ``patch_size``.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.default_rng(0).standard_normal((32, 32))
    >>> patches = extract_patches(img, patch_size=8, step=8)
    >>> patches.shape  # 4×4 non-overlapping grid
    (16, 8, 8)

    >>> img_rgb = np.random.default_rng(1).standard_normal((32, 32, 3))
    >>> p = extract_patches(img_rgb, patch_size=8, step=8)
    >>> p.shape
    (16, 8, 8, 3)
    """
    if image.ndim == 2:
        H, W = image.shape
        patches = []
        for i in range(0, H - patch_size + 1, step):
            for j in range(0, W - patch_size + 1, step):
                # Slice a (patch_size, patch_size) window from the image
                patches.append(image[i : i + patch_size, j : j + patch_size])
        return np.array(patches)
    elif image.ndim == 3:
        H, W, _C = image.shape
        patches = []
        for i in range(0, H - patch_size + 1, step):
            for j in range(0, W - patch_size + 1, step):
                # Slice a (patch_size, patch_size, C) window, keeping all channels
                patches.append(image[i : i + patch_size, j : j + patch_size, :])
        return np.array(patches)
    else:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")


class WaveletTransform(BaseTransform):
    """Multi-level 2-D wavelet decomposition for image data.

    Wraps PyWavelets ``wavedec2`` / ``waverec2`` to provide the standard
    ``fit`` / ``transform`` / ``inverse_transform`` interface expected by
    RBIG pipeline components.

    The forward transform maps each ``(H, W)`` image to a flat coefficient
    vector of length ``H * W`` (for periodization boundary mode the coefficient
    array has the same number of elements as the input image).

    Requires PyWavelets (``pip install PyWavelets``).

    Parameters
    ----------
    wavelet : str, default ``"haar"``
        Wavelet name accepted by :func:`pywt.Wavelet`, e.g. ``"haar"``,
        ``"db2"``, ``"sym4"``.
    level : int, default 1
        Decomposition depth.  Higher levels produce coarser approximation
        sub-bands.
    mode : str, default ``"periodization"``
        Signal extension mode passed to PyWavelets.  ``"periodization"``
        ensures the output coefficient array has the same total size as the
        input.

    Attributes
    ----------
    original_shape_ : tuple of int
        Shape ``(N, H, W)`` of the training data passed to :meth:`fit`.
    coeff_slices_ : list
        PyWavelets slicing metadata needed to pack/unpack the coefficient
        array.  Set during :meth:`fit`.
    coeff_shape_ : tuple of int
        Shape of the 2-D coefficient array produced by
        :func:`pywt.coeffs_to_array`.

    Notes
    -----
    The mapping from image to coefficients is

    .. math::

        (N,\\, H \\cdot W) \\xrightarrow{\\text{wavedec2}}
        (N,\\, H \\cdot W)

    For ``level=1`` and ``wavelet="haar"`` the four sub-bands are:
    approximation (LL), horizontal detail (LH), vertical detail (HL), and
    diagonal detail (HH).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 8, 8))  # 50 grayscale 8×8 images
    >>> wt = WaveletTransform(wavelet="haar", level=1)
    >>> wt.fit(X)  # doctest: +ELLIPSIS
    WaveletTransform(...)
    >>> Xt = wt.transform(X)
    >>> Xt.shape
    (50, 64)
    >>> Xr = wt.inverse_transform(Xt)
    >>> Xr.shape
    (50, 8, 8)
    """

    def __init__(
        self, wavelet: str = "haar", level: int = 1, mode: str = "periodization"
    ):
        self.wavelet = wavelet
        self.level = level
        self.mode = mode

    def fit(self, X: np.ndarray, y=None) -> WaveletTransform:
        """Compute and store coefficient layout from the first sample.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, H, W)``
            Training images.  Only the first sample is used to determine the
            coefficient array shape; the data values are not retained.

        Returns
        -------
        self : WaveletTransform
        """
        import pywt

        self.pywt_ = pywt
        self.original_shape_ = X.shape  # store (N, H, W) for reference
        test = X[0]  # single image of shape (H, W)
        coeffs = pywt.wavedec2(test, self.wavelet, level=self.level, mode=self.mode)
        self.coeff_slices_ = None
        # coeffs_to_array packs all sub-bands into one 2-D array
        arr, self.coeff_slices_ = pywt.coeffs_to_array(coeffs)
        self.coeff_shape_ = arr.shape  # e.g. (H, W) for periodization
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Decompose images into flattened wavelet coefficients.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, H, W)``
            Images to transform.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, H*W)``
            Flattened coefficient vectors, one row per image.
        """
        import pywt

        result = []
        for xi in X:  # xi shape: (H, W)
            coeffs = pywt.wavedec2(xi, self.wavelet, level=self.level, mode=self.mode)
            arr, _ = pywt.coeffs_to_array(coeffs)  # arr shape: coeff_shape_
            result.append(arr.ravel())  # flatten to 1-D coefficient vector
        return np.array(result)  # (N, H*W)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct images from flattened wavelet coefficients.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, H*W)``
            Flattened coefficient vectors produced by :meth:`transform`.

        Returns
        -------
        Xr : np.ndarray, shape ``(N, H, W)``
            Reconstructed images.
        """
        import pywt

        result = []
        for xi in X:  # xi shape: (H*W,)
            arr = xi.reshape(self.coeff_shape_)  # restore 2-D coefficient array
            coeffs = pywt.array_to_coeffs(
                arr, self.coeff_slices_, output_format="wavedec2"
            )
            img = pywt.waverec2(coeffs, self.wavelet, mode=self.mode)  # (H, W)
            result.append(img)
        return np.array(result)  # (N, H, W)


from rbig._src.base import Bijector


class ImageBijector(Bijector):
    """Abstract base class for bijective image transforms.

    Manages the conversion between the flattened representation
    ``(N, C·H·W)`` expected by RBIG and the 4-D tensor ``(N, C, H, W)``
    used internally by spatial transforms.

    Subclasses must implement :meth:`fit`, :meth:`transform`, and
    :meth:`inverse_transform`.  The default :meth:`get_log_det_jacobian`
    returns zeros, which is correct for all orthonormal transforms defined
    in this module (``|det J| = 1``).

    Attributes
    ----------
    C_ : int
        Number of channels (set during :meth:`fit`).
    H_ : int
        Image height in pixels (set during :meth:`fit`).
    W_ : int
        Image width in pixels (set during :meth:`fit`).

    Notes
    -----
    The two helper methods implement:

    .. math::

        \\text{_to_tensor}: (N, C \\cdot H \\cdot W)
            \\longrightarrow (N, C, H, W)

        \\text{_to_flat}: (N, C, H, W)
            \\longrightarrow (N, C \\cdot H \\cdot W)
    """

    def _to_tensor(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat ``(N, C*H*W)`` array to tensor ``(N, C, H, W)``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Flattened image batch.

        Returns
        -------
        tensor : np.ndarray, shape ``(N, C, H, W)``
        """
        N = X.shape[0]
        C, H, W = self.C_, self.H_, self.W_
        return X.reshape(N, C, H, W)

    def _to_flat(self, X: np.ndarray) -> np.ndarray:
        """Reshape tensor ``(N, C, H, W)`` to flat ``(N, C*H*W)``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C, H, W)``
            Image tensor batch.

        Returns
        -------
        flat : np.ndarray, shape ``(N, C*H*W)``
        """
        N = X.shape[0]
        return X.reshape(N, -1)

    def fit(self, X: np.ndarray, y=None) -> ImageBijector:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample log |det J| = 0 (orthonormal transform).

        Parameters
        ----------
        X : np.ndarray, shape ``(N, D)``

        Returns
        -------
        log_det : np.ndarray, shape ``(N,)``
            All-zero array because the Jacobian determinant is ±1 for every
            orthonormal linear map.
        """
        return np.zeros(X.shape[0])


class OrthogonalWaveletLayer(ImageBijector):
    """Single-level 2-D DWT "squeeze" bijector.

    Applies a single-level 2-D discrete wavelet transform to each spatial
    channel, splitting each ``(H, W)`` feature map into four ``(H/2, W/2)``
    sub-bands and concatenating them along the channel axis.

    The forward transform is

    .. math::

        (N,\\, C,\\, H,\\, W)
        \\xrightarrow{\\text{DWT}}
        (N,\\, 4C,\\, H/2,\\, W/2)

    Sub-band channel ordering (repeated ``C`` times):

    * ``4c + 0`` — LL (approximation, cA)
    * ``4c + 1`` — LH (horizontal detail, cH)
    * ``4c + 2`` — HL (vertical detail, cV)
    * ``4c + 3`` — HH (diagonal detail, cD)

    The Jacobian determinant is 1 for orthonormal wavelets (e.g. Haar),
    so ``log|det J| = 0``.

    Requires PyWavelets (``pip install PyWavelets``).

    Parameters
    ----------
    wavelet : str, default ``"haar"``
        Wavelet name accepted by :func:`pywt.dwt2`.
    C : int, default 1
        Number of input channels.
    H : int, default 8
        Input image height.  Must be even.
    W : int, default 8
        Input image width.  Must be even.

    Attributes
    ----------
    C_ : int
        Fitted number of channels.
    H_ : int
        Fitted image height.
    W_ : int
        Fitted image width.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 1 * 8 * 8))  # N=10, C=1, H=8, W=8
    >>> layer = OrthogonalWaveletLayer(wavelet="haar", C=1, H=8, W=8)
    >>> layer.fit(X)  # doctest: +ELLIPSIS
    OrthogonalWaveletLayer(...)
    >>> Xt = layer.transform(X)
    >>> Xt.shape  # N=10, 4*C=4, H/2=4, W/2=4 → flat 4*4*4=64
    (10, 64)
    >>> Xr = layer.inverse_transform(Xt)
    >>> Xr.shape
    (10, 64)
    """

    def __init__(self, wavelet: str = "haar", C: int = 1, H: int = 8, W: int = 8):
        self.wavelet = wavelet
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray, y=None) -> OrthogonalWaveletLayer:
        """Store spatial dimensions; no data-dependent fitting required.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``

        Returns
        -------
        self : OrthogonalWaveletLayer
        """
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply single-level 2-D DWT and flatten output.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Flattened image batch.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, 4*C * H//2 * W//2)``
            Flattened wavelet coefficient batch.
        """
        import pywt

        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = []
        for n in range(N):
            channels = []
            for c in range(self.C_):
                # 2-D DWT on single (H, W) channel
                coeffs = pywt.dwt2(imgs[n, c], self.wavelet)
                cA, (cH, cV, cD) = coeffs  # each sub-band: (H/2, W/2)
                # Stack sub-bands: LL, LH, HL, HH
                channels.extend([cA, cH, cV, cD])
            result.append(np.stack(channels, axis=0))  # (4C, H/2, W/2)
        out = np.array(result)  # (N, 4C, H/2, W/2)
        return out.reshape(N, -1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct images via single-level 2-D inverse DWT.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, 4*C * H//2 * W//2)``
            Flattened coefficient batch from :meth:`transform`.

        Returns
        -------
        Xr : np.ndarray, shape ``(N, C*H*W)``
            Reconstructed flattened image batch.
        """
        import pywt

        N = X.shape[0]
        H2, W2 = self.H_ // 2, self.W_ // 2
        # Unpack into (N, 4C, H/2, W/2) tensor
        imgs = X.reshape(N, 4 * self.C_, H2, W2)
        result = []
        for n in range(N):
            channels = []
            for c in range(self.C_):
                # Extract the four sub-bands for channel c
                cA = imgs[n, 4 * c]  # LL approximation
                cH = imgs[n, 4 * c + 1]  # LH horizontal detail
                cV = imgs[n, 4 * c + 2]  # HL vertical detail
                cD = imgs[n, 4 * c + 3]  # HH diagonal detail
                reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
                channels.append(reconstructed)
            result.append(np.stack(channels, axis=0))
        out = np.array(result)  # (N, C, H, W)
        return out.reshape(N, -1)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros: orthonormal DWT has ``|det J| = 1``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, D)``

        Returns
        -------
        log_det : np.ndarray, shape ``(N,)``
        """
        return np.zeros(X.shape[0])


class HartleyRotation(ImageBijector):
    """Discrete Hartley Transform — real-to-real orthonormal rotation.

    The 2-D Discrete Hartley Transform (DHT) is defined as

    .. math::

        H(\\mathbf{x}) = \\operatorname{Re}\\bigl(\\text{FFT}(\\mathbf{x})\\bigr)
                        - \\operatorname{Im}\\bigl(\\text{FFT}(\\mathbf{x})\\bigr)

    and is normalised by ``1/√(H·W)`` to make it orthonormal (unitary).
    Because the DHT is its own inverse (self-inverse), the same operation is
    applied in both :meth:`transform` and :meth:`inverse_transform`.

    Since the transform is orthonormal ``log|det J| = 0`` for all inputs.

    Parameters
    ----------
    C : int, default 1
        Number of image channels.
    H : int, default 8
        Image height in pixels.
    W : int, default 8
        Image width in pixels.

    Attributes
    ----------
    C_ : int
        Fitted number of channels.
    H_ : int
        Fitted image height.
    W_ : int
        Fitted image width.

    Notes
    -----
    The normalised DHT satisfies

    .. math::

        H(H(\\mathbf{x})) = \\mathbf{x}

    making it a self-inverse bijection.  The scaling factor is
    :math:`1 / \\sqrt{H \\cdot W}`.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((5, 64))  # N=5, C=1, H=8, W=8
    >>> layer = HartleyRotation(C=1, H=8, W=8)
    >>> layer.fit(X)  # doctest: +ELLIPSIS
    HartleyRotation(...)
    >>> Xt = layer.transform(X)
    >>> Xr = layer.inverse_transform(Xt)
    >>> np.allclose(X, Xr, atol=1e-10)
    True
    """

    def __init__(self, C: int = 1, H: int = 8, W: int = 8):
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray, y=None) -> HartleyRotation:
        """Store spatial dimensions; no data-dependent fitting required.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``

        Returns
        -------
        self : HartleyRotation
        """
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def _dht2(self, x: np.ndarray) -> np.ndarray:
        """Compute un-normalised 2-D Discrete Hartley Transform.

        .. math::

            H_{m,n} = \\operatorname{Re}(F_{m,n}) - \\operatorname{Im}(F_{m,n})

        where :math:`F = \\text{FFT2}(x)`.

        Parameters
        ----------
        x : np.ndarray, shape ``(H, W)``
            Single real-valued image channel.

        Returns
        -------
        h : np.ndarray, shape ``(H, W)``
            Un-normalised DHT coefficients.
        """
        from scipy.fft import fft2

        X_fft = fft2(x)  # complex FFT2, shape (H, W)
        # DHT = Re(FFT) - Im(FFT)
        return X_fft.real - X_fft.imag

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalised 2-D DHT to every image channel.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Flattened image batch.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, C*H*W)``
            DHT coefficients scaled by ``1/√(H*W)``.
        """
        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                # Normalise by 1/√(H·W) to make the transform orthonormal
                result[n, c] = self._dht2(imgs[n, c]) / np.sqrt(self.H_ * self.W_)
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse DHT (identical to the forward transform).

        The normalised DHT satisfies ``H(H(x)) = x``, so the forward and
        inverse transforms are the same function.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``

        Returns
        -------
        Xr : np.ndarray, shape ``(N, C*H*W)``
        """
        # DHT is self-inverse (up to scale factor already applied in transform)
        return self.transform(X)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros: normalised DHT has ``|det J| = 1``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, D)``

        Returns
        -------
        log_det : np.ndarray, shape ``(N,)``
        """
        return np.zeros(X.shape[0])


class DCTRotation(ImageBijector):
    """Type-II orthonormal 2-D Discrete Cosine Transform rotation.

    Applies the 2-D DCT-II with orthonormal normalisation (``norm="ortho"``)
    to each spatial channel.  Because the ortho-normalised DCT is an
    orthogonal matrix, ``log|det J| = 0`` for all inputs.

    Parameters
    ----------
    C : int, default 1
        Number of image channels.
    H : int, default 8
        Image height in pixels.
    W : int, default 8
        Image width in pixels.

    Attributes
    ----------
    C_ : int
        Fitted number of channels.
    H_ : int
        Fitted image height.
    W_ : int
        Fitted image width.

    Notes
    -----
    For an orthonormal DCT matrix :math:`\\mathbf{D}` acting on the
    vectorised image :math:`\\mathbf{x}`:

    .. math::

        \\mathbf{y} = \\mathbf{D}\\,\\mathbf{x},
        \\quad
        \\log |\\det J| = \\log |\\det \\mathbf{D}| = 0

    because :math:`\\mathbf{D}` is orthogonal (``det = ±1``).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((5, 64))  # N=5, C=1, H=8, W=8
    >>> layer = DCTRotation(C=1, H=8, W=8)
    >>> layer.fit(X)  # doctest: +ELLIPSIS
    DCTRotation(...)
    >>> Xt = layer.transform(X)
    >>> Xr = layer.inverse_transform(Xt)
    >>> np.allclose(X, Xr, atol=1e-10)
    True
    """

    def __init__(self, C: int = 1, H: int = 8, W: int = 8):
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray, y=None) -> DCTRotation:
        """Store spatial dimensions; no data-dependent fitting required.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``

        Returns
        -------
        self : DCTRotation
        """
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply orthonormal 2-D DCT-II to every image channel.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Flattened image batch.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, C*H*W)``
            Orthonormal DCT-II coefficients.
        """
        from scipy.fft import dctn

        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                # norm="ortho" yields the orthonormal variant of the DCT-II
                result[n, c] = dctn(imgs[n, c], norm="ortho")
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply orthonormal 2-D inverse DCT (DCT-III scaled).

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            DCT coefficient batch from :meth:`transform`.

        Returns
        -------
        Xr : np.ndarray, shape ``(N, C*H*W)``
            Reconstructed image batch.
        """
        from scipy.fft import idctn

        N = X.shape[0]
        imgs = X.reshape(N, self.C_, self.H_, self.W_)  # (N, C, H, W)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                # idctn with norm="ortho" is the exact inverse of dctn with norm="ortho"
                result[n, c] = idctn(imgs[n, c], norm="ortho")
        return self._to_flat(result)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros: orthonormal DCT has ``log|det J| = 0``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, D)``

        Returns
        -------
        log_det : np.ndarray, shape ``(N,)``
        """
        return np.zeros(X.shape[0])


class RandomChannelRotation(ImageBijector):
    """Orthogonal 1×1 convolution that mixes information across channels.

    Generates a random orthogonal matrix ``Q ∈ ℝ^{C×C}`` via QR
    decomposition and applies it as a 1×1 convolution, independently at every
    spatial location.  This is equivalent to multiplying the ``(H*W, C)``
    spatial-pixel matrix by ``Q^T`` on the right:

    .. math::

        (N,\\, H{\\cdot}W,\\, C)\\;
        \\overset{\\times\\,\\mathbf{Q}^\\top}{\\longrightarrow}\\;
        (N,\\, H{\\cdot}W,\\, C)

    Because ``Q`` is orthogonal, ``log|det J| = 0``.

    Parameters
    ----------
    C : int, default 1
        Number of image channels.
    H : int, default 8
        Image height in pixels.
    W : int, default 8
        Image width in pixels.
    random_state : int or None, default None
        Seed for the random number generator used to initialise ``Q``.

    Attributes
    ----------
    rotation_matrix_ : np.ndarray, shape ``(C, C)``
        The fitted orthogonal rotation matrix ``Q``.

    Notes
    -----
    The forward pass reshapes the tensor as

    .. math::

        (N, C, H, W)
        \\xrightarrow{\\text{transpose}} (N \\cdot H \\cdot W, C)
        \\xrightarrow{\\times\\,\\mathbf{Q}^\\top} (N \\cdot H \\cdot W, C)
        \\xrightarrow{\\text{reshape}} (N, C, H, W)

    The inverse uses :math:`\\mathbf{Q}^{-1} = \\mathbf{Q}^\\top`, so

    .. math::

        (N \\cdot H \\cdot W, C)
        \\xrightarrow{\\times\\,\\mathbf{Q}} (N \\cdot H \\cdot W, C)

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((8, 3 * 8 * 8))  # N=8, C=3, H=8, W=8
    >>> layer = RandomChannelRotation(C=3, H=8, W=8, random_state=42)
    >>> layer.fit(X)  # doctest: +ELLIPSIS
    RandomChannelRotation(...)
    >>> Xt = layer.transform(X)
    >>> Xr = layer.inverse_transform(Xt)
    >>> np.allclose(X, Xr, atol=1e-10)
    True
    """

    def __init__(
        self,
        C: int = 1,
        H: int = 8,
        W: int = 8,
        random_state: int | None = None,
    ):
        self.C = C
        self.H = H
        self.W = W
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RandomChannelRotation:
        """Draw a random orthogonal rotation matrix via QR decomposition.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Training data (used only to confirm shape; values are ignored).

        Returns
        -------
        self : RandomChannelRotation
        """
        rng = np.random.default_rng(self.random_state)
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        # Sample a random Gaussian matrix and orthogonalise via QR
        A = rng.standard_normal((self.C_, self.C_))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix_ = Q  # (C, C) orthogonal matrix
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Mix channels with the random orthogonal matrix.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Flattened image batch.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, C*H*W)``
            Channel-rotated flattened image batch.
        """
        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        # Rearrange to (N*H*W, C) for batch matrix multiply
        spatial = imgs.transpose(0, 2, 3, 1).reshape(-1, self.C_)
        # Apply Q^T: (N*H*W, C) @ (C, C)^T → (N*H*W, C)
        rotated = spatial @ self.rotation_matrix_.T
        # Restore tensor shape (N, C, H, W) then flatten
        result = rotated.reshape(N, self.H_, self.W_, self.C_).transpose(0, 3, 1, 2)
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the channel rotation using Q^T (since Q^{-1} = Q^T).

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Rotated flattened image batch from :meth:`transform`.

        Returns
        -------
        Xr : np.ndarray, shape ``(N, C*H*W)``
            Reconstructed flattened image batch.
        """
        N = X.shape[0]
        imgs = X.reshape(N, self.C_, self.H_, self.W_)  # (N, C, H, W)
        # Rearrange to (N*H*W, C)
        spatial = imgs.transpose(0, 2, 3, 1).reshape(-1, self.C_)
        # Multiply by Q (the transpose of Q^T) to invert: Q^{-1} = Q^T
        rotated = spatial @ self.rotation_matrix_  # (N*H*W, C) @ (C, C)
        result = rotated.reshape(N, self.H_, self.W_, self.C_).transpose(0, 3, 1, 2)
        return self._to_flat(result)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros: orthogonal rotation has ``log|det J| = 0``.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, D)``

        Returns
        -------
        log_det : np.ndarray, shape ``(N,)``
        """
        return np.zeros(X.shape[0])


class ImageRBIG:
    """RBIG orchestrator for image data.

    Alternates between marginal Gaussianisation and an orthonormal spatial
    rotation for ``n_layers`` steps, progressively pushing the joint
    distribution of image pixels towards a multivariate Gaussian.

    Each layer applies:

    1. :class:`~rbig._src.marginal.MarginalGaussianize` — maps every feature
       dimension to a standard normal marginal distribution.
    2. An orthonormal rotation selected by ``strategy`` — decorrelates
       features without changing the differential entropy.

    Parameters
    ----------
    n_layers : int, default 10
        Number of (marginal + rotation) layer pairs to apply.
    C : int, default 1
        Number of image channels passed to the rotation layers.
    H : int, default 8
        Image height in pixels passed to the rotation layers.
    W : int, default 8
        Image width in pixels passed to the rotation layers.
    strategy : str, default ``"dct"``
        Rotation strategy.  One of:

        * ``"dct"`` — Type-II orthonormal DCT (:class:`DCTRotation`).
        * ``"hartley"`` — Discrete Hartley Transform
          (:class:`HartleyRotation`).
        * ``"random_channel"`` — Random orthogonal channel mixing
          (:class:`RandomChannelRotation`).

        Any unknown string falls back to ``"dct"``.
    random_state : int or None, default None
        Base seed for rotation layers that use randomness (``random_channel``).
        Layer ``i`` uses seed ``random_state + i``.
    verbose : bool or int, default=False
        Controls progress bar display.  ``False`` (or ``0``) disables all
        progress bars.  ``True`` (or ``1``) shows a progress bar for the
        ``fit`` loop.  ``2`` additionally shows progress bars for
        ``transform`` and ``inverse_transform``.

    Attributes
    ----------
    layers_ : list of tuple (MarginalGaussianize, ImageBijector)
        Fitted (marginal, rotation) pairs in application order.
    X_transformed_ : np.ndarray, shape ``(N, C*H*W)``
        Final transformed representation after the last layer.

    Notes
    -----
    The composed forward transform for a single sample :math:`\\mathbf{x}` is

    .. math::

        \\mathbf{z} = (R_L \\circ G_L \\circ \\cdots \\circ R_1 \\circ G_1)(\\mathbf{x})

    where :math:`G_\\ell` is marginal Gaussianisation and :math:`R_\\ell` is an
    orthonormal rotation at layer :math:`\\ell`.  Because each rotation is
    orthonormal, the total log-determinant is determined entirely by the
    marginal transforms.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 64))  # 50 images, C=1, H=8, W=8
    >>> model = ImageRBIG(n_layers=3, C=1, H=8, W=8, strategy="dct", random_state=0)
    >>> model.fit(X)  # doctest: +ELLIPSIS
    ImageRBIG(...)
    >>> Xt = model.transform(X)
    >>> Xt.shape
    (50, 64)
    >>> Xr = model.inverse_transform(Xt)
    >>> Xr.shape
    (50, 64)
    """

    def __init__(
        self,
        n_layers: int = 10,
        C: int = 1,
        H: int = 8,
        W: int = 8,
        strategy: str = "dct",
        random_state: int | None = None,
        verbose: bool | int = False,
    ):
        self.n_layers = n_layers
        self.C = C
        self.H = H
        self.W = W
        self.strategy = strategy
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: np.ndarray, y=None) -> ImageRBIG:
        """Fit all (marginal, rotation) layer pairs sequentially.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Training image batch in flattened format.

        Returns
        -------
        self : ImageRBIG
        """
        from rbig._src._progress import maybe_tqdm
        from rbig._src.marginal import MarginalGaussianize

        self.layers_ = []
        Xt = X.copy()  # working copy updated layer by layer
        pbar = maybe_tqdm(
            range(self.n_layers),
            verbose=self.verbose,
            level=1,
            desc="Fitting ImageRBIG",
            total=self.n_layers,
        )
        for i in pbar:
            # Step 1: marginal Gaussianisation
            marginal = MarginalGaussianize()
            Xt_m = marginal.fit_transform(Xt)
            # Step 2: orthonormal spatial rotation
            rotation = self._make_rotation(seed=i)
            rotation.fit(Xt_m)
            self.layers_.append((marginal, rotation))
            Xt = rotation.transform(Xt_m)  # update for the next iteration
        self.X_transformed_ = Xt  # final representation after all layers
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all fitted layers in forward order.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Image batch to transform.

        Returns
        -------
        Xt : np.ndarray, shape ``(N, C*H*W)``
            Gaussianised representation.
        """
        from rbig._src._progress import maybe_tqdm

        Xt = X.copy()
        layers_iter = maybe_tqdm(
            self.layers_,
            verbose=self.verbose,
            level=2,
            desc="Transforming",
            total=len(self.layers_),
        )
        for marginal, rotation in layers_iter:
            Xt = rotation.transform(marginal.transform(Xt))
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all fitted layers in reverse order.

        Parameters
        ----------
        X : np.ndarray, shape ``(N, C*H*W)``
            Gaussianised representation to invert.

        Returns
        -------
        Xr : np.ndarray, shape ``(N, C*H*W)``
            Reconstructed image batch in the original domain.
        """
        from rbig._src._progress import maybe_tqdm

        Xt = X.copy()
        # Reverse the layer list; apply inverse of each (rotation first, then marginal)
        layers_iter = maybe_tqdm(
            list(reversed(self.layers_)),
            verbose=self.verbose,
            level=2,
            desc="Inverse transforming",
            total=len(self.layers_),
        )
        for marginal, rotation in layers_iter:
            Xt = marginal.inverse_transform(rotation.inverse_transform(Xt))
        return Xt

    def _make_rotation(self, seed: int = 0):
        """Instantiate the rotation layer for the given layer index.

        Parameters
        ----------
        seed : int, default 0
            Layer index; combined with ``random_state`` for reproducibility.

        Returns
        -------
        rotation : ImageBijector
            An unfitted rotation bijector of the type specified by
            ``self.strategy``.
        """
        # Combine base seed with layer index so each layer gets a unique seed
        rng_seed = (self.random_state or 0) + seed
        if self.strategy == "dct":
            return DCTRotation(C=self.C, H=self.H, W=self.W)
        elif self.strategy == "hartley":
            return HartleyRotation(C=self.C, H=self.H, W=self.W)
        elif self.strategy == "random_channel":
            return RandomChannelRotation(
                C=self.C, H=self.H, W=self.W, random_state=rng_seed
            )
        else:
            # Unknown strategy: fall back to DCT
            return DCTRotation(C=self.C, H=self.H, W=self.W)
