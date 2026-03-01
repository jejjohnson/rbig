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


from rbig._src.base import Bijector


class ImageBijector(Bijector):
    """Base class handling flattened <-> image tensor reshaping."""

    def _to_tensor(self, X: np.ndarray) -> np.ndarray:
        """Reshape flat (N, C*H*W) to (N, C, H, W)."""
        N = X.shape[0]
        C, H, W = self.C_, self.H_, self.W_
        return X.reshape(N, C, H, W)

    def _to_flat(self, X: np.ndarray) -> np.ndarray:
        """Reshape (N, C, H, W) to (N, C*H*W)."""
        N = X.shape[0]
        return X.reshape(N, -1)

    def fit(self, X: np.ndarray) -> ImageBijector:
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class OrthogonalWaveletLayer(ImageBijector):
    """DWT (Haar/Daubechies) squeeze: (N,C,H,W) -> (N,4C,H/2,W/2)."""

    def __init__(self, wavelet: str = "haar", C: int = 1, H: int = 8, W: int = 8):
        self.wavelet = wavelet
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray) -> OrthogonalWaveletLayer:
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        import pywt

        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = []
        for n in range(N):
            channels = []
            for c in range(self.C_):
                coeffs = pywt.dwt2(imgs[n, c], self.wavelet)
                cA, (cH, cV, cD) = coeffs
                channels.extend([cA, cH, cV, cD])
            result.append(np.stack(channels, axis=0))  # (4C, H/2, W/2)
        out = np.array(result)  # (N, 4C, H/2, W/2)
        return out.reshape(N, -1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        import pywt

        N = X.shape[0]
        H2, W2 = self.H_ // 2, self.W_ // 2
        imgs = X.reshape(N, 4 * self.C_, H2, W2)
        result = []
        for n in range(N):
            channels = []
            for c in range(self.C_):
                cA = imgs[n, 4 * c]
                cH = imgs[n, 4 * c + 1]
                cV = imgs[n, 4 * c + 2]
                cD = imgs[n, 4 * c + 3]
                reconstructed = pywt.idwt2((cA, (cH, cV, cD)), self.wavelet)
                channels.append(reconstructed)
            result.append(np.stack(channels, axis=0))
        out = np.array(result)  # (N, C, H, W)
        return out.reshape(N, -1)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class HartleyRotation(ImageBijector):
    """Discrete Hartley Transform (real-to-real, orthogonal, self-inverse)."""

    def __init__(self, C: int = 1, H: int = 8, W: int = 8):
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray) -> HartleyRotation:
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def _dht2(self, x: np.ndarray) -> np.ndarray:
        """2D Discrete Hartley Transform."""
        from scipy.fft import fft2

        X_fft = fft2(x)
        return X_fft.real - X_fft.imag

    def transform(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                result[n, c] = self._dht2(imgs[n, c]) / np.sqrt(self.H_ * self.W_)
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        # DHT is self-inverse (up to scale factor already applied in transform)
        return self.transform(X)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class DCTRotation(ImageBijector):
    """Discrete Cosine Transform (Type II, orthonormal)."""

    def __init__(self, C: int = 1, H: int = 8, W: int = 8):
        self.C = C
        self.H = H
        self.W = W

    def fit(self, X: np.ndarray) -> DCTRotation:
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        from scipy.fft import dctn

        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                result[n, c] = dctn(imgs[n, c], norm="ortho")
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        from scipy.fft import idctn

        N = X.shape[0]
        imgs = X.reshape(N, self.C_, self.H_, self.W_)
        result = np.zeros_like(imgs)
        for n in range(N):
            for c in range(self.C_):
                result[n, c] = idctn(imgs[n, c], norm="ortho")
        return self._to_flat(result)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class RandomChannelRotation(ImageBijector):
    """Orthogonal 1x1 convolution (rotates channel vectors)."""

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

    def fit(self, X: np.ndarray) -> RandomChannelRotation:
        rng = np.random.default_rng(self.random_state)
        self.C_ = self.C
        self.H_ = self.H
        self.W_ = self.W
        A = rng.standard_normal((self.C_, self.C_))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix_ = Q  # (C, C)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        imgs = self._to_tensor(X)  # (N, C, H, W)
        spatial = imgs.transpose(0, 2, 3, 1).reshape(-1, self.C_)
        rotated = spatial @ self.rotation_matrix_.T
        result = rotated.reshape(N, self.H_, self.W_, self.C_).transpose(0, 3, 1, 2)
        return self._to_flat(result)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        imgs = X.reshape(N, self.C_, self.H_, self.W_)
        spatial = imgs.transpose(0, 2, 3, 1).reshape(-1, self.C_)
        rotated = spatial @ self.rotation_matrix_  # Q^{-1} = Q^T
        result = rotated.reshape(N, self.H_, self.W_, self.C_).transpose(0, 3, 1, 2)
        return self._to_flat(result)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(X.shape[0])


class ImageRBIG:
    """Orchestrator class composing marginal + rotation layers for images."""

    def __init__(
        self,
        n_layers: int = 10,
        C: int = 1,
        H: int = 8,
        W: int = 8,
        strategy: str = "dct",
        random_state: int | None = None,
    ):
        self.n_layers = n_layers
        self.C = C
        self.H = H
        self.W = W
        self.strategy = strategy
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> ImageRBIG:
        from rbig._src.marginal import MarginalGaussianize

        self.layers_ = []
        Xt = X.copy()
        for i in range(self.n_layers):
            marginal = MarginalGaussianize()
            Xt_m = marginal.fit_transform(Xt)
            rotation = self._make_rotation(seed=i)
            rotation.fit(Xt_m)
            self.layers_.append((marginal, rotation))
            Xt = rotation.transform(Xt_m)
        self.X_transformed_ = Xt
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        for marginal, rotation in self.layers_:
            Xt = rotation.transform(marginal.transform(Xt))
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        Xt = X.copy()
        for marginal, rotation in reversed(self.layers_):
            Xt = marginal.inverse_transform(rotation.inverse_transform(Xt))
        return Xt

    def _make_rotation(self, seed: int = 0):
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
            return DCTRotation(C=self.C, H=self.H, W=self.W)
