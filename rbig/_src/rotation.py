"""Rotation transforms for RBIG.

This module provides linear rotation transforms used as the second step in
each RBIG (Rotation-Based Iterative Gaussianization) layer.  After marginal
Gaussianization, a rotation is applied to mix the dimensions, allowing
subsequent marginal steps to reduce higher-order statistical dependencies.

Available transforms
--------------------
PCARotation
    PCA-based rotation with optional whitening (z = Lambda^{-1/2} V^T (x - mu)).
ICARotation
    ICA-based rotation (Picard solver or FastICA fallback): s = W K x.
RandomRotation
    Uniformly random orthogonal rotation drawn from the Haar measure via QR.
RandomOrthogonalProjection
    Semi-orthogonal random projection, possibly reducing dimensionality.
GaussianRandomProjection
    Johnson-Lindenstrauss style projection with Gaussian entries.
OrthogonalDimensionalityReduction
    Full orthogonal rotation followed by optional dimension truncation.
PicardRotation
    ICA rotation via the Picard algorithm (preferred) or FastICA fallback.

References
----------
Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
537-549. https://doi.org/10.1109/TNN.2011.2106511
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from rbig._src.base import BaseTransform


class PCARotation(BaseTransform):
    """PCA-based rotation with optional whitening (decorrelation + rescaling).

    Fits a standard PCA (via scikit-learn's :class:`~sklearn.decomposition.PCA`)
    and uses it as a linear rotation transform.  When ``whiten=True`` (default),
    each principal component is additionally rescaled by the reciprocal square
    root of its eigenvalue so the output has unit variance per component::

        z = Lambda^{-1/2} V^T (x - mu)

    where ``V`` in R^{D x K} is the matrix of leading eigenvectors (principal
    axes), ``Lambda`` in R^{K x K} is the diagonal eigenvalue matrix, and
    ``mu`` is the sample mean.  When ``whiten=False``, the rescaling is
    omitted and the transform is a pure rotation::

        z = V ^ T(x - mu)

    Parameters
    ----------
    n_components : int or None, default None
        Number of principal components to retain.  If ``None``, all D
        components are kept.
    whiten : bool, default True
        If True, divide each component by sqrt(lambda_i) to decorrelate
        *and* normalise variance.  If False, only rotate (and center).

    Attributes
    ----------
    pca_ : sklearn.decomposition.PCA
        Fitted PCA object containing eigenvectors, eigenvalues, and the
        sample mean.

    Notes
    -----
    The log-absolute-Jacobian determinant for the whitening transform is::

        log|det J| = -1/2 * sum_i log(lambda_i)

    because each whitening factor Lambda^{-1/2} contributes
    ``-1/2 * log(lambda_i)`` per component.  For a pure rotation
    (``whiten=False``), the determinant is 1 and the log is 0.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import PCARotation
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> pca_rot = PCARotation(whiten=True).fit(X)
    >>> Z = pca_rot.transform(X)
    >>> Z.shape
    (200, 4)
    >>> ldj = pca_rot.log_det_jacobian(X)
    >>> ldj.shape
    (200,)
    """

    def __init__(self, n_components: int | None = None, whiten: bool = True):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: np.ndarray, y=None) -> PCARotation:
        """Fit PCA to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCARotation
            Fitted transform instance.
        """
        # Stores eigenvectors, eigenvalues, and mean in pca_
        self.pca_ = PCA(n_components=self.n_components, whiten=self.whiten)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply PCA rotation (and optional whitening) to X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Rotated (and optionally whitened) data.
        """
        return self.pca_.transform(X)  # (N, D) -> (N, K)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the PCA rotation (and optional whitening).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Data in the PCA / whitened space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
        """
        return self.pca_.inverse_transform(X)  # (N, K) -> (N, D)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute Jacobian determinant (constant for linear transforms).

        For a whitening PCA the Jacobian determinant is constant::

            log|det J| = -1/2 * sum_i log(lambda_i)

        where ``lambda_i`` are the PCA eigenvalues (``explained_variance_``).
        For a plain rotation (``whiten=False``), ``|det J| = 1`` and the
        log is 0.

        .. note::
            This method is only valid when the transform is square (i.e.
            ``n_components`` is ``None`` or equals the number of input
            features).  A dimensionality-reducing PCA (``n_components`` <
            ``n_features``) is not bijective and its Jacobian determinant is
            undefined.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine the number of samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Constant per-sample log absolute Jacobian determinant.
        """
        if self.whiten:
            # Whitening scales each eigendirection by lambda_i^{-1/2}
            # -> log|det| = -1/2 * sum log(lambda_i)
            log_det = -0.5 * np.sum(np.log(self.pca_.explained_variance_))
        else:
            # Pure rotation: |det Q| = 1 -> log|det| = 0
            log_det = 0.0
        return np.full(X.shape[0], log_det)


class ICARotation(BaseTransform):
    """ICA-based rotation using the Picard algorithm or FastICA fallback.

    Fits an Independent Component Analysis (ICA) model that learns a linear
    unmixing matrix.  When the optional ``picard`` package is available, it
    is used for faster and more accurate convergence::

        s = W K x

    where ``K`` in R^{K x D} is a pre-whitening matrix and ``W`` in
    R^{K x K} is the ICA unmixing matrix.  The combined transform is
    ``W K``.

    If ``picard`` is not installed, :class:`sklearn.decomposition.FastICA`
    is used as a drop-in replacement.

    Parameters
    ----------
    n_components : int or None, default None
        Number of independent components.  If ``None``, all D components
        are estimated (square unmixing matrix).
    random_state : int or None, default None
        Seed for reproducible ICA initialisation.

    Attributes
    ----------
    K_ : np.ndarray of shape (n_components, n_features) or None
        Pre-whitening matrix from the Picard solver.  ``None`` when using
        the FastICA fallback.
    W_ : np.ndarray of shape (n_components, n_components) or None
        ICA unmixing matrix from the Picard solver.  ``None`` when using
        FastICA.
    ica_ : sklearn.decomposition.FastICA or None
        Fitted FastICA object used when Picard is unavailable.
    n_features_in_ : int
        Number of input features (set only when using Picard).

    Notes
    -----
    The log-absolute-Jacobian determinant is::

        log|det J| = log|det(W K)|

    for the Picard path, or ``log|det(components_)|`` for FastICA.  The
    Jacobian is constant (independent of x) for any linear transform.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import ICARotation
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> ica = ICARotation(random_state=0).fit(X)
    >>> S = ica.transform(X)
    >>> S.shape
    (200, 3)
    """

    def __init__(
        self, n_components: int | None = None, random_state: int | None = None
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> ICARotation:
        """Fit the ICA model (Picard if available, otherwise FastICA).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : ICARotation
            Fitted transform instance.
        """
        try:
            from picard import picard

            n = X.shape[1] if self.n_components is None else self.n_components
            # Picard expects data as (n_features, n_samples), so transpose X
            K, W, _ = picard(
                X.T,
                n_components=n,
                random_state=self.random_state,
                max_iter=500,
                tol=1e-5,
            )
            self.K_ = K  # whitening matrix, shape (K, D)
            self.W_ = W  # unmixing matrix, shape (K, K)
            self.n_features_in_ = X.shape[1]
        except ImportError:
            from sklearn.decomposition import FastICA

            # Fall back to FastICA when picard is not installed
            self.ica_ = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=500,
            )
            self.ica_.fit(X)
            self.K_ = None  # signals that FastICA path is active
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the ICA unmixing to X: s = W K x.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original (mixed) space.

        Returns
        -------
        S : np.ndarray of shape (n_samples, n_components)
            Estimated independent components.
        """
        if self.K_ is None:
            # FastICA path: uses sklearn's built-in transform
            return self.ica_.transform(X)
        # Picard path: first whiten X with K, then unmix with W
        Xw = X @ self.K_.T  # (N, D) @ (D, K) -> (N, K)  whitening step
        return Xw @ self.W_.T  # (N, K) @ (K, K) -> (N, K)  unmixing step

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the ICA unmixing: x = (W K)^{-1} s.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Independent-component representation.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original mixed space.
        """
        if self.K_ is None:
            return self.ica_.inverse_transform(X)
        # Invert unmixing W then whitening K using pseudo-inverses
        Xw = X @ np.linalg.pinv(self.W_).T  # (N, K) -> (N, K)
        return Xw @ np.linalg.pinv(self.K_).T  # (N, K) -> (N, D)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute Jacobian determinant (constant for linear transforms).

        Computes ``log|det(W K)|`` (Picard path) or
        ``log|det(components_)|`` (FastICA path).  The result is replicated
        for every sample since the Jacobian of a linear transform is constant.

        .. note::
            This method is only valid when the unmixing matrix is square (i.e.
            ``n_components`` is ``None`` or equals the number of input
            features).  A non-square unmixing matrix is not bijective and its
            Jacobian determinant is undefined.  A :exc:`ValueError` is raised
            in that case.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine the number of samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Constant per-sample log absolute Jacobian determinant.

        Raises
        ------
        ValueError
            If the unmixing matrix is not square (``n_components != n_features``).
        """
        if self.K_ is None:
            W = self.ica_.components_  # shape (K, D) or (D, D)
            if W.shape[0] != W.shape[1]:
                raise ValueError(
                    "ICARotation.log_det_jacobian is only defined for square "
                    "unmixing matrices. Got components_ with shape "
                    f"{W.shape}. Ensure that `n_components` is None or "
                    "equals the number of features."
                )
            log_det = np.log(np.abs(np.linalg.det(W)))
        else:
            # Combined unmixing matrix: W @ K, shape (K, D)
            WK = self.W_ @ self.K_
            if WK.shape[0] != WK.shape[1]:
                raise ValueError(
                    "ICARotation.log_det_jacobian is only defined for square "
                    "unmixing matrices. Got W @ K with shape "
                    f"{WK.shape}. Ensure that `n_components` is None or "
                    "equals the number of features."
                )
            log_det = np.log(np.abs(np.linalg.det(WK)))
        return np.full(X.shape[0], log_det)


from rbig._src.base import RotationBijector


class RandomRotation(RotationBijector):
    """Random orthogonal rotation drawn from the Haar measure via QR.

    Generates a uniformly random orthogonal matrix Q in R^{D x D} by QR
    decomposing a matrix of i.i.d. standard-normal entries and applying a
    sign correction to ensure the result is Haar-uniform::

        A ~ N(0, 1)^{D x D},  A = Q R,  Q <- Q * diag(sign(diag(R)))

    The sign correction guarantees that Q is sampled uniformly from the
    orthogonal group O(D) (the Haar measure).

    Parameters
    ----------
    random_state : int or None, default None
        Seed for reproducible rotation matrix generation.

    Attributes
    ----------
    rotation_matrix_ : np.ndarray of shape (n_features, n_features)
        The sampled orthogonal rotation matrix Q.

    Notes
    -----
    Because Q is orthogonal, ``|det Q| = 1`` and::

        log|det J| = log|det Q| = 0

    This is the default implementation inherited from
    :class:`~rbig._src.base.RotationBijector`.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import RandomRotation
    >>> rng_data = np.random.default_rng(0)
    >>> X = rng_data.standard_normal((100, 4))
    >>> rot = RandomRotation(random_state=42).fit(X)
    >>> Z = rot.transform(X)
    >>> Z.shape
    (100, 4)
    >>> Xr = rot.inverse_transform(Z)
    >>> np.allclose(X, Xr)
    True
    """

    def __init__(self, random_state: int | None = None):
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RandomRotation:
        """Sample a Haar-uniform orthogonal rotation matrix of size D x D.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (used only to infer the dimensionality D).

        Returns
        -------
        self : RandomRotation
            Fitted transform instance with ``rotation_matrix_`` set.
        """
        rng = np.random.default_rng(self.random_state)
        n_features = X.shape[1]
        # Draw a random D x D Gaussian matrix
        A = rng.standard_normal((n_features, n_features))
        Q, R = np.linalg.qr(A)
        # Sign correction: multiply columns of Q by sign(diag(R)) for Haar measure
        Q *= np.sign(np.diag(R))  # ensures uniform distribution on O(D)
        self.rotation_matrix_ = Q  # shape (D, D)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Rotate X by the sampled orthogonal matrix: z = Q x.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Rotated data.
        """
        return X @ self.rotation_matrix_.T  # (N, D) @ (D, D) -> (N, D)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the rotation: x = Q^T z = Q^{-1} z.

        Because Q is orthogonal, its inverse equals its transpose.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Rotated data.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original space.
        """
        return X @ self.rotation_matrix_  # Q^{-1} = Q^T -> (N, D) @ (D, D)


class RandomOrthogonalProjection(RotationBijector):
    """Semi-orthogonal random projection from D to K dimensions via QR.

    Generates a semi-orthogonal matrix P in R^{D x K} (K <= D) whose
    columns are orthonormal, obtained by taking the first K columns of a
    QR decomposition of a random Gaussian matrix::

        A ~ N(0, 1)^{D x K},  A = Q R,  P = Q[:, :K]

    The forward transform projects D-dimensional input to K dimensions::

        z = X P   where P in R^{D x K}

    Parameters
    ----------
    n_components : int or None, default None
        Output dimensionality K.  If ``None``, K = D (square case).
    random_state : int or None, default None
        Seed for reproducible matrix generation.

    Attributes
    ----------
    projection_matrix_ : np.ndarray of shape (n_features, n_components)
        Semi-orthogonal projection matrix P with orthonormal columns.
    input_dim_ : int
        Input dimensionality D.
    output_dim_ : int
        Output dimensionality K.

    Notes
    -----
    When K = D the matrix is fully orthogonal and ``log|det J| = 0``.
    When K < D the transform is not invertible and both
    :meth:`inverse_transform` and :meth:`get_log_det_jacobian` raise
    :class:`NotImplementedError`.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import RandomOrthogonalProjection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> proj = RandomOrthogonalProjection(random_state=0).fit(X)
    >>> Z = proj.transform(X)
    >>> Z.shape
    (100, 4)
    """

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> RandomOrthogonalProjection:
        """Build the semi-orthogonal projection matrix P.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (used only to infer D = n_features).

        Returns
        -------
        self : RandomOrthogonalProjection
            Fitted transform instance.
        """
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        # Random Gaussian seed matrix; QR gives orthonormal columns
        A = rng.standard_normal((D, K))
        Q, _ = np.linalg.qr(A)
        self.projection_matrix_ = Q[:, :K]  # (D, K)  semi-orthogonal basis
        self.input_dim_ = D
        self.output_dim_ = K
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X from D to K dimensions: z = X P.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        return X @ self.projection_matrix_  # (N, D) @ (D, K) -> (N, K)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the projection (only valid for square case K = D).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Projected data.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original space (exact only when K = D).

        Raises
        ------
        NotImplementedError
            If ``n_components < n_features`` (projection is not invertible).
        """
        if self.output_dim_ < self.input_dim_:
            raise NotImplementedError(
                "RandomOrthogonalProjection with n_components < input dimension "
                "is not bijective; inverse_transform is undefined."
            )
        return X @ self.projection_matrix_.T  # exact inverse only when square (N, D)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros for the square (bijective) case.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine n_samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Zeros, because ``|det P| = 1`` for a square orthogonal matrix.

        Raises
        ------
        NotImplementedError
            If ``n_components < n_features`` (Jacobian determinant undefined).
        """
        if self.output_dim_ < self.input_dim_:
            raise NotImplementedError(
                "RandomOrthogonalProjection with n_components < input dimension "
                "does not have a well-defined Jacobian determinant."
            )
        # For a square orthogonal matrix, |det(J)| = 1, so log|det(J)| = 0.
        return np.zeros(X.shape[0])


class GaussianRandomProjection(RotationBijector):
    """Johnson-Lindenstrauss style random projection with Gaussian entries.

    Constructs a random projection matrix M in R^{D x K} whose entries are
    drawn i.i.d. from N(0, 1/K)::

        M_ij ~ N(0, 1/K)

    The 1/K normalisation approximately preserves pairwise Euclidean
    distances (Johnson-Lindenstrauss lemma)::

        (1 - eps)||x - y||^2 <= ||Mx - My||^2 <= (1 + eps)||x - y||^2

    with high probability when K = O(eps^{-2} log n).

    Parameters
    ----------
    n_components : int or None, default None
        Output dimensionality K.  If ``None``, K = D (square case).
    random_state : int or None, default None
        Seed for reproducible matrix generation.

    Attributes
    ----------
    matrix_ : np.ndarray of shape (n_features, n_components)
        The random projection matrix with entries ~ N(0, 1/K).

    Notes
    -----
    Unlike :class:`RandomOrthogonalProjection`, the columns of this matrix
    are *not* orthogonal, so ``|det M| != 1`` in general.
    :meth:`get_log_det_jacobian` returns zeros as an approximation.
    For density estimation where accuracy matters, prefer
    :class:`RandomOrthogonalProjection` or :class:`RandomRotation`.

    The inverse uses the Moore-Penrose pseudoinverse computed by
    :func:`numpy.linalg.pinv`.

    References
    ----------
    Johnson, W. B., & Lindenstrauss, J. (1984). Extensions of Lipschitz
    mappings into a Hilbert space. *Contemporary Mathematics*, 26, 189-206.

    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import GaussianRandomProjection
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> grp = GaussianRandomProjection(random_state=0).fit(X)
    >>> Z = grp.transform(X)
    >>> Z.shape
    (100, 4)
    """

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> GaussianRandomProjection:
        """Build the Gaussian random projection matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (used only to infer D = n_features).

        Returns
        -------
        self : GaussianRandomProjection
            Fitted transform instance.
        """
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        # Entries drawn from N(0, 1), then scaled by 1/sqrt(K) for distance preservation
        self.matrix_ = rng.standard_normal((D, K)) / np.sqrt(K)  # (D, K)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X: z = X M.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Projected data.
        """
        return X @ self.matrix_  # (N, D) @ (D, K) -> (N, K)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Approximate inverse via the Moore-Penrose pseudoinverse.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Projected data.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Approximately recovered original data.
        """
        # Pseudoinverse: M^+ = (M^T M)^{-1} M^T; here we use pinv(M).T
        return X @ np.linalg.pinv(self.matrix_).T  # (N, K) @ (K, D) -> (N, D)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros (approximation; Gaussian projections are not isometric).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine n_samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Zeros (approximate; the true log-det is generally non-zero).
        """
        return np.zeros(X.shape[0])


class OrthogonalDimensionalityReduction(RotationBijector):
    """Full orthogonal rotation followed by optional dimension truncation.

    Applies a D x D orthogonal rotation Q (drawn from the Haar measure via
    QR) and then retains only the first K <= D components::

        z = (X Q^T)[:, :K]

    The rotation is sampled fresh at ``fit`` time from a square
    standard-normal matrix processed through QR with sign correction.

    Parameters
    ----------
    n_components : int or None, default None
        Number of output dimensions K.  If ``None``, K = D (no truncation).
    random_state : int or None, default None
        Seed for reproducible rotation matrix generation.

    Attributes
    ----------
    rotation_matrix_ : np.ndarray of shape (n_features, n_features)
        Full D x D orthogonal rotation matrix Q.
    n_components_ : int
        Number of retained output dimensions K.
    input_dim_ : int
        Input dimensionality D.

    Notes
    -----
    When K = D the transform is a bijection and::

        log|det J| = log|det Q| = 0

    When K < D the transform is not invertible; both
    :meth:`inverse_transform` and :meth:`get_log_det_jacobian` raise
    :class:`NotImplementedError`.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import OrthogonalDimensionalityReduction
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 4))
    >>> odr = OrthogonalDimensionalityReduction(random_state=0).fit(X)
    >>> Z = odr.transform(X)
    >>> Z.shape
    (100, 4)
    """

    def __init__(
        self,
        n_components: int | None = None,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None) -> OrthogonalDimensionalityReduction:
        """Sample a Haar-uniform D x D rotation matrix.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data (used only to infer D = n_features).

        Returns
        -------
        self : OrthogonalDimensionalityReduction
            Fitted transform instance.
        """
        rng = np.random.default_rng(self.random_state)
        D = X.shape[1]
        K = self.n_components if self.n_components is not None else D
        # QR of a random Gaussian matrix gives a Haar-uniform orthogonal matrix
        A = rng.standard_normal((D, D))
        Q, _ = np.linalg.qr(A)
        self.rotation_matrix_ = Q  # (D, D)
        self.n_components_ = K
        self.input_dim_ = D
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Rotate then truncate: z = (X Q^T)[:, :K].

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Rotated and (optionally) truncated data.
        """
        Xr = X @ self.rotation_matrix_.T  # (N, D) @ (D, D) -> (N, D)  full rotation
        return Xr[:, : self.n_components_]  # (N, K)  keep first K components

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the rotation (only valid for square case K = D).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Rotated data.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original space.

        Raises
        ------
        NotImplementedError
            If ``n_components < n_features`` (not invertible).
        """
        if self.n_components_ < self.input_dim_:
            raise NotImplementedError(
                "OrthogonalDimensionalityReduction with n_components < input dimension "
                "is not bijective; inverse_transform is undefined."
            )
        return X @ self.rotation_matrix_  # (N, D) @ (D, D) -> (N, D)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros for the square (bijective) case.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine n_samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Zeros, because ``|det Q| = 1`` for any orthogonal Q.

        Raises
        ------
        NotImplementedError
            If ``n_components < n_features`` (Jacobian determinant undefined).
        """
        if self.n_components_ < self.input_dim_:
            raise NotImplementedError(
                "OrthogonalDimensionalityReduction with n_components < input dimension "
                "does not have a well-defined Jacobian determinant."
            )
        return np.zeros(X.shape[0])


class PicardRotation(RotationBijector):
    """ICA rotation via the Picard algorithm with a FastICA fallback.

    Fits an ICA model that learns maximally statistically-independent
    sources.  When the optional ``picard`` package is available, it solves::

        K, W = picard(X^T)
        s = W K x

    where ``K`` in R^{K x D} is the pre-whitening matrix and ``W`` in
    R^{K x K} is the Picard unmixing matrix.  The log-det-Jacobian is::

        log|det J| = log|det(W K)|

    If ``picard`` is not installed (or incompatible), :class:`sklearn
    .decomposition.FastICA` is used as a fallback.

    Parameters
    ----------
    n_components : int or None, default None
        Number of independent components K.  If ``None``, K = D.
    extended : bool, default False
        If True, use the extended Picard algorithm that can handle both
        super- and sub-Gaussian sources (passed directly to ``picard``).
    random_state : int or None, default None
        Seed for reproducible initialisation.
    max_iter : int, default 500
        Maximum number of ICA iterations.
    tol : float, default 1e-5
        Convergence tolerance for the ICA algorithm.

    Attributes
    ----------
    K_ : np.ndarray of shape (n_components, n_features) or None
        Pre-whitening matrix (Picard path).  ``None`` when using FastICA.
    W_ : np.ndarray of shape (n_components, n_components) or None
        Unmixing matrix (Picard path).  ``None`` when using FastICA.
    use_picard_ : bool
        True if the Picard solver was used; False if FastICA was used.
    ica_ : sklearn.decomposition.FastICA or None
        Fitted FastICA model (FastICA path only).

    Notes
    -----
    The log-det-Jacobian is::

        log|det J| = log|det(W K)|

    for the Picard path, or ``log|det(components_)|`` for the FastICA path.
    The Jacobian is constant because the transform is linear.

    :meth:`get_log_det_jacobian` raises :class:`ValueError` if the unmixing
    matrix is not square (i.e. ``n_components != n_features``).

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Ablin, P., Cardoso, J.-F., & Gramfort, A. (2018). Faster Independent
    Component Analysis by Preconditioning with Hessian Approximations.
    *IEEE Transactions on Signal Processing*, 66(15), 4040-4049.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.rotation import PicardRotation
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> pic = PicardRotation(random_state=0).fit(X)
    >>> S = pic.transform(X)
    >>> S.shape
    (200, 3)
    """

    def __init__(
        self,
        n_components: int | None = None,
        extended: bool = False,
        random_state: int | None = None,
        max_iter: int = 500,
        tol: float = 1e-5,
    ):
        self.n_components = n_components
        self.extended = extended
        self.random_state = random_state
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: np.ndarray, y=None) -> PicardRotation:
        """Fit ICA (Picard if available, otherwise FastICA).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PicardRotation
            Fitted transform instance.
        """
        try:
            from picard import picard

            n = X.shape[1] if self.n_components is None else self.n_components
            # Picard expects (n_features, n_samples); returns K (whitening) and W (unmixing)
            K, W, _ = picard(
                X.T,  # (D, N)
                n_components=n,
                random_state=self.random_state,
                max_iter=self.max_iter,
                tol=self.tol,
                extended=self.extended,
            )
            self.K_ = K  # pre-whitening matrix, shape (K, D)
            self.W_ = W  # ICA unmixing matrix, shape (K, K)
            self.use_picard_ = True
        except (ImportError, TypeError):
            from sklearn.decomposition import FastICA

            # FastICA fallback when picard is unavailable or incompatible
            self.ica_ = FastICA(
                n_components=self.n_components,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
            self.ica_.fit(X)
            self.K_ = None
            self.use_picard_ = False
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the ICA unmixing: s = W K x.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original (mixed) space.

        Returns
        -------
        S : np.ndarray of shape (n_samples, n_components)
            Estimated independent components.
        """
        if not self.use_picard_:
            return self.ica_.transform(X)
        # Picard path: whiten then unmix
        # (N, D) @ (D, K) -> (N, K) whitening; then (N, K) @ (K, K) -> (N, K) unmixing
        return (X @ self.K_.T) @ self.W_.T

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the ICA unmixing: x = (W K)^{-1} s.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_components)
            Independent-component representation.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original mixed space.
        """
        if not self.use_picard_:
            return self.ica_.inverse_transform(X)
        # Invert unmixing W then whitening K, each via pseudo-inverse
        return (X @ np.linalg.pinv(self.W_).T) @ np.linalg.pinv(self.K_).T

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute Jacobian determinant: log|det(W K)| (constant).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine n_samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Constant per-sample log absolute Jacobian determinant.

        Raises
        ------
        ValueError
            If the unmixing matrix is not square
            (``n_components != n_features``).
        """
        if not self.use_picard_:
            W = self.ica_.components_
            if W.shape[0] != W.shape[1]:
                raise ValueError(
                    "PicardRotation.get_log_det_jacobian is only defined for square "
                    "unmixing matrices when using the FastICA fallback. Got "
                    f"components_ with shape {W.shape}. Ensure that `n_components` "
                    "is None or equals the number of features."
                )
            log_det = np.log(np.abs(np.linalg.det(W)))
        else:
            WK = self.W_ @ self.K_  # combined unmixing matrix, shape (K, D)
            if WK.shape[0] != WK.shape[1]:
                raise ValueError(
                    "PicardRotation.get_log_det_jacobian is only defined for square "
                    "unmixing matrices when using the Picard solver. Got "
                    f"W @ K with shape {WK.shape}. Ensure that `n_components` "
                    "is None or equals the number of features."
                )
            log_det = np.log(np.abs(np.linalg.det(WK)))
        return np.full(X.shape[0], log_det)
