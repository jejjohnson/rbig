from abc import ABC, abstractmethod

import numpy as np


class BaseTransform(ABC):
    """Abstract base class for all RBIG transforms.

    Defines the common interface shared by every learnable data transformation
    in this library: fitting to data, forward mapping, and its inverse.
    Subclasses that support density estimation should also implement
    ``log_det_jacobian``.

    Notes
    -----
    The change-of-variables formula for a normalizing flow relates the density
    of the input ``x`` to a base density ``p_Z`` via a bijection ``f``:

        log p(x) = log p_Z(f(x)) + log|det J_f(x)|

    where ``J_f(x)`` is the Jacobian of ``f`` evaluated at ``x``.
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseTransform":
        """Fit the transform to data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to estimate any internal parameters.

        Returns
        -------
        self : BaseTransform
            The fitted transform instance.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted forward transform to data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        ...

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted inverse transform to data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the transformed (latent) space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original input space.
        """
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data X, then return the transformed result.

        Equivalent to calling ``self.fit(X).transform(X)`` but may be
        overridden for efficiency when a single pass suffices.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to fit and transform.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log absolute determinant of the Jacobian evaluated at X.

        For a transform f, this returns ``log|det J_f(x)|`` per sample,
        which is the volume-correction term required in the change-of-variables
        formula for density estimation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant of the Jacobian.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError


class BaseITMeasure(ABC):
    """Abstract base class for information-theoretic (IT) measures.

    Subclasses implement specific measures such as mutual information,
    total correlation, or entropy from data, potentially conditioning on
    an optional second variable ``Y``.

    Notes
    -----
    Common IT quantities include:

    * **Entropy**: H(X) = −𝔼_X[log p(x)]
    * **Mutual information**: I(X; Y) = H(X) + H(Y) − H(X, Y)
    * **Total correlation**: TC(X) = ∑ᵢ H(Xᵢ) − H(X)
    """

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray | None = None) -> "BaseITMeasure":
        """Fit the IT measure to data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Primary data array.
        Y : np.ndarray of shape (n_samples, n_targets) or None, optional
            Secondary data array, used for joint or conditional measures.

        Returns
        -------
        self : BaseITMeasure
            The fitted measure instance.
        """
        ...

    @abstractmethod
    def score(self) -> float:
        """Return the scalar value of the fitted IT measure.

        Returns
        -------
        value : float
            The computed information-theoretic quantity (e.g., entropy in
            nats, mutual information in bits, total correlation in nats).
        """
        ...


class Bijector(ABC):
    """Abstract base class for invertible transformations (bijectors).

    A bijector implements a differentiable, invertible map ``f : ℝᵈ → ℝᵈ``
    and provides the log absolute determinant of its Jacobian.  These are
    the building blocks of normalizing flows.

    The density of a random variable ``X = f⁻¹(Z)`` where ``Z ~ p_Z`` is:

        log p(x) = log p_Z(f(x)) + log|det J_f(x)|

    Notes
    -----
    Concrete subclasses must implement :meth:`fit`, :meth:`transform`,
    :meth:`inverse_transform`, and :meth:`get_log_det_jacobian`.
    ``log_det_jacobian`` is provided as a convenience alias for the last
    method, for compatibility with ``RBIGLayer``.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    From ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537–549. https://doi.org/10.1109/TNN.2011.2106511
    """

    @abstractmethod
    def fit(self, X: np.ndarray) -> "Bijector":
        """Fit the bijector to data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : Bijector
            The fitted bijector.
        """
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward bijection f(x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Data mapped to the latent space.
        """
        ...

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse bijection f⁻¹(z).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the latent space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original space.
        """
        ...

    @abstractmethod
    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute log|det J_f(x)| per sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant of the forward Jacobian J_f.
        """
        ...

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the bijector to X, then return f(X).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to fit and transform.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Alias for get_log_det_jacobian for compatibility with RBIGLayer.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points.

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Per-sample log|det J_f(x)|.
        """
        return self.get_log_det_jacobian(X)


class MarginalBijector(Bijector):
    """Abstract bijector for independent, per-dimension (marginal) transforms.

    Each feature dimension is transformed by a separate invertible function.
    Because the transform is applied independently to each coordinate, the
    Jacobian is diagonal and its log-determinant is the sum of per-dimension
    log-derivatives:

        log|det J_f(x)| = ∑ᵢ log|f′(xᵢ)|

    Subclasses implement concrete marginal mappings such as empirical CDF
    Gaussianization, quantile transform, or kernel density estimation.

    Notes
    -----
    In RBIG, the marginal step maps each dimension to a standard Gaussian via

        z = Φ⁻¹(F̂ₙ(x))

    where F̂ₙ is the estimated marginal CDF and Φ⁻¹ is the standard normal
    quantile function (probit).
    """

    # inherits all abstract methods


class RotationBijector(Bijector):
    """Abstract bijector for orthogonal rotation transforms.

    Rotation matrices Q satisfy QᵀQ = I and |det Q| = 1, so the
    log-absolute-determinant of the Jacobian is exactly zero:

        log|det J_Q(x)| = log|det Q| = log 1 = 0

    This default implementation of ``get_log_det_jacobian`` returns a
    zero vector of length ``n_samples``, which concrete subclasses (e.g.
    PCA, ICA, random orthogonal) can inherit without override.

    Notes
    -----
    In RBIG, the rotation step de-correlates the marginally Gaussianized
    data, driving the joint distribution closer to a standard multivariate
    Gaussian with each iteration.
    """

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Return zeros because |det Q| = 1 for any orthogonal matrix Q.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points (used only to determine n_samples).

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Array of zeros; rotations do not change volume.
        """
        # Orthogonal matrices preserve volume: log|det Q| = 0 for all x
        return np.zeros(X.shape[0])


class CompositeBijector(Bijector):
    """A bijector that chains a sequence of bijectors in order.

    Applies bijectors ``f₁, f₂, …, fₖ`` in sequence so that the composite
    map is ``g = fₖ ∘ … ∘ f₂ ∘ f₁``.  The log-det-Jacobian of the
    composition follows the chain rule:

        log|det J_g(x)| = ∑ₖ log|det J_fₖ(xₖ₋₁)|

    where ``xₖ₋₁ = fₖ₋₁ ∘ … ∘ f₁(x)`` is the input to the k-th bijector.

    Parameters
    ----------
    bijectors : list of Bijector
        Ordered list of bijectors to chain.  They are applied left-to-right
        during ``transform`` and right-to-left during ``inverse_transform``.

    Attributes
    ----------
    bijectors : list of Bijector
        The constituent bijectors in application order.

    Notes
    -----
    Fitting is done sequentially: each bijector is fitted to the output of
    the previous one, so that the full model is trained in a single
    ``fit`` call.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.base import CompositeBijector
    >>> from rbig._src.marginal import MarginalGaussianize
    >>> from rbig._src.rotation import PCARotation
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 4))
    >>> cb = CompositeBijector([MarginalGaussianize(), PCARotation()])
    >>> cb.fit(X)  # doctest: +ELLIPSIS
    <rbig._src.base.CompositeBijector ...>
    >>> Z = cb.transform(X)
    >>> Z.shape
    (200, 4)
    """

    def __init__(self, bijectors: list):
        self.bijectors = bijectors

    def fit(self, X: np.ndarray) -> "CompositeBijector":
        """Fit each bijector sequentially on the output of the previous one.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : CompositeBijector
            The fitted composite bijector.
        """
        Xt = X.copy()  # working copy; shape (n_samples, n_features)
        for b in self.bijectors:
            # fit bijector b on current Xt, then advance Xt to b's output
            Xt = b.fit_transform(Xt)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply all bijectors left-to-right: g(x) = fₖ(… f₁(x) …).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Data after passing through every bijector in sequence.
        """
        Xt = X.copy()  # shape (n_samples, n_features)
        for b in self.bijectors:
            Xt = b.transform(Xt)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the composite map: g⁻¹(z) = f₁⁻¹(… fₖ⁻¹(z) …).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the latent space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original input space.
        """
        Xt = X.copy()  # shape (n_samples, n_features)
        # reverse order to undo the forward composition
        for b in reversed(self.bijectors):
            Xt = b.inverse_transform(Xt)
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Sum log|det Jₖ| over all bijectors (chain rule).

        Uses the chain rule for Jacobian determinants:

            log|det J_g(x)| = ∑ₖ log|det J_fₖ(xₖ₋₁)|

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points.

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Per-sample sum of log-det-Jacobians across all bijectors.
        """
        Xt = X.copy()  # shape (n_samples, n_features)
        log_det = np.zeros(X.shape[0])  # accumulator, shape (n_samples,)
        for b in self.bijectors:
            # add log|det Jₖ| at the *current* intermediate input Xt
            log_det += b.get_log_det_jacobian(Xt)
            # advance Xt to the output of bijector b for the next iteration
            Xt = b.transform(Xt)
        return log_det
