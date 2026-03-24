"""RBIG Model - Rotation-Based Iterative Gaussianization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

from rbig._src.marginal import MarginalGaussianize
from rbig._src.rotation import PCARotation


@dataclass
class RBIGLayer:
    """Single RBIG layer: marginal Gaussianization followed by rotation.

    One iteration of the RBIG algorithm applies two successive bijections:

    1. **Marginal Gaussianization** – maps each feature independently to a
       standard Gaussian via its empirical CDF and the probit function:

           z = Φ⁻¹(F̂ₙ(x))

       where F̂ₙ is the estimated marginal CDF and Φ⁻¹ is the standard
       normal quantile function.

    2. **Rotation/whitening** – applies a linear transform R (default: PCA
       whitening) to de-correlate the Gaussianized features:

           y = R · z

    The full single-layer transform is therefore:

        y = R · Φ⁻¹(F̂ₙ(x))

    Parameters
    ----------
    marginal : MarginalGaussianize, optional
        Marginal Gaussianization transform (fitted per feature).
        Defaults to a new ``MarginalGaussianize`` instance.
    rotation : PCARotation, optional
        Rotation transform applied after marginal Gaussianization.
        Defaults to a new ``PCARotation`` instance.

    Attributes
    ----------
    marginal : MarginalGaussianize
        Fitted marginal transform.
    rotation : PCARotation
        Fitted rotation transform.

    Notes
    -----
    The layer log-det-Jacobian is the sum of the marginal and rotation
    contributions:

        log|det J_layer(x)| = log|det J_marginal(x)| + log|det J_rotation|
                             = ∑ᵢ log|Φ⁻¹′(F̂ₙ(xᵢ)) · f̂ₙ(xᵢ)| + log|det J_rotation|

    The rotation term ``log|det J_rotation|`` is zero only when the rotation is
    strictly orthogonal (``|det R| = 1``).  The default
    ``PCARotation(whiten=True)`` includes per-component scaling and therefore
    has a generally non-zero ``log_det_jacobian``.  More general rotations such
    as ICA are also not guaranteed to be orthogonal.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    From ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537–549. https://doi.org/10.1109/TNN.2011.2106511

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.model import RBIGLayer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((500, 3))
    >>> layer = RBIGLayer()
    >>> layer.fit(X)
    RBIGLayer(...)
    >>> Z = layer.transform(X)
    >>> Z.shape
    (500, 3)
    """

    marginal: MarginalGaussianize = field(default_factory=MarginalGaussianize)
    rotation: PCARotation = field(default_factory=PCARotation)

    def fit(self, X: np.ndarray, y=None) -> RBIGLayer:
        """Fit the marginal and rotation transforms to data X.

        First fits the marginal Gaussianizer on X, applies it to obtain the
        intermediate Gaussianized representation, then fits the rotation on
        that intermediate representation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for sklearn pipeline compatibility.

        Returns
        -------
        self : RBIGLayer
            The fitted layer.
        """
        Xm = self.marginal.fit_transform(
            X
        )  # shape (n_samples, n_features) - Gaussianized
        self.rotation.fit(Xm)  # fit rotation on the Gaussianized data
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply marginal Gaussianization then rotation: y = R · Φ⁻¹(F̂ₙ(x)).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Y : np.ndarray of shape (n_samples, n_features)
            Transformed data after Gaussianization and rotation.
        """
        Xm = self.marginal.transform(
            X
        )  # marginal Gaussianization, shape (n_samples, n_features)
        return self.rotation.transform(Xm)  # rotation, shape (n_samples, n_features)

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for this layer at input X.

        The total log-det-Jacobian is the sum of contributions from the
        marginal step and the rotation step:

            log|det J_layer(x)| = log|det J_marginal(x)| + log|det J_rotation(z)|

        For strictly orthogonal rotations (e.g. ``RandomRotation``), the
        rotation term is zero.  For ``PCARotation(whiten=True)`` the rotation
        includes a per-component rescaling, so its term is generally non-zero.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points.

        Returns
        -------
        ldj : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant of the layer Jacobian.
        """
        Xm = self.marginal.transform(
            X
        )  # intermediate Gaussianized data, shape (n_samples, n_features)
        # marginal log-det + rotation log-det (non-zero for PCARotation with whiten=True)
        return self.marginal.log_det_jacobian(X) + self.rotation.log_det_jacobian(Xm)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the layer: apply inverse rotation then inverse marginal.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the layer's output (latent) space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original input space.
        """
        Xr = self.rotation.inverse_transform(
            X
        )  # undo rotation, shape (n_samples, n_features)
        return self.marginal.inverse_transform(Xr)  # undo marginal Gaussianization


class AnnealedRBIG(TransformerMixin, BaseEstimator):
    """Rotation-Based Iterative Gaussianization (RBIG).

    RBIG is a density estimation and data transformation method that
    iteratively Gaussianizes multivariate data by alternating between:

    1. **Marginal Gaussianization**: mapping each feature to a Gaussian
       using its empirical CDF and the probit transform.
    2. **Rotation**: applying an orthogonal matrix (PCA or ICA) to
       de-correlate the Gaussianized features.

    The process repeats until the total correlation (TC) of the
    transformed data converges.  After fitting, the model represents a
    normalizing flow whose density is given by the change-of-variables
    formula:

        log p(x) = log p_Z(f(x)) + log|det J_f(x)|

    where ``f`` is the composition of all fitted layers and ``p_Z`` is a
    standard multivariate Gaussian.

    Parameters
    ----------
    n_layers : int, default=100
        Maximum number of RBIG layers to apply.  Early stopping via
        ``patience`` may halt training before this limit.
    rotation : str, default="pca"
        Rotation method: ``"pca"`` (PCA with whitening), ``"ica"``
        (Independent Component Analysis), or ``"random"`` (Haar-distributed
        orthogonal rotation).
    patience : int, default=10
        Number of consecutive layers showing a TC change smaller than
        ``tol`` before training stops early.  (Formerly ``zero_tolerance``,
        which is still accepted but deprecated.)
    tol : float or "auto", default=1e-5
        Convergence threshold for the per-layer change in total correlation:
        ``|TC(k) − TC(k−1)| < tol``.  When set to ``"auto"``, the tolerance
        is chosen adaptively based on the number of training samples using
        an empirically calibrated lookup table.
    random_state : int or None, default=None
        Seed for the random number generator used by stochastic components
        such as ICA or random rotations.
    strategy : list or None, default=None
        Optional per-layer override list.  Each entry may be a string
        (rotation name) or a ``(rotation_name, marginal_name)`` pair.
        Entries cycle if the list is shorter than ``n_layers``.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    layers_ : list of RBIGLayer
        Fitted RBIG layers in application order.
    tc_per_layer_ : list of float
        Total correlation of the data after each fitted layer.
    log_det_train_ : np.ndarray of shape (n_samples,)
        Accumulated per-sample log-det-Jacobian over all layers,
        computed on the training data during ``fit``.
    X_transformed_ : np.ndarray of shape (n_samples, n_features)
        Training data after passing through all fitted layers.

    Notes
    -----
    Total correlation is defined as:

        TC(X) = ∑ᵢ H(Xᵢ) − H(X)

    where H(Xᵢ) is the marginal entropy of the i-th feature and H(X) is
    the joint entropy.  For a fully Gaussianized, independent dataset,
    TC = 0.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    From ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537–549. https://doi.org/10.1109/TNN.2011.2106511

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.model import AnnealedRBIG
    >>> rng = np.random.default_rng(42)
    >>> X = rng.standard_normal((300, 4))
    >>> model = AnnealedRBIG(n_layers=20, rotation="pca")
    >>> model.fit(X)
    <rbig._src.model.AnnealedRBIG object at ...>
    >>> Z = model.transform(X)
    >>> Z.shape
    (300, 4)
    >>> model.score(X)  # mean log-likelihood in nats
    -5.65...
    """

    def __init__(
        self,
        n_layers: int = 100,
        rotation: str = "pca",
        patience: int = 10,
        tol: float | str = 1e-5,
        random_state: int | None = None,
        strategy: list | None = None,
        *,
        zero_tolerance: int | None = None,
    ):
        if zero_tolerance is not None:
            import warnings

            warnings.warn(
                "zero_tolerance is deprecated, use patience instead",
                FutureWarning,
                stacklevel=2,
            )
            patience = zero_tolerance
        self.n_layers = n_layers
        self.rotation = rotation
        self.patience = patience
        self.zero_tolerance = zero_tolerance
        self.tol = tol
        self.random_state = random_state
        self.strategy = strategy

    def fit(self, X: np.ndarray, y=None) -> AnnealedRBIG:
        """Fit the RBIG model by iteratively Gaussianizing X.

        At each layer k the algorithm:

        1. Builds a new :class:`RBIGLayer` with the configured marginal and
           rotation transforms.
        2. Fits the layer on the current working copy ``Xt``.
        3. Accumulates the per-sample log-det-Jacobian:
           ``log_det_train_ += log|det J_k(Xt)|``.
        4. Advances ``Xt`` through the layer: ``Xt = f_k(Xt)``.
        5. Measures residual total correlation: ``TC(Xt) = ∑ᵢ H(Xᵢ) − H(X)``.
        6. Stops early when TC has not changed by more than ``tol`` for
           ``patience`` consecutive layers.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for sklearn pipeline compatibility.

        Returns
        -------
        self : AnnealedRBIG
            The fitted model.
        """
        X = validate_data(self, X)
        n_samples, n_features = X.shape
        if n_samples < 2:
            raise ValueError(
                f"RBIG requires at least 2 samples to estimate marginal CDFs, "
                f"got n_samples = {n_samples}."
            )
        self.n_features_in_ = n_features  # remember input dimensionality
        self.layers_: list[RBIGLayer] = []
        self.tc_per_layer_: list[float] = []

        # Validate and resolve tolerance
        if self.tol == "auto":
            tol = self._get_information_tolerance(n_samples)
        elif isinstance(self.tol, int | float):
            tol = float(self.tol)
        else:
            raise ValueError(f"tol must be a float or 'auto', got {self.tol!r}")
        self.tol_: float = tol  # store resolved tolerance for inspection

        Xt = X.copy()  # working copy; shape (n_samples, n_features)
        self.log_det_train_ = np.zeros(
            n_samples
        )  # accumulated log|det J|; shape (n_samples,)
        zero_count = 0  # consecutive non-improving layer counter

        for i in range(self.n_layers):
            # Build layer i with the appropriate marginal and rotation components
            layer = RBIGLayer(
                marginal=self._make_marginal(layer_index=i),
                rotation=self._make_rotation(layer_index=i),
            )
            layer.fit(Xt)
            # Accumulate log|det J_i(Xt)| before advancing Xt
            self.log_det_train_ += layer.log_det_jacobian(Xt)
            Xt = layer.transform(Xt)  # advance to next representation
            self.layers_.append(layer)

            # Measure residual total correlation: TC = sum_i H(Xi) - H(X)
            tc = self._total_correlation(Xt)
            self.tc_per_layer_.append(tc)

            if i > 0:
                # Check convergence: how much did TC improve this layer?
                delta = abs(self.tc_per_layer_[-2] - tc)
                if delta < tol:
                    zero_count += 1
                else:
                    zero_count = 0  # reset on any significant improvement

            # Stop early if TC has been flat for patience consecutive layers
            if zero_count >= self.patience:
                break

        # Store the fully transformed training data for efficient entropy estimation
        self.X_transformed_ = Xt  # shape (n_samples, n_features)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map X to the Gaussian latent space through all fitted layers.

        Applies each fitted :class:`RBIGLayer` in order:
        ``Z = fₖ(… f₂(f₁(x)) …)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Data in the approximately Gaussian latent space.
        """
        check_is_fitted(self)
        Xt = validate_data(self, X, reset=False).copy()
        for layer in self.layers_:
            Xt = layer.transform(Xt)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map latent-space data back to the original input space.

        Applies layers in reverse order:
        ``x = f₁⁻¹(… fₖ₋₁⁻¹(fₖ⁻¹(z)) …)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the latent (approximately Gaussian) space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data recovered in the original input space.
        """
        check_is_fitted(self)
        Xt = validate_data(self, X, reset=False).copy()
        for layer in reversed(self.layers_):
            Xt = layer.inverse_transform(Xt)
        return Xt

    def fit_transform(self, X: np.ndarray, y=None) -> np.ndarray:
        """Fit the model to X and return the latent-space representation.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Transformed data in the latent space.
        """
        return self.fit(X).transform(X)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-sample log-likelihood under the fitted density model.

        Uses the change-of-variables formula for normalizing flows:

            log p(x) = log p_Z(f(x)) + log|det J_f(x)|

        where ``p_Z = 𝒩(0, I)`` is the standard Gaussian base density,
        ``f`` is the composition of all fitted layers, and ``J_f(x)`` is
        the Jacobian of ``f`` at ``x``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points at which to evaluate the log-likelihood.

        Returns
        -------
        log_prob : np.ndarray of shape (n_samples,)
            Per-sample log-likelihood in nats.

        Notes
        -----
        The log-det-Jacobian is accumulated layer by layer to avoid
        recomputing intermediate representations:

            log|det J_f(x)| = ∑ₖ log|det J_fₖ(xₖ₋₁)|
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        Xt = X.copy()  # shape (n_samples, n_features)
        log_det_jac = np.zeros(X.shape[0])  # accumulator; shape (n_samples,)
        for layer in self.layers_:
            # Accumulate log|det Jₖ| before advancing through layer k
            log_det_jac += layer.log_det_jacobian(Xt)
            Xt = layer.transform(Xt)  # xₖ = fₖ(xₖ₋₁)
        # log p_Z(z) = sum_i log N(z_i; 0, 1); shape (n_samples,)
        log_pz = np.sum(stats.norm.logpdf(Xt), axis=1)
        # change-of-variables: log p(x) = log p_Z(f(x)) + log|det J_f(x)|
        return log_pz + log_det_jac

    def score(self, X: np.ndarray, y=None) -> float:
        """Mean log-likelihood of samples X under the fitted density.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to evaluate.
        y : ignored
            Not used, present for sklearn pipeline compatibility.

        Returns
        -------
        mean_log_prob : float
            Average per-sample log-likelihood in nats.
        """
        return float(np.mean(self.score_samples(X)))

    def entropy(self) -> float:
        """Differential entropy of the fitted distribution in nats.

        Estimated from the training data using:

            H(X) = −𝔼_X[log p(x)]

        The expectation is approximated by the sample mean over the training
        set.  The log-likelihoods are obtained via the efficient cached path
        :meth:`score_samples_raw_` which reuses pre-computed quantities from
        ``fit``.

        Returns
        -------
        h : float
            Estimated entropy in nats.  Always ≥ 0 for continuous
            distributions.

        Notes
        -----
        This is equivalent to ``-self.score(X_train)`` but avoids the cost
        of re-passing training data through all layers.
        """
        check_is_fitted(self)
        return float(-np.mean(self.score_samples_raw_()))

    def score_samples_raw_(self) -> np.ndarray:
        """Log-likelihood for the stored training data without recomputing layers.

        Reuses ``X_transformed_`` and ``log_det_train_`` cached during
        :meth:`fit`, so the cost is a single Gaussian log-pdf evaluation
        rather than a full forward pass through all layers.

        Returns
        -------
        log_prob : np.ndarray of shape (n_samples,)
            Per-sample log-likelihood of the training data in nats.
        """
        # log p_Z evaluated at the pre-computed transformed training data
        log_pz = np.sum(
            stats.norm.logpdf(self.X_transformed_), axis=1
        )  # shape (n_samples,)
        # add the accumulated log-det-Jacobian stored during fit
        return log_pz + self.log_det_train_

    def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        """Generate samples from the learned distribution.

        Draws i.i.d. standard Gaussian samples in the latent space and maps
        them back to the data space via the inverse normalizing flow.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : int or None, optional
            Seed for the random number generator.  If ``None``, a random
            seed is used.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_features_in_)
            Samples in the original data space.
        """
        check_is_fitted(self)
        rng = np.random.default_rng(random_state)
        Z = rng.standard_normal((n_samples, self.n_features_in_))  # latent samples
        return self.inverse_transform(Z)

    def predict_proba(
        self,
        X: np.ndarray,
        domain: str = "input",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Return probability density estimates for X.

        Uses the change-of-variables formula via the full Jacobian matrix
        to compute the density in the requested domain.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data points to evaluate.
        domain : str, default="input"
            Which domain to return densities in:

            - ``"input"`` — density in the original data space:
              ``p(x) = p_Z(f(x)) · |det J_f(x)|``
            - ``"transform"`` — density in the Gaussian latent space:
              ``p_Z(f(x)) = ∏ᵢ φ(fᵢ(x))``
            - ``"both"`` — returns a tuple ``(p_input, p_transform)``

        Returns
        -------
        proba : np.ndarray of shape (n_samples,) or tuple
            Probability density estimates.  When ``domain="both"``, returns
            ``(p_input, p_transform)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        jac, Xt = self.jacobian(X, return_X_transform=True)

        # Work in log-space for numerical stability
        log_p_transform = np.sum(stats.norm.logpdf(Xt), axis=1)

        if domain == "transform":
            p_transform = np.exp(log_p_transform)
            p_transform = np.where(np.isfinite(p_transform), p_transform, 0.0)
            return p_transform

        # Input-domain density via change of variables (log-space)
        _sign, log_abs_det = np.linalg.slogdet(jac)
        log_p_input = log_p_transform + log_abs_det
        p_input = np.exp(log_p_input)
        p_input = np.where(np.isfinite(p_input), p_input, 0.0)

        if domain == "input":
            return p_input
        if domain == "both":
            p_transform = np.exp(log_p_transform)
            p_transform = np.where(np.isfinite(p_transform), p_transform, 0.0)
            return p_input, p_transform
        raise ValueError(
            f"Unknown domain: {domain!r}. Use 'input', 'transform', or 'both'."
        )

    def jacobian(
        self,
        X: np.ndarray,
        return_X_transform: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Compute the full Jacobian matrix of the RBIG transform.

        For each sample, returns the ``(n_features, n_features)`` Jacobian
        matrix ``df/dx`` of the composition of all fitted layers.  Uses the
        seed-dimension approach from the legacy implementation: for each input
        dimension ``idim``, a unit vector is propagated through the chain of
        per-feature marginal derivatives and rotation matrices.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data at which to evaluate the Jacobian.
        return_X_transform : bool, default False
            If True, also return the fully transformed data ``f(X)`` (computed
            as a side-effect of the Jacobian calculation).

        Returns
        -------
        jac : np.ndarray of shape (n_samples, n_features, n_features)
            Full Jacobian matrix per sample.  ``jac[n, i, j]`` is the partial
            derivative ``df_i/dx_j`` for the n-th sample.
        X_transformed : np.ndarray of shape (n_samples, n_features)
            Only returned when ``return_X_transform=True``.  The data after
            passing through all layers.
        """
        check_is_fitted(self)
        n_samples, n_features = X.shape

        # ── Forward pass: collect per-layer derivatives and rotation matrices ──
        derivs_per_layer = []  # each: (n_samples, n_features)
        rotmats_per_layer = []  # each: (n_features, n_features)

        Xt = X.copy()
        for layer in self.layers_:
            if not hasattr(layer.marginal, "_per_feature_log_deriv"):
                raise NotImplementedError(
                    f"Jacobian computation requires a marginal with "
                    f"_per_feature_log_deriv(); "
                    f"{type(layer.marginal).__name__} does not support this."
                )
            # Per-feature marginal derivatives and transformed data in one pass
            log_d, Xt_marginal = layer.marginal._per_feature_log_deriv(
                Xt, return_transform=True
            )
            derivs_per_layer.append(np.exp(log_d))

            # Rotation matrix in row-vector convention: y = z @ R
            rot = self._extract_rotation_matrix(layer.rotation)
            rotmats_per_layer.append(rot)

            # Advance through rotation only
            Xt = layer.rotation.transform(Xt_marginal)

        # ── Seed-dimension loop: propagate unit vectors through the chain ──
        jac = np.zeros((n_samples, n_features, n_features))

        for idim in range(n_features):
            # Initialize seed: unit vector in dimension idim
            XX = np.zeros((n_samples, n_features))
            XX[:, idim] = 1.0

            for derivs, R in zip(derivs_per_layer, rotmats_per_layer, strict=True):
                # Chain rule: XX_new = diag(derivs) @ XX @ R
                XX = (derivs * XX) @ R

            jac[:, :, idim] = XX

        if return_X_transform:
            return jac, Xt
        return jac

    @staticmethod
    def _extract_rotation_matrix(rotation) -> np.ndarray:
        """Extract the effective rotation matrix in row-vector convention.

        For PCA with whitening the effective matrix is
        ``components_.T / sqrt(explained_variance_)`` so that
        ``y = (x - mu) @ R``.

        Parameters
        ----------
        rotation : BaseTransform
            A fitted rotation object (PCARotation, ICARotation, etc.).

        Returns
        -------
        R : np.ndarray of shape (n_features, n_features)
            Rotation matrix such that ``y = x @ R`` (ignoring mean shift).
        """
        from rbig._src.rotation import ICARotation, PCARotation

        if isinstance(rotation, PCARotation):
            R = rotation.pca_.components_.T.copy()
            if rotation.whiten:
                R /= np.sqrt(rotation.pca_.explained_variance_)[np.newaxis, :]
            return R

        if isinstance(rotation, ICARotation):
            # ICA unmixing: W = components_, transform is x @ W.T
            if hasattr(rotation, "K_") and rotation.K_ is not None:
                # Picard path: y = (x @ K.T) @ W.T
                return rotation.K_.T @ rotation.W_.T
            return rotation.ica_.components_.T.copy()

        # Generic fallback: try to get rotation_matrix_ attribute.
        # These rotations apply X @ rotation_matrix_.T, so transpose
        # to match the y = x @ R convention used by PCA/ICA above.
        if hasattr(rotation, "rotation_matrix_"):
            return rotation.rotation_matrix_.T.copy()

        raise TypeError(
            f"Cannot extract rotation matrix from {type(rotation).__name__}. "
            f"Jacobian computation requires PCARotation, ICARotation, or an "
            f"object with a rotation_matrix_ attribute."
        )

    def _make_rotation(self, layer_index: int = 0):
        """Instantiate the rotation component for a given layer.

        Parameters
        ----------
        layer_index : int, default=0
            Index of the layer being constructed.  Used when cycling through
            a ``strategy`` list.

        Returns
        -------
        rotation : RotationBijector
            An unfitted rotation bijector instance.
        """
        if self.strategy is not None:
            # cycle through the strategy list to select rotation for this layer
            idx = layer_index % len(self.strategy)
            entry = self.strategy[idx]
            rotation_name = entry[0] if isinstance(entry, list | tuple) else entry
            return self._get_component(rotation_name, "rotation", layer_index)
        if self.rotation == "pca":
            return PCARotation(whiten=True)
        elif self.rotation == "ica":
            from rbig._src.rotation import ICARotation

            return ICARotation(random_state=self.random_state)
        elif self.rotation == "random":
            from rbig._src.rotation import RandomRotation

            seed = (self.random_state or 0) + layer_index
            return RandomRotation(random_state=seed)
        else:
            raise ValueError(
                f"Unknown rotation: {self.rotation}. Use 'pca', 'ica', or 'random'."
            )

    def _make_marginal(self, layer_index: int = 0):
        """Instantiate the marginal Gaussianization component for a given layer.

        Parameters
        ----------
        layer_index : int, default=0
            Index of the layer being constructed.  Used when cycling through
            a ``strategy`` list.

        Returns
        -------
        marginal : MarginalBijector
            An unfitted marginal Gaussianizer instance.
        """
        if self.strategy is not None:
            # cycle through the strategy list to select marginal for this layer
            idx = layer_index % len(self.strategy)
            entry = self.strategy[idx]
            marginal_name = (
                entry[1] if isinstance(entry, list | tuple) else "gaussianize"
            )
            return self._get_component(marginal_name, "marginal", layer_index)
        return MarginalGaussianize()

    def _get_component(self, name: str, kind: str, seed: int = 0):
        """Instantiate a rotation or marginal component by name.

        Parameters
        ----------
        name : str
            Component name, e.g. ``"pca"``, ``"ica"``, ``"gaussianize"``.
        kind : str
            Either ``"rotation"`` or ``"marginal"``.
        seed : int, default=0
            Layer index added to ``random_state`` to vary seeds per layer.

        Returns
        -------
        component : Bijector
            An unfitted bijector of the requested kind.
        """
        rng_seed = (self.random_state or 0) + seed
        if kind == "rotation":
            return self._make_rotation_by_name(name, rng_seed)
        return self._make_marginal_by_name(name, rng_seed)

    def _make_rotation_by_name(self, name: str, seed: int):
        """Instantiate a rotation bijector from its string name.

        Parameters
        ----------
        name : str
            One of ``"pca"``, ``"ica"``, or ``"random"``.
        seed : int
            Random seed for stochastic rotations.

        Returns
        -------
        rotation : RotationBijector
            The corresponding unfitted rotation instance.

        Raises
        ------
        ValueError
            If ``name`` is not a recognised rotation type.
        """
        if name == "pca":
            return PCARotation(whiten=True)
        if name == "ica":
            from rbig._src.rotation import ICARotation

            return ICARotation(random_state=seed)
        if name == "random":
            from rbig._src.rotation import RandomRotation

            return RandomRotation(random_state=seed)
        raise ValueError(f"Unknown rotation: {name!r}. Use 'pca', 'ica', or 'random'.")

    def _make_marginal_by_name(self, name: str, seed: int):
        """Instantiate a marginal Gaussianizer from its string name.

        Parameters
        ----------
        name : str
            One of ``"gaussianize"`` / ``"empirical"``, ``"quantile"``,
            ``"kde"``, ``"gmm"``, or ``"spline"``.
        seed : int
            Random seed for stochastic marginal estimators.

        Returns
        -------
        marginal : MarginalBijector
            The corresponding unfitted marginal Gaussianizer instance.

        Raises
        ------
        ValueError
            If ``name`` is not a recognised marginal type.
        """
        if name in ("gaussianize", "empirical", None):
            return MarginalGaussianize()
        if name == "quantile":
            from rbig._src.marginal import QuantileGaussianizer

            return QuantileGaussianizer(random_state=seed)
        if name == "kde":
            from rbig._src.marginal import KDEGaussianizer

            return KDEGaussianizer()
        if name == "gmm":
            from rbig._src.marginal import GMMGaussianizer

            return GMMGaussianizer(random_state=seed)
        if name == "spline":
            from rbig._src.marginal import SplineGaussianizer

            return SplineGaussianizer()
        raise ValueError(
            f"Unknown marginal: {name!r}. Use 'gaussianize', 'quantile', 'kde', 'gmm', or 'spline'."
        )

    @staticmethod
    def _get_information_tolerance(n_samples: int) -> float:
        """Compute a sample-size-adaptive convergence tolerance.

        Interpolates from an empirically calibrated lookup table mapping
        dataset size to an appropriate TC-change threshold.  Larger datasets
        can resolve finer changes in total correlation, so the tolerance
        decreases with sample count.

        Parameters
        ----------
        n_samples : int
            Number of training samples.

        Returns
        -------
        tol : float
            Adaptive tolerance value.
        """
        from scipy.interpolate import interp1d

        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        return float(interp1d(xxx, yyy, fill_value="extrapolate")(n_samples))

    @staticmethod
    def _calculate_negentropy(X: np.ndarray) -> np.ndarray:
        """Negentropy of each marginal: J(xᵢ) = H(Gauss) − H(xᵢ) ≥ 0.

        Negentropy measures how far a distribution is from Gaussian.  It is
        zero if and only if the distribution is Gaussian.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data whose per-feature negentropy is computed.

        Returns
        -------
        neg_entropy : np.ndarray of shape (n_features,)
            Non-negative negentropy for each feature dimension.

        Notes
        -----
        The negentropy is computed as:

            J(xᵢ) = H(𝒩(μᵢ, σᵢ²)) − H(xᵢ)

        where H(𝒩(μ, σ²)) = ½(1 + log(2πσ²)) is the Gaussian entropy with
        the same variance.
        """
        from rbig._src.densities import marginal_entropy

        # Gaussian entropy for a Gaussian with the same variance: 0.5*(1 + log(2*pi*var))
        gauss_h = 0.5 * (1 + np.log(2 * np.pi)) + 0.5 * np.log(np.var(X, axis=0))
        marg_h = marginal_entropy(X)  # empirical marginal entropy per feature
        return gauss_h - marg_h  # shape (n_features,); always >= 0

    @staticmethod
    def _total_correlation(X: np.ndarray) -> float:
        """Total correlation of X: TC(X) = ∑ᵢ H(Xᵢ) − H(X).

        Total correlation (also called multi-information) quantifies the
        statistical dependence among all features jointly.  It equals zero
        when all features are mutually independent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data whose total correlation is measured.

        Returns
        -------
        tc : float
            Total correlation in nats.  Non-negative by the subadditivity of
            entropy.

        Notes
        -----
        The joint entropy ``H(X)`` is estimated under a Gaussian assumption
        (using the log-determinant of the covariance matrix), while the
        marginal entropies ``H(Xᵢ)`` are estimated empirically.
        """
        from rbig._src.densities import joint_entropy_gaussian, marginal_entropy

        marg_h = marginal_entropy(X)  # per-feature entropy; shape (n_features,)
        joint_h = joint_entropy_gaussian(X)  # Gaussian approximation to joint entropy
        return float(np.sum(marg_h) - joint_h)
