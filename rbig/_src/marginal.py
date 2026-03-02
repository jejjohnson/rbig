"""Marginal Gaussianization transforms.

Each transform maps individual feature dimensions independently to uniform
[0, 1] or standard Gaussian N(0, 1).  These one-dimensional mappings form
the marginal step of each RBIG (Rotation-Based Iterative Gaussianization)
layer.

Because every dimension is transformed independently, the Jacobian is
diagonal and its log-determinant equals the sum of per-dimension
log-derivatives::

    log|det J(x)| = sum_i log|f_i'(x_i)|

All classes expose ``fit`` / ``transform`` / ``inverse_transform`` compatible
with scikit-learn conventions.  Subclasses of
:class:`~rbig._src.base.Bijector` additionally implement
:meth:`get_log_det_jacobian` for density estimation via the
change-of-variables formula.

References
----------
Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
537-549. https://doi.org/10.1109/TNN.2011.2106511
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtri

from rbig._src.base import BaseTransform, Bijector


class MarginalUniformize(BaseTransform):
    """Transform each marginal to uniform [0, 1] using the empirical CDF.

    For each feature dimension *i*, the empirical CDF is estimated from the
    training data with a mid-point (Hazen) continuity correction::

        u_hat = F_hat_n(x) = (rank(x, X_train) + 0.5) / N

    where *rank* is the number of training samples <= x (left-sided
    ``searchsorted``) and *N* is the number of training samples.  The
    ``+0.5`` shift avoids the degenerate values 0 and 1 for in-sample
    boundary points.

    Parameters
    ----------
    bound_correct : bool, default True
        If True, clip the output to ``[eps, 1 - eps]`` to prevent exact 0
        or 1, which is useful when feeding the result into a probit or
        logit function.
    eps : float, default 1e-6
        Half-width of the clipping margin when ``bound_correct=True``.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_samples, n_features)
        Column-wise sorted training data.  Serves as empirical quantile
        nodes for both the forward transform and piecewise-linear inversion.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The mid-point empirical CDF (Hazen plotting position) is::

        F_hat_n(x) = (rank + 0.5) / N

    The inverse is approximated by piecewise-linear interpolation between
    the sorted support values and their corresponding uniform probabilities
    ``np.linspace(0, 1, N)``.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import MarginalUniformize
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> uni = MarginalUniformize().fit(X)
    >>> U = uni.transform(X)
    >>> U.shape
    (100, 2)
    >>> bool(U.min() > 0.0) and bool(U.max() < 1.0)
    True
    >>> Xr = uni.inverse_transform(U)
    >>> Xr.shape
    (100, 2)
    """

    def __init__(self, bound_correct: bool = True, eps: float = 1e-6):
        self.bound_correct = bound_correct
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalUniformize:
        """Fit the transform by storing sorted training values per feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.  Each column is sorted and stored as the empirical
            support (quantile nodes) for that feature.

        Returns
        -------
        self : MarginalUniformize
            Fitted transform instance.
        """
        # Sort each column independently to obtain empirical quantile nodes
        self.support_ = np.sort(X, axis=0)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to [0, 1] via the mid-point empirical CDF.

        Applies ``u = (rank + 0.5) / N`` to every column independently.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Uniformized data in [0, 1] (or ``[eps, 1 - eps]`` when
            ``bound_correct=True``).
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            Xt[:, i] = self._uniformize(X[:, i], self.support_[:, i])
        if self.bound_correct:
            # Clip to (eps, 1-eps) to prevent boundary issues downstream
            Xt = np.clip(Xt, self.eps, 1 - self.eps)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map uniform [0, 1] values back to the original space.

        Uses piecewise-linear interpolation between the stored sorted support
        values and their corresponding uniform probabilities
        ``np.linspace(0, 1, N)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the uniform [0, 1] space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Interpolate: uniform grid [0, 1] -> sorted training values
            Xt[:, i] = np.interp(
                X[:, i],
                np.linspace(0, 1, len(self.support_[:, i])),
                self.support_[:, i],
            )
        return Xt

    @staticmethod
    def _uniformize(x: np.ndarray, support: np.ndarray) -> np.ndarray:
        """Compute the mid-point empirical CDF for a single feature.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            New data values to evaluate the empirical CDF at.
        support : np.ndarray of shape (n_train,)
            Sorted training values used as the empirical quantile nodes.

        Returns
        -------
        u : np.ndarray of shape (n_samples,)
            Empirical CDF values: ``(rank + 0.5) / n_train``.
        """
        n = len(support)
        # Left-sided searchsorted gives the number of training points <= x
        ranks = np.searchsorted(support, x, side="left")
        # Mid-point shift (+0.5) avoids exact 0 and 1
        return (ranks + 0.5) / n


class MarginalGaussianize(BaseTransform):
    """Transform each marginal to standard Gaussian using empirical CDF + probit.

    Combines a mid-point empirical CDF estimate with the Gaussian probit
    (quantile) function Phi^{-1} to map each feature to an approximately
    standard-normal marginal::

        z = Phi ^ {-1}(F_hat_n(x))

    where ``F_hat_n(x) = (rank + 0.5) / N`` is the mid-point empirical CDF
    and ``Phi^{-1}`` is the inverse standard-normal CDF (probit).

    Parameters
    ----------
    bound_correct : bool, default True
        Clip the intermediate uniform value to ``[eps, 1 - eps]`` before
        applying the probit to prevent +/-inf outputs at the tails.
    eps : float, default 1e-6
        Clipping margin for the uniform intermediate value.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_samples, n_features)
        Column-wise sorted training data (empirical quantile nodes).
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-absolute Jacobian determinant needed for density estimation is::

        log|dz/dx| = log f_hat_n(x) - log phi(Phi^{-1}(F_hat_n(x)))

    where ``f_hat_n`` is the empirical density estimated from the spacing of
    adjacent sorted training values, and ``phi`` is the standard-normal PDF.
    This is computed in :meth:`log_det_jacobian`.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import MarginalGaussianize
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> mg = MarginalGaussianize().fit(X)
    >>> Z = mg.transform(X)
    >>> Z.shape
    (200, 3)
    >>> abs(float(Z.mean())) < 0.5
    True
    """

    def __init__(self, bound_correct: bool = True, eps: float = 1e-6):
        self.bound_correct = bound_correct
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalGaussianize:
        """Fit by storing the column-wise sorted training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to build the per-feature empirical CDF.

        Returns
        -------
        self : MarginalGaussianize
            Fitted transform instance.
        """
        # Sorted columns serve as empirical quantile nodes
        self.support_ = np.sort(X, axis=0)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to N(0, 1) via empirical CDF then probit.

        Applies ``z = Phi^{-1}(F_hat_n(x))`` column by column.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Gaussianized data; each column has approximately N(0, 1) marginal.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Step 1: empirical CDF -> uniform value in (0, 1)
            u = MarginalUniformize._uniformize(X[:, i], self.support_[:, i])
            if self.bound_correct:
                # Clip to avoid Phi^{-1}(0) = -inf or Phi^{-1}(1) = +inf
                u = np.clip(u, self.eps, 1 - self.eps)
            # Step 2: probit transform Phi^{-1}(u) -> standard normal
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map standard-normal values back to the original space.

        Applies the normal CDF Phi to obtain uniform values, then uses
        piecewise-linear interpolation through the sorted support to recover
        approximate original-space values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized (standard-normal) space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Invert probit: z -> Phi(z) in (0, 1)
            u = stats.norm.cdf(X[:, i])
            # Invert empirical CDF via linear interpolation
            Xt[:, i] = np.interp(
                u, np.linspace(0, 1, len(self.support_[:, i])), self.support_[:, i]
            )
        return Xt

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for marginal Gaussianization.

        For g(x) = Phi^{-1}(F_n(x)):
            log|dg/dx| = log f_n(x_i) - log phi(g(x_i))

        where f_n is estimated from the spacing of the empirical support.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_jac : np.ndarray of shape (n_samples,)
            Per-sample sum of per-feature log-derivatives.

        Notes
        -----
        The empirical density is approximated as::

            f_hat_n(x) ~= 1 / (N * spacing)

        where *spacing* is the gap between adjacent sorted training values
        at the location of x.  Tied values produce zero spacings; these are
        replaced by the minimum positive spacing to keep the log-density
        finite.
        """
        Xt = self.transform(X)  # Gaussianized output, shape (N, D)
        log_jac = np.zeros(X.shape[0])  # accumulator, shape (N,)
        n = self.support_.shape[0]  # number of training samples
        for i in range(self.n_features_):
            support_i = self.support_[:, i]  # sorted training values, shape (n,)
            spacings = np.diff(support_i)  # gaps between consecutive support points
            pos_sp = spacings[spacings > 0]
            min_sp = pos_sp.min() if len(pos_sp) > 0 else 1e-10
            safe_sp = np.where(spacings > 0, spacings, min_sp)
            # Pad to n elements: first element uses the first spacing
            sp_full = np.concatenate([[safe_sp[0]], safe_sp])
            ranks = np.clip(np.searchsorted(support_i, X[:, i], side="left"), 0, n - 1)
            local_spacing = sp_full[ranks]  # spacing at each sample's location
            # Empirical log-density: log f_hat_n(x) ~= -log(N * spacing)
            log_f_i = -np.log(n * local_spacing)
            # Log standard-normal PDF at Gaussianized value: log phi(z_i)
            log_phi_gi = stats.norm.logpdf(Xt[:, i])
            # Chain rule: log|dz/dx| = log f_hat_n(x) - log phi(z)
            log_jac += log_f_i - log_phi_gi
        return log_jac


class MarginalKDEGaussianize(BaseTransform):
    """Transform each marginal to Gaussian using a KDE-estimated CDF.

    A kernel density estimate (KDE) with a Gaussian kernel is fitted to each
    feature dimension.  The cumulative integral of the KDE serves as a smooth
    approximation to the marginal CDF, which is then composed with the probit
    function Phi^{-1} to Gaussianize each dimension::

        z = Phi ^ {-1}(F_KDE(x))

    where ``F_KDE(x) = integral_{-inf}^{x} f_KDE(t) dt`` and ``f_KDE`` is
    the Gaussian-kernel density estimate.

    Parameters
    ----------
    bw_method : str, float, or None, default None
        Bandwidth selection method passed to
        :class:`scipy.stats.gaussian_kde`.  ``None`` uses Scott's rule;
        ``'silverman'`` uses Silverman's rule; a scalar sets the bandwidth
        factor directly.
    eps : float, default 1e-6
        Clipping margin to prevent ``Phi^{-1}(0) = -inf`` or
        ``Phi^{-1}(1) = +inf``.

    Attributes
    ----------
    kdes_ : list of scipy.stats.gaussian_kde
        One fitted KDE object per feature dimension.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The inverse transform inverts the KDE CDF numerically via Brent's method
    (:func:`scipy.optimize.brentq`) searching in [-100, 100].  Samples
    outside this range default to 0.0.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import MarginalKDEGaussianize
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 2))
    >>> kde_g = MarginalKDEGaussianize().fit(X)
    >>> Z = kde_g.transform(X)
    >>> Z.shape
    (50, 2)
    """

    def __init__(self, bw_method: str | float | None = None, eps: float = 1e-6):
        self.bw_method = bw_method
        self.eps = eps

    def fit(self, X: np.ndarray) -> MarginalKDEGaussianize:
        """Fit a Gaussian KDE to each feature dimension.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : MarginalKDEGaussianize
            Fitted transform instance.
        """
        self.kdes_ = []
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            # Fit an independent Gaussian KDE per feature
            self.kdes_.append(stats.gaussian_kde(X[:, i], bw_method=self.bw_method))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to N(0, 1) via KDE CDF then probit.

        Computes ``z = Phi^{-1}(F_KDE(x))`` per feature using numerical
        integration of the fitted KDE.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Gaussianized data.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Integrate KDE from -inf to each sample value to get CDF
            u = np.array(
                [self.kdes_[i].integrate_box_1d(-np.inf, xi) for xi in X[:, i]]
            )
            # Clip to avoid +/-inf from the probit function
            u = np.clip(u, self.eps, 1 - self.eps)
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map standard-normal values back to the original space.

        Numerically inverts the KDE CDF via Brent's root-finding method.
        For each sample *j* and feature *i*, solves::

            F_KDE(x) = Phi(z_j)

        searching on the interval [-100, 100].

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
            Samples that fail root-finding are set to 0.0.
        """
        from scipy.optimize import brentq

        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            for j, xj in enumerate(X[:, i]):
                # Map z -> u in (0, 1) via normal CDF
                u = stats.norm.cdf(xj)
                try:
                    # Numerically invert F_KDE(x) = u via root-finding
                    Xt[j, i] = brentq(
                        lambda x, u=u, i=i: (
                            self.kdes_[i].integrate_box_1d(-np.inf, x) - u
                        ),
                        -100,
                        100,
                    )
                except ValueError:
                    # Root not found in [-100, 100]; fall back to zero
                    Xt[j, i] = 0.0
        return Xt


class QuantileGaussianizer(Bijector):
    """Gaussianize each marginal using sklearn's QuantileTransformer.

    Wraps :class:`sklearn.preprocessing.QuantileTransformer` configured with
    ``output_distribution='normal'`` to map each feature to an approximately
    standard-normal distribution.  The quantile transform is a step-function
    CDF estimate that is particularly robust to outliers.

    Parameters
    ----------
    n_quantiles : int, default 1000
        Number of quantile nodes used to define the piecewise-linear mapping.
        Capped at ``n_samples`` during ``fit`` to avoid requesting more
        quantiles than there are training points.
    random_state : int or None, default 0
        Seed for reproducible subsampling inside ``QuantileTransformer``.

    Attributes
    ----------
    qt_ : sklearn.preprocessing.QuantileTransformer
        Fitted quantile transformer with ``output_distribution='normal'``.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-absolute-Jacobian is estimated via central finite differences::

        dz_i/dx_i ~= (z_i(x + eps*e_i) - z_i(x - eps*e_i)) / (2*eps)

    with ``eps = 1e-5``.  This approximation may be inaccurate near
    discontinuities of the piecewise quantile function.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import QuantileGaussianizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 3))
    >>> qg = QuantileGaussianizer().fit(X)
    >>> Z = qg.transform(X)
    >>> Z.shape
    (200, 3)
    >>> Xr = qg.inverse_transform(Z)
    >>> Xr.shape
    (200, 3)
    """

    def __init__(self, n_quantiles: int = 1000, random_state: int | None = 0):
        self.n_quantiles = n_quantiles
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> QuantileGaussianizer:
        """Fit the quantile transformer to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : QuantileGaussianizer
            Fitted bijector instance.
        """
        from sklearn.preprocessing import QuantileTransformer

        # Cap quantile count so it cannot exceed the available samples
        n_quantiles = min(self.n_quantiles, X.shape[0])
        self.qt_ = QuantileTransformer(
            n_quantiles=n_quantiles,
            output_distribution="normal",
            random_state=self.random_state,
        )
        self.qt_.fit(X)
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the quantile transform: x -> z approximately N(0, 1).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_features)
            Gaussianized data.
        """
        return self.qt_.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the quantile transform: z -> x.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized (standard-normal) space.

        Returns
        -------
        Xr : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
        """
        return self.qt_.inverse_transform(X)

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Estimate log |det J| by finite differences on the quantile transform.

        Uses a small perturbation ``eps = 1e-5`` in each dimension::

            dz_i/dx_i ~= (z_i(x + eps*e_i) - z_i(x - eps*e_i)) / (2*eps)

        and sums the log-absolute-derivatives::

            log|det J| = sum_i log|dz_i/dx_i|

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant approximation.

        Notes
        -----
        The quantile transform is piecewise-linear; the finite-difference
        derivative equals the local slope and is exact within each segment.
        """
        eps = 1e-5
        log_det = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            dummy_plus = X.copy()
            dummy_plus[:, i] = X[:, i] + eps
            dummy_minus = X.copy()
            dummy_minus[:, i] = X[:, i] - eps
            y_plus = self.qt_.transform(dummy_plus)[:, i]
            y_minus = self.qt_.transform(dummy_minus)[:, i]
            # Central-difference derivative for dimension i
            deriv = (y_plus - y_minus) / (2 * eps)
            log_det += np.log(np.maximum(np.abs(deriv), 1e-300))
        return log_det


class KDEGaussianizer(Bijector):
    """Gaussianize each marginal using a KDE-estimated CDF and probit.

    Fits a Gaussian kernel density estimate (KDE) to each feature dimension,
    then maps samples to standard-normal values via::

        z = Phi ^ {-1}(F_KDE(x))

    where ``F_KDE(x) = integral_{-inf}^{x} f_KDE(t) dt`` is the smooth
    KDE-based CDF and ``Phi^{-1}`` is the Gaussian probit (inverse CDF).

    Parameters
    ----------
    bw_method : str, float, or None, default None
        Bandwidth selection passed to :class:`scipy.stats.gaussian_kde`.
        ``None`` uses Scott's rule; ``'silverman'`` uses Silverman's rule;
        a scalar sets the smoothing factor directly.
    eps : float, default 1e-6
        Clipping margin applied to the CDF value before the probit to
        prevent ``Phi^{-1}(0) = -inf`` or ``Phi^{-1}(1) = +inf``.

    Attributes
    ----------
    kdes_ : list of scipy.stats.gaussian_kde
        One fitted KDE per feature dimension.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-det-Jacobian uses the analytic KDE density::

        log|dz/dx| = log f_KDE(x) - log phi(z)

    where ``phi`` is the standard-normal PDF evaluated at the Gaussianized
    value ``z = Phi^{-1}(F_KDE(x))``.

    The inverse transform uses Brent's root-finding algorithm to numerically
    invert ``F_KDE(x) = Phi(z)`` on the interval [-100, 100].

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import KDEGaussianizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 2))
    >>> kde = KDEGaussianizer().fit(X)
    >>> Z = kde.transform(X)
    >>> Z.shape
    (100, 2)
    >>> ldj = kde.get_log_det_jacobian(X)
    >>> ldj.shape
    (100,)
    """

    def __init__(self, bw_method: str | float | None = None, eps: float = 1e-6):
        self.bw_method = bw_method
        self.eps = eps

    def fit(self, X: np.ndarray) -> KDEGaussianizer:
        """Fit a Gaussian KDE to each feature dimension.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KDEGaussianizer
            Fitted bijector instance.
        """
        self.kdes_ = []
        self.n_features_ = X.shape[1]
        for i in range(self.n_features_):
            # Independent KDE per feature
            self.kdes_.append(stats.gaussian_kde(X[:, i], bw_method=self.bw_method))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to N(0, 1) via KDE CDF then probit.

        Computes ``z = Phi^{-1}(F_KDE(x))`` for each feature independently.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Gaussianized data with approximately standard-normal marginals.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Numerical integration of KDE from -inf to each sample value
            u = np.array(
                [self.kdes_[i].integrate_box_1d(-np.inf, xi) for xi in X[:, i]]
            )
            # Clip CDF values away from boundaries before probit
            u = np.clip(u, self.eps, 1 - self.eps)
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map standard-normal values back to the original space.

        Numerically inverts ``F_KDE(x) = Phi(z)`` using Brent's method
        on the interval [-100, 100] per sample and feature.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized (standard-normal) space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
            Samples for which root-finding fails default to 0.0.
        """
        from scipy.optimize import brentq

        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            for j, xj in enumerate(X[:, i]):
                # Map z -> u = Phi(z) in (0, 1)
                u = stats.norm.cdf(xj)
                try:
                    # Find x such that F_KDE(x) = u
                    Xt[j, i] = brentq(
                        lambda x, u=u, i=i: (
                            self.kdes_[i].integrate_box_1d(-np.inf, x) - u
                        ),
                        -100,
                        100,
                    )
                except ValueError:
                    # Root not bracketed in [-100, 100]; use zero as fallback
                    Xt[j, i] = 0.0
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute log |det J| using the analytic KDE density.

        Because the Jacobian is diagonal (each feature transformed
        independently)::

            log|det J| = sum_i log|dz_i/dx_i|
                       = sum_i [log f_KDE(x_i) - log phi(z_i)]

        where ``phi`` is the standard-normal PDF evaluated at
        ``z_i = Phi^{-1}(F_KDE(x_i))``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant.
        """
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            # Evaluate KDE density (used as the empirical marginal PDF)
            pdf = self.kdes_[i](X[:, i])
            # Compute KDE CDF via numerical integration
            u = np.array(
                [self.kdes_[i].integrate_box_1d(-np.inf, xi) for xi in X[:, i]]
            )
            u = np.clip(u, self.eps, 1 - self.eps)
            g = ndtri(u)  # Gaussianized value z = Phi^{-1}(u)
            log_phi = stats.norm.logpdf(g)  # log phi(z)
            # log|dz/dx| = log f_KDE(x) - log phi(z)
            log_det += np.log(np.maximum(pdf, 1e-300)) - log_phi
        return log_det


class GMMGaussianizer(Bijector):
    """Gaussianize each marginal using a Gaussian Mixture Model (GMM) CDF.

    Fits a univariate GMM with ``n_components`` Gaussian components to each
    feature dimension, then maps samples to standard-normal values via the
    analytic GMM CDF::

        F_GMM(x) = sum_k w_k * Phi((x - mu_k) / sigma_k)

    followed by the probit function::

        z = Phi ^ {-1}(F_GMM(x))

    where ``Phi`` is the standard-normal CDF, ``w_k`` are mixture weights,
    and ``mu_k``, ``sigma_k`` are the component means and standard deviations.

    Parameters
    ----------
    n_components : int, default 5
        Number of mixture components.  Capped at
        ``max(1, min(n_components, n_samples // 5, n_samples))`` during
        ``fit`` to avoid over-fitting on small data sets.
    random_state : int or None, default 0
        Seed for reproducible GMM initialisation.

    Attributes
    ----------
    gmms_ : list of sklearn.mixture.GaussianMixture
        One fitted 1-D GMM per feature dimension.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-det-Jacobian uses the analytic GMM density::

        log|dz/dx| = log f_GMM(x) - log phi(z)

    where ``f_GMM(x) = sum_k w_k * phi((x - mu_k) / sigma_k) / sigma_k``
    is the GMM PDF and ``phi`` is the standard-normal PDF.

    The inverse transform numerically inverts the GMM CDF via Brent's
    method on [-50, 50]; samples outside this range default to 0.0.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import GMMGaussianizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((200, 2))
    >>> gmm = GMMGaussianizer(n_components=3).fit(X)
    >>> Z = gmm.transform(X)
    >>> Z.shape
    (200, 2)
    >>> ldj = gmm.get_log_det_jacobian(X)
    >>> ldj.shape
    (200,)
    """

    def __init__(self, n_components: int = 5, random_state: int | None = 0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: np.ndarray) -> GMMGaussianizer:
        """Fit a univariate GMM to each feature dimension.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : GMMGaussianizer
            Fitted bijector instance.

        Notes
        -----
        The number of mixture components is capped at
        ``max(1, min(n_components, n_samples // 5, n_samples))`` to avoid
        over-fitting when ``n_samples`` is small.
        """
        from sklearn.mixture import GaussianMixture

        self.gmms_ = []
        self.n_features_ = X.shape[1]
        n_samples = X.shape[0]
        # Cap n_components to avoid GMMs with more components than data points
        n_components = max(1, min(self.n_components, n_samples // 5, n_samples))
        for i in range(self.n_features_):
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=self.random_state,
            )
            # Reshape to (n_samples, 1) as required by sklearn GMM API
            gmm.fit(X[:, i : i + 1])
            self.gmms_.append(gmm)
        return self

    def _cdf(self, gmm, x: np.ndarray) -> np.ndarray:
        """Compute the GMM CDF at points x (1-D).

        Evaluates the mixture CDF::

            F_GMM(x) = sum_k w_k * Phi((x - mu_k) / sigma_k)

        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Fitted 1-D GMM.
        x : np.ndarray of shape (n_samples,)
            Query points.

        Returns
        -------
        cdf : np.ndarray of shape (n_samples,)
            GMM CDF values in [0, 1].
        """
        weights = gmm.weights_  # mixture weights, shape (K,)
        means = gmm.means_.ravel()  # component means, shape (K,)
        stds = np.sqrt(gmm.covariances_.ravel())  # component stds, shape (K,)
        cdf = np.zeros_like(x, dtype=float)
        for w, mu, sigma in zip(weights, means, stds, strict=False):
            # Weighted sum of normal CDFs: w_k * Phi((x - mu_k) / sigma_k)
            cdf += w * stats.norm.cdf(x, loc=mu, scale=sigma)
        return cdf

    def _pdf(self, gmm, x: np.ndarray) -> np.ndarray:
        """Compute the GMM PDF at points x (1-D).

        Evaluates the mixture density::

            f_GMM(x) = sum_k w_k * phi((x - mu_k) / sigma_k) / sigma_k

        Parameters
        ----------
        gmm : sklearn.mixture.GaussianMixture
            Fitted 1-D GMM.
        x : np.ndarray of shape (n_samples,)
            Query points.

        Returns
        -------
        pdf : np.ndarray of shape (n_samples,)
            GMM PDF values (>= 0).
        """
        weights = gmm.weights_
        means = gmm.means_.ravel()
        stds = np.sqrt(gmm.covariances_.ravel())
        pdf = np.zeros_like(x, dtype=float)
        for w, mu, sigma in zip(weights, means, stds, strict=False):
            # Weighted sum of normal PDFs: w_k * phi((x - mu_k) / sigma_k) / sigma_k
            pdf += w * stats.norm.pdf(x, loc=mu, scale=sigma)
        return pdf

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to N(0, 1) via GMM CDF then probit.

        Applies ``z = Phi^{-1}(F_GMM(x))`` to each feature independently.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Gaussianized data with approximately standard-normal marginals.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Evaluate analytic GMM CDF
            u = self._cdf(self.gmms_[i], X[:, i])
            # Clip away from boundaries before probit
            u = np.clip(u, 1e-6, 1 - 1e-6)
            Xt[:, i] = ndtri(u)
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map standard-normal values back to the original space.

        Numerically inverts ``F_GMM(x) = Phi(z)`` per sample via Brent's
        method on the interval [-50, 50].

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized (standard-normal) space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
            Samples for which root-finding fails default to 0.0.
        """
        from scipy.optimize import brentq

        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            for j, xj in enumerate(X[:, i]):
                # Map z -> u = Phi(z) in (0, 1)
                u = stats.norm.cdf(xj)
                try:
                    # Numerically solve F_GMM(x) = u for x
                    Xt[j, i] = brentq(
                        lambda x, u=u, i=i: (
                            self._cdf(self.gmms_[i], np.array([x]))[0] - u
                        ),
                        -50,
                        50,
                    )
                except ValueError:
                    # Root not found in [-50, 50]; fall back to zero
                    Xt[j, i] = 0.0
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute log |det J| using the analytic GMM density.

        Because the Jacobian is diagonal (each feature transformed
        independently)::

            log|det J| = sum_i log|dz_i/dx_i|
                       = sum_i [log f_GMM(x_i) - log phi(z_i)]

        where ``z_i = Phi^{-1}(F_GMM(x_i))`` and ``phi`` is the
        standard-normal PDF.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant.
        """
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            # Evaluate GMM CDF and clip to avoid probit boundary issues
            u = self._cdf(self.gmms_[i], X[:, i])
            u = np.clip(u, 1e-6, 1 - 1e-6)
            g = ndtri(u)  # z = Phi^{-1}(F_GMM(x))
            pdf = self._pdf(self.gmms_[i], X[:, i])  # f_GMM(x)
            log_phi = stats.norm.logpdf(g)  # log phi(z)
            # log|dz/dx| = log f_GMM(x) - log phi(z)
            log_det += np.log(np.maximum(pdf, 1e-300)) - log_phi
        return log_det


class SplineGaussianizer(Bijector):
    """Gaussianize each marginal using monotone PCHIP spline interpolation.

    Estimates the marginal CDF from empirical quantiles and fits a
    shape-preserving (monotone) cubic Hermite spline (PCHIP) from
    original-space quantile values to the corresponding Gaussian quantiles.
    The forward transform is::

        z = S(x)

    where ``S`` is the fitted :class:`scipy.interpolate.PchipInterpolator`
    mapping data values to standard-normal quantiles.  Because PCHIP
    preserves monotonicity, the mapping is guaranteed to be invertible.

    Parameters
    ----------
    n_quantiles : int, default 200
        Number of quantile nodes used to fit the splines.  Capped at
        ``n_samples`` when fewer training samples are available.
    eps : float, default 1e-6
        Clipping margin applied to the Gaussian quantile grid to keep
        the spline endpoints finite (avoids +/-inf at boundary quantiles).

    Attributes
    ----------
    splines_ : list of scipy.interpolate.PchipInterpolator
        Forward splines (x -> z) per feature, mapping original-space
        values to standard-normal quantiles.
    inv_splines_ : list of scipy.interpolate.PchipInterpolator
        Inverse splines (z -> x) per feature, mapping Gaussian quantiles
        back to original-space values.
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-det-Jacobian uses the analytic first derivative of the spline::

        log|dz/dx| = log|S'(x)|

    where ``S'`` is the first derivative of the PCHIP forward spline,
    evaluated via ``spline(x, 1)`` (the derivative-order argument).

    Duplicate x-values (arising from discrete or constant features) are
    removed before fitting to ensure strict monotonicity.

    References
    ----------
    Laparra, V., Camps-Valls, G., & Malo, J. (2011). Iterative Gaussianization:
    from ICA to Random Rotations. *IEEE Transactions on Neural Networks*, 22(4),
    537-549.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.marginal import SplineGaussianizer
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 3))
    >>> sg = SplineGaussianizer(n_quantiles=100).fit(X)
    >>> Z = sg.transform(X)
    >>> Z.shape
    (300, 3)
    >>> ldj = sg.get_log_det_jacobian(X)
    >>> ldj.shape
    (300,)
    """

    def __init__(self, n_quantiles: int = 200, eps: float = 1e-6):
        self.n_quantiles = n_quantiles
        self.eps = eps

    def fit(self, X: np.ndarray) -> SplineGaussianizer:
        """Fit forward and inverse PCHIP splines for each feature.

        For each dimension, ``n_quantiles`` evenly-spaced probability levels
        are mapped to their empirical quantile values in the data, and the
        corresponding Gaussian quantile values ``Phi^{-1}(p)`` are computed.
        PCHIP interpolants are then fitted in both directions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : SplineGaussianizer
            Fitted bijector instance.
        """
        from scipy.interpolate import PchipInterpolator

        self.splines_ = []
        self.inv_splines_ = []
        self.n_features_ = X.shape[1]
        # Use at most n_samples quantile nodes
        n_q = min(self.n_quantiles, X.shape[0])
        # Probability grid: n_q evenly-spaced points in [0, 1]
        quantiles = np.linspace(0, 1, n_q)
        # Corresponding Gaussian quantiles Phi^{-1}(p), clipped away from +/-inf
        g_q = ndtri(np.clip(quantiles, self.eps, 1 - self.eps))
        for i in range(self.n_features_):
            xi_sorted = np.sort(X[:, i])
            # Empirical quantile values at each probability level
            x_q = np.quantile(xi_sorted, quantiles)
            # Remove duplicate x values so PchipInterpolator gets a strictly
            # increasing sequence (duplicates arise with discrete/tied data).
            x_q_u, idx = np.unique(x_q, return_index=True)
            g_q_u = g_q[idx]
            self.splines_.append(PchipInterpolator(x_q_u, g_q_u))
            self.inv_splines_.append(PchipInterpolator(g_q_u, x_q_u))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the forward spline map: x -> z = S(x).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data in the original space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Gaussianized data with approximately standard-normal marginals.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Evaluate the forward PCHIP spline at the input values
            Xt[:, i] = self.splines_[i](X[:, i])
        return Xt

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the inverse spline map: z -> x = S^{-1}(z).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data in the Gaussianized space.

        Returns
        -------
        Xt : np.ndarray of shape (n_samples, n_features)
            Data approximately recovered in the original input space.
        """
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            # Evaluate the inverse PCHIP spline at the Gaussian values
            Xt[:, i] = self.inv_splines_[i](X[:, i])
        return Xt

    def get_log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Compute log |det J| using the analytic spline first derivative.

        Because the Jacobian is diagonal::

            log|det J| = sum_i log|S'(x_i)|

        where ``S'`` is the first derivative of the PCHIP forward spline,
        evaluated via ``spline(x, 1)`` (the derivative order argument).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input points at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_det : np.ndarray of shape (n_samples,)
            Per-sample log absolute determinant.
        """
        log_det = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            deriv = self.splines_[i](X[:, i], 1)  # first derivative
            log_det += np.log(np.maximum(np.abs(deriv), 1e-300))
        return log_det
