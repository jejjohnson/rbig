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
from rbig._src.spline import RQSpline


def make_cdf_monotonic(cdf: np.ndarray) -> np.ndarray:
    """Enforce monotonicity on an empirical CDF via running maximum.

    Replaces each value with the cumulative maximum so that the result is
    non-decreasing.

    Parameters
    ----------
    cdf : np.ndarray
        1-D or 2-D array of CDF values.  For 2-D input, monotonicity is
        enforced independently along each column (axis 0).

    Returns
    -------
    cdf_monotonic : np.ndarray
        Array of same shape as *cdf* with non-decreasing values along axis 0.
    """
    return np.maximum.accumulate(cdf, axis=0)


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

    def __init__(
        self,
        bound_correct: bool = True,
        eps: float = 1e-6,
        pdf_extension: float = 0.0,
        pdf_resolution: int = 1000,
    ):
        self.bound_correct = bound_correct
        self.eps = eps
        self.pdf_extension = pdf_extension
        self.pdf_resolution = pdf_resolution

    def fit(self, X: np.ndarray, y=None) -> MarginalUniformize:
        """Fit the transform by storing sorted training values per feature.

        When ``pdf_extension > 0``, a histogram-based CDF pipeline is used
        instead of the default empirical CDF.  This extends the support by
        ``pdf_extension`` percent of the data range and builds an interpolated,
        monotonic CDF on a grid of ``pdf_resolution`` points.

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
        self.n_features_ = X.shape[1]

        if self.pdf_extension > 0:
            self._fit_histogram_cdf(X)
        else:
            # Sort each column independently to obtain empirical quantile nodes
            self.support_ = np.sort(X, axis=0)
        return self

    def _fit_histogram_cdf(self, X: np.ndarray) -> None:
        """Build per-feature histogram CDF with extended support."""
        n_samples = X.shape[0]
        self.cdf_support_ = []
        self.cdf_values_ = []
        self.pdf_support_ = []
        self.pdf_values_ = []

        for i in range(self.n_features_):
            xi = X[:, i]
            x_min, x_max = xi.min(), xi.max()

            # Handle constant-valued feature: trivial linear CDF
            if x_min == x_max:
                support = np.array([x_min - 1.0, x_min, x_min + 1.0])
                cdf_vals = np.array([0.0, 0.5, 1.0])
                pdf_sup = np.array([x_min - 1.0, x_min, x_min + 1.0])
                pdf_vals = np.array([0.0, 1.0, 0.0])
                self.cdf_support_.append(support)
                self.cdf_values_.append(cdf_vals)
                self.pdf_support_.append(pdf_sup)
                self.pdf_values_.append(pdf_vals)
                continue

            support_ext = (self.pdf_extension / 100) * abs(x_max - x_min)

            # Build histogram bins: sqrt(n) + 1 edges
            n_bin_edges = int(np.sqrt(float(n_samples)) + 1)
            bin_edges = np.linspace(x_min, x_max, n_bin_edges)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            counts, _ = np.histogram(xi, bin_edges)
            bin_size = bin_edges[1] - bin_edges[0]

            # Empirical PDF with zero-padded edges
            pdf_support = np.concatenate(
                [
                    [bin_centers[0] - bin_size],
                    bin_centers,
                    [bin_centers[-1] + bin_size],
                ]
            )
            empirical_pdf = np.concatenate(
                [
                    [0.0],
                    counts / (np.sum(counts) * bin_size),
                    [0.0],
                ]
            )

            # CDF from cumulative counts with extended support
            c_sum = np.cumsum(counts)
            cdf = (1 - 1 / n_samples) * c_sum / n_samples
            incr_bin = bin_size / 2

            new_bin_edges = np.concatenate(
                [
                    [x_min - support_ext],
                    [x_min],
                    bin_centers + incr_bin,
                    [x_max + support_ext + incr_bin],
                ]
            )
            extended_cdf = np.concatenate(
                [
                    [0.0],
                    [1.0 / n_samples],
                    cdf,
                    [1.0],
                ]
            )

            # Interpolate onto fine grid, enforce monotonicity, normalize
            new_support = np.linspace(
                new_bin_edges[0], new_bin_edges[-1], self.pdf_resolution
            )
            learned_cdf = np.interp(new_support, new_bin_edges, extended_cdf)
            uniform_cdf = make_cdf_monotonic(learned_cdf)
            uniform_cdf /= uniform_cdf.max()

            self.cdf_support_.append(new_support)
            self.cdf_values_.append(uniform_cdf)
            self.pdf_support_.append(pdf_support)
            self.pdf_values_.append(empirical_pdf)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map each feature to [0, 1] via the mid-point empirical CDF.

        Applies ``u = (rank + 0.5) / N`` to every column independently.
        When ``pdf_extension > 0``, uses interpolation with the stored
        histogram-based CDF grid instead.

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
        if self.pdf_extension > 0:
            for i in range(self.n_features_):
                Xt[:, i] = np.interp(X[:, i], self.cdf_support_[i], self.cdf_values_[i])
        else:
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
        ``np.linspace(0, 1, N)``.  When ``pdf_extension > 0``, uses the
        inverted histogram CDF grid instead.

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
        if self.pdf_extension > 0:
            for i in range(self.n_features_):
                # Ensure strictly increasing xp for np.interp by
                # dropping duplicate CDF values
                cdf_vals = self.cdf_values_[i]
                cdf_sup = self.cdf_support_[i]
                unique_mask = np.concatenate([[True], np.diff(cdf_vals) > 0])
                Xt[:, i] = np.interp(
                    X[:, i], cdf_vals[unique_mask], cdf_sup[unique_mask]
                )
        else:
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
    n_quantiles : int or None, default None
        Memory cap for the stored empirical quantile nodes.  ``None`` keeps
        the full sorted training columns (``O(n·d)`` fitted memory); an
        integer stores that many evenly spaced empirical quantiles instead
        (``O(n_quantiles·d)``), like sklearn's ``QuantileTransformer``.
    tail : {None, "gaussian", "pareto"}, default None
        Parametric tail extension beyond the ``q_tail`` quantiles.
        ``None`` keeps the classic clipping behavior (desired for anomaly
        scoring).  ``"gaussian"`` matches a normal through two empirical
        quantiles per side (exact for truly Gaussian data);``"pareto"``
        fits a generalized Pareto to the seam exceedances.  Both are
        C⁰-continuous at the seam with a small C¹ discontinuity.
    q_tail : float, default 0.95
        Quantile beyond which the parametric tail replaces the empirical
        CDF (mirrored below at ``1 - q_tail``).
    dither : bool, default False
        Break ties in discrete/ordinal features with seeded uniform jitter
        at 1% of the feature's minimum positive gap, applied consistently
        at fit and transform time.
    random_state : int or None, default None
        Seed for the dithering jitter.

    Attributes
    ----------
    support_ : np.ndarray of shape (n_nodes, n_features)
        Per-feature empirical quantile nodes (sorted training columns, or
        the ``n_quantiles`` grid when capped).
    n_features_ : int
        Number of feature dimensions seen during ``fit``.
    constant_mask_ : np.ndarray of shape (n_features,)
        True for zero-variance features, which use a centred identity map.
    tails_ : list of dict or None
        Per-feature parametric tail specifications (when ``tail`` is set).

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

    def __init__(
        self,
        bound_correct: bool = True,
        eps: float = 1e-6,
        n_quantiles: int | None = None,
        tail: str | None = None,
        q_tail: float = 0.95,
        dither: bool = False,
        random_state: int | None = None,
    ):
        self.bound_correct = bound_correct
        self.eps = eps
        self.n_quantiles = n_quantiles
        self.tail = tail
        self.q_tail = q_tail
        self.dither = dither
        self.random_state = random_state

    def _dither_jitter(self, X: np.ndarray) -> np.ndarray:
        """Seeded uniform jitter at 1% of each feature's min positive gap."""
        rng = np.random.default_rng(self.random_state)
        jitter = rng.uniform(-0.5, 0.5, size=X.shape)
        return X + jitter * self.dither_scale_[None, :]

    def fit(self, X: np.ndarray, y=None) -> MarginalGaussianize:
        """Fit by storing per-feature empirical quantile nodes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data used to build the per-feature empirical CDF.

        Returns
        -------
        self : MarginalGaussianize
            Fitted transform instance.
        """
        X = np.asarray(X, dtype=float)
        self.n_features_ = X.shape[1]

        # Dither scale: 1% of the min positive gap per feature (0 disables).
        scales = np.zeros(self.n_features_)
        if self.dither:
            for i in range(self.n_features_):
                gaps = np.diff(np.unique(X[:, i]))
                scales[i] = 0.01 * gaps.min() if gaps.size else 0.0
        self.dither_scale_ = scales  # (n_features,)
        if self.dither:
            X = self._dither_jitter(X)

        # Zero-variance features fall back to a centred identity map.
        self.constant_mask_ = X.max(axis=0) == X.min(axis=0)  # (n_features,)
        self.offsets_ = np.where(self.constant_mask_, X.mean(axis=0), 0.0)
        if self.constant_mask_.any():
            import warnings

            idx = np.flatnonzero(self.constant_mask_).tolist()
            warnings.warn(
                f"Features {idx} have zero variance; using an identity "
                "(centred) marginal for them.",
                UserWarning,
                stacklevel=2,
            )

        # Quantile nodes: full sorted columns, or a memory-capped grid of
        # n_quantiles evenly spaced empirical quantiles (QuantileTransformer
        # style) — fitted memory drops from O(n·d) to O(n_quantiles·d).
        if self.n_quantiles is not None and self.n_quantiles < X.shape[0]:
            probs = np.linspace(0.0, 1.0, self.n_quantiles)
            self.support_ = np.quantile(X, probs, axis=0)
        else:
            self.support_ = np.sort(X, axis=0)

        self.kdes_ = [
            None
            if self.constant_mask_[i]
            else stats.gaussian_kde(self.support_[:, i].copy())
            for i in range(self.n_features_)
        ]

        if self.tail is not None:
            if self.tail not in ("gaussian", "pareto"):
                raise ValueError(
                    f"Unknown tail {self.tail!r}. Use None, 'gaussian', or 'pareto'."
                )
            self._fit_tails(X)
        return self

    def _fit_tails(self, X: np.ndarray) -> None:
        """Fit per-feature parametric tails beyond the ``q_tail`` quantiles.

        ``"gaussian"`` matches a normal through two empirical quantiles per
        side (C⁰-continuous at the seam by construction; recovers the exact
        probit map for truly Gaussian data).  ``"pareto"`` fits a
        generalized Pareto to the seam exceedances (falling back to the
        Gaussian tail when fewer than 5 exceedances exist).
        """
        q_hi, q_lo = self.q_tail, 1.0 - self.q_tail
        self.tails_ = []
        for i in range(self.n_features_):
            if self.constant_mask_[i]:
                self.tails_.append(None)
                continue
            col = X[:, i]
            spec: dict = {"q_hi": q_hi, "q_lo": q_lo}
            spec["x_hi"] = float(np.quantile(col, q_hi))
            spec["x_lo"] = float(np.quantile(col, q_lo))
            for side, x_seam, q_seam, q_mid in (
                ("hi", spec["x_hi"], q_hi, (1.0 + q_hi) / 2.0),
                ("lo", spec["x_lo"], q_lo, q_lo / 2.0),
            ):
                kind = self.tail
                if kind == "pareto":
                    exceed = (
                        col[col > x_seam] - x_seam
                        if side == "hi"
                        else x_seam - col[col < x_seam]
                    )
                    if exceed.size >= 5:
                        c, _loc, scale = stats.genpareto.fit(exceed, floc=0.0)
                        if c < 0.0:
                            # A negative shape gives a *bounded* tail (infinite
                            # log-density beyond the endpoint).  Clamp to the
                            # exponential boundary case, scale by its MLE.
                            c, scale = 0.0, float(exceed.mean())
                        spec[side] = {
                            "kind": "pareto",
                            "c": float(c),
                            "scale": float(scale),
                        }
                        continue
                    kind = "gaussian"  # too few exceedances: fall back
                # Gaussian tail via two-point quantile matching.
                x_mid = float(np.quantile(col, q_mid))
                z_seam, z_mid = ndtri(q_seam), ndtri(q_mid)
                sigma = (x_mid - x_seam) / (z_mid - z_seam)
                sigma = max(sigma, 1e-12)
                mu = x_seam - sigma * z_seam
                spec[side] = {
                    "kind": "gaussian",
                    "mu": float(mu),
                    "sigma": float(sigma),
                }
            self.tails_.append(spec)

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
        X = np.asarray(X, dtype=float)
        if self.dither:
            X = self._dither_jitter(X)
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            if self.constant_mask_[i]:
                # Zero-variance feature: centred identity map.
                Xt[:, i] = X[:, i] - self.offsets_[i]
                continue
            # Step 1: empirical CDF -> uniform value in (0, 1)
            u = MarginalUniformize._uniformize(X[:, i], self.support_[:, i])
            if self.bound_correct:
                # Clip to avoid Phi^{-1}(0) = -inf or Phi^{-1}(1) = +inf
                u = np.clip(u, self.eps, 1 - self.eps)
            # Step 2: probit transform Phi^{-1}(u) -> standard normal
            Xt[:, i] = ndtri(u)
            if self.tail is not None:
                # Override tail regions with the (unclipped) parametric maps.
                Xt[:, i] = self._tail_probit(X[:, i], Xt[:, i], i)
        return Xt

    def _tail_probit(self, x: np.ndarray, z: np.ndarray, i: int) -> np.ndarray:
        """Replace probit values beyond the seams with the parametric tails.

        For the Gaussian tail the composition ``Phi^{-1}(Phi((x-mu)/sigma))``
        collapses to ``(x-mu)/sigma`` exactly, so arbitrarily far inputs map
        to finite, strictly increasing latents with no clipping.
        """
        spec = self.tails_[i]
        z = z.copy()
        hi = x > spec["x_hi"]
        lo = x < spec["x_lo"]
        if hi.any():
            t = spec["hi"]
            if t["kind"] == "gaussian":
                z[hi] = (x[hi] - t["mu"]) / t["sigma"]
            else:  # pareto: F(x) = q_hi + (1-q_hi) * F_gpd(x - x_hi)
                y = x[hi] - spec["x_hi"]
                u = spec["q_hi"] + (1.0 - spec["q_hi"]) * stats.genpareto.cdf(
                    y, t["c"], scale=t["scale"]
                )
                z[hi] = ndtri(np.clip(u, 1e-300, 1.0 - 1e-16))
        if lo.any():
            t = spec["lo"]
            if t["kind"] == "gaussian":
                z[lo] = (x[lo] - t["mu"]) / t["sigma"]
            else:  # pareto: F(x) = q_lo * SF_gpd(x_lo - x)
                y = spec["x_lo"] - x[lo]
                u = spec["q_lo"] * stats.genpareto.sf(y, t["c"], scale=t["scale"])
                z[lo] = ndtri(np.clip(u, 1e-300, 1.0 - 1e-16))
        return z

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
        X = np.asarray(X, dtype=float)
        Xt = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            if self.constant_mask_[i]:
                Xt[:, i] = X[:, i] + self.offsets_[i]
                continue
            # Invert probit: z -> Phi(z) in (0, 1)
            u = stats.norm.cdf(X[:, i])
            # Invert empirical CDF via linear interpolation
            Xt[:, i] = np.interp(
                u, np.linspace(0, 1, len(self.support_[:, i])), self.support_[:, i]
            )
            if self.tail is not None:
                Xt[:, i] = self._tail_ppf(X[:, i], u, Xt[:, i], i)
        return Xt

    def _tail_ppf(
        self, z: np.ndarray, u: np.ndarray, x: np.ndarray, i: int
    ) -> np.ndarray:
        """Replace inverse-CDF values beyond the seams with the tail inverses.

        Gaussian tails invert directly from the latent ``z`` (``x = mu +
        sigma * z``), so the inverse stays exact even where ``Phi(z)``
        saturates to 1 in float64.
        """
        spec = self.tails_[i]
        x = x.copy()
        hi = u > spec["q_hi"]
        lo = u < spec["q_lo"]
        if hi.any():
            t = spec["hi"]
            if t["kind"] == "gaussian":
                x[hi] = t["mu"] + t["sigma"] * z[hi]
            else:
                frac = (u[hi] - spec["q_hi"]) / (1.0 - spec["q_hi"])
                frac = np.clip(frac, 0.0, 1.0 - 1e-16)
                x[hi] = spec["x_hi"] + stats.genpareto.ppf(
                    frac, t["c"], scale=t["scale"]
                )
        if lo.any():
            t = spec["lo"]
            if t["kind"] == "gaussian":
                x[lo] = t["mu"] + t["sigma"] * z[lo]
            else:
                frac = 1.0 - u[lo] / spec["q_lo"]
                frac = np.clip(frac, 0.0, 1.0 - 1e-16)
                x[lo] = spec["x_lo"] - stats.genpareto.ppf(
                    frac, t["c"], scale=t["scale"]
                )
        return x

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for marginal Gaussianization.

        For g(x) = Phi^{-1}(F_n(x)):
            log|dg/dx| = log f_n(x_i) - log phi(g(x_i))

        where f_n is estimated from a Gaussian KDE fitted to the training
        data for each feature.

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
        The empirical density is approximated via a Gaussian KDE (one per
        feature) fitted during :meth:`fit`.  Bandwidth is selected
        automatically using Scott's rule (the default in
        :func:`scipy.stats.gaussian_kde`).  The KDE objects are cached
        in ``self.kdes_`` so that ``log_det_jacobian`` and repeated calls
        to ``_per_feature_log_deriv`` do not re-fit the KDEs.
        """
        return np.sum(self._per_feature_log_deriv(X), axis=1)

    def _per_feature_log_deriv(
        self, X: np.ndarray, return_transform: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Per-feature log |dz_i/dx_i| via cached KDE density estimates.

        Uses the per-feature Gaussian KDEs stored in ``self.kdes_`` (fitted
        during :meth:`fit` with Scott's rule bandwidth) to evaluate the
        marginal density f_n(x_i) at each query point.  The log-derivative
        is then ``log f_n(x_i) - log phi(z_i)`` where ``z_i`` is the
        Gaussianized value.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data at which to evaluate the per-feature log-derivatives.
        return_transform : bool, default False
            If True, also return the Gaussianized output to avoid recomputing.

        Returns
        -------
        log_derivs : np.ndarray of shape (n_samples, n_features)
            Per-feature log |dz_i/dx_i| for each sample.
        Xt : np.ndarray of shape (n_samples, n_features)
            Only returned when ``return_transform=True``.
        """
        X = np.asarray(X, dtype=float)
        Xt = self.transform(X)  # Gaussianized output, shape (N, D)
        if self.dither:
            # Evaluate densities at the same jittered inputs transform saw.
            X = self._dither_jitter(X)
        log_derivs = np.zeros_like(X, dtype=float)
        for i in range(self.n_features_):
            if self.constant_mask_[i]:
                # Identity marginal: log|dz/dx| = 0.
                continue
            # KDE-based density estimate for feature i
            # After pickle with readonly memmap, the KDE's internal arrays
            # may be read-only.  Re-create the KDE from a writable copy of
            # the dataset if necessary.
            kde = self.kdes_[i]
            if not kde.dataset.flags.writeable:
                kde = stats.gaussian_kde(kde.dataset.copy())
                self.kdes_[i] = kde
            xi = np.ascontiguousarray(X[:, i])
            log_f_i = np.log(np.maximum(kde(xi), 1e-300))
            if self.tail is not None:
                log_f_i = self._tail_log_density(xi, log_f_i, i)
            # Log standard-normal PDF at Gaussianized value: log phi(z_i)
            log_phi_gi = stats.norm.logpdf(Xt[:, i])
            # Chain rule: log|dz/dx| = log f(x) - log phi(z)
            log_derivs[:, i] = log_f_i - log_phi_gi
        if return_transform:
            return log_derivs, Xt
        return log_derivs

    def _tail_log_density(self, x: np.ndarray, log_f: np.ndarray, i: int) -> np.ndarray:
        """Replace log-density values beyond the seams with the tail densities."""
        spec = self.tails_[i]
        log_f = log_f.copy()
        hi = x > spec["x_hi"]
        lo = x < spec["x_lo"]
        if hi.any():
            t = spec["hi"]
            if t["kind"] == "gaussian":
                log_f[hi] = stats.norm.logpdf(x[hi], loc=t["mu"], scale=t["sigma"])
            else:
                y = x[hi] - spec["x_hi"]
                log_f[hi] = np.log1p(-spec["q_hi"]) + stats.genpareto.logpdf(
                    y, t["c"], scale=t["scale"]
                )
        if lo.any():
            t = spec["lo"]
            if t["kind"] == "gaussian":
                log_f[lo] = stats.norm.logpdf(x[lo], loc=t["mu"], scale=t["sigma"])
            else:
                y = spec["x_lo"] - x[lo]
                log_f[lo] = np.log(spec["q_lo"]) + stats.genpareto.logpdf(
                    y, t["c"], scale=t["scale"]
                )
        return log_f

    def to_dict(self) -> dict:
        """Serialize the fitted marginal to a dict of plain arrays.

        Returns
        -------
        state : dict
            Fitted state; consumed by :meth:`from_dict`.  The per-feature
            KDEs are rebuilt deterministically from ``support_``.
        """
        return {
            "class": type(self).__name__,
            "params": self.get_params(deep=False),
            "support_": np.asarray(self.support_),
            "n_features_": int(self.n_features_),
            "constant_mask_": np.asarray(self.constant_mask_),
            "offsets_": np.asarray(self.offsets_),
            "dither_scale_": np.asarray(self.dither_scale_),
            "tails_": getattr(self, "tails_", None),
        }

    @classmethod
    def from_dict(cls, state: dict) -> MarginalGaussianize:
        """Rebuild a fitted marginal from :meth:`to_dict` output."""
        if state.get("class") != cls.__name__:
            raise ValueError(
                f"State is for {state.get('class')!r}, not {cls.__name__}."
            )
        obj = cls(**state["params"])
        obj.support_ = np.asarray(state["support_"], dtype=float)
        obj.n_features_ = int(state["n_features_"])
        obj.constant_mask_ = np.asarray(state["constant_mask_"], dtype=bool)
        obj.offsets_ = np.asarray(state["offsets_"], dtype=float)
        obj.dither_scale_ = np.asarray(state["dither_scale_"], dtype=float)
        if state.get("tails_") is not None:
            obj.tails_ = state["tails_"]
        obj.kdes_ = [
            None
            if obj.constant_mask_[i]
            else stats.gaussian_kde(obj.support_[:, i].copy())
            for i in range(obj.n_features_)
        ]
        return obj


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

    def fit(self, X: np.ndarray, y=None) -> MarginalKDEGaussianize:
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

    def log_det_jacobian(self, X: np.ndarray) -> np.ndarray:
        """Log |det J| for the KDE-CDF Gaussianization.

        For ``g(x) = Phi^{-1}(F_KDE(x))`` the per-feature log-derivative is::

            log|dg/dx| = log f_KDE(x) - log phi(g(x))

        where ``f_KDE`` is the fitted kernel density itself — the exact
        derivative of the smooth KDE CDF, so this quantity matches central
        finite differences of :meth:`transform` to high accuracy.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data at which to evaluate the log-det-Jacobian.

        Returns
        -------
        log_jac : np.ndarray of shape (n_samples,)
            Per-sample sum of per-feature log-derivatives.
        """
        Xt = self.transform(X)  # Gaussianized output, shape (N, D)
        log_jac = np.zeros(X.shape[0])
        for i in range(self.n_features_):
            xi = np.ascontiguousarray(X[:, i])
            log_f_i = np.log(np.maximum(self.kdes_[i](xi), 1e-300))
            log_jac += log_f_i - stats.norm.logpdf(Xt[:, i])
        return log_jac


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

    def fit(self, X: np.ndarray, y=None) -> QuantileGaussianizer:
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

    def fit(self, X: np.ndarray, y=None) -> KDEGaussianizer:
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

    def fit(self, X: np.ndarray, y=None) -> GMMGaussianizer:
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
    """Gaussianize each marginal using a monotonic rational-quadratic spline.

    Each feature is Gaussianized with an independent :class:`RQSpline` -- the
    shared 1D rational-quadratic spline primitive that also backs the
    SIG/GIS layers.  The forward transform is::

        z = g(x)

    where ``g`` composes a KDE-estimated CDF with the Gaussian quantile
    function.  The spline is monotone by construction (positive knot
    derivatives) and its inverse is analytic, so the mapping is exactly
    invertible.

    Parameters
    ----------
    n_quantiles : int, default 200
        Number of spline knots.  Capped at ``n_samples`` when fewer training
        samples are available.  (Maps to :class:`RQSpline`'s ``n_knots``.)
    eps : float, default 1e-6
        Clipping margin applied to CDF values before the probit, keeping the
        target knots finite.

    Attributes
    ----------
    splines_ : list of RQSpline
        One fitted rational-quadratic spline per feature, each mapping
        original-space values to standard-normal values (and back).
    n_features_ : int
        Number of feature dimensions seen during ``fit``.

    Notes
    -----
    The log-det-Jacobian uses the analytic spline derivative returned by the
    forward pass::

        log|dz/dx| = log|g'(x)|

    Between knots ``g`` is a rational-quadratic interpolant; outside the
    outermost knots it extends linearly, so both the transform and its
    inverse are finite and analytic everywhere.

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

    def fit(self, X: np.ndarray, y=None) -> SplineGaussianizer:
        """Fit one rational-quadratic spline per feature.

        For each dimension an :class:`RQSpline` is fitted with up to
        ``n_quantiles`` knots, estimating the marginal CDF by KDE and placing
        the target knots at the corresponding Gaussian quantiles.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : SplineGaussianizer
            Fitted bijector instance.
        """
        self.n_features_ = X.shape[1]
        # One shared RQ-spline primitive per feature dimension.  The same
        # primitive backs the SIG/GIS layers (see rbig._src.spline).
        self.splines_ = [
            RQSpline(n_knots=self.n_quantiles, eps=self.eps).fit(X[:, i])
            for i in range(self.n_features_)
        ]
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
            # Forward RQ-spline Gaussianization for feature i
            Xt[:, i], _ = self.splines_[i].forward(X[:, i])
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
            # Analytic inverse of the RQ spline for feature i
            Xt[:, i], _ = self.splines_[i].inverse(X[:, i])
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
            # Analytic log|dz/dx| from the RQ spline forward pass
            _, log_dz = self.splines_[i].forward(X[:, i])
            log_det += log_dz
        return log_det
