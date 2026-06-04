"""Shared 1D monotonic rational-quadratic (RQ) spline primitive.

This module provides :class:`RQSpline`, a one-dimensional, strictly
monotonic rational-quadratic spline used as the core 1D Gaussianization
transform throughout the library.  The same primitive is applied whether
the input is a raw feature dimension (``X[:, k]``) or a projected slice
(``X @ a_k``); only the data fed in differs.

The forward map sends data to an (approximately) standard-normal variable
by composing an estimated marginal CDF with the Gaussian quantile
function::

    z = Phi ^ {-1}(F_hat(x))

The CDF ``F_hat`` is estimated with a Gaussian kernel density estimate
(KDE).  Knots are placed on the data support; the target ``z`` values are
the Gaussian quantiles ``Phi^{-1}(F_hat(x_knot))``; and the spline
derivatives at the knots are the analytic Gaussianization derivatives::

    g'(x) = f_hat(x) / phi(g(x))

which are strictly positive, guaranteeing a monotone (hence invertible)
transform.  Between knots the map is a rational-quadratic interpolant
(Durkan et al. 2019); outside the outermost knots it extends linearly,
so both the forward map and its inverse are analytic everywhere.

References
----------
Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural
Spline Flows. *NeurIPS*. https://arxiv.org/abs/1906.04032

Dai, B., & Seljak, U. (2020). Sliced Iterative Normalizing Flows.
https://arxiv.org/abs/2007.00674

sinflow ``MonotonicRationalQuadraticSpline``:
https://github.com/minaskar/sinflow
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.special import ndtri


class RQSpline:
    """Strictly monotonic 1D rational-quadratic spline for Gaussianization.

    A single fitted instance represents one scalar bijection ``x -> z`` that
    maps a 1D sample to an approximately standard-normal variable, together
    with its analytic inverse and log-derivative.

    Parameters
    ----------
    n_knots : int, default 1000
        Number of spline knots placed across the data support.  Capped at
        the number of available samples during :meth:`fit`.
    kde_bw : str, float, or None, default None
        Bandwidth selection passed to :class:`scipy.stats.gaussian_kde`
        when estimating the CDF.  ``None`` uses Scott's rule.
    eps : float, default 1e-6
        Clipping margin applied to CDF values before the probit
        ``Phi^{-1}`` to keep the target knots finite.
    min_derivative : float, default 1e-3
        Lower clip on the spline knot derivatives.  Prevents zero or
        negative slopes (which would break monotonicity) in regions of
        vanishing estimated density.

    Attributes
    ----------
    x_knots_ : np.ndarray of shape (n_bins + 1,)
        Strictly increasing knot positions in data space.
    y_knots_ : np.ndarray of shape (n_bins + 1,)
        Strictly increasing knot positions in Gaussian (latent) space.
    derivatives_ : np.ndarray of shape (n_bins + 1,)
        Positive spline derivatives ``dz/dx`` at each knot.
    identity_ : bool
        True when the input had no spread (constant feature); the spline
        then falls back to a centred identity map.

    Notes
    -----
    The rational-quadratic interpolant on bin ``k`` (Durkan et al. 2019),
    with bin width ``w = x_{k+1} - x_k``, height ``h = z_{k+1} - z_k``,
    slope ``s = h / w``, knot derivatives ``d_k, d_{k+1}`` and local
    coordinate ``xi = (x - x_k) / w`` is::

        z = z_k + h * (s * xi^2 + d_k * xi * (1 - xi)) / D
        D = s + (d_{k+1} + d_k - 2 s) * xi * (1 - xi)

    with derivative::

        dz/dx = s^2 * (d_{k+1} xi^2 + 2 s xi (1-xi) + d_k (1-xi)^2) / D^2

    The inverse solves a quadratic in ``xi`` analytically, so
    ``inverse(forward(x)) == x`` to numerical precision.

    Examples
    --------
    >>> import numpy as np
    >>> from rbig._src.spline import RQSpline
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(2000)
    >>> spline = RQSpline(n_knots=200).fit(x)
    >>> z, log_dzdx = spline.forward(x)
    >>> x_rec, _ = spline.inverse(z)
    >>> bool(np.allclose(x_rec, x, atol=1e-4))
    True
    """

    def __init__(
        self,
        n_knots: int = 1000,
        kde_bw: str | float | None = None,
        eps: float = 1e-6,
        min_derivative: float = 1e-3,
    ):
        self.n_knots = n_knots
        self.kde_bw = kde_bw
        self.eps = eps
        self.min_derivative = min_derivative

    def fit(
        self,
        x: np.ndarray,
        n_knots: int | None = None,
        kde_bw: str | float | None = None,
    ) -> RQSpline:
        """Estimate knots and derivatives from 1D data.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            One-dimensional training data (a raw feature or a projected
            slice).  Multi-dimensional input is flattened.
        n_knots : int or None, optional
            Override for ``self.n_knots`` for this fit.
        kde_bw : str, float, or None, optional
            Override for ``self.kde_bw`` for this fit.

        Returns
        -------
        self : RQSpline
            The fitted spline.
        """
        x = np.asarray(x, dtype=float).ravel()
        n_knots = int(self.n_knots if n_knots is None else n_knots)
        kde_bw = self.kde_bw if kde_bw is None else kde_bw

        x_min, x_max = float(x.min()), float(x.max())
        # Degenerate (constant) feature: fall back to a centred identity map.
        if x_max - x_min <= 0.0 or x.size < 2:
            self.identity_ = True
            self.x_offset_ = float(x.mean()) if x.size else 0.0
            self.x_knots_ = np.array([x_min - 1.0, x_min, x_min + 1.0])
            self.y_knots_ = self.x_knots_ - self.x_offset_
            self.derivatives_ = np.ones_like(self.x_knots_)
            return self
        self.identity_ = False

        # Use at most n_samples knots; need at least two bins.
        n_knots = max(3, min(n_knots, x.size))

        kde = stats.gaussian_kde(x, bw_method=kde_bw)
        bw = float(np.sqrt(kde.covariance[0, 0]))

        # Knot positions on a regular grid over the data support.
        x_knots = np.linspace(x_min, x_max, n_knots)

        # Analytic Gaussian-KDE CDF: F(t) = mean_i Phi((t - x_i) / bw).
        cdf = stats.norm.cdf((x_knots[:, None] - x[None, :]) / bw).mean(axis=1)
        # Enforce monotonicity and keep away from the probit's infinities.
        cdf = np.maximum.accumulate(cdf)
        cdf = np.clip(cdf, self.eps, 1.0 - self.eps)
        y_knots = ndtri(cdf)

        # Drop knots that are not strictly increasing in z (the CDF can
        # saturate to the clip bounds in the tails, producing ties).
        keep = np.concatenate([[True], np.diff(y_knots) > 0.0])
        x_knots, y_knots = x_knots[keep], y_knots[keep]
        if x_knots.size < 2:
            # KDE produced a degenerate CDF; fall back to identity.
            self.identity_ = True
            self.x_offset_ = float(x.mean())
            self.x_knots_ = np.array([x_min - 1.0, x_min, x_min + 1.0])
            self.y_knots_ = self.x_knots_ - self.x_offset_
            self.derivatives_ = np.ones_like(self.x_knots_)
            return self

        # Gaussianization derivative g'(x) = f(x) / phi(g(x)) at each knot.
        pdf = kde(x_knots)
        derivatives = pdf / np.maximum(stats.norm.pdf(y_knots), 1e-300)
        derivatives = np.clip(derivatives, self.min_derivative, 1.0 / self.eps)

        self.x_knots_ = x_knots
        self.y_knots_ = y_knots
        self.derivatives_ = derivatives
        return self

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map data to Gaussian space: ``z = g(x)``.

        Parameters
        ----------
        x : np.ndarray of shape (n_samples,)
            Input values in data space.

        Returns
        -------
        z : np.ndarray of shape (n_samples,)
            Gaussianized values.
        log_dz_dx : np.ndarray of shape (n_samples,)
            Per-sample log absolute derivative ``log|dz/dx|``.
        """
        x = np.asarray(x, dtype=float).ravel()
        if self.identity_:
            return x - self.x_offset_, np.zeros_like(x)

        xk, yk, d = self.x_knots_, self.y_knots_, self.derivatives_
        z = np.empty_like(x)
        log_dz_dx = np.empty_like(x)

        left = x < xk[0]
        right = x > xk[-1]
        interior = ~(left | right)

        # Linear tails extend the boundary knot derivative.
        z[left] = yk[0] + d[0] * (x[left] - xk[0])
        log_dz_dx[left] = np.log(d[0])
        z[right] = yk[-1] + d[-1] * (x[right] - xk[-1])
        log_dz_dx[right] = np.log(d[-1])

        if np.any(interior):
            xi_ = x[interior]
            # Bin index k such that xk[k] <= x < xk[k+1].
            k = np.searchsorted(xk, xi_, side="right") - 1
            k = np.clip(k, 0, xk.size - 2)
            w = xk[k + 1] - xk[k]
            h = yk[k + 1] - yk[k]
            s = h / w
            dk, dk1 = d[k], d[k + 1]
            xi = (xi_ - xk[k]) / w
            xi1 = 1.0 - xi

            denom = s + (dk1 + dk - 2.0 * s) * xi * xi1
            z[interior] = yk[k] + h * (s * xi * xi + dk * xi * xi1) / denom
            deriv = (
                s
                * s
                * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
                / (denom * denom)
            )
            log_dz_dx[interior] = np.log(np.maximum(deriv, 1e-300))

        return z, log_dz_dx

    def inverse(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map Gaussian space back to data space: ``x = g^{-1}(z)``.

        Parameters
        ----------
        z : np.ndarray of shape (n_samples,)
            Values in Gaussian (latent) space.

        Returns
        -------
        x : np.ndarray of shape (n_samples,)
            Recovered data-space values.
        log_dx_dz : np.ndarray of shape (n_samples,)
            Per-sample log absolute derivative ``log|dx/dz|`` of the inverse.
        """
        z = np.asarray(z, dtype=float).ravel()
        if self.identity_:
            return z + self.x_offset_, np.zeros_like(z)

        xk, yk, d = self.x_knots_, self.y_knots_, self.derivatives_
        x = np.empty_like(z)
        log_dx_dz = np.empty_like(z)

        left = z < yk[0]
        right = z > yk[-1]
        interior = ~(left | right)

        x[left] = xk[0] + (z[left] - yk[0]) / d[0]
        log_dx_dz[left] = -np.log(d[0])
        x[right] = xk[-1] + (z[right] - yk[-1]) / d[-1]
        log_dx_dz[right] = -np.log(d[-1])

        if np.any(interior):
            zi = z[interior]
            k = np.searchsorted(yk, zi, side="right") - 1
            k = np.clip(k, 0, yk.size - 2)
            w = xk[k + 1] - xk[k]
            h = yk[k + 1] - yk[k]
            s = h / w
            dk, dk1 = d[k], d[k + 1]
            delta = dk1 + dk - 2.0 * s
            eta = zi - yk[k]

            # Solve the quadratic a*xi^2 + b*xi + c = 0 for xi in [0, 1].
            a = h * (s - dk) + eta * delta
            b = h * dk - eta * delta
            c = -s * eta
            discriminant = np.maximum(b * b - 4.0 * a * c, 0.0)
            xi = 2.0 * c / (-b - np.sqrt(discriminant))
            xi = np.clip(xi, 0.0, 1.0)
            xi1 = 1.0 - xi

            x[interior] = xk[k] + xi * w
            denom = s + delta * xi * xi1
            deriv = (
                s
                * s
                * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1)
                / (denom * denom)
            )
            log_dx_dz[interior] = -np.log(np.maximum(deriv, 1e-300))

        return x, log_dx_dz
