"""Marginal Gaussianization: fit, transform, inverse, and entropy."""
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm

BOUNDS_THRESHOLD = 1e-7


def bin_estimation(n_samples: int, rule: str = "standard") -> int:
    """Estimate the number of histogram bins for n_samples."""
    if rule == "sturge":
        return int(np.ceil(1 + 3.322 * np.log10(n_samples)))
    elif rule == "standard":
        return int(np.ceil(np.sqrt(n_samples)))
    else:
        raise ValueError(f"Unrecognized bin estimation rule: {rule}")


def make_cdf_monotonic(cdf: np.ndarray) -> np.ndarray:
    """Ensure a CDF array is monotonically non-decreasing."""
    return np.maximum.accumulate(cdf)


def fit_marginal_params(
    X: np.ndarray,
    extension: int = 10,
    n_quantiles: int = 1000,
) -> Dict:
    """Fit marginal Gaussianization parameters for a 1-D array.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples,)
        1-D input data.
    extension : int
        Percentage by which to extend the support beyond the data range.
    n_quantiles : int
        Number of support points for the CDF/PPF interpolation.

    Returns
    -------
    params : dict
        Keys: uniform_cdf_support, uniform_cdf, empirical_pdf_support,
        empirical_pdf, _cdf_fn, _ppf_fn, x_bounds.
    """
    X = np.asarray(X, dtype=float).ravel()
    n_samples = len(X)

    # Compute empirical PDF via histogram
    n_bins = bin_estimation(n_samples)
    counts, bin_edges = np.histogram(X, bins=n_bins, range=(X.min(), X.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_size = bin_edges[1] - bin_edges[0]

    pdf_support = np.concatenate(
        [[bin_centers[0] - bin_size], bin_centers, [bin_centers[-1] + bin_size]]
    )
    empirical_pdf = np.concatenate(
        [[0.0], counts / (counts.sum() * bin_size + 1e-12), [0.0]]
    )

    # Extended support for CDF
    ext_range = (extension / 100.0) * (X.max() - X.min())
    x_min = X.min() - ext_range
    x_max = X.max() + ext_range

    new_support = np.linspace(x_min, x_max, n_quantiles)

    # Build empirical CDF from cumulative histogram counts
    c_sum = np.cumsum(counts)
    cdf_values = (1.0 - 1.0 / n_samples) * c_sum / n_samples

    # Anchor points: 0 at x_min, small at X.min(), monotone through data, 1 at x_max
    incr_bin = bin_size / 2.0
    cdf_x = np.concatenate(
        [
            [x_min],
            [X.min()],
            bin_centers + incr_bin,
            [X.max() + ext_range],
        ]
    )
    cdf_y = np.concatenate([[0.0], [1.0 / n_samples], cdf_values, [1.0]])

    raw_cdf_fn = interp1d(cdf_x, cdf_y, kind="linear", fill_value=(0.0, 1.0), bounds_error=False)
    uniform_cdf = make_cdf_monotonic(raw_cdf_fn(new_support))
    uniform_cdf = uniform_cdf / uniform_cdf.max()

    cdf_fn = interp1d(new_support, uniform_cdf, kind="linear", fill_value="extrapolate", bounds_error=False)
    ppf_fn = interp1d(uniform_cdf, new_support, kind="linear", fill_value="extrapolate", bounds_error=False)

    return {
        "uniform_cdf_support": new_support,
        "uniform_cdf": uniform_cdf,
        "empirical_pdf_support": pdf_support,
        "empirical_pdf": empirical_pdf,
        "_cdf_fn": cdf_fn,
        "_ppf_fn": ppf_fn,
        "x_bounds": [x_min, x_max],
    }


def marginal_gaussianize(X: np.ndarray, params: Dict) -> np.ndarray:
    """Gaussianize 1-D data using pre-fitted params.

    Applies the CDF then the normal quantile function (PPF).
    """
    X = np.asarray(X, dtype=float).ravel()
    u = params["_cdf_fn"](X)
    u = np.clip(u, BOUNDS_THRESHOLD, 1.0 - BOUNDS_THRESHOLD)
    return norm.ppf(u)


def marginal_gaussianize_inverse(Z: np.ndarray, params: Dict) -> np.ndarray:
    """Invert marginal Gaussianization: Gaussian → original domain."""
    Z = np.asarray(Z, dtype=float).ravel()
    u = norm.cdf(Z)
    u = np.clip(u, BOUNDS_THRESHOLD, 1.0 - BOUNDS_THRESHOLD)
    return params["_ppf_fn"](u)


def log_pdf_marginal(X: np.ndarray, params: Dict) -> np.ndarray:
    """Log PDF of the marginal distribution at X."""
    X = np.asarray(X, dtype=float).ravel()
    pdf = interp1d(
        params["empirical_pdf_support"],
        params["empirical_pdf"],
        kind="linear",
        fill_value=1e-12,
        bounds_error=False,
    )(X)
    pdf = np.clip(pdf, 1e-12, None)
    return np.log(pdf)


def entropy_marginal(
    data: np.ndarray,
    bin_est: str = "standard",
    correction: bool = True,
) -> np.ndarray:
    """Per-dimension marginal entropy of multivariate data.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, d_dimensions)
    bin_est : str
        Bin estimation rule ('standard' or 'sturge').
    correction : bool
        Apply Shannon-Miller correction.

    Returns
    -------
    H : np.ndarray, shape (d_dimensions,)
    """
    data = np.atleast_2d(data)
    n_samples, d_dimensions = data.shape
    n_bins = bin_estimation(n_samples, rule=bin_est)

    H = np.zeros(d_dimensions)
    for i in range(d_dimensions):
        col = data[:, i]
        counts, bin_edges = np.histogram(col, bins=n_bins, range=(col.min(), col.max()))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        delta = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 1.0

        # Shannon entropy with optional Miller-Madow correction
        corr = 0.0
        if correction:
            corr = 0.5 * (np.sum(counts > 0) - 1) / max(counts.sum(), 1)

        from scipy.stats import entropy as _sci_entropy

        H[i] = _sci_entropy(counts, base=2) + corr + np.log2(abs(delta) + 1e-12)

    return H
