import numpy as np
from scipy.stats import entropy as sci_entropy, norm, uniform
from scipy.interpolate import interp1d


def make_cdf_monotonic(cdf):
    """Take a cdf and sequentially readjust values to force monotonicity."""
    corrected_cdf = cdf.copy()
    for i in range(1, len(corrected_cdf)):
        if corrected_cdf[i] <= corrected_cdf[i - 1]:
            if abs(corrected_cdf[i - 1]) > 1e-14:
                corrected_cdf[i] = corrected_cdf[i - 1] + 1e-14
            elif corrected_cdf[i - 1] == 0:
                corrected_cdf[i] = 1e-80
            else:
                corrected_cdf[i] = corrected_cdf[i - 1] + 10 ** (
                    np.log10(abs(corrected_cdf[i - 1]))
                )
    return corrected_cdf


def entropy(hist_counts, correction=None):
    if not (correction is None):
        correction = 0.5 * (np.sum(hist_counts > 0) - 1) / hist_counts.sum()
    else:
        correction = 0.0
    return sci_entropy(hist_counts, base=2) + correction


def bin_estimation(n_samples, rule="standard"):
    if rule == "sturge":
        n_bins = int(np.ceil(1 + 3.322 * np.log10(n_samples)))
    elif rule == "standard":
        n_bins = int(np.ceil(np.sqrt(n_samples)))
    else:
        raise ValueError(f"Unrecognized bin estimation rule: {rule}")
    return n_bins


def entropy_marginal(data, bin_est="standard", correction=True):
    """Calculates the marginal entropy per dimension of a multidimensional dataset."""
    n_samples, d_dimensions = data.shape
    n_bins = bin_estimation(n_samples, rule=bin_est)
    H = np.zeros(d_dimensions)
    for idim in range(d_dimensions):
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim],
            bins=n_bins,
            range=(data[:, idim].min(), data[:, idim].max()),
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        delta = bin_centers[3] - bin_centers[2]
        H[idim] = entropy(hist_counts, correction=correction) + np.log2(delta)
    return H


def univariate_make_uniform(uni_data, extension, precision):
    """Transform univariate data to have approximately uniform distribution."""
    n_samps = len(uni_data)
    support_extension = (extension / 100) * abs(np.max(uni_data) - np.min(uni_data))
    bin_edges = np.linspace(
        np.min(uni_data), np.max(uni_data), int(np.sqrt(np.float64(n_samps)) + 1)
    )
    bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
    counts, _ = np.histogram(uni_data, bin_edges)
    bin_size = bin_edges[2] - bin_edges[1]
    pdf_support = np.hstack(
        (bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size)
    )
    empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))
    c_sum = np.cumsum(counts)
    cdf = (1 - 1 / n_samps) * c_sum / n_samps
    incr_bin = bin_size / 2
    new_bin_edges = np.hstack(
        (
            np.min(uni_data) - support_extension,
            np.min(uni_data),
            bin_centers + incr_bin,
            np.max(uni_data) + support_extension + incr_bin,
        )
    )
    extended_cdf = np.hstack((0.0, 1.0 / n_samps, cdf, 1.0))
    new_support = np.linspace(new_bin_edges[0], new_bin_edges[-1], int(precision))
    learned_cdf = interp1d(new_bin_edges, extended_cdf, fill_value="extrapolate")
    uniform_cdf = make_cdf_monotonic(learned_cdf(new_support))
    uniform_cdf /= np.max(uniform_cdf)
    uni_uniform_data = interp1d(new_support, uniform_cdf, fill_value="extrapolate")(uni_data)
    return (
        uni_uniform_data,
        {
            "empirical_pdf_support": pdf_support,
            "empirical_pdf": empirical_pdf,
            "uniform_cdf_support": new_support,
            "uniform_cdf": uniform_cdf,
        },
    )


def univariate_make_normal(uni_data, extension, precision, base="gauss"):
    """Transform univariate data to have approximately normal distribution."""
    data_uniform, params = univariate_make_uniform(uni_data.T, extension, precision)
    if base == "gauss":
        return norm.ppf(data_uniform).T, params
    elif base == "uniform":
        return uniform.ppf(data_uniform).T, params
    else:
        raise ValueError(f"Unrecognized base dist: {base}.")


def univariate_invert_uniformization(uni_uniform_data, trans_params):
    """Invert the marginal uniformization transform."""
    return interp1d(
        trans_params["uniform_cdf"],
        trans_params["uniform_cdf_support"],
        fill_value="extrapolate",
    )(uni_uniform_data)


def univariate_invert_normalization(uni_gaussian_data, trans_params, base="gauss"):
    """Invert the marginal normalization."""
    if base == "gauss":
        uni_uniform_data = norm.cdf(uni_gaussian_data)
    elif base == "uniform":
        uni_uniform_data = uniform.cdf(uni_gaussian_data)
    else:
        raise ValueError(f"Unrecognized base dist.: {base}.")
    return univariate_invert_uniformization(uni_uniform_data, trans_params)
