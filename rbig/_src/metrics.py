import numpy as np
from rbig._src.marginal import entropy_marginal


def information_reduction(x_data, y_data, tol_dimensions=None, correction=True):
    """Computes the multi-information (total correlation) reduction after a linear transformation."""
    err_msg = "Number of samples for x and y should be equal."
    np.testing.assert_equal(x_data.shape, y_data.shape, err_msg=err_msg)
    n_samples, n_dimensions = x_data.shape
    if tol_dimensions is None or 0:
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        tol_dimensions = np.interp(n_samples, xxx, yyy)
    hx = entropy_marginal(x_data, correction=correction)
    hy = entropy_marginal(y_data, correction=correction)
    I = np.sum(hy) - np.sum(hx)
    II = np.sqrt(np.sum((hy - hx) ** 2))
    p = 0.25
    if II < np.sqrt(n_dimensions * p * tol_dimensions ** 2) or I < 0:
        I = 0.0
    return float(I)


def entropy(hist_counts, correction=None):
    from scipy.stats import entropy as sci_entropy
    if not (correction is None):
        correction = 0.5 * (np.sum(hist_counts > 0) - 1) / hist_counts.sum()
    else:
        correction = 0.0
    return sci_entropy(hist_counts, base=2) + correction


def neg_entropy_normal(data):
    """Calculate marginal negative entropy per dimension."""
    from scipy import stats
    n_samples, d_dimensions = data.shape
    n_bins = int(np.ceil(np.sqrt(n_samples)))
    neg = np.zeros(d_dimensions)
    for idim in range(d_dimensions):
        [hist_counts, bin_edges] = np.histogram(
            a=data[:, idim], bins=n_bins, range=(data[:, idim].min(), data[:, idim].max())
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        delta = bin_centers[3] - bin_centers[2]
        pg = stats.norm.pdf(bin_centers, 0, 1)
        kde_model = stats.gaussian_kde(data[:, idim])
        hx = kde_model.pdf(bin_centers)
        px = hx / (hx.sum() * delta)
        idx = np.where((px > 0) & (pg > 0))
        neg[idim] = delta * (px[idx] * np.log2(px[idx] / pg[idx])).sum()
    return neg


def total_correlation(rbig_model):
    """Compute total correlation from a fitted RBIG model."""
    return rbig_model.residual_info.sum()


def entropy_from_rbig(rbig_model, correction=True):
    """Compute entropy from a fitted RBIG model."""
    return entropy_marginal(rbig_model.X_fit_, correction=correction).sum() - rbig_model.mutual_information
