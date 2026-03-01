import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d


def compute_jacobian(rbig_model, X, return_X_transform=False):
    """Compute the full n_samples x n_features x n_features Jacobian matrix."""
    n_samples, n_components = X.shape
    jacobian = np.zeros((n_samples, n_components, n_components))
    X_transformed = X.copy()
    XX = np.zeros(shape=(n_samples, n_components))
    XX[:, 0] = np.ones(shape=n_samples)
    gaussian_pdf = np.zeros(shape=(n_samples, n_components, rbig_model.n_layers))
    igaussian_pdf = np.zeros(shape=(n_samples, n_components))
    for ilayer in range(rbig_model.n_layers):
        for idim in range(n_components):
            data_uniform = interp1d(
                rbig_model.gauss_params[ilayer][idim]["uniform_cdf_support"],
                rbig_model.gauss_params[ilayer][idim]["uniform_cdf"],
                fill_value="extrapolate",
            )(X_transformed[:, idim])
            igaussian_pdf[:, idim] = norm.ppf(data_uniform)
            gaussian_pdf[:, idim, ilayer] = interp1d(
                rbig_model.gauss_params[ilayer][idim]["empirical_pdf_support"],
                rbig_model.gauss_params[ilayer][idim]["empirical_pdf"],
                fill_value="extrapolate",
            )(X_transformed[:, idim]) * (1 / norm.pdf(igaussian_pdf[:, idim]))
        XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, rbig_model.rotation_matrix[ilayer])
        X_transformed = np.dot(igaussian_pdf, rbig_model.rotation_matrix[ilayer])
    jacobian[:, :, 0] = XX
    if n_components > 1:
        for idim in range(n_components):
            XX = np.zeros(shape=(n_samples, n_components))
            XX[:, idim] = np.ones(n_samples)
            for ilayer in range(rbig_model.n_layers):
                XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, rbig_model.rotation_matrix[ilayer])
            jacobian[:, :, idim] = XX
    if return_X_transform:
        return jacobian, X_transformed
    else:
        return jacobian
