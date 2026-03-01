import numpy as np
import warnings
import logging
import sys
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, FastICA
from sklearn.utils.validation import check_is_fitted
from scipy.stats import norm, uniform, ortho_group
from scipy.interpolate import interp1d

from rbig._src.marginal import make_cdf_monotonic, entropy_marginal
from rbig._src.metrics import information_reduction

logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s: %(levelname)s: %(message)s")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")


class RBIG(BaseEstimator, TransformerMixin):
    """Rotation-Based Iterative Gaussian-ization (RBIG)."""

    def __init__(
        self,
        n_layers=1000,
        rotation_type="PCA",
        pdf_resolution=1000,
        pdf_extension=None,
        random_state=None,
        verbose: int = 0,
        tolerance=None,
        zero_tolerance=60,
        entropy_correction=True,
        rotation_kwargs=None,
        base="gauss",
    ):
        self.n_layers = n_layers
        self.rotation_type = rotation_type
        self.pdf_resolution = pdf_resolution
        self.pdf_extension = pdf_extension
        self.random_state = random_state
        self.verbose = verbose
        self.tolerance = tolerance
        self.zero_tolerance = zero_tolerance
        self.entropy_correction = entropy_correction
        self.rotation_kwargs = rotation_kwargs
        self.base = base

    def fit(self, X):
        X = check_array(X, ensure_2d=True)
        self._fit(X)
        return self

    def _fit(self, data):
        data = check_array(data, ensure_2d=True)
        if self.pdf_extension is None:
            self.pdf_extension = 10
        if self.pdf_resolution is None:
            self.pdf_resolution = 2 * np.round(np.sqrt(data.shape[0]))
        self.X_fit_ = data
        gauss_data = np.copy(data)
        n_samples, n_dimensions = np.shape(data)
        if self.zero_tolerance is None:
            self.zero_tolerance = self.n_layers + 1
        if self.tolerance is None:
            self.tolerance = self._get_information_tolerance(n_samples)
        self.residual_info = list()
        self.gauss_params = list()
        self.rotation_matrix = list()
        for layer in range(self.n_layers):
            if self.verbose > 1:
                print("Completed {} iterations of RBIG.".format(layer + 1))
            layer_params = list()
            for idim in range(n_dimensions):
                gauss_data[:, idim], temp_params = self.univariate_make_normal(
                    gauss_data[:, idim], self.pdf_extension, self.pdf_resolution
                )
                layer_params.append(temp_params)
            self.gauss_params.append(layer_params)
            gauss_data_prerotation = gauss_data.copy()
            if self.verbose == 2:
                print(gauss_data.min(), gauss_data.max())
            if self.rotation_type == "random":
                rand_ortho_matrix = ortho_group.rvs(n_dimensions)
                gauss_data = np.dot(gauss_data, rand_ortho_matrix)
                self.rotation_matrix.append(rand_ortho_matrix)
            elif self.rotation_type.lower() == "ica":
                if self.rotation_kwargs is not None:
                    ica_model = FastICA(random_state=self.random_state, **self.rotation_kwargs)
                else:
                    ica_model = FastICA(random_state=self.random_state)
                gauss_data = ica_model.fit_transform(gauss_data)
                self.rotation_matrix.append(ica_model.components_.T)
            elif self.rotation_type.lower() == "pca":
                if self.rotation_kwargs is not None:
                    pca_model = PCA(random_state=self.random_state, **self.rotation_kwargs)
                else:
                    pca_model = PCA(random_state=self.random_state)
                gauss_data = pca_model.fit_transform(gauss_data)
                self.rotation_matrix.append(pca_model.components_.T)
            else:
                raise ValueError("Rotation type " + self.rotation_type + " not recognized")
            self.residual_info.append(
                information_reduction(gauss_data, gauss_data_prerotation, self.tolerance)
            )
            if self._stopping_criteria(layer):
                break
        self.residual_info = np.array(self.residual_info)
        self.gauss_data = gauss_data
        self.mutual_information = np.sum(self.residual_info)
        self.n_layers = len(self.gauss_params)
        return self

    def _stopping_criteria(self, layer):
        stop_ = False
        if layer > self.zero_tolerance:
            aux_residual = np.array(self.residual_info)
            if np.abs(aux_residual[-self.zero_tolerance:]).sum() == 0:
                self.rotation_matrix = self.rotation_matrix[:-50]
                self.gauss_params = self.gauss_params[:-50]
                stop_ = True
            else:
                stop_ = False
        return stop_

    def transform(self, X):
        check_is_fitted(self, ["gauss_params", "rotation_matrix"])
        n_dimensions = np.shape(X)[1]
        X_transformed = np.copy(X)
        for layer in range(self.n_layers):
            data_layer = X_transformed
            for idim in range(n_dimensions):
                data_layer[:, idim] = interp1d(
                    self.gauss_params[layer][idim]["uniform_cdf_support"],
                    self.gauss_params[layer][idim]["uniform_cdf"],
                    fill_value="extrapolate",
                )(data_layer[:, idim])
                data_layer[:, idim] = norm.ppf(data_layer[:, idim])
            X_transformed = np.dot(data_layer, self.rotation_matrix[layer])
        return X_transformed

    def inverse_transform(self, X):
        check_is_fitted(self, ["gauss_params", "rotation_matrix"])
        n_dimensions = np.shape(X)[1]
        X_input_domain = np.copy(X)
        for layer in range(self.n_layers - 1, -1, -1):
            if self.verbose > 1:
                print("Completed {} inverse iterations of RBIG.".format(layer + 1))
            X_input_domain = np.dot(X_input_domain, self.rotation_matrix[layer].T)
            temp = X_input_domain
            for idim in range(n_dimensions):
                temp[:, idim] = self.univariate_invert_normalization(
                    temp[:, idim], self.gauss_params[layer][idim]
                )
            X_input_domain = temp
        return X_input_domain

    def _get_information_tolerance(self, n_samples):
        xxx = np.logspace(2, 8, 7)
        yyy = [0.1571, 0.0468, 0.0145, 0.0046, 0.0014, 0.0001, 0.00001]
        return interp1d(xxx, yyy, fill_value="extrapolate")(n_samples)

    def jacobian(self, X, return_X_transform=False):
        n_samples, n_components = X.shape
        jacobian = np.zeros((n_samples, n_components, n_components))
        X_transformed = X.copy()
        XX = np.zeros(shape=(n_samples, n_components))
        XX[:, 0] = np.ones(shape=n_samples)
        gaussian_pdf = np.zeros(shape=(n_samples, n_components, self.n_layers))
        igaussian_pdf = np.zeros(shape=(n_samples, n_components))
        for ilayer in range(self.n_layers):
            for idim in range(n_components):
                data_uniform = interp1d(
                    self.gauss_params[ilayer][idim]["uniform_cdf_support"],
                    self.gauss_params[ilayer][idim]["uniform_cdf"],
                    fill_value="extrapolate",
                )(X_transformed[:, idim])
                igaussian_pdf[:, idim] = norm.ppf(data_uniform)
                gaussian_pdf[:, idim, ilayer] = interp1d(
                    self.gauss_params[ilayer][idim]["empirical_pdf_support"],
                    self.gauss_params[ilayer][idim]["empirical_pdf"],
                    fill_value="extrapolate",
                )(X_transformed[:, idim]) * (1 / norm.pdf(igaussian_pdf[:, idim]))
            XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer])
            X_transformed = np.dot(igaussian_pdf, self.rotation_matrix[ilayer])
        jacobian[:, :, 0] = XX
        if n_components > 1:
            for idim in range(n_components):
                XX = np.zeros(shape=(n_samples, n_components))
                XX[:, idim] = np.ones(n_samples)
                for ilayer in range(self.n_layers):
                    XX = np.dot(gaussian_pdf[:, :, ilayer] * XX, self.rotation_matrix[ilayer])
                jacobian[:, :, idim] = XX
        if return_X_transform:
            return jacobian, X_transformed
        else:
            return jacobian

    def predict_proba(self, X, n_trials=1, chunksize=2000, domain="input"):
        component_wise_std = np.std(X, axis=0) / 20
        n_samples, n_components = X.shape
        prob_data_gaussian_domain = np.zeros(shape=(n_samples, n_trials))
        prob_data_input_domain = np.zeros(shape=(n_samples, n_trials))
        for itrial in range(n_trials):
            jacobians = np.zeros(shape=(n_samples, n_components, n_components))
            if itrial < n_trials:
                data_aux = X + component_wise_std[None, :]
            else:
                data_aux = X
            data_temp = np.zeros(data_aux.shape)
            jacobians, data_temp = self.jacobian(data_aux, return_X_transform=True)
            jacobians[np.isnan(jacobians)] = 0.0
            det_jacobians = np.linalg.det(jacobians)
            prob_data_gaussian_domain[:, itrial] = np.prod(
                (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power(data_temp, 2)), axis=1
            )
            prob_data_gaussian_domain[np.isnan(prob_data_gaussian_domain)] = 0.0
            prob_data_input_domain[:, itrial] = prob_data_gaussian_domain[:, itrial] * np.abs(det_jacobians)
            prob_data_input_domain[np.isnan(prob_data_input_domain)] = 0.0
        prob_data_input_domain = prob_data_input_domain.mean(axis=1)
        prob_data_gaussian_domain = prob_data_gaussian_domain.mean(axis=1)
        det_jacobians = det_jacobians.mean()
        self.jacobians = jacobians
        self.det_jacobians = det_jacobians
        if domain == "input":
            return prob_data_input_domain
        elif domain == "transform":
            return prob_data_gaussian_domain
        elif domain == "both":
            return prob_data_input_domain, prob_data_gaussian_domain

    def entropy(self, correction=None):
        if (correction is None) or (correction is False):
            correction = self.entropy_correction
        return entropy_marginal(self.X_fit_, correction=correction).sum() - self.mutual_information

    def total_correlation(self):
        return self.residual_info.sum()

    def univariate_make_normal(self, uni_data, extension, precision):
        data_uniform, params = self.univariate_make_uniform(uni_data.T, extension, precision)
        if self.base == "gauss":
            return norm.ppf(data_uniform).T, params
        elif self.base == "uniform":
            return uniform.ppf(data_uniform).T, params
        else:
            raise ValueError(f"Unrecognized base dist: {self.base}.")

    def univariate_make_uniform(self, uni_data, extension, precision):
        n_samps = len(uni_data)
        support_extension = (extension / 100) * abs(np.max(uni_data) - np.min(uni_data))
        bin_edges = np.linspace(np.min(uni_data), np.max(uni_data), int(np.sqrt(np.float64(n_samps)) + 1))
        bin_centers = np.mean(np.vstack((bin_edges[0:-1], bin_edges[1:])), axis=0)
        counts, _ = np.histogram(uni_data, bin_edges)
        bin_size = bin_edges[2] - bin_edges[1]
        pdf_support = np.hstack((bin_centers[0] - bin_size, bin_centers, bin_centers[-1] + bin_size))
        empirical_pdf = np.hstack((0.0, counts / (np.sum(counts) * bin_size), 0.0))
        c_sum = np.cumsum(counts)
        cdf = (1 - 1 / n_samps) * c_sum / n_samps
        incr_bin = bin_size / 2
        new_bin_edges = np.hstack((
            np.min(uni_data) - support_extension,
            np.min(uni_data),
            bin_centers + incr_bin,
            np.max(uni_data) + support_extension + incr_bin,
        ))
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

    def univariate_invert_normalization(self, uni_gaussian_data, trans_params):
        if self.base == "gauss":
            uni_uniform_data = norm.cdf(uni_gaussian_data)
        elif self.base == "uniform":
            uni_uniform_data = uniform.cdf(uni_gaussian_data)
        else:
            raise ValueError(f"Unrecognized base dist.: {self.base}.")
        uni_data = self.univariate_invert_uniformization(uni_uniform_data, trans_params)
        return uni_data

    def univariate_invert_uniformization(self, uni_uniform_data, trans_params):
        return interp1d(
            trans_params["uniform_cdf"],
            trans_params["uniform_cdf_support"],
            fill_value="extrapolate",
        )(uni_uniform_data)
