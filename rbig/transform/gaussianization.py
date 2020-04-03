from typing import Callable, Optional, Union

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_random_state

from rbig.base import DensityMixin, DensityTransformerMixin

# from rbig.density import Histogram
from rbig.transform.gauss_icdf import InverseGaussCDF
from rbig.transform.histogram import MarginalHistogramTransform


class HistogramGaussianization(BaseEstimator, DensityTransformerMixin, DensityMixin):
    """This performs a univariate transformation on a datasets.
    
    Assuming that the data is independent across features, this
    applies a transformation on each feature independently. The inverse 
    transformation is the marginal cdf applied to each of the features
    independently and the inverse transformation is the marginal inverse
    cdf applied to the features independently.
    """

    def __init__(
        self, nbins: Optional[Union[int, str]] = "auto", alpha: float = 1e-5
    ) -> None:
        self.nbins = nbins
        self.alpha = alpha

    def fit(self, X, y=None):

        # Uniformization
        self.marg_uniformer_ = MarginalHistogramTransform(
            nbins=self.nbins, alpha=self.alpha
        )
        self.marg_uniformer_.fit(X)

        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Marginal Uniformization
        X = self.marg_uniformer_.transform(X)

        # 2. Marginal Gaussianization
        X = InverseGaussCDF().transform(X)

        return X

    def log_abs_det_jacobian(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # marginal uniformization
        # print("X: ", X.min(), X.max())
        Xu_der = self.marg_uniformer_.log_abs_det_jacobian(X)
        # print("Xu Jacobian:", Xu_der.min(), Xu_der.max())
        X = self.marg_uniformer_.transform(X)
        # print("X_u:", X.min(), X.max())

        # inverse CDF gaussian
        # X = InverseGaussCDF().transform(X)
        # print("Xg:", X.min(), X.max())

        Xg_der = InverseGaussCDF().log_abs_det_jacobian(X)
        # print("Xg jacobian:", Xg_der.min(), Xg_der.max())
        # print(f"#Nans: {np.count_nonzero(~np.isnan(Xg_der))} / {Xg_der.shape[0]}")

        return Xu_der + Xg_der

    def inverse_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:

        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        # 1. Inverse Gaussianization
        X = InverseGaussCDF().inverse_transform(X)

        # 2. Inverse Uniformization
        X = self.marg_uniformer_.inverse_transform(X)

        return X

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Returns the log determinant abs jacobian of the inputs.
        
        Parameters
        ----------
        X : np.ndarray
            Inputs to be transformed
        
        y: Not used, only for compatibility
        
        """

        # Marginal Gaussianization Transformation
        # check inputs
        X = check_array(X, ensure_2d=True, copy=False)

        x_logprob = stats.norm().logpdf(self.transform(X))

        return (x_logprob + self.log_abs_det_jacobian(X)).sum(axis=1)