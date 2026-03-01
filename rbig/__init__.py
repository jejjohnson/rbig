"""RBIG: Rotation-Based Iterative Gaussianization."""

from rbig._src.base import BaseITMeasure, BaseTransform
from rbig._src.densities import (
    entropy_reduction,
    gaussian_entropy,
    joint_entropy_gaussian,
    marginal_entropy,
    total_correlation,
)
from rbig._src.image import (
    WaveletTransform,
    extract_patches,
    matrix_to_patches,
    patches_to_matrix,
)
from rbig._src.marginal import (
    MarginalGaussianize,
    MarginalKDEGaussianize,
    MarginalUniformize,
)
from rbig._src.metrics import (
    entropy_normal_approx,
    kl_divergence_rbig,
    mutual_information_rbig,
    negentropy,
    total_correlation_rbig,
)
from rbig._src.model import AnnealedRBIG, RBIGLayer
from rbig._src.parametric import BoxCoxTransform, LogitTransform, QuantileTransform
from rbig._src.rotation import ICARotation, PCARotation
from rbig._src.xarray_image import matrix_to_xr_image, xr_apply_rbig, xr_image_to_matrix
from rbig._src.xarray_st import matrix_to_xr_st, xr_rbig_fit_transform, xr_st_to_matrix
from rbig._version import __version__

__all__ = [
    "AnnealedRBIG",
    "BaseITMeasure",
    "BaseTransform",
    "BoxCoxTransform",
    "ICARotation",
    "LogitTransform",
    "MarginalGaussianize",
    "MarginalKDEGaussianize",
    "MarginalUniformize",
    "PCARotation",
    "QuantileTransform",
    "RBIGLayer",
    "WaveletTransform",
    "__version__",
    "entropy_normal_approx",
    "entropy_reduction",
    "extract_patches",
    "gaussian_entropy",
    "joint_entropy_gaussian",
    "kl_divergence_rbig",
    "marginal_entropy",
    "matrix_to_patches",
    "matrix_to_xr_image",
    "matrix_to_xr_st",
    "mutual_information_rbig",
    "negentropy",
    "patches_to_matrix",
    "total_correlation",
    "total_correlation_rbig",
    "xr_apply_rbig",
    "xr_image_to_matrix",
    "xr_rbig_fit_transform",
    "xr_st_to_matrix",
]
