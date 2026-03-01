"""RBIG: Rotation-Based Iterative Gaussianization."""
from rbig._src.densities import (
    log_prob,
    score_samples,
)
from rbig._src.image import (
    ImageRBIG,
    extract_patches_2d,
    image_gaussianize,
    reconstruct_from_patches_2d,
)
from rbig._src.marginal import (
    entropy_marginal,
    fit_marginal_params,
    marginal_gaussianize,
    marginal_gaussianize_inverse,
)
from rbig._src.metrics import (
    entropy_rbig,
    information_reduction,
    mutual_information,
    total_correlation,
)
from rbig._src.model import AnnealedRBIG
from rbig._src.parametric import (
    HistogramUniformization,
    KDEUniformization,
    QuantileUniformization,
    fit_parametric_marginal,
    parametric_gaussianize,
    parametric_gaussianize_inverse,
)
from rbig._src.rotation import (
    PCARotation,
    RandomRotation,
    apply_rotation,
    apply_rotation_inverse,
    fit_rotation,
)
from rbig._version import __version__

__all__ = [
    "__version__",
    "AnnealedRBIG",
    "fit_marginal_params",
    "marginal_gaussianize",
    "marginal_gaussianize_inverse",
    "entropy_marginal",
    "PCARotation",
    "RandomRotation",
    "fit_rotation",
    "apply_rotation",
    "apply_rotation_inverse",
    "information_reduction",
    "total_correlation",
    "entropy_rbig",
    "mutual_information",
    "score_samples",
    "log_prob",
    "HistogramUniformization",
    "KDEUniformization",
    "QuantileUniformization",
    "fit_parametric_marginal",
    "parametric_gaussianize",
    "parametric_gaussianize_inverse",
    "ImageRBIG",
    "image_gaussianize",
    "extract_patches_2d",
    "reconstruct_from_patches_2d",
]
