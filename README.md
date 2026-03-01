# py-rbig: Rotation-Based Iterative Gaussianization

[![Tests](https://github.com/jejjohnson/rbig/actions/workflows/test.yml/badge.svg)](https://github.com/jejjohnson/rbig/actions)

A Python package implementing Rotation-Based Iterative Gaussianization (RBIG) for multivariate data analysis, density estimation, and information-theoretic measures.

## Installation

```bash
pip install -e ".[test,xarray]"
```

## Quick Start

```python
import numpy as np
from rbig import AnnealedRBIG

# Generate some data
X = np.random.randn(500, 5)

# Fit RBIG model
model = AnnealedRBIG(n_layers=100).fit(X)

# Transform to Gaussian domain
Z = model.transform(X)

# Recover original data
X_rec = model.inverse_transform(Z)

# Log probability
log_p = model.score_samples(X)

# Information measures
tc = model.total_correlation()
h = model.entropy()
```

## Package Structure

```
rbig/
├── _src/
│   ├── base.py          # Abstract base classes (Bijector, RotationTransform, ...)
│   ├── marginal.py      # Marginal Gaussianization (fit, transform, inverse, entropy)
│   ├── rotation.py      # PCA and random rotation transforms
│   ├── model.py         # AnnealedRBIG model
│   ├── metrics.py       # Information metrics (TC, entropy, MI)
│   ├── densities.py     # Density estimation (log_prob, score_samples)
│   ├── parametric.py    # Parametric marginal transforms (histogram, KDE, quantile, ...)
│   ├── image.py         # Image Gaussianization via patches
│   ├── xarray_st.py     # xarray spatiotemporal support
│   └── xarray_image.py  # xarray image support
├── _version.py
└── py.typed
```

## Features

- **AnnealedRBIG**: Full RBIG model with forward/inverse transforms, log-probability, total correlation, and entropy
- **Marginal Gaussianization**: Non-parametric CDF-based marginal transforms
- **Rotations**: PCA and random orthogonal rotations
- **Parametric Transforms**: Histogram, KDE, QuantileTransformer, Gaussian mixture, logistic, Laplace, uniform, and normal marginals
- **Image Support**: Patch-based RBIG for grayscale and multi-channel images
- **xarray Integration**: Apply RBIG to `xr.Dataset` and `xr.DataArray` objects

## References

- Laparra et al. (2011): *Iterative Gaussianization: from ICA to Random Rotations*. [arXiv:1602.00229](https://arxiv.org/abs/1602.00229)
- Original MATLAB implementation: [http://isp.uv.es/rbig.html](http://isp.uv.es/rbig.html)

