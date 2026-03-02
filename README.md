# rbig

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/jejjohnson/rbig/actions/workflows/ci.yaml/badge.svg)](https://github.com/jejjohnson/rbig/actions)

**Rotation-Based Iterative Gaussianization — density estimation, IT measures, and generative modeling**

---

## Overview

RBIG is a density destructor: an invertible, differentiable transformation that maps any multivariate distribution to a standard Gaussian. It works by alternating between two steps:

1. **Marginal Gaussianization** — independently transform each dimension to Gaussian using an empirical CDF-based mapping.
2. **Rotation** — apply an orthonormal linear transform (PCA, ICA, random, or Picard) to mix the dimensions.

After enough iterations the joint distribution converges to a standard Gaussian. Because each step is invertible and differentiable, RBIG provides exact likelihood evaluation via the change-of-variables formula:

$$p_x(x) = p_z\!\left(\mathcal{G}_\theta(x)\right) \left|\nabla_x \mathcal{G}_\theta(x)\right|$$

This also makes RBIG a natural tool for computing information-theoretic measures (entropy, mutual information, total correlation, KL-divergence) without assuming any parametric form.

---

## Key Features

- **Multiple marginal Gaussianization methods**: Quantile Transform, KDE, Gaussian Mixture Model, Spline
- **Multiple rotation strategies**: PCA, ICA, Random, Picard
- **Information theory measures**: entropy, mutual information, total correlation, KL-divergence
- **Image processing support**: DCT, Hartley, and Wavelet transforms
- **Xarray integration**: spatiotemporal data with named dimensions
- **Parametric distributions**: Gaussian, Exponential, Gamma, Beta, and more

---

## Installation

```bash
pip install rbig
```

Optional extras:

```bash
pip install "rbig[image]"   # wavelet/DCT image support
pip install "rbig[xarray]"  # spatiotemporal xarray integration
pip install "rbig[all]"     # everything
```

---

## Quick Start

```python
import numpy as np
from rbig import AnnealedRBIG

# Generate correlated 2D data
rng = np.random.RandomState(42)
data = rng.randn(1000, 2) @ [[1, 0.8], [0.8, 1]]

# Fit RBIG and transform to Gaussian
model = AnnealedRBIG(n_layers=50, rotation="pca")
Z = model.fit_transform(data)

# Z is now approximately standard Gaussian
```

---

## Documentation

Full documentation lives in the [`docs/`](docs/) directory and is built with MkDocs Material.

---

## Citation

If you use RBIG in your research, please cite:

> V. Laparra, G. Camps-Valls, and J. Malo,
> "Iterative Gaussianization: from ICA to Random Rotations,"
> *IEEE Transactions on Neural Networks*, 22(4):537–549, 2011.
> [arXiv:1602.00229](https://arxiv.org/abs/1602.00229)

---

## License

MIT — see [LICENSE](LICENSE) for details.
