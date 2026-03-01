# RBIG - Rotation-Based Iterative Gaussianization

[![codecov](https://codecov.io/gh/jejjohnson/rbig/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/rbig)

A Python package implementing Rotation-Based Iterative Gaussianization (RBIG) for multivariate data analysis and information-theoretic measures.

## Installation

```bash
pip install py-rbig
```

Or with uv:
```bash
uv add py-rbig
```

## Features

- **RBIG**: Core Rotation-Based Iterative Gaussianization algorithm
- **AnnealedRBIG**: Modern implementation with annealed convergence
- **RBIGMI**: Mutual information estimation between multivariate datasets
- **RBIGKLD**: KL divergence estimation between multivariate datasets
- **compute_jacobian**: Full Jacobian matrix computation

## Module Structure

| Module | Description |
|--------|-------------|
| `rbig/_src/base.py` | Base RBIG class |
| `rbig/_src/model.py` | AnnealedRBIG model |
| `rbig/_src/mi.py` | Mutual information via RBIG |
| `rbig/_src/kld.py` | KL divergence via RBIG |
| `rbig/_src/metrics.py` | Information-theoretic metrics |
| `rbig/_src/jacobian.py` | Jacobian computation |
| `rbig/_src/marginal.py` | Marginal Gaussianization utilities |
| `rbig/_src/rotation.py` | Rotation utilities |

## Quick Start

```python
import numpy as np
from rbig import RBIG, RBIGMI, RBIGKLD

# Fit RBIG model
X = np.random.randn(1000, 5)
model = RBIG(n_layers=100, rotation_type='PCA')
model.fit(X)

# Transform to Gaussian
Z = model.transform(X)

# Compute mutual information
X1 = np.random.randn(500, 3)
X2 = np.random.randn(500, 3)
mi_model = RBIGMI(n_layers=50)
mi_model.fit(X1, X2)
mi = mi_model.mutual_information()

# Compute KL divergence
kld_model = RBIGKLD(n_layers=50)
kld_model.fit(X1, X2)
kld = kld_model.get_kld()
```

## References

- Laparra, V., et al. "Iterative Gaussianization: from ICA to Random Rotations." IEEE Transactions on Neural Networks 22.4 (2011): 537-549.
- https://arxiv.org/abs/1602.00229
