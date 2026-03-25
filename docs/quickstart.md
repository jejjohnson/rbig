# Quickstart

Get from zero to results in under five minutes.

---

## Part 1 — Density Estimation & Generative Modeling

### Fit RBIG to data

```python
import numpy as np
from rbig import AnnealedRBIG

# Generate a 2-D sin-wave dataset
rng = np.random.RandomState(42)
n = 2_000
x = np.abs(2 * rng.randn(1, n))
y = np.sin(x) + 0.25 * rng.randn(1, n)
data = np.vstack((x, y)).T  # (2000, 2)

# Fit RBIG
model = AnnealedRBIG(n_layers=50, rotation="pca", patience=10, random_state=42)
model.fit(data)
```

### Transform to Gaussian space

```python
import matplotlib.pyplot as plt

Z = model.transform(data)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hexbin(data[:, 0], data[:, 1], gridsize=30, cmap="Blues", mincnt=1)
axes[0].set_title("Original data")
axes[1].hexbin(Z[:, 0], Z[:, 1], gridsize=30, cmap="Purples", mincnt=1)
axes[1].set_title("After RBIG (≈ standard Gaussian)")
plt.tight_layout()
plt.show()
```

### Generate new samples

```python
samples = model.sample(n_samples=1_000, random_state=0)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hexbin(data[:, 0], data[:, 1], gridsize=30, cmap="Blues", mincnt=1)
axes[0].set_title("Training data")
axes[1].hexbin(samples[:, 0], samples[:, 1], gridsize=30, cmap="Oranges", mincnt=1)
axes[1].set_title("Generated samples")
plt.tight_layout()
plt.show()
```

### Estimate log-probabilities

```python
log_probs = model.score_samples(data)  # per-sample log p(x)
mean_ll = model.score(data)            # mean log-likelihood

fig, ax = plt.subplots(figsize=(5, 4))
h = ax.scatter(data[:, 0], data[:, 1], s=8, c=log_probs, cmap="Reds")
ax.set_title("Data coloured by log p(x)")
plt.colorbar(h, ax=ax, label="log p(x)")
plt.tight_layout()
plt.show()
```

For a full walkthrough with theory, see the [RBIG Walk-Through notebook](notebooks/03_rbig_walkthrough.ipynb).

---

## Part 2 — Information Theory Measures

RBIG can estimate classical information-theoretic quantities as a by-product of the Gaussianization. Here we show all of them on 2-D data.

### Setup

```python
import numpy as np
from rbig import (
    AnnealedRBIG,
    mutual_information_rbig,
    kl_divergence_rbig,
    total_correlation_rbig,
    entropy_rbig,
    information_summary,
)

rng = np.random.RandomState(42)
N = 1_000

# Two correlated 2-D random vectors
mean = np.zeros(4)
cov = np.eye(4)
cov[0, 2] = cov[2, 0] = 0.8  # x0 ↔ y0
cov[1, 3] = cov[3, 1] = 0.5  # x1 ↔ y1
joint = rng.multivariate_normal(mean, cov, size=N)

X = joint[:, :2]  # (N, 2)
Y = joint[:, 2:]  # (N, 2)
XY = np.hstack([X, Y])  # (N, 4)
```

### Fit models

```python
kwargs = dict(n_layers=50, rotation="pca", random_state=42)

model_x = AnnealedRBIG(**kwargs)
model_y = AnnealedRBIG(**kwargs)
model_xy = AnnealedRBIG(**kwargs)

model_x.fit(X)
model_y.fit(Y)
model_xy.fit(XY)
```

### Total Correlation

```python
tc = total_correlation_rbig(XY)
print(f"Total Correlation: {tc:.4f} nats")
```

### Entropy

```python
H = entropy_rbig(model_xy, XY)
print(f"Differential entropy H(X,Y): {H:.4f} nats")
```

### Mutual Information

```python
mi = mutual_information_rbig(model_x, model_y, model_xy)
print(f"Mutual Information I(X;Y): {mi:.4f} nats")
```

### KL-Divergence

```python
# How different is X from Y?
kld = kl_divergence_rbig(model_x, Y)
print(f"KL(X || Y): {kld:.4f} nats")
```

### Full summary

```python
summary = information_summary(model_xy, XY)
print(summary)
```

### Available IT measures

| Measure | Function / Method | Description |
|---------|-------------------|-------------|
| Total Correlation | `total_correlation_rbig(X)` | Multivariate mutual information (redundancy) |
| Differential Entropy | `entropy_rbig(model, X)` | Entropy of the fitted distribution |
| Mutual Information | `mutual_information_rbig(m_x, m_y, m_xy)` | Dependence between two random vectors |
| KL-Divergence | `kl_divergence_rbig(model_P, X_Q)` | Divergence between two distributions |
| Log-Likelihood | `model.score_samples(X)` | Per-sample log p(x) via change-of-variables |
| Mean Log-Likelihood | `model.score(X)` | Average log-likelihood |
| Entropy (model) | `model.entropy()` | Entropy of the fitted density |
| Info Summary | `information_summary(model, X)` | Dict with entropy, TC, and NLL |

For detailed examples, see the [Information Theory notebook](notebooks/06_information_theory.ipynb) and [Dependence Detection notebooks](notebooks/09_dependence_1d.ipynb).
