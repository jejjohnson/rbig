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

RBIG estimates information-theoretic quantities using the **per-layer TC
reduction** approach from Laparra et al. (2011, 2020).  Each RBIG layer
removes statistical dependence; summing these reductions gives the total
correlation — no Jacobian estimation needed.

### Quick estimates (Level 0 — data only)

The simplest API: pass data, get an IT estimate.

```python
import numpy as np
from rbig import estimate_tc, estimate_entropy, estimate_mi, estimate_kld

rng = np.random.RandomState(42)

# Two correlated 2-D random vectors
cov = np.eye(4)
cov[0, 2] = cov[2, 0] = 0.8
cov[1, 3] = cov[3, 1] = 0.5
joint = rng.multivariate_normal(np.zeros(4), cov, size=1_000)
X, Y = joint[:, :2], joint[:, 2:]

tc = estimate_tc(joint, random_state=42)
h  = estimate_entropy(joint, random_state=42)
mi = estimate_mi(X, Y, random_state=42)
kl = estimate_kld(X, Y, random_state=42)

print(f"TC(X,Y):    {tc:.4f} nats")
print(f"H(X,Y):     {h:.4f} nats")
print(f"MI(X; Y):   {mi:.4f} nats")
print(f"KLD(X || Y): {kl:.4f} nats")
```

### Pre-fitted models (Level 1 — more control)

Fit models once, then compute multiple measures without re-fitting.

```python
from rbig import (
    AnnealedRBIG,
    total_correlation_rbig_reduction,
    entropy_rbig_reduction,
    mutual_information_rbig_reduction,
)

kwargs = dict(n_layers=50, rotation="pca", random_state=42)
model_x = AnnealedRBIG(**kwargs)
model_y = AnnealedRBIG(**kwargs)

model_x.fit(X)
model_y.fit(Y)

tc = total_correlation_rbig_reduction(model_x)
h  = entropy_rbig_reduction(model_x, X)
mi = mutual_information_rbig_reduction(model_x, model_y, X, Y, rbig_kwargs=kwargs)

print(f"TC(X):    {tc:.4f} nats")
print(f"H(X):     {h:.4f} nats")
print(f"MI(X; Y): {mi:.4f} nats")
```

### Model-level access (Level 2)

```python
model = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model.fit(joint)

# Per-layer TC values (recorded during fit)
print(f"Input TC:    {model.tc_per_layer_[0]:.4f}")
print(f"Residual TC: {model.tc_per_layer_[-1]:.4f}")
print(f"TC removed:  {model.total_correlation_reduction():.4f}")
```

### Change-of-variables approach (also available)

The package also supports the standard normalizing-flow density approach
using `log p(x) = log p_Z(f(x)) + log|det J|`:

```python
from rbig import entropy_rbig, mutual_information_rbig, kl_divergence_rbig

model_xy = AnnealedRBIG(**kwargs)
model_xy.fit(np.hstack([X, Y]))

H_cov   = entropy_rbig(model_xy, np.hstack([X, Y]))
mi_cov  = mutual_information_rbig(model_x, model_y, model_xy)
kld_cov = kl_divergence_rbig(model_x, Y)
```

### Available IT measures

| Measure | RBIG-way (recommended) | Change-of-variables |
|---------|----------------------|---------------------|
| Total Correlation | `estimate_tc(X)` | `total_correlation_rbig(X)` |
| Entropy | `estimate_entropy(X)` | `entropy_rbig(model, X)` |
| Mutual Information | `estimate_mi(X, Y)` | `mutual_information_rbig(m_x, m_y, m_xy)` |
| KL-Divergence | `estimate_kld(X, Y)` | `kl_divergence_rbig(model_P, X_Q)` |
| Log-Likelihood | — | `model.score_samples(X)` |
| TC Reduction | `model.total_correlation_reduction()` | — |

For detailed examples, see the [Information Theory notebook](notebooks/06_information_theory.ipynb) and [Dependence Detection notebooks](notebooks/09_dependence_1d.ipynb).
