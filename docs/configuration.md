# Configuration Guide

How to choose hyperparameters for `AnnealedRBIG`.

---

## Quick Start (Recommended Defaults)

```python
from rbig import AnnealedRBIG

model = AnnealedRBIG(
    n_layers=100,    # generous upper bound
    rotation="pca",  # best general-purpose choice
    patience=10,     # early stopping window
    tol=1e-5,        # convergence threshold
)
model.fit(X)
```

These defaults work well for most datasets. Read on to understand when and
how to adjust them.

---

## Parameters

### `n_layers` (int, default=100)

Maximum number of RBIG layers (iterations). Each layer applies one marginal
Gaussianization + one rotation.

| Scenario | Suggested range |
|----------|:--------------:|
| Low-dimensional (D ≤ 5), simple structure | 20–50 |
| Moderate (5 < D ≤ 50) | 50–100 |
| High-dimensional (D > 50) or strong nonlinearity | 100–200 |

In practice, **early stopping via `patience`** usually halts training well
before `n_layers` is reached. Setting `n_layers` too low risks under-fitting;
setting it high just means unused budget — the computational cost is only
paid for layers actually fitted.

Check `len(model.layers_)` after fitting to see how many layers were used.

---

### `rotation` (str, default="pca")

Which orthogonal rotation to apply after each marginal Gaussianization step.

| Value | Method | When to use |
|-------|--------|------------|
| `"pca"` | PCA with whitening | **Default.** Removes second-order dependence explicitly → fastest convergence in most cases. |
| `"ica"` | Independent Component Analysis | Data with known independent source structure. Slower per layer than PCA. |
| `"random"` | Haar-distributed random orthogonal | Very high-dimensional data (PCA eigendecomposition is $O(D^3)$). Also useful for ensemble averaging of IT estimates. |

**Rule of thumb:** Start with `"pca"`. Only switch to `"random"` if
`D` is large enough that PCA becomes a bottleneck, or if you need
stochastic diversity across runs.

See the [Rotation Choices notebook](notebooks/08_rotation_choices.ipynb) for
a visual comparison.

---

### `patience` (int, default=10)

Number of consecutive layers where the total correlation (TC) change is
below `tol` before training stops early.

| Value | Effect |
|-------|--------|
| Small (3–5) | Aggressive early stopping. Faster, but may stop too soon on noisy data. |
| Default (10) | Good balance. Tolerates short plateaus before converging. |
| Large (20+) | Conservative. Use when TC curve has long flat regions before dropping again. |

**Tip:** Plot `model.tc_per_layer_` after fitting. If the curve has a long
tail of small-but-nonzero TC reductions, increase patience.

---

### `tol` (float or "auto", default=1e-5)

Convergence threshold: training stops when `|TC(k) − TC(k−1)| < tol` for
`patience` consecutive layers.

| Value | Effect |
|-------|--------|
| `1e-3` | Loose. Converges quickly but may leave residual dependence. |
| `1e-5` (default) | Tight. Good for density estimation and IT measures. |
| `1e-7` | Very tight. Use for high-precision entropy/MI estimates. |
| `"auto"` | Adaptive: sets tol based on number of training samples. |

**When to use `"auto"`:** When you're unsure about the right tolerance and
want a data-driven choice. The adaptive lookup scales tol with sample size
(more samples → tighter tolerance).

---

### `random_state` (int or None, default=None)

Seed for stochastic components (ICA initialization, random rotations). Set
this for **reproducible results**. Has no effect when `rotation="pca"`
(PCA is deterministic).

---

### `strategy` (list or None, default=None)

Per-layer override for rotation and marginal choices. Each entry can be:

- A string: rotation name (e.g., `"pca"`, `"random"`)
- A tuple: `(rotation_name, marginal_name)`

The list cycles if shorter than `n_layers`.

```python
# Example: alternate PCA and random rotations
model = AnnealedRBIG(
    n_layers=100,
    strategy=["pca", "random"],
)
```

This is an advanced feature — the default (`None`) uses the same rotation
for every layer, which is fine for nearly all use cases.

---

### `verbose` (bool or int, default=False)

| Value | Behaviour |
|-------|-----------|
| `False` / `0` | Silent |
| `True` / `1` | Progress bar during `fit()` |
| `2` | Progress bars for `fit`, `transform`, `inverse_transform`, and `score_samples` |

---

## Diagnosing Fit Quality

After fitting, inspect these attributes:

```python
# How many layers were actually used?
print(f"Layers: {len(model.layers_)}")

# Did TC converge?
import matplotlib.pyplot as plt
plt.plot(model.tc_per_layer_)
plt.xlabel("Layer")
plt.ylabel("Total Correlation (nats)")
plt.title("TC convergence")
plt.show()

# Final TC (should be near zero for good fit)
print(f"Final TC: {model.tc_per_layer_[-1]:.6f}")
```

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `len(model.layers_) == n_layers` | Hit the layer budget before converging | Increase `n_layers` |
| TC curve still dropping at the end | Stopped too early | Increase `n_layers` or `patience` |
| TC curve flat but nonzero | Residual higher-order dependence | Try `rotation="ica"` or increase `n_layers` |
| TC oscillates | Numerical instability (rare) | Check for constant/duplicate features in input data |

---

## Computational Cost

Each RBIG layer has cost approximately:

| Step | Complexity |
|------|-----------|
| Marginal Gaussianization | $O(N \cdot D \cdot \log N)$ — sorting per feature |
| PCA rotation | $O(N \cdot D^2 + D^3)$ — covariance + eigendecomposition |
| Random rotation | $O(N \cdot D^2)$ — matrix multiply only |
| TC estimation | $O(N \cdot D)$ — KDE marginal entropies |

For large $D$, the PCA eigendecomposition dominates. Switching to
`rotation="random"` removes the $D^3$ term at the cost of more layers.

---

## Marginal Transform Choice

By default, `AnnealedRBIG` uses `MarginalGaussianize` (empirical CDF + probit)
inside each layer. For custom marginals, use the `strategy` parameter or
build your own pipeline with `RBIGLayer`:

```python
from rbig import RBIGLayer, SplineGaussianizer, PCARotation

layer = RBIGLayer(
    marginal=SplineGaussianizer(n_quantiles=200),
    rotation=PCARotation(whiten=True),
)
```

See the [Marginal Transforms notebook](notebooks/01_marginal_transforms.ipynb)
for a comparison of all 7 available marginal methods.
