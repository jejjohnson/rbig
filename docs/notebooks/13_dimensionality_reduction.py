# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Dimensionality-Reducing Rotations
#
# Standard RBIG rotations are **square** (D → D) and bijective. But `rbig` also
# provides rotations that **reduce dimensionality** (D → K, where K < D). These
# are useful for high-dimensional data where full RBIG is too expensive, but they
# come with important trade-offs.
#
# **Contents**
#
# 1. [Theory — Bijectivity and Density Estimation](#theory)
# 2. [Available Methods](#available-methods)
# 3. [PCA with Truncation](#pca-with-truncation)
# 4. [OrthogonalDimensionalityReduction](#orthogonaldimensionalityreduction)
# 5. [RandomOrthogonalProjection](#randomorthogonalprojection)
# 6. [GaussianRandomProjection](#gaussianrandomprojection)
# 7. [Comparison](#comparison)
# 8. [Summary](#summary)

# %% [markdown]
# ---
# ## Theory
#
# ### Bijectivity and the Jacobian
#
# A normalizing flow requires **bijective** (invertible) transforms so that the
# change-of-variables formula holds:
#
# $$\log p(x) = \log p_z(f(x)) + \log|\det J_f(x)|$$
#
# When K = D, the rotation is square and orthogonal, so $|\det J| = 1$ and the
# transform is exactly invertible.
#
# **When K < D**, the transform is a projection — it discards D − K dimensions
# of information. This means:
#
# - The transform is **not bijective** (you cannot recover the original data)
# - The **Jacobian determinant is undefined** (the Jacobian is not square)
# - **Density estimation via change-of-variables is invalid**
#
# ### When Is Dim-Reduction Useful?
#
# Even though we lose bijectivity, dimensionality reduction is valuable for:
#
# - **Exploratory analysis** — visualize or summarize high-dimensional data
# - **Feature extraction** — reduce dimensionality before a downstream task
# - **Computational savings** — RBIG on K << D features is much faster
# - **Approximate density estimation** — if most variance is captured in K
#   components, the density estimate may still be useful (with caveats)

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from rbig import (
    PCARotation,
    RandomOrthogonalProjection,
    GaussianRandomProjection,
    OrthogonalDimensionalityReduction,
)

# %% [markdown]
# ---
# ## Available Methods
#
# | Class | Method | Orthogonal? | Inverse (K<D)? | log\|det J\| (K<D)? |
# |-------|--------|:-----------:|:--------------:|:-------------------:|
# | `PCARotation` | PCA eigenvectors | Yes | Lossy reconstruction | Undefined |
# | `OrthogonalDimensionalityReduction` | Haar rotation + truncation | Yes | Not supported | Not supported |
# | `RandomOrthogonalProjection` | QR of random matrix | Yes (columns) | Not supported | Not supported |
# | `GaussianRandomProjection` | Gaussian random matrix | No | Pseudoinverse (approx) | Returns 0 (approx) |

# %% [markdown]
# ---
# ## Data
#
# We create a 10-D dataset with most variance concentrated in 3 directions,
# mimicking a common real-world scenario where data lives on a low-dimensional
# manifold.

# %%
rng = np.random.default_rng(42)
N = 1_000
D = 10
K = 3  # target dimensionality

# 3 strong signal directions + 7 noise directions
signal = rng.standard_normal((N, 3)) * np.array([5.0, 3.0, 1.5])
noise = 0.1 * rng.standard_normal((N, 7))
data_full = np.hstack([signal, noise])

# Apply a random rotation so the structure isn't axis-aligned
Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
X = data_full @ Q.T

print(f"Data shape: {X.shape}")
print(f"Singular values: {np.linalg.svd(X, compute_uv=False)[:6].round(1)}")

# %% [markdown]
# The first 3 singular values are much larger than the rest — the data is
# effectively 3-dimensional embedded in 10-D space.

# %% [markdown]
# ---
# ## PCA with Truncation
#
# `PCARotation(n_components=K)` retains only the top-K principal components.
# The inverse maps back to D dimensions but **loses the discarded components**.

# %%
pca = PCARotation(n_components=K, whiten=True)
pca.fit(X)
Z_pca = pca.transform(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {Z_pca.shape}")

# Lossy reconstruction
X_rec_pca = pca.inverse_transform(Z_pca)
rec_err = np.mean((X - X_rec_pca) ** 2)
print(f"Reconstruction MSE: {rec_err:.4f}")

# %%
# Explained variance
var_explained = pca.pca_.explained_variance_ratio_
print("Explained variance per component:")
for i, v in enumerate(var_explained):
    bar = "█" * int(v * 50)
    print(f"  PC {i}: {v:.3f} {bar}")
print(f"  Total: {var_explained.sum():.3f}")

# %% [markdown]
# ---
# ## OrthogonalDimensionalityReduction
#
# Applies a full D×D Haar-random orthogonal rotation, then **truncates** to
# the first K dimensions. Unlike PCA, the rotation is random (not
# data-dependent), so it doesn't preferentially capture high-variance directions.

# %%
orth_dr = OrthogonalDimensionalityReduction(n_components=K, random_state=42)
orth_dr.fit(X)
Z_orth = orth_dr.transform(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {Z_orth.shape}")

# Inverse is not supported for K < D
try:
    orth_dr.inverse_transform(Z_orth)
except NotImplementedError as e:
    print(f"Inverse: {e}")

# %% [markdown]
# ---
# ## RandomOrthogonalProjection
#
# Generates a semi-orthogonal D×K matrix (orthonormal columns) via QR
# decomposition. Projects data onto a random K-dimensional subspace.

# %%
rand_orth = RandomOrthogonalProjection(n_components=K, random_state=42)
rand_orth.fit(X)
Z_rand_orth = rand_orth.transform(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {Z_rand_orth.shape}")
print(f"Projection matrix shape: {rand_orth.projection_matrix_.shape}")

# Verify orthonormality of columns
P = rand_orth.projection_matrix_
print(f"P^T P ≈ I_K: {np.allclose(P.T @ P, np.eye(K), atol=1e-10)}")

# %% [markdown]
# ---
# ## GaussianRandomProjection
#
# Uses a random Gaussian matrix (entries ~ N(0, 1/K)) for
# Johnson-Lindenstrauss style projection. Columns are **not orthogonal**, but
# pairwise distances are approximately preserved.
#
# This is the only dim-reducing rotation that provides an approximate inverse
# (via pseudoinverse) and an approximate log-det-Jacobian (returns 0).

# %%
gauss_proj = GaussianRandomProjection(n_components=K, random_state=42)
gauss_proj.fit(X)
Z_gauss = gauss_proj.transform(X)

print(f"Input shape:  {X.shape}")
print(f"Output shape: {Z_gauss.shape}")

# Approximate inverse via pseudoinverse
X_rec_gauss = gauss_proj.inverse_transform(Z_gauss)
rec_err_gauss = np.mean((X - X_rec_gauss) ** 2)
print(f"Reconstruction MSE (pseudoinverse): {rec_err_gauss:.4f}")

# %% [markdown]
# ---
# ## Comparison
#
# ### Variance Captured
#
# We measure how much of the original data's variance each projection retains.

# %%
total_var = np.var(X, axis=0).sum()

projections = {
    "PCA (K=3)": Z_pca,
    "OrthogonalDimRed": Z_orth,
    "RandomOrthProj": Z_rand_orth,
    "GaussianRandProj": Z_gauss,
}

print(f"{'Method':<22s} {'Var retained':>14s} {'% of total':>12s}")
print("-" * 50)
for name, Z in projections.items():
    var_retained = np.var(Z, axis=0).sum()
    pct = 100 * var_retained / total_var
    print(f"{name:<22s} {var_retained:>14.2f} {pct:>11.1f}%")

# %% [markdown]
# PCA retains the most variance by design — it picks the directions of maximum
# variance. Random projections retain less because they don't know which
# directions matter.

# %% [markdown]
# ### Pairwise Distance Preservation
#
# The Johnson-Lindenstrauss lemma guarantees that random projections
# approximately preserve pairwise distances. Let's verify.

# %%
from scipy.spatial.distance import pdist

# Use a subset for speed
idx = rng.choice(N, size=200, replace=False)
d_orig = pdist(X[idx])

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, Z) in zip(axes, projections.items()):
    d_proj = pdist(Z[idx])
    ax.scatter(d_orig, d_proj, s=2, alpha=0.3)
    ax.plot([0, d_orig.max()], [0, d_orig.max()], "r--", alpha=0.5)
    corr = np.corrcoef(d_orig, d_proj)[0, 1]
    ax.set_title(f"{name}\nr = {corr:.3f}", fontsize=10)
    ax.set_xlabel("Original distances")
    ax.set_ylabel("Projected distances")

plt.suptitle("Pairwise distance preservation (K=3, D=10)", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Reconstruction Error

# %%
reconstructors = {
    "PCA (K=3)": (pca, Z_pca),
    "GaussianRandProj": (gauss_proj, Z_gauss),
}

print(f"{'Method':<22s} {'MSE':>10s} {'Max |err|':>10s}")
print("-" * 44)
for name, (model, Z) in reconstructors.items():
    X_rec = model.inverse_transform(Z)
    mse = np.mean((X - X_rec) ** 2)
    max_err = np.abs(X - X_rec).max()
    print(f"{name:<22s} {mse:>10.4f} {max_err:>10.4f}")

print("\n(OrthogonalDimRed and RandomOrthProj do not support inverse_transform for K < D)")

# %% [markdown]
# ---
# ## Summary
#
# ### When to Use Each Method
#
# | Method | Use when... |
# |--------|------------|
# | `PCARotation(n_components=K)` | You want **maximum variance** in K dims; reconstruction needed |
# | `OrthogonalDimensionalityReduction` | You want a random but **orthogonal** projection |
# | `RandomOrthogonalProjection` | You want an **orthonormal** projection matrix (columns are orthonormal) |
# | `GaussianRandomProjection` | You need **approximate inverse** or JL distance preservation |
#
# ### Bijectivity Trade-offs
#
# | Property | K = D | K < D |
# |----------|:-----:|:-----:|
# | Bijective | Yes | **No** |
# | Exact inverse | Yes | **No** (lossy or unsupported) |
# | log\|det J\| | 0 | **Undefined** (or approximate) |
# | Valid for density estimation | Yes | **No** (approximate at best) |
#
# **Rule of thumb**: If you need density estimation or generative sampling
# (the core RBIG use case), use **square rotations** (K = D). Use
# dimensionality-reducing rotations for exploration, visualization, or
# preprocessing before a non-density-estimation task.
