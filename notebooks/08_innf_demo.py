# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # RBIG as an Invertible Neural Network Flow (INNF Demo)
#
# This notebook demonstrates RBIG as an example of an **Invertible Neural
# Network Flow** (INNF) — a generative model based on composable invertible
# transformations.
#
# We show the RBIG algorithm step by step on a 2-D sin-wave dataset using the
# new composable API:
#
# - `RBIGLayer` — a single invertible layer (marginal Gaussianization + PCA)
# - `AnnealedRBIG` — the full iterative model

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from rbig import AnnealedRBIG, MarginalGaussianize, PCARotation, RBIGLayer

matplotlib.use("Agg")
sns.set_style("whitegrid")

# %% [markdown]
# ## Data
#
# 2-D sin-wave dataset (same as the original `innf_demo.ipynb`).
#
# > The original notebook used `make_toy_data("rbig_sin_wave", ...)` from the
# > `destructive-deep-learning` package.  We generate the data directly here.

# %%
seed = 123
rng = np.random.RandomState(seed=seed)
n_samples = 10_000

x = np.abs(2 * rng.randn(1, n_samples))
y = np.sin(x) + 0.25 * rng.randn(1, n_samples)
X = np.vstack((x, y)).T

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[:, 0], X[:, 1], s=1, c="red", alpha=0.4)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Original sin-wave data")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RBIG Algorithm — Step by Step (1 Layer)
#
# A single RBIG layer has two sub-steps:
#
# 1. **Marginal Gaussianization** — map each feature to N(0, 1) using its
#    empirical CDF and the probit function.
# 2. **Rotation (PCA)** — decorrelate the Gaussianized features via a whitening
#    PCA rotation.

# %%
# Build and fit a single layer
layer1 = RBIGLayer(
    marginal=MarginalGaussianize(),
    rotation=PCARotation(whiten=True),
)
layer1.fit(X)

# %% [markdown]
# ### Step I — After Marginal Gaussianization
#
# We can inspect the intermediate state by applying only the marginal transform.

# %%
X_mg = layer1.marginal.transform(X)

fig = plt.figure(figsize=(5, 5))
g = sns.jointplot(x=X_mg[:, 0], y=X_mg[:, 1], kind="hex", color="red")
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
g.ax_joint.set_title("After marginal Gaussianization")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step II — After Rotation (1 full RBIG layer)

# %%
X_layer1 = layer1.transform(X)

fig = plt.figure(figsize=(5, 5))
g = sns.jointplot(x=X_layer1[:, 0], y=X_layer1[:, 1], kind="hex", color="red")
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
g.ax_joint.set_title("After 1 RBIG layer (marginal + PCA rotation)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RBIG Algorithm — Multiple Layers
#
# We show the output after 1, 2, 3, 4, 5, and 6 layers.

# %%
n_layer_list = [1, 2, 3, 4, 5, 6]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, n in zip(axes.ravel(), n_layer_list, strict=False):
    model = AnnealedRBIG(
        n_layers=n,
        rotation="pca",
        zero_tolerance=n + 1,  # never stop early in this demo
        random_state=seed,
    )
    Z = model.fit_transform(X)
    ax.scatter(Z[:, 0], Z[:, 1], s=1, alpha=0.3, color="red")
    ax.set_title(f"{n} layer{'s' if n > 1 else ''}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("RBIG output at increasing numbers of layers", y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Full RBIG Algorithm (Converged)
#
# We now run `AnnealedRBIG` until the total correlation converges.

# %%
rbig_full = AnnealedRBIG(
    n_layers=1000,
    rotation="pca",
    zero_tolerance=30,
    random_state=seed,
)
rbig_full.fit(X)
Z_full = rbig_full.transform(X)

print(f"Converged after {len(rbig_full.layers_)} layers")

fig = plt.figure(figsize=(5, 5))
g = sns.jointplot(x=Z_full[:, 0], y=Z_full[:, 1], kind="hex", color="red")
g.ax_joint.set_xticks([])
g.ax_joint.set_yticks([])
g.ax_joint.set_title(f"Converged RBIG ({len(rbig_full.layers_)} layers)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Properties of the Learned Flow

# %% [markdown]
# ### Invertibility
#
# RBIG is an exact invertible transform.

# %%
X_reconstructed = rbig_full.inverse_transform(Z_full)
residual = np.abs(X - X_reconstructed).mean()
print(f"Mean absolute reconstruction error: {residual:.4e}")

# %% [markdown]
# ### TC Convergence Curve

# %%
fig, ax = plt.subplots()
ax.plot(rbig_full.tc_per_layer_)
ax.set_xlabel("Layer index")
ax.set_ylabel("Total Correlation (nats)")
ax.set_title("TC decreases as RBIG converges to N(0, I)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Log-Likelihood (Density Estimation)
#
# RBIG implements the change-of-variables formula for density estimation:
#
# $$\log p(x) = \log p_Z(f(x)) + \log|\det J_f(x)|$$

# %%
log_probs = rbig_full.score_samples(X)
print(
    f"Log-likelihood  — mean: {log_probs.mean():.3f}, "
    f"min: {log_probs.min():.3f}, max: {log_probs.max():.3f}"
)

fig, ax = plt.subplots()
h = ax.scatter(X[:, 0], X[:, 1], s=1, c=np.exp(log_probs), cmap="Reds")
ax.set_title("Data coloured by estimated density p(x)")
ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(h, ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Generative Sampling
#
# Because RBIG is invertible we can synthesize new samples by mapping standard
# Gaussian noise through the inverse transform.

# %%
X_synth = rbig_full.sample(n_samples=n_samples, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], s=1, alpha=0.3, c="red")
axes[0].set_title("Original data")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].scatter(X_synth[:, 0], X_synth[:, 1], s=1, alpha=0.3, c="blue")
axes[1].set_title("Synthesized data (RBIG samples)")
axes[1].set_xticks([])
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# RBIG implements a simple, composable normalizing flow:
#
# | Property | RBIG |
# |---|---|
# | Invertible | ✓ (exact) |
# | Density estimation | ✓ (`score_samples`) |
# | Generative sampling | ✓ (`sample`) or `inverse_transform` |
# | Convergence criterion | TC convergence |
# | Building block | `RBIGLayer` = `MarginalGaussianize` + `PCARotation` |
