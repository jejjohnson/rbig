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
# # Marginal Uniformization
#
# This notebook demonstrates how to use `MarginalUniformize` and
# `MarginalKDEGaussianize` from the new `rbig` API to transform marginal
# distributions to a uniform [0, 1] distribution.
#
# The marginal uniformization step is the first building block of the RBIG
# algorithm: before applying the probit (inverse Gaussian CDF) transform, we
# map each feature to the uniform distribution using an empirical CDF.

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from rbig import MarginalKDEGaussianize, MarginalUniformize

matplotlib.use("Agg")
plt.style.use("seaborn-v0_8-paper")

# %% [markdown]
# ## Data
#
# We draw samples from a Gamma distribution (a skewed, non-uniform marginal) to
# demonstrate the transform.

# %%
seed = 123
n_samples = 10_000
a = 4  # shape parameter for Gamma

# initialise data distribution
data_dist = stats.gamma(a=a)

# draw samples — shape (n_samples, 1) required by the new API
X = data_dist.rvs(size=(n_samples, 1), random_state=seed)

fig, ax = plt.subplots()
ax.set_title("Original Gamma Samples")
sns.histplot(X[:, 0], ax=ax, bins=50, kde=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Method I — Empirical CDF (MarginalUniformize)
#
# `MarginalUniformize` uses the empirical CDF (rank-based) to map each marginal
# to [0, 1].  It is deterministic and efficient for large datasets.

# %% [markdown]
# ### Fit the model

# %%
marg_unif = MarginalUniformize(bound_correct=True, eps=1e-6)
marg_unif.fit(X)

# %% [markdown]
# ### Transform: original → uniform

# %%
Xu = marg_unif.transform(X)

fig, ax = plt.subplots()
ax.set_title("After MarginalUniformize: should be ≈ Uniform[0,1]")
sns.histplot(Xu[:, 0], ax=ax, bins=50)
ax.set_xlabel("u")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inverse transform: uniform → original

# %%
X_approx = marg_unif.inverse_transform(Xu)

fig, ax = plt.subplots()
ax.set_title("After Inverse Transform: should recover original distribution")
sns.histplot(X_approx[:, 0], ax=ax, bins=50, kde=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Verify round-trip accuracy

# %%
residual = np.abs(X - X_approx).mean()
print(f"Mean absolute round-trip error: {residual:.4e}")

# %% [markdown]
# ## Method II — KDE-based Gaussianization (MarginalKDEGaussianize)
#
# `MarginalKDEGaussianize` estimates the CDF via Kernel Density Estimation (KDE)
# and then applies the probit transform Φ⁻¹ to map samples to a standard
# Gaussian distribution.  This is smoother than the empirical-CDF approach and
# produces a Gaussian (not uniform) output.

# %% [markdown]
# ### Fit the model

# %%
marg_kde = MarginalKDEGaussianize(bw_method="scott", eps=1e-6)
marg_kde.fit(X)

# %% [markdown]
# ### Transform: original → Gaussian

# %%
Xg = marg_kde.transform(X)

fig, ax = plt.subplots()
ax.set_title("After MarginalKDEGaussianize: should be ≈ N(0,1)")
sns.histplot(Xg[:, 0], ax=ax, bins=50, kde=True)
ax.set_xlabel("z")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inverse transform: Gaussian → original
#
# > **Note**: The KDE inverse transform uses a numerical root-finding algorithm
# > (`scipy.optimize.brentq`) so it is slower than the forward transform,
# > especially for large datasets.

# %%
# Use a small subset for the inverse to keep runtime reasonable
X_sub = X[:500]
Xg_sub = marg_kde.transform(X_sub)
X_approx_kde = marg_kde.inverse_transform(Xg_sub)

residual_kde = np.abs(X_sub - X_approx_kde).mean()
print(f"Mean absolute KDE round-trip error (n=500): {residual_kde:.4e}")

# %% [markdown]
# ### Comparison: empirical vs. KDE density estimate

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title("MarginalUniformize output")
sns.histplot(marg_unif.transform(X)[:, 0], ax=axes[0], bins=50)
axes[0].set_xlabel("u  (should be Uniform[0,1])")

axes[1].set_title("MarginalKDEGaussianize output")
sns.histplot(marg_kde.transform(X)[:, 0], ax=axes[1], bins=50, kde=True)
axes[1].set_xlabel("z  (should be N(0,1))")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# | Transform | Output distribution | Speed | Use case |
# |---|---|---|---|
# | `MarginalUniformize` | Uniform [0, 1] | Fast (rank-based) | Pre-processing step in RBIG |
# | `MarginalKDEGaussianize` | Standard Gaussian | Slower (KDE + root-find) | Smooth density estimation |
#
# Both transforms implement `.fit()`, `.transform()`, and `.inverse_transform()`
# following the scikit-learn estimator API.
