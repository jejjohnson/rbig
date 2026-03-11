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
# # Marginal Gaussianization via Inverse CDF
#
# This notebook demonstrates how to use `MarginalGaussianize` from the new
# `rbig` API to transform marginal distributions to a standard Gaussian via the
# empirical CDF followed by the probit (Φ⁻¹) transform.
#
# This "inverse-CDF Gaussianization" is the core marginal step inside each RBIG
# layer.

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from rbig import MarginalGaussianize

plt.style.use("seaborn-v0_8-paper")

# %% [markdown]
# ## Data
#
# We start from a **Uniform[0, 1]** distribution to cleanly illustrate the
# inverse-CDF step: the empirical CDF of uniform data is approximately the
# identity, so the probit transform maps it directly to N(0, 1).

# %%
seed = 123
n_samples = 10_000

# Uniform distribution — easy to verify the Gaussianization analytically
data_dist = stats.uniform()
X = data_dist.rvs(size=(n_samples, 1), random_state=seed)

fig, ax = plt.subplots()
ax.set_title("Original Uniform[0,1] Samples")
sns.histplot(X[:, 0], ax=ax, bins=50)
ax.set_xlabel("x")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## MarginalGaussianize
#
# `MarginalGaussianize` performs the following two-step transform **per feature**:
#
# 1. Map to uniform via empirical CDF: $u_i = \hat{F}(x_i)$
# 2. Apply the probit: $z_i = \Phi^{-1}(u_i)$
#
# After fitting on training data the transform is applied to any new samples
# sharing the same marginal distribution.

# %% [markdown]
# ### Fit the model

# %%
marg_gauss = MarginalGaussianize(bound_correct=True, eps=1e-6)
marg_gauss.fit(X)

# %% [markdown]
# ### Transform: original → Gaussian

# %%
Xg = marg_gauss.transform(X)

fig, ax = plt.subplots()
ax.set_title("After MarginalGaussianize: should be ≈ N(0,1)")
sns.histplot(Xg[:, 0], ax=ax, bins=50, kde=True)
ax.set_xlabel("z")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Inverse transform: Gaussian → original

# %%
X_approx = marg_gauss.inverse_transform(Xg)

fig, ax = plt.subplots()
ax.set_title("After Inverse Transform: should recover Uniform[0,1]")
sns.histplot(X_approx[:, 0], ax=ax, bins=50)
ax.set_xlabel("x")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Verify round-trip accuracy

# %%
residual = np.abs(X - X_approx).mean()
print(f"Mean absolute round-trip error: {residual:.4e}")

# %% [markdown]
# ### Log-determinant of the Jacobian
#
# For normalizing flows and density estimation we need the log |det J| of the
# transform.  `MarginalGaussianize` provides `log_det_jacobian(X)` which returns
# a per-sample scalar (sum over features).

# %%
log_jac = marg_gauss.log_det_jacobian(X)
print(
    f"Log |det J|  — min: {log_jac.min():.3f}, max: {log_jac.max():.3f}, "
    f"mean: {log_jac.mean():.3f}"
)

# %% [markdown]
# ## Generalisation: Skewed Distribution
#
# Let us also apply `MarginalGaussianize` to a Gamma-distributed input to show
# that it correctly handles non-symmetric distributions.

# %%
a = 4
data_gamma = stats.gamma(a=a)
X_gamma = data_gamma.rvs(size=(n_samples, 1), random_state=seed)

marg_gauss_gamma = MarginalGaussianize()
marg_gauss_gamma.fit(X_gamma)
Xg_gamma = marg_gauss_gamma.transform(X_gamma)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].set_title(f"Original Gamma(a={a})")
sns.histplot(X_gamma[:, 0], ax=axes[0], bins=50, kde=True)
axes[0].set_xlabel("x")

axes[1].set_title("After MarginalGaussianize")
sns.histplot(Xg_gamma[:, 0], ax=axes[1], bins=50, kde=True)
axes[1].set_xlabel("z")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary
#
# `MarginalGaussianize` provides:
#
# | Method | Description |
# |---|---|
# | `fit(X)` | Stores sorted support for each feature |
# | `transform(X)` | Empirical CDF → probit → Gaussian |
# | `inverse_transform(X)` | Gaussian → uniform CDF → original support |
# | `log_det_jacobian(X)` | Per-sample log |det J| for density estimation |
#
# This transform is the marginal step used inside every `RBIGLayer`.
