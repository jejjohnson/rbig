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
# # Information Theory Measures with RBIG
#
# This notebook demonstrates how to estimate classical information-theoretic
# quantities using the new `rbig` functional API:
#
# | Measure | Function |
# |---|---|
# | Total Correlation | `total_correlation_rbig(X)` |
# | Entropy | `marginal_entropy(X)`, `AnnealedRBIG.entropy()` |
# | Mutual Information | `mutual_information_rbig(model_X, model_Y, model_XY)` |
# | KL Divergence | `kl_divergence_rbig(model_P, X_Q)` |

# %%
import numpy as np
from sklearn.utils import check_random_state

from rbig import (
    AnnealedRBIG,
    kl_divergence_rbig,
    marginal_entropy,
    mutual_information_rbig,
    total_correlation_rbig,
)


# %% [markdown]
# ---
# ## Total Correlation
#
# Total Correlation (TC), also called multi-information, measures the statistical
# dependence among all variables:
#
# $$\mathrm{TC}(X) = \sum_i H(X_i) - H(X)$$

# %% [markdown]
# ### Sample Data

# %%
n_samples = 1_000
d_dimensions = 3
seed = 123

rng = check_random_state(seed)

# Generate correlated Gaussian data via a random linear mixing matrix
data_original = rng.randn(n_samples, d_dimensions)
A = rng.rand(d_dimensions, d_dimensions)
data = data_original @ A

# Covariance matrix
C = A.T @ A
vv = np.diag(C)

# %% [markdown]
# ### Analytical TC for Gaussian data
#
# For a multivariate Gaussian $X \sim \mathcal{N}(0, \Sigma)$:
#
# $$\mathrm{TC}(X) = \sum_i \frac{1}{2}\log\sigma_i^2 - \frac{1}{2}\log|\Sigma|$$

# %%
tc_analytical = np.log(np.sqrt(vv)).sum() - 0.5 * np.log(np.linalg.det(C))
print(f"TC (analytical): {tc_analytical:.4f} nats")

# %% [markdown]
# ### RBIG-based TC estimate
#
# `total_correlation_rbig(X)` directly estimates TC from samples without fitting
# a full RBIG model.

# %%
tc_rbig = total_correlation_rbig(data)
print(f"TC (RBIG/direct): {tc_rbig:.4f} nats")

# %% [markdown]
# We can also fit a full `AnnealedRBIG` model and use the TC tracked per layer.
# After convergence the sum of per-layer TC reductions approximates the initial TC.

# %%
rbig_tc_model = AnnealedRBIG(
    n_layers=30,
    rotation="pca",
    patience=10,
    random_state=seed,
)
rbig_tc_model.fit(data)

# tc_per_layer_ records TC *after* each layer; use total_correlation_rbig for the
# initial (pre-RBIG) TC of the raw data.
tc_rbig_model = total_correlation_rbig(data)
print(f"TC (AnnealedRBIG estimate from raw data): {tc_rbig_model:.4f} nats")

# %% [markdown]
# ---
# ## Entropy

# %% [markdown]
# ### Sample Data

# %%
n_samples = 1_000
rng = check_random_state(seed)

data_original = rng.randn(n_samples, d_dimensions)
A = rng.rand(d_dimensions, d_dimensions)
data = data_original @ A

# %% [markdown]
# ### Gaussian plug-in entropy estimate
#
# We use the Gaussian entropy formula as a reference,
# $$H(X) = \frac{1}{2}\log|2\pi e \Sigma|$$
# but the value below is a plug-in / estimated quantity based on sampled data.

# %%
Hx_marginals = marginal_entropy(data_original)
H_analytical = Hx_marginals.sum() + np.log(np.abs(np.linalg.det(A)))
print(f"H (Gaussian plug-in estimate): {H_analytical:.4f} nats")

# %% [markdown]
# ### RBIG entropy estimate
#
# `AnnealedRBIG.entropy()` returns the differential entropy in nats.

# %%
ent_rbig_model = AnnealedRBIG(
    n_layers=30,
    rotation="pca",
    patience=10,
    random_state=seed,
)
ent_rbig_model.fit(data)
H_rbig = ent_rbig_model.entropy()
print(f"H (RBIG):        {H_rbig:.4f} nats")

# %% [markdown]
# ---
# ## Mutual Information
#
# Mutual information between two random vectors X and Y:
#
# $$\mathrm{MI}(X;Y) = H(X) + H(Y) - H(X,Y)$$
#
# Using RBIG we fit separate models for X, Y, and [X, Y] concatenated.

# %% [markdown]
# ### Sample Data

# %%
n_samples = 1_000
rng = check_random_state(seed)

A = rng.rand(2 * d_dimensions, 2 * d_dimensions)
C = A @ A.T
mu = np.zeros(2 * d_dimensions)

dat_all = rng.multivariate_normal(mu, C, n_samples)

CX = C[:d_dimensions, :d_dimensions]
CY = C[d_dimensions:, d_dimensions:]

X = dat_all[:, :d_dimensions]
Y = dat_all[:, d_dimensions:]

# %% [markdown]
# ### Analytical MI for Gaussian data

# %%
H_X = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * CX))
H_Y = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * CY))
H = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * C))

mi_analytical = H_X + H_Y - H
print(f"MI (analytical): {mi_analytical:.4f} nats")

# %% [markdown]
# ### RBIG-based MI estimate
#
# `mutual_information_rbig(model_X, model_Y, model_XY)` computes
# MI = H(X) + H(Y) - H(X,Y) using fitted `AnnealedRBIG` models.

# %%
rbig_kwargs = dict(
    n_layers=30, rotation="pca", patience=10, random_state=seed
)

model_X = AnnealedRBIG(**rbig_kwargs).fit(X)
model_Y = AnnealedRBIG(**rbig_kwargs).fit(Y)
model_XY = AnnealedRBIG(**rbig_kwargs).fit(np.hstack([X, Y]))

mi_rbig = mutual_information_rbig(model_X, model_Y, model_XY)
print(f"MI (RBIG):       {mi_rbig:.4f} nats")

# %% [markdown]
# ---
# ## Kullback-Leibler Divergence (KLD)
#
# $$\mathrm{KL}(P \| Q) = \int p(x) \log\frac{p(x)}{q(x)}\,dx$$
#
# We estimate this using RBIG by:
# 1. Fitting a model on samples from **P**.
# 2. Evaluating that model's log-likelihood on samples from **Q**.
#
# `kl_divergence_rbig(model_P, X_Q)` computes `KL(P‖Q)`.

# %% [markdown]
# ### Sample Data

# %%
n_samples = 1_000
mu_offset = 0.4  # controls how different the two distributions are
rng = check_random_state(seed)

A = rng.rand(d_dimensions, d_dimensions)
cov = A @ A.T
cov = cov / cov.max()  # normalise

mu_x = np.zeros(d_dimensions)
mu_y = np.ones(d_dimensions) * mu_offset

X_p = rng.multivariate_normal(mu_x, cov, n_samples)
X_q = rng.multivariate_normal(mu_y, cov, n_samples)

# %% [markdown]
# ### Analytical KLD for Gaussian distributions
#
# Since both distributions share the same covariance `cov`, the formula simplifies:
#
# $$\mathrm{KL}(\mathcal{N}(\mu_x,\Sigma) \| \mathcal{N}(\mu_y,\Sigma))
# = \frac{1}{2}(\mu_y - \mu_x)^\top \Sigma^{-1} (\mu_y - \mu_x)$$

# %%
# Use the known covariance matrix (not sample estimates) for the exact value
kld_analytical = 0.5 * (mu_y - mu_x) @ np.linalg.inv(cov) @ (mu_y - mu_x).T
print(f"KLD (analytical): {kld_analytical:.4f} nats")

# %% [markdown]
# ### RBIG-based KLD estimate

# %%
kld_rbig_model = AnnealedRBIG(
    n_layers=30,
    rotation="pca",
    patience=10,
    random_state=seed,
)
kld_rbig_model.fit(X_p)

kld_rbig = kl_divergence_rbig(kld_rbig_model, X_q)
print(f"KLD (RBIG):       {kld_rbig:.4f} nats")

# %% [markdown]
# ---
# ## Summary
#
# | Measure | Old API | New API |
# |---|---|---|
# | Total Correlation | `RBIG(...).fit(X).mutual_information * log(2)` | `total_correlation_rbig(X)` |
# | Entropy | `RBIG(...).fit(X).entropy(correction=True) * log(2)` | `AnnealedRBIG(...).fit(X).entropy()` |
# | Mutual Info | `RBIGMI(...).fit(X,Y).mutual_information() * log(2)` | `mutual_information_rbig(mX, mY, mXY)` |
# | KL Divergence | `RBIGKLD(...).fit(X,Y).kld * log(2)` | `kl_divergence_rbig(model_P, X_Q)` |
#
# The new API uses **nats** (natural logarithm) throughout; multiply by
# `np.log2(np.e) ≈ 1.4427` to convert to bits.
