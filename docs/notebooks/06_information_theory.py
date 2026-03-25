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
# # Information Theory Measures with RBIG
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/rbig/blob/main/docs/notebooks/06_information_theory.ipynb)
#
# This notebook demonstrates two approaches for estimating information-theoretic
# quantities with RBIG:
#
# | Approach | Method | Pros |
# |---|---|---|
# | **RBIG-way** (recommended) | Per-layer TC reduction | No Jacobian estimation; more robust |
# | **Change-of-variables** | `log p(x) = log p_Z(f(x)) + log\|det J\|` | Fast (cached Jacobian); standard NF technique |
#
# For mathematical definitions, see the [Information Theory Measures note](../notes/information_theory_measures.md).

# %% [markdown]
# > **Colab / fresh environment?** Run the cell below to install `rbig` from
# > GitHub. Skip if already installed.

# %%
# !pip install "rbig[all] @ git+https://github.com/jejjohnson/rbig.git" -q

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
diff = mu_y - mu_x
kld_analytical = 0.5 * diff @ np.linalg.solve(cov, diff.T)
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
# ## RBIG-way Estimation (Per-Layer TC Reduction)
#
# The **RBIG-way** approach (Laparra et al. 2011, 2020) avoids Jacobian
# estimation entirely. Instead, it tracks how much total correlation each
# layer removes and sums those reductions.
#
# The `estimate_*` functions provide the simplest API — just pass data:

# %%
from rbig import estimate_tc, estimate_entropy, estimate_mi, estimate_kld

# Reuse the correlated data from the MI section above
rbig_kw = dict(n_layers=30, rotation="pca", patience=10, random_state=seed)

tc_red = estimate_tc(dat_all, **rbig_kw)
h_red = estimate_entropy(dat_all, **rbig_kw)
mi_red = estimate_mi(X, Y, **rbig_kw)

print(f"TC  (RBIG-way): {tc_red:.4f} nats")
print(f"H   (RBIG-way): {h_red:.4f} nats")
print(f"MI  (RBIG-way): {mi_red:.4f} nats")

# %% [markdown]
# ### KLD via RBIG-way

# %%
kld_red = estimate_kld(X_p, X_q, **rbig_kw)
print(f"KLD (RBIG-way):  {kld_red:.4f} nats")
print(f"KLD (analytical): {kld_analytical:.4f} nats")

# %% [markdown]
# ### Comparison: Change-of-Variables vs RBIG-way
#
# Both approaches should agree for well-converged models.  The RBIG-way
# approach is generally more robust because it avoids estimating
# `log|det J|`, which can introduce bias.

# %%
from rbig import entropy_rbig_reduction, total_correlation_rbig_reduction

# Compare on the MI data
model_joint = AnnealedRBIG(**rbig_kw).fit(dat_all)

h_cov = model_joint.entropy()  # change-of-variables
h_red2 = entropy_rbig_reduction(model_joint, dat_all)  # RBIG-way
tc_cov = total_correlation_rbig(dat_all)  # single-shot KDE+Gaussian
tc_red2 = total_correlation_rbig_reduction(model_joint)  # RBIG-way

print(f"{'Measure':<8} {'Change-of-Vars':>14} {'RBIG-way':>10} {'Analytical':>12}")
print(f"{'H':8} {h_cov:14.4f} {h_red2:10.4f} {H:12.4f}")
print(f"{'TC':8} {tc_cov:14.4f} {tc_red2:10.4f} {'':>12}")

# %% [markdown]
# ---
# ## Summary
#
# | Measure | RBIG-way (`estimate_*`) | Change-of-Variables |
# |---|---|---|
# | Total Correlation | `estimate_tc(X)` | `total_correlation_rbig(X)` |
# | Entropy | `estimate_entropy(X)` | `entropy_rbig(model, X)` / `model.entropy()` |
# | Mutual Info | `estimate_mi(X, Y)` | `mutual_information_rbig(mX, mY, mXY)` |
# | KL Divergence | `estimate_kld(X, Y)` | `kl_divergence_rbig(model_P, X_Q)` |
#
# All values are in **nats** (natural logarithm); multiply by
# `np.log2(np.e) ≈ 1.4427` to convert to bits.

# %% [markdown]
# ---
# ## See Also
#
# - [Information Theory Measures](../notes/information_theory_measures.md) — formal definitions of TC, entropy, MI, and KLD
# - [Measuring Dependence: 1D Variables](./09_dependence_1d.ipynb) — MI for detecting nonlinear dependence in 1D
# - [Measuring Dependence: 2D Variables](./10_dependence_2d.ipynb) — MI for multivariate dependence
# - [Information Theory with Synthetic Stock-Market Data](./11_real_world_it.ipynb) — practical IT application
