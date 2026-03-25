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
# A key advantage of RBIG is that information-theoretic quantities fall out
# naturally from the Gaussianization process. This notebook demonstrates
# **two approaches** for estimating them, and compares both against analytical
# values on Gaussian data where ground truth is available.
#
# <figure align="center">
# <img src="../pics/rbig_it/Fig_1.png" alt="IT measures" width="500">
# <figcaption>Information Theory measures computable via RBIG.</figcaption>
# </figure>
#
# | Approach | How it works | Pros |
# |----------|-------------|------|
# | **Change-of-variables** | Fit a full RBIG model, use the learned density | Single model, exact density |
# | **RBIG-way** | Sum per-layer TC reductions | No Jacobian needed, more stable |
#

# %% [markdown]
# > **Colab / fresh environment?** Run the cell below to install `rbig` from
# > GitHub. Skip if already installed.

# %%
# !pip install "rbig[all] @ git+https://github.com/jejjohnson/rbig.git" -q

# %%
import numpy as np
from scipy import stats

from rbig import (
    AnnealedRBIG,
    entropy_rbig,
    estimate_entropy,
    estimate_kld,
    estimate_mi,
    estimate_tc,
    kl_divergence_rbig,
    marginal_entropy,
    mutual_information_rbig,
    total_correlation_rbig,
)

# %% [markdown]
# ---
# ## Shared Data Setup
#
# We use correlated Gaussian data throughout so we can compare all estimates
# against **closed-form analytical values**.

# %%
seed = 42
rng = np.random.RandomState(seed)
n_samples = 1_000
d = 2  # dimensionality per block

# Joint covariance for [X, Y] with cross-correlations
C_full = np.eye(2 * d)
C_full[0, d] = C_full[d, 0] = 0.8  # x0 ↔ y0
C_full[1, d + 1] = C_full[d + 1, 1] = 0.5  # x1 ↔ y1

joint = rng.multivariate_normal(np.zeros(2 * d), C_full, size=n_samples)
X = joint[:, :d]
Y = joint[:, d:]
XY = joint

# Marginal covariances
CX = C_full[:d, :d]
CY = C_full[d:, d:]

# For KLD: shifted-mean distribution
mu_shift = np.array([0.5, 0.0])
X_shifted = rng.multivariate_normal(mu_shift, CX, size=n_samples)

# Common RBIG settings
rbig_kw = dict(n_layers=20, rotation="pca", patience=10, random_state=seed)

# %% [markdown]
# ---
# ## 1. Total Correlation
#
# Total correlation measures the overall statistical dependence among all
# dimensions of a random vector — how far the joint distribution is from
# being fully independent:
#
# $$\mathrm{TC}(X) = \sum_i H(X_i) - H(X) = D_\text{KL}\left[ p(\mathbf{x}) \| \prod_d p(x_d) \right]$$
#
# TC is zero if and only if all dimensions are independent. In 2D, TC equals
# the mutual information.
#

# %% [markdown]
# ### Analytical

# %%
tc_true_X = 0.5 * (np.sum(np.log(np.diag(CX))) - np.log(np.linalg.det(CX)))
tc_true_XY = 0.5 * (np.sum(np.log(np.diag(C_full))) - np.log(np.linalg.det(C_full)))
print(f"TC(X)  analytical: {tc_true_X:.4f} nats")
print(f"TC(XY) analytical: {tc_true_XY:.4f} nats")

# %% [markdown]
# ### Change-of-variables (single-shot KDE + Gaussian joint)

# %%
tc_cov_X = total_correlation_rbig(X)
tc_cov_XY = total_correlation_rbig(XY)
print(f"TC(X)  change-of-vars: {tc_cov_X:.4f} nats")
print(f"TC(XY) change-of-vars: {tc_cov_XY:.4f} nats")

# %% [markdown]
# ### RBIG-way (per-layer TC reduction)

# %%
tc_rbig_X = estimate_tc(X, **rbig_kw)
tc_rbig_XY = estimate_tc(XY, **rbig_kw)
print(f"TC(X)  RBIG-way: {tc_rbig_X:.4f} nats")
print(f"TC(XY) RBIG-way: {tc_rbig_XY:.4f} nats")

# %% [markdown]
# ---
# ## 2. Entropy
#
# Entropy measures the uncertainty of a random variable — the expected
# information content:
#
# $$H(X) = -\int p(x) \log p(x) \, dx$$
#
# For Gaussian data: $H(X) = \frac{1}{2}\log\lvert 2\pi e\,\Sigma \rvert$
#
# **RBIG-way**: $H(X) = \sum_d H(X_d) - \mathrm{TC}(X)$ — marginal entropies minus
# the RBIG-accumulated total correlation (no Jacobian needed).
#
# **Change-of-variables**: $H(X) = -\mathbb{E}[\log p(x)]$ using the normalizing-flow
# density $\log p(x) = \log p_Z(f(x)) + \log\lvert\det J\rvert$.
#

# %% [markdown]
# ### Analytical

# %%
H_X_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * CX))
H_XY_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * C_full))
print(f"H(X)  analytical: {H_X_true:.4f} nats")
print(f"H(XY) analytical: {H_XY_true:.4f} nats")

# %% [markdown]
# ### Change-of-variables

# %%
model_X = AnnealedRBIG(**rbig_kw).fit(X)
model_XY = AnnealedRBIG(**rbig_kw).fit(XY)

H_X_cov = model_X.entropy()
H_XY_cov = model_XY.entropy()
print(f"H(X)  change-of-vars: {H_X_cov:.4f} nats")
print(f"H(XY) change-of-vars: {H_XY_cov:.4f} nats")

# %% [markdown]
# ### RBIG-way

# %%
H_X_rbig = estimate_entropy(X, **rbig_kw)
H_XY_rbig = estimate_entropy(XY, **rbig_kw)
print(f"H(X)  RBIG-way: {H_X_rbig:.4f} nats")
print(f"H(XY) RBIG-way: {H_XY_rbig:.4f} nats")

# %% [markdown]
# ---
# ## 3. Mutual Information
#
# Mutual information quantifies how much knowing $X$ tells you about $Y$
# (and vice versa). It is zero if and only if $X$ and $Y$ are independent —
# unlike linear correlation, MI captures **all** forms of dependence:
#
# $$\mathrm{MI}(X;Y) = H(X) + H(Y) - H(X,Y)$$
#
# **RBIG-way**: $\mathrm{MI}(X;Y) = \mathrm{TC}([G_X(X),\, G_Y(Y)])$ —
# Gaussianize each block independently, then measure the TC of the concatenation.
#

# %% [markdown]
# ### Analytical

# %%
H_Y_true = 0.5 * np.log(np.linalg.det(2 * np.pi * np.e * CY))
mi_true = H_X_true + H_Y_true - H_XY_true
print(f"MI(X;Y) analytical: {mi_true:.4f} nats")

# %% [markdown]
# ### Change-of-variables (3 separate RBIG models)

# %%
model_Y = AnnealedRBIG(**rbig_kw).fit(Y)
mi_cov = mutual_information_rbig(model_X, model_Y, model_XY)
print(f"MI(X;Y) change-of-vars: {mi_cov:.4f} nats")

# %% [markdown]
# ### RBIG-way (2 models + TC measurement)

# %%
mi_rbig = estimate_mi(X, Y, **rbig_kw)
print(f"MI(X;Y) RBIG-way: {mi_rbig:.4f} nats")

# %% [markdown]
# ---
# ## 4. KL Divergence
#
# KL-divergence measures how one probability distribution $P$ differs from a
# reference distribution $Q$. It is asymmetric and always non-negative:
#
# $$\mathrm{KL}(P \| Q) = \int p(x) \log\frac{p(x)}{q(x)}\,dx$$
#
# **RBIG-way**: $\mathrm{KL}(P_X \| P_Y) = \sum_d D(Z_d \| \mathcal{N}(0,1)) + \mathrm{TC}(Z)$
# where $Z = G_Y(X)$ — apply Y's Gaussianization to X's samples, then measure
# per-marginal KL to standard Gaussian plus the TC of Z.
#
# **Change-of-variables**: Uses cross-scoring of P's density model on Q's samples.
#

# %% [markdown]
# ### Analytical
#
# Same covariance, shifted mean:
# $$\mathrm{KL}(\mathcal{N}(\mu, \Sigma) \| \mathcal{N}(0, \Sigma)) = \tfrac{1}{2}\mu^\top \Sigma^{-1} \mu$$

# %%
kld_true = 0.5 * mu_shift @ np.linalg.solve(CX, mu_shift)
print(f"KLD analytical: {kld_true:.4f} nats")

# %% [markdown]
# ### Change-of-variables

# %%
# Fit model on the reference (unshifted) distribution
model_ref = AnnealedRBIG(**rbig_kw).fit(X)
kld_cov = kl_divergence_rbig(model_ref, X_shifted)
print(f"KLD change-of-vars: {kld_cov:.4f} nats")

# %% [markdown]
# ### RBIG-way

# %%
kld_rbig = estimate_kld(X_shifted, X, **rbig_kw)
print(f"KLD RBIG-way: {kld_rbig:.4f} nats")

# %% [markdown]
# ---
# ## Comparison Table
#
# All measures side-by-side: analytical ground truth vs. the two RBIG approaches.

# %%
print(f"{'Measure':<12} {'Analytical':>12} {'Change-of-Vars':>16} {'RBIG-way':>12}")
print("-" * 56)
print(f"{'TC(X)':12} {tc_true_X:12.4f} {tc_cov_X:16.4f} {tc_rbig_X:12.4f}")
print(f"{'TC(X,Y)':12} {tc_true_XY:12.4f} {tc_cov_XY:16.4f} {tc_rbig_XY:12.4f}")
print(f"{'H(X)':12} {H_X_true:12.4f} {H_X_cov:16.4f} {H_X_rbig:12.4f}")
print(f"{'H(X,Y)':12} {H_XY_true:12.4f} {H_XY_cov:16.4f} {H_XY_rbig:12.4f}")
print(f"{'MI(X;Y)':12} {mi_true:12.4f} {mi_cov:16.4f} {mi_rbig:12.4f}")
print(f"{'KLD':12} {kld_true:12.4f} {kld_cov:16.4f} {kld_rbig:12.4f}")

# %% [markdown]
# ---
# ## See Also
#
# - [Measuring Dependence: 1D Variables](09_dependence_1d.ipynb) — MI for detecting nonlinear dependence in 1D
# - [Measuring Dependence: 2D Variables](10_dependence_2d.ipynb) — MI for multivariate dependence
# - [Information Theory with Synthetic Stock-Market Data](11_real_world_it.ipynb) — IT measures on financial data
#
# ### References
#
# * Nonlinear Extraction of "Independent Components" of elliptically symmetric
#   densities using radial Gaussianization — Lyu & Simoncelli (2008) —
#   [PDF](https://www.cns.nyu.edu/pub/lcv/lyu08a.pdf)
#
