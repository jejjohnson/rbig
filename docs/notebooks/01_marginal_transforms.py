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
# # Marginal Transforms
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/rbig/blob/main/docs/notebooks/01_marginal_transforms.ipynb)
#
# Marginal Gaussianization is the first building block of the RBIG algorithm.
# It transforms each feature **independently** to a standard Gaussian — a
# necessary preprocessing step before rotation can mix the dimensions.
#
# This notebook covers:
#
# - **Why** marginal transforms matter for density estimation and normalizing flows
# - **How** the two-step pipeline (CDF → uniform → probit → Gaussian) works
# - **All 7 implementations** available in `rbig`, with a head-to-head comparison
#

# %% [markdown]
# > **Colab / fresh environment?** Run the cell below to install `rbig` from
# > GitHub. Skip if already installed.

# %%
# !pip install "rbig[all] @ git+https://github.com/jejjohnson/rbig.git" -q

# %% [markdown]
# ---
# ## Theory
#
# ### Why Marginal Gaussianization?
#
# Many statistical methods assume Gaussian data. Marginal Gaussianization makes
# that assumption true — at least marginally — by applying an invertible,
# dimension-wise transform. Because each dimension is handled independently, the
# Jacobian is **diagonal**, making the log-determinant cheap to compute
# ($O(D)$ instead of $O(D^3)$). This is critical for density estimation in
# normalizing flows.
#
# ### The Pipeline
#
# For each dimension $d$, the transform is a two-step composition:
#
# $$x_d \xrightarrow{F_d} u_d \xrightarrow{\Phi^{-1}} z_d$$
#
# 1. **Uniformization** — Estimate the empirical CDF $F_d$ and apply it to
#    obtain a uniform variable $u_d \in [0, 1]$:
#
#    $$u_d = F_d(x_d) = \int_{-\infty}^{x_d} p_d(x_d') \, dx_d'$$
#
# 2. **Gaussianization** — Apply the inverse Gaussian CDF (probit) to map from
#    uniform to Gaussian:
#
#    $$z_d = \Phi^{-1}(u_d)$$
#
# ### Log-Determinant Jacobian
#
# The key property is that the derivative of the CDF is the PDF: $dF/dx = f(x)$.
# Combined with the probit step, the full log-determinant Jacobian is:
#
# $$\log|\det J| = \sum_{d=1}^{D} \log\left|\frac{dz_d}{dx_d}\right|$$
#
# This diagonal structure means we only need per-dimension derivatives — no
# matrix determinants.
#
# ### Boundary Issues
#
# At test time, points outside the training support can cause the probit to
# diverge ($\Phi^{-1}(0) = -\infty$, $\Phi^{-1}(1) = +\infty$). Two
# strategies exist:
#
# - **Clip to boundaries** — map outliers to $[\epsilon, 1-\epsilon]$
#   (simple, used by `MarginalGaussianize`)
# - **KDE tails** — use a smooth KDE whose tails extend beyond the training
#   range (used by `MarginalKDEGaussianize`)
#
# See the [Boundary Issues notebook](07_boundary_issues.ipynb) for a detailed
# comparison.
#
# ### CDF Estimation Methods
#
# The key design choice is **how to estimate $F_d$**. Different estimators trade
# off smoothness, accuracy, and speed:
#
# | Class | CDF Method | Best for |
# |-------|-----------|----------|
# | `MarginalUniformize` | Empirical (rank) | Uniformization step |
# | `MarginalGaussianize` | Empirical + probit | Default in `RBIGLayer` |
# | `MarginalKDEGaussianize` | KDE + probit | Smooth CDF, small data |
# | `QuantileGaussianizer` | Quantile (sklearn) | Large data |
# | `KDEGaussianizer` | KDE + probit (bijector) | Analytic Jacobian |
# | `GMMGaussianizer` | GMM CDF + probit | Multimodal marginals |
# | `SplineGaussianizer` | PCHIP spline | Smooth + fast + accurate |
#

# %%
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

from rbig import (
    MarginalUniformize,
    MarginalGaussianize,
    MarginalKDEGaussianize,
    QuantileGaussianizer,
    KDEGaussianizer,
    GMMGaussianizer,
    SplineGaussianizer,
)

plt.style.use("seaborn-v0_8-paper")

# %% [markdown]
# ---
# ## Data
#
# We use a **Gamma(4)** distribution — a skewed, non-Gaussian marginal that
# clearly shows the effect of each transform. Its asymmetry makes it easy to
# see whether the output is truly Gaussian.
#

# %%
seed = 123
n_samples = 3_000
a = 4  # Gamma shape parameter

data_dist = stats.gamma(a=a)
X = data_dist.rvs(size=(n_samples, 1), random_state=seed)

fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(X[:, 0], ax=ax, bins=50, kde=True, color="steelblue")
ax.set_title(f"Original Gamma(a={a}) samples")
ax.set_xlabel("x")
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## Step 1 — Marginal Uniformization
#
# The first step maps each marginal to a Uniform[0, 1] distribution using the
# empirical CDF. `MarginalUniformize` uses a rank-based estimator with Hazen
# continuity correction:
#
# $$u = \hat{F}_n(x) = \frac{\text{rank}(x) + 0.5}{N}$$
#
# This is useful on its own (e.g., for copula models) and is the intermediate
# step inside every Gaussianizer.
#

# %%
marg_unif = MarginalUniformize(bound_correct=True, eps=1e-6)
marg_unif.fit(X)
Xu = marg_unif.transform(X)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(X[:, 0], ax=axes[0], bins=50, kde=True, color="steelblue")
axes[0].set_title("Original Gamma(4)")
axes[0].set_xlabel("x")

sns.histplot(Xu[:, 0], ax=axes[1], bins=50, color="darkorange")
axes[1].set_title("After MarginalUniformize")
axes[1].set_xlabel("u ∈ [0, 1]")
plt.tight_layout()
plt.show()

# %%
# Round-trip accuracy
X_rt = marg_unif.inverse_transform(Xu)
print(f"Mean |round-trip error|: {np.abs(X - X_rt).mean():.2e}")

# %% [markdown]
# ---
# ## Step 2 — Marginal Gaussianization
#
# `MarginalGaussianize` combines the empirical CDF with the probit transform
# $\Phi^{-1}$ in a single step: $z = \Phi^{-1}(\hat{F}_n(x))$.
#
# Why probit? Because $\Phi^{-1}$ maps Uniform[0,1] exactly to $\mathcal{N}(0,1)$
# — so if the CDF estimate is accurate, the output is Gaussian by construction.
#

# %%
marg_gauss = MarginalGaussianize(bound_correct=True, eps=1e-6)
marg_gauss.fit(X)
Xg = marg_gauss.transform(X)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.histplot(X[:, 0], ax=axes[0], bins=50, kde=True, color="steelblue")
axes[0].set_title("Original Gamma(4)")
axes[0].set_xlabel("x")

sns.histplot(Xg[:, 0], ax=axes[1], bins=50, kde=True, color="seagreen")
axes[1].set_title("After MarginalGaussianize")
axes[1].set_xlabel("z ≈ N(0,1)")
plt.tight_layout()
plt.show()

# %%
# Round-trip accuracy
X_rt2 = marg_gauss.inverse_transform(Xg)
print(f"Mean |round-trip error|: {np.abs(X - X_rt2).mean():.2e}")

# %%
# Log-determinant of the Jacobian (needed for density estimation)
log_jac = marg_gauss.log_det_jacobian(X)
print(
    f"log |det J| — min: {log_jac.min():.3f}, max: {log_jac.max():.3f}, "
    f"mean: {log_jac.mean():.3f}"
)

# %% [markdown]
# ---
# ## All Available Methods
#
# Beyond the two core transforms above, `rbig` provides several alternative
# marginal Gaussianizers. Each estimates the CDF differently, offering different
# tradeoffs in smoothness, speed, and accuracy. The right choice depends on your
# data size, whether you need an analytic Jacobian, and how smooth you want the
# transform to be.
#

# %% [markdown]
# ### MarginalKDEGaussianize
#
# Uses a Gaussian KDE to estimate the CDF, then applies probit. Smoother than
# the empirical CDF, but slower (inverse uses numerical root-finding).

# %%
marg_kde = MarginalKDEGaussianize(bw_method="scott", eps=1e-6)
marg_kde.fit(X)
Xg_kde = marg_kde.transform(X)

# %% [markdown]
# ### QuantileGaussianizer
#
# Wraps sklearn's `QuantileTransformer` with `output_distribution='normal'`.
# Rank-based with piecewise-linear interpolation between quantile nodes.

# %%
quant_gauss = QuantileGaussianizer(n_quantiles=1000, random_state=0)
quant_gauss.fit(X)
Xg_quant = quant_gauss.transform(X)

# %% [markdown]
# ### KDEGaussianizer
#
# Similar to `MarginalKDEGaussianize` but implements the `Bijector` interface
# with an analytic `get_log_det_jacobian` using the KDE density directly.

# %%
kde_gauss = KDEGaussianizer(bw_method="scott", eps=1e-6)
kde_gauss.fit(X)
Xg_kde2 = kde_gauss.transform(X)

# %% [markdown]
# ### GMMGaussianizer
#
# Fits a univariate Gaussian Mixture Model per feature and uses the analytic
# GMM CDF: $F_\text{GMM}(x) = \sum_k w_k \Phi\!\left(\frac{x - \mu_k}{\sigma_k}\right)$.

# %%
gmm_gauss = GMMGaussianizer(n_components=5, random_state=0)
gmm_gauss.fit(X)
Xg_gmm = gmm_gauss.transform(X)

# %% [markdown]
# ### SplineGaussianizer
#
# Fits a monotone PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
# spline from data quantiles to Gaussian quantiles. Smooth, fast, and
# analytically differentiable.

# %%
spline_gauss = SplineGaussianizer(n_quantiles=200, eps=1e-6)
spline_gauss.fit(X)
Xg_spline = spline_gauss.transform(X)

# %% [markdown]
# ---
# ## Comparison
#
# ### Visual comparison of all outputs

# %%
results = {
    "MarginalGaussianize": Xg,
    "MarginalKDEGaussianize": Xg_kde,
    "QuantileGaussianizer": Xg_quant,
    "KDEGaussianizer": Xg_kde2,
    "GMMGaussianizer": Xg_gmm,
    "SplineGaussianizer": Xg_spline,
}

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (name, Zg) in zip(axes.ravel(), results.items()):
    sns.histplot(Zg[:, 0], ax=ax, bins=50, kde=True, color="seagreen", stat="density")
    # overlay ideal N(0,1)
    xx = np.linspace(-4, 4, 200)
    ax.plot(xx, stats.norm.pdf(xx), "k--", alpha=0.5, label="N(0,1)")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("z")
    ax.set_xlim(-4, 4)
    ax.legend(fontsize=8)

plt.suptitle("All Gaussianization methods vs. ideal N(0,1)", y=1.01, fontsize=13)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Round-trip accuracy

# %%
models = {
    "MarginalGaussianize": marg_gauss,
    "MarginalKDEGaussianize": marg_kde,
    "QuantileGaussianizer": quant_gauss,
    "KDEGaussianizer": kde_gauss,
    "GMMGaussianizer": gmm_gauss,
    "SplineGaussianizer": spline_gauss,
}

print(f"{'Method':<25s} {'Mean |error|':>12s}")
print("-" * 40)
for name, model in models.items():
    Zg = model.transform(X)
    if name == "MarginalKDEGaussianize":
        # KDE inverse is slow — use a subset
        X_sub = X[:500]
        Zg_sub = model.transform(X_sub)
        X_rec = model.inverse_transform(Zg_sub)
        err = np.abs(X_sub - X_rec).mean()
        print(f"{name:<25s} {err:>12.2e}  (n=500)")
    else:
        X_rec = model.inverse_transform(Zg)
        err = np.abs(X - X_rec).mean()
        print(f"{name:<25s} {err:>12.2e}")

# %% [markdown]
# ### Log-determinant Jacobian

# %%
print(f"{'Method':<25s} {'mean log|det J|':>16s}")
print("-" * 44)
for name, model in models.items():
    for method in ("log_det_jacobian", "get_log_det_jacobian"):
        fn = getattr(model, method, None)
        if fn is not None:
            try:
                ldj = fn(X)
                print(f"{name:<25s} {ldj.mean():>16.4f}")
                break
            except (NotImplementedError, AttributeError):
                continue
    else:
        print(f"{name:<25s} {'N/A':>16s}")

# %% [markdown]
# ---
# ## Summary
#
# | Class | CDF Method | Output | Inverse | log\|det J\| | Best for |
# |-------|-----------|--------|---------|-------------|----------|
# | `MarginalUniformize` | Empirical (rank) | Uniform | Fast (interp) | N/A | Pre-processing / uniformization step |
# | `MarginalGaussianize` | Empirical + probit | Gaussian | Fast (interp) | Yes | Default in `RBIGLayer` |
# | `MarginalKDEGaussianize` | KDE + probit | Gaussian | Slow (root-find) | Yes | Smooth CDF, small data |
# | `QuantileGaussianizer` | Quantile (sklearn) | Gaussian | Fast (interp) | Yes (finite diff) | Large data, sklearn integration |
# | `KDEGaussianizer` | KDE + probit | Gaussian | Slow (root-find) | Yes (analytic) | When analytic Jacobian needed |
# | `GMMGaussianizer` | GMM CDF + probit | Gaussian | Slow (root-find) | Yes (analytic) | Multimodal marginals |
# | `SplineGaussianizer` | PCHIP spline | Gaussian | Fast (spline) | Yes (analytic) | Smooth + fast + accurate |
#
# All transforms follow the scikit-learn API: `.fit(X)`, `.transform(X)`,
# `.inverse_transform(Z)`.
#
# ### References
#
# * Quantile Transformation — [sklearn](https://stats.stackexchange.com/questions/325570/quantile-transformation-with-gaussian-distribution-sklearn-implementation) | [PyTorch](https://github.com/MilesCranmer/differentiable_quantile_transform)
# * KDE — [Density Estimation](../notes/density_estimation.md#kernel-density-estimation)
# * Splines — [Neural Spline Flows](https://github.com/bayesiains/nsf) | [TF Probability](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/rational_quadratic_spline.py)
#
