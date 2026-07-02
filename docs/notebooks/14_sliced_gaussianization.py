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
# # Sliced Iterative Gaussianization (GIS) — Core Demo
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/rbig/blob/main/docs/notebooks/14_sliced_gaussianization.ipynb)
#
# **Gaussianization via Iterative Slicing (GIS)** — from
# [Dai & Seljak (2020)](https://arxiv.org/abs/2007.00674) — Gaussianizes data
# with a greedy stack of *sliced* layers. Where RBIG rotates **all**
# dimensions and Gaussianizes **every** marginal at each step, a GIS layer:
#
# 1. **learns** $K$ orthonormal projection directions $\mathbf{A}$ that maximize
#    the sliced Wasserstein distance to a Gaussian reference (i.e. the
#    directions along which the data is *most* non-Gaussian), and
# 2. Gaussianizes only those $K$ 1D slices with monotonic rational-quadratic
#    (RQ) splines, leaving the orthogonal complement untouched.
#
# This notebook fits GIS on 2D distributions, **visualizes the learned
# directions layer by layer**, verifies the output is Gaussian, and closes
# with a brief comparison against `AnnealedRBIG` on the same data.
#
# For the mathematical details, see the
# [theory page](../notes/sliced_iterative_gaussianization.md).

# %% [markdown]
# > **Colab / fresh environment?** Run the cell below to install `rbig` from
# > GitHub. Skip if already installed.

# %%
# !pip install "rbig[all] @ git+https://github.com/jejjohnson/rbig.git" -q

# %%
# %matplotlib inline
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from rbig import GIS, AnnealedRBIG

plt.style.use("seaborn-v0_8-paper")

# %% [markdown]
# ## Data
#
# We use two classic 2D benchmarks:
#
# - a **banana** distribution (a Gaussian bent along a parabola), and
# - a **mixture of Gaussians** (multi-modal).


# %%
def make_banana(n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, 2))
    return np.stack([x[:, 0], x[:, 1] + 0.5 * x[:, 0] ** 2 - 1.0], axis=1)


def make_gmm(n_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    means = np.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.5]])
    labels = rng.integers(0, len(means), size=n_samples)
    return means[labels] + 0.6 * rng.standard_normal((n_samples, 2))


n_samples = 5_000
X_banana = make_banana(n_samples, seed=0)
X_gmm = make_gmm(n_samples, seed=1)

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, X, title in zip(axes, [X_banana, X_gmm], ["Banana", "Mixture of Gaussians"]):
    ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.3, color="steelblue")
    ax.set_title(title)
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Fit GIS
#
# Defaults follow the paper / `sinflow`: `n_directions=None` selects
# $K = D/2$ (here $K=1$ per layer), directions are found by Stiefel-manifold
# optimization, and early stopping monitors the held-out log-likelihood.

# %%
t0 = time.perf_counter()
gis_banana = GIS(n_layers=100, random_state=0).fit(X_banana)
t_banana = time.perf_counter() - t0

t0 = time.perf_counter()
gis_gmm = GIS(n_layers=100, random_state=0).fit(X_gmm)
t_gmm = time.perf_counter() - t0

print(f"Banana: {gis_banana.n_layers_} layers kept, {t_banana:.1f}s")
print(f"GMM:    {gis_gmm.n_layers_} layers kept, {t_gmm:.1f}s")

# %% [markdown]
# Early stopping kicked in far below the `n_layers=100` budget — a handful
# of *targeted* layers suffices in 2D.
#
# ## Visualizing the Learned Directions
#
# Each fitted layer stores its orthonormal directions in `layer.A_`
# (shape `(D, K)`). We replay the data through the flow — whitening first,
# then layer by layer — and draw each layer's direction on top of the
# representation it saw. The arrow is the 1D slice the layer Gaussianizes;
# everything orthogonal to it passes through unchanged.


# %%
def plot_layer_directions(model: GIS, X: np.ndarray, max_layers: int = 6):
    """Scatter the running representation with each layer's direction."""
    Xt = model.whitener_.transform(X) if model.whitener_ is not None else X.copy()
    n_show = min(max_layers, model.n_layers_)
    fig, axes = plt.subplots(2, (n_show + 1) // 2, figsize=(12, 7))
    axes = np.atleast_1d(axes).ravel()
    for i in range(n_show):
        layer = model.layers_[i]
        ax = axes[i]
        ax.scatter(Xt[:, 0], Xt[:, 1], s=2, alpha=0.25, color="steelblue")
        # A_: (D, K); draw each direction through the data mean.
        center = Xt.mean(axis=0)
        for k in range(layer.A_.shape[1]):
            a = layer.A_[:, k]
            scale = 3.0
            ax.annotate(
                "",
                xy=center + scale * a,
                xytext=center - scale * a,
                arrowprops={"arrowstyle": "-|>", "color": "crimson", "lw": 2},
            )
        ax.set_title(f"Layer {i + 1} input + direction")
        ax.set_aspect("equal")
        Xt, _ = layer.transform(Xt)
    for ax in axes[n_show:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


fig = plot_layer_directions(gis_banana, X_banana)
fig.suptitle("Banana — learned slice per layer", y=1.02)
plt.show()

# %%
fig = plot_layer_directions(gis_gmm, X_gmm)
fig.suptitle("GMM — learned slice per layer", y=1.02)
plt.show()

# %% [markdown]
# The Stiefel optimizer picks the direction along which the current
# representation deviates most from a Gaussian — the parabolic ridge of the
# banana, the mode-separation axes of the mixture — rather than a fixed PCA
# or random axis. After each layer, the residual non-Gaussianity lives along
# new directions, which the next layer then targets.
#
# ## Is the Output Gaussian?
#
# The forward transform should map the data to (approximately)
# $\mathcal{N}(\mathbf{0}, \mathbf{I}_2)$.

# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
for row, (X, model, name) in enumerate(
    [(X_banana, gis_banana, "Banana"), (X_gmm, gis_gmm, "GMM")]
):
    Z = model.transform(X)
    ax = axes[row, 0]
    ax.scatter(Z[:, 0], Z[:, 1], s=2, alpha=0.3, color="seagreen")
    circle = plt.Circle((0, 0), 3.0, color="black", fill=False, ls="--", lw=1)
    ax.add_patch(circle)
    ax.set_title(f"{name}: transformed (3σ circle)")
    ax.set_aspect("equal")

    grid = np.linspace(-4, 4, 200)
    for j in range(2):
        ax = axes[row, j + 1]
        ax.hist(Z[:, j], bins=60, density=True, alpha=0.6, color="seagreen")
        ax.plot(grid, stats.norm.pdf(grid), "k--", lw=1.5, label="N(0, 1)")
        ax.set_title(f"{name}: latent dim {j + 1}")
        ax.legend()
plt.tight_layout()
plt.show()

# %%
for X, model, name in [(X_banana, gis_banana, "Banana"), (X_gmm, gis_gmm, "GMM")]:
    Z = model.transform(X)
    # Normality of each latent marginal (D'Agostino-Pearson).
    pvals = [stats.normaltest(Z[:, j]).pvalue for j in range(Z.shape[1])]
    cov = np.cov(Z.T)
    print(
        f"{name:7s} latent normaltest p-values: "
        f"{np.array2string(np.array(pvals), precision=3)}  "
        f"| cov diag: {np.diag(cov).round(2)} off-diag: {cov[0, 1]:+.3f}"
    )

# %% [markdown]
# ## Invertibility
#
# Every layer is an exact bijection (the RQ spline inverse is analytic), so
# the full flow inverts to numerical precision.

# %%
for X, model, name in [(X_banana, gis_banana, "Banana"), (X_gmm, gis_gmm, "GMM")]:
    X_rec = model.inverse_transform(model.transform(X))
    err = np.max(np.abs(X_rec - X))
    print(f"{name:7s} max |inverse(transform(X)) - X| = {err:.2e}")

# %% [markdown]
# ## Brief Comparison: GIS vs RBIG
#
# Same data, same evaluation: held-out mean log-likelihood, number of layers,
# and wall-clock fit time. (A detailed convergence study lives in the
# [SIG vs RBIG convergence notebook](15_sig_vs_rbig_convergence.ipynb).)

# %%
X_test_banana = make_banana(n_samples, seed=42)
X_test_gmm = make_gmm(n_samples, seed=43)

results = {}
for X_tr, X_te, gis_model, name in [
    (X_banana, X_test_banana, gis_banana, "Banana"),
    (X_gmm, X_test_gmm, gis_gmm, "GMM"),
]:
    t0 = time.perf_counter()
    rbig_model = AnnealedRBIG(n_layers=100, random_state=0).fit(X_tr)
    t_rbig = time.perf_counter() - t0
    results[name] = {
        "GIS": (gis_model.score(X_te), gis_model.n_layers_),
        "RBIG": (rbig_model.score(X_te), len(rbig_model.layers_)),
        "t_rbig": t_rbig,
    }

print(f"{'data':8s} {'model':6s} {'test LL':>9s} {'layers':>7s}")
for name, res in results.items():
    for m in ["GIS", "RBIG"]:
        ll, nl = res[m]
        print(f"{name:8s} {m:6s} {ll:9.3f} {nl:7d}")

# %% [markdown]
# On these 2D targets GIS reaches a better held-out log-likelihood with a
# comparable (or smaller) layer count: each layer attacks the most
# non-Gaussian direction directly, and the RQ-spline marginals generalize
# more gracefully than histogram CDFs near the support boundary. The gap
# grows with dimension — see the
# [convergence notebook](15_sig_vs_rbig_convergence.ipynb).

# %% [markdown]
# ---
# ## Summary
#
# - `GIS` stacks layers that **learn** where to slice (Stiefel-manifold
#   K-SWD maximization) and Gaussianize only those slices with RQ splines.
# - A handful of targeted layers Gaussianizes classic 2D benchmarks, and the
#   flow inverts to numerical precision.
# - Compared to RBIG, GIS uses far fewer (but individually more expensive)
#   layers for similar held-out log-likelihood.
#
# ## See Also
#
# - [Theory — RBIG vs SIG/GIS](../notes/sliced_iterative_gaussianization.md)
# - [SIG vs RBIG Convergence](15_sig_vs_rbig_convergence.ipynb)
# - [Density Estimation Quality](16_gis_density_estimation.ipynb)
# - [Sampling Quality](17_sig_sampling_quality.ipynb)
# - [Boundary Behavior](18_sig_boundary_behavior.ipynb)
