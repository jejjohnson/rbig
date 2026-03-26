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
# # Measuring Dependence: Multivariate (2D) Variables
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jejjohnson/rbig/blob/main/docs/notebooks/10_dependence_2d.ipynb)
#
# When both $X$ and $Y$ are **vectors** (not scalars), measuring dependence
# becomes harder. Kernel methods like CKA are popular but bandwidth-sensitive.
# This notebook shows that RBIG-based MI provides a non-parametric alternative
# that captures multivariate nonlinear dependence.
#

# %% [markdown]
# > **Colab / fresh environment?** Run the cell below to install `rbig` from
# > GitHub. Skip if already installed.

# %%
# !pip install "rbig[all] @ git+https://github.com/jejjohnson/rbig.git" -q

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from rbig import AnnealedRBIG, mutual_information_rbig

# %% [markdown]
# ## Dataset 1: Asymmetric nonlinearity
#
# $$
# \begin{aligned}
# y_0 &= (2 x_0)^2 + \varepsilon \\
# y_1 &= 0.2 x_0 + \sqrt{2|x_1|} + 0.1 \varepsilon
# \end{aligned}
# $$
#
# Cross-dimensional dependencies with different nonlinear forms — $y_0$
# depends only on $x_0$ (quadratic), while $y_1$ depends on both $x_0$
# (linear) and $x_1$ (square-root).

# %%
rng = np.random.RandomState(42)
N = 1000

x1 = rng.randn(N, 2)
noise1 = rng.randn(N, 1)
y1 = np.column_stack([
    (2 * x1[:, 0]) ** 2 + noise1.ravel(),
    0.2 * x1[:, 0] + np.sqrt(2 * np.abs(x1[:, 1])) + 0.1 * noise1.ravel(),
])

# %% [markdown]
# ### Scatter plots: all pairwise relationships

# %%
fig, axes = plt.subplots(2, 2, figsize=(8, 7))
labels = [("$x_0$", "$y_0$"), ("$x_1$", "$y_0$"), ("$x_0$", "$y_1$"), ("$x_1$", "$y_1$")]
pairs = [(0, 0), (1, 0), (0, 1), (1, 1)]
for ax, (xi, yi), (xl, yl) in zip(axes.ravel(), pairs, labels):
    ax.scatter(x1[:, xi], y1[:, yi], alpha=0.5, s=10)
    ax.set(xlabel=xl, ylabel=yl)
fig.suptitle("Dataset 1: Asymmetric nonlinearity", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Classical measures

# %%
from sklearn.metrics.pairwise import rbf_kernel


def hsic(K, L):
    """Biased HSIC estimator from centered kernel matrices."""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # centering matrix
    Kc = H @ K @ H
    Lc = H @ L @ H
    return np.trace(Kc @ Lc) / (n - 1) ** 2


def normalized_hsic(K, L):
    """CKA: HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L)), bounded [0, 1]."""
    return hsic(K, L) / np.sqrt(hsic(K, K) * hsic(L, L))


# --- Linear kernels ---
K_lin1 = x1 @ x1.T
L_lin1 = y1 @ y1.T
cka_lin1 = normalized_hsic(K_lin1, L_lin1)

# --- RBF kernels (median heuristic for length scale) ---
from scipy.spatial.distance import pdist

sigma_x1 = np.median(pdist(x1, "euclidean"))
sigma_y1 = np.median(pdist(y1, "euclidean"))
K_rbf1 = rbf_kernel(x1, gamma=1.0 / (2 * sigma_x1**2))
L_rbf1 = rbf_kernel(y1, gamma=1.0 / (2 * sigma_y1**2))
cka_rbf1 = normalized_hsic(K_rbf1, L_rbf1)

# Spearman on stacked [X, Y]
stacked = np.hstack([x1, y1])
spearman_matrix = stats.spearmanr(stacked).statistic
# Extract cross-correlation block (X cols vs Y cols)
spearman_xy = spearman_matrix[:2, 2:]
spearman_fro = np.linalg.norm(spearman_xy, "fro")

print("Dataset 1 — classical measures:")
print(f"  CKA linear:  {cka_lin1:.4f}")
print(f"  CKA RBF:     {cka_rbf1:.4f}")
print(f"  Spearman cross-block Frobenius: {spearman_fro:.4f}")

# %% [markdown]
# ### Mutual Information via RBIG

# %%
model_x1 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model_y1 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model_xy1 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)

model_x1.fit(x1)
model_y1.fit(y1)
model_xy1.fit(np.hstack([x1, y1]))

mi1 = mutual_information_rbig(model_x1, model_y1, model_xy1)
icc1 = np.sqrt(np.maximum(0, 1 - np.exp(-2 * mi1)))

print(f"  MI (RBIG): {mi1:.4f} nats")
print(f"  ICC:       {icc1:.4f}")

# %% [markdown]
# ## Dataset 2: Symmetric quadratic (higher noise)
#
# $$
# \begin{aligned}
# y_0 &= (2 x_0)^2 + 3\varepsilon_0 \\
# y_1 &= (2 x_1)^2 + 3\varepsilon_1
# \end{aligned}
# $$
#
# Independent per-dimension quadratics with higher noise — weaker signal.

# %%
x2 = rng.randn(N, 2)
y2 = np.column_stack([
    (2 * x2[:, 0]) ** 2 + 3 * rng.randn(N),
    (2 * x2[:, 1]) ** 2 + 3 * rng.randn(N),
])

fig, axes = plt.subplots(2, 2, figsize=(8, 7))
for ax, (xi, yi), (xl, yl) in zip(axes.ravel(), pairs, labels):
    ax.scatter(x2[:, xi], y2[:, yi], alpha=0.5, s=10)
    ax.set(xlabel=xl, ylabel=yl)
fig.suptitle("Dataset 2: Symmetric quadratic (high noise)", fontsize=13)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Classical measures + MI

# %%
# Linear CKA
K_lin2 = x2 @ x2.T
L_lin2 = y2 @ y2.T
cka_lin2 = normalized_hsic(K_lin2, L_lin2)

# RBF CKA (median heuristic)
sigma_x2 = np.median(pdist(x2, "euclidean"))
sigma_y2 = np.median(pdist(y2, "euclidean"))
K_rbf2 = rbf_kernel(x2, gamma=1.0 / (2 * sigma_x2**2))
L_rbf2 = rbf_kernel(y2, gamma=1.0 / (2 * sigma_y2**2))
cka_rbf2 = normalized_hsic(K_rbf2, L_rbf2)

spearman2 = stats.spearmanr(np.hstack([x2, y2])).statistic
spearman_xy2 = spearman2[:2, 2:]
spearman_fro2 = np.linalg.norm(spearman_xy2, "fro")

model_x2 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model_y2 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model_xy2 = AnnealedRBIG(n_layers=50, rotation="pca", random_state=42)
model_x2.fit(x2)
model_y2.fit(y2)
model_xy2.fit(np.hstack([x2, y2]))

mi2 = mutual_information_rbig(model_x2, model_y2, model_xy2)
icc2 = np.sqrt(np.maximum(0, 1 - np.exp(-2 * mi2)))

print("Dataset 2 — classical measures:")
print(f"  CKA linear:  {cka_lin2:.4f}")
print(f"  CKA RBF:     {cka_rbf2:.4f}")
print(f"  Spearman cross-block Frobenius: {spearman_fro2:.4f}")
print(f"  MI (RBIG): {mi2:.4f} nats")
print(f"  ICC:       {icc2:.4f}")

# %% [markdown]
# ## Comparison summary
#
# | Metric | Dataset 1 (asymmetric) | Dataset 2 (symmetric, noisy) |
# |--------|:---------------------:|:---------------------------:|
# | CKA linear | low | low |
# | CKA RBF | **moderate–high** | **moderate** |
# | Spearman cross-Frobenius | low | low |
# | MI (RBIG) | **high** | **moderate** |
# | ICC | **high** | **moderate** |
#
# The RBF kernel captures nonlinear dependence that the linear kernel misses
# entirely — similar to how MI outperforms Pearson/Spearman. However, CKA RBF
# depends on the bandwidth choice, while MI (RBIG) is non-parametric.
#
# Again, MI detects nonlinear multivariate dependence that classical
# matrix-based measures largely miss. Dataset 2 shows lower MI due to
# the higher noise level, correctly reflecting the weaker signal.

# %% [markdown]
# ---
# ## See Also
#
# - [Information Theory Measures](06_information_theory.ipynb) — formal definitions of MI and TC
# - [Measuring Dependence: 1D Variables](09_dependence_1d.ipynb) — MI for detecting nonlinear dependence in 1D
# - [Real-World IT](11_real_world_it.ipynb) — IT measures on synthetic financial data
#
