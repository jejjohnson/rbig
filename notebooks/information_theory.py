# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Information Theory Measures w/ RBIG

# %%
import sys

# MacOS
sys.path.insert(0, '/Users/eman/Documents/code_projects/rbig/')
sys.path.insert(0, '/home/emmanuel/code/py_packages/py_rbig/src')

# ERC server
sys.path.insert(0, '/home/emmanuel/code/rbig/')


import numpy as np
import warnings
from time import time
from rbig.rbig import RBIGKLD, RBIG, RBIGMI, entropy_marginal
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

warnings.filterwarnings('ignore') # get rid of annoying warnings

%load_ext autoreload
%autoreload 2

# %% [markdown]
# ---
# ## Total Correlation

# %%
#Parameters
n_samples = 10000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)

# %% [markdown]
# #### Sample Data

# %%
# Generate random normal data
data_original = rng.randn(n_samples, d_dimensions)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

data = data_original @ A

# covariance matrix
C = A.T @ A
vv = np.diag(C)

# %% [markdown]
# #### Calculate Total Correlation

# %%
tc_original = np.log(np.sqrt(vv)).sum() - 0.5 * np.log(np.linalg.det(C))

print(f"TC: {tc_original:.4f}")

# %% [markdown]
# ### RBIG - TC

# %%
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
pdf_extension = 10
pdf_resolution = None
tolerance = None

# Initialize RBIG class
tc_rbig_model = RBIG(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance,
                  pdf_extension=pdf_extension,
                  pdf_resolution=pdf_resolution)

# fit model to the data
tc_rbig_model.fit(data);

# %%
tc_rbig = tc_rbig_model.mutual_information * np.log(2)
print(f"TC (RBIG): {tc_rbig:.4f}")
print(f"TC: {tc_original:.4f}")

# %% [markdown]
# ---
# ## Entropy

# %% [markdown]
# #### Sample Data

# %%
#Parameters
n_samples = 5000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)

# Generate random normal data
data_original = rng.randn(n_samples, d_dimensions)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

data = data_original @ A


# %% [markdown]
# #### Calculate Entropy

# %%
Hx = entropy_marginal(data)

H_original = Hx.sum() + np.log2(np.abs(np.linalg.det(A)))

H_original *= np.log(2)

print(f"H: {H_original:.4f}")

# %% [markdown]
# ### Entropy RBIG

# %%
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
pdf_extension = None
pdf_resolution = None
tolerance = None

# Initialize RBIG class
ent_rbig_model = RBIG(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance)

# fit model to the data
ent_rbig_model.fit(data);

# %%
H_rbig = ent_rbig_model.entropy(correction=True) * np.log(2)
print(f"Entropy (RBIG): {H_rbig:.4f}")
print(f"Entropy: {H_original:.4f}")

# %% [markdown]
# ---
# ## Mutual Information

# %% [markdown]
# #### Sample Data

# %%
#Parameters
n_samples = 10000
d_dimensions = 10

seed = 123

rng = check_random_state(seed)

# Generate random Data
A = rng.rand(2 * d_dimensions, 2 * d_dimensions)

# Covariance Matrix
C = A @ A.T
mu = np.zeros((2 * d_dimensions))

dat_all = rng.multivariate_normal(mu, C, n_samples)

CX = C[:d_dimensions, :d_dimensions]
CY = C[d_dimensions:, d_dimensions:]

X = dat_all[:, :d_dimensions]
Y = dat_all[:, d_dimensions:]

# %% [markdown]
# #### Calculate Mutual Information

# %%
H_X = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(CX)))
H_Y = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(CY)))
H = 0.5 * np.log(2 * np.pi * np.exp(1) * np.abs(np.linalg.det(C)))

mi_original = H_X + H_Y - H
mi_original *= np.log(2)

print(f"MI: {mi_original:.4f}")

# %% [markdown]
# ### RBIG - Mutual Information

# %%
%%time 
n_layers = 10000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
tolerance = None

# Initialize RBIG class
rbig_model = RBIGMI(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance)

# fit model to the data
rbig_model.fit(X, Y);

# %%
H_rbig = rbig_model.mutual_information() * np.log(2)

print(f"MI (RBIG): {H_rbig:.4f}")
print(f"MI: {mi_original:.4f}")

# %% [markdown]
# ---
# ## Kullback-Leibler Divergence (KLD)

# %% [markdown]
# #### Sample Data

# %%
#Parameters
n_samples = 10000
d_dimensions = 10
mu = 0.4          # how different the distributions are

seed = 123

rng = check_random_state(seed)

# Generate random Data
A = rng.rand(d_dimensions, d_dimensions)

# covariance matrix
cov = A @ A.T

# Normalize cov mat
cov = A / A.max()

# create covariance matrices for x and y
cov_x = np.eye(d_dimensions)
cov_y = cov_x.copy()

mu_x = np.zeros(d_dimensions) + mu
mu_y = np.zeros(d_dimensions)

# generate multivariate gaussian data
X = rng.multivariate_normal(mu_x, cov_x, n_samples)
Y = rng.multivariate_normal(mu_y, cov_y, n_samples)


# %% [markdown]
# #### Calculate KLD

# %%
kld_original = 0.5 * ((mu_y - mu_x) @ np.linalg.inv(cov_y) @ (mu_y - mu_x).T +
                      np.trace(np.linalg.inv(cov_y) @ cov_x) -
                      np.log(np.linalg.det(cov_x) / np.linalg.det(cov_y)) - d_dimensions)

print(f'KLD: {kld_original:.4f}')

# %% [markdown]
# ### RBIG - KLD

# %%
X.min(), X.max()

# %%
Y.min(), Y.max()

# %%
%%time

n_layers = 100000
rotation_type = 'PCA'
random_state = 0
zero_tolerance = 60
tolerance = None
pdf_extension = 10
pdf_resolution = None
verbose = 0

# Initialize RBIG class
kld_rbig_model = RBIGKLD(n_layers=n_layers, 
                  rotation_type=rotation_type, 
                  random_state=random_state, 
                  zero_tolerance=zero_tolerance,
                  tolerance=tolerance,
                     pdf_resolution=pdf_resolution,
                    pdf_extension=pdf_extension,
                    verbose=verbose)

# fit model to the data
kld_rbig_model.fit(X, Y);

# %%
# Save KLD value to data structure
kld_rbig= kld_rbig_model.kld*np.log(2)

print(f'KLD (RBIG): {kld_rbig:.4f}')
print(f'KLD: {kld_original:.4f}')

# %%

