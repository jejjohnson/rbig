import numpy as np
from scipy.stats import norm


def gaussian_pdf(x):
    return norm.pdf(x)


def gaussian_log_pdf(x):
    return norm.logpdf(x)


def empirical_pdf(data, bins=None, range=None):
    counts, bin_edges = np.histogram(data, bins=bins or 'auto', range=range)
    return counts, bin_edges


def get_prob_gauss(X):
    return np.prod(norm.pdf(X), axis=1)
