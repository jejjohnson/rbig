"""Tests for progress bar integration."""

import numpy as np
import pytest


def test_maybe_tqdm_silent():
    from rbig._src._progress import maybe_tqdm

    result = list(maybe_tqdm(range(5), verbose=False, level=1))
    assert result == [0, 1, 2, 3, 4]


def test_maybe_tqdm_active():
    from rbig._src._progress import maybe_tqdm

    result = list(maybe_tqdm(range(5), verbose=True, level=1))
    assert result == [0, 1, 2, 3, 4]


def test_maybe_tqdm_level_filtering():
    from rbig._src._progress import maybe_tqdm

    # verbose=1 should not show level=2 bars
    result = list(maybe_tqdm(range(3), verbose=1, level=2))
    assert result == [0, 1, 2]


def test_verbose_fit(simple_2d):
    from rbig import AnnealedRBIG

    model = AnnealedRBIG(n_layers=5, verbose=True)
    model.fit(simple_2d)
    assert len(model.layers_) > 0


def test_verbose_2_transform(simple_2d):
    from rbig import AnnealedRBIG

    model = AnnealedRBIG(n_layers=5, verbose=2)
    model.fit(simple_2d)
    Xt = model.transform(simple_2d)
    assert Xt.shape == simple_2d.shape


@pytest.fixture
def simple_2d():
    rng = np.random.default_rng(42)
    n = 200
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    return rng.multivariate_normal(mean, cov, size=n)
