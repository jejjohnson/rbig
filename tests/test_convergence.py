"""Tests for the shared StoppingCriterion module."""

from __future__ import annotations

import numpy as np
import pytest

from rbig._src.convergence import StoppingCriterion


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_invalid_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        StoppingCriterion(metric="nope")


@pytest.mark.parametrize("metric", ["log_likelihood", "swd", "total_correlation"])
def test_metric_returns_finite(rng, metric):
    crit = StoppingCriterion(metric=metric, random_state=0)
    X = rng.standard_normal((300, 3))
    crit.update(X, log_det=np.zeros(300))
    assert np.isfinite(crit.best_score())
    assert len(crit.history_) == 1


def test_split_shapes(rng):
    crit = StoppingCriterion(validation_fraction=0.25)
    X = rng.standard_normal((400, 4))
    X_tr, X_val = crit.split(X, random_state=0)
    assert X_tr.shape[0] == 300
    assert X_val.shape[0] == 100
    assert X_tr.shape[1] == X_val.shape[1] == 4


def test_split_zero_fraction_returns_full(rng):
    crit = StoppingCriterion(validation_fraction=0.0)
    X = rng.standard_normal((50, 2))
    X_tr, X_val = crit.split(X)
    assert X_tr.shape == X_val.shape == X.shape


def test_early_stop_triggers_with_patience():
    crit = StoppingCriterion(
        metric="total_correlation", patience=3, validation_fraction=0.0
    )
    rng = np.random.default_rng(1)
    # Same data each iteration => metric never improves after the first.
    X = rng.standard_normal((300, 3))
    stops = [crit.update(X) for _ in range(6)]
    # First update sets the baseline; then 3 non-improving updates stop it.
    assert stops[:3] == [False, False, False]
    assert stops[3] is True


def test_log_likelihood_improves_when_more_gaussian(rng):
    crit = StoppingCriterion(metric="log_likelihood", validation_fraction=0.0)
    # A poorly scaled representation has lower log-likelihood than N(0, I).
    bad = rng.standard_normal((500, 3)) * 5.0
    good = rng.standard_normal((500, 3))
    crit.update(good, log_det=np.zeros(500))
    s_good = crit.history_[-1]
    crit2 = StoppingCriterion(metric="log_likelihood", validation_fraction=0.0)
    crit2.update(bad, log_det=np.zeros(500))
    s_bad = crit2.history_[-1]
    assert s_good > s_bad


def test_best_score_before_update_raises():
    crit = StoppingCriterion()
    with pytest.raises(ValueError, match="No updates"):
        crit.best_score()


def test_swd_smaller_for_gaussian(rng):
    crit_g = StoppingCriterion(metric="swd", random_state=0, validation_fraction=0.0)
    crit_g.update(rng.standard_normal((1000, 3)))
    crit_b = StoppingCriterion(metric="swd", random_state=0, validation_fraction=0.0)
    crit_b.update(rng.exponential(1.0, (1000, 3)))
    assert crit_g.best_score() < crit_b.best_score()
