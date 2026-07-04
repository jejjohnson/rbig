"""Test configuration and shared fixtures."""

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_2d(rng):
    """Simple 2D Gaussian dataset."""
    n = 200
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    return rng.multivariate_normal(mean, cov, size=n)


@pytest.fixture
def simple_5d(rng):
    """Simple 5D dataset."""
    n = 300
    cov = np.eye(5)
    cov[0, 1] = cov[1, 0] = 0.3
    return rng.multivariate_normal(np.zeros(5), cov, size=n)


@pytest.fixture
def uniform_2d(rng):
    """2D uniform data."""
    return rng.uniform(0.01, 0.99, size=(200, 2))


# ── Golden-file plumbing ──────────────────────────────────────────────────────

GOLDEN_DIR = Path(__file__).parent / "golden"


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Re-pin golden regression arrays instead of comparing against them.",
    )


@pytest.fixture
def golden(request):
    """Pin or compare fixed-seed arrays under ``tests/golden/<name>.npz``.

    Usage: ``golden("my_case", array)`` — with ``--update-golden`` the array
    is (re)written; otherwise it is compared against the pinned copy with
    ``np.testing.assert_allclose``.
    """

    def _check(name: str, array: np.ndarray, rtol: float = 1e-10, atol: float = 1e-12):
        path = GOLDEN_DIR / f"{name}.npz"
        if request.config.getoption("--update-golden"):
            GOLDEN_DIR.mkdir(exist_ok=True)
            np.savez_compressed(path, array=np.asarray(array))
            pytest.skip(f"golden file {name} updated")
        if not path.exists():
            pytest.fail(
                f"Missing golden file {path}. Run pytest with --update-golden "
                f"to pin it."
            )
        pinned = np.load(path)["array"]
        np.testing.assert_allclose(np.asarray(array), pinned, rtol=rtol, atol=atol)

    return _check
