# rbig

[![Tests](https://github.com/jejjohnson/rbig/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/rbig/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/rbig/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/rbig/actions/workflows/lint.yml)
[![Deploy Docs](https://github.com/jejjohnson/rbig/actions/workflows/docs.yml/badge.svg)](https://github.com/jejjohnson/rbig/actions/workflows/docs.yml)
[![codecov](https://codecov.io/gh/jejjohnson/rbig/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/rbig)
[![PyPI version](https://img.shields.io/pypi/v/rbig.svg)](https://pypi.org/project/rbig/)
[![Python versions](https://img.shields.io/pypi/pyversions/rbig.svg)](https://pypi.org/project/rbig/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

**Rotation-Based Iterative Gaussianization — density estimation, IT measures, and generative modeling**

---

## Overview

RBIG is a density destructor: an invertible, differentiable transformation that maps any multivariate distribution to a standard Gaussian. It works by alternating between two steps:

1. **Marginal Gaussianization** — independently transform each dimension to Gaussian using an empirical CDF-based mapping.
2. **Rotation** — apply an orthonormal linear transform (PCA, ICA, random, or Picard) to mix the dimensions.

After enough iterations the joint distribution converges to a standard Gaussian. Because each step is invertible and differentiable, RBIG provides exact likelihood evaluation via the change-of-variables formula:

$$p_x(x) = p_z\!\left(\mathcal{G}_\theta(x)\right) \left|\nabla_x \mathcal{G}_\theta(x)\right|$$

This also makes RBIG a natural tool for computing information-theoretic measures (entropy, mutual information, total correlation, KL-divergence) without assuming any parametric form.

---

## Key Features

- **Multiple marginal Gaussianization methods**: Quantile Transform, KDE, Gaussian Mixture Model, Spline
- **Multiple rotation strategies**: PCA, ICA, Random, Picard
- **Information theory measures**: entropy, mutual information, total correlation, KL-divergence
- **Image processing support**: DCT, Hartley, and Wavelet transforms
- **Xarray integration**: spatiotemporal data with named dimensions
- **Parametric distributions**: Gaussian, Exponential, Gamma, Beta, and more

---

## Installation

```bash
pip install rbig
```

Optional extras:

```bash
pip install "rbig[image]"   # wavelet/DCT image support
pip install "rbig[xarray]"  # spatiotemporal xarray integration
pip install "rbig[all]"     # everything
```

---

## Quick Start

```python
import numpy as np
from rbig import AnnealedRBIG

# Generate correlated 2D data
rng = np.random.RandomState(42)
data = rng.randn(1000, 2) @ [[1, 0.8], [0.8, 1]]

# Fit RBIG and transform to Gaussian
model = AnnealedRBIG(n_layers=50, rotation="pca")
Z = model.fit_transform(data)

# Z is now approximately standard Gaussian
```

---

## Repository Layout

```
rbig/
├── rbig/                                 # Main package code (flat layout)
├── tests/                                # pytest test suite
├── docs/                                 # MkDocs documentation source
├── notebooks/                            # Jupyter notebooks
├── .github/
│   ├── workflows/                        # GitHub Actions CI/CD workflows
│   ├── instructions/                     # Copilot custom instructions
│   ├── copilot-instructions.md           # Copilot behavioural config
│   ├── dependabot.yml                    # Automated dependency PRs
│   └── labeler.yml                       # Automatic PR labelling rules
├── pyproject.toml                        # Project metadata & tool config
├── uv.lock                              # Reproducible lockfile
├── Makefile                              # Self-documenting task runner
├── mkdocs.yml                            # Documentation site configuration
├── .pre-commit-config.yaml               # Git hook definitions
├── release-please-config.json            # Automated release & changelog config
├── .env.example                          # Template for local environment variables
├── AGENTS.md                             # Standing instructions for AI coding agents
├── CODE_REVIEW.md                        # Code review standards
└── CHANGELOG.md                          # Auto-generated changelog
```

---

## Developer Quick Start

```bash
# Prerequisites: uv (https://github.com/astral-sh/uv)
git clone https://github.com/jejjohnson/rbig.git
cd rbig
make install      # install all dependency groups + pre-commit hooks
make test         # run tests
make docs-serve   # preview docs locally
```

All common tasks are available via `make`:

| Target | Description |
|--------|-------------|
| `make help` | Print all targets with descriptions |
| `make install` | Install all dependency groups via uv |
| `make init` | Bootstrap `.env` from `.env.example` |
| `make test` | Run tests (no coverage) |
| `make test-cov` | Run tests with coverage report |
| `make lint` | Lint with ruff (no autofix) |
| `make format` | Format with ruff (format + autofix) |
| `make precommit` | Run pre-commit hooks on all files |
| `make build` | Build wheel and sdist |
| `make clean` | Remove build artefacts and caches |
| `make docs` | Build documentation site |
| `make docs-serve` | Preview documentation locally |
| `make docs-deploy` | Deploy documentation to GitHub Pages |
| `make docs-clean` | Remove built docs |
| `make version` | Display package version and git hash |

---

## CI/CD Workflows

| Workflow | File | Trigger | What it does |
|----------|------|---------|-------------|
| Tests | `ci.yml` | push / PR to main | pytest + Codecov upload |
| Lint | `lint.yml` | push / PR to main | `ruff check` + `ruff format --check` |
| Deploy Docs | `docs.yml` | push to main | `mkdocs gh-deploy` |
| Release Please | `release-please.yml` | push to main | automated release PR + changelog |
| CodeQL | `codeql.yml` | push / PR / schedule | security static analysis |
| Conventional Commits | `conventional-commits.yml` | PR | validates PR title format |
| PR Labeler | `label-pr.yml` | PR | applies path-based labels |
| Pre-commit Autoupdate | `pre-commit-autoupdate.yml` | weekly schedule | bumps hook revisions, opens PR |
| Dependency Review | `dependency-review.yml` | PR | reviews dependency changes |

---

## Contributing

- See [`AGENTS.md`](AGENTS.md) for AI agent guidelines and coding principles
- See [`CODE_REVIEW.md`](CODE_REVIEW.md) for code review standards
- Report issues at the [issue tracker](https://github.com/jejjohnson/rbig/issues)

---

## Documentation

Full documentation lives in the [`docs/`](docs/) directory and is built with MkDocs Material.

---

## Citation

If you use RBIG in your research, please cite:

> V. Laparra, G. Camps-Valls, and J. Malo,
> "Iterative Gaussianization: from ICA to Random Rotations,"
> *IEEE Transactions on Neural Networks*, 22(4):537–549, 2011.
> [arXiv:1602.00229](https://arxiv.org/abs/1602.00229)

---

## Further Reading

| Tool | Documentation |
|------|--------------|
| uv | <https://docs.astral.sh/uv/> |
| ruff | <https://docs.astral.sh/ruff/> |
| hatchling | <https://hatch.pypa.io/latest/> |
| pytest | <https://docs.pytest.org/> |
| MkDocs Material | <https://squidfunk.github.io/mkdocs-material/> |
| mkdocstrings | <https://mkdocstrings.github.io/> |
| pre-commit | <https://pre-commit.com/> |
| Release Please | <https://github.com/googleapis/release-please> |
| Conventional Commits | <https://www.conventionalcommits.org/> |

---

## License

MIT — see [LICENSE](LICENSE) for details.
