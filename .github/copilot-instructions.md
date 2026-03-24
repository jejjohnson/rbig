# Copilot Instructions

## Project Overview

- **Project**: rbig — Rotation-Based Iterative Gaussianization
- **Python**: 3.10+
- **Package Manager**: uv
- **Layout**: flat layout (`rbig/`)
- **Testing**: pytest
- **Docs**: MkDocs + Material + mkdocstrings + mkdocs-jupyter

## Build & Test Commands

```bash
uv sync --all-extras --all-groups   # Install all dependencies
uv run pytest tests -v              # Run tests
uv run ruff check rbig/             # Lint code
uv run ruff format rbig/            # Format code
```

## Before Every Commit — Mandatory Checklist

**All checks must pass before any commit.** CI runs them on the entire repo, so always run the commands below from the repo root.

```bash
# 1. Tests — zero failures required
uv run pytest tests -v

# 2. Lint — run on the ENTIRE repo (includes tests/)
uv run ruff check .

# 3. Format check — run on the ENTIRE repo
uv run ruff format --check .
```

> **Common pitfall**: Running `ruff check rbig/` instead of `ruff check .` misses import-sorting errors in `tests/`. The CI workflow runs `ruff check .`. Always use `.` (repo root), not a subdirectory.

## Key Directories

| Path | Purpose |
|------|---------|
| `rbig/_src/` | Main package source code |
| `tests/` | Test suite |
| `docs/` | Documentation (MkDocs) |
| `docs/notebooks/` | Jupyter notebooks |

## Behavioral Guidelines

### Do Not Nitpick
- Ignore style issues that linters/formatters catch (formatting, import order, quote style)
- Don't suggest changes to code you weren't asked to modify
- Match existing patterns even if you'd do it differently

### Always Propose Tests
When implementing features or fixing bugs:
1. Write a test that verifies the expected behavior
2. Implement the change
3. Verify the test passes

### Never Suggest Without a Proposal
Bad: "You should add validation here"
Good: "Add validation here. Proposed implementation:"
```python
if value < 0:
    raise ValueError('Value must be non-negative')
```

### Simplicity First
- No abstractions for single-use code
- No speculative features beyond what was asked
- If 200 lines could be 50, propose the simpler version

### Surgical Changes
- Only modify lines directly related to the request
- Don't refactor adjacent code
- Don't add docstrings/comments to code you didn't change
- Remove only imports/functions that YOUR changes made unused

## Code Review

For all code review tasks, follow the guidance in `/CODE_REVIEW.md`.
