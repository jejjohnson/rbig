# Agent Guidelines

This file contains standing instructions for **all** coding agents working on this repository (Copilot, Claude, Gemini, etc.).

---

## Karpathy Coding Principles

Four behavioral principles to reduce the most common LLM coding mistakes. These bias toward caution over speed — for trivial tasks, use judgment.

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:

- State assumptions explicitly. If uncertain, ask before writing code.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Test: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:

- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

Test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into declarative goals with verification:

- "Add validation" → Write tests for invalid inputs, then make them pass
- "Fix the bug" → Write a test that reproduces it, then make it pass
- "Optimize X" → Write the naive correct version first, then optimize while preserving correctness

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Before Every Commit

**All agents must** verify that every one of the following passes before creating a commit or reporting progress. No exceptions.

1. **Tests** – `uv run pytest -q` must have 0 failures.
2. **Lint** – `uv run ruff check rbig/` must report no issues.
3. **Format** – `uv run ruff format --check rbig/` must report no files to reformat.

> **Note**: Type checking (e.g., `ty`, `mypy`) is not yet configured for this project. It will be added in the future. Once enabled, agents must also pass the type checker before committing.

> **Common pitfall**: Running `ruff check rbig/` misses lint errors in `tests/` and `scripts/`. CI runs ruff on the entire repo — consider also running `ruff check .` (repo root) to catch issues outside `rbig/`.

## Development Environment

**IMPORTANT**: Always use `uv run` when running Python tools or scripts (e.g., `pytest`, `python`, `ruff`, `mkdocs`, `pre-commit`) so they run in the project environment. You do **not** need `uv run` for non-Python shell commands (e.g., `git`, `ls`, `cat`). Do NOT use the system Python directly.

This project targets **Python 3.10+**.

## Pull Request Descriptions

**Never replace or remove an existing PR title or description.** When reporting progress on a PR that already has a title and description, only append new checklist items or update the status of existing ones. The original content must be preserved in full.

This is a common failure mode: an agent called to make a small follow-up change will supply a fresh description scoped only to its own work, silently discarding all prior context. Always read the existing description first and treat it as the base.

**Do not change** the PR title or description between sessions except to:
- Correct a conventional-commits format violation in the title.
- Append new items to the description checklist.

Never rewrite the existing description; only add to it.

## Plans

Plans and design documents go in `.plans/` (gitignored, never committed). If a plan needs to be tracked long-term, create a GitHub issue with the same detail instead. **Never commit plan files to the repository.**

## GIT Safety Rules

- **NEVER** push to `main` or merge into `main` unless the user explicitly says "push to main" or "merge to main".
- **NEVER** push to any remote branch or run `git push` unless the user explicitly asks you to push. Only commit locally.
- Always work on feature branches.
- When the user says "merge the changes" or "merge the branch", they mean push the local branch to the remote — NOT merge into main.
- Always confirm before any action that affects shared branches (main, production, etc.).

## Documentation

This repo uses **MkDocs + Material + mkdocstrings + mkdocs-jupyter** for documentation.

- **Serve locally**: `uv run --group docs mkdocs serve`
- **Build static site**: `uv run --group docs mkdocs build`
- **Deploy to GitHub Pages**: `uv run --group docs mkdocs gh-deploy --force`
- **Auto-deploy**: the `pages.yml` workflow deploys automatically on every push to `main`

When writing docstrings, use **Google style** (enforced by `mkdocstrings` config).

Notebooks in `docs/` may be stored as `.ipynb` files or as `jupytext`-paired `.py` files.

## Commit Messages

All commit messages **must** follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Valid types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:
- `feat: add new data loading utility`
- `fix: correct off-by-one error in slice computation`
- `docs: update installation instructions`
- `chore: bump ruff to 0.9.7`

PR titles must also follow Conventional Commits format (same rule as commit messages).
