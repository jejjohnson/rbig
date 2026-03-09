.PHONY: help install_mamba install_macos install_linux update_macos update_linux docs-serve docs-build docs-clean
.DEFAULT_GOAL = help

# ANSI Color Codes for pretty terminal output
BLUE   := \033[36m
YELLOW := \033[33m
GREEN  := \033[32m
RED    := \033[31m
RESET  := \033[0m

PYTHON = python
VERSION = 3.13
NAME = rbig
ROOT = ./
PIP = pip
SHELL = bash
PKGROOT = rbig
TESTS = tests
NOTEBOOKS_DIR = notebooks

help:	## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Formatting
.PHONY: uv-format
uv-format: ## Run ruff formatter
	@printf "$(YELLOW)>>> Formatting code with ruff...$(RESET)\n"
	@uv run ruff format rbig tests
	@uv run ruff check --fix rbig tests
	@printf "$(GREEN)>>> Codebase formatted successfully.$(RESET)\n"

.PHONY: uv-lint
uv-lint: ## Run ruff check
	@printf "$(YELLOW)>>> Executing static analysis...$(RESET)\n"
	@uv run ruff check rbig tests
	@printf "$(GREEN)>>> Linting checks passed.$(RESET)\n"

.PHONY: uv-pre-commit
uv-pre-commit: ## Run all pre-commit hooks
	@printf "$(YELLOW)>>> Running pre-commit hooks on all files...$(RESET)\n"
	@uv run pre-commit run --all-files
	@printf "$(GREEN)>>> Pre-commit checks passed.$(RESET)\n"

##@ Testing
.PHONY: install
install: ## Install all project dependencies
	@printf "$(YELLOW)>>> Initiating environment synchronization and dependency installation...$(RESET)\n"
	@uv sync --all-extras
	@uv run pre-commit install
	@printf "$(GREEN)>>> Environment is ready and pre-commit hooks are active.$(RESET)\n"

.PHONY: uv-sync
uv-sync: ## Update lock file and sync dependencies using uv
	@printf "$(YELLOW)>>> Updating and syncing dependencies with uv...$(RESET)\n"
	@uv lock --upgrade
	@uv sync --all-extras
	@printf "$(GREEN)>>> uv environment synchronized.$(RESET)\n"

.PHONY: uv-test
uv-test: ## Run pytest with coverage using uv
	@printf "$(YELLOW)>>> Launching test suite with verbosity...$(RESET)\n"
	@uv run pytest tests -v
	@printf "$(GREEN)>>> All tests passed.$(RESET)\n"

##@ Documentation
.PHONY: docs-serve
docs-serve: ## Serve docs locally with live reload (http://127.0.0.1:8000)
	@printf "$(YELLOW)>>> Starting MkDocs dev server...$(RESET)\n"
	@uv run mkdocs serve

.PHONY: docs-build
docs-build: ## Build static docs site into site/ directory
	@printf "$(YELLOW)>>> Building documentation site...$(RESET)\n"
	@uv run mkdocs build --strict
	@printf "$(GREEN)>>> Documentation built successfully in site/.$(RESET)\n"

.PHONY: docs-clean
docs-clean: ## Remove built docs
	@printf "$(YELLOW)>>> Cleaning built documentation...$(RESET)\n"
	@rm -rf site/
	@printf "$(GREEN)>>> Documentation cleaned.$(RESET)\n"
