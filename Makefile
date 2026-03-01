.PHONY: help install install-dev test lint format clean

help:
	@echo "Available targets:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linter"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"

install:
	uv pip install -e .

install-dev:
	uv pip install -e ".[test,dev]"

test:
	uv run pytest tests/ -v --cov=rbig --cov-report=xml

lint:
	uv run ruff check rbig/ tests/

format:
	uv run ruff format rbig/ tests/

clean:
	rm -rf build/ dist/ *.egg-info .coverage reports/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
