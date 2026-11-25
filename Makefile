.PHONY: setup lint fmt typecheck test test-int check usage

setup:            ## create .venv, install deps and git hooks
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

fmt:
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck:
	uv run mypy

test:             ## offline unit tests
	uv run pytest

test-int:         ## integration tests (API keys / Docker / server tunnel)
	uv run pytest -m integration

check: lint typecheck test

usage:            ## API spend grouped by role
	uv run hgrag usage --by role
