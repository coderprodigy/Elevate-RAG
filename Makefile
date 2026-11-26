# Developer commands. lint, typecheck and test run offline once `make install` has
# synced the environment: --frozen never re-resolves the lockfile and --offline
# forbids network access.

UV      ?= uv
RUN     := $(UV) run --frozen --offline
OFFLINE := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LANGFUSE_TRACING_ENABLED=false

.PHONY: help install install-all lint format typecheck test test-integration check

help:
	@echo "install           Sync core + dev dependencies from uv.lock"
	@echo "install-all       Also install optional extras (lightrag, ml, providers, db, eval, tracing)"
	@echo "lint              ruff lint + format check"
	@echo "format            Apply ruff fixes and formatting"
	@echo "typecheck         mypy (strict) over src/elevate"
	@echo "test              Unit + property tests, offline (integration tests excluded)"
	@echo "test-integration  Integration tests (need Docker DBs / API keys)"
	@echo "check             lint + typecheck + test"

install:
	$(UV) sync --frozen

install-all:
	$(UV) sync --frozen --all-extras

lint:
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck:
	$(RUN) mypy

test:
	$(OFFLINE) $(RUN) pytest -m "not integration"

test-integration:
	$(RUN) pytest -m integration

check: lint typecheck test
