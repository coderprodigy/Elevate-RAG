.PHONY: install check demo

install:
	uv sync --locked --extra graph --extra router --extra faiss

check:
	uv run --no-sync ruff check src tests scripts
	uv run --no-sync ruff format --check src tests scripts
	uv run --no-sync mypy src
	uv run --no-sync pytest --cov=elevate_rag --cov-report=term-missing

demo:
	uv run --no-sync elevate-rag prepare-fixture data/synthetic
	uv run --no-sync elevate-rag run data/synthetic --mode V4 --split all
