# Elevate-RAG

Research harness for ECE 750: one shared index serving vector and graph retrieval, a learned
query router on top, and measurements of when each retrieval path wins.

## Quick start

```bash
make install        # uv sync --frozen (core + dev tools)
uv run elevate --help
make check          # lint + typecheck + test, all offline
```

Optional integrations are extras: `uv sync --extra lightrag --extra ml` or `make install-all`.
Copy `.env.example` to `.env` to configure API keys and database credentials.
