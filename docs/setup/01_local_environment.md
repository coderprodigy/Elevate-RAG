# 01 — Local environment

Your machine: Apple Silicon Mac, 16 GB RAM, Python 3.12.3, **no Docker installed**, and the project folder
is not yet a git repository.

## 1. Python tooling (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # or: brew install uv
uv --version
```

uv manages the virtual environment and the lockfile. Phase 0 creates `pyproject.toml`, and after that
`make setup` (which runs `uv sync`) installs everything. You don't need to install packages by hand.

## 2. Git

```bash
git --version                       # Xcode command-line tools provide it
git clone https://github.com/coderprodigy/Elevate-RAG.git && cd Elevate-RAG
```

Phase 0 runs `git init` and writes `.gitignore`. If the team will share code, create an **empty private
GitHub repository** (no README) and send me its URL. I'll add it as the remote, but I'll only push when you ask.

> The folder path contains spaces. Everything works, but quote paths in your own shell commands.

## 3. Docker runtime

Pick one:

| Option | Notes |
|---|---|
| **OrbStack** (recommended on Mac) | `brew install orbstack`. Lighter on memory and starts faster. Free for personal and academic use, but check its licence terms. |
| Docker Desktop | `brew install --cask docker`. The standard choice; heavier on memory. |

Then:

1. Set the memory limit to **8 GB** (Docker Desktop: Settings → Resources; OrbStack: Settings → System).
   That leaves about 8 GB for Python, embedders and the IDE.
2. Check: `docker run --rm hello-world` and `docker compose version`.

**Memory plan on 16 GB.** Don't run every database at once. The Compose files are split into stacks:

| Stack | Services | Approx. RAM | When |
|---|---|---|---|
| `core` | Qdrant + Neo4j | 3–4 GB | Phases 3–5, 7 |
| `bench-vector` | Milvus *or* pgvector (one profile at a time) | 2–4 GB | Phase 6 |
| `bench-graph` | Memgraph *or* Postgres | 2–3 GB | Phase 6 |

Fairness rule (same CPU and RAM limits for every engine) is enforced in the Compose files. Close heavy
apps while timing benchmarks.

## 4. Optional: Ollama (local small models)

Only needed if you want to run a small open generator locally instead of through a hosted endpoint:

```bash
brew install ollama && ollama serve &
ollama pull gemma4:12b        # confirm the exact tag at freeze; about 8 GB at 4-bit
```

On 16 GB this works but is slow for thousands of queries. See [04_compute_for_open_models.md](04_compute_for_open_models.md).

## 5. Disk space

Allow about **25 GB** free: datasets (about 3 GB raw), embedding caches, Docker volumes and model weights for
local embedders and rerankers (about 3 GB).
