# Setup

This document lists everything you must provide to run Elevate-RAG: software, hardware, API keys,
Docker databases and local models. Datasets are covered in [DATASETS.md](DATASETS.md) and models in
[MODELS.md](MODELS.md).

Nothing on this page is needed for the offline path. `make check` and the Phase 0–4 exit checks run
on the bundled synthetic Tiny_Sample with no-setup backends (FAISS or NanoVectorDB, NetworkX) and
deterministic fake models. Add the items below only when you want real data, real models or real
databases.

Run `uv run elevate verify-setup` at any point. It prints one `pass`, `fail` or `skipped` line per
check (Python version, dependencies, environment variables, dataset files, local models, Docker),
and each line names the section of these docs to read.

## Contents

1. [Required software](#1-required-software)
2. [Hardware assumptions](#2-hardware-assumptions)
3. [Installation](#3-installation)
4. [Environment variables and API keys](#4-environment-variables-and-api-keys)
5. [Docker databases](#5-docker-databases)
6. [Prerequisites per phase](#6-prerequisites-per-phase)
7. [Troubleshooting](#7-troubleshooting)

## 1. Required software

| Software | Version | Needed for | Notes |
|---|---|---|---|
| Python | 3.11 or 3.12 | everything | `pyproject.toml` requires `>=3.11`. uv can install it for you (`uv python install 3.11`). |
| uv | 0.5 or newer | everything | Package and environment manager. Reads `uv.lock`, which pins every dependency. Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` or `brew install uv`. |
| Git | any recent | everything | Run_Records store the git commit hash, so run from a git checkout. |
| GNU Make | any | convenience | The `make` targets wrap `uv run`. You can run the underlying commands directly instead. |
| Docker Engine | 24 or newer | Phase 5 only | Docker Desktop on macOS/Windows is fine. |
| Docker Compose | v2.20 or newer (`docker compose`, not `docker-compose`) | Phase 5 only | v2 is needed to apply the `deploy.resources.limits` CPU and memory caps in `docker/*.yml`. |
| Ollama or vLLM | current release | optional | Only if you serve open-weight LLMs locally. See [MODELS.md §3](MODELS.md#3-obtaining-and-serving-local-open-weight-models). |

Check what you have:

```bash
python3 --version
uv --version
git --version
docker --version && docker compose version   # Phase 5 only
```

## 2. Hardware assumptions

| Resource | Minimum (offline Tiny_Sample) | Recommended (full experiments) | Why |
|---|---|---|---|
| CPU | 4 cores, x86-64 or Apple Silicon | 8+ cores | Exit checks are designed to finish in under 15 minutes on a laptop CPU. Phase 5 gives each DB container 2 CPUs by default. |
| RAM | 8 GB | 16–32 GB | Local embedder and reranker need about 2–4 GB each. Each Docker DB is capped at 4 GB by default (`ELEVATE_DB_MEMORY`). |
| Disk | 2 GB | 40–100 GB | Datasets need about 1–4 GB depending on which ones you download (sizes in [DATASETS.md](DATASETS.md)). Local embedder and reranker weights add about 5 GB. A local 35B-class generator adds 20–70 GB depending on quantisation. Caches, indexes and Docker volumes grow with corpus size. |
| GPU | not needed | optional | Speeds up local embedding and reranking. Needed only if you serve open-weight generator LLMs locally (see below). |

GPU guidance for local generators (rough figures; check the model card and your server's docs):

- Gemma 4 12B: about 8–10 GB of VRAM at 4-bit quantisation, about 24 GB at 16-bit.
- Qwen3.6-35B-A3B (mixture of experts, 3B active): about 20–24 GB at 4-bit, 70+ GB at 16-bit.
- Apple Silicon: unified memory of 32 GB or more runs 4-bit quantised 30B-class models through
  Ollama, more slowly than a dedicated GPU.

Without a suitable GPU, point the generator role at an API model or a hosted OpenAI-compatible
endpoint instead. See [MODELS.md §4](MODELS.md#4-substituting-an-unavailable-model).

## 3. Installation

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Get the code
git clone <your-repo-url> Elevate-RAG
cd Elevate-RAG

# 3. Install core + dev dependencies from the lockfile (offline-capable afterwards)
make install                 # = uv sync --frozen

# 4. Create your environment file
cp .env.example .env         # then edit only the values you need (section 4)

# 5. Check the install
uv run elevate --version
make check                   # lint + typecheck + unit/property tests, all offline
uv run elevate verify-setup  # one pass/fail/skipped line per prerequisite
uv run elevate exit-check --phase 0
```

### Optional extras

The core install is enough for every exit check. Real integrations are opt-in extras defined in
`pyproject.toml`:

| Extra | Installs | Needed when |
|---|---|---|
| `lightrag` | `lightrag-hku==1.5.7` | Running real LightRAG indexing and graph retrieval (Phase 3 onward). |
| `ml` | `torch`, `sentence-transformers` | Local embedders and cross-encoder rerankers (Phase 2 onward). |
| `providers` | `openai`, `anthropic` | Any API model, and OpenAI-compatible local servers (Ollama, vLLM). |
| `db` | `qdrant-client`, `pymilvus`, `psycopg`, `pgvector`, `neo4j` | Docker database backends (Phase 5). The Neo4j driver also talks to Memgraph. |
| `eval` | `ragas` | RAGAS faithfulness, context recall and context precision (Phase 1 onward, real runs). |
| `tracing` | `langfuse` | Sending traces to Langfuse. Without it, traces go to local JSONL files. |
| `all` | everything above | Full experiments. |

```bash
make install-all                                       # every extra
uv sync --frozen --extra lightrag --extra ml --extra providers   # a subset
```

`uv sync` installs exactly the extras you name and removes the others, so list every extra you
want in one command.

`torch` from the `ml` extra is large (about 1 GB on macOS and CPU-only Linux, several GB with CUDA).

### Developer commands

| Command | What it does | Network needed |
|---|---|---|
| `make lint` | ruff lint and format check | no |
| `make typecheck` | mypy strict over `src/elevate` | no |
| `make test` | unit and property tests, integration tests excluded | no |
| `make test-integration` | tests marked `integration`; each skips unless its Docker DB or API key is present | yes |
| `make check` | lint + typecheck + test | no |

## 4. Environment variables and API keys

All settings are read from the process environment or from `.env` in the repository root.
`.env.example` lists every supported variable with a placeholder. Secret values are never written to
logs, Run_Records or error messages. Do not commit `.env`.

### 4.1 API keys you may need to provide

Only provide keys for providers that `configs/models.yaml` actually points a role at (see
[MODELS.md](MODELS.md)). With the default offline configuration no key is needed.

| Variable | Provider | Where to sign up | Used for | Phases |
|---|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI | <https://platform.openai.com/api-keys> (billing must be enabled) | Cheap API extraction LLM, LLM judge, R3 router classifier, RAGAS, optional API embedder | 1 (judge, RAGAS), 3 (extraction), 4 (R3), 6, 7 |
| `OPENAI_BASE_URL` | OpenAI or any OpenAI-compatible host | n/a | Change only to use another OpenAI-compatible endpoint | as above |
| `ANTHROPIC_API_KEY` | Anthropic | <https://console.anthropic.com/settings/keys> (Claude Console) | Claude Sonnet 5.5 generator in the E6 generator comparison | 2–4 (if Claude is the generator), 6 |
| `GOOGLE_API_KEY` | Google AI Studio | <https://aistudio.google.com/apikey> | Optional: Gemma or Gemini through the Google API instead of serving Gemma locally | 6 (optional) |
| `HF_TOKEN` | Hugging Face | <https://huggingface.co/settings/tokens> (a read token is enough) | Downloading model weights and datasets; required only for gated repositories, and it raises download rate limits | 1 (datasets from the Hub), 2 onward (local models) |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Langfuse | <https://cloud.langfuse.com> (or self-host, <https://langfuse.com/docs>) | Optional hosted tracing. Without these, traces go to `ELEVATE_TRACE_DIR` as JSONL | any (optional) |
| `VLLM_API_KEY` | your own vLLM server | none; you choose the value when starting `vllm serve --api-key ...` | Authenticating to a local vLLM server | 2 onward (optional) |

The judge must come from a different model family than the generator (Requirement 3.8). With
Claude, Qwen and Gemma as the three generators, the planned judge is an OpenAI model, so plan on an
OpenAI key for any run that reports judge accuracy.

**Cost control.** Paid calls are capped by `ELEVATE_RUN_SPENDING_CAP_USD` (per Run, default
`5.00`) and `ELEVATE_PROJECT_SPENDING_CAP_USD` (whole project, default `200.00`). Set either to `0`
to forbid paid calls. When a cap is reached the Run stops, keeps completed results, and resumes from
the cache next time. Before indexing a full corpus with an API extractor, run the 5% cost estimate
first (Requirement 4.5), because graph extraction is the largest cost in the project.

### 4.2 Paths and run settings

| Variable | Default | Meaning |
|---|---|---|
| `ELEVATE_DATA_DIR` | `./data` | Raw and processed datasets. Layout in [DATASETS.md](DATASETS.md#expected-layout). |
| `ELEVATE_RESULTS_DIR` | `./results` | Run_Records (per-query JSONL + summary JSON), manifests, report assets. |
| `ELEVATE_CACHE_DIR` | `./.cache/elevate` | Content-hash cache of LLM, embedding and retrieval results. Keep it between runs: it is what makes runs resumable and avoids paying twice. |
| `ELEVATE_TRACE_DIR` | `./logs/traces` | Local JSONL traces when Langfuse is not configured. |
| `ELEVATE_SEED` | `42` | Global seed for Python, NumPy and torch. Recorded in every Run_Record. |
| `HF_HOME` | `./.cache/huggingface` | Hugging Face download cache for model weights and Hub datasets. |

### 4.3 Local model servers

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server. |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible server. |

### 4.4 Database credentials (Phase 5)

You choose these values; there is no sign-up. Defaults in `.env.example` match the Compose files.
Change every `change-me` password before exposing a port beyond localhost.

| Variable(s) | Backend |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | Qdrant |
| `MILVUS_URI`, `MILVUS_TOKEN` | Milvus |
| `PGVECTOR_DSN` (and `PGVECTOR_USER`, `PGVECTOR_PASSWORD`, `PGVECTOR_DB`, `PGVECTOR_PORT` for the container) | pgvector |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` (password at least 8 characters) | Neo4j |
| `MEMGRAPH_URI`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD` | Memgraph |
| `POSTGRES_GRAPH_DSN` (and `POSTGRES_GRAPH_USER`, `POSTGRES_GRAPH_PASSWORD`, `POSTGRES_GRAPH_DB`, `POSTGRES_GRAPH_PORT` for the container) | Postgres graph tables |
| `ELEVATE_DB_CPUS`, `ELEVATE_DB_MEMORY` (defaults `2`, `4g`); `ELEVATE_DB_AUX_CPUS`, `ELEVATE_DB_AUX_MEMORY` for Milvus side services | Resource limits shared by every DB container |

For the Postgres stacks, the container variables and the DSN must agree (same user, password,
database and port).

## 5. Docker databases

Docker is needed only for Phase 5 (vector and graph database comparison) and for the reference
configuration in Phase 6 (Qdrant + Neo4j). Every other phase uses embedded backends. When Docker is
absent, Phase 5 exit checks mark the affected backends as `skipped`, and Phase 6 substitutes the
no-setup backends (Requirement 27.2).

One Compose file per stack lives in `docker/`, with pinned images and identical CPU and memory
limits (see `docker/README.md`):

| Stack | File | Image | Host port(s) |
|---|---|---|---|
| Qdrant | `docker/qdrant.yml` | `qdrant/qdrant:v1.13.4` | 6333, 6334 |
| Milvus | `docker/milvus.yml` | `milvusdb/milvus:v2.5.4` + etcd + MinIO | 19530, 9091 |
| pgvector | `docker/pgvector.yml` | `pgvector/pgvector:0.8.0-pg16` | 5432 |
| Neo4j | `docker/neo4j.yml` | `neo4j:5.26.2-community` | 7687, 7474 |
| Memgraph | `docker/memgraph.yml` | `memgraph/memgraph:2.21.0` | 7688 |
| Postgres graph tables | `docker/postgres-graph.yml` | `postgres:16.6` | 5433 |

```bash
uv sync --frozen --extra db                                   # client libraries
docker compose -f docker/qdrant.yml --env-file .env up -d     # start one stack
docker compose -f docker/qdrant.yml ps                        # wait for "healthy"
uv run elevate verify-setup                                   # Docker + backend checks
docker compose -f docker/qdrant.yml down                      # stop, keep data
docker compose -f docker/qdrant.yml down -v                   # stop and delete data
```

Run stacks one at a time for benchmarks so they do not compete for CPU and memory. Running all six
at once needs about 25 GB of RAM for the containers alone at the default limits. Free ports 5432 and
7687 if a local Postgres or Neo4j is already running.

When a configured backend fails its health check, Elevate-RAG stops with an error that names the
backend and points to the matching subsection below.

### Qdrant

Start `docker/qdrant.yml`. Check: `curl http://localhost:6333/healthz`. `QDRANT_URL` must match the
host port.

### Milvus

Start `docker/milvus.yml`. It starts three containers (etcd, MinIO, Milvus) and takes up to a
minute to become healthy. Check: `curl http://localhost:9091/healthz`.

### pgvector

Start `docker/pgvector.yml`. Check: `docker compose -f docker/pgvector.yml exec pgvector pg_isready`.
The image ships the `vector` extension; the database user needs permission to run
`CREATE EXTENSION vector` (the default `elevate` user owns its database, so it does).

### Neo4j

Start `docker/neo4j.yml`. The browser is at <http://localhost:7474>. `NEO4J_PASSWORD` must be at
least 8 characters, or the container refuses to start.

### Memgraph

Start `docker/memgraph.yml`. It listens on host port 7688 so it can run beside Neo4j. It uses the
Bolt protocol, so the `neo4j` Python driver connects to it.

### Postgres graph tables

Start `docker/postgres-graph.yml`. It listens on host port 5433 so it can run beside pgvector.

## 6. Prerequisites per phase

"Required" means the phase's exit check cannot run without it. "Optional" means the item is needed
only for real-data, real-model or real-database runs; without it the exit check marks the dependent
check as `skipped` and names the missing item.

| Phase | Required | Optional |
|---|---|---|
| 0: setup, config, gateway, cache, tracing | Python 3.11+, uv, git, `make install` | Any API key (provider smoke tests); Langfuse keys (hosted tracing); Docker (Compose files are checked but not started) |
| 1: datasets, splits, chunking, evaluation harness | Nothing beyond Phase 0 (uses the bundled Tiny_Sample) | Downloaded datasets ([DATASETS.md](DATASETS.md)); `OPENAI_API_KEY` for the LLM judge and RAGAS (`eval` extra); 100 hand-labelled answers for judge validation (Cohen's kappa ≥ 0.6) |
| 2: vector baseline V0–V4 | Nothing beyond Phase 0 | `ml` extra plus local embedder and reranker weights ([MODELS.md](MODELS.md)); a generator (API key or local server); real dev data for V4 tuning |
| 3: graph layer and graph modes | Nothing beyond Phase 0 | `lightrag` extra; extraction LLM (`OPENAI_API_KEY` for the cheap API extractor), with a spending cap set; reference systems (Microsoft GraphRAG, HippoRAG 2) and alternative frameworks (LlamaIndex, Neo4j GraphRAG) installed separately when those adapters are used |
| 4: routers R0–R4 and oracle | Nothing beyond Phase 0 | `OPENAI_API_KEY` (or another provider) for the R3 LLM classifier; RAGRouter-Bench for 5-fold cross-validation |
| 5: database comparison | Nothing beyond Phase 0 (backends without a healthy container are skipped) | Docker Engine + Compose v2, `db` extra, each stack in `docker/`; a quiet machine with recorded specs for fair benchmarks |
| 6: model comparison, ablations, experiment runner | Nothing beyond Phase 0 (reduced matrix on fakes) | Keys or local servers for every model in the comparison (Claude Sonnet 5.5, Qwen3.6-35B-A3B, Gemma 4 12B, OpenAI extractor and judge); GPU for local generators; full datasets; Docker for the Qdrant + Neo4j reference configuration |
| 7: statistics, error analysis, annotation, report | Nothing beyond Phase 0 | Human annotators: two people for the 50-triple audit, judge-validation labels, error tagging (guidelines in `docs/annotation/`) |
| End-to-end (`elevate e2e --tiny`) | Nothing beyond Phase 0 | Everything above for `elevate e2e --config <file>` on full data |

Suggested order for the first real end-to-end slice (Natural Questions + MultiHop-RAG Tiny_Sample,
FAISS + NetworkX, one local embedder, one reranker, one API LLM):

1. `make install-all`, `cp .env.example .env`, set `OPENAI_API_KEY` and a low
   `ELEVATE_RUN_SPENDING_CAP_USD`.
2. Download NQ and MultiHop-RAG ([DATASETS.md](DATASETS.md)).
3. Download the reference embedder and reranker ([MODELS.md §3](MODELS.md#3-obtaining-and-serving-local-open-weight-models)).
4. `uv run elevate verify-setup` until every line you care about passes.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `uv sync --frozen` fails with a lockfile mismatch | Someone changed `pyproject.toml` without `uv lock`. Run `uv lock` and commit the new `uv.lock`. |
| `make test` tries to reach the network | It sets `HF_HUB_OFFLINE=1` and `uv run --offline`. Run `make install` once with network access first. |
| An exit check line says `skipped` | The named prerequisite is missing. This is expected on the offline path; follow the referenced section to enable it. |
| `ConfigError` naming a file and field | Fix the named field in that config file. Unknown fields are rejected on purpose. |
| `BackendError` naming a database | Start the stack in [section 5](#5-docker-databases) and check its credentials. |
| `BudgetExceeded` | Raise the cap or stop. Completed work is cached; rerunning resumes without repeating paid calls. |
| Hugging Face 401 or 403 | Accept the model licence on its Hub page, then set `HF_TOKEN`. |
| Port already in use | Stop the local service using it, or change the host port in `.env` and the matching URL or DSN. |
