# Elevate-RAG

A reproducible research platform for the question in *Hybrid Vector + Graph RAG: Project Design &
Evaluation Plan*: **when does graph structure pay for itself, and can a router capture that benefit
automatically?**

One shared index (chunks, dense vectors, BM25, a LightRAG knowledge graph) serves every retrieval
path: vector baselines V0–V4, LightRAG `naive/local/global/hybrid/mix`, and routers R0–R4. All paths
share the same reranker, token budget and generation prompt, so retrieval mode is the only
variable. Experiments (E1–E6, A1–A4) are config-driven, resumable, cost-capped and report bootstrap
CIs and paired tests.

> **Status (2026-11-27):** software implemented and tested (CI: lint, types, tests with 86 %
> coverage, Docker image, cluster dry run). Seven datasets prepared from pinned revisions;
> model-free baselines measured. Gate G1, the first end-to-end run with real models at $0, passed
> for the pipeline with 3B and with 7B/8B local models ([`results/`](results/README.md)).
> **The research experiments E1–E6 have not been run yet**: they need GPU time (the cluster
> setup is ready, untested on a real cluster) or an API budget, and two human annotators to
> validate the judge. Details: [`docs/PROGRESS.md`](docs/PROGRESS.md),
> [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

## Quick start (offline, no credentials, ~20 s)

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 if missing).

```bash
uv sync --locked --extra faiss       # venv from the lockfile
uv run elevate-rag demo              # synthetic corpus → index → all modes/routers → report
open demo_output/report/REPORT.md    # every number is labelled DEMO (stand-in models)
```

The demo uses deterministic stand-ins (hash embedder, lexical reranker, heuristic extractor,
extractive generator). They exercise the real pipeline, LightRAG included, but are **not**
research models. Docker only: `make image && make image-smoke`.

## Running for real

**[docs/RUNNING.md](docs/RUNNING.md)** covers every setup step by step and how to reproduce each
committed result:

| Where | Models | Entry point |
|---|---|---|
| Laptop / desktop | Ollama (Qwen2.5 3B/7B, Llama 3.x judge), $0 | `scripts/run_pipeline.sh` |
| Workstation with NVIDIA GPUs | vLLM (Qwen3.6-35B-A3B, phi-4 judge), $0 | `scripts/run_pipeline.sh` |
| College / HPC cluster (SLURM) | vLLM, sharded job arrays, Apptainer optional | `infrastructure/cluster/submit.sh` |
| Any machine with API keys | Claude / OpenAI, hard spending cap | `elevate-rag experiment run` |
| AWS EC2 | any of the above | [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) |

The research workflow in one block (details and options in RUNNING.md):

```bash
uv run elevate-rag data prepare all                                     # pinned, checksummed
uv run elevate-rag index build  PIPELINE.yaml --set dataset=multihop_rag  # chunks, vectors, graph
uv run elevate-rag index validate PIPELINE.yaml --set dataset=multihop_rag --audit-out triples.csv
uv run elevate-rag experiment validate configs/experiments/E2_retrieval_modes.yaml --pipeline PIPELINE.yaml
uv run elevate-rag experiment estimate configs/experiments/E2_retrieval_modes.yaml --pipeline PIPELINE.yaml
uv run elevate-rag experiment run      configs/experiments/E2_retrieval_modes.yaml --pipeline PIPELINE.yaml
uv run elevate-rag report --experiment E2 --out results/E2 --reference V4
```

Human validation (two annotators, never fabricated) and external systems (Microsoft GraphRAG,
HippoRAG 2, …, scored by the same evaluator via `elevate-rag experiment external`) are in
RUNNING.md as well.

## Configuration

* `configs/pipelines/`:

  | Config | Use |
  |---|---|
  | `reference.yaml` | the plan's reference (needs an API budget and a judge choice) |
  | `cluster_gpu.yaml`, `cluster_gpu_7b.yaml` | all-open models on vLLM, for GPU servers |
  | `local_free.yaml`, `local_7b.yaml` | Ollama on a laptop, $0 |
  | `local_dev.yaml`, `cpu_small.yaml` | reference variants without Docker / on a laptop CPU |
  | `demo_offline.yaml` | offline stand-ins |

  A config can `extends:` another; a model role the child defines with its own `model:` replaces
  the parent's role entirely.
* `configs/experiments/`: E1–E6, A1–A4, G1 (laptop gate) and S0 (cluster smoke tier). Each
  variant changes one factor. `--pipeline`, `--datasets`, `--limit` and `--shard K/N` re-target a
  spec without editing it.
* `--set key=value` overrides any pipeline field from the CLI.
* Environment (`.env`, see [`.env.example`](.env.example)): API keys, database passwords, model
  server URLs and storage locations (`ELEVATE_DATA_DIR`, `ELEVATE_INDEX_DIR`, `ELEVATE_CACHE_DIR`,
  `ELEVATE_RUNS_DIR`). Run ids depend only on what changes outputs: not on paths, server URLs,
  concurrency or timeouts, so runs resume across disks, machines and cluster nodes.

## Repository layout

```
src/elevate_rag/
  data/          dataset loaders (pinned revisions), preparation, splits, synthetic corpus
  indexing/      chunking and the shared chunk store
  retrieval/     BM25 and rank fusion
  graph/         LightRAG engine, entity resolution, graph quality, communities
  storage/       vector (exact, FAISS, Qdrant, Milvus, pgvector) and graph (NetworkX, Neo4j,
                 Memgraph, Postgres) adapters
  routing/       routers R0–R4, fallback, oracle
  generation/    context assembly (shared token budget) and the answer generator
  providers/     LLM / embedding / reranker clients, cost ledger, retries
  evaluation/    metrics, LLM judge, harness, human-annotation tooling
  experiments/   experiment specs → resumable, shardable runs; external-system predictions
  reporting/     tables, paired tests, figures
  cli/           the `elevate-rag` command
configs/         pipeline and experiment YAML
infrastructure/
  docker/        docker-compose for the databases (Qdrant, Neo4j, Milvus, pgvector, Memgraph)
  cluster/       SLURM jobs, vLLM start-up, Apptainer image definition, site settings (env.sh)
scripts/         run_pipeline.sh, download_models.py, model-free measurements, EC2 setup
tests/           unit, integration (live DBs, skipped when absent), end-to-end
docs/            requirements, architecture, decisions, progress, running, deployment
results/         committed results with write-ups; raw runs are git-ignored
```

## Development

```bash
make check        # ruff + mypy + pytest (what CI runs)
make cov          # coverage report
make db-up        # live DBs, so the integration tests run instead of skipping
make cluster-dry  # the SLURM jobs `submit.sh smoke` would submit (no cluster needed)
```

Tests are offline. DB-service conformance tests run when `ELEVATE_*` URLs point at live services
and are reported as skipped otherwise. CI (`.github/workflows/ci.yml`) runs lint, format, types,
shellcheck, a cluster dry run and the tests from the lockfile, and builds the Docker image and
runs the offline demo inside it.

## Documentation

| Document | Contents |
|---|---|
| [docs/RUNNING.md](docs/RUNNING.md) | **how to run and reproduce everything**: laptop, GPU, cluster, API |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | running on AWS EC2: sizing, security, setup, operations |
| [results/README.md](results/README.md) | every committed result, what it shows, how it was produced |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | requirements extracted from the plan and spec |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components, invariants, data flow |
| [docs/DECISIONS.md](docs/DECISIONS.md) | decisions with rationale and alternatives |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | phases, gates and acceptance criteria |
| [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md) | requirement → code → tests → status |
| [docs/PROGRESS.md](docs/PROGRESS.md) | what was verified, with evidence |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | limitations, blockers, human-dependent tasks |

Dataset licences are listed by `elevate-rag data list`. Large data, indexes, caches and runs are
git-ignored.
