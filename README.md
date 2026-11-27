# Elevate-RAG

A reproducible research platform for the question in *Hybrid Vector + Graph RAG: Project Design &
Evaluation Plan*: **when does graph structure pay for itself, and can a router capture that benefit
automatically?**

One shared index (chunks, dense vectors, BM25, a LightRAG knowledge graph) serves every retrieval
path: vector baselines V0–V4, LightRAG `naive/local/global/hybrid/mix`, and routers R0–R4. All paths
share the same reranker, token budget and generation prompt, so retrieval mode is the only
variable. Experiments (E1–E6, A1–A4) are config-driven, resumable, cost-capped and report bootstrap
CIs and paired tests.

> **Status (2026-11-26):** the software is implemented and tested (126 tests, 86 % coverage;
> Qdrant and Neo4j adapters verified against live servers). Datasets are prepared and model-free
> baselines are measured. **No research experiment requiring an LLM has been run yet**: that
> needs an API budget or a local model server, the judge-model choice and human annotators. See
> [`docs/PROGRESS.md`](docs/PROGRESS.md) and [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

## Quick start (offline, no credentials, ~20 s)

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is installed by uv if missing).

```bash
uv sync --locked --extra faiss       # venv from the lockfile
uv run elevate-rag demo              # synthetic corpus → index → all modes/routers → report
open demo_output/report/REPORT.md    # every number is labelled DEMO (stand-in models)
```

Or with Docker only: `make image && make image-smoke`.

## Deployment (AWS EC2)

The platform is a batch CLI. On a server it runs as a Docker image next to the database
containers, with all state on one mounted data volume and no inbound ports except SSH.
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) covers instance sizing, security, setup and
operations:

```bash
bash scripts/ec2_setup.sh                       # Docker + Compose on Ubuntu 24.04
cp .env.example .env && chmod 600 .env          # passwords, API keys
make db-up && make image && make image-smoke    # databases, app image, offline self-test
```

The demo uses deterministic offline stand-ins (hash embedder, lexical reranker, heuristic
extractor, extractive generator). They exercise the real pipeline, LightRAG included, but are
**not** research models.

## Research workflow

```bash
cp .env.example .env                 # fill in only what you use; .env is git-ignored
uv run elevate-rag env               # Python, packages, Docker, which credentials are present

# 1. Data (pinned Hugging Face revisions, checksums, 20 % dev / 80 % test, manifests)
uv run elevate-rag data prepare all
uv run elevate-rag data info hotpotqa

# 2. Databases (reference = Qdrant + Neo4j; PROFILE=all for E4)
make db-up

# 3. Choose the judge (different model family from every generator) and a spending cap in
#    configs/pipelines/reference.yaml, then pilot-index 5 % and extrapolate cost (plan §Risks)
uv run elevate-rag index pilot configs/pipelines/reference.yaml --set dataset=multihop_rag

# 4. Build and validate the graph (gate G3: answer-entity coverage ≥ 60 %; triple audit sample)
uv run elevate-rag index build configs/pipelines/reference.yaml --set dataset=multihop_rag
uv run elevate-rag index validate configs/pipelines/reference.yaml --set dataset=multihop_rag \
    --audit-out results/audit/multihop_rag_triples.csv

# 5. Experiments: validate → estimate → dry-run → run (completed runs are skipped on re-run)
uv run elevate-rag experiment validate configs/experiments/E1_vector_baselines.yaml
uv run elevate-rag experiment estimate configs/experiments/E2_retrieval_modes.yaml
uv run elevate-rag experiment run configs/experiments/E2_retrieval_modes.yaml --dry-run
uv run elevate-rag experiment run configs/experiments/E2_retrieval_modes.yaml

# 6. Routers: train R2 on dev data only, then E3
uv run elevate-rag experiment run configs/experiments/E3_labels_dev.yaml
uv run elevate-rag router train runs/routers/r2-type --scheme type --dataset ragrouter
uv run elevate-rag router train runs/routers/r2-outcome --scheme outcome --runs runs/E3-labels \
    --dataset multihop_rag --dataset hotpotqa
uv run elevate-rag router cv --dataset ragrouter --leave-one-dataset-out

# 7. Level-A database benchmarks (matched recall@10 = 0.95 vs exact search)
uv run elevate-rag bench vector configs/pipelines/reference.yaml --backends exact,faiss,qdrant,milvus,pgvector
uv run elevate-rag bench graph configs/pipelines/reference.yaml --backends networkx,neo4j,memgraph,postgres

# 8. Human validation (never fabricated): judge sample, triple audit, failure analysis
uv run elevate-rag annotate judge-sample runs/E2/<run_dir> results/annotation/judge_sample.csv
uv run elevate-rag annotate judge-agreement results/annotation/judge_sample.csv a.csv b.csv runs/E2/<run_dir>
uv run elevate-rag annotate failures runs/E2/<run_dir> results/annotation/failures_local.csv

# 9. Reports: tables, McNemar + Holm, figures
uv run elevate-rag report --runs-root runs --out results/report --reference V4
```

External reference systems (Microsoft GraphRAG, HippoRAG 2, LlamaIndex PGI, Neo4j GraphRAG) run
with their own tooling and are scored through the same evaluator:
`elevate-rag experiment external preds.jsonl --system hipporag2 --dataset musique`.

Model-free measurements on real data (BM25 retrieval, rule and TF-IDF routers):
`uv run python scripts/model_free_measurements.py`.

## Configuration

* `configs/pipelines/reference.yaml`: the plan's reference configuration with verified model IDs
  and revisions. `local_dev.yaml` uses the same models without Docker; `cpu_small.yaml` runs on a
  laptop CPU; `demo_offline.yaml` is offline.
* `configs/experiments/*.yaml`: E1–E6 and A1–A4. Each variant changes one factor via dotted
  overrides (`retrieval.context_budget: 4096`) or picks a mode/router.
* `configs/pipelines/local_free.yaml`: a $0 configuration with small local models (Ollama) for
  smoke-testing the full pipeline; its numbers are not comparable to the reference.
* `--set key=value` overrides any field from the CLI.
* Environment (`.env`, see [`.env.example`](.env.example)): API keys, database URLs and passwords,
  and storage locations (`ELEVATE_DATA_DIR`, `ELEVATE_CACHE_DIR`, `ELEVATE_INDEX_DIR`,
  `ELEVATE_RUNS_DIR`). Storage locations are not part of a run's identity, so moving data between
  machines does not invalidate completed runs.

## Repository layout

```
src/elevate_rag/
  data/          dataset loaders (pinned revisions), preparation, splits, synthetic corpus
  indexing/      chunking and the shared chunk store
  retrieval/     BM25 and rank fusion
  graph/         LightRAG engine, entity resolution, graph quality, communities
  storage/       vector (Exact, FAISS, Qdrant, Milvus, pgvector) and graph (NetworkX, Neo4j,
                 Memgraph, Postgres) adapters
  routing/       routers R0–R4, fallback, oracle
  generation/    context assembly (shared token budget) and the answer generator
  providers/     LLM / embedding / reranker clients, cost ledger, retries
  evaluation/    metrics, LLM judge, harness, human-annotation tooling
  experiments/   experiment specs → resumable runs; external-system predictions
  reporting/     tables, paired tests, figures
  cli/           the `elevate-rag` command
configs/         pipeline and experiment YAML
infrastructure/  docker-compose for the databases
scripts/         model-free measurements, EC2 host setup
tests/           unit, integration (live DBs, skipped when absent), end-to-end
docs/            requirements, architecture, decisions, progress, deployment
results/         committed results (model-free measurements); run outputs are git-ignored
```

## Development

```bash
make check        # ruff + mypy + pytest (what CI runs)
make cov          # coverage report
make db-up        # live DBs, so the integration tests run instead of skipping
```

Tests are offline. DB-service conformance tests run when `ELEVATE_*` URLs point at live services
and are reported as skipped otherwise. CI (`.github/workflows/ci.yml`) runs lint, types and tests
from the lockfile, and builds the Docker image and runs the offline demo inside it.

## Documentation

| Document | Contents |
|---|---|
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | requirements extracted from the plan and spec |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components, invariants, data flow |
| [docs/DECISIONS.md](docs/DECISIONS.md) | decisions with rationale and alternatives |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | phases, gates and acceptance criteria |
| [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md) | requirement → code → tests → status |
| [docs/PROGRESS.md](docs/PROGRESS.md) | what was verified, with evidence |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | limitations, blockers, human-dependent tasks |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | running on AWS EC2: sizing, security, setup, operations |

Dataset licences are listed by `elevate-rag data list`. Large data, indexes, caches and runs are
git-ignored.
