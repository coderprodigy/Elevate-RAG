# CLAUDE.md

ECE 750 group project: **Hybrid Vector + Graph RAG** with a learned query router. The project measures
when graph retrieval beats vector retrieval, and whether a router can capture that benefit cheaply.

- Plan and phase status: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md). Read §1.1 (decisions) and the current phase before changing code.
- Setup the team does by hand: [docs/setup/README.md](docs/setup/README.md)
- What is still open, by phase: [docs/PENDING.md](docs/PENDING.md). Update it at the end of every step.

## Commands

```bash
make setup        # uv sync (creates .venv) + pre-commit hooks
make check        # lint + typecheck + unit tests: run before declaring any step done
make test         # unit tests only (no network, no keys, no Docker)
make test-int     # integration tests (need API keys / Docker / server tunnel)
make usage        # API spend by role (same as: uv run hgrag usage --by role)
uv run hgrag --help
uv run hgrag eval run --pipeline v0 --experiment g1_smoke --limit 20 --model claude-haiku-5-5 --dry-run
uv run hgrag index build -e bge-small          # free: chunks, gold map, BM25, embeddings, HNSW
uv run hgrag index evaluate -d multihop_rag    # free: V1-V4 retrieval recall on dev
uv run hgrag exp plan --exp e2_modes           # every run of an experiment: cost bound + blockers
uv run hgrag graph estimate                    # graph indexing cost upper bound (no calls)
uv run hgrag router cv                         # free: R2 classifier CV on RAGRouter-Bench
uv run hgrag report build                      # all tables + figures from runs/ (no calls)
```

The project path contains spaces, so always quote it in shell commands.

## Working agreement

- Implement **one phase sub-step at a time**, then stop and report: what changed, the test results, and
  what the user must do next. The user tracks Claude Code usage between steps.
- **Never start a paid run estimated above `budget.confirm_above_usd` ($2) without first showing the
  `--dry-run` estimate and getting a yes.**
- Don't commit or push unless asked. Never commit `.env`, `data/` or `runs/`.
- Waterloo GPU server: needed from day 4 (vLLM for open models, Qdrant/Neo4j). Pause and ask the user for
  server details then (see [docs/setup/05_waterloo_servers.md](docs/setup/05_waterloo_servers.md)).
  Never ask for passwords or keys in chat.

## Architecture rules (do not break these)

1. **All paid calls go through `hgrag.llm.gateway.LLMGateway`**: it handles the cache, usage ledger and
   budget guard. Only `src/hgrag/llm/providers/` may import `anthropic` or `openai`; a test enforces this.
2. **We own generation.** LightRAG is used for retrieval only. Every pipeline (V0–V4 and the graph
   modes) answers through the same generator prompt.
3. **Our chunker is canonical.** LightRAG receives our chunks; chunk IDs are content hashes and must
   match across the vector and graph indexes.
4. **One variable at a time.** Within an experiment, only the factor under test may differ from the
   reference config.
5. **The test split is locked.** `split=test` runs only when the experiment config has `frozen: true`.
   Tune on dev only.
6. **Same context budget** (token cap) for every retrieval path before generation.
7. Local models (embedders, rerankers) download from Hugging Face anonymously unless `HF_TOKEN`
   is set; never rely on the cached CLI login token (an expired one breaks downloads).
8. LightRAG: one *workspace* per graph (process-wide namespaces); pipelines must `aclose()`
   their LightRAG storages, and the runner does this after every run.
9. Claude 5.5 models reject `temperature`. Model specs carry `supports_temperature: false`, and
   determinism comes from the response cache. Don't add a server-side model fallback, because it would
   silently swap models mid-experiment; log refusals instead.

## Code layout

```
src/hgrag/
  config.py        layered YAML → Pydantic (configs/base, models, budget, datasets .yaml)
  core/            hashing, run IDs, shared types
  llm/             gateway, cache (SQLite), ledger (SQLite), budget, pricing, providers/
  data/            Phase 1: loaders (one per source format), splits (frozen), data cards
  commands/        CLI groups: data, index, eval, graph, router, bench, exp, report
  pipelines/       Pipeline protocol + Prediction; v0 closed-book; v1-v4 vector RAG; registry
  generation/      prompts (versioned) and answer conventions (refusal phrase, citations)
  eval/            runner, scoring, metrics/, stats, judge, pairwise, faithfulness/evidence recall,
                   human labels + kappa, experiments (frozen flag)
  cli.py           Typer CLI `hgrag`
  index/           chunking, gold-chunk maps, embedders, BM25, FAISS HNSW, index build
  retrieval/       searcher (bm25|dense|hybrid + rerank), RRF, token budget, free retrieval eval
  graph/           LightRAG adapter (our chunks in, aquery_data out), build, GraphIndex, quality, audit
  router/          R0-R4 routers, outcomes from E2 runs, offline evaluation, RAGRouter CV
  bench/           vector + graph store adapters and Level A benchmark harnesses
  analysis/        collect runs, paired comparisons, tables, figures (validated palette), errors
configs/           models.yaml = frozen model registry + prices; roles map job → model
configs/retrieval.yaml  chunkers, embedders, reranker, HNSW, search settings (k, budget)
configs/graph.yaml      LightRAG backends (local / qdrant_neo4j / milvus_memgraph / postgres), query caps
configs/router.yaml     router paths, rules, classifier, fallback thresholds
docker/                 database stacks for E4 (equal limits, localhost only)
configs/experiments/  one YAML per experiment; `frozen: true` unlocks --split test
data/indexes/<dataset>/<chunker>/  chunks, gold map, bm25/, emb-<embedder>/ (ignored)
runs/<exp>/<dataset>__<pipeline>__<split>/  config, predictions, usage, scores, metrics (ignored)
tests/unit         fast, offline; uses the `fake` provider
tests/integration  marked `integration`
tests/fixtures/tiny_corpus   15 questions over a fictional corpus, one per query type
reports/           generated tables and figures for the final report (tracked)
data/processed/*/splits.json  frozen dev/test splits (tracked; everything else in data/ is ignored)
```

## Coding standards

- Python 3.12, full type hints, `mypy --strict` on `src/`, `ruff` for lint and format (line length 100).
- Small modules that each do one job. Docstrings state units (tokens, ms, USD).
- No hidden globals: pass config objects explicitly. Secrets only from `.env`.
- Metrics and statistics are pure functions, unit-tested against hand-computed values.
- Every run writes its resolved config, git SHA, model IDs and timestamps under `runs/<exp>/<run_id>/`.
- Prefer clear names over comments; comment only the *why* of non-obvious choices.
- Add dependencies in the phase that needs them (`uv add <pkg>`), not earlier.
