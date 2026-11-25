# Hybrid Vector + Graph RAG — Implementation Plan

Source design: *Hybrid Vector + Graph RAG: Project Design & Evaluation Plan* (Nov 20, 2026).
This file turns that design into an engineering plan: architecture decisions, repo layout, and
nine phases, each with a definition of done and a test you can run.

Setup you must do yourself (accounts, keys, datasets, Docker, compute) is collected in
[docs/setup/README.md](docs/setup/README.md).

---

## Implementation status (2026-11-25)

All eight phases are **implemented and tested offline**: 127 unit tests pass, `ruff` and
`mypy --strict` are clean. Paid and server runs are tracked in [docs/PENDING.md](docs/PENDING.md);
problems and their fixes are in [docs/PROBLEMS.md](docs/PROBLEMS.md).

| Phase | Code | Verified offline | Waiting on |
|---|---|---|---|
| 0 Foundations | ☑ | gateway, cache, ledger, budget | API key (live ping) |
| 1 Data | ☑ | 7 datasets built, splits frozen | — |
| 2 Eval harness | ☑ | runner, metrics, stats, judge, kappa, variance | API key + judge model |
| 3 Vector V1–V4 | ☑ | indexes for 6 corpora; dev retrieval measured | generator (Haiku stand-in OK); Qwen3 embeddings (running) |
| 4 Graph (LightRAG) | ☑ | real LightRAG build + all 5 modes on fixture | API key (sample builds ≈ $1.91) |
| 5 Router | ☑ | R0–R4 + oracle; RAGRouter CV 0.964 | E2 runs |
| 6 Databases | ☑ | FAISS / NetworkX benches | Docker on the Waterloo server |
| 7 Experiments | ☑ | planner + 9 experiment YAMLs | model keys for open models (day 4) |
| 8 Analysis | ☑ | tables, figures, comparisons, error tagging | real runs |

## 0. How we will work (and how you track usage)

**Implementation cadence.** I implement **one phase at a time**, usually as 2–4 sub-steps.
After each sub-step I stop and report what changed, which tests ran and their results, and anything
you must do before the next step. You decide when to continue, so you can check your Claude Code usage
between steps.

**Spend tracking inside the project.** From Phase 0, every paid call goes through one metered
gateway:

| Mechanism | What it gives you |
|---|---|
| Usage ledger (`runs/usage.sqlite`) | One row per LLM, embedding or rerank call: time, run ID, phase, role, model, input/output/cached tokens, cost in USD, latency, cache hit |
| `hgrag usage` CLI | Spend grouped by day, phase, role, model or run, e.g. `hgrag usage --by role --since 2026-12-01` |
| Budget guard | A per-run cap and a global cap in `configs/budget.yaml`. A run that would go over stops cleanly, saves a checkpoint, and can be resumed |
| `--dry-run` | Prints the planned calls and the estimated cost, then exits without spending anything |
| `--limit N` | Runs any job on N questions first, so you can smoke-test it cheaply |
| Response cache | Calls are cached by content hash (protocol rule 6), so a rerun costs $0 and the ledger marks it as a cache hit |

**Rule:** I will never start a paid run over a small threshold (default $2) without showing you
the dry-run estimate first.

---

## 1. Lead assessment of the design

The design is strong. It has a clear hypothesis, a strong baseline (V4), an oracle upper bound,
paired statistics, judge validation, and caching. The issues below are what I'd fix or decide before
writing code. Each one is reflected in the phases.

### 1.1 Decisions I am making (please tell me if you disagree)

| # | Decision | Why |
|---|---|---|
| D1 | **We own generation. LightRAG is used for retrieval only** (context/data-only query APIs). Every path, V0–V4 and every graph mode, sends its context through our one generator prompt. | If LightRAG's own answer prompt were used, the prompt would change along with the retrieval mode, which breaks protocol rule 1. It is also the only way to enforce "cite passage numbers" and "insufficient information" in the same way on every path. |
| D2 | **Our chunker is the single source of truth, and LightRAG consumes our chunks** through its custom chunking hook. The design doc does it the other way round ("Phase 1 reads LightRAG's chunk table"). | We keep sentence-boundary chunking, paragraph mode and the chunk-size ablations under our control. A test checks that the vector and graph indexes contain the same set of chunk IDs. |
| D3 | **One metered LLM gateway** (`hgrag.llm`) with provider adapters: the official `anthropic` SDK for Claude, the `openai` SDK for OpenAI and for OpenAI-compatible hosts of open-weight models, and an optional local Ollama adapter. LightRAG's `llm_model_func` and `embedding_func`, the RAGAS LLM and the judge all wrap this gateway. | One place for caching, the cost ledger, budget caps, retries and Langfuse tracing. Nothing can spend money without it being recorded. |
| D4 | **A uniform "context unit" and token cap.** Retrieval returns typed units (chunk, entity, relation). The reranker scores chunk units, and a single `ContextBudget` trims every path to the same token limit (default 8k, ablated at 4k/8k/16k) before generation. LightRAG's own `max_*_tokens` limits are also set as a first cap. | Prevents a comparison from turning into a comparison of context length, which the design doc warns about. |
| D5 | **The test split is locked in code.** `split=test` refuses to run unless the experiment config has `frozen: true`, which you set at gate G2/G4. | Enforces "never tune on the test portion" in code instead of relying on discipline. |
| D6 | **Configuration** is layered YAML (`base` → dataset → experiment → CLI overrides), validated by Pydantic. Each run writes a resolved `config.yaml`, git SHA, model IDs and dates next to its results. | Reproducibility appendix for free. |
| D7 | **Tooling:** Python 3.12, `uv`, `ruff` (lint and format), `mypy` on `src/`, `pytest`, `pre-commit`, a `Makefile`, and a Typer CLI `hgrag`. | Clean, conventional and fast. |

### 1.2 Issues found in the design doc, and how they are resolved

Status (agreed 2026-11-24): every item below is **accepted as written**. Items 4–11 were delegated to
me, the engineering lead. Item 2 now uses the Waterloo servers instead of a hosted endpoint, item 3 adds
Docker or Apptainer on those servers, and item 12 uses the compressed schedule in §6.

1. **"Temperature 0" isn't possible on current Claude models.** (The design compares Claude Haiku 5.5
   and Sonnet 5.5 as extractors and Sonnet 5.5 as the strong API generator, so this applies.) Claude Haiku 5.5 and Sonnet 5.5
   return HTTP 400 for non-default `temperature`/`top_p`/`top_k`, and the Anthropic API has no `seed`.
   *Fix:* use temperature 0 wherever a provider allows it (open-weight hosts, OpenAI). For Claude, use
   default sampling and the lowest thinking/effort setting that still works (Haiku 5.5: thinking off at
   `effort: low`; Sonnet 5.5: `thinking: {type: "between_tools"}`). Make reruns deterministic with the
   response cache, and **measure run-to-run variance** (50 questions × 3 uncached runs) as a reported
   number in the methodology section.
2. **The "small, local" extractor (Qwen3.6-35B-A3B) won't run on a 16 GB Mac**, and neither will the
   mid-tier generators or Qwen3-Embedding-8B. *Fix (decided):* serve open-weight models with **vLLM on
   the Waterloo GPU servers**, behind an OpenAI-compatible endpoint that the Mac reaches over an SSH
   tunnel. See [docs/setup/05_waterloo_servers.md](docs/setup/05_waterloo_servers.md). A hosted endpoint
   ([04](docs/setup/04_compute_for_open_models.md)) is the fallback if server access is delayed.
   Small embedders and rerankers (bge-small, bge-reranker-v2-m3, Qwen3-Embedding-0.6B) run locally on MPS.
3. **Docker isn't installed, and 16 GB RAM can't run every database at once.** *Fix (decided):* run the
   database stacks on the Waterloo server: Docker or Podman if allowed, otherwise Apptainer running the
   same images. They're reached over the same SSH tunnel. Separate stacks (`core`: Qdrant + Neo4j;
   `bench-*`: one engine at a time) keep memory limits fair, and the same Compose files also work on a
   Mac with OrbStack. FAISS and NetworkX remain the setup-free fallbacks, so Phases 0–3 never wait
   on database setup.
4. **The HotpotQA and MuSiQue distractor settings are per question.** If each question kept its own
   10–20 paragraphs, retrieval would be trivial. *Fix:* pool the paragraphs of all sampled questions
   (train/dev/test) into **one corpus per dataset**, deduplicated by (title, text), as Han et al. do.
5. **MultiHop-RAG gold evidence is fact text, not passage IDs.** *Fix:* map each evidence fact to the
   chunk(s) that contain it (normalised fuzzy containment, threshold tuned on 50 hand-checked
   cases), and define recall@k at chunk level. The mapping rate is reported in the data card.
6. **RAGRouter-Bench contains MuSiQue questions**, so they can leak into our MuSiQue test split when R2
   is trained on RAGRouter-Bench. *Fix:* remove any RAGRouter-Bench question whose normalised text
   matches a question in any of our test splits.
7. **Outcome labels for the router (label 2) need every mode run on the training data.** *Fix:* E2 runs
   all modes on **dev and test** splits. The dev results provide the outcome labels, and the cost is
   included in the budget.
8. **"Cheapest mode" for the oracle needs a cost ordering.** *Fix:* order modes by measured mean
   tokens per query on dev, per dataset. Expected order: naive < local < global < hybrid < mix.
9. **RAGAS faithfulness makes several LLM calls per answer.** *Fix:* route RAGAS through the gateway
   using the judge model, and run it on a fixed, stratified 200-question subset per system unless the
   budget allows more.
10. **Statistical power.** An 80% test split of NQ gives 400 test questions (and MuSiQue 480), so the
    minimum detectable effect is closer to 5–7 points than 3–5. The planned power check on dev
    (Phase 2 tooling) decides whether to enlarge samples.
11. **Judge validation must happen before any result counts** (kappa ≥ 0.6). *Fix:* it is an exit
    criterion of Phase 3, which is the first phase that produces real answers, not something left
    for the end.
12. **Timeline.** The design assumed ten weeks; the project actually has about 20 days. *Decided:*
    build all of the code in about one week, then spend the remaining time on full runs (which are
    limited by API rate limits and GPU time, not by coding), analysis and writing. See §6. The
    Must tier in §5 is protected first.

### 1.3 Rough indexing cost

These are order-of-magnitude figures, to be checked by the 5%-sample measurement in Phase 4.

At list prices, Claude Haiku 5.5 costs $0.10 per million input tokens and $0.50 per million output
tokens (prompts up to 100K tokens); Sonnet 5.5 costs $2 and $10. Assume about 3k input and 1k output
tokens per chunk for extraction:

| Extractor | Cost per chunk | ~10k chunks (HotpotQA-size corpus) |
|---|---|---|
| Haiku 5.5 | ≈ $0.0008 | ≈ $8–16 (including gleaning) |
| Sonnet 5.5 | ≈ $0.016 | ≈ $160–320 |

This is why the design limits the strong-extractor comparison to two datasets. Prices go in
`configs/models.yaml` and are confirmed when model IDs are frozen.

---

## 2. Architecture

```
Offline (per dataset)                                    Online (per query)
──────────────────────                                   ────────────────────────────────────────
raw dataset ─► loader ─► corpus.jsonl / queries.jsonl    query ─► Router (R0–R4)
                          │                                         │ picks a path
                          ▼                                         ▼
                      Chunker (canonical chunk IDs)       ┌─ VectorPath: BM25 | dense | hybrid RRF
                     ┌────┴──────────┐                    ├─ GraphPath: LightRAG local|global|hybrid (retrieval only)
                     ▼               ▼                    └─ FusedPath: LightRAG mix
              Embedder → VectorStore   LightRAG (our chunks,         │ ContextUnits
              BM25 index               metered LLM) → GraphStore     ▼
                                       + entity/relation vectors   Reranker → ContextBudget (token cap)
                                                                    ▼
                                                                  Generator (one prompt, cites [n],
                                                                  "insufficient information")
                                                                    ▼
                                                                  Prediction → Evaluation harness
All LLM / embed / rerank calls ─► LLM Gateway (cache · ledger · budget · Langfuse)
```

**Core interfaces** (`src/hgrag/core/types.py`, `protocols.py`):

- `Document`, `Chunk(chunk_id, doc_id, text, n_tokens, meta)`,
  `Query(qid, text, answers, gold_chunk_ids, query_type, answerable, split)`
- `ContextUnit(kind: chunk|entity|relation, id, text, score, source_chunk_ids)`
- `Retriever.retrieve(query) -> list[ContextUnit]`
- `Pipeline.run(query) -> Prediction(answer, cited_units, contexts, route, usage, latency)`
- `Router.route(query) -> RouteDecision(mode, confidence, features)`
- `VectorStore` and `GraphStore` protocols with FAISS/Qdrant/Milvus/pgvector and
  NetworkX/Neo4j/Memgraph/Postgres adapters

## 3. Repository layout

```
ECE_750_Group_Project/
├── IMPLEMENTATION_PLAN.md          ← this file
├── README.md                       quickstart (Phase 0)
├── pyproject.toml  uv.lock  Makefile  .pre-commit-config.yaml  .env.example
├── configs/
│   ├── base.yaml                   chunking, top-k, token cap, seeds
│   ├── models.yaml                 frozen model IDs + prices (filled in at week-1 freeze)
│   ├── budget.yaml                 per-run and global USD caps
│   ├── datasets/{nq,hotpotqa,musique,multihop_rag,graphrag_bench,ragrouter_bench}.yaml
│   └── experiments/{e1_baselines,e2_modes,e3_router,e4_db,e5_models,e6_interaction,ablations}.yaml
├── docker/
│   ├── compose.core.yaml           Qdrant + Neo4j (reference DBs)
│   ├── compose.bench-vector.yaml   Milvus, pgvector (one profile at a time)
│   └── compose.bench-graph.yaml    Memgraph, Postgres graph tables
├── src/hgrag/
│   ├── config.py  cli.py  logging.py
│   ├── core/          types.py, protocols.py, ids.py (content hashing)
│   ├── llm/           gateway.py, providers/{anthropic,openai_compat,ollama}.py,
│   │                  cache.py, ledger.py, budget.py, pricing.py, tracing.py
│   ├── data/          loaders/<dataset>.py, normalize.py, splits.py, gold_mapping.py, datacard.py
│   ├── chunking/      sentence_chunker.py, paragraph_chunker.py
│   ├── embedding/     local_st.py, api.py
│   ├── stores/        vector/{faiss,qdrant,milvus,pgvector}.py, graph/{networkx,neo4j,memgraph,postgres}.py
│   ├── retrieval/     bm25.py, dense.py, fusion.py (RRF), rerank.py, context_budget.py
│   ├── graph/         lightrag_adapter.py, quality.py (coverage, shape), audit.py, llamaindex_fallback.py
│   ├── generation/    prompts/*.txt, generator.py, refusal.py
│   ├── pipelines/     closed_book.py (V0), vector.py (V1–V4), graph.py (LightRAG modes), routed.py
│   ├── router/        rules.py (R1), classifier.py (R2), llm_router.py (R3), fallback.py (R4),
│   │                  oracle.py, labels.py
│   ├── eval/          runner.py (resumable), metrics/{qa,retrieval,refusal,judge,ragas,pairwise,router,efficiency}.py,
│   │                  stats.py (bootstrap, McNemar, paired bootstrap, Holm, power), human_labels.py
│   ├── bench/         vectordb_bench.py, graphdb_bench.py
│   └── analysis/      tables.py, figures.py, error_analysis.py
├── tests/  unit/  integration/ (marked, need Docker or keys)  fixtures/tiny_corpus/
├── data/   raw/  processed/          (git-ignored; see docs/setup/03_datasets.md)
├── runs/   <experiment>/<run_id>/{config.yaml, predictions.jsonl, metrics.json, usage.json}
├── reports/  figures/  tables/
└── docs/   setup/  adr/  annotation_guidelines.md
```

## 4. Phases

Each phase lists its **goal**, **work items**, **tests and how to verify**, **exit criteria**,
**owner role** (the design doc's five roles) and **target weeks**. Phase 6 (infrastructure)
can run alongside Phases 3–5.

---

### Phase 0 — Foundations (week 1) · *Infra lead + Data/Eval lead*

> **Status (2026-11-24): done, apart from a live call.** `make check` passes: ruff, mypy --strict and 27
> unit tests. Still pending: one real `hgrag llm ping` once `ANTHROPIC_API_KEY` is in `.env`.
> Langfuse tracing is deferred, because the ledger already covers spend; it can be added as a gateway hook.

**Goal:** a clean skeleton in which every later phase plugs into the same config, gateway and CLI.

Work items
1. `git init`, `.gitignore` (data, runs, `.env`, caches), `pyproject.toml` with uv, ruff, mypy,
   pytest, pre-commit, and Makefile targets (`setup`, `lint`, `test`, `test-int`, `usage`).
2. `config.py`: Pydantic models for the layered YAML. `configs/models.yaml` with **role → frozen model
   ID → price** entries.
3. **LLM gateway** (D3): provider adapters (Anthropic SDK, OpenAI-compatible, Ollama), a SQLite
   response cache keyed by hash(model, params, messages), the usage ledger, budget guard, retry with
   backoff on 429/5xx, and an optional Langfuse trace.
4. CLI skeleton: `hgrag usage`, `hgrag config show`, `hgrag llm ping --role generator`.
5. Tiny fixture corpus (about 20 hand-written docs and 15 questions covering every query type) used
   by all unit and integration tests.

Tests
- `make test`: config validation, cache hit and miss, ledger cost arithmetic, budget guard aborts at the cap.
- `hgrag llm ping --role extractor` (costs well under $0.01). Then run `hgrag usage`, which shows one paid row; ping again and it shows one cache-hit row costing $0.

Exit: lint and tests pass in CI-like `make check`; one real call is recorded in the ledger.

---

### Phase 1 — Data layer (weeks 1–2) · *Data/Eval lead* → **gate G1 (part 1)**

> **Status (2026-11-24): done.** All 7 datasets download automatically and build in about 30 s; splits
> are frozen and tracked in git; the summary is in [reports/data_cards.md](reports/data_cards.md);
> 46 unit tests pass.
> Differences from the plan below:
> - MuSiQue comes from a Hugging Face mirror, so there is no manual step.
> - RAGRouter-Bench is on Hugging Face. Its medical domain is entirely GraphRAG-Bench medical, so
>   all 1,896 of those questions are dropped and R2 trains on 3 domains.
> - GraphRAG-Bench is built as two datasets (novel and medical).
> - MultiHop-RAG and GraphRAG-Bench gold evidence links to the article or book; mapping it to chunks
>   happens in Phase 3.

**Goal:** all datasets in one normalised format with gold evidence, query types and frozen splits.

Work items
1. Loaders → `data/processed/<ds>/{corpus.jsonl, queries.jsonl}` using the schema `doc_id, title, text, source`
   and `qid, question, answers[], gold_doc_ids[], gold_facts[], query_type, hop_count, answerable`.
   - **NQ:** DPR NQ-dev; corpus = gold passages + hard negatives of the 500 sampled questions.
   - **HotpotQA** distractor dev: 1,000 questions (all dev questions are level *hard*); pooled corpus (§1.2-4).
   - **MuSiQue:** 500 answerable + 100 unanswerable from the `full` dev file; hop count from the ID prefix.
   - **MultiHop-RAG:** 609 articles, all queries; gold facts mapped later to chunks (§1.2-5).
   - **GraphRAG-Bench:** novel + medical corpora with their task levels. Optionally cap corpus size, decided after a size check.
   - **RAGRouter-Bench:** router-only; with the dedup against our test sets (§1.2-6).
2. `splits.py`: deterministic 80/20 test/dev split, seeded and stratified by `query_type`, written once to
   `splits.json` and then treated as read-only.
3. `datacard.py`: a counts and statistics markdown table per dataset (feeds report §4a).

Tests
- Schema validation of every record; split determinism (same seed gives identical files); no qid appears in
  both splits; every gold doc ID exists in the corpus; expected counts per dataset.
- `hgrag data build --dataset all` followed by `hgrag data card`.

Exit: data cards committed; `splits.json` frozen (hash recorded).

---

### Phase 2 — Evaluation harness + V0 (weeks 2–3) · *Data/Eval lead* → **gate G1**

> **Status (2026-11-24): 2a and 2b done.**
> - 2a: `hgrag eval run|score`, the V0 pipeline, a resumable runner with dry run, test-split lock and
>   confirm threshold, EM/F1, retrieval, refusal and efficiency metrics, and statistics. 72 unit tests
>   pass. The G1 smoke dry run has an upper bound of $0.017.
> - 2b: LLM judge, pairwise judge (order-swapped), faithfulness and GraphRAG-Bench evidence recall
>   (our own RAGAS-style implementation through the gateway, not the `ragas` package), router
>   metrics, and the human-labelling CLI with kappa. Live runs wait for API keys (docs/PENDING.md A1, A3).

**Goal:** one runner and one metric suite that every pipeline uses through the `Pipeline` interface.

Work items
1. `eval/runner.py`: batched async execution, resumable (skips qids already in `predictions.jsonl`),
   with `--limit`, `--dry-run`, `--split dev|test` (test guarded by D5), and per-run `usage.json` from the ledger.
2. Metrics: EM and token F1 (SQuAD normalisation); recall@k and nDCG@10 against gold chunk IDs; refusal
   accuracy; **LLM-judge correctness** (correct, partial or wrong, fixed prompt, from a model family not used
   by any generator); pairwise judge with order swapping; router metrics (accuracy, macro-F1, regret, tokens
   saved); efficiency (p50/p95 latency, tokens per query, $ per 1,000 queries); RAGAS faithfulness and
   context precision/recall through the gateway.
3. `stats.py`: bootstrap 95% CI (1,000 resamples), McNemar, paired bootstrap, Holm–Bonferroni, power/MDE helper.
4. **V0 closed-book pipeline** (generator with no context).
5. `human_labels.py`: a small CLI to label answers and triples blind (answer order and system hidden), plus Cohen's kappa.

Tests
- Metric unit tests with hand-computed expected values (EM/F1 edge cases, nDCG, McNemar on a known table,
  bootstrap CI covers the true mean on synthetic data).
- **G1 check:** `hgrag eval --pipeline v0 --dataset all --split dev --limit 20` runs end to end, writes
  `metrics.json`, and the usage report shows the cost.

Exit (G1): loaders, splits and the harness run end to end on 20 questions per dataset.

---

### Phase 3 — Vector RAG baseline V1–V4 (weeks 2–4) · *Vector RAG lead* → **gate G2**

> **Status (2026-11-24): code done; paid runs pending.** Indexes are built for all 6 corpora, and
> free retrieval evaluation works (`hgrag index evaluate`).
> Differences from the plan below:
> - FAISS (HNSW) is the Phase 3 store; Qdrant moves to the day-4 server stack.
> - GraphRAG-Bench evidence is paraphrased, so its retrieval is scored with LLM-checked evidence
>   recall, not string-matched gold chunks.
> - hit@k is added (the standard NQ retrieval metric).
> - SQLite cache and ledger made thread-safe (a regression test was added).

**Goal:** a strong, tuned vector baseline (deliverable 1, E1, RQ1).

Work items
1. Chunkers: a sentence-boundary chunker (512/64 default; 256 and 1,024 for the ablation) and a paragraph
   chunker (Wikipedia datasets). Chunk IDs are content hashes (D2).
2. Embedders: local sentence-transformers (bge-small, Qwen3-Embedding-0.6B on MPS) and API embedders. The
   model name and version are stored with every vector. Embedding cache on disk.
3. Stores: FAISS (flat as the exact reference, HNSW) and Qdrant, with HNSW `M`, `ef_construction` and
   `ef_search` from config.
4. BM25 (`bm25s`), dense, and hybrid RRF (k = 60), each returning the top 50; cross-encoder reranker
   (bge-reranker-v2-m3) keeps the top 5–10; `ContextBudget` (D4).
5. Generator prompt: numbered passages, must cite `[n]`, must answer "insufficient information" when the
   context lacks the answer. Parser extracts citations and refusals.
6. Pipelines V1–V4; dev tuning of chunk size and top-k for V4.
7. **Judge validation:** two people each label the same 100 V4 dev answers → kappa ≥ 0.6, otherwise revise
   the judge prompt and repeat.
8. **Variance check** (§1.2-1): 50 questions × 3 uncached runs.

Tests
- Unit: RRF against a hand-worked example; the context budget never exceeds its cap; the citation parser.
- Integration on the fixture corpus: hybrid recall@10 ≥ dense recall@10; the reranker improves nDCG.
- `hgrag eval --exp e1_baselines --split dev` produces the V0–V4 table with confidence intervals.

Exit (G2): V4 configuration frozen in `configs/experiments/e1_baselines.yaml` (`frozen: true`); judge kappa
recorded; **E1 test-split run** done.

---

### Phase 4 — Graph RAG layer (weeks 3–6) · *Graph RAG lead* → **gate G3**

> **Status (2026-11-25): code done; paid builds pending (keys).**
> - LightRAG 1.5.8 is fed **our chunks** through `chunking_func`, with `file_path` = `doc_id`; a test
>   checks the chunk sets are identical.
> - Retrieval uses `aquery_data` and nothing else; every pipeline answers with our shared generator.
> - LightRAG's extract and keyword roles call our metered gateway.
> - Pipelines `lr_naive|local|global|hybrid|mix` share the reranker, `final_k` and token budget
>   with V4. Keyword-call cost is attributed to each query.
> - `hgrag graph estimate|build [--sample]|quality|audit|audit-label|audit-kappa`.
> - Upper bound for all 6 graphs with Haiku 5.5: **$38.21** (5% samples: about $1.91).
> - One LightRAG workspace per graph (see PROBLEMS P15).
> - Not built (Could tier): LlamaIndex fallback, community layer, Microsoft GraphRAG / HippoRAG 2
>   references, and the embedding-similarity entity-merge ablation (LightRAG merges by exact name).

**Goal:** a LightRAG graph over the same chunks and embeddings; all retrieval modes evaluated (deliverable 2, E2, RQ2, RQ6).

Work items
1. `lightrag_adapter.py`: pinned `lightrag-hku` (v1.5.7 per the design, confirmed at the freeze); our
   chunks injected (D2); `llm_model_func` and `embedding_func` wrap the gateway; per-role models (extract,
   keywords); NetworkX + FAISS/NanoVectorDB first, then Neo4j + Qdrant.
2. **Cost-first indexing:** `hgrag graph estimate --dataset X` → index a **5% sample**, measure tokens, and
   extrapolate; you approve; then the full index (resumable, cached).
3. Entity resolution: normalised-name merging plus an embedding-similarity threshold (ablation A2).
4. Retrieval-only graph pipelines for `local`, `global`, `hybrid` and `mix`; LightRAG's `naive` mode is
   run too, as a check against our V2.
5. **Graph quality checks:** answer-entity coverage; graph shape (nodes, edges, average degree, clustering,
   duplicate rate); triple-audit sampler (50 triples, two labellers, kappa).
6. Fallback rule: if coverage stays below 60% after tuning, rebuild that dataset's graph with LlamaIndex
   `SchemaLLMPathExtractor` behind the same `GraphStore` interface.
7. Optional community layer (ablation A4) and reference systems (Microsoft GraphRAG and HippoRAG 2 on two
   small datasets), both stretch tier.

Tests
- **Chunk equality test:** the set of chunk IDs in the vector index equals the set in LightRAG's chunk store.
- Every entity and relation has at least one source chunk ID that exists (traceability).
- Fixture-corpus integration test: build the graph, and `local` mode retrieves the expected entity for 3 known questions.
- `hgrag graph quality --dataset all` prints coverage and shape.

Exit (G3): every graph has answer-entity coverage ≥ 60% and an audited triple sample; **E2 runs on dev
and test** (dev runs feed the router's outcome labels).

---

### Phase 5 — Router R0–R4 + oracle (weeks 5–7) · *Router lead*

> **Status (2026-11-25): code done.**
> - Routers choose a *path* (`naive`=V4, `local`, `global`, `mix`; configs/router.yaml). Their
>   accuracy and cost are taken from the E2 per-mode runs, so evaluation is offline and free;
>   only R3 calls an LLM.
> - Levels: R0 static; R1 rules; R2 TF-IDF+SVM or embeddings+logistic regression, trained on
>   type labels and on outcome labels (oracle path on dev), leave-one-dataset-out; R3 LLM;
>   R4 fallback (threshold tuned on dev); oracle.
> - Reports regret, tokens saved vs `mix`, routing accuracy and macro-F1, and path share.
> - **First result:** R2 TF-IDF+SVM 5-fold CV on RAGRouter-Bench (deduplicated, 3 domains):
>   accuracy **0.964**, macro-F1 **0.963** (paper: 0.932 / 0.928 on all 4 domains).
> - Commands: `hgrag router cv`, `hgrag router evaluate --experiment e2_modes`.

**Goal:** deliverable 3, E3, RQ3.

Work items
1. R0 static (from E2 results, no new spend); R1 rules (keywords, entity count, question word);
   R2 TF-IDF+SVM and a small-encoder classifier; R3 zero- and few-shot LLM classifier (one short call);
   R4 confidence fallback (cheap path first, escalate to `mix` on a low top rerank score or "insufficient information").
2. Labels: (a) query-type labels from the datasets; (b) outcome labels, i.e. the cheapest correct mode on dev, using
   the mode cost ordering (§1.2-8).
3. Oracle and regret; cost saved against always running `mix`.
4. Evaluation: 5-fold cross-validation on RAGRouter-Bench (deduplicated); **leave-one-dataset-out** for R2;
   end-to-end routed runs on the test splits. **Routed runs reuse cached E2 retrievals and answers**, so
   only R3 and the R4 escalations cost new money.

Tests
- Unit: rule firing on crafted queries; the oracle picks the cheapest correct mode on a synthetic table;
  regret = 0 when the router equals the oracle.
- The R2 classifier reproduces sensible CV metrics on RAGRouter-Bench; R4 threshold tuned on dev only.

Exit: the accuracy-against-cost scatter (one point per router level) with confidence intervals.

---

### Phase 6 — Infrastructure + database comparison (weeks 4–8, in parallel) · *Infra lead*

> **Status (2026-11-25): code done; runs need Docker on the server.**
> - Compose stacks (`docker/`) with equal CPU and memory limits, localhost-only ports and pinned
>   images.
> - Vector adapters: FAISS flat/HNSW, Qdrant, pgvector, Milvus. Graph adapters: NetworkX,
>   Neo4j, Memgraph (Bolt), Postgres tables.
> - Level A harnesses: matched recall (smallest `ef` with recall@10 ≥ 0.95 vs exact), latency
>   p50/p95/p99, QPS at 1/8/32 clients, filtered search, write-to-searchable, memory
>   (docker stats), recall-vs-QPS curve. The graph harness reports load, lookup and 1/2-hop
>   latency, with every answer checked against NetworkX.
> - Level B = `hgrag eval run --graph-backend qdrant_neo4j|milvus_memgraph|postgres --limit 300`.
> - Commands: `hgrag bench vector|graph|health`.
> - First observation (synthetic test): HNSW at matched `ef` drops to about 0.83 recall under a
>   10%-selective filter, while exact search stays at 1.0. This is why filtered search is
>   reported separately.

**Goal:** E4, RQ4, and the reproducibility appendix.

Work items
1. Docker Compose stacks with pinned image tags, CPU and memory limits, and health checks; `make db-up STACK=core`.
2. **Level A, vector stores:** tune `ef` or `nprobe` per engine until recall@10 = 0.95 against exact FAISS
   search; then measure QPS at 1/8/32 clients, p50/p95/p99 latency, build time, peak RAM and disk, filtered
   search speed, and write-to-searchable delay; plus a recall-against-QPS curve per engine. 1,000 warm-up
   queries, 3 repeats, median reported.
3. **Level A, graph stores** (MultiHop-RAG graph): bulk load, 1- and 2-hop fetch for 1,000 random entities,
   lookup by name, memory, `local`-mode latency.
4. **Level B:** swap the backend inside LightRAG; fixed 300-question subset; accuracy, recall, latency,
   setup hours and lines of configuration.
5. Optional: Weaviate, Chroma and Redis through VectorDBBench (Level A only).

Tests
- `hgrag bench health` (every backend answers); a recall check against FAISS flat; benchmark output schema.

Exit: recall-against-QPS figure, latency table, setup-effort table; Compose files and engine versions in `reports/appendix/`.

---

### Phase 7 — Model comparison + ablations (weeks 6–8)

> **Status (2026-11-25): code done.**
> - Experiment YAMLs (`configs/experiments/`): `g1_smoke`, `e1_baselines`, `e2_modes`, `e4_db`,
>   `e5_models`, `e5_extractors`, `e6_interaction`, `a1_chunk`, `a3_budget`. Each is a `base` arm
>   plus `vary` (one factor at a time, checked by `violations()`) or `grid` (factorial).
> - `hgrag exp list|plan|run --exp <name> [--split]`. The plan prices every arm × dataset
>   instantly and lists *blocked* runs with the reason (missing graph or index, unknown model
>   key), so it doubles as a to-do list.
> - Factors: pipeline, generator model, chunker, embedder, reranker, extractor, graph backend,
>   context budget.
> - Not built: A2 entity-merge threshold (LightRAG merges by exact name) and A4 community
>   layer. Both are Could tier.

### Phase 7 (original section) · *Vector lead (E5 retrieval models), Graph lead (extractors), Eval lead (E6)*

**Goal:** E5, E6, A1–A4, RQ5.

Work items
1. `hgrag exp plan --exp e5_models` expands the one-factor-at-a-time matrix around the reference
   configuration and prints run count, cache reuse and **estimated cost**; you approve.
2. E5: embedders (4), rerankers (3), generators (3) on all datasets; extractors (3) on MultiHop-RAG and
   GraphRAG-Bench medical only.
3. E6: mode (naive, local, mix) × generator (3) on HotpotQA, MuSiQue and GraphRAG-Bench.
4. Ablations on MultiHop-RAG: chunk size, entity-merge threshold, context budget 4k/8k/16k, 1- against
   2-hop, community layer on/off.

Tests
- Dry-run matrix snapshot test (the expected run IDs); each run's `config.yaml` differs from the
  reference in exactly one field (enforces protocol rule 1).

Exit: factor-importance table (Δ accuracy with CI, Δ cost) and the interaction plot.

---

### Phase 8 — Analysis, error analysis, report assets (weeks 8–10) · *everyone* → **gate G4**

> **Status (2026-11-25): code done.**
> - `hgrag report build` regenerates every table (`reports/tables/*.md`) and figure
>   (`reports/figures/*.png|svg`) from `runs/` and `reports/`, with no API calls. Figures:
>   accuracy by query type and system, router accuracy vs cost, recall-vs-QPS, failure causes,
>   factor importance, and the mode × generator interaction.
> - The palette passed the dataviz validator (CVD and normal vision). Every figure has a
>   markdown table as its table view.
> - `hgrag report compare` runs paired McNemar and bootstrap tests with Holm correction and
>   reports "no clear difference" below the MDE.
> - `hgrag report errors-sample` / `errors-tag`: 50 failures per system with a heuristic cause
>   suggestion that a person confirms.

Work items
1. G4: all runs logged and checked against the ledger; the tag `results-frozen` set in git.
2. Statistics pass: CIs, McNemar with Holm correction, "no clear difference" below the minimum detectable effect.
3. Error analysis: sample 50 failures per main system; a tagging CLI with the 6 causes; failure-cause chart.
4. Figures and tables for the report: architecture, accuracy by query type and mode (the central figure),
   router scatter, recall-against-QPS, failure causes, cost per 1,000 queries.
5. Appendix: prompts, hyperparameters, Compose files, model IDs and dates, total spend (`hgrag usage --total`).

Tests: `make report-assets` regenerates every figure and table from `runs/` alone (no API calls; checked
with the network off).

---

## 5. Scope tiers (if time or budget is short)

| Tier | Contents |
|---|---|
| **Must** | NQ, HotpotQA, MuSiQue, MultiHop-RAG; V0–V4; LightRAG naive/local/global/hybrid/mix; R0, R1, R2, R4 + oracle; FAISS+NetworkX and Qdrant+Neo4j; reference configuration; judge validation; core figures |
| **Should** | GraphRAG-Bench; R3; E5 embedder/reranker/generator; E6; ablations A1 and A3 |
| **Could** | Microsoft GraphRAG and HippoRAG 2 references; Milvus, pgvector, Memgraph, Postgres graph; LlamaIndex fallback (unless triggered); community layer; UltraDomain pairwise; VectorDBBench extras; extractor comparison with Sonnet 5.5 |

## 6. Timeline and gates (compressed: about 20 days, decided 2026-11-24)

The week numbers in §4 come from the original 10-week design. This compressed schedule replaces them.

| Days | Engineering | Long-running jobs | Gate |
|---|---|---|---|
| 1 | P0 foundations, P1 data layer | dataset downloads | model IDs frozen |
| 2 | P2 harness + V0 | V0 on dev | **G1** |
| 3 | P3 vector baseline | embeddings; V1–V4 on dev; judge labelling | |
| 4 | **Waterloo server setup** (vLLM + DBs); P4 graph adapter | 5% extraction sample, then cost approval | **G2** |
| 5 | P4 graph quality; P5 router | full graph indexing (background, several hours) | |
| 6 | P6 benchmarks; P7 experiment planner | E1/E2 test runs | **G3** |
| 7 | P8 analysis tooling | E3 router runs | |
| 8–14 | fixes only | E4–E6, ablations, reference systems as budget allows | **G4** (about day 14) |
| 15–20 | — | — | report writing, figures, buffer |

## 7. Engineering standards

- Type hints everywhere and `mypy` on `src/`; small modules that each do one job; functions with docstrings
  that state units (tokens, ms, USD).
- No hidden globals: everything comes from config objects passed in explicitly. No secrets in code; `.env` only.
- Every paid call goes through the gateway (lint rule: `anthropic`/`openai` are imported only in `llm/providers/`).
- Pure functions for metrics and statistics, unit-tested with hand-computed values.
- Every run writes its resolved config, git SHA, model IDs and timestamps, so each result can be reproduced from `runs/` and the cache.
- Pull requests are small and one person reviews each; `make check` must pass before merging.

## 8. Inputs needed from you before each phase

| Before | You provide | Guide |
|---|---|---|
| Phase 0 | uv, git, GitHub repo (optional), Docker runtime | [01_local_environment.md](docs/setup/01_local_environment.md) |
| Phase 0 | API keys and a budget cap; agree the frozen model list | [02_accounts_api_keys_and_models.md](docs/setup/02_accounts_api_keys_and_models.md) |
| Phase 1 | Manual dataset downloads (MuSiQue, RAGRouter-Bench if not on Hugging Face) | [03_datasets.md](docs/setup/03_datasets.md) |
| Phase 3 | Two people for the 100-answer judge labelling | Phase 3 above |
| Phase 4 | Approval of the extraction cost estimate; two people for the 50-triple audit | Phase 4 above |
| Phase 4/7 | Compute for open-weight models | [04_compute_for_open_models.md](docs/setup/04_compute_for_open_models.md) |
| Each gate | A 15-minute team review to sign off | §6 |
