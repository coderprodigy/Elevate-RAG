# Implementation Plan: Hybrid Vector-Graph RAG (Elevate-RAG)

## Overview

The plan implements Elevate-RAG in Python 3.11+ following the design's phases 0 to 7, then the end-to-end command. Each phase builds on the previous one, runs offline on the bundled synthetic Tiny_Sample with deterministic fakes (FakeLLMGateway, FakeEmbedder, FakeReranker, FAISS/NanoVectorDB, NetworkX), and ends with a checkpoint that runs `elevate exit-check --phase N` and writes `docs/phase-reports/phase-N.md`. Property-based tests use Hypothesis with at least 100 examples and carry the tag `Feature: hybrid-vector-graph-rag, Property {number}: {property_text}`.

Every exit-check lives in its own module `src/elevate/exitchecks/phase_N.py` and registers itself with the exit-check framework, so phases never edit each other's check code.

## Tasks

### Phase 0: Repository, configuration, gateway, setup docs

- [x] 1. Scaffold the Python project and tooling
  - Create `pyproject.toml` (package `elevate`, `src/` layout, console script `elevate`), pinned dependencies and a lockfile (`uv.lock` or `requirements.lock`)
  - Create the package directories from the design's repository layout with empty `__init__.py` files
  - Add `src/elevate/cli.py` with a Typer/argparse root command and subcommand stubs: `verify-setup`, `exit-check`, `run`, `experiment`, `e2e`
  - Configure ruff (lint), mypy (strict on public functions), pytest and Hypothesis; add `make lint`, `make typecheck`, `make test` (or equivalent scripts) that run offline
  - Create `.env.example` listing every supported environment variable with placeholder values
  - _Requirements: 1.1, 1.5, 1.8, 37.1_

- [x] 2. Implement core utilities and configuration
  - [x] 2.1 Implement error taxonomy, IDs and token counting
    - Create `core/errors.py` with `ElevateError`, `ConfigError`, `BackendError`, `ModelCallError`, `BudgetExceeded`, `DatasetError`, `EmbeddingMismatchError`, `TuningOnTestError`
    - Create `core/tokens.py` (tiktoken-based counter with a deterministic fallback) and `core/ids.py` (content-hash IDs)
    - _Requirements: 1.3, 2.6, 3.5, 4.4, 6.9, 7.3, 11.2_

  - [x] 2.2 Implement Config_Loader with typed schema and secrets
    - Create `config/schema.py` with pydantic models (`extra="forbid"`) for experiment, backend, model-role, threshold and cap settings, using the design's default thresholds
    - Create `config/loader.py` that raises `ConfigError` naming file, field and expected type; loads secrets from env/`.env` as `SecretStr`; rejects judge and generator in the same model family
    - Create `configs/defaults.yaml`
    - _Requirements: 1.2, 1.3, 1.4, 3.8_

  - [x] 2.3 Implement seeding and Run_Record lifecycle
    - Create `core/seeding.py` (`set_global_seeds` for Python, NumPy, torch if installed, returning `SeedRecord`)
    - Create `core/run_record.py` with `RunItem`, `RunRecord`, `open_run` capturing config snapshot (secrets redacted), git commit hash, library versions, start time and seeds
    - _Requirements: 1.6, 1.7, 1.4_

  - [ ]* 2.4 Write unit tests for configuration and run records
    - Test unknown/missing/wrong-type field errors, judge-family rejection, secret redaction in snapshots and error messages, seed recording
    - _Requirements: 1.2, 1.3, 1.4, 1.6, 1.7, 3.8_

- [x] 3. Implement Model_Registry, Cache, Budget_Tracker, Tracer and LLM_Gateway
  - [x] 3.1 Implement Model_Registry
    - Create `gateway/registry.py` and `configs/models.yaml` with provider, model ID, version, context length and input/output price per million tokens for embedder, reranker, extraction, generator and judge roles
    - _Requirements: 3.1_

  - [x] 3.2 Implement content-hash Cache
    - Create `gateway/cache.py` with a SQLite or disk-backed store; key = hash(input content, model ID, parameters); `get_or_call`
    - _Requirements: 4.1, 4.2_

  - [ ]* 3.3 Write property test for cache idempotence
    - **Property 1: Cache idempotence**
    - Use a call-counting fake provider; issue the same request twice and assert one provider call and identical results
    - **Validates: Requirements 4.1, 4.2**

  - [x] 3.4 Implement Budget_Tracker and Tracer
    - Create `gateway/budget.py`: per-Run and per-project accumulation of uncached cost, `BudgetExceeded` at the Spending_Cap with spent amount and remaining work, 5% sample extrapolation `estimate`
    - Create `gateway/tracer.py`: Langfuse tracer when credentials exist, local JSONL tracer otherwise
    - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7_

  - [x] 3.5 Implement LLM_Gateway and providers
    - Create `gateway/gateway.py` routing every call through the registry role; temperature 0; seed passthrough where supported; exponential backoff retry (5 attempts, base 2s) then `ModelCallError`; one JSON repair pass and one re-request then mark malformed; log model ID, call date, tokens, latency, cost; charge budget only on uncached calls; emit trace
    - Create `gateway/providers/` with OpenAI-compatible, Anthropic and local (Ollama/vLLM) adapters plus `FakeLLMGateway`/`FakeProvider` with scripted and record-replay responses
    - Support separate extraction and generator settings within one Run
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.9, 4.3, 37.3_

  - [ ]* 3.6 Write unit tests for the gateway
    - Test retry and final error message, JSON repair then re-request then malformed, temperature/seed passthrough, budget stop persisting results, local JSONL tracing
    - _Requirements: 3.3, 3.4, 3.5, 3.6, 3.9, 4.4, 4.7_

- [x] 4. Implement store protocols and no-setup backends
  - [x] 4.1 Implement Vector_Store protocol with FAISS and NanoVectorDB backends
    - Create `stores/vector/base.py` (insert, delete, top-k search, filtered search by `source`, `index_params`, `health_check`) and `faiss_store.py`, `nano_store.py`
    - Default to FAISS/NanoVectorDB when no backend is configured; `health_check` reports name, version, status and raises `BackendError` referencing Setup_Docs
    - _Requirements: 2.1, 2.3, 2.5, 2.6_

  - [x] 4.2 Implement Graph_Store protocol with NetworkX backend
    - Create `stores/graph/base.py` (node upsert, edge upsert, entity lookup by name, 1-hop and 2-hop neighbours, `health_check`) and `networkx_store.py`
    - Add an `InMemoryGraphStore` dict-adjacency reference implementation for tests
    - _Requirements: 2.2, 2.3, 2.5_

  - [x] 4.3 Add Docker Compose stacks
    - Create `docker/qdrant.yml`, `milvus.yml`, `pgvector.yml`, `neo4j.yml`, `memgraph.yml`, `postgres-graph.yml` with pinned image versions and explicit CPU/memory limits
    - _Requirements: 2.4, 26.3_

  - [ ]* 4.4 Write unit tests for embedded stores
    - Test insert/delete/search/filter for FAISS and NanoVectorDB; upsert/lookup/1-hop/2-hop for NetworkX; health-check failure message
    - _Requirements: 2.1, 2.2, 2.5, 2.6_

- [ ] 5. Implement setup docs, Tiny_Sample fixture, verify-setup and exit-check framework
  - [x] 5.1 Write Setup_Docs
    - Create `docs/SETUP.md` (software versions, hardware assumptions, install commands, API keys with sign-up location, env var and phases needing them, per-phase required/optional prerequisites)
    - Create `docs/DATASETS.md` (NQ, HotpotQA distractor, MuSiQue, MultiHop-RAG, GraphRAG-Bench, RAGRouter-Bench, optional UltraDomain: source, licence, file layout, disk size, verify command)
    - Create `docs/MODELS.md` (every registry entry, obtaining/serving local open-weight models, substituting unavailable models)
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 Create the bundled synthetic Tiny_Sample fixture
    - Create `fixtures/tiny_sample/` with a synthetic corpus and up to 20 questions covering every Query_Type, including unanswerable questions, gold passage IDs, and scripted fake LLM responses for extraction, generation and judging
    - _Requirements: 5.7, 7.4_

  - [ ] 5.3 Implement verify-setup and the exit-check framework
    - Create `exitchecks/framework.py`: check registry, `CheckResult` (pass/fail/skipped with prerequisite name and Setup_Docs reference), summary printing, exit code 0 on pass and non-zero on fail
    - Wire `elevate exit-check --phase N` and `elevate verify-setup` in `cli.py`; verify-setup checks Python version, dependencies, env vars, dataset files, local models and Docker, one line per check
    - _Requirements: 5.6, 36.2, 36.3, 36.4, 36.5_

  - [ ] 5.4 Implement the Phase 0 exit-check
    - Create `exitchecks/phase_0.py`: load config, open a Run with seeds, call FakeLLMGateway twice to confirm caching, check budget/tracer output, health-check FAISS and NetworkX
    - _Requirements: 36.2, 1.6, 4.2, 2.3_

  - [ ]* 5.5 Write unit tests for the exit-check framework
    - Test exit codes on pass and injected failure, skipped output naming the prerequisite
    - _Requirements: 36.3, 36.4, 36.5_

- [ ] 6. Checkpoint - Phase 0 exit-check and report
  - Run `elevate exit-check --phase 0` offline on the Tiny_Sample, plus lint, type-check and unit tests
  - Write `docs/phase-reports/phase-0.md` with implemented components, exit-check command and observed output, test results, deviations, known limitations, and items the user must address (API keys, Docker, datasets)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 1: Data, splits, chunking, evaluation harness (gate G1)

- [ ] 7. Implement data models and dataset loaders
  - [ ] 7.1 Implement data models and JSONL IO
    - Create `data/models.py` with `Document`, `Question`, `QueryType` and `data/jsonl.py` read/write helpers
    - _Requirements: 6.1, 6.10_

  - [ ]* 7.2 Write property test for corpus round-trip
    - **Property 2: Corpus serialisation round-trip**
    - **Validates: Requirements 6.10**

  - [ ] 7.3 Implement Dataset_Loaders
    - Create `data/loaders/` for NQ (500 questions, gold + distractors), HotpotQA (1,000 hard distractor dev), MuSiQue (500 answerable + 100 unanswerable), MultiHop-RAG (full set incl. null), GraphRAG-Bench (novel + medical, levels 1–4) and RAGRouter-Bench
    - Map source labels to Query_Type; preserve gold passage IDs; seeded selection; `DatasetError` naming file, missing field and DATASETS.md section; enforce gold-passage referential integrity at write time
    - Add `elevate data load <dataset>` and file verification commands referenced in DATASETS.md
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.11, 6.12, 5.3_

  - [ ]* 7.4 Write property test for gold-passage referential integrity
    - **Property 3: Gold-passage referential integrity**
    - Generate synthetic raw dataset records; assert every gold ID exists in the corpus
    - **Validates: Requirements 6.11**

  - [ ]* 7.5 Write property test for dataset load determinism
    - **Property 4: Dataset load determinism**
    - **Validates: Requirements 6.12**

  - [ ]* 7.6 Write unit tests for loaders
    - Use small raw-format fixtures per dataset; test selection counts, Query_Type mapping, `answerable` flag and schema error messages
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_

- [ ] 8. Implement splits and Tiny_Sample selection
  - [ ] 8.1 Implement Splitter
    - Create `data/splits.py`: 20/80 dev/test stratified by Query_Type with fixed seed; `tiny_sample` of at most 20 dev questions with gold and distractor documents; `guard_tuning(split)` raising `TuningOnTestError`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 8.2 Write property test for split partition
    - **Property 5: Split partition**
    - **Validates: Requirements 7.1, 7.2**

- [ ] 9. Implement chunking and the shared Chunk_Store
  - [ ] 9.1 Implement Chunker
    - Create `chunking/chunker.py`: sentence-boundary chunking, default 512 tokens with 64 overlap, alternatives 256, 1,024 and paragraph-level; deterministic content-hash `chunk_id`; character offsets; gold passage → chunk ID mapping
    - _Requirements: 8.1, 8.2, 8.3, 8.7_

  - [ ] 9.2 Implement the LightRAG bridge and Chunk_Store adapter
    - Create `lightrag_bridge/chunking.py` exposing `Chunker.as_lightrag_func()` for LightRAG's `chunking_func`
    - Create `stores/chunk_store.py`: read adapter over LightRAG's chunk KV store, with an in-memory implementation for tests
    - _Requirements: 8.3, 12.8_

  - [ ]* 9.3 Write property test for chunk size
    - **Property 6: Chunk size invariant**
    - **Validates: Requirements 8.4**

  - [ ]* 9.4 Write property test for chunk coverage
    - **Property 7: Chunk coverage**
    - **Validates: Requirements 8.5**

  - [ ]* 9.5 Write property test for chunk determinism
    - **Property 8: Chunk determinism**
    - **Validates: Requirements 8.6**

- [ ] 10. Implement the Evaluation_Harness and LLM_Judge
  - [ ] 10.1 Implement answer and retrieval metrics
    - Create `eval/metrics.py`: SQuAD-style normalisation, EM, token F1, Recall@{5,10,20,50}, nDCG@10 against gold chunk IDs
    - _Requirements: 9.1, 9.2_

  - [ ]* 10.2 Write property test for normalisation idempotence
    - **Property 9: Answer normalisation idempotence**
    - **Validates: Requirements 9.3**

  - [ ]* 10.3 Write property test for metric reflexivity
    - **Property 10: Metric reflexivity**
    - **Validates: Requirements 9.4**

  - [ ] 10.4 Implement harness aggregation and Run_Record output
    - Create `eval/harness.py`: refusal accuracy and false-refusal rate, latency p50/p95, tokens per query, cost per 1,000 queries, overall and per-Query_Type breakdown, write per-query JSONL plus summary JSON
    - _Requirements: 9.8, 9.9, 9.10, 9.11_

  - [ ] 10.5 Implement RAGAS and GraphRAG-Bench evidence recall adapters
    - Create `eval/ragas_adapter.py` (faithfulness, context recall, context precision through the LLM_Gateway judge role) and `eval/evidence_recall.py`
    - _Requirements: 9.6, 9.7_

  - [ ] 10.6 Implement LLM_Judge
    - Create `eval/judge.py`: versioned grading prompt returning correct/partial/wrong with rationale; pairwise comprehensiveness/diversity/empowerment judged twice with order swapped, ties on disagreement; unjudged items excluded and counted
    - Wire judge correctness into the harness as the primary accuracy metric
    - _Requirements: 9.5, 10.1, 10.2, 10.3, 10.4_

  - [ ] 10.7 Implement Cohen's kappa and judge validation command
    - Create `stats/agreement.py` with `cohens_kappa`; add `elevate judge validate <labels.csv>` reporting kappa and pass at ≥ 0.6
    - _Requirements: 10.5, 32.3_

  - [ ]* 10.8 Write property test for kappa agreement
    - **Property 24: Kappa agreement**
    - **Validates: Requirements 32.4**

  - [ ]* 10.9 Write unit tests for harness and judge
    - Test refusal metrics, per-type breakdown, order-swap tie logic, unjudged counting
    - _Requirements: 9.8, 9.10, 10.3, 10.4_

  - [ ] 10.10 Implement the Phase 1 exit-check (gate G1)
    - Create `exitchecks/phase_1.py`: load, split, chunk the Tiny_Sample; run a closed-book and a trivial retrieval pipeline on 20 questions end to end with fakes; print the metrics summary
    - _Requirements: 10.6, 36.2_

- [ ] 11. Checkpoint - Phase 1 exit-check and report
  - Run `elevate exit-check --phase 1` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-1.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as downloading datasets and providing 100 hand-labelled judge answers)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 2: Vector baseline V0–V4 (gate G2)

- [ ] 12. Implement vector indexing
  - [ ] 12.1 Implement Embedders
    - Create `vector/embedder.py`: `FakeEmbedder` (hash-based) and `SentenceTransformerEmbedder`, both via LLM_Gateway caching; store model name and version with each vector; `EmbeddingMismatchError` naming both IDs when query and index models differ
    - _Requirements: 11.1, 11.2_

  - [ ] 12.2 Implement HNSW parameters and BM25_Index
    - Extend `stores/vector/faiss_store.py` with configured M, ef_construction, ef_search recorded via `index_params` into the Run_Record
    - Create `stores/bm25.py` indexing exactly the Chunk_Store chunk IDs; add `check_shared_chunks()` asserting Vector_Store, BM25_Index and Chunk_Store ID sets are equal
    - _Requirements: 11.3, 11.4, 11.5_

  - [ ]* 12.3 Write unit tests for indexing
    - Test embedding mismatch error, recorded HNSW params, shared-chunk invariant on the Tiny_Sample
    - _Requirements: 11.2, 11.3, 11.5_

- [ ] 13. Implement retrieval and reranking
  - [ ] 13.1 Implement RRF fusion
    - Create `vector/rrf.py` with `rrf_fuse(rankings, k=60)` and deterministic tie-breaking by chunk ID
    - _Requirements: 12.3_

  - [ ]* 13.2 Write property test for RRF confluence
    - **Property 11: RRF confluence**
    - **Validates: Requirements 12.4**

  - [ ]* 13.3 Write property test for RRF monotonicity
    - **Property 12: RRF monotonicity**
    - **Validates: Requirements 12.5**

  - [ ] 13.4 Implement Rerankers
    - Create `vector/reranker.py`: `FakeReranker` and cross-encoder reranker returning top-n (5–10, default 8) sorted by descending score
    - _Requirements: 12.6_

  - [ ]* 13.5 Write property test for reranker subset and ordering
    - **Property 13: Reranker subset and ordering**
    - **Validates: Requirements 12.7**

  - [ ] 13.6 Implement Vector_Retriever
    - Create `vector/retriever.py`: BM25, dense and hybrid RRF modes returning top 50 chunks read from the Chunk_Store, optional rerank
    - _Requirements: 12.1, 12.2, 12.3, 12.8_

- [ ] 14. Implement context building and generation
  - [ ] 14.1 Implement Context_Builder
    - Create `generation/context.py`: numbered passages, entity blocks and relation blocks, truncated to the Context_Token_Cap (default 8,000); same reranker and cap for vector, graph and fused paths
    - _Requirements: 13.1, 13.3_

  - [ ]* 14.2 Write property test for context token budget
    - **Property 14: Context token budget**
    - **Validates: Requirements 13.2**

  - [ ] 14.3 Implement Generator
    - Create `generation/generator.py` and a versioned prompt file: cite passage numbers, answer "insufficient information" when context lacks the answer; parse citations, map to chunk IDs, record invalid citations
    - _Requirements: 13.4, 13.5, 13.6_

  - [ ]* 14.4 Write unit tests for citation parsing
    - Test valid, invalid and missing citations and refusal detection
    - _Requirements: 13.5, 13.6_

- [ ] 15. Implement vector variants and the Phase 2 exit-check
  - [ ] 15.1 Implement V0–V4 pipelines
    - Create `vector/variants.py` and `pipelines/vector_pipeline.py`: V0 closed-book, V1 BM25, V2 dense, V3 hybrid RRF, V4 hybrid + reranker; `elevate run vector --variant Vn`; contamination flag when V0 EM ≥ configured threshold (default 0.25)
    - _Requirements: 14.1, 14.2, 14.4_

  - [ ] 15.2 Implement V4 dev-only tuning (gate G2)
    - Create `vector/tuning.py`: grid search over top-k and reranker top-n on dev only (guarded by `guard_tuning`), write frozen config to `configs/frozen/v4.vN.yaml`
    - _Requirements: 14.3, 7.3_

  - [ ] 15.3 Implement the Phase 2 exit-check
    - Create `exitchecks/phase_2.py`: run V0–V4 on the Tiny_Sample with FAISS and fakes, print recall and accuracy per variant
    - _Requirements: 14.5, 36.2_

- [ ] 16. Checkpoint - Phase 2 exit-check and report
  - Run `elevate exit-check --phase 2` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-2.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as local embedder/reranker download and V4 tuning on real dev data)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 3: Graph layer, retrieval modes, quality checks (gate G3)

- [ ] 17. Implement extraction and graph construction
  - [ ] 17.1 Implement Extractor
    - Create `graph/extractor.py`: entities (name, type, description) and relations (source, relation, target, description, weight) with source `chunk_id`; cache by chunk hash, prompt version and model ID; record malformed chunks as failed and continue; report tokens, wall-clock time and cost
    - _Requirements: 15.1, 15.2, 15.4, 15.5, 15.6_

  - [ ] 17.2 Implement Entity_Resolver
    - Create `graph/resolver.py`: merge by normalised name, then by profile-embedding cosine above Merge_Threshold (default 0.88) using deterministic union-find
    - _Requirements: 16.1_

  - [ ]* 17.3 Write property test for resolution idempotence
    - **Property 16: Entity-resolution idempotence**
    - **Validates: Requirements 16.2**

  - [ ]* 17.4 Write property test for resolution reduction
    - **Property 17: Entity-resolution reduction**
    - **Validates: Requirements 16.3**

  - [ ] 17.5 Implement Graph_Builder and LightRAG construction
    - Create `lightrag_bridge/factory.py` constructing a LightRAG instance with our chunking_func, role-routed `llm_model_func`/`embedding_func` and configured storages
    - Create `graph/builder.py`: extraction → resolution → summarisation of descriptions above the merge limit (default 8) → embed entity profiles and relation descriptions into the Vector_Store → write nodes/edges to the Graph_Store
    - _Requirements: 16.4, 16.5, 3.7_

  - [ ]* 17.6 Write property test for graph provenance
    - **Property 15: Graph provenance**
    - **Validates: Requirements 15.3**

  - [ ] 17.7 Implement Community_Builder
    - Create `graph/community.py`: optional Leiden detection (graspologic or igraph) and one LLM report per community
    - _Requirements: 16.6_

  - [ ]* 17.8 Write unit tests for extraction failures and summarisation
    - Test failed-chunk counting, cache hits on re-extraction, summarisation trigger
    - _Requirements: 15.4, 15.5, 16.4_

- [ ] 18. Implement graph retrieval modes
  - [ ] 18.1 Implement Graph_Retriever
    - Create `graph/retriever.py` calling LightRAG `aquery_data` for naive, local, global, hybrid and mix; map to QueryParam; depth-2 neighbours in local mode; mix passes combined results through the Reranker; record chunk, entity and relation IDs; feed our Context_Builder and Generator
    - Add `elevate run graph --mode <mode>`
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 17.7, 13.3_

  - [ ]* 18.2 Write unit tests for graph retrieval
    - Use a fake LightRAG query_data response; test mode mapping, depth-2 inclusion and recorded IDs
    - _Requirements: 17.4, 17.6, 17.7_

- [ ] 19. Implement graph quality checks
  - [ ] 19.1 Implement Graph_Quality_Checker
    - Create `graph/quality.py`: Answer_Entity_Coverage, node/edge counts, average degree, clustering coefficient, duplicate-entity rate; mark G3 failure below 60% and print the fallback rebuild command
    - _Requirements: 18.1, 18.2, 18.3_

  - [ ] 19.2 Implement Fallback_Extractor
    - Create `graph/fallback_extractor.py` wrapping LlamaIndex SchemaLLMPathExtractor with a fixed schema; selectable in Graph_Builder config
    - _Requirements: 18.4_

  - [ ] 19.3 Implement triple audit export
    - Create `analysis/annotation.py` with CSV `export` (item IDs, content, per-annotator label columns, guideline reference) and `export_triple_audit` sampling 50 triples with source chunk text using a fixed seed
    - _Requirements: 18.5, 32.1_

- [ ] 20. Implement reference and alternative system adapters and the Phase 3 exit-check
  - [ ] 20.1 Implement SystemAdapter and reference systems
    - Create `lightrag_bridge/adapters/base.py` (`SystemAdapter.run(question) -> RunItem`) and adapters for Microsoft GraphRAG and HippoRAG 2, enabled only when installed/configured
    - _Requirements: 19.1, 19.3_

  - [ ] 20.2 Implement alternative framework adapters
    - Create adapters for LlamaIndex PropertyGraphIndex and the Neo4j GraphRAG package over the shared Chunk_Store, scored by the same harness
    - _Requirements: 19.2, 19.3_

  - [ ] 20.3 Implement the Phase 3 exit-check
    - Create `exitchecks/phase_3.py`: build a graph on the Tiny_Sample with NetworkX and fake extraction, run all five modes, print accuracy, recall and context tokens per mode plus graph quality stats
    - _Requirements: 17.8, 18.1, 36.2_

- [ ] 21. Checkpoint - Phase 3 exit-check and report
  - Run `elevate exit-check --phase 3` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-3.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as extraction cost estimate approval, triple audit annotation and optional reference systems)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 4: Router levels R0–R4 and oracle

- [ ] 22. Implement routers
  - [ ] 22.1 Implement Router protocol, R0 and R1
    - Create `router/base.py` (`RouteDecision` with mode, level, reason), `router/static.py` (R0) and `router/rules.py` (R1) reading `configs/routers/r1_rules.yaml` (keywords, named-entity count, question word) with a default mode guaranteeing totality
    - _Requirements: 20.1, 20.2, 20.3, 20.4_

  - [ ]* 22.2 Write property test for router totality
    - **Property 18: Router totality**
    - Cover R0, R1, R2 (fitted on synthetic data) and R3 with a fake LLM
    - **Validates: Requirements 20.3**

  - [ ] 22.3 Implement R2 classifier router
    - Create `router/classifier.py`: TF-IDF + SVM (optional small encoder), training on query-type or outcome labels, leave-one-dataset-out and 5-fold CV utilities, `save`/`load` via joblib
    - _Requirements: 21.1, 21.2, 21.3, 21.4_

  - [ ]* 22.4 Write property test for R2 save/load round-trip
    - **Property 19: R2 model save/load round-trip**
    - **Validates: Requirements 21.6**

  - [ ] 22.5 Implement R3 LLM classifier router
    - Create `router/llm.py`: zero-/few-shot classification into fact, multi-hop, comparison, global through LLM_Gateway, mapped to a mode, with fallback mode on unparseable output
    - _Requirements: 21.5_

  - [ ] 22.6 Implement R4 confidence fallback router
    - Create `router/fallback.py`: run cheap mode, escalate once to mix when top reranker score is below the dev-tuned threshold or answer equals the Refusal_String; record tokens and cost of both attempts; add dev-only threshold tuning
    - _Requirements: 22.1, 22.2, 22.3, 22.4, 22.5, 7.3_

  - [ ]* 22.7 Write unit tests for R1 and R4
    - Test rule matches, escalation triggers, at most one escalation, recorded attempt costs
    - _Requirements: 20.2, 22.2, 22.3, 22.4, 22.5_

- [ ] 23. Implement oracle and router evaluation
  - [ ] 23.1 Implement Oracle_Router
    - Create `router/oracle.py`: cheapest judge-correct mode per question; cheapest mode and `unsolvable` flag when none is correct
    - _Requirements: 23.1, 23.2_

  - [ ] 23.2 Implement router metrics and cached replay
    - Create `eval/router_metrics.py`: routing accuracy, macro-F1, end-to-end accuracy, regret, cost saved vs always-mix; compute router results from cached per-mode Run_Records without new LLM calls
    - Add `elevate run router --level Rn`
    - _Requirements: 23.3, 23.5_

  - [ ]* 23.3 Write property test for oracle regret bound
    - **Property 20: Oracle regret bound**
    - **Validates: Requirements 23.4**

  - [ ] 23.4 Implement the Phase 4 exit-check
    - Create `exitchecks/phase_4.py`: run R0–R4 and the oracle on the Tiny_Sample, print accuracy, regret and cost per Router_Level
    - _Requirements: 23.6, 36.2_

- [ ] 24. Checkpoint - Phase 4 exit-check and report
  - Run `elevate exit-check --phase 4` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-4.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as RAGRouter-Bench download and R4 threshold tuning on real dev data)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 5: Vector and graph database comparison

- [ ] 25. Implement Docker-backed stores
  - [ ] 25.1 Implement Qdrant, Milvus and pgvector Vector_Stores
    - Create `stores/vector/qdrant_store.py`, `milvus_store.py`, `pgvector_store.py` with health checks and HNSW params; register with LightRAG storage selection
    - _Requirements: 2.1, 2.5, 2.6, 11.3_

  - [ ] 25.2 Implement Neo4j, Memgraph and Postgres graph Graph_Stores
    - Create `stores/graph/neo4j_store.py`, `memgraph_store.py`, `postgres_store.py` with health checks
    - _Requirements: 2.2, 2.5, 2.6_

  - [ ]* 25.3 Write gated integration tests for Docker backends
    - Skip when the backend health check fails; test the full protocol surface
    - _Requirements: 37.4_

  - [ ]* 25.4 Write property test for graph backend equivalence
    - **Property 21: Graph backend equivalence**
    - Offline: compare NetworkxGraphStore with InMemoryGraphStore on generated graphs; additionally run across any healthy Docker graph backends
    - **Validates: Requirements 25.4**

- [ ] 26. Implement DB_Benchmark
  - [ ] 26.1 Implement Level A vector benchmark
    - Create `experiments/db_bench/vector_bench.py`: tune ef_search/nprobe to recall@10 ≥ 0.95 vs exact FAISS; QPS at 1/8/32 clients; p50/p95/p99; build time, peak RAM, disk, filtered search, write-to-searchable delay; recall-vs-QPS curves; 1,000-query warm-up, 3 repeats, median
    - _Requirements: 24.1, 24.2, 24.3, 24.4, 24.5_

  - [ ] 26.2 Implement Level A graph benchmark
    - Create `experiments/db_bench/graph_bench.py`: bulk load time and memory, 1-hop/2-hop latency for 1,000 seeded random entities, lookup by name, LightRAG local-mode latency per backend
    - _Requirements: 25.1, 25.2, 25.3_

  - [ ] 26.3 Implement Level B comparison and fairness records
    - Create `experiments/db_bench/level_b.py`: swap LightRAG storage backend, re-run a seeded 300-question subset, report accuracy, recall, latency, config line count and user-entered setup hours; record image versions, resource limits, machine spec and HNSW params
    - _Requirements: 26.1, 26.2, 26.3, 26.4_

  - [ ] 26.4 Implement the Phase 5 exit-check
    - Create `exitchecks/phase_5.py`: run Level A on the Tiny_Sample for every healthy backend (FAISS, NanoVectorDB, NetworkX always), mark others skipped, print the result table
    - _Requirements: 26.5, 36.2, 36.5_

- [ ] 27. Checkpoint - Phase 5 exit-check and report
  - Run `elevate exit-check --phase 5` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-5.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as starting Docker stacks and entering setup hours)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 6: Model comparison, ablations, experiment runner (gate G4)

- [ ] 28. Implement the Experiment_Runner
  - [ ] 28.1 Define experiment schema and matrix files
    - Create `experiments/schema.py` and `configs/experiments/E1..E6.yaml`, `A1..A4.yaml` (dataset, split, pipeline, Router_Level, backends, model roles); reference configuration with no-setup substitution; extractor comparisons restricted to MultiHop-RAG and GraphRAG-Bench medical; E6 mode × generator factorial; A1 chunk sizes, A2 merge thresholds {0.82, 0.88, 0.94}, A3 caps {4k, 8k, 16k}, A4 community on/off and depth 1/2
    - _Requirements: 29.1, 27.2, 27.3, 27.4, 28.1, 28.2, 28.3, 28.4_

  - [ ] 28.2 Implement runner execution, resume, dry run and manifest
    - Create `experiments/runner.py`: execute Runs, resume from last completed query via persisted Run_Records and Cache, dry run listing planned Runs with estimated tokens and cost and no model calls, frozen run manifest with Run_Record paths and config hashes (gate G4); wire `elevate experiment <file> [--dry-run]`
    - _Requirements: 29.2, 29.3, 29.4_

  - [ ] 28.3 Implement one-factor model comparison and ablation reporting
    - Create `experiments/comparison.py`: validate exactly one varied model role per comparison, report accuracy delta vs reference and cost per 1,000 queries, report context tokens next to accuracy for ablations
    - _Requirements: 27.1, 27.5, 28.5_

  - [ ]* 28.4 Write unit tests for the runner
    - Test resume after injected interruption, dry run making zero model calls, manifest contents, one-factor validation error
    - _Requirements: 27.1, 29.2, 29.3, 29.4_

  - [ ] 28.5 Implement the Phase 6 exit-check
    - Create `exitchecks/phase_6.py`: execute a reduced matrix (one comparison per model role, one value pair per ablation) on the Tiny_Sample and print the summary
    - _Requirements: 29.5, 36.2_

- [ ] 29. Checkpoint - Phase 6 exit-check and report
  - Run `elevate exit-check --phase 6` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-6.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as reviewing dry-run cost before full matrix and scope cuts)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### Phase 7: Statistics, analysis, annotation, reporting

- [ ] 30. Implement the Stats_Engine
  - [ ] 30.1 Implement statistical methods
    - Create `stats/engine.py`: 95% bootstrap CI (1,000 resamples, fixed seed, percentile clipped to contain the point estimate), McNemar, paired bootstrap, Holm–Bonferroni within families, minimum detectable effect from dev variance
    - _Requirements: 30.1, 30.2, 30.3, 30.4_

  - [ ]* 30.2 Write property test for bootstrap interval
    - **Property 22: Bootstrap interval invariant**
    - **Validates: Requirements 30.5**

  - [ ]* 30.3 Write property test for Holm–Bonferroni correction
    - **Property 23: Multiple-comparison correction invariant**
    - **Validates: Requirements 30.6**

- [ ] 31. Implement annotation tooling and error analysis
  - [ ] 31.1 Complete the Annotation_Tool
    - Extend `analysis/annotation.py` with `import_sheet` validating labels against allowed sets and reporting invalid item IDs, kappa between annotators and between humans and LLM_Judge (reusing `stats/agreement.py`), judge-validation export
    - Write `docs/annotation/triple-audit.md`, `judge-validation.md`, `error-tagging.md`
    - _Requirements: 32.1, 32.2, 32.3, 32.6_

  - [ ]* 31.2 Write property test for annotation sheet round-trip
    - **Property 25: Annotation sheet round-trip**
    - **Validates: Requirements 32.5**

  - [ ] 31.3 Implement Error_Analyzer
    - Create `analysis/errors.py`: seeded sample of 50 failures with question, gold, prediction, context, route and judge rationale; fixed failure-cause tags; import tagged sheets and aggregate counts and chart data per system
    - _Requirements: 31.1, 31.2, 31.3_

  - [ ]* 31.4 Write unit tests for annotation import and error aggregation
    - Test invalid-label reporting and failure-cause counts
    - _Requirements: 31.3, 32.2_

- [ ] 32. Implement the Report_Builder and the Phase 7 exit-check
  - [ ] 32.1 Implement figures, tables and cost reporting
    - Create `report/builder.py`: architecture diagram, accuracy by Query_Type and mode, router cost-vs-accuracy scatter, recall-vs-QPS curves, failure-cause chart, cost per 1,000 queries table, RQ1–RQ6 tables with 95% CIs, indexing tokens/time/storage MB/query tokens per dataset and pipeline, estimated-vs-actual budget summary; read only from the frozen manifest when it exists
    - _Requirements: 33.1, 33.2, 33.3, 34.1, 34.2_

  - [ ] 32.2 Implement appendix export and report template
    - Create `report/appendix.py` exporting prompts, hyperparameters, Docker Compose files, full result tables and annotation guidelines; add `report/template.md` with all required sections
    - _Requirements: 33.4, 33.5_

  - [ ]* 32.3 Write snapshot tests for the Report_Builder
    - Test generated tables and figure data on Tiny_Sample Run_Records and manifest-only reading
    - _Requirements: 33.2, 33.3_

  - [ ] 32.4 Implement the Phase 7 exit-check
    - Create `exitchecks/phase_7.py`: compute stats on Tiny_Sample Run_Records, export and re-import annotation and error sheets, generate report assets
    - _Requirements: 36.2, 30.1, 33.1_

- [ ] 33. Checkpoint - Phase 7 exit-check and report
  - Run `elevate exit-check --phase 7` offline on the Tiny_Sample, plus lint, type-check, unit and property tests
  - Write `docs/phase-reports/phase-7.md` (implemented components, exit-check output, test results, deviations, limitations, user items such as human annotation and report writing)
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 36.1, 36.2_

### End-to-end pipeline

- [ ] 34. Implement the end-to-end command
  - [ ] 34.1 Implement `elevate e2e`
    - Create `pipelines/e2e.py`: stages load → split → chunk → build vector and graph indexes → run V4, local, mix, R4 → evaluate → stats → report assets; `--tiny` uses no-setup backends and fakes, `--config` uses configured backends and models; persist stage outputs, stop on failure reporting the stage and error, resume from completed stages; print Run_Record location, metrics summary and asset paths
    - _Requirements: 35.1, 35.2, 35.3, 35.4_

  - [ ]* 34.2 Write integration test for `e2e --tiny`
    - Assert success offline, printed locations, and resume after an injected stage failure
    - _Requirements: 35.1, 35.3, 35.4, 37.3_

  - [ ] 34.3 Finalise README
    - Write `README.md` with architecture, directory layout, phase list and the exit-check command per phase
    - _Requirements: 37.5_

- [ ] 35. Final checkpoint - Ensure all tests pass
  - Run `elevate e2e --tiny`, all eight exit-checks, lint, type-check, unit, property and available integration tests
  - Ensure all tests pass, ask the user if questions arise.
  - _Requirements: 35.1, 36.3, 37.1, 37.2_

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; the property tests cover Properties 1–25 from the design.
- The recommended minimum viable slice (NQ + MultiHop-RAG Tiny_Sample, FAISS + NetworkX, naive/local/mix, R1, EM/F1, recall@k) is reachable after task 22.1.
- All exit-checks run offline with deterministic fakes; real models, datasets and Docker databases are opt-in through configuration, and unavailable prerequisites are reported as skipped.
- Each checkpoint writes `docs/phase-reports/phase-N.md` listing what the user must provide before running on real data.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "4.3", "5.1", "5.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "3.1", "3.2"] },
    { "id": 3, "tasks": ["2.4", "3.3", "3.4", "4.1", "4.2"] },
    { "id": 4, "tasks": ["3.5", "4.4"] },
    { "id": 5, "tasks": ["3.6", "5.3"] },
    { "id": 6, "tasks": ["5.4"] },
    { "id": 7, "tasks": ["5.5"] },
    { "id": 8, "tasks": ["6"] },
    { "id": 9, "tasks": ["7.1"] },
    { "id": 10, "tasks": ["7.2", "7.3", "8.1", "9.1", "10.1"] },
    { "id": 11, "tasks": ["7.4", "7.5", "7.6", "8.2", "9.2", "9.3", "9.4", "9.5", "10.2", "10.3", "10.4", "10.5", "10.7"] },
    { "id": 12, "tasks": ["10.6", "10.8"] },
    { "id": 13, "tasks": ["10.9", "10.10"] },
    { "id": 14, "tasks": ["11"] },
    { "id": 15, "tasks": ["12.1", "13.1", "13.4", "14.1"] },
    { "id": 16, "tasks": ["12.2", "13.2", "13.3", "13.5", "14.2", "14.3"] },
    { "id": 17, "tasks": ["12.3", "13.6", "14.4"] },
    { "id": 18, "tasks": ["15.1"] },
    { "id": 19, "tasks": ["15.2"] },
    { "id": 20, "tasks": ["15.3"] },
    { "id": 21, "tasks": ["16"] },
    { "id": 22, "tasks": ["17.1", "17.2"] },
    { "id": 23, "tasks": ["17.3", "17.4", "17.5"] },
    { "id": 24, "tasks": ["17.6", "17.7", "17.8", "18.1"] },
    { "id": 25, "tasks": ["18.2", "19.1", "19.3", "20.1"] },
    { "id": 26, "tasks": ["19.2", "20.2"] },
    { "id": 27, "tasks": ["20.3"] },
    { "id": 28, "tasks": ["21"] },
    { "id": 29, "tasks": ["22.1", "22.3", "22.5", "23.1"] },
    { "id": 30, "tasks": ["22.2", "22.4", "22.6", "23.2"] },
    { "id": 31, "tasks": ["22.7", "23.3"] },
    { "id": 32, "tasks": ["23.4"] },
    { "id": 33, "tasks": ["24"] },
    { "id": 34, "tasks": ["25.1", "25.2"] },
    { "id": 35, "tasks": ["25.3", "25.4", "26.1", "26.2"] },
    { "id": 36, "tasks": ["26.3"] },
    { "id": 37, "tasks": ["26.4"] },
    { "id": 38, "tasks": ["27"] },
    { "id": 39, "tasks": ["28.1"] },
    { "id": 40, "tasks": ["28.2"] },
    { "id": 41, "tasks": ["28.3", "28.4"] },
    { "id": 42, "tasks": ["28.5"] },
    { "id": 43, "tasks": ["29"] },
    { "id": 44, "tasks": ["30.1", "31.1", "31.3"] },
    { "id": 45, "tasks": ["30.2", "30.3", "31.2", "31.4", "32.1"] },
    { "id": 46, "tasks": ["32.2"] },
    { "id": 47, "tasks": ["32.3", "32.4"] },
    { "id": 48, "tasks": ["33"] },
    { "id": 49, "tasks": ["34.1"] },
    { "id": 50, "tasks": ["34.2", "34.3"] },
    { "id": 51, "tasks": ["35"] }
  ]
}
```
