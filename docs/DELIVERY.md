# Elevate-RAG local delivery

Raw evidence paths below refer to local workspace files. See the [checkpoint evidence](evidence/README.md) for the published summaries.

The [Phase 2 delivery report](FINAL_DELIVERY.md) records the subsequent audit and the latest verification.

The repository contains an offline-verified hybrid retrieval and experimentation platform on the assigned `atlas` worktree. The research study remains open. There were no Git commits or pushes during implementation, and no paid model calls were made.

## Architecture and repository structure

| Directory | Contents |
| --- | --- |
| [src/elevate_rag](../src/elevate_rag) | Normalization, shared chunking, vector and graph retrieval, model providers, context, citations, routing, statistics, experiment execution and CLI. |
| [configs](../configs) | Offline configuration, reference candidate, and suite example. |
| [tests](../tests) | Data and metric unit tests, mocked HTTP contracts, actual local library integration and end-to-end experiments. |
| [infrastructure/docker](../infrastructure/docker) | Isolated database profiles and pgvector setup. |
| [docs](.) | Requirements, architecture, protocol, traceability, known issues and final research report outline. |
| `artifacts/`, `data/`, `results/` | Ignored local cache, indexes, upstream data, normalized corpora, run checkpoints, CSV and JSONL outputs. |

Offline ingestion creates one shared chunk catalog. Lexical and dense retrieval resolve to those IDs. LightRAG reads the same physical passage ANN index as the dense baselines and translates its native IDs back to original corpus offsets. The reranker and token-budgeted generator use the same evidence contract. V0 stays closed book and V1–V3 bypass reranking. Graph profiles influence candidate selection. The generator receives attributable passages with a common prompt. The judge and offline oracle use reference data only after generation. [Architecture](ARCHITECTURE.md), [implementation decisions](DECISIONS.md)

## Local data verification

| Dataset | Documents | Development | Held out | Verification |
| --- | ---: | ---: | ---: | --- |
| Synthetic | 20 | 4 | 16 | Complete offline pipeline run on all 20 questions |
| Natural Questions | 18,647 | 100 | 400 | Annotated development sample with answer candidates and distractors |
| MultiHop-RAG | 609 | 511 | 2,045 | Real corpus and query release checksummed and parsed |
| HotpotQA distractor | 9,787 | 200 | 800 | Author-provided Parquet validation file filtered to 1,000 hard questions |
| MuSiQue-Full | 7,982 | 120 | 480 | 500 answerable and 100 unanswerable examples with distinct source IDs |
| GraphRAG-Bench novel | 20 | 402 | 1,608 | Full published novel inputs and evidence ingested |
| GraphRAG-Bench medical | 1 | 412 | 1,650 | Full published medical inputs and evidence ingested |
| RAGRouter-Bench | 21,460 | 1,545 | 6,182 | All 7,727 queries across the four published domains |

Real input and output checksums are in the respective `data/<dataset>/manifest.json`. The GraphRAG-Bench scorer has access to the preserved evidence text and triples. Literal span coverage remains missing where evidence is paraphrased. The RAGRouter-Bench classification checks use all 7,727 benchmark queries with the grouped five-fold and leave-one-domain-out protocols. The main-corpus check uses 1,745 development queries across six datasets and the shared route labels. These classifier outcomes measure label prediction. Five-fold (local evidence, `artifacts/completion/ragrouter-bench-five-fold-grouped.json`), domain holdout (local evidence, `artifacts/completion/ragrouter-bench-domain-holdout.json`), dataset holdout (local evidence, `artifacts/completion/router-main-dataset-holdout.json`)

## Supported paths

- Vector V0–V4 with BM25, dense retrieval, RRF and cross-encoder reranking.
- LightRAG naive, local, global, hybrid and mix. Optional one/two-hop settings, embedding-based entity merge and Leiden reports.
- R0 static, R1 rules, R2 TF-IDF/SVM with query-type and outcome labels, R3 structured LLM classification, and R4 at most one fallback. Independent offline oracle, calibration, correctness, regret and cost reporting.
- Independent vector adapters for Qdrant, Milvus, PostgreSQL/pgvector and FAISS. LightRAG graph adapters for Neo4j, Memgraph, PostgreSQL graph tables and NetworkX.
- Provider roles for embedding, reranking, extraction, keywords, generation, router and judge. The providers support local Sentence Transformers, compatible chat/embedding endpoints, Jina rerank and explicit synthetic fixtures.
- Immutable experiment input manifests, SQLite recovery, aligned metrics, run summaries, paired statistics, and local audit and plot outputs.

## Environment and execution

Verified on Apple M4 arm64, 16 GB RAM, macOS 26.6.2, Python 3.12.13, and uv 0.11.8. The offline stack uses NumPy 2.4.6, LightRAG 1.5.8, FAISS 1.16.0, NetworkX 3.6.1, scikit-learn 1.9.1, and httpx 0.28.1. Database and scoring extras use the matching versions in [uv.lock](../uv.lock). The full model server and external framework runtime environments have separate prerequisites.

```sh
uv sync --locked --extra graph --extra router --extra faiss
uv run --no-sync elevate-rag prepare-fixture data/synthetic
uv run --no-sync elevate-rag run data/synthetic --split all --mode V4 --run-id local-demo
make check
```

## Verification evidence

- Clean locked installation in a separate local virtual environment and a 20-question V4 smoke run with zero execution errors. Install log (local evidence, `artifacts/delivery/clean-install.log`), smoke output (local evidence, `artifacts/delivery/clean-smoke.json`)
- Each V0–V4 synthetic arm completed 20 questions. The V4 and R4 runs each answered or refused all 20 fixture questions as expected. R4 escalated three questions once. The graph held 40 nodes and 20 edges with mapped passage provenance. V4 run (local evidence, `results/synthetic/verified-v4/report.md`), R4 run (local evidence, `results/synthetic/verified-r4/report.md`), graph inspection (local evidence, `artifacts/delivery/audit/graph-quality.json`)
- FAISS microbenchmarks produced measured recall, throughput, latency and searchable-write results on four synthetic queries. NetworkX graph checks produced lookup, one/two-hop, local retrieval and isolated bulk-load measurements. These artifacts are software smokes. Vector results (local evidence, `artifacts/delivery/vector-benchmark/vector-benchmark.json`), graph results (local evidence, `artifacts/delivery/graph-benchmark/graph-benchmark.json`), figures (local evidence, `artifacts/delivery/plots`)
- RAGAS 0.4.4 scored a fixture with mocked model responses. The LlamaIndex 0.14.26 schema adapter used the real extractor with a mocked model response. No live judge or extractor requests were made.
- The default suite passed 40 tests and skipped three opt-in service cases. Ruff, mypy and the dependency lock check passed. Core line coverage is 86.5% across `config`, `data`, `evaluation`, `experiments`, `graph`, `graph_passages`, `pipeline`, `providers`, `retrieval`, `routing`, `statistics`, `suite` and `tuning`. Package-wide line coverage is 74.6%. Tests (local evidence, `artifacts/completion/tests.txt`), coverage (local evidence, `artifacts/completion/coverage.json`)
- FAISS physical sharing was verified by reusing a single persisted index file through vector and graph construction and checking both readers against the same vector records. The graph pilot kept its own index, and full R4 completed 20 fixture questions without execution errors. Pilot (local evidence, `artifacts/completion/pilot.json`), full graph (local evidence, `artifacts/completion/graph.json`), R4 report (local evidence, `results/synthetic/shared-index-r4/report.md`)
- The fixture tuning pass froze a configuration on annotated development questions. The frozen configuration then completed all 16 held-out fixture questions without execution errors. Tuning report (local evidence, `artifacts/completion/v4-tuning/tuning.json`), frozen fixture run (local evidence, `results/synthetic/frozen-fixture-v4/report.md`)
- The native LlamaIndex 0.14.26 and Neo4j GraphRAG 1.24.0 subprocesses each processed two questions through a loopback fixture model service. All four launchers share an immutable execution manifest and isolated local endpoints. The Langfuse SDK produced in-memory stage traces with metadata and redacted failures. LlamaIndex run (local evidence, `artifacts/completion/native-llamaindex/output/report.md`), Neo4j GraphRAG run (local evidence, `artifacts/completion/native-neo4j-apoc/output/report.md`)
- Synthetic integration checks passed against live local Qdrant 1.16.3, Milvus 2.6.11, Neo4j 5.26.0, Memgraph 3.7.0 and PostgreSQL 16 with both pgvector 0.8.1 and Apache AGE 1.6.0. The Qdrant check inspected the physical passage collection and HNSW configuration. Qdrant (local evidence, `artifacts/completion/live-qdrant.txt`), Milvus (local evidence, `artifacts/completion/live-milvus.txt`), Neo4j (local evidence, `artifacts/completion/live-qdrant-neo4j.txt`), Memgraph (local evidence, `artifacts/completion/live-memgraph.txt`), PostgreSQL (local evidence, `artifacts/completion/live-postgres.txt`)

## Experiment status

| Gate or block | State |
| --- | --- |
| G1 | Offline 20-question integration passed. All six main benchmark corpora and RAGRouter-Bench are normalized. Reference-model QA remains pending. |
| G2 / E1 | V0–V4 software paths and the plan for all 30 baseline arms are verified. Real V4 development tuning and freeze require the candidate model backends. The offline preflight could not find pinned model files. Dry run (local evidence, `artifacts/completion/plans/e1-dev.json`), preflight (local evidence, `artifacts/completion/reference-tuning-preflight.log`) |
| G3 / E2 | Persisted LightRAG modes passed on synthetic data. Real coverage, the 50-triple double annotation and full native framework comparisons remain open. |
| E3 | Routing and oracle code passed on fixtures. Both RAGRouter-Bench validations and the main dataset classification checks ran. Routed answer comparisons and outcome-label validation require reference runs. |
| E4 | All specified vector and graph stores passed fixture checks. Full controlled measurements remain pending. |
| E5–E6 | Suite validation and factor-oriented reports are available. Real factor and interaction runs remain open. |
| G4 / RQ1–RQ6 | Held-out experiments were not run. No research results were frozen and no model, graph or router advantage is claimed. |

Check [traceability](REQUIREMENTS_TRACEABILITY.md) for implementation status by requirement and [known issues](KNOWN_ISSUES.md) for unverified paths. The reference study requires the approved model infrastructure and spending cap, human graph and judge annotation, and execution of the held-out blocks after development freeze.
