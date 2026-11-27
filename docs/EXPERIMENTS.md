# Experimental protocol

## Environment and reference candidate

The reference candidate uses BGE-M3 1,024-dimensional embeddings, BGE reranker v2 m3, Claude Haiku 5.5 for extraction and keywords, Qwen3.6-35B-A3B for generation, an independent DeepSeek V4 Pro judge, and Qdrant/Neo4j. Model identifiers and tokenizer revisions are recorded in [the candidate configuration](../configs/reference.toml). Qwen must be served by a compatible local or approved server with the recorded weights, quantization, context length and serving revision. The candidate cap is zero. The current M4 host has 16 GB RAM and about 15 GB of free disk at the time of testing. The unquantized 35B weights exceed that memory and disk. No replacement research configuration was selected. [BGE-M3](https://huggingface.co/BAAI/bge-m3), [reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3), [Qwen](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [Claude](https://platform.claude.com/docs/en/models/overview), [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/)

```sh
uv sync --locked --extra graph --extra router --extra faiss --extra datasets \
  --extra models --extra storage --extra evaluation --extra report --extra community --extra schema
docker compose -f infrastructure/docker/compose.yaml --profile reference config
```

Set service credentials and endpoints in the environment or your runner. The application reads credential environment variables and never writes `.env`. Use a unique Compose project and storage namespace when reusing a host. The database profiles and port overrides are in [the Compose file](../infrastructure/docker/compose.yaml). LightRAG database adapters also read backend-specific environment variables. [LightRAG SDK](https://github.com/HKUDS/LightRAG)

Build the PostgreSQL profile with `docker compose -f infrastructure/docker/compose.yaml --profile postgres up --build -d postgres`. Its local image installs Apache AGE so LightRAG graph storage and pgvector can run in the same database. A fresh profile volume initializes both extensions. [Apache AGE](https://age.apache.org/age-manual/master/intro/setup.html)

The Milvus profile runs in standalone mode with embedded etcd on container loopback. The official v2.6.11 image download and extraction need several gigabytes of host disk. [Milvus standalone script](https://raw.githubusercontent.com/milvus-io/milvus/v2.6.11/scripts/standalone_embed.sh)

The shared passage index exposes HNSW parameters through project configuration. Both dense baselines and LightRAG passage retrieval use that same physical backend. LightRAG’s native entity and relationship stores retain their own profile-index settings. Record those effective settings before E4. Container resource consumption and setup effort must be collected from the actual database host. In-process FAISS and NetworkX use the process limits available on the benchmark host.

## Execution

Examples use a local research configuration whose references and budgets are already approved.

```sh
uv run --no-sync elevate-rag --config configs/reference.toml matrix E3
uv run --no-sync elevate-rag --config configs/reference.toml suite E1 configs/suite.example.toml --split dev
uv run --no-sync elevate-rag --config configs/reference.toml run data/multihop-rag --mode V4 --split dev --dry-run
uv run --no-sync elevate-rag --config configs/reference.toml index data/multihop-rag --graph --pilot
uv run --no-sync elevate-rag --config configs/reference.toml index data/multihop-rag --graph
uv run --no-sync elevate-rag --config configs/reference.toml inspect-graph data/multihop-rag artifacts/audits/multihop
uv run --no-sync elevate-rag --config configs/reference.toml run data/multihop-rag --mode V4 --split dev --run-id multihop-v4-dev
```

`suite` validates by default. Run it with `--execute` when the correct indexes and providers are present. Populate the dataset paths and model tiers in the suite input. E4 Level B uses 300 deterministic questions, cycles over 1,000 development warm-up requests, and runs three measured reproductions with caches bypassed. Benchmark Level A is available through `benchmark-vector` and `benchmark-graph`. The vector benchmark selects the first ef whose observed exact-FAISS recall@10 reaches 0.95. Ties at the exact cutoff receive equal credit. Graph Level A accepts `--bulk-load` to measure isolated persistence of the source graph snapshot. Report a missing matched point when none of the probed settings reaches the target.

If development answer-entity coverage stays below 60% after allowed tuning, install the `schema` extra and set `graph_extractor = "llamaindex-schema"` in a copy of the fixed reference configuration. Build the pilot and the new graph, then repeat the development audit. The fixed `universal-qa-v1` schema and the extraction choice are versioned independently.

Run V0 on each main benchmark and use a paired accuracy delta against V0 to report retrieval gains. Tune only on development. Keep final configuration, model snapshots, prompts and Docker settings with the run manifest. Generation is temperature zero. Determinism can still depend on provider and hardware.

## Definitions

- Recall@K is distinct annotated evidence recovered in the first K unique candidates before reranking, divided by distinct annotated evidence. The V4 tuning workflow separately scores its selected reranker depths. A span is matched by overlap. Positional nDCG@10 uses binary passage relevance and the ideal ranking up to ten relevant catalog passages. Context evidence recall and precision use the passages actually sent to generation. Missing gold evidence yields a missing value.
- Exact match and token F1 remove numbered citations, punctuation and English articles, normalize whitespace and use the best matching reference. The independent judge grades correct, partial or wrong with scores 1, 0.5 and 0. Binary correctness requires correct. Unanswerable queries use refusal accuracy. Execution failures count as wrong for binary correctness and remain visible. Other unavailable scores have their own denominator.
- RAGAS metrics are enabled using `ragas_metrics = true` and the evaluation extra. Model traffic uses the project's budgeted judge transport. Xiang benchmark ingestion and export retain evidence text for the official scripts. Report these metrics under their exact scoring protocol. [RAGAS metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), [GraphRAG-Bench evaluators](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/tree/main/Evaluation)
- The offline oracle chooses a correctly answering mode by known estimated direct USD cost, then known tokens, then stable mode name. Report regret and savings against aligned static runs. Classifier training accepts development outcomes and rejects held-out artifacts. `calibrate-router` uses paired development anchors for R4.
- RAGRouter-Bench cross-validation measures native query-type labels. Identical questions after case and whitespace normalization stay in the same five-fold group. Leave-one-domain-out training excludes questions with the same text as a test question. The main-corpus leave-one-dataset-out check uses only development questions and maps each corpus’s labels to the same fixed R2 route taxonomy before fitting or scoring.
- CIs use 1,000 seeded question-level resamples. Paired comparisons require identical question IDs. Binary tests use exact McNemar and report Holm-adjusted probabilities across compared arms. Power analysis uses development discordance. Pairwise judging counts an ordered win only when both answer orders agree.
- Costs labeled estimated use provider-reported token counts and the configured rate card. The spend ledger begins with conservative reservations and reconciles known counts. Query reports retain per-role usage, retries and cache status. Local compute cost and provider invoices require separate observations.
- For the triple audit, two annotators independently label 50 randomly selected triples as correct, partial or wrong using the supplied passages. For the judge audit they label 100 sampled answers using the question and reference. `audit-agreement` calculates Cohen’s kappa from actual labels. A judge kappa below 0.6 requires revision on development data.

## Development V4 freeze

`tune-v4` reads the development partition and grades the reranked passages against its gold evidence. It keeps candidate count, fusion and models fixed, selects by evidence recall followed by precision and nDCG, and writes a configuration and an aligned trial report. Pass the generated `v4.json` as `--config` for subsequent runs.

```sh
uv run --no-sync elevate-rag --config configs/reference.toml tune-v4 \
  data/multihop-rag artifacts/tuning/reference-v4 --limit 100
```

## Optional local tracing

Install the `tracing` extra and set `langfuse_tracing = true` in the active configuration. Set `LANGFUSE_BASE_URL` to a loopback instance and set the project key variables in the environment. Trace data includes stage names, counts, timing, experiment IDs, model IDs, usage and error types. Questions, answers, passages and raw errors remain in the appropriate local artifacts. [Langfuse Python SDK](https://python.reference.langfuse.com/langfuse)

## External workflow

`export-reference` writes text, documents, question IDs and checksums for a standalone framework. Copy `configs/reference-framework.example.toml` and set exact model IDs, direct loopback endpoints and the installed framework version. Run `run-reference DATASET EXPORTED OUT --python PATH --settings FILE` with the chosen framework's isolated Python. The Neo4j path requires a dedicated empty database and its two credential environment variables. Microsoft uses global search. The result is checkpointed alongside the execution manifest. Use `grade-reference DATASET OUT SCORED` with the same independent judge configuration used for the internal experiment. `import-reference` accepts one JSONL object per query with `id`, `status`, `answer`, `contexts` and optional measured latency, tokens and estimated cost. Keep those measurements in isolated reports. [LlamaIndex PropertyGraphIndex](https://developers.llamaindex.ai/python/framework/module_guides/indexing/lpg_index_guide/), [Neo4j GraphRAG](https://neo4j.com/docs/neo4j-graphrag-python/current/), [Microsoft GraphRAG](https://microsoft.github.io/graphrag/), [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

For the native Neo4j GraphRAG comparison, enable APOC Core for the SDK’s entity resolution. In the Compose profile, set `ELEVATE_NEO4J_PLUGINS='["apoc"]'` and use a separate Compose project and port. The LightRAG Neo4j database path has a separate verification check. [Neo4j Docker plugins](https://neo4j.com/docs/operations-manual/current/docker/plugins/)

Use `export-graphrag-eval` to prepare saved native or isolated outcomes for the upstream evaluation script. Use its detailed retrieval output with `attach-graphrag-metrics` to generate an evaluated copy. Obtain separate approval for any provider expenditure made by upstream tools.
