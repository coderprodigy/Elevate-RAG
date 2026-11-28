# Requirements Traceability Matrix

Status legend (spec §24):
**IV** implemented & verified · **IN** implemented, not externally verified (needs services /
credentials / hardware) · **PI** partially implemented · **BL** blocked (external) · **NI** not implemented.

Code paths are relative to `src/elevate_rag/`. Tests: `tests/…`. Evidence for IV rows = the tests
named (all passing: `171 passed`, live-DB tests deselected while Docker is down) and/or the real-data artefact named.

## Research questions

| ID | Implementation | Experiment spec | Status |
|---|---|---|---|
| RQ1 | V0–V4 in `pipeline.py`; reports | `configs/experiments/E1_vector_baselines.yaml` | IN (runner verified on synthetic; BM25 V1 measured on real data → `results/model_free/`) · research run BL (X1–X3) |
| RQ2 | LightRAG modes + per-type breakdowns (`harness.summarize`, `fig_accuracy_by_type`) | `E2_retrieval_modes.yaml` | IN · BL (X1–X3) |
| RQ3 | `routing/*`, `oracle.routing_report` | `E3_labels_dev.yaml`, `E3_routers.yaml` | IN · R1/R2 classification measured on real labels · end-to-end BL |
| RQ4 | `storage/*`, `benchmarks/dbbench.py`, E4 Level B | `E4_databases_levelB.yaml`, `bench` CLI | exact/FAISS/NetworkX IV · server DBs IN (X4) |
| RQ5 | provider configs, one-factor variants, E6 factorial | `E5_model_factors.yaml`, `E5_extractors.yaml`, `E6_mode_x_generator.yaml` | IN · BL (X1–X3, X6) |
| RQ6 | index manifest (`timings`, `usage_delta`, `storage_bytes`), per-query tokens/cost | every run | IV (recorded in demo runs) |

## Functional requirements

| ID | Code | Tests | Status |
|---|---|---|---|
| FR-D1 loaders (NQ, HotpotQA, MuSiQue, MultiHop-RAG, GraphRAG-Bench ×2, RAGRouter) | `data/datasets.py` | `unit/test_data.py` (real schemas); real prepare runs (manifests in `data/processed/*/manifest.json`) | IV |
| FR-D2 normalised schemas | `types.py` (`Document`, `Question`) | `unit/test_data.py` | IV |
| FR-D3 IDs, validation, dedupe, checksums, manifests, sampling, malformed records, cache | `data/prepare.py`, `datasets.deterministic_sample` | `test_dedup_…`, `test_reference_validation_and_store_checksums`, `test_graphrag_bench_and_ragrouter` (rejected record reported) | IV |
| FR-D4 20/80 split, no test tuning | `prepare.stratified_split`; `router train` refuses test; R4 tuned on dev | `test_split_is_…`, `test_cli_smoke` (test-split training refused), `test_experiment_run_score_report_and_resume` (dev tuning) | IV |
| FR-D5 gold never online | `Question.to_query`, `harness.run_questions` passes `Query` only | `test_offline_pipeline_never_sees_gold_fields` | IV |
| FR-D6 UltraDomain | – | – | NI (optional) |
| FR-D7 synthetic fixtures | `data/synthetic.py` (flagged `synthetic: true`) | used throughout | IV |
| Team goal G-B (not in the plan): audio/video ingestion | `data/media.py`, `data/media_backends.py` (speech), `data/media_music.py`, `data/media_vision.py`, `data/media_tags.py`, `data ingest-media`, `scripts/make_media_demo.py` (D-37, D-38) | `unit/test_media.py` (CI, no extra), `integration/test_media_backends.py`, `test_media_music.py` (tempo/key on calibration signals, OpenMP isolation), `test_media_vision.py` (cuts ±0.5 s); end-to-end runs on the synthetic demo set | IV · team clips open |
| FR-V1 chunking 512/64, sentence-aware, ablations, paragraph | `indexing/chunker.py` | `unit/test_chunking_retrieval.py` | IV |
| FR-V2 one embedder, ID stored with vectors | `providers/embeddings.py`, index manifest `embedder.id`, `open()` rejects mismatch | `test_embedding_dimension_mismatch_is_rejected`, `test_extends_overrides_and_hashes` | IV |
| FR-V3 HNSW params recorded/fixed | `VectorStoreConfig`, index manifest `vector_store.hnsw` | FAISS test | IV (FAISS) · IN (servers) |
| FR-V4 V0–V4, top-50 → top-k, RRF 60 | `pipeline.retrieve`, `retrieval/lexical.py` | `test_rrf_matches_formula`, `test_all_modes_answer_…` | IV |
| FR-V5 numbered passages, citations, refusal | `generation/generator.py` | `unit/test_generation.py` | IV |
| FR-V6 shared chunk store, deterministic ties | `ChunkStore`, BM25/RRF/exact tie-breaks | `test_bm25_ties_…`, `test_exact_store_…` | IV |
| FR-G1 same chunks + provenance | `graph/lightrag_engine.py` (`ainsert_custom_chunks`, chunk map, `verify_chunk_identity`) | `integration/test_chunk_identity_and_provenance` | IV (LightRAG 1.5.7, offline extractor) |
| FR-G2 LLM extraction | LightRAG extraction with our provider stack | integration tests; real-LLM extraction BL (X1) | IN |
| FR-G3 name + embedding resolution, threshold | `graph/resolution.py` → `amerge_entities` | `test_plan_merges_…`, `test_entity_merge_through_lightrag_api` | IV |
| FR-G4 description summarisation | LightRAG `force_llm_summary_on_merge` (config) | – | IN |
| FR-G5 dual indexing | LightRAG entity/relation VDBs + graph store | integration tests | IV |
| FR-G6 modes + 1/2-hop | `lightrag_engine.retrieve`, `_expand_two_hop` | `test_every_mode_…`, `test_two_hop_expansion_…` | IV |
| FR-G7 community ablation | `graph/community.py` (Leiden + reports) | `test_community_layer_ablation` | IV |
| FR-G8 quality checks | `graph/quality.py`, `index validate` | `test_answer_entity_coverage_rules`, `test_graph_shape_…`, CLI smoke | IV (tooling) · gate G3 BL (X1) |
| FR-G9 50-triple audit + kappa | **removed by owner decision D-36** (no annotators); graph quality is checked automatically by FR-G8 (G3 coverage, shape, duplicate rate) | – | NI (deviation, D-36) |
| FR-G10 graph context budget | `generation/context.py` | `test_kg_share_…` | IV |
| FR-G11 LlamaIndex fallback | – | – | NI (KNOWN_ISSUES P1) |
| FR-R0 static | `Router` level static | `test_router_levels_…` | IV |
| FR-R1 rules | `router.classify_rules` | `test_rule_router` + real-label accuracy (`results/model_free`) | IV |
| FR-R2 TF-IDF+SVM, both labels, persistence, 5-fold, LODO | `TfidfSvmRouter`, `cross_validate`, `oracle.outcome_training_labels`, `router train/cv` | `test_tfidf_svm_router_…`, `test_oracle_…`, CLI tests; RAGRouter CV in `results/model_free` | IV |
| FR-R3 LLM classifier, validated output | `Router._decide_llm` | `test_router_levels_…` (invalid output → R1 fallback, recorded) | IV (offline) · real LLM IN |
| FR-R4 confidence fallback, one step, logged | `routing/fallback.py` | `test_tune_threshold_…`, `test_should_escalate_reasons`, e2e fallback | IV |
| FR-R5 oracle offline | `routing/oracle.py` (not imported online) | `test_oracle_and_routing_report` | IV |
| FR-R6 router metrics | `oracle.routing_report` | same | IV |
| FR-DB1 vector adapters | `storage/vector.py` | exact/FAISS tests; `integration/test_db_services.py` | IV exact+FAISS+**Qdrant (live)** · IN Milvus/pgvector (X4) |
| FR-DB2 graph adapters | `storage/graph.py` | NetworkX tests; db-services tests | IV NetworkX+**Neo4j (live)** · IN Memgraph/PG (X4) |
| FR-DB3 compose | `infrastructure/docker/docker-compose.yml` | `docker compose config` validated | IN (X4) |
| FR-DB4 Level A vector | `benchmarks/dbbench.bench_vector_store`, `bench vector` | `test_vector_bench_…`, `test_faiss_hnsw_…`, CLI bench | IV (exact, FAISS) · IN servers |
| FR-DB5 Level A graph | `bench_graph_store`, `bench graph` | `test_graph_bench_networkx`, CLI | IV (NetworkX) · IN servers |
| FR-DB6 Level B | `E4_databases_levelB.yaml` (LightRAG storage swap, limit 300) | spec validates | IN · BL (X1, X4) |
| FR-DB7 no silent fallback | adapters raise `ConfigError`/`StorageError` | `test_server_stores_fail_loudly_…` | IV |
| FR-M1 per-role providers | `ModelsConfig`, `providers/*` | provider tests | IV |
| FR-M2 verified IDs, frozen revisions | `configs/pipelines/reference.yaml` (HF SHAs, Claude IDs) | `test_repository_configs_are_valid` | IV (verified 2026-11-26) |
| FR-M3 reference config | `reference.yaml` | same | IN (X1, X6) |
| FR-M4 independent judge | `experiments.validate` | `test_validate_enforces_protocol_rules` | IV |
| FR-M5 dimension validation, index versioning | `BaseEmbedder.embed`, `index_key`, `open()` | `test_embedding_dimension_mismatch_…`, config test | IV |
| FR-C1 same prompt, citation validation | `Generator` | generation tests | IV |
| FR-C2 temperature 0 / seeds | OpenAI-compat sends `temperature=0, seed`; Claude unsupported → logged | `test_openai_compat_…`, `test_anthropic_provider_contract` | IV |
| FR-C3 tokenizer-aware 4k/8k/16k | `assemble_context`, A3 variants | context tests | IV |
| FR-C4 untrusted context | system prompt + delimiters | `test_prompt_marks_passages_untrusted…` | IV |
| FR-C5 evidence recorded | `QueryRecord.context_item_ids/context_items` | e2e tests | IV |
| FR-E1 recall@k, nDCG@10 | `evaluation/metrics.py` | metric tests; real BM25 results | IV |
| FR-E2 context recall/precision, faithfulness | `evaluation/ragas_metrics.py` (`ragas` 0.4.3 package, spec `ragas: true`); `evaluation/llm_metrics.py` (native, for paid judges) | `test_ragas_scores_answerable_rows_…`, `test_llm_metrics_follow_ragas_definitions`; live run with Llama-3.1-8B via Ollama (3 G1 questions) | IV |
| FR-E3 evidence recall / context relevance | string evidence recall (valid only when evidence is verbatim) | `test_evidence_recall_…`, `test_paraphrased_evidence_…` | PI (P3) |
| FR-E4 EM / F1 | metrics | tests | IV |
| FR-E5 judge correctness | `evaluation/judge.py`; second judge `models.judge2`, agreement `harness.judge_agreement` | `test_correctness_judge`, `test_judge_agreement_ignores_rule_verdicts_…`, `test_runs_write_traceable_records_with_two_judges` | IV |
| FR-E6 refusal accuracy | `refusal_correct`, unanswerable rule | judge + e2e tests | IV |
| FR-E7 pairwise, both orders | `PairwiseJudge`, `evaluation/pairwise.py`, `experiment pairwise` | `test_pairwise_judge_…`, `test_compare_runs_…` | IV (logic) |
| FR-E8 efficiency metrics | `summarize.efficiency`, index manifest | `test_summarize_…` | IV |
| FR-E9 judge validation kappa ≥ 0.6 | **replaced (D-36)**: GroUSE calibration (`evaluation/judge_calibration.py`, `judge calibrate`) + inter-judge κ ≥ 0.6 → `judge_validation` status per run | `test_grouse_conditions_map_…`, `test_calibrate_…`, `test_judge_validation_requires_…`, `test_revalidate_…`; real calibration of Llama-3.1-8B (`results/judge_calibration/`) | IV (deviation from human labels, D-36) |
| FR-E10 statistics | `stats.py`, `reporting.paired_comparisons` | `unit/test_metrics_stats.py` | IV |
| FR-E11 error analysis | **removed by owner decision D-36** (manual cause labelling) | – | NI (deviation, D-36) |
| FR-E12 failures counted, undefined ≠ 0 | `score_rows`, `summarize` (`__undefined`, `error_rate`, cost None) | `test_summarize_counts_failures_and_undefined` | IV |
| FR-X1 runner E1–E6, A1–A4 | `experiments/runner.py`, `configs/experiments/` | e2e experiment + CLI tests; all specs expand & validate | IV |
| FR-X2 one factor at a time; extractor datasets | variants; `validate` | `test_validate_enforces_protocol_rules` | IV |
| FR-X3 external systems isolated | `experiments/external.py` | `test_external_predictions_contract` | PI (P4) |
| FR-X4 dry-run, validation, estimate, subsets, resume, run IDs, immutable manifests, JSONL, tables | runner + harness; `--pipeline/--datasets/--limit/--shard` | `test_run_questions_resumes_after_crash`, `test_experiment_run_…_resume`, `test_select_shard_partitions_runs_exactly_once`, `test_load_spec_can_retarget_pipeline_limit_and_datasets`, CLI dry-run | IV |
| FR-X5 5 % pilot + cap | `index pilot`, `UsageLedger.check_budget` | `test_budget_cap_…` (unit + e2e) | IV (cap) · IN (pilot with real LLM) |
| FR-O1 structured logs (+Langfuse) | `utils.configure_logging` (JSON, redaction) | `test_json_log_formatter_redacts_secrets` | PI (Langfuse NI, P5) |
| FR-O2 per-query record | `pipeline.QueryRecord` | e2e tests (fields asserted) | IV |
| FR-O3 figures | `reporting/report.py`; architecture in `docs/ARCHITECTURE.md` | `test_report_tables_and_all_figures` | IV (from real run files only) |
| FR-O4 actual vs estimated, cached vs uncached | `CallRecord.cost_is_estimate/cached/tokens_are_estimate`, estimates flagged | provider tests | IV |
| FR-U1 CLI | `cli/main.py` | `test_cli_smoke`, `test_cli_workflows.py` | IV |
| FR-U2 offline demo, labelled | `demo.py`, `elevate-rag demo` | run manually (see PROGRESS) | IV |

## Non-functional

| ID | Evidence | Status |
|---|---|---|
| NFR-1 reproducibility | `uv.lock`, `exclude-newer`, pinned dataset/model revisions, config hashes, caches; committed run records with git commit (`tracking.py`, `results/runs/`), optional MLflow | IV |
| NFR-2 fairness | shared path in `ElevateRAG.answer`; validation rules | IV |
| NFR-3 cost safety | cap checked before calls; `$0` default; `BudgetExceededError` aborts | IV |
| NFR-4 reliability | retries/backoff/timeouts (`retry_async`), bounded semaphores, resumable indexing/runs, crash-safe JSONL | IV |
| NFR-5 quality | ruff clean, mypy clean (53 files), 134 tests, 86 % coverage, CI workflow (+ Docker image job) | IV locally (CI result on GitHub not observed from here) |
| NFR-6 security | `.env` git-ignored (local DB passwords only), secrets redacted in logs, no credentials in repo, DB ports bound to localhost | IV |
| NFR-7 simplicity | no empty modules; offline providers are real algorithms (D-18) | IV |
| NFR-8 portability (laptop, GPU workstation, SLURM cluster, EC2) | `docs/RUNNING.md`; `scripts/run_pipeline.sh`; `infrastructure/cluster/` (SLURM, vLLM, Apptainer); location- and endpoint-independent run ids (D-29, D-30) | laptop: IV (G1 runs); cluster: PI (dry-run + stub-tested, not yet on a real cluster) |
