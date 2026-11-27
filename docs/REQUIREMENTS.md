# Requirements

The pasted Master Implementation Specification is the implementation request. Its incorporated project proposal is the 24-page design and evaluation plan dated November 20, 2026. The proposal supplies the experiment design and gates. The latest user instruction requires that all work stay local and uncommitted on `atlas`.

| Scope | Acceptance evidence |
| --- | --- |
| RQ1 / E1 | V0–V4 use the same frozen corpus and holdout across Natural Questions, HotpotQA, MuSiQue, MultiHop-RAG and both GraphRAG-Bench domains. V0 quantifies closed-book accuracy. |
| RQ2 / E2 | LightRAG naive, local, global, hybrid and mix keep shared chunk IDs and a shared generation budget. Compare by query type. Isolate LlamaIndex, Neo4j GraphRAG, Microsoft GraphRAG and HippoRAG 2. |
| RQ3 / E3 | R0–R4, two R2 label strategies and an offline cheapest-correct oracle. Report accuracy, macro-F1, regret, fallback frequency and savings vs mix. Five-fold RAGRouter-Bench and leave-one-dataset-out validation. |
| RQ4 / E4 | Qdrant, Milvus, pgvector, FAISS, Neo4j, Memgraph, PostgreSQL graph tables and NetworkX. E4 uses MultiHop-RAG and HotpotQA. Match vector recall@10 to 0.95 against exact FAISS at concurrency 1/8/32. Graph Level A uses 1,000 MultiHop-RAG entities. Level B uses a fixed 300-question subset, 1,000 warm-up queries and three measured repetitions. Record latency, throughput, indexing, visibility delay, filtered search, RAM, disk and setup effort. |
| RQ5 / E5–E6 | One factor at a time for embedding, reranking, extraction and generation. Extractor swaps only on MultiHop-RAG and GraphRAG-Bench medical. Cross naive/local/mix with three generator tiers on HotpotQA, MuSiQue and GraphRAG-Bench. |
| RQ6 | Indexing time, consumption and storage, query latency and consumption, dollars per 1,000. A representative 5% graph sample precedes authorized full indexing. |
| Data | NQ 500; HotpotQA 1,000 hard dev; MuSiQue 500 answerable plus 100 unanswerable; full MultiHop-RAG; full Xiang novel and medical GraphRAG-Bench; router benchmark separately; optional UltraDomain. Reproducible stratified 20/80 split, version and integrity tracked, gold never enters online code. |
| Graph quality | Shared 512-token chunks and embeddings, provenance, typed nodes and weighted edges, two-stage entity resolution, 1/2-hop and community ablations, graph shape and coverage, 50-triple double audit. Below 60% after permitted tuning triggers the schema fallback. |
| Evaluation | Recall@K, nDCG@10, context precision and recall, benchmark evidence and relevance, EM/F1, three-class judge, faithfulness and refusal, double-order global judging, 100-answer double audit, 95% intervals using 1,000 resamples, paired tests, Holm correction and dev power. |
| Engineering | Validated config, CLI, independent stages, model/storage adapters, bounded and cached calls, explicit budget, checkpoints, immutable manifests, JSONL/CSV/reporting, 85% meaningful core test coverage, CI and clean install. |

## Controlled ablations

Use 256/512/1,024-token chunks, Wikipedia paragraph boundaries, entity-merge threshold, 4,096/8,192/16,384 context tokens, graph depth and optional community reports. Preserve 64-token overlap unless a development ablation specifies otherwise. Reference retrieval collects 50 candidates with RRF constant 60 and reranks to 5–10.

## Research gates

G1 exercises 20 questions end to end. G2 tunes V4 exclusively on development data and freezes it. G3 requires corpus coverage and two human annotations per sampled triple. G4 freezes authorized measurements before analysis. Graph extraction quality and judge-human kappa use real annotations. The implementation and study have separate acceptance states.
