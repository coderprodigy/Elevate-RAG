# Elevate-RAG — Requirements

Source of truth: *Hybrid Vector + Graph RAG: Project Design & Evaluation Plan* (20 Nov 2026,
"the plan") and the *Elevate-RAG Master Implementation Specification* ("the spec").
Every requirement has a stable ID used in `REQUIREMENTS_TRACEABILITY.md`.

Priority: **M** = must (plan or spec says so explicitly), **S** = should, **C** = could.

## 1. Research questions (plan §Goals)

| ID | Requirement | P |
|----|-------------|---|
| RQ1 | Measure how much tuned vector RAG (hybrid + reranker) closes the gap to graph RAG: accuracy and recall per dataset | M |
| RQ2 | Accuracy by query type (single-hop, multi-hop, comparison, temporal, global, null) for each retrieval mode | M |
| RQ3 | Router accuracy, oracle regret and cost saved vs. best static pipeline | M |
| RQ4 | DB choice effect: recall at matched ANN settings, p95 latency, build time, memory | M |
| RQ5 | Model factor effect: accuracy change per factor and cost / 1,000 queries | M |
| RQ6 | Graph RAG cost: indexing tokens/time, query tokens, storage size (collected in every run) | M |

## 2. Functional requirements

### Data (plan §Phase 1 step 1, §Datasets; spec §6)
| ID | Requirement | P |
|----|-------------|---|
| FR-D1 | Loaders for NQ (500 q, gold + distractor passages), HotpotQA distractor (1,000 hard dev), MuSiQue (500 answerable + 100 unanswerable), MultiHop-RAG (full), GraphRAG-Bench Xiang et al. (novel + medical, full), RAGRouter-Bench (router only, 5-fold CV) | M |
| FR-D2 | Normalised corpus JSONL schema `doc_id,title,text,source`; separate question schema with answers, query type, answerability, gold evidence IDs | M |
| FR-D3 | Deterministic IDs, schema validation, duplicate detection, checksums, manifests, reproducible sampling, explicit malformed-record handling, download cache, configurable locations | M |
| FR-D4 | 20 % dev / 80 % test split per dataset; never tune on test; router training only on dev or external data | M |
| FR-D5 | Gold evidence / answers never visible to online retrieval or generation | M |
| FR-D6 | Optional UltraDomain sensemaking set | C |
| FR-D7 | Synthetic deterministic corpus for offline tests, clearly labelled | M |

### Vector RAG (plan §Phase 1; spec §7)
| ID | Requirement | P |
|----|-------------|---|
| FR-V1 | Sentence-aware chunking, default 512 tokens / 64 overlap; ablations 256 / 1,024 and paragraph chunks | M |
| FR-V2 | One reference embedder; model name + version stored with every vector | M |
| FR-V3 | HNSW index with recorded `M`, `ef_construction`, `ef_search`, fixed across systems | M |
| FR-V4 | Baselines V0 closed-book, V1 BM25, V2 dense, V3 hybrid RRF (k=60), V4 hybrid + cross-encoder reranker; top-50 candidates → top 5–10 | M |
| FR-V5 | Generation with numbered passages, citations, "insufficient information" refusal | M |
| FR-V6 | Sparse and dense retrieval share the chunk store; deterministic tie-breaking | M |

### Graph RAG (plan §Phase 2; spec §8)
| ID | Requirement | P |
|----|-------------|---|
| FR-G1 | Graph built from the same chunks; every entity/relation has source-chunk provenance | M |
| FR-G2 | LLM extraction of entities (name, type, description) and relations (src, tgt, description, keywords, weight) | M |
| FR-G3 | Entity resolution: normalised name, then embedding similarity above an ablatable threshold | M |
| FR-G4 | Description summarisation for heavily merged entities | M |
| FR-G5 | Dual indexing: entity/relation profiles embedded in the vector store, nodes/edges in the graph store | M |
| FR-G6 | Retrieval modes naive / local / global / hybrid / mix; 1-hop vs 2-hop ablation | M |
| FR-G7 | Optional community layer (clustering + LLM report) as ablation only | S |
| FR-G8 | Graph quality checks before QA: answer-entity coverage, node/edge count, avg degree, clustering coeff., duplicate rate, build time, tokens, storage | M |
| FR-G9 | 50-triple manual audit export, two annotators, Cohen's kappa | M |
| FR-G10 | Context-token budget for graph modes from day one | M |
| FR-G11 | Fallback: coverage < 60 % after tuning → rebuild with LlamaIndex `SchemaLLMPathExtractor` | S |

### Routing (plan §Router; spec §9)
| ID | Requirement | P |
|----|-------------|---|
| FR-R0 | R0 static modes, compared with V4 | M |
| FR-R1 | R1 rules: keywords, patterns, question word, entity count, comparison/temporal/global cues; configurable | M |
| FR-R2 | R2 TF-IDF + SVM (optional small encoder); dataset-type labels and outcome labels; persistence, versioning, 5-fold CV and leave-one-dataset-out | M |
| FR-R3 | R3 LLM classifier with validated structured output and failure handling; latency/token/cost tracked separately | M |
| FR-R4 | R4 cheap path first, escalate once to `mix` on low rerank score or refusal; threshold tuned on dev; everything logged | M |
| FR-R5 | Oracle = cheapest correct mode, offline only | M |
| FR-R6 | Report routing accuracy, macro-F1, end-to-end accuracy, regret, token/cost saved vs always-`mix`, fallback rate, per-type breakdown | M |

### Databases (plan §Database comparison; spec §10)
| ID | Requirement | P |
|----|-------------|---|
| FR-DB1 | Vector adapters: Qdrant, Milvus, pgvector, FAISS (+ in-process exact fallback) | M |
| FR-DB2 | Graph adapters: Neo4j, Memgraph, PostgreSQL tables, NetworkX | M |
| FR-DB3 | Docker Compose with pinned versions, health checks, volumes, resource limits, no hard-coded ports/credentials | M |
| FR-DB4 | Level A vector: matched recall@10 = 0.95 vs exact; recall-vs-QPS curve; QPS at 1/8/32 clients; p50/95/99; build time; RAM/disk; filtered search; write-to-searchable delay | M |
| FR-DB5 | Level A graph: bulk load, entity lookup, 1-/2-hop latency on 1,000 random entities, memory, local-mode latency | M |
| FR-DB6 | Level B: swap LightRAG backend, fixed 300-question subset, accuracy/recall/latency/setup effort; 1,000 warm-ups, 3 repeats, median | M |
| FR-DB7 | Never silently substitute a fallback DB in a real experiment | M |

### Models (plan §Model comparison; spec §11)
| ID | Requirement | P |
|----|-------------|---|
| FR-M1 | Independent provider config per role: embedder, reranker, extractor, keyword, generator, router, judge | M |
| FR-M2 | Local and API providers; no invented model IDs; freeze IDs/revisions before experiments | M |
| FR-M3 | Reference config: strong open embedder, open reranker, cheap API extractor, mid open generator, Qdrant + Neo4j | M |
| FR-M4 | Judge from a different family than every generator | M |
| FR-M5 | Embedding-dimension validation; embedder change ⇒ separate index version | M |

### Generation & context (spec §12)
| ID | Requirement | P |
|----|-------------|---|
| FR-C1 | Same prompt across modes; citations validated; no fabricated citations | M |
| FR-C2 | Temperature 0 / seeds where supported; log when unsupported | M |
| FR-C3 | Tokenizer-aware context budgets 4k / 8k / 16k; one reference budget | M |
| FR-C4 | Retrieved text treated as untrusted data | M |
| FR-C5 | Record exact evidence passed to generator | M |

### Evaluation & statistics (plan §Metrics, §Protocol, §Statistics; spec §13, §15)
| ID | Requirement | P |
|----|-------------|---|
| FR-E1 | Recall@k, nDCG@10 (own script) | M |
| FR-E2 | RAGAS context recall/precision and faithfulness | M |
| FR-E3 | GraphRAG-Bench evidence recall / context relevance | S |
| FR-E4 | EM, token F1 with SQuAD normalisation | M |
| FR-E5 | LLM-judge correctness (correct/partial/wrong) – primary metric | M |
| FR-E6 | Refusal accuracy on unanswerable questions | M |
| FR-E7 | Pairwise comprehensiveness / diversity / empowerment, both orders, win only if both agree | M |
| FR-E8 | Efficiency: build time/tokens, storage, latency p50/p95, tokens per query, $/1k queries | M |
| FR-E9 | Judge validation: 100 human labels × 2 annotators, kappa ≥ 0.6 gate | M |
| FR-E10 | Bootstrap 95 % CI (1,000 resamples), paired bootstrap, McNemar, Holm–Bonferroni, effect sizes, MDE/power check | M |
| FR-E11 | Error analysis: 50 failures per system, six cause categories, failure-cause chart | M |
| FR-E12 | Failed queries counted explicitly, never silently dropped; unsupported metrics reported as missing, not zero | M |

### Experiments (plan §Experiment matrix; spec §14)
| ID | Requirement | P |
|----|-------------|---|
| FR-X1 | Config-driven runner for E1–E6 and A1–A4 (chunk size, merge threshold, context budget, graph depth, community layer) | M |
| FR-X2 | One-factor-at-a-time from reference; full factorial only mode × generator (E6); extractor comparison only on MultiHop-RAG + GraphRAG-Bench medical | M |
| FR-X3 | External reference systems (MS GraphRAG, HippoRAG 2, LlamaIndex PGI, Neo4j GraphRAG) isolated from controlled comparisons | S |
| FR-X4 | Dry run, validation, cost estimate, subsets, resume/checkpoint, failed-run recovery, run IDs, immutable manifests, per-question JSONL, aggregate tables | M |
| FR-X5 | 5 % indexing pilot + extrapolation; hard spending cap requiring approval | M |

### Reporting & observability (plan §Report template; spec §20)
| ID | Requirement | P |
|----|-------------|---|
| FR-O1 | Structured logs; optional Langfuse tracing | M |
| FR-O2 | Per-query record: run ID, question ID, dataset version, config hash, mode, route, chunk IDs, scores, context size, model IDs, stage timings, tokens, cost, answer, citations, eval, errors | M |
| FR-O3 | Figures: architecture, accuracy by type × mode, router cost-vs-accuracy, recall-vs-QPS, factor importance, mode × generator interaction, failure causes, cost / 1k queries | M |
| FR-O4 | Actual vs estimated cost distinguishable; cached vs uncached distinguished | M |

### Developer experience (spec §19)
| ID | Requirement | P |
|----|-------------|---|
| FR-U1 | CLI: env check, download, prepare, index, validate index, ask, inspect graph, train router, run experiment, benchmark, evaluate, report | M |
| FR-U2 | Offline demo without paid credentials; demo results separated from research results | M |

## 3. Non-functional requirements

| ID | Requirement | P |
|----|-------------|---|
| NFR-1 | Reproducibility: lockfile, frozen model IDs, config hashing, content-hash caching, seeds | M |
| NFR-2 | Fairness: one variable at a time; same chunks, embeddings, prompt, top-k, budget, reranker | M |
| NFR-3 | Cost safety: no unapproved paid calls; spending cap; cache everything | M |
| NFR-4 | Reliability: bounded concurrency, timeouts, retries with backoff, idempotent resumable indexing, cancellation | M |
| NFR-5 | Quality: typed code, Ruff, mypy, pytest, ≥ 85 % coverage of core logic, CI | M |
| NFR-6 | Security: no committed secrets, `.env.example`, never overwrite `.env`, credentials redacted in logs | M |
| NFR-7 | Simplicity: no unnecessary abstractions, no empty modules, no fake implementations | M |

## 4. Out of scope (plan)
Multimodal documents, generator fine-tuning, access control / multi-tenancy, agentic multi-round
retrieval beyond one fallback step.

## 5. Ambiguities and how they were resolved
See `DECISIONS.md` for the full record. Summary:

1. **"Same reranker for all paths"** vs graph modes that return entities/relations, not passages →
   the shared reranker reorders *chunk* items for every mode; KG items keep LightRAG's order; one
   shared token budget applies to the assembled context (D-07).
2. **Temperature 0** is rejected by current Claude models (API version 2026-10-21) → the provider
   omits it, logs `temperature_unsupported`, and reproducibility relies on the response cache (D-09).
3. **NQ corpus**: the plan says "gold passages plus distractors" without a source → `florin-hf/nq_open_gold`
   gold passages; distractors are other questions' gold passages (D-12).
4. **HotpotQA "hard"**: the distractor dev set is entirely `level=hard` → sample 1,000 of it (D-12).
5. **Judge family**: proposal generators include Claude, Gemma and Qwen → judge must be from another
   family; no ID is invented, the judge must be set at kickoff and validation fails until then (D-15).
6. **RAGRouter-Bench**: hosted as `Chaplain0908/RAGRouter` on Hugging Face (D-12).
