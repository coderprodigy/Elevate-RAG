# Requirements Document

## Introduction

Elevate-RAG is an ECE 750 research system that builds one shared index serving both vector retrieval and graph retrieval, places a learned query router on top, and measures when each retrieval path wins. The framework is LightRAG (HKUDS, MIT licence, v1.5.7). LlamaIndex PropertyGraphIndex and the Neo4j GraphRAG package are benchmarked alternatives; Microsoft GraphRAG and HippoRAG 2 are reference systems.

The working hypothesis is that vector retrieval wins single-hop and detail questions, graph retrieval wins multi-hop, comparison and global questions, and a router beats either path alone at lower cost than always running both. The system answers six research questions (RQ1 to RQ6) over five datasets plus a router benchmark, reporting answer accuracy, retrieval recall, faithfulness, latency and cost, each with 95% confidence intervals.

### Phased delivery

The implementation is divided into phases. Each phase ends with an exit check: a single documented command that runs on the Tiny_Sample with no-setup backends and prints a pass/fail result.

| Phase | Scope | Requirements |
|---|---|---|
| 0 | Repository, configuration, setup and dataset instructions, LLM gateway, caching, tracing | 1–5 |
| 1 | Dataset loaders, splits, chunking, evaluation harness (gate G1) | 6–10 |
| 2 | Vector baseline V0–V4 (gate G2) | 11–14 |
| 3 | Graph layer, graph retrieval modes, graph quality checks (gate G3) | 15–19 |
| 4 | Router levels R0–R4 and oracle | 20–23 |
| 5 | Vector and graph database comparison | 24–26 |
| 6 | Model comparison, ablations, experiment runner (gate G4) | 27–29 |
| 7 | Statistics, error analysis, human annotation tooling, figures and report assets | 30–34 |
| All | End-to-end pipeline, phase reports, cross-cutting quality | 35–37 |

### Engineering lead assessment

- **Indexing cost is the dominant risk.** LLM entity extraction over five corpora can exceed the budget. The plan mitigates this with a Tiny_Sample first, a 5% index-and-extrapolate cost estimate, content-hash caching and a hard spending cap (Requirements 4, 5, 15).
- **Database setup time can stall progress.** Every phase must run with no-setup backends (FAISS or NanoVectorDB for vectors, NetworkX for graphs) before Docker databases are introduced in Phase 5 (Requirement 2).
- **Model identifiers in the source plan are forward-dated** (for example Claude Sonnet 5.5, Qwen3.6-35B-A3B, Gemma 4 12B). Models are therefore configuration values frozen in a model registry rather than hard-coded, so unavailable models can be substituted without code changes (Requirement 3).
- **Experiment matrix size (about 170 runs) is large for 10 weeks.** Runs are defined declaratively and resumable, so the team can cut scope by removing matrix entries rather than code (Requirement 29).
- **Human-only work** (triple audit, judge validation labels, report writing) is supported by export/import tooling and agreement statistics, not automated (Requirement 32).
- **Recommended minimum viable end-to-end slice:** Natural Questions and MultiHop-RAG Tiny_Sample, FAISS + NetworkX, one local embedder, one reranker, one API LLM, modes naive/local/mix, router R1, EM/F1 and recall@k. This slice should run end to end before any scaling.

### Out of scope

Multimodal input, fine-tuning generator models, access control and multi-tenancy, and agentic multi-round retrieval beyond one fallback step.

## Glossary

- **Elevate_RAG**: The complete system described in this document.
- **CLI**: The command-line entry point of Elevate_RAG used to run every phase, experiment and exit check.
- **Config_Loader**: The component that loads and validates experiment configuration files and environment variables.
- **Model_Registry**: The configuration file listing every frozen model identifier (embedder, reranker, extraction LLM, generator LLM, judge LLM) with provider, version and price per token.
- **LLM_Gateway**: The single component through which every LLM and embedding API call is made; responsible for seeding, temperature, caching, logging, token counting and spend tracking.
- **Cache**: The content-hash keyed store of LLM outputs, embeddings and retrieval results.
- **Budget_Tracker**: The component that accumulates monetary cost of model calls and enforces the Spending_Cap.
- **Spending_Cap**: A configured maximum total spend in US dollars for a run and for the project.
- **Tracer**: The component that records per-call traces (Langfuse when configured, local log files otherwise).
- **Setup_Docs**: The user-facing documentation files `docs/SETUP.md`, `docs/DATASETS.md` and `docs/MODELS.md` describing prerequisites the user must provide.
- **Exit_Check**: A single documented CLI command per phase that runs on the Tiny_Sample and reports pass or fail.
- **Phase_Report**: A markdown file `docs/phase-reports/phase-N.md` summarising what a phase implemented, how to verify it, known limitations and open items.
- **Dataset_Loader**: A per-dataset component that converts raw dataset files into Normalised_Corpus and Question_Set files.
- **Normalised_Corpus**: A JSONL file where each record holds `doc_id`, `title`, `text` and `source`.
- **Question_Set**: A JSONL file where each record holds `question_id`, `question`, `answers`, `gold_passage_ids`, `query_type`, `answerable` and `split`.
- **Query_Type**: One of single-hop, multi-hop, comparison, temporal, global, or null (no answer in corpus).
- **Splitter**: The component that assigns each question to the dev split (20%) or test split (80%).
- **Tiny_Sample**: A deterministic subset of at most 20 questions and their supporting and distractor documents per dataset, used for exit checks and gate G1.
- **Chunker**: The component that splits documents into chunks shared by every retrieval path.
- **Chunk_Store**: The persistent table of chunks with `chunk_id`, `doc_id`, text, token count and offsets; the single source of chunks for every pipeline, read from LightRAG's chunk table.
- **Embedder**: The component that converts chunk text, entity profiles and relation descriptions into vectors.
- **Vector_Store**: An abstraction over vector database backends (FAISS, NanoVectorDB, Qdrant, Milvus, pgvector).
- **Graph_Store**: An abstraction over graph database backends (NetworkX, Neo4j, Memgraph, Postgres graph tables).
- **BM25_Index**: The lexical index over chunks in the Chunk_Store.
- **Vector_Retriever**: The component that performs BM25, dense and hybrid (Reciprocal Rank Fusion) retrieval over chunks.
- **RRF**: Reciprocal Rank Fusion with constant k=60.
- **Reranker**: The cross-encoder component that rescores candidate passages and trims them to the configured top-n.
- **Context_Builder**: The component that assembles retrieved passages, entities and relations into a numbered context limited by the Context_Token_Cap.
- **Context_Token_Cap**: The configured maximum number of tokens of retrieved context passed to the Generator (default 8,000).
- **Generator**: The component that produces an answer from a question and numbered context using a fixed prompt.
- **Refusal_String**: The exact answer text "insufficient information".
- **Vector_Variant**: One of V0 closed-book, V1 BM25, V2 dense, V3 hybrid RRF, V4 hybrid RRF plus Reranker.
- **Extractor**: The component that uses an extraction LLM to produce entities and relations from chunks.
- **Entity_Resolver**: The component that merges duplicate entities by normalised name and then by embedding similarity above the Merge_Threshold.
- **Merge_Threshold**: The configured cosine-similarity threshold used by the Entity_Resolver.
- **Graph_Builder**: The component that runs extraction, resolution, description summarisation and dual indexing into the Vector_Store and Graph_Store.
- **Community_Builder**: The optional component that runs Leiden community detection and produces LLM community reports.
- **Graph_Retriever**: The component that performs LightRAG retrieval modes naive, local, global, hybrid and mix.
- **Retrieval_Mode**: One of the Vector_Variants or one of naive, local, global, hybrid, mix.
- **Graph_Quality_Checker**: The component that computes answer-entity coverage and graph shape statistics.
- **Answer_Entity_Coverage**: The share of gold answer entities in a Question_Set that appear as nodes in the graph.
- **Fallback_Extractor**: The LlamaIndex SchemaLLMPathExtractor with a fixed schema, used when Answer_Entity_Coverage is below 60%.
- **Router**: The component that selects a Retrieval_Mode per query using one of the Router_Levels.
- **Router_Level**: One of R0 static, R1 rules, R2 lightweight classifier, R3 LLM classifier, R4 confidence fallback.
- **Oracle_Router**: The post-hoc router that selects the cheapest Retrieval_Mode that answered a question correctly.
- **Evaluation_Harness**: The component that computes retrieval, answer, faithfulness, router and efficiency metrics for a run.
- **LLM_Judge**: The component that grades answers as correct, partial or wrong, and performs pairwise comparisons, using a model family different from the Generator.
- **Run**: One execution of a configured pipeline over one dataset split, producing a Run_Record.
- **Run_Record**: The persisted output of a Run: per-query predictions, retrieved IDs, metrics, tokens, latency, cost, configuration snapshot, model IDs, call dates and git commit hash.
- **Experiment_Runner**: The component that executes the experiment matrix (E1–E6, A1–A4) from declarative configuration files.
- **DB_Benchmark**: The component that runs Level A micro-benchmarks and Level B end-to-end database comparisons.
- **Stats_Engine**: The component that computes bootstrap confidence intervals, significance tests and multiple-comparison corrections.
- **Annotation_Tool**: The component that exports annotation sheets for human labelling, imports completed sheets and computes Cohen's kappa.
- **Error_Analyzer**: The component that samples failures and aggregates human-assigned failure causes.
- **Report_Builder**: The component that produces figures and tables for the final report.

## Requirements

### Requirement 1: Project structure, configuration and reproducibility (Phase 0)

**User Story:** As a team member, I want one reproducible project layout with validated configuration, so that every run can be repeated exactly.

#### Acceptance Criteria

1. THE Elevate_RAG SHALL provide a Python package, a lockfile of pinned dependency versions, and a CLI entry point covering every phase.
2. WHEN a configuration file is loaded, THE Config_Loader SHALL validate every field against a typed schema and return the validated configuration object.
3. IF a configuration file contains an unknown field, a missing required field or a value of the wrong type, THEN THE Config_Loader SHALL stop with an error message naming the file, the field and the expected type.
4. THE Config_Loader SHALL read API keys and database credentials from environment variables or a `.env` file and SHALL exclude secret values from logs, Run_Records and error messages.
5. THE Elevate_RAG SHALL provide a `.env.example` file listing every supported environment variable with a placeholder value.
6. WHEN a Run starts, THE Elevate_RAG SHALL set fixed random seeds for Python, NumPy and any ML framework in use, and SHALL record the seeds in the Run_Record.
7. WHEN a Run starts, THE Elevate_RAG SHALL record the configuration snapshot, git commit hash, library versions and start time in the Run_Record.
8. THE Elevate_RAG SHALL provide lint, type-check and unit-test commands that run without network access or API keys.

### Requirement 2: No-setup backends and Docker stacks (Phase 0)

**User Story:** As a team member, I want every phase to run with embedded backends before Docker databases exist, so that database setup never blocks progress.

#### Acceptance Criteria

1. THE Vector_Store SHALL expose one interface for insert, delete, top-k search, filtered search by `source`, and index-parameter reporting across every vector backend.
2. THE Graph_Store SHALL expose one interface for node upsert, edge upsert, entity lookup by name, 1-hop neighbour fetch and 2-hop neighbour fetch across every graph backend.
3. WHERE no database backend is configured, THE Elevate_RAG SHALL use FAISS or NanoVectorDB as the Vector_Store and NetworkX as the Graph_Store.
4. THE Elevate_RAG SHALL provide one Docker Compose file per database stack (Qdrant, Milvus, pgvector, Neo4j, Memgraph, Postgres graph tables) with pinned image versions and explicit CPU and memory limits.
5. WHEN a database backend is selected, THE Elevate_RAG SHALL run a connectivity health check and report backend name, version and status before indexing.
6. IF a configured database backend fails the health check, THEN THE Elevate_RAG SHALL stop with an error message that names the backend and references the matching section of the Setup_Docs.

### Requirement 3: Model registry and LLM gateway (Phase 0)

**User Story:** As a researcher, I want all model calls routed through one gateway with frozen model IDs, so that results are reproducible and model substitutions need no code change.

#### Acceptance Criteria

1. THE Model_Registry SHALL record, for every model role (embedder, reranker, extraction LLM, generator LLM, judge LLM), the provider, exact model identifier, version, context length and price per million input and output tokens.
2. THE LLM_Gateway SHALL send every LLM and embedding request through the Model_Registry entry for the requested role.
3. THE LLM_Gateway SHALL issue generation requests with temperature 0.
4. WHEN an LLM call completes, THE LLM_Gateway SHALL log the exact model identifier, call date, prompt tokens, completion tokens, latency and cost.
5. IF an LLM call fails with a transient error, THEN THE LLM_Gateway SHALL retry with exponential backoff up to the configured maximum attempts and then raise an error naming the model and the last failure reason.
6. WHEN an LLM response is expected to be JSON and fails to parse, THE LLM_Gateway SHALL attempt one repair pass and one re-request before recording the output as malformed.
7. THE LLM_Gateway SHALL support separate model settings for the extraction LLM and the generator LLM within the same Run.
8. IF the Judge model and the Generator model in a configuration belong to the same model family, THEN THE Config_Loader SHALL stop with an error naming both models.
9. WHERE the model provider supports a seed parameter, THE LLM_Gateway SHALL pass the configured seed with every generation request.

### Requirement 4: Caching, tracing and budget control (Phase 0)

**User Story:** As a researcher with a fixed budget, I want cached calls, traced usage and a hard spending cap, so that experiments do not repeat paid work or exceed the budget.

#### Acceptance Criteria

1. WHEN an LLM call, embedding call or retrieval is requested, THE Cache SHALL compute a key from the content hash of the input, model identifier and parameters, and SHALL return the stored result on a key match.
2. FOR ALL cacheable requests, issuing the same request twice SHALL produce one provider call and identical results (idempotence property).
3. THE Budget_Tracker SHALL accumulate the cost of every uncached model call per Run and per project.
4. IF accumulated cost reaches the Spending_Cap, THEN THE Budget_Tracker SHALL stop the Run, persist completed results, and report spent amount and remaining work.
5. WHEN an indexing Run is requested with cost estimation enabled, THE Budget_Tracker SHALL index a 5% sample, extrapolate total tokens and cost for the full corpus, and report the estimate before full indexing starts.
6. WHERE Langfuse credentials are configured, THE Tracer SHALL send every LLM call trace to Langfuse.
7. WHERE Langfuse credentials are absent, THE Tracer SHALL write every LLM call trace to a local JSONL log file.

### Requirement 5: Setup and dataset instruction documents (Phase 0)

**User Story:** As the project owner, I want written instructions for everything I must provide, so that I can prepare keys, hardware, models and datasets without reading code.

#### Acceptance Criteria

1. THE Setup_Docs SHALL include `docs/SETUP.md` listing required software (Python version, package manager, Docker and Docker Compose versions), hardware assumptions (CPU, RAM, disk, optional GPU) and step-by-step installation commands.
2. THE Setup_Docs SHALL list every API key the user must provide, the provider sign-up location, the environment variable name and which phases need the key.
3. THE Setup_Docs SHALL include `docs/DATASETS.md` giving, for Natural Questions, HotpotQA distractor, MuSiQue, MultiHop-RAG, GraphRAG-Bench, RAGRouter-Bench and optional UltraDomain, the download source, licence note, expected file layout under the data directory, disk size and the CLI command that verifies the files.
4. THE Setup_Docs SHALL include `docs/MODELS.md` listing every Model_Registry entry, how to obtain or serve local open-weight models, and how to substitute a model that is unavailable.
5. THE Setup_Docs SHALL state, for each phase, which user-provided prerequisites are required and which are optional.
6. WHEN the user runs the setup verification command, THE CLI SHALL check Python version, installed dependencies, environment variables, dataset files, local model availability and Docker availability, and SHALL print one pass, fail or skipped line per check with a reference to the relevant Setup_Docs section.
7. THE Elevate_RAG SHALL include a bundled synthetic Tiny_Sample fixture so that Exit_Checks for Phases 0 to 4 run without downloaded datasets.

### Requirement 6: Dataset loading and normalisation (Phase 1)

**User Story:** As a researcher, I want every dataset converted into one corpus and question format, so that all pipelines consume identical inputs.

#### Acceptance Criteria

1. WHEN a Dataset_Loader is run on raw dataset files, THE Dataset_Loader SHALL write a Normalised_Corpus and a Question_Set to the processed data directory.
2. THE Dataset_Loader SHALL preserve the gold supporting-passage IDs for every question that has them in the source dataset.
3. THE Dataset_Loader SHALL assign a Query_Type to every question from the source labels (MuSiQue hop count, HotpotQA type, MultiHop-RAG type, GraphRAG-Bench level).
4. THE Dataset_Loader for Natural Questions SHALL select 500 questions and build the corpus from gold passages plus distractor passages.
5. THE Dataset_Loader for HotpotQA SHALL select 1,000 hard-level distractor dev questions.
6. THE Dataset_Loader for MuSiQue SHALL select 500 answerable and 100 unanswerable questions and set the `answerable` field accordingly.
7. THE Dataset_Loader for MultiHop-RAG SHALL load the full query set including null queries.
8. THE Dataset_Loader for GraphRAG-Bench SHALL load the novel and medical subsets with level 1 to 4 labels.
9. IF a raw dataset file is missing or has an unexpected schema, THEN THE Dataset_Loader SHALL stop with an error naming the file, the missing field and the `docs/DATASETS.md` section.
10. FOR ALL Normalised_Corpus records, writing then reading the JSONL file SHALL produce records equal to the originals (round-trip property).
11. FOR ALL questions with gold passage IDs, THE Dataset_Loader SHALL ensure every gold passage ID exists in the Normalised_Corpus (referential integrity invariant).
12. THE Dataset_Loader SHALL apply a fixed seed so that repeated loads of the same raw files produce identical outputs.

### Requirement 7: Splits and tiny samples (Phase 1)

**User Story:** As a researcher, I want fixed dev/test splits and a tiny sample, so that tuning never touches test data and smoke tests run in minutes.

#### Acceptance Criteria

1. THE Splitter SHALL assign 20% of each Question_Set to dev and 80% to test, stratified by Query_Type, with a fixed seed.
2. FOR ALL Question_Sets, the dev and test splits SHALL be disjoint and their union SHALL equal the full Question_Set (partition invariant).
3. WHEN a tuning operation (threshold, hyperparameter or prompt selection) is requested on the test split, THE Elevate_RAG SHALL stop with an error stating that tuning is restricted to dev.
4. THE Splitter SHALL produce a Tiny_Sample per dataset of at most 20 dev questions with their gold and distractor documents.

### Requirement 8: Chunking (Phase 1)

**User Story:** As a researcher, I want one shared chunking step, so that vector and graph paths retrieve over identical chunks.

#### Acceptance Criteria

1. THE Chunker SHALL split documents into chunks of at most 512 tokens with 64 tokens of overlap, breaking at sentence boundaries, as the default configuration.
2. WHERE an alternative chunk configuration (256 tokens, 1,024 tokens, or paragraph-level for Wikipedia datasets) is selected, THE Chunker SHALL apply the selected configuration.
3. THE Chunker SHALL write every chunk to the Chunk_Store with `chunk_id`, `doc_id`, text, token count and character offsets.
4. FOR ALL documents, every chunk SHALL have a token count at or below the configured maximum (size invariant).
5. FOR ALL documents, concatenating chunk texts with overlaps removed SHALL reproduce the document text modulo whitespace normalisation (coverage property).
6. FOR ALL documents, chunking the same document twice with the same configuration SHALL produce identical `chunk_id` values and texts (determinism property).
7. THE Elevate_RAG SHALL map each gold supporting-passage ID to the set of chunk IDs derived from that passage.

### Requirement 9: Evaluation harness (Phase 1)

**User Story:** As a researcher, I want one harness that scores every run the same way, so that results are comparable across pipelines.

#### Acceptance Criteria

1. THE Evaluation_Harness SHALL compute Recall@k for k in {5, 10, 20, 50} and nDCG@10 against gold passages for every Run that retrieves passages.
2. THE Evaluation_Harness SHALL compute exact match and token F1 after normalisation (lowercase, punctuation removal, article removal, whitespace collapse).
3. FOR ALL answer strings, normalising an already normalised string SHALL return the same string (idempotence property).
4. FOR ALL answer strings, exact match of a string against itself SHALL equal 1 and token F1 SHALL equal 1.
5. THE Evaluation_Harness SHALL compute LLM_Judge correctness (correct, partial, wrong) as the primary accuracy metric.
6. THE Evaluation_Harness SHALL compute RAGAS faithfulness, context recall and context precision.
7. THE Evaluation_Harness SHALL compute GraphRAG-Bench evidence recall for GraphRAG-Bench Runs.
8. THE Evaluation_Harness SHALL compute refusal accuracy on unanswerable questions as the share of unanswerable questions answered with the Refusal_String, and the false-refusal rate on answerable questions.
9. THE Evaluation_Harness SHALL compute latency p50 and p95, prompt and completion tokens per query, and cost per 1,000 queries.
10. THE Evaluation_Harness SHALL report every metric overall and broken down by Query_Type.
11. WHEN a Run completes, THE Evaluation_Harness SHALL write the Run_Record to the results directory as per-query JSONL plus a summary JSON file.

### Requirement 10: LLM judge (Phase 1)

**User Story:** As a researcher, I want a validated, order-robust LLM judge, so that judge bias does not drive conclusions.

#### Acceptance Criteria

1. THE LLM_Judge SHALL use a fixed, versioned grading prompt and return one label of correct, partial or wrong with a short rationale.
2. WHEN a pairwise comparison is requested, THE LLM_Judge SHALL judge comprehensiveness, diversity and empowerment twice with answer order swapped.
3. WHEN the two order-swapped pairwise judgments disagree, THE LLM_Judge SHALL record the comparison as a tie.
4. IF the LLM_Judge output cannot be parsed into a valid label after the LLM_Gateway repair step, THEN THE LLM_Judge SHALL record the item as unjudged and exclude the item from accuracy with a count of unjudged items reported.
5. WHEN the judge validation command is run with 100 hand-labelled answers, THE Evaluation_Harness SHALL compute Cohen's kappa between the LLM_Judge and the human labels and report pass when kappa is at least 0.6.
6. WHEN the user runs the Phase 1 Exit_Check, THE CLI SHALL load, split, chunk and evaluate a closed-book and a trivial retrieval pipeline on 20 Tiny_Sample questions end to end and print the metrics summary (gate G1).

### Requirement 11: Vector indexing (Phase 2)

**User Story:** As a researcher, I want a dense vector index with fixed, recorded parameters, so that vector backends are compared fairly.

#### Acceptance Criteria

1. THE Embedder SHALL store the embedding model name and version with every stored vector.
2. IF a query embedding model identifier differs from the model identifier stored with the index, THEN THE Vector_Retriever SHALL stop with an error naming both identifiers.
3. THE Vector_Store SHALL build HNSW indexes with configured M, ef_construction and ef_search values and record the values in the Run_Record.
4. THE BM25_Index SHALL index the same chunk IDs held in the Chunk_Store.
5. FOR ALL indexed chunks, the set of chunk IDs in the Vector_Store, the BM25_Index and the Chunk_Store SHALL be equal (shared-chunk invariant).

### Requirement 12: Vector retrieval and reranking (Phase 2)

**User Story:** As a researcher, I want BM25, dense, hybrid and reranked retrieval over the shared chunks, so that I have a tuned vector baseline.

#### Acceptance Criteria

1. WHEN a query is submitted in BM25 mode, THE Vector_Retriever SHALL return the top 50 chunks ranked by BM25 score.
2. WHEN a query is submitted in dense mode, THE Vector_Retriever SHALL return the top 50 chunks ranked by vector similarity.
3. WHEN a query is submitted in hybrid mode, THE Vector_Retriever SHALL fuse BM25 and dense rankings with RRF (k=60) and return the top 50 chunks.
4. FOR ALL ranked lists, RRF fusion SHALL produce the same result regardless of the order in which input lists are supplied (confluence property).
5. FOR ALL RRF inputs, a chunk ranked first in every input list SHALL be ranked first in the fused list (monotonicity property).
6. WHEN reranking is enabled, THE Reranker SHALL rescore candidate chunks with the configured cross-encoder and return the configured top-n between 5 and 10.
7. FOR ALL reranker outputs, the returned chunks SHALL be a subset of the input candidates sorted by descending reranker score (subset and ordering invariant).
8. THE Vector_Retriever SHALL read chunks from the Chunk_Store so that V1, V3 and V4 use the same chunks as LightRAG naive mode.

### Requirement 13: Context building and generation (Phase 2)

**User Story:** As a researcher, I want a fixed generation prompt with citations and refusal, so that answer quality differences come from retrieval only.

#### Acceptance Criteria

1. THE Context_Builder SHALL number every passage, entity block and relation block in the context and keep the total context at or below the Context_Token_Cap.
2. FOR ALL retrieval outputs, the context produced by the Context_Builder SHALL have a token count at or below the Context_Token_Cap (budget invariant).
3. THE Context_Builder SHALL apply the same Reranker and Context_Token_Cap to vector, graph and fused paths.
4. THE Generator SHALL use one fixed, versioned prompt that instructs the model to cite passage numbers and to answer with the Refusal_String when the context lacks the answer.
5. WHEN the Generator returns an answer, THE Generator SHALL parse cited passage numbers and map the numbers to chunk IDs in the Run_Record.
6. IF the Generator cites a passage number not present in the context, THEN THE Generator SHALL record the citation as invalid in the Run_Record.

### Requirement 14: Vector variants V0 to V4 (Phase 2)

**User Story:** As a researcher, I want the five vector variants runnable by name, so that I can answer RQ1.

#### Acceptance Criteria

1. WHEN variant V0 is run, THE Elevate_RAG SHALL generate answers with no retrieved context to measure contamination.
2. WHEN variants V1, V2, V3 or V4 are run, THE Elevate_RAG SHALL use BM25, dense, hybrid RRF, or hybrid RRF plus Reranker retrieval respectively.
3. WHEN V4 hyperparameters are tuned, THE Elevate_RAG SHALL tune only on the dev split and write the frozen V4 configuration to a versioned file (gate G2).
4. WHEN V0 accuracy on a dataset exceeds a configured contamination threshold, THE Evaluation_Harness SHALL flag the dataset as contamination-suspect in the Run summary.
5. WHEN the user runs the Phase 2 Exit_Check, THE CLI SHALL run V0 to V4 on the Tiny_Sample with FAISS and print recall and accuracy per variant.

### Requirement 15: Entity and relation extraction (Phase 3)

**User Story:** As a researcher, I want LLM extraction of entities and relations from the shared chunks with provenance, so that the graph is grounded in the same text as the vector index.

#### Acceptance Criteria

1. WHEN a chunk is processed, THE Extractor SHALL produce entities as (name, type, description) and relations as (source, relation, target, description, weight).
2. THE Extractor SHALL attach the source `chunk_id` to every extracted entity and relation.
3. FOR ALL entities and relations in the graph, every referenced source `chunk_id` SHALL exist in the Chunk_Store (provenance invariant).
4. THE Extractor SHALL cache extraction outputs by chunk content hash, extraction prompt version and model identifier.
5. IF the extraction output for a chunk is malformed after the LLM_Gateway repair step, THEN THE Extractor SHALL record the chunk as failed, continue with remaining chunks, and report the failure count.
6. WHEN an extraction Run completes, THE Extractor SHALL report total indexing tokens, wall-clock time and cost.

### Requirement 16: Entity resolution and graph construction (Phase 3)

**User Story:** As a researcher, I want duplicate entities merged and the graph dual-indexed, so that graph retrieval finds the right nodes.

#### Acceptance Criteria

1. THE Entity_Resolver SHALL merge entities with identical normalised names, then merge entities whose profile embedding cosine similarity exceeds the Merge_Threshold.
2. FOR ALL entity sets, running the Entity_Resolver on an already resolved entity set SHALL produce the same entity set (idempotence property).
3. FOR ALL entity sets, the number of entities after resolution SHALL be at most the number before resolution (metamorphic property).
4. WHEN an entity accumulates more descriptions than the configured merge limit, THE Graph_Builder SHALL summarise the descriptions with the extraction LLM into one description.
5. THE Graph_Builder SHALL embed entity profiles and relation descriptions into the Vector_Store and write nodes and edges to the Graph_Store.
6. WHERE the community layer is enabled, THE Community_Builder SHALL run Leiden community detection and generate one LLM report per community.

### Requirement 17: Graph retrieval modes (Phase 3)

**User Story:** As a researcher, I want LightRAG naive, local, global, hybrid and mix modes over the shared index, so that I can compare graph and vector paths (RQ2).

#### Acceptance Criteria

1. WHEN a query is submitted in local mode, THE Graph_Retriever SHALL retrieve entities matching low-level keywords, their 1-hop neighbours and their source chunks.
2. WHEN a query is submitted in global mode, THE Graph_Retriever SHALL retrieve relations matching high-level keywords and their source chunks.
3. WHEN a query is submitted in hybrid mode, THE Graph_Retriever SHALL combine local-mode and global-mode results.
4. WHEN a query is submitted in mix mode, THE Graph_Retriever SHALL combine graph results with vector chunk results and pass the combined set through the Reranker.
5. WHEN a query is submitted in naive mode, THE Graph_Retriever SHALL perform dense-only chunk retrieval.
6. WHERE graph depth is configured to 2, THE Graph_Retriever SHALL include 2-hop neighbours in local mode.
7. THE Graph_Retriever SHALL record the chunk IDs, entity IDs and relation IDs used in each context in the Run_Record.
8. WHEN the user runs the Phase 3 Exit_Check, THE CLI SHALL build a graph on the Tiny_Sample with NetworkX, run all five modes and print accuracy, recall and context tokens per mode.

### Requirement 18: Graph quality checks (Phase 3)

**User Story:** As a researcher, I want the graph quality checked before QA, so that poor extraction does not invalidate graph results.

#### Acceptance Criteria

1. THE Graph_Quality_Checker SHALL compute Answer_Entity_Coverage for each dataset graph.
2. THE Graph_Quality_Checker SHALL compute node count, edge count, average degree, clustering coefficient and duplicate-entity rate.
3. IF Answer_Entity_Coverage is below 60% after tuning, THEN THE Graph_Quality_Checker SHALL mark the graph as failing gate G3 and report the rebuild command using the Fallback_Extractor.
4. WHEN the Fallback_Extractor is selected, THE Graph_Builder SHALL rebuild the graph with the LlamaIndex SchemaLLMPathExtractor and the fixed schema.
5. WHEN a triple audit export is requested, THE Annotation_Tool SHALL export 50 randomly sampled triples with source chunk text using a fixed seed.

### Requirement 19: Reference and alternative graph systems (Phase 3)

**User Story:** As a researcher, I want Microsoft GraphRAG, HippoRAG 2 and the alternative frameworks runnable through the same harness, so that LightRAG results are put in context.

#### Acceptance Criteria

1. WHERE a reference system (Microsoft GraphRAG or HippoRAG 2) is configured, THE Elevate_RAG SHALL run the reference system through an adapter that returns answers and retrieved passages in the Run_Record format.
2. WHERE an alternative framework (LlamaIndex PropertyGraphIndex or Neo4j GraphRAG package) is configured, THE Elevate_RAG SHALL run the framework through the same adapter interface over the shared Chunk_Store.
3. THE Evaluation_Harness SHALL score reference and alternative system Runs with the same metrics as LightRAG Runs.

### Requirement 20: Static and rule-based routing (Phase 4)

**User Story:** As a researcher, I want static and rule-based routers, so that I have simple routing baselines.

#### Acceptance Criteria

1. WHERE Router_Level R0 is selected, THE Router SHALL send every query to the configured single Retrieval_Mode (naive, local, global or mix).
2. WHERE Router_Level R1 is selected, THE Router SHALL select a Retrieval_Mode from keyword rules ("compare", "both", "overall", "main themes"), named-entity count and question word, defined in a configuration file.
3. FOR ALL queries, the Router SHALL return exactly one Retrieval_Mode from the configured set (totality invariant).
4. THE Router SHALL record the selected mode, Router_Level and decision reason for every query in the Run_Record.

### Requirement 21: Learned and LLM routing (Phase 4)

**User Story:** As a researcher, I want a lightweight classifier and an LLM classifier router, so that I can test whether learning improves routing.

#### Acceptance Criteria

1. WHERE Router_Level R2 is selected, THE Router SHALL predict a Query_Type or target mode with a TF-IDF plus SVM classifier or a configured small encoder trained on the dev or training split.
2. THE Router SHALL support training R2 on query-type labels from datasets and on outcome labels (cheapest correct mode on the training split).
3. WHEN R2 is evaluated across datasets, THE Router SHALL support leave-one-dataset-out evaluation.
4. WHEN RAGRouter-Bench is used, THE Router SHALL support 5-fold cross-validation.
5. WHERE Router_Level R3 is selected, THE Router SHALL classify each query as fact, multi-hop, comparison or global with zero-shot or few-shot prompting through the LLM_Gateway and map the class to a mode.
6. FOR ALL trained R2 models, saving then loading the model SHALL produce identical predictions on the same queries (round-trip property).

### Requirement 22: Confidence fallback routing (Phase 4)

**User Story:** As a researcher, I want a cheap-first router that escalates when unsure, so that I can trade cost for accuracy.

#### Acceptance Criteria

1. WHERE Router_Level R4 is selected, THE Router SHALL run the configured cheap mode first.
2. WHEN the top Reranker score of the cheap-mode result is below the dev-tuned threshold, THE Router SHALL escalate the query to mix mode.
3. WHEN the cheap-mode answer equals the Refusal_String, THE Router SHALL escalate the query to mix mode.
4. THE Router SHALL perform at most one escalation per query.
5. THE Router SHALL record the tokens and cost of both the cheap and escalated attempts in the Run_Record.

### Requirement 23: Oracle and router evaluation (Phase 4)

**User Story:** As a researcher, I want router decisions scored against an oracle, so that I can answer RQ3.

#### Acceptance Criteria

1. WHEN per-mode Run_Records exist for a question set, THE Oracle_Router SHALL assign each question the cheapest Retrieval_Mode that the LLM_Judge marked correct.
2. IF no mode answered a question correctly, THEN THE Oracle_Router SHALL assign the cheapest mode and mark the question as unsolvable.
3. THE Evaluation_Harness SHALL report routing accuracy, macro-F1, end-to-end accuracy, regret (oracle accuracy minus router accuracy) and cost saved relative to always-mix for each Router_Level.
4. FOR ALL router evaluations, regret SHALL be greater than or equal to zero (oracle bound invariant).
5. THE Evaluation_Harness SHALL compute router end-to-end results from cached per-mode Run_Records when every routed mode has a cached result, without new LLM calls.
6. WHEN the user runs the Phase 4 Exit_Check, THE CLI SHALL run R0 to R4 and the Oracle_Router on the Tiny_Sample and print accuracy, regret and cost per Router_Level.

### Requirement 24: Vector database micro-benchmark, Level A (Phase 5)

**User Story:** As a researcher, I want vector databases benchmarked on the project's embeddings, so that I can tell whether vector DB choice changes quality or only speed and cost (RQ4).

#### Acceptance Criteria

1. THE DB_Benchmark SHALL tune ef_search or nprobe per vector backend until recall@10 reaches 0.95 against exact FAISS search.
2. THE DB_Benchmark SHALL measure queries per second at 1, 8 and 32 concurrent clients and latency p50, p95 and p99.
3. THE DB_Benchmark SHALL measure index build time, peak RAM, disk usage, filtered search by `source`, and write-to-searchable delay.
4. THE DB_Benchmark SHALL produce recall-versus-QPS curve data per vector backend.
5. WHEN a benchmark measurement starts, THE DB_Benchmark SHALL run a 1,000-query warm-up, repeat each measurement 3 times and report the median.

### Requirement 25: Graph database micro-benchmark, Level A (Phase 5)

**User Story:** As a researcher, I want graph databases benchmarked on the MultiHop-RAG graph, so that graph DB choice is compared fairly.

#### Acceptance Criteria

1. THE DB_Benchmark SHALL measure bulk load time and memory for each graph backend on the MultiHop-RAG graph.
2. THE DB_Benchmark SHALL measure 1-hop and 2-hop fetch latency for 1,000 random entities selected with a fixed seed, and entity lookup latency by name.
3. THE DB_Benchmark SHALL measure LightRAG local-mode query latency per graph backend.
4. FOR ALL graph backends, 1-hop and 2-hop fetches for the same entity SHALL return the same neighbour set (backend equivalence property).

### Requirement 26: Database end-to-end comparison, Level B and fairness (Phase 5)

**User Story:** As a researcher, I want end-to-end runs with swapped backends under equal conditions, so that DB comparisons are fair and reproducible.

#### Acceptance Criteria

1. WHEN a Level B comparison is run, THE DB_Benchmark SHALL swap the LightRAG storage backend and re-run a fixed 300-question subset selected with a fixed seed.
2. THE DB_Benchmark SHALL report accuracy, recall, end-to-end latency, and configuration line count per backend, and SHALL accept setup hours as a user-entered value.
3. THE DB_Benchmark SHALL run every database container with the same CPU and memory limits on the same machine and the same HNSW parameters.
4. THE DB_Benchmark SHALL record container image versions, resource limits and machine specification in the benchmark record.
5. WHEN the user runs the Phase 5 Exit_Check, THE CLI SHALL run Level A on the Tiny_Sample for every backend whose health check passes, mark other backends as skipped, and print the result table.

### Requirement 27: Model comparison (Phase 6)

**User Story:** As a researcher, I want model roles varied one at a time, so that I can measure which model choices matter most (RQ5).

#### Acceptance Criteria

1. THE Experiment_Runner SHALL support varying exactly one model role (embedder, reranker, extraction LLM or generator) per comparison while holding all other roles at the reference configuration.
2. THE Experiment_Runner SHALL define the reference configuration as a strong open embedder, open reranker, cheap API extractor, mid-size open generator, Qdrant and Neo4j, with no-setup backends substituted when the Docker stacks are unavailable.
3. THE Experiment_Runner SHALL restrict extractor comparisons to MultiHop-RAG and GraphRAG-Bench medical.
4. THE Experiment_Runner SHALL support the full factorial of mode (naive, local, mix) by generator (3 models) on HotpotQA, MuSiQue and GraphRAG-Bench (E6).
5. THE Evaluation_Harness SHALL report accuracy delta per factor relative to the reference configuration and cost per 1,000 queries.

### Requirement 28: Ablations (Phase 6)

**User Story:** As a researcher, I want ablations on MultiHop-RAG, so that I can attribute effects to design choices.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL run ablation A1 over chunk sizes 256, 512 and 1,024 tokens.
2. THE Experiment_Runner SHALL run ablation A2 over at least three Merge_Threshold values.
3. THE Experiment_Runner SHALL run ablation A3 over Context_Token_Cap values of 4,000, 8,000 and 16,000 tokens.
4. THE Experiment_Runner SHALL run ablation A4 with the community layer on and off, and with graph depth 1 and 2.
5. THE Evaluation_Harness SHALL report context tokens next to accuracy for every ablation Run.

### Requirement 29: Experiment runner (Phase 6)

**User Story:** As a team lead, I want the experiment matrix defined declaratively and resumable, so that the team can scale or cut scope without code changes.

#### Acceptance Criteria

1. THE Experiment_Runner SHALL read experiments E1 to E6 and A1 to A4 from declarative configuration files listing dataset, split, pipeline, Router_Level, backends and model roles.
2. WHEN an experiment is interrupted, THE Experiment_Runner SHALL resume from the last completed query using persisted Run_Records and the Cache.
3. WHEN a dry run is requested, THE Experiment_Runner SHALL list every planned Run with estimated tokens and cost without making model calls.
4. WHEN all experiment Runs are complete, THE Experiment_Runner SHALL write a frozen run manifest listing every Run_Record path and configuration hash (gate G4).
5. WHEN the user runs the Phase 6 Exit_Check, THE CLI SHALL execute a reduced matrix of one comparison per model role and one value pair per ablation on the Tiny_Sample and print the summary.

### Requirement 30: Statistical analysis (Phase 7)

**User Story:** As a researcher, I want confidence intervals and significance tests, so that reported differences are statistically supported.

#### Acceptance Criteria

1. THE Stats_Engine SHALL compute 95% bootstrap confidence intervals with 1,000 resamples and a fixed seed for every headline metric.
2. THE Stats_Engine SHALL apply McNemar's test to paired binary outcomes and the paired bootstrap test to paired continuous outcomes.
3. THE Stats_Engine SHALL apply Holm–Bonferroni correction within each family of comparisons.
4. THE Stats_Engine SHALL compute the minimum detectable effect for a given sample size from dev-split variance.
5. FOR ALL metric samples, the bootstrap confidence interval SHALL contain the point estimate (interval invariant).
6. FOR ALL p-value families, Holm–Bonferroni adjusted p-values SHALL be greater than or equal to the raw p-values and at most 1 (correction invariant).

### Requirement 31: Error analysis (Phase 7)

**User Story:** As a researcher, I want failures sampled and tagged by cause, so that I can explain why each system fails.

#### Acceptance Criteria

1. WHEN an error analysis export is requested for a system, THE Error_Analyzer SHALL sample 50 failed questions with a fixed seed and export question, gold answer, prediction, retrieved context, route and judge rationale.
2. THE Error_Analyzer SHALL use the failure-cause tags retrieval miss, extraction miss, wrong route, context overload, generation error and judge error.
3. WHEN tagged error sheets are imported, THE Error_Analyzer SHALL aggregate failure-cause counts per system and produce failure-cause chart data.

### Requirement 32: Human annotation tooling (Phase 7)

**User Story:** As an annotator, I want export and import of annotation sheets with agreement scores, so that manual triple audits and judge validation are reliable.

#### Acceptance Criteria

1. THE Annotation_Tool SHALL export annotation sheets as CSV files with item IDs, item content, label columns per annotator and an annotation guideline reference.
2. WHEN completed sheets are imported, THE Annotation_Tool SHALL validate that every label belongs to the allowed label set and report the item IDs of invalid labels.
3. THE Annotation_Tool SHALL compute Cohen's kappa between two annotators for the triple audit and between human labels and LLM_Judge labels.
4. FOR ALL pairs of identical label sequences with more than one distinct label, Cohen's kappa SHALL equal 1 (agreement property).
5. FOR ALL annotation sheets, exporting then importing an unmodified sheet SHALL reproduce the original items (round-trip property).
6. THE Elevate_RAG SHALL provide annotation guidelines for the triple audit, judge validation and error tagging in `docs/annotation/`.

### Requirement 33: Figures and report assets (Phase 7)

**User Story:** As a report author, I want required figures and tables generated from Run_Records, so that the report is consistent with results.

#### Acceptance Criteria

1. THE Report_Builder SHALL generate the architecture diagram, accuracy by Query_Type and mode, router cost-versus-accuracy scatter, recall-versus-QPS curves, failure-cause chart and cost per 1,000 queries table.
2. THE Report_Builder SHALL generate one results table per research question (RQ1 to RQ6) with 95% confidence intervals.
3. THE Report_Builder SHALL read only from the frozen run manifest when the manifest exists.
4. THE Report_Builder SHALL export appendix material: prompts, hyperparameters, Docker Compose files, full result tables and annotation guidelines.
5. THE Elevate_RAG SHALL provide a report template with front matter, introduction, background, system design, evaluation methodology, results per RQ, analysis, limitations and conclusion sections.

### Requirement 34: Cost and indexing reporting (Phase 7)

**User Story:** As a researcher, I want the cost of graph RAG measured, so that I can answer RQ6.

#### Acceptance Criteria

1. THE Evaluation_Harness SHALL report, per dataset and pipeline, indexing tokens, indexing wall-clock time, storage size in MB, query tokens and cost per 1,000 queries.
2. THE Report_Builder SHALL produce a budget summary comparing estimated cost to actual cost per experiment.

### Requirement 35: End-to-end pipeline

**User Story:** As the project owner, I want one command that runs the full pipeline from raw data to report assets, so that I can verify the system works end to end.

#### Acceptance Criteria

1. WHEN the end-to-end command is run on the Tiny_Sample with no-setup backends, THE CLI SHALL load data, split, chunk, build the vector and graph indexes, run V4, local, mix and R4, evaluate, compute statistics and generate report assets.
2. WHEN the end-to-end command is run on a full dataset with a configured experiment file, THE CLI SHALL execute the same stages with the configured backends and models.
3. WHEN the end-to-end command completes, THE CLI SHALL print the location of the Run_Records, the metrics summary and the generated assets.
4. IF any stage of the end-to-end command fails, THEN THE CLI SHALL stop, report the failed stage and the error, and keep outputs of completed stages for resumption.

### Requirement 36: Phase reports and exit checks

**User Story:** As the project owner, I want a report and a verification command after each phase, so that I can check progress at intervals.

#### Acceptance Criteria

1. WHEN a phase is completed, THE Elevate_RAG SHALL include a Phase_Report describing implemented components, the Exit_Check command and its observed output, test results, deviations from this plan, known limitations, and items the user must address.
2. THE Elevate_RAG SHALL provide one Exit_Check per phase that completes on the Tiny_Sample within 15 minutes on a laptop CPU, excluding first-time model downloads.
3. WHEN an Exit_Check passes, THE CLI SHALL exit with status code 0 and print a pass summary.
4. IF an Exit_Check fails, THEN THE CLI SHALL exit with a non-zero status code and print the failing check and its cause.
5. WHERE an Exit_Check depends on an unavailable user-provided prerequisite, THE CLI SHALL mark the dependent check as skipped, name the prerequisite, and reference the Setup_Docs section.

### Requirement 37: Code quality and testing

**User Story:** As a maintainer, I want clean, typed and tested code, so that team members can understand and extend the system.

#### Acceptance Criteria

1. THE Elevate_RAG SHALL use type hints on all public functions and pass the configured static type checker and linter with zero errors.
2. THE Elevate_RAG SHALL include unit tests for every pure component and property-based tests for the properties stated in Requirements 4, 6, 7, 8, 9, 12, 13, 15, 16, 20, 21, 23, 25, 30 and 32.
3. THE Elevate_RAG SHALL run unit and property tests offline by replacing LLM, embedding and database calls with deterministic fakes.
4. THE Elevate_RAG SHALL provide integration tests for database backends and model providers that run only when the corresponding prerequisite is available.
5. THE Elevate_RAG SHALL provide a README describing the architecture, directory layout, phase list and the Exit_Check command per phase.
