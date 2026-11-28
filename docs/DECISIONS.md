# Architecture & Implementation Decisions

Each entry: context → decision → alternatives considered → consequences. IDs are referenced
from code comments and the traceability matrix.

## D-01 Python 3.12, uv, `exclude-newer` date gate
- **Decision:** Python 3.12 virtualenv managed by `uv`; `pyproject.toml` sets
  `[tool.uv] exclude-newer = 2026-11-12`, so every resolved package is ≥ 2 weeks old at kickoff.
  `uv.lock` freezes exact versions.
- **Alternatives:** hand-pinning each dependency (error-prone); Python 3.14 (LightRAG supports it, but
  several scientific wheels lag).
- **Consequence:** upgrading means deliberately moving the date and re-locking.

## D-02 LightRAG 1.5.7 is the graph engine; Elevate-RAG owns the chunk store
- **Context:** the plan requires one shared index, identical chunks for every pipeline, and "Phase 1
  as its own module that reads LightRAG's chunk table".
- **Decision:** Elevate-RAG chunks documents once (`indexing/chunker.py`) and feeds those exact
  chunks to LightRAG through `LightRAG.ainsert_custom_chunks(full_text, chunks, doc_id)`.
  LightRAG's chunk ID is reproduced with its public helper
  `lightrag.utils_pipeline.make_custom_chunk_id(doc_id, text)`, giving a bijection between
  Elevate-RAG chunk IDs and LightRAG chunk IDs. Vector baselines read the same chunk store.
- **Alternatives:** (a) `ainsert` (LightRAG re-chunks with its own fixed-token splitter; chunk
  identity would differ from V1–V4); (b) re-implementing the graph pipeline natively (duplicates
  LightRAG, violates the spec).
- **Risk:** `ainsert_custom_chunks` is marked "deprecated" in a comment but is the documented,
  journaled path for caller-chunked content in 1.5.7. A regression test
  (`tests/integration/test_lightrag_adapter.py`) fails loudly if the behaviour or ID scheme changes.

## D-03 Graph retrieval through `aquery_data`, generation through Elevate-RAG
- **Decision:** graph modes call `LightRAG.aquery_data(query, QueryParam(mode=...))`, which returns
  the structured entities, relations and chunks LightRAG would have sent to its LLM. Elevate-RAG
  then applies the *shared* reranker, *shared* token budget and *shared* generation prompt.
  LightRAG's internal reranker is disabled (`enable_rerank=False`).
- **Why:** this is the only way to keep prompt, budget and reranker identical across V0–V4 and all
  LightRAG modes (plan protocol rule 1) while reusing LightRAG's retrieval logic unchanged.
- **Consequence:** LightRAG's own answer prompt is not used. E2 therefore measures *retrieval
  mode*, not LightRAG's prompt engineering — documented as a deliberate control.

## D-04 Vector baselines are implemented natively; `naive` remains LightRAG's
- **Decision:** V1 (BM25), V2 (dense), V3 (RRF k=60), V4 (V3 + cross-encoder) live in
  `retrieval/`. V2 uses the *same* cached embeddings LightRAG stores, so V2 and LightRAG `naive`
  differ only in the vector store / top-k plumbing. Both are reported.
- **BM25:** own implementation (Okapi BM25, k1=0.9, b=0.4 — Pyserini defaults) over the shared
  tokenizer; deterministic tie-break by chunk ID. Avoids an extra dependency and makes scores
  auditable.

## D-05 Storage adapters: two layers
- **Decision:** (1) Level-B experiments swap LightRAG's own storage classes
  (`QdrantVectorDBStorage`, `MilvusVectorDBStorage`, `PGVectorStorage`, `FaissVectorDBStorage`,
  `Neo4JStorage`, `MemgraphStorage`, `PGGraphStorage`, `NetworkXStorage`) by config — no
  re-implementation. (2) Level-A micro-benchmarks and V2–V4 dense retrieval use small Elevate-RAG
  adapters (`storage/vector/*`, `storage/graph/*`) so each database can be timed in isolation.
- **Fallbacks:** `exact` (numpy) vector store and `networkx` graph store need no services. A run
  whose config names Qdrant never silently falls back: connection failure is a hard error (FR-DB7).

## D-06 Token counting with tiktoken `o200k_base`
- LightRAG 1.5.7 tokenises with tiktoken for `gpt-4o-mini` (= `o200k_base`). Using the same
  encoding for chunking, budgets and cost estimates keeps all token numbers comparable.
  Provider-reported usage (when available) is recorded separately as *actual* tokens.

## D-07 Shared reranking and context budget across heterogeneous modes
- **Context:** graph modes return entity profiles and relation descriptions as well as chunks.
- **Decision:** context is a list of typed items (`chunk`, `entity`, `relation`, `community`).
  The shared reranker scores **chunk** items for every mode that has a reranker. One token budget
  (reference 8,192; ablations 4,096 / 16,384) is enforced by `generation/context.py`: KG items may
  use at most `kg_share` (default 0.5) of the budget, unused KG budget flows to chunks; items are
  never cut mid-sentence-free — an item that does not fit is truncated at a token boundary only if
  ≥ 64 tokens remain, otherwise skipped. Vector modes get the full budget for chunks.
- **Alternative:** rerank KG items with the cross-encoder too — rejected: cross-encoders are
  trained on passages, not entity profiles, and would distort LightRAG's graph ranking.

## D-08 Router candidate modes and cost order
- Router chooses among a configurable set, default `["vector", "local", "global", "mix"]` where
  `vector` = V4. The oracle's "cheapest" uses measured mean tokens per mode on the dev split;
  configured static order (`vector < local < global < hybrid < mix`) only breaks ties.

## D-09 Determinism and the Claude temperature restriction
- Current Claude models reject `temperature`, `top_p`, `top_k` (API version 2026-10-21) and do not
  allow thinking to be disabled on Opus 5.5 / Sonnet 5.5. The Anthropic provider therefore omits
  sampling parameters, uses `output_config.effort` (configurable, default `low` for
  extraction/routing) and records `sampling_controls="unsupported"` in every call record.
  Reproducibility of Claude-backed results relies on the response cache (D-10).
- OpenAI-compatible endpoints (vLLM, Ollama, llama.cpp, OpenAI) receive `temperature=0` and `seed`.

## D-10 Caching: SQLite key-value store, content-hashed keys
- One SQLite file per namespace (`embeddings`, `llm`, `rerank`, `retrieval`). Keys are SHA-256 of a
  canonical JSON of *everything* that affects the output (provider, model, revision, prompt
  version, parameters, input). Corrupt rows are detected (JSON decode / checksum), deleted and
  recomputed. Cache hits are flagged in call records so benchmarks can separate them.
- **Alternative:** diskcache / Redis — extra dependency or service for no benefit at this scale.

## D-11 No silent server-side model fallback
- Anthropic's server-side `fallbacks` would answer some requests with a *different model*, which
  breaks "log the exact model ID of every call" (plan protocol rule 2). It is **off** by default and
  configurable; the provider logs the model actually reported in each response and treats a
  `refusal` stop reason as a recorded failure.

## D-12 Dataset sources (verified 2026-11-26)
| Dataset | Source | Notes |
|---|---|---|
| NQ | `florin-hf/nq_open_gold` (dev) | NQ-open questions with gold Wikipedia passage; distractors = other questions' gold passages, deduplicated by `idx_gold_in_corpus` |
| HotpotQA | `hotpotqa/hotpot_qa` distractor / validation | dev is all `level=hard`; corpus = pooled context paragraphs of the sampled questions |
| MuSiQue | `bdsaglam/musique` `musique_full_v1.0_dev.jsonl` | contains answerable + unanswerable; corpus = pooled paragraphs |
| MultiHop-RAG | `yixuantt/MultiHopRAG` | 609 news articles, 2,556 queries; gold docs by URL, gold facts as evidence strings |
| GraphRAG-Bench | `GraphRAG-Bench/GraphRAG-Bench` | novel + medical corpora; evidence sentences as gold strings |
| RAGRouter-Bench | `Chaplain0908/RAGRouter` | JSONL, labels `single_hop / multi_hop / summary` |

## D-13 Gold evidence: document IDs and evidence strings
- Questions carry `gold_doc_ids` and/or `gold_evidence`. Recall@k / nDCG@10 use document IDs
  where they exist; evidence-string recall (normalised containment, or token-recall ≥ 0.8 for long
  facts) is used where gold is text (GraphRAG-Bench, RAGRouter-Bench, MultiHop-RAG facts).

## D-14 External reference systems are isolated
- Microsoft GraphRAG, HippoRAG 2, LlamaIndex PGI and Neo4j GraphRAG cannot share LightRAG's
  internal index. They are integrated through a *predictions contract*: an external run writes
  per-question JSONL (`question_id, answer, contexts, tokens, latency`) that the same evaluator
  scores. They are never mixed into within-LightRAG mode comparisons.

## D-15 Judge model must be chosen, never defaulted
- The proposal's generators span Claude, Gemma and Qwen. The judge must come from another
  family. No judge model ID is shipped as default; `experiments validate` fails until
  `models.judge` is set and its family differs from all generator families in the run.

## D-16 Local hardware limits (this workstation)
- Apple M-series, 16 GB RAM, ~16 GB free disk, Docker installed but daemon not running, no API
  credentials in the environment. Consequences: the 35B mid-tier generator cannot run locally;
  Qwen3-Embedding-8B is not practical; real experiments require either API credentials or a GPU
  host serving an OpenAI-compatible endpoint. The offline demo uses deterministic providers and
  is labelled as such in every output.

## D-17 Git policy for this session
- The user instructed: **do not push or commit anything**. All work stays uncommitted in the
  `nova` worktree. (The spec allowed local commits; the explicit user instruction overrides it.)

## D-18 Offline providers are real algorithms, not stubs
- `hash` embedder (feature hashing of word n-grams, L2-normalised), `lexical` reranker (BM25-style
  query-passage overlap), `extractive` generator (picks the best-supported sentence, cites it, or
  refuses when overlap is below a threshold) and `heuristic` extractor (capitalised-phrase entities
  + sentence co-occurrence relations in LightRAG's delimiter format). They are deterministic, make
  the full pipeline runnable offline, and are clearly labelled as *not* research configurations.

## D-19 LLM-graded RAG metrics implemented natively (RAGAS definitions, not the package)
- **Context:** the plan names RAGAS for context recall/precision and faithfulness.
- **Decision:** `evaluation/llm_metrics.py` implements the three RAGAS metric *definitions* on the
  project's provider stack, so every grading call is cached, budget-checked and recorded in the
  ledger. The `ragas` package (optional extra) routes calls through LangChain wrappers that bypass
  both, and its result could not be verified here without API credentials.
- **Consequence:** numbers are comparable to RAGAS in definition, not bit-identical. If the team
  must report the `ragas` package itself, run it on the saved `scored.jsonl` contexts; this is
  listed in KNOWN_ISSUES.

## D-20 One LightRAG workspace per physical index
- **Found by testing:** LightRAG keeps process-wide shared dictionaries keyed by
  `workspace:namespace`. With the default empty workspace, two indexes opened in one process
  (e.g. the experiment runner moving from dataset A to B) share document status and graph state,
  and LightRAG then silently skips "already committed" documents.
- **Decision:** `workspace = "elevate_" + hash(resolved index path)`. Regression test:
  `test_two_indexes_in_one_process_are_isolated`.

## D-21 PostgreSQL graph = `PGTableGraphStorage`
- LightRAG's `PGGraphStorage` requires the Apache AGE extension (absent from the pinned pgvector
  image). `PGTableGraphStorage` stores the graph in plain relational tables, which is exactly the
  plan's "PostgreSQL (graph tables): tests whether a dedicated graph DB is needed at all".

## D-22 Crash-safe JSONL
- A crash mid-write leaves a truncated last line without a newline; appending after it would glue
  the next row onto it and corrupt the file mid-way. `run_questions` calls
  `repair_jsonl_tail` before resuming (found by `test_run_questions_resumes_after_crash`).

## D-23 R1 rule set includes relation-of-entity chains
- "the founder/spouse/capital/director of X" is the canonical bridge pattern of 2-hop questions;
  the first rule set missed it. Added as an explicit, separately reported feature
  (`relational_of`).

## D-24 Self-hosted models declare price 0 explicitly
- `openai-compatible` endpoints are paid or free depending on who hosts them. A provider with no
  price is "unknown" (blocked when the cap is 0; cost reported as missing). Self-hosted configs set
  `price_*_per_mtok: 0.0` explicitly; set a GPU cost per Mtok if the team wants it counted.

## D-25 Figures: small multiples instead of colour-coded series for many variants
- The central RQ2 figure can have 10+ variants; colour slots run out at 8 and series would be
  silently dropped. It is drawn as one panel per query type with labelled rows, so identity never
  depends on colour.

## D-26 Offline re-preparation
- **Found in use:** a DNS outage made `data prepare` fail for datasets whose files were already
  downloaded and verified, because the Hub API was queried for checksums first.
- **Decision:** each file's size/SHA-256/blob ID at the pinned revision is cached in
  `<file>.meta.json` at first download (immutable for a pinned revision); later runs re-hash the
  file against it without network. Regression test:
  `test_download_reuses_verified_files_without_network`.

## D-27 Project owner decisions (2026-11-26)
- **Providers available:** OpenAI and Anthropic (Claude).
- **Budget: $0 — no paid calls.** `budget.max_usd` stays 0, so the cap blocks every OpenAI/Claude
  call before it is made. Only free local models may run; the reference configuration
  (Claude Haiku extraction, Qwen3.6-35B generation) therefore cannot run as specified.
- **Judge:** undecided. Recommendation when budget allows: an OpenAI model as judge with Claude +
  open models as generators (satisfies the different-family rule, D-15).
- **Annotators:** undecided (needed only for G3 audit, judge validation, failure analysis).
- **Docker:** approved. Local DB passwords were generated into a new git-ignored `.env`
  (no API keys written); per-container memory set to 2 GB because Docker Desktop has ~8 GB.

## D-28 Local models stopped for disk space (owner decision, 2026-11-26)
- Free disk fell to ~4 GB (another Docker project shares the machine); the local model plan needs
  ~6 GB. Owner chose to stop local models. `configs/pipelines/local_free.yaml` stays as the
  ready-to-run $0 configuration (Qwen2.5-3B-Instruct generator/extractor, Llama-3.2-3B judge from
  a different family, bge-small embedder, bge-reranker-v2-m3) for when ≥ 10 GB is free.

## D-29 Deployment shape and location-independent run identity (2026-11-26)

* **Decision:** deploy as a CLI Docker image next to the compose databases, with all mutable state
  on one mounted volume (`/data`) and DB ports bound to localhost; storage locations come from the
  environment and are excluded from `config_hash()`.
* **Why:** the platform is a batch research tool, not a service, so no inbound ports are needed.
  Run ids must identify the *experiment*, not where its files live, or moving data between a
  laptop and EC2 silently invalidates resumable runs.
* **Alternatives:** an HTTP API service (no requirement for one; adds an attack surface);
  keeping paths in the hash (breaks resume across machines).

## D-30 Run identity excludes operational model settings (2026-11-27)

* **Decision:** `config_hash()` (and so every run id) ignores, per model, `base_url`,
  `api_key_env`, `max_concurrency`, `timeout_s`, `max_retries`, and LightRAG's `llm_max_async` /
  `max_parallel_insert`, in addition to storage paths (D-29).
* **Why:** on a cluster every job lands on a different node, so the vLLM URL differs per
  submission; with the URL in the hash a resubmitted job would redo completed runs. None of these
  fields changes a model's output (seeded requests).
* **Alternatives:** keep them and pin a fixed hostname (impossible on SLURM); hash only an
  allow-list of fields (brittle when new output-affecting fields are added).

## D-31 `extends`: a child model role with its own `model:` replaces the parent role (2026-11-27)

* **Decision:** deep merge as before, except for `models.<role>` mappings that name a `model`,
  which replace the parent's role. Partial role overrides (no `model:`) still merge.
* **Why:** found while writing the cluster config: `local_free.yaml` inherited the reference
  extractor's Anthropic-only `effort: low`, which was sent to Ollama (ignored there, but it would
  be an unknown request field for vLLM and it polluted the model identity).
* **Consequence:** indexes and caches of `local_free.yaml` built before the fix have a different
  key; `results/g1` (3B) predates it. Model inputs were unaffected (Ollama ignores the field).

## D-32 Cluster execution: vLLM per job, sharded experiments, in-process storage (2026-11-27)

* **Decision:** SLURM jobs start their own vLLM servers (generator and judge on disjoint GPUs,
  ports derived from the job id), experiments are split with `--shard K/N` into job arrays, and
  the cluster config uses FAISS + NetworkX in-process storage. Apptainer is optional (uv venv
  works without root).
* **Why:** university clusters rarely allow Docker or long-lived services; per-job servers need
  no admin support. Queries never write to the index (verified: LightRAG files untouched during
  a full experiment), so shards share one index; SQLite caches are per shard because SQLite
  locking is unreliable across nodes.
* **Alternatives:** one long job serving all shards (simpler but no parallelism and fragile to
  time limits); Qdrant/Neo4j as Apptainer instances (possible for E4; not needed for E1–E3).
* **Status:** dry-run and stub-tested only; tier S0 is the first real test.

## D-33 All-open cluster models (2026-11-27)

* **Decision:** `cluster_gpu.yaml` uses Qwen/Qwen3.6-35B-A3B (the reference generator) for every
  generation role, thinking disabled per its model card, and microsoft/phi-4 (MIT, different
  family) as judge; `cluster_gpu_7b.yaml` swaps in Qwen/Qwen2.5-7B-Instruct for smaller GPUs.
  All revisions pinned and verified ungated on 2026-11-27; vLLM 0.33.0 (the newest release at
  least two weeks old; the card requires ≥ 0.19).
* **Why:** the owner's decision was $0 spend (D-27); the cluster is free compute. This makes the
  cluster runs a factor variant (E5) of the reference, which is how results must be labelled.

## D-34 Judge-sample items keyed by run id (2026-11-27)

* **Decision:** judge-sample item ids are `j<i>|<run_id>|<question_id>|<mode>`; agreement checks
  that the annotators labelled exactly the exported sample.
* **Why:** a sample spanning a router run and a mode run can contain the same question answered
  in the same mode by two runs; keyed without the run id, one verdict silently overwrote the
  other and the κ would compare a human label with the wrong answer's verdict.

## D-35 Spacing-insensitive entity identity (2026-11-27)

* **Decision:** entity resolution groups names by `entity_key` (camel case split, normalised,
  spaces removed) instead of the exact normalised name; graph-quality duplicate counting and the
  G3 answer-entity coverage use the same idea.
* **Why:** Qwen2.5-7B wrote 54 % of entity names without spaces in G1 ("HansenClarke"). Exact
  matching left 84 split duplicates in the graph and made G3 coverage report 2/4 instead of 4/4.
  Removing spaces after normalising keeps "iPhone" == "iphone"; splitting camel case before
  normalising keeps article stripping ("TheClickFive" == "The Click Five").
* **Risk:** rare false merges of names that differ only in spacing; embedding-stage merges keep
  the entity-type guard. The A2 ablation measures resolution's effect either way.
* **Consequence:** graphs built before this change (both G1 graphs) keep their duplicates until
  rebuilt; coverage was re-validated (7B: 4/4; 3B: 4/4, unchanged).

## D-36 Automated judge validation and traceable run records replace human annotation (owner decision, 2026-11-27)

* **Decision (project owner):** remove the human-annotation gates (100-answer judge check with
  two annotators, 50-triple audit, 50-failure cause labelling) and the tooling behind them
  (`elevate-rag annotate …`, the exported CSVs). In their place:
  1. **Primary metric: the benchmarks' official, judge-free answer metrics**, exact match and
     token F1 with SQuAD normalisation (refusal accuracy for unanswerable questions). Paired
     tests (McNemar, Holm) are reported on EM and on the judge verdict.
  2. **Two LLM judges from different families** (`models.judge`, `models.judge2`) score every
     answer; their agreement (Cohen's κ over the answers both judged with an LLM, refusal-rule
     verdicts excluded) is reported per run.
  3. **Judge calibration on GroUSE** (Muller et al., COLING 2025; MIT; pinned revision and
     SHA-256), a published meta-evaluation benchmark whose expected verdicts were written by
     experts: `elevate-rag judge calibrate`. The 63 cases that apply to a reference-based
     correctness judge are used (mapping in `evaluation/judge_calibration.py`).
  4. A run's judge accuracy is **validated** only when both judges pass calibration (≥ 80 %)
     and κ ≥ 0.6 on that run's answers; otherwise every report and record marks it
     **unvalidated** with the reasons.
  5. **RAGAS** (package `ragas` 0.4.3, Apache-2.0) adds faithfulness, context precision/recall
     and answer correctness, scored by the judge endpoint (spec `ragas: true` or `--ragas`).
     Its calls bypass the cost ledger and cache, so only a $0 self-hosted judge is accepted.
     The native implementation (D-19) stays for paid judges.
  6. **Every completed run writes a committed record** to `results/runs/<experiment>/<run_id>.json`
     and a line in `results/runs/INDEX.csv`: dataset version, variant, config hash, index key,
     every model with its revision, git commit/branch/dirty flag captured when the run started,
     headline metrics with CIs, judge agreement and validation, efficiency and cost. Records
     contain no host names or absolute paths. `elevate-rag results sync` rebuilds them from raw
     runs (e.g. copied back from a cluster) and re-checks judge validation.
  7. **MLflow** (Apache-2.0, extra `tracking`, `--mlflow` / `ELEVATE_MLFLOW=1`) logs the same
     parameters, metrics and tags plus the run's summary/manifest to a local store.
* **Why:** the team has no annotator time. Every automated evaluation framework surveyed
  (RAGAS, DeepEval, MLflow evaluate, Phoenix) itself relies on an LLM judge, so none removes
  the need to check the judge; EM/F1 need no judge, GroUSE supplies expert-written expected
  verdicts, and cross-family agreement detects judges that disagree on this run's data. Hosted
  platforms (LangSmith, Braintrust) were rejected: paid, and every question, answer and context
  would leave the project.
* **What is lost (stated in the report):** no human check of answers *on our datasets*; two
  LLM judges can share a blind spot; GroUSE is small (63 applicable cases) and English-only; the
  triple audit (FR-G9) and failure-cause labelling (FR-E11) are no longer done. Graph quality is
  still checked automatically (G3 coverage, shape, duplicate rate).
* **First measurement:** Llama-3.1-8B (the laptop judge) passes 49/63 GroUSE cases (0.78,
  below 0.8): 35/36 complete answers rated `correct`, but all 9 partially complete answers
  were also rated `correct` and 4 of 18 answers containing none of the reference were rated
  `partial`: it is too lenient, as G1 suspected (`results/judge_calibration/`).
* **Second judge:** IBM Granite-3.3-8B-Instruct (ungated, Apache-2.0; revision pinned in
  `cluster_gpu.yaml`; `granite3.3:8b` on Ollama), a family different from the Qwen generators
  and the phi/Llama judges.
* **Run ids:** an unset `judge2` is left out of the config hash, so runs from before this change
  keep their ids; configs that now define `judge2` (`local_7b`, `cluster_gpu*`) get new ids
  because their evaluation definition changed.

## D-37 Audio/video ingestion as a dataset loader (2026-11-27)

* **Decision:** clips become an ordinary dataset (`elevate-rag data ingest-media`,
  docs/MEDIA_INGESTION_DESIGN.md). Each clip produces timestamped transcript windows, sized to
  stay one chunk, plus a clip card. faster-whisper 1.2.1 transcribes with a pinned CTranslate2
  Whisper model (`small` on laptops, `large-v3` on GPUs) and deterministic settings.
* **Why:**
  * Retrieval, both judges, RAGAS and the run records need no change.
  * A retrieved chunk maps to exactly one time span, so answers can cite clip and time.
  * faster-whisper needs no system ffmpeg (PyAV bundles it), runs on CPU and CUDA, and is MIT;
    the models are ungated and MIT.
* **No branch:** the change is additive. The `media` extra only adds packages (checked on a copy
  of the lockfile), and no schema, chunker, writer or config hash changes.
* **Pins:** `av<19` (PyAV 19 removed `metadata_errors`, which faster-whisper 1.2.1 passes).
* **Isolation:** ingestion never imports PyTorch or FAISS (a test asserts it), so the macOS
  OpenMP clash (`utils.guard_openmp`) cannot gain a third member.
* **Licences:** a `licences.csv` row is required for every clip and recorded in the manifest;
  clips are never committed.

## D-38 Media v2: scene captions and music details (2026-11-27)

* **Decision:** `ingest-media --scenes` detects scene cuts by colour frame difference (PyAV +
  numpy), captions each scene's middle frame through an OpenAI-compatible vision model, and adds
  `kind: scene` documents. `--music` adds tempo, key (Krumhansl–Schmuckler on librosa chroma),
  level, brightness and a speech / music / mixed / silence label to every clip card. Both are off
  by default.
* **Why:** the picture and the music become searchable with no change downstream. Gold for
  picture questions is opt-in (`"kinds": ["scene"]`), so v1 question files keep their meaning.
* **Captioner outside the config and the provider stack:** images are list-valued message
  content, which the shared `BaseLLM` (token counting on strings) does not handle. A small
  dedicated client with the existing response cache avoids changing shared code (the v1 branch
  rule). Consequence: no cost ledger, so local unpriced servers only.
* **Models:** laptop `granite3.2-vision:2b` (Apache-2.0, Ollama); GPU Qwen2.5-VL-7B-Instruct
  (Apache-2.0, vLLM). Qwen2.5-VL-3B was rejected: its card states no Apache licence.
* **Dependencies:** extra `music = ["librosa>=0.11,<1"]` (librosa 1.x needs Python 3.12; the
  project supports 3.11), additions only. Numba is forced to its `workqueue` threading layer.
* **Tags (v2c):** `--tags` uses CLAP (`laion/clap-htsat-fused` @ `365dea6e…`, Apache-2.0;
  the only CLAP release with safetensors weights) in a separate process, gated by the librosa
  content label. CLAP's own speech/music judgement was unreliable on the demo audio. The tags
  are unvalidated on real music and are descriptive metadata only.
