# Progress Log

Workspace `Elevate-RAG-nova`, branch `nova`. **Nothing committed or pushed** (user instruction).

## Verified state (2026-11-26)

| Check | Command | Result |
|---|---|---|
| Install from lockfile | `uv sync --locked --extra faiss` / `uv lock --check` | consistent (199 packages, `exclude-newer 2026-11-12`) |
| Lint / format | `ruff check` / `ruff format --check` | clean (72 files incl. scripts) |
| Types | `mypy` (pydantic plugin) | clean (53 source files) |
| Tests | `pytest -q --cov` | **134 passed, 4 skipped**, 86 % line+branch coverage (live Qdrant/Neo4j tests deselected while Docker is broken; they passed on 2026-11-26) |
| DB conformance | Qdrant + Neo4j live in Docker | **passed**; Milvus, pgvector, Memgraph, PG-graph not started → skipped with reason |
| Offline demo | `elevate-rag demo` | 13 variants (V0–V4, 5 LightRAG modes, R1, R3, R4) end to end, report generated, all labelled DEMO |
| Real datasets | `elevate-rag data prepare …` | all 7 prepared from pinned revisions, checksums verified (table below) |
| Model-free measurements | `scripts/model_free_measurements.py` | `results/model_free/RESULTS.md` |

### Prepared datasets

| Dataset | Docs | Questions | Dev / Test | Notes |
|---|---:|---:|---|---|
| nq | 4,839 | 500 | 100 / 400 | gold passages + distractors |
| hotpotqa | 9,801 | 1,000 | 200 / 800 | 6 duplicate paragraphs merged, gold IDs remapped |
| musique | 8,171 | 600 | 120 / 480 | 500 answerable + 100 unanswerable; answerable twins excluded |
| multihop_rag | 609 | 2,556 | 511 / 2,045 | evidence 100 % verbatim |
| graphrag_bench_medical | 1 | 2,062 | 413 / 1,649 | evidence 0.45 % verbatim → string evidence recall withheld |
| graphrag_bench_novel | 20 | 2,010 | 401 / 1,609 | upstream ID `Novel-55f0c0e2` shared by 2 questions → kept both (`#1`, `#2`) |
| ragrouter | – | 7,727 | 1,545 / 6,182 | router-only; both evidence formats normalised |

## Defects found and fixed by testing or real data

1. LightRAG in-process state leaked between indexes (shared workspace) → one workspace per index (D-20).
2. Graph-mode runs after a vector-only build silently ran without a graph → `open()` now raises.
3. A truncated JSONL tail corrupted resumed runs → tail repair before resuming (D-22).
4. The R1 rules missed "the founder/spouse/… of X" bridges → explicit rule (D-23).
5. The central figure dropped variants beyond 8 colour slots → small multiples (D-25).
6. The Hub client stalled; a DNS outage blocked offline re-preparation → verified resumable
   downloader plus cached metadata (D-26).
7. GraphRAG-Bench paraphrased evidence made string evidence recall meaningless → measured
   `evidence_verbatim_rate`, metric withheld when below 0.9.
8. RAGRouter-Bench rejected 3,414 records with `{doc_id, text}` evidence → both shapes accepted.
9. A duplicate upstream question ID in GraphRAG-Bench novel → deterministic disambiguation.

## Experiment execution status

| Block | Software | Research run |
|---|---|---|
| G1 (end-to-end, real models) | done with $0 local models | **passed for the pipeline** (`results/g1/G1_RESULTS.md`); judge and extractor too weak for research numbers |
| E1 (V0–V4) | ready; V1 retrieval measured on 4 datasets | **not run**: needs LLM (X1), judge (X2), budget (X3) |
| E2 (modes) | ready | not run (X1–X3) |
| E3 (routers) | ready; R1 and R2 classification measured on real labels | end-to-end not run (X1–X3) |
| E4 (DBs) | exact / FAISS / NetworkX benchmarked on synthetic data | server DBs not run (X4) |
| E5 / E6 | ready (specs validate) | not run (X1–X3, X6) |
| A1–A4 | ready | not run (X1–X3) |
| Judge validation (D-36, replaces the human gates) | GroUSE calibration + two-judge κ, run records | Llama-3.1-8B calibrated: 0.78 (below 0.8); no run validated yet |

## Session 2 (2026-11-26, after owner decisions D-27)

* Docker started; reference stack (Qdrant 1.15.5 + Neo4j 5.26.12) healthy; adapters verified live.
* Four infrastructure bugs found and fixed (see KNOWN_ISSUES).
* Ollama + sentence-transformers installed; `local_free.yaml` ($0 config) written and validated;
  20-question HotpotQA distractor subset prepared (`hotpotqa_g1`, 200 paragraphs) for gate G1.
* Model downloads stopped at owner's decision (disk 98 % full); partial download removed,
  Ollama service stopped. **No local-LLM run has been executed.**

## Session 3 (2026-11-26): ready to push and deploy

* **Run identity no longer depends on storage paths.** `config_hash()` included `data_dir`,
  `cache_dir` and `index_dir`, so moving data (e.g. to an EC2 volume) changed every run id and
  completed runs would have been re-run → location fields excluded (D-29, regression-tested).
* Storage locations come from `ELEVATE_DATA_DIR` / `ELEVATE_CACHE_DIR` / `ELEVATE_INDEX_DIR` /
  `ELEVATE_RUNS_DIR` (they were documented in `.env.example` but never read).
* Database ports bind to `127.0.0.1` by default (`ELEVATE_BIND_HOST`).
* Added `Dockerfile` (multi-stage, locked deps, non-root, tiktoken vocabulary baked in so it runs
  offline), `.dockerignore`, `.python-version`, `scripts/ec2_setup.sh`, `docs/DEPLOYMENT.md`,
  Makefile `image` / `image-smoke` / `db-ps`, and a CI job that builds the image and runs the
  demo in it. The image is **not yet built locally** (disk; see KNOWN_ISSUES).
* `elevate-rag env` no longer crashes on a hung Docker daemon.
* Checks: ruff + format clean (src, tests, scripts), mypy clean, **126 passed, 4 skipped**, 86 %.

## Session 4 (2026-11-27): gate G1 with real local models

* Ollama Qwen2.5-3B (generator/extractor) + Llama-3.2-3B (judge) + bge embedder/reranker, $0.
* Graph built on 200 HotpotQA paragraphs (91 min): 1,026 entities, 355 relations, 60 % isolated.
* Six variants on 20 questions (36 min). Judge accuracy 0.45–0.70, no significant difference vs
  V4 after Holm. The 3B judge accepted clearly wrong answers → human κ check and a stronger
  judge are required. Full write-up: `results/g1/G1_RESULTS.md`.
* Five defects found and fixed (OpenMP clash, Ollama context truncation, extractor loops,
  cache-distorted latency, deprecated merge argument). Docker's VM disk returns I/O errors, so
  Qdrant/Neo4j were not used for G1 (see KNOWN_ISSUES).
* Checks: ruff + mypy clean; **128 passed, 4 skipped** (2 live-DB tests deselected while Docker
  is broken).

## Session 5 (2026-11-27 overnight): G1 with 7B models, cluster readiness, clean-up

* **G1 re-run with Qwen2.5-7B + Llama-3.1-8B** (`results/g1_7b/G1_7B_RESULTS.md`): graph 194 min,
  six variants 62 min, $0. Judge accuracy 0.50–0.65, gold answer in 55–60 % of responses (3B run:
  30–45 %), EM 0.25–0.35, no variant significantly different from V4. The 8B judge's verdicts
  agree with the judge-free check far better than the 3B judge's; the 3B "mix wins" signal was
  judge noise.
* **Defects found and fixed (all regression-tested):**
  1. run ids depended on model-server URLs, concurrency and timeouts → excluded (D-30);
  2. `extends` leaked provider-specific params across providers (Anthropic `effort` sent to
     Ollama) → child model roles with `model:` replace the parent role (D-31);
  3. judge-sample items collided across runs answering the same question in the same mode →
     keyed by run id; agreement checks the annotators labelled exactly the sample (D-34);
  4. entity resolution, duplicate counting and G3 coverage missed space-stripped names (54 % of
     7B entities) → spacing-insensitive entity key (D-35);
  5. `judge_agreement` ignored its sample file; `fig_interaction` had an unused parameter (dead
     code found by vulture).
* **New:** `--pipeline/--datasets/--limit/--shard K/N` for experiments; `scripts/run_pipeline.sh`
  (one command: build → validate → run → report); `scripts/download_models.py`;
  `configs/pipelines/local_7b.yaml`, `cluster_gpu.yaml`, `cluster_gpu_7b.yaml`;
  `configs/experiments/G1_smoke_local_7b.yaml`, `S0_cluster_smoke.yaml`;
  `infrastructure/cluster/` (SLURM jobs, vLLM start-up, Apptainer definition, `env.sh`);
  `docs/RUNNING.md`; `results/README.md`; CI shellcheck + cluster dry run.
* **Cluster path:** dry-run and stub-tested only (no SLURM or GPU here). Tier S0 is the first
  real test.
* **Human annotation packets ready:** `results/g1*/judge_sample.csv`,
  `results/g1*/triple_audit_sample.csv`.

## Session 6 (2026-11-27): README report, EC2 GPU guide, two script fixes

* README rewritten as the project report: problem statement, step-by-step work, results, what
  remains, and exact local / EC2 (GPU) / cluster run steps.
* `scripts/run_pipeline.sh` ran the experiment with the spec's own pipeline instead of the one
  the index was built with (e.g. S0 names `cluster_gpu.yaml`; running it with
  `cluster_gpu_7b.yaml` would have mismatched) → `--pipeline` passed through. Run ids of the
  committed G1 runs are unchanged (checked by dry run).
* phi-4's context limit is 16,384 tokens (its `config.json`), but the judge was started with
  `--max-model-len 32768`, which vLLM rejects → `JUDGE_MAX_MODEL_LEN=16384` in
  `infrastructure/cluster/env.sh`, used by `serve_models.sh` (stub-tested) and RUNNING.md.
* Every experiment spec validates against `cluster_gpu_7b.yaml`; E5/E6 Claude variants are
  refused at budget $0, as intended.

## Session 7 (2026-11-27): automated evaluation and traceable results (owner decision D-36)

* **Human annotation removed** (owner decision): `elevate-rag annotate …`, the triple-audit
  export, the failure-cause chart and the exported CSVs in `results/g1*/` are deleted.
* **Replacement, all $0 and local:**
  * the benchmarks' official EM / F1 are the primary metrics; paired tests run on EM and on the
    judge verdict;
  * a second judge (`models.judge2`, Granite-3.3-8B: third model family) and per-run Cohen's κ;
  * judge calibration on GroUSE (`elevate-rag judge calibrate`; MIT, pinned + SHA-256);
  * a `validated` / `unvalidated` status per run with reasons;
  * the `ragas` package (0.4.3) for faithfulness / context precision / context recall / answer
    correctness (`ragas: true` or `--ragas`);
  * a committed record per run (`results/runs/…json` + `INDEX.csv`: git commit captured at run
    start, config hash, data version, models, metrics with CIs) and optional MLflow logging
    (`--mlflow`); `elevate-rag results sync` rebuilds records and re-checks validation.
* **Verified live (Ollama, laptop):** Llama-3.1-8B calibrated on all 63 applicable GroUSE cases:
  0.78, too lenient on incomplete answers (9/9 rated `correct`). RAGAS + record + MLflow on 3
  G1 questions through the real pipeline: answers and EM identical to the original G1-7B run.
  Records written for the 12 committed G1 runs (`results/runs/`).
* **Cluster / EC2:** a third vLLM server for the second judge (`serve_models.sh`,
  `JUDGE2_*` in `env.sh`; stub-tested); the index job calibrates the judges; the report job runs
  `results sync`. `INDEX.csv` is written atomically (parallel shards).
* **Dependency fix:** `ragas` 0.4.3 fails to import with `langchain-community` 0.4 → pinned
  `<0.4` in the `ragas` extra.
* Checks: ruff + format clean, mypy clean (55 files), **140 passed** (live-DB tests deselected
  while Docker is broken).

## Session 8 (2026-11-27): audio/video ingestion v1 (D-37)

* `elevate-rag data ingest-media` turns a folder of clips into a normal dataset:
  * faster-whisper 1.2.1, pinned Whisper `small` / `large-v3`, deterministic settings;
  * timestamped transcript windows, one chunk each, plus a clip card;
  * licences required; transcripts cached;
  * time spans mapped to gold windows; word error rate against optional reference transcripts.
  Design: `docs/MEDIA_INGESTION_DESIGN.md`; how to run: RUNNING.md §7.
* **Nothing existing changed.** The `media` extra only adds packages (`faster-whisper`,
  `ctranslate2`, `av`, `onnxruntime`, `flatbuffers`), and no schema, chunker, writer or config
  hash changed. Ingestion never loads PyTorch or FAISS (tested).
* **Found while building:**
  * PyAV 19 is incompatible with faster-whisper 1.2.1 (`metadata_errors`) → `av<19` pinned;
  * a long segment was repeated as a whole extra window → overlap only when short;
  * a missing extra gave a raw import error → clear message;
  * cards showed raw FFmpeg names ("mov,mp4,…", "mp3float") → normalised;
  * reference transcripts with spelled-out numbers inflated word error rate → written form.
* **Verified live:**
  * Whisper `small` on CPU transcribed a generated sentence exactly.
  * The synthetic demo set (3 clips: WAV, MP3, MP4 video; 12 questions with exact time spans)
    ingests in 14 s: 9 documents, 12 questions, none rejected, evidence 100 % verbatim, word
    error rate 6–15 %. All errors are misspelt invented names ("Wellmora", "Donhold").
  * Re-ingesting hit the transcript cache.
* Checks: ruff + format clean, mypy clean (57 files), **153 passed**, 86 % coverage; the media
  tests also pass in a CI-like environment without the extra (backend tests skipped).

## Session 9 (2026-11-27): media experiments, vocabulary hint, media v2 (D-38)

* **Media demo, end to end** (`results/media_demo/MEDIA_DEMO_RESULTS.md`):
  * 3 synthetic clips, 12 questions, local 7B models, 4 modes.
  * Without a vocabulary, EM was 0.36–0.45, with 4 of 11 answers lost to misspelt names
    although retrieval was right.
  * Whisper `large-v3` misspelt the same names. A `vocabulary.txt` (`hotwords`) fixed all six,
    and EM rose to **0.82** in every mode.
  * The remaining errors are the 7B generator's reasoning.
* **Media v2 implemented** (`docs/MEDIA_V2_DESIGN.md`, D-38):
  * `--scenes`: colour-difference cut detection (PyAV + numpy); keyframes; captions through an
    OpenAI-compatible vision model (laptop: `granite3.2-vision:2b`; GPU: Qwen2.5-VL-7B);
    `kind: scene` documents; picture questions opt in with `"kinds": ["scene"]`.
  * `--music`: tempo, key (Krumhansl–Schmuckler), level, brightness and a speech / music /
    mixed / silence label on the clip card (librosa 0.11, extra `music`).
* **Calibration and checks:**
  * Tempo is within ±3 BPM on 90/120/150 BPM click tracks, and keys are right on A minor and
    C major progressions (confidence > 0.9).
  * The cuts in a generated video are found within ±0.5 s.
  * No OpenMP runtime is loaded on the media path; the check detects PyTorch's `libomp` as a
    positive control.
* **Live, laptop:**
  * The 2B vision model read the on-screen words in 3 of 6 demo scenes and invented people on
    plain title cards. It is the laptop fallback only.
  * Speech clips were labelled `speech` with key "unclear", as they should be.
* **Fixed:**
  * the RAGAS client closed after its event loop ended;
  * grey-level cut detection missed colour cuts;
  * a key was reported for drums-only audio.
* **Disk:** removed the 3B Ollama models and Whisper `large-v3` (about 7 GB, owner's go-ahead);
  pulled `granite3.2-vision:2b` (2.4 GB).
* **v2c tags** (`--tags`): CLAP zero-shot genre / mood / instrument in a separate process,
  gated by the librosa content label. CLAP's own gate failed on the demo (instrumental →
  "noise", talk → "music"), and ungated tags called a talk "jazz 0.88".
* **Media on the GPU:** README §6 step 10b (Whisper `large-v3` with cuBLAS 12 / cuDNN 9,
  Qwen2.5-VL-7B through vLLM, then experiments with `--datasets team_media`).
* **Run 3 (media v2, 16 questions):**
  * The key question was answered right in every mode, and tempo came out 117.5 against 120.
  * Picture questions scored 0 of 2: the 2B captioner never read those cards.
  * The extra scene documents cost one speech answer in three modes.
* **Test fix:** a tag test imported PyTorch into the pytest process, so a later FAISS test
  tripped the OpenMP guard. It now checks for the packages without importing them.
* Checks: ruff + format clean, mypy clean (60 files), **171 passed**; `uv lock --check` OK. The
  media, music and tags extras only add packages to the lockfile.
