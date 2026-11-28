# Running and reproducing Elevate-RAG

This is the single guide for running the platform anywhere and for reproducing every result in
`results/`. Pick the row that matches your hardware, do the common setup, then follow that
section.

| Setup | Models | Cost | Section |
|---|---|---|---|
| Any machine, offline | deterministic stand-ins (demo only) | $0 | [1](#1-offline-demo-any-machine) |
| Laptop / desktop, no GPU server | small open models via Ollama | $0 | [2](#2-laptop-or-desktop-with-local-models-ollama) |
| One machine with NVIDIA GPU(s) | open models via vLLM | $0 | [3](#3-workstation-with-nvidia-gpus-vllm) |
| **College / HPC cluster (SLURM)** | open models via vLLM, sharded jobs | $0 | [4](#4-college--hpc-cluster-slurm) |
| Any machine + API keys | Claude / OpenAI models | paid, capped | [5](#5-api-models-paid-capped) |
| AWS EC2 | any of the above | instance cost | [DEPLOYMENT.md](DEPLOYMENT.md) |

## 0. Common setup

```bash
git clone <your-repo-url> elevate-rag && cd elevate-rag
curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh     # or: pip install uv==0.11.8
uv sync --locked --extra faiss                              # core; add --extra local-models for
                                                            # sentence-transformers (PyTorch)
cp .env.example .env                                        # only what you use; git-ignored
uv run elevate-rag env                                      # what is installed / reachable
uv run elevate-rag data prepare all                         # 7 datasets, pinned + checksummed
uv run elevate-rag data prepare hotpotqa --n-questions 20 --as hotpotqa_g1   # G1 subset
```

Storage locations default to the checkout (`data/`, `indexes/`, `runs/`); move them with
`ELEVATE_DATA_DIR`, `ELEVATE_INDEX_DIR`, `ELEVATE_CACHE_DIR`, `ELEVATE_RUNS_DIR`. Run ids do not
depend on these paths or on model-server URLs, so runs resume after moving data or machines.

Every long step is resumable: re-running the same command skips finished graph documents and
completed runs. Nothing spends money unless a config declares a non-zero price **and**
`budget.max_usd` allows it.

Every completed run writes a small, committable record to `results/runs/<experiment>/<run_id>.json`
and a line in `results/runs/INDEX.csv` (section 6), so every number can be traced to its code
commit, config, data version and models.

## Reproducing the committed results

| Result | What it is | Command | Time / hardware |
|---|---|---|---|
| `results/model_free/` | BM25 retrieval and router accuracy on real data (no LLM) | `uv run python scripts/model_free_measurements.py` | minutes, CPU only |
| `results/g1/` | Gate G1, Qwen2.5-3B + Llama-3.2-3B, 20 questions | section 2, `local_free.yaml` | ~2 h, 16 GB Apple-silicon laptop |
| `results/g1_7b/` | Gate G1, Qwen2.5-7B + Llama-3.1-8B, same 20 questions | section 2, `local_7b.yaml` | ~4.3 h, same laptop (graph 3.2 h) |
| `results/judge_calibration/` | GroUSE calibration of each judge model | `uv run elevate-rag judge calibrate configs/pipelines/local_7b.yaml` | ~5 min per 8B judge on a laptop |
| `results/media_demo/` | audio/video ingestion end to end on 3 synthetic clips, 4 variants | section 7 (demo set) | ~1 h on a laptop (RAGAS scoring dominates) |
| `results/runs/` | one record per completed run + `INDEX.csv` | written by every `experiment run`; rebuild with `uv run elevate-rag results sync` | seconds |
| `demo_output/` (not committed) | offline pipeline demo, every number labelled DEMO | `uv run elevate-rag demo` | ~20 s |

Expected differences on re-runs: local model servers are seeded (`seed: 0`) but GPU kernels are
not bit-identical across hardware, so a few answers can change; the statistics in each report
(bootstrap CIs, McNemar) show how much that matters. `local_7b.yaml` now defines a second judge,
so re-running G1-7B with it creates new run ids (the predictions come from the response cache). The `results/g1` run predates two config
fixes (Anthropic's `effort` field was inherited and sent to Ollama, which ignores it; operational
fields were part of run ids), so a re-run gets new run ids but the same model inputs.

## 1. Offline demo (any machine)

```bash
uv sync --locked --extra faiss
uv run elevate-rag demo          # synthetic corpus → index → 13 variants → demo_output/report
```

Checks the whole pipeline (LightRAG included) with stand-in models. Not research results.

## 2. Laptop or desktop with local models (Ollama)

Needs ~8 GB RAM for the 3B pair, 16 GB for the 7B/8B pair, and 5–12 GB disk for weights.

```bash
uv sync --locked --extra faiss --extra local-models --extra ragas --extra tracking
uv run python scripts/download_models.py configs/pipelines/local_7b.yaml   # embedder, reranker

# Ollama: the context window MUST be >= 8192 (the default 4096 silently truncates prompts)
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &          # macOS/Homebrew: brew services stop ollama first
ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b && ollama pull granite3.3:8b
# (granite3.3:8b is the second judge; 3B pair: qwen2.5:3b-instruct + llama3.2:3b, one judge)

# Gate G1 end to end (index + graph → G3 checks → 6 variants → report), in-process storage
scripts/run_pipeline.sh configs/pipelines/local_7b.yaml \
  configs/experiments/G1_smoke_local_7b.yaml hotpotqa_g1 results/g1_7b \
  --set vector_store.backend=exact --set lightrag.vector_storage=NanoVectorDBStorage \
  --set graph_store.backend=networkx --set lightrag.graph_storage=NetworkXStorage
```

For the 3B pair use `configs/pipelines/local_free.yaml`, `configs/experiments/G1_smoke_local.yaml`
and `results/g1`. With Docker running, drop the four `--set` flags to use Qdrant + Neo4j
(`make db-up` first).

Laptop notes (all learned the hard way, see `results/g1/G1_RESULTS.md`):
* macOS: FAISS and the PyTorch models cannot share a process (duplicate OpenMP runtime); the code
  refuses it. Use `exact` (fine up to ~100k chunks) or Qdrant.
* `run_pipeline.sh` keeps a Mac awake with `caffeinate`; keep it on power with the lid open.
* Watch for silent truncation: `grep "truncated = 1" /opt/homebrew/var/log/ollama.log`.

## 3. Workstation with NVIDIA GPU(s) (vLLM)

```bash
uv sync --locked --extra faiss --extra local-models
uv run python scripts/download_models.py configs/pipelines/cluster_gpu.yaml
python -m venv .venv-vllm && .venv-vllm/bin/pip install vllm==0.33.0   # separate env
# generator (2 GPUs) and judge (1 GPU); the served names must equal the config's model ids
CUDA_VISIBLE_DEVICES=0,1 .venv-vllm/bin/vllm serve Qwen/Qwen3.6-35B-A3B \
  --revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7 --port 8000 --tensor-parallel-size 2 \
  --max-model-len 32768 --seed 0 --reasoning-parser qwen3 --language-model-only &
CUDA_VISIBLE_DEVICES=2 .venv-vllm/bin/vllm serve microsoft/phi-4 \
  --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --port 8001 --max-model-len 16384 --seed 0 &
CUDA_VISIBLE_DEVICES=3 .venv-vllm/bin/vllm serve ibm-granite/granite-3.3-8b-instruct \
  --revision 51dd4bc2ade4059a6bd87649d68aa11e4fb2529b --port 8002 --max-model-len 16384 --seed 0 &
export ELEVATE_GEN_URL=http://127.0.0.1:8000/v1 ELEVATE_JUDGE_URL=http://127.0.0.1:8001/v1 \
       ELEVATE_JUDGE2_URL=http://127.0.0.1:8002/v1

scripts/run_pipeline.sh configs/pipelines/cluster_gpu.yaml \
  configs/experiments/S0_cluster_smoke.yaml hotpotqa_g1 results/s0
```

Then any experiment: `uv run elevate-rag experiment run configs/experiments/E2_retrieval_modes.yaml
--pipeline configs/pipelines/cluster_gpu.yaml` (build each dataset's index first with
`uv run elevate-rag index build configs/pipelines/cluster_gpu.yaml --set dataset=<name>`).

### Model sizes

| Role | Model (pinned revision in the config) | bf16 weights | Fits on |
|---|---|---|---|
| generator/extractor (default) | Qwen/Qwen3.6-35B-A3B | 71.9 GB | 2×80 GB (TP 2) or 4×48 GB (TP 4) |
| generator/extractor (small) | Qwen/Qwen2.5-7B-Instruct (`cluster_gpu_7b.yaml`) | 15.2 GB | 1×24 GB+ |
| judge | microsoft/phi-4 | 29.3 GB | 1×40 GB+ (or 2×24 GB, TP 2) |
| second judge | ibm-granite/granite-3.3-8b-instruct | 16.3 GB | 1×24 GB+ |
| embedder / reranker | Qwen3-Embedding-0.6B / bge-reranker-v2-m3 | 1.2 / 2.3 GB | any GPU, or CPU |

KV cache needs room on top of the weights; `--max-model-len 32768` for the generator is plenty
(prompts stay below 16k tokens). phi-4's context limit is 16,384 tokens, so serve it with
`--max-model-len 16384` (judge prompts are short).

## 4. College / HPC cluster (SLURM)

Everything lives in `infrastructure/cluster/`:

| File | Purpose |
|---|---|
| `env.sh` | **the only file to edit**: paths (scratch), partitions, account, models, GPUs |
| `submit.sh` | entry point: `prepare`, `smoke`, `index`, `experiment`, `status` |
| `gpu_job.sbatch` | GPU job: starts vLLM (generator + two judges), then index build + judge calibration, or one experiment shard |
| `cpu_job.sbatch` | report job: `results sync` (run records, judge validation) then the report |
| `serve_models.sh` | splits the job's GPUs, starts the three vLLM servers (`JUDGE2_TP=0`: two), waits, stops them at exit |
| `elevate-rag.def` | Apptainer image (if your cluster has no uv/Python 3.12) |

### One-time setup (login node, needs internet)

```bash
git clone <your-repo-url> elevate-rag && cd elevate-rag
nano infrastructure/cluster/env.sh          # ELEVATE_ROOT (scratch), partitions, account, GPUs
# Option A: uv venv in the checkout (no root needed)
curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh
uv sync --locked --extra faiss --extra local-models
# Option B: Apptainer image (then: export ELEVATE_SIF=$ELEVATE_ROOT/elevate-rag.sif in env.sh)
apptainer build --fakeroot "$ELEVATE_ROOT/elevate-rag.sif" infrastructure/cluster/elevate-rag.def

infrastructure/cluster/submit.sh prepare    # datasets, model weights (~120 GB), vLLM image
```

Compute nodes then run fully offline (`HF_HUB_OFFLINE=1`, `uv run --offline`).

### Scale tiers (run in order; each one validates the next)

| Tier | Command | What it proves |
|---|---|---|
| 0 smoke | `infrastructure/cluster/submit.sh smoke` | the cluster stack works: 20 questions, 6 variants, report at `$ELEVATE_ROOT/results/S0/` |
| 1 one dataset | `submit.sh index multihop_rag` then `LIMIT=200 EXP_DATASETS=multihop_rag submit.sh experiment configs/experiments/E2_retrieval_modes.yaml 3` | throughput and cost per 1k queries at moderate size |
| 2 main block | `submit.sh index nq hotpotqa musique multihop_rag graphrag_bench_medical graphrag_bench_novel`, then `submit.sh experiment configs/experiments/E1_vector_baselines.yaml 6` and `… E2_retrieval_modes.yaml 6` | RQ1 / RQ2 on full test splits |
| 3 routers, factors, ablations | E3 (labels on dev → `router train` → E3_routers), E5, E6, A_ablations; A1 needs `INDEX_SETS="chunking.strategy=paragraph" submit.sh index …` | RQ3–RQ5 |

`submit.sh experiment SPEC [SHARDS]` validates the spec on the login node first, submits a job
array of SHARDS tasks (`--shard K/N` splits the runs exactly once) and a report job that runs
when the array ends. `PIPELINE=configs/pipelines/cluster_gpu_7b.yaml` (with the matching
`GEN_MODEL`/`GEN_REVISION`/`GEN_TP=1`) uses the small generator. `DRY_RUN=1` prints the sbatch
commands without submitting.

Operational notes:
* Each array task starts its own vLLM servers (ports derived from the job id, so tasks sharing
  a node do not clash); model loading takes a few minutes per task, so prefer fewer, longer
  shards.
* Each shard gets its own SQLite cache directory (SQLite locking is unreliable across nodes).
* Queries never write to the index, so shards share one index safely.
* Logs: `$ELEVATE_ROOT/runs/logs/` (job output, vLLM logs, per-step logs).
* Copy results back: `rsync -av cluster:$ELEVATE_ROOT/results/runs/ results/runs/` (records;
  commit them) and `rsync -av cluster:$ELEVATE_ROOT/results/judge_calibration/ results/judge_calibration/`.
  Reports are in `$ELEVATE_ROOT/results/<EXP>/`.

**Status:** the cluster scripts were checked for syntax, dry-run submission (`DRY_RUN=1`) and the
vLLM start/wait/stop logic with a stub server; they have **not yet run on a real SLURM cluster
or GPU**. Tier 0 is the test: if it fails, the job log names the step.

## 5. API models (paid, capped)

`configs/pipelines/reference.yaml` is the plan's reference (Claude Haiku extractor, open
generator). It refuses to run until a judge from another model family is chosen and
`budget.max_usd` is raised. Always estimate first:

```bash
uv run elevate-rag index pilot configs/pipelines/reference.yaml --set dataset=multihop_rag
uv run elevate-rag experiment estimate configs/experiments/E2_retrieval_modes.yaml
```

The ledger checks the cap before every paid call and aborts the run when it would be exceeded.

## 6. Evaluation, judge validation and run records (all setups)

**Metrics.** The primary answer metrics are the benchmarks' official, judge-free ones: exact
match and token F1 (SQuAD normalisation), refusal accuracy on unanswerable questions, plus
retrieval recall@k / nDCG@10. Two LLM judges from different model families (`models.judge`,
`models.judge2`) grade every answer correct / partial / wrong. With `ragas: true` in the spec
(or `--ragas`), the [RAGAS](https://docs.ragas.io) package adds faithfulness, context precision,
context recall and answer correctness, scored by the judge server (self-hosted, $0 only).

**Judge validation (automatic, no annotators; D-36).** A run's judge accuracy counts as
`validated` only when

1. both judges pass calibration on [GroUSE](https://arxiv.org/abs/2409.06595), a published
   benchmark of expert-written expected verdicts (≥ 80 % of its 63 applicable cases), and
2. the two judges agree on this run's answers with Cohen's κ ≥ 0.6.

```bash
uv run elevate-rag judge calibrate PIPELINE.yaml     # once per judge model; writes results/judge_calibration/
```

`run_pipeline.sh` and the cluster index job run it automatically. Otherwise reports and records
mark judge accuracy `unvalidated` with the reasons, and EM/F1 stay the numbers to cite.

**Run records (traceability).** Every completed run writes
`results/runs/<experiment>/<run_id>.json` and updates `results/runs/INDEX.csv`:

| Field | Traces |
|---|---|
| `config_hash`, `index_key`, `variant`, `split`, `limit` | the exact experimental configuration (re-run it with the same spec + pipeline) |
| `dataset`, `dataset_version` | the prepared data (checksummed manifest) |
| `models.<role>` | every model id with its pinned revision |
| `code.git_commit`, `git_branch`, `git_dirty`, `git_from_run` | the code that produced the run (captured when the run started) |
| `metrics.*` | headline metrics, each with mean, 95 % bootstrap CI and n |
| `judge_agreement`, `judge_validation` | κ between the judges, calibration results, status and reasons |
| `efficiency`, `cost_usd`, `raw_run_dir` | latency, tokens, cost; where the raw predictions are (relative to `ELEVATE_RUNS_DIR`) |

Commit `results/runs/` and `results/judge_calibration/` after each block; raw predictions stay in
`runs/` (git-ignored). Records hold no host names or absolute paths. If `git_dirty` is true the
run used uncommitted code: commit first for fully traceable results.

```bash
uv run elevate-rag results sync                      # rebuild records from runs/ (e.g. after rsync from EC2 / the cluster)
uv run elevate-rag experiment run SPEC --mlflow      # also log to MLflow (extra `tracking`)
uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db   # browse runs, compare metrics
```

`--results-dir` / `ELEVATE_RESULTS_DIR` moves the records (default `results/`); `--no-record`
disables them.

## 7. Audio and video datasets

Turns a folder of clips into a normal dataset, so every mode, judge and metric above works on it.
The design is in [MEDIA_INGESTION_DESIGN.md](MEDIA_INGESTION_DESIGN.md).

```bash
uv sync --locked --extra faiss --extra local-models --extra media   # PyAV bundles FFmpeg: no system ffmpeg
# your clips: .wav .mp3 .m4a .flac .ogg .opus .aiff .mp4 .mov .mkv .webm, plus
#   licences.csv      file,licence,source_url,notes   (one row per clip; required)
#   questions.jsonl   {"question", "answers", "clip", "start_s", "end_s", "query_type", "evidence"?}
#                     cross-clip: "clips" + "spans"; clip details: "card": true; "answerable": false
#   vocabulary.txt    optional: names/terms in the clips, one per line (biases transcription; see below)
#   references/<clip>.txt   optional checked transcript → word error rate in the manifest
uv run elevate-rag data ingest-media clips/ --as my_media            # laptop: Whisper small (CPU, int8)
uv run elevate-rag data ingest-media clips/ --as my_media --model large-v3   # GPU (CUDA, float16)
uv run elevate-rag index build configs/pipelines/local_7b.yaml --set dataset=my_media
uv run elevate-rag ask configs/pipelines/local_7b.yaml "When was the lab founded?" --set dataset=my_media
```

* Each clip becomes timestamped transcript windows (default 60 s, `--window-s`) and one clip card
  (duration, container, codecs, tags). Answers cite e.g. `talk.wav [01:00–02:00]`.
* Transcripts are cached per file content + model + settings in `$ELEVATE_CACHE_DIR/media/`:
  re-running (e.g. with another `--window-s`) does not transcribe again.
* **Put the names in `vocabulary.txt`.** Speech models misspell names they have never heard
  (people, places, products). On the demo clips, `small` without a vocabulary misspelt all six
  invented names, and so did `large-v3`. With the names listed, `small` got all six right and
  word error rate fell from 6–15 % to 2–10 %. The vocabulary is part of the transcript cache key
  and the manifest.
* `small` and `large-v3` transcripts are different dataset versions; report numbers from one.
* The manifest (`data/processed/<name>/manifest.json`) lists every clip's SHA-256, licence,
  language, windows, dropped (likely hallucinated) segments, word error rate, and every file
  skipped.

**Music details and video scenes (v2, optional):**

```bash
uv sync --locked --extra faiss --extra local-models --extra media --extra music
# tempo, key, loudness and speech/music/mixed on every clip card (librosa)
uv run elevate-rag data ingest-media clips/ --as my_media --music
# one captioned document per video scene, from a vision model on an OpenAI-compatible server
ollama pull granite3.2-vision:2b      # laptop (2.4 GB); GPU: vLLM Qwen/Qwen2.5-VL-7B-Instruct
uv run elevate-rag data ingest-media clips/ --as my_media --scenes --captioner granite3.2-vision:2b
```

* Scenes are cut where sampled frames change colour (`--scene-threshold`, `--min-scene-s`,
  `--max-scene-s`). Each scene's middle frame is captioned (cached), and becomes a
  `kind: scene` document such as `tour.mp4 [00:12–00:25] (scene)`.
* A question about the picture rather than the speech marks its gold with
  `"kinds": ["scene"]` (default: transcript windows only).
* `--tags` (with `--music`) adds genre, mood and instrument tags to clips labelled music or mixed.
  It uses CLAP (`laion/clap-htsat-fused`, Apache-2.0, safetensors) in a **separate process**,
  because PyTorch must not share a process with the speech model. Only clips the music analysis
  labels music or mixed are tagged: on speech, a closed label list still picks a confident-looking
  genre (a talk came out "jazz, 0.88"). The tags are unvalidated on real music, so treat them as
  descriptive metadata.
* All flags are off by default; datasets built without them are unchanged.
* GPU / EC2 run with `large-v3` and Qwen2.5-VL-7B: README §6 step 10b.

**Demo set (macOS):** three synthetic clips (WAV, MP3 and an MP4 video) spoken with `say`, with
exact gold time spans, then the experiment end to end:

```bash
uv run python scripts/make_media_demo.py data/raw/media_demo     # also writes vocabulary.txt
uv run elevate-rag data ingest-media data/raw/media_demo --as media_demo --window-s 15 --max-window-s 25
# with scenes and music (needs granite3.2-vision:2b and the `music` extra):
uv run elevate-rag data ingest-media data/raw/media_demo --as media_demo_v2 --window-s 15 \
  --max-window-s 25 --music --scenes --captioner granite3.2-vision:2b \
  --questions data/raw/media_demo/questions_v2.jsonl
scripts/run_pipeline.sh configs/pipelines/local_7b.yaml configs/experiments/MEDIA_demo_local_7b.yaml \
  media_demo results/media_demo \
  --set vector_store.backend=exact --set lightrag.vector_storage=NanoVectorDBStorage \
  --set graph_store.backend=networkx --set lightrag.graph_storage=NetworkXStorage
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| segfault / "OMP: Error #15" on macOS | FAISS + PyTorch OpenMP clash | `vector_store.backend=exact` or `qdrant` |
| graph-mode answers ignore instructions | model server context < prompt | Ollama `OLLAMA_CONTEXT_LENGTH=8192`; vLLM `--max-model-len` |
| extraction takes minutes per chunk | small model in a repetition loop | `lightrag.extract_max_tokens`, `entity_extract_max_gleaning: 0` |
| run stalls overnight on a laptop | system sleep | `caffeinate -i` (done by `run_pipeline.sh`) |
| `Qdrant … Input/output error` | Docker VM disk broken (host disk filled) | restart Docker Desktop, `make db-up` |
| `graph requested but not built` | experiment before index build | build the index with the same pipeline and storage overrides |
| a variant shows latency `n/a` | its LLM calls were served from cache | run it against an empty `ELEVATE_CACHE_DIR` to measure latency |
| `environment variable … is not set` | config needs `ELEVATE_GEN_URL` etc. | start the model servers / export the URL |
| `judge_status: unvalidated` in a report | a judge is uncalibrated, failed GroUSE, or κ < 0.6 | read `judge_validation.reasons` in the run record; run `judge calibrate`; cite EM/F1 |
| `No module named 'ragas'` / `'mlflow'` | optional extras missing | `uv sync --locked --extra ragas --extra tracking` (plus your other extras) |
| `media ingestion needs the extra` | `media` extra missing | `uv sync --locked --extra media` (plus your other extras) |
| `clip(s) without a licence row` | a clip has no row in `licences.csv` | add `file,licence,source_url,notes` for it |
| names misspelt in transcripts | the model has never heard them (a larger model does not fix it) | list them in `vocabulary.txt`; check `wer_vs_reference` |
