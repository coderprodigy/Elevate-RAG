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

## Reproducing the committed results

| Result | What it is | Command | Time / hardware |
|---|---|---|---|
| `results/model_free/` | BM25 retrieval and router accuracy on real data (no LLM) | `uv run python scripts/model_free_measurements.py` | minutes, CPU only |
| `results/g1/` | Gate G1, Qwen2.5-3B + Llama-3.2-3B, 20 questions | section 2, `local_free.yaml` | ~2 h, 16 GB Apple-silicon laptop |
| `results/g1_7b/` | Gate G1, Qwen2.5-7B + Llama-3.1-8B, same 20 questions | section 2, `local_7b.yaml` | ~4.3 h, same laptop (graph 3.2 h) |
| `demo_output/` (not committed) | offline pipeline demo, every number labelled DEMO | `uv run elevate-rag demo` | ~20 s |

Expected differences on re-runs: local model servers are seeded (`seed: 0`) but GPU kernels are
not bit-identical across hardware, so a few answers can change; the statistics in each report
(bootstrap CIs, McNemar) show how much that matters. The `results/g1` run predates two config
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
uv sync --locked --extra faiss --extra local-models
uv run python scripts/download_models.py configs/pipelines/local_7b.yaml   # embedder, reranker

# Ollama: the context window MUST be >= 8192 (the default 4096 silently truncates prompts)
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &          # macOS/Homebrew: brew services stop ollama first
ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b     # or qwen2.5:3b-instruct + llama3.2:3b

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
  --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --port 8001 --max-model-len 32768 --seed 0 &
export ELEVATE_GEN_URL=http://127.0.0.1:8000/v1 ELEVATE_JUDGE_URL=http://127.0.0.1:8001/v1

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
| judge | microsoft/phi-4 | 29.3 GB | 1×40 GB+ |
| embedder / reranker | Qwen3-Embedding-0.6B / bge-reranker-v2-m3 | 1.2 / 2.3 GB | any GPU, or CPU |

KV cache needs room on top of the weights; `--max-model-len 32768` is plenty (prompts stay
below 16k tokens).

## 4. College / HPC cluster (SLURM)

Everything lives in `infrastructure/cluster/`:

| File | Purpose |
|---|---|
| `env.sh` | **the only file to edit**: paths (scratch), partitions, account, models, GPUs |
| `submit.sh` | entry point: `prepare`, `smoke`, `index`, `experiment`, `status` |
| `gpu_job.sbatch` | GPU job: starts vLLM (generator + judge), then index build or one experiment shard |
| `cpu_job.sbatch` | report job |
| `serve_models.sh` | splits the job's GPUs, starts both vLLM servers, waits, stops them at exit |
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

infrastructure/cluster/submit.sh prepare    # datasets, model weights (~105 GB), vLLM image
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
* Copy results back: `rsync -av cluster:$ELEVATE_ROOT/results/ results/cluster/`.

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

## 6. Human validation (all setups)

```bash
uv run elevate-rag annotate judge-sample runs/<EXP>/<run_dir> results/annotation/judge_sample.csv
# two people fill in the label columns independently, then:
uv run elevate-rag annotate judge-agreement results/annotation/judge_sample.csv a.csv b.csv runs/<EXP>/<run_dir>
# triples: each annotator fills the `label` column of a copy of triple_audit_sample.csv
uv run elevate-rag annotate triple-agreement triples_a.csv triples_b.csv
uv run elevate-rag annotate failures runs/<EXP>/<run_dir> results/annotation/failures.csv
```

No judge-based number should be reported until judge-vs-human κ ≥ 0.6.

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
