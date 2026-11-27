# Elevate-RAG

A reproducible research platform for comparing **vector RAG**, **graph RAG** and **routers that
choose between them**, built from *Hybrid Vector + Graph RAG: Project Design & Evaluation Plan*
and the *Elevate-RAG Master Implementation Specification*.

> **Status (2026-11-27)**
> * **Done:** the software is built and tested (134 tests, 86 % coverage). Seven datasets are
>   prepared from pinned versions. Model-free baselines are measured. The first end-to-end gate
>   (G1) passed with free local models, at 3B and at 7B/8B size.
> * **Not done:** the research experiments E1–E6 and A1–A4. They need GPU time (EC2 or the
>   college cluster) and two human annotators to validate the LLM judge.
> * **Untested:** the EC2 and cluster scripts are ready but have **not yet run on a real GPU
>   machine**.
>
> This README is the full report: [problem](#1-problem-statement), [what was done](#2-what-has-been-done-step-by-step),
> [results](#3-results-so-far), [what remains](#4-what-still-needs-to-be-done),
> [run locally](#5-run-it-locally), [run on EC2](#6-run-the-scaled-version-on-aws-ec2-gpu),
> [college cluster](#7-run-on-the-college-cluster-slurm).

---

## 1. Problem statement

Retrieval-augmented generation (RAG) answers a question by retrieving text from a corpus and
giving it to an LLM. There are two main ways to retrieve:

* **Vector RAG** splits documents into chunks, embeds them and retrieves the chunks most similar
  to the question. It is cheap and strong on single-fact questions.
* **Graph RAG** (here [LightRAG](https://github.com/HKUDS/LightRAG)) first has an LLM extract
  entities and relations into a knowledge graph, then retrieves along that graph. It promises
  better multi-hop and "global" answers, but building the graph costs many LLM calls.

Published comparisons disagree. They often compare a weak vector baseline with a tuned graph
system, or change several things at once (different chunks, prompts, models or context sizes),
so the gain cannot be attributed to graph structure. The project asks:

> **When does graph structure pay for itself, and can a router capture that benefit
> automatically, sending each question to the cheapest retrieval mode that answers it?**

| ID | Research question | Experiment |
|---|---|---|
| RQ1 | How much of the gap to graph RAG does a **tuned** vector baseline (hybrid BM25 + dense + reranker) close? | E1, E2 |
| RQ2 | Which retrieval mode wins for which **query type** (single-hop, multi-hop, comparison, temporal, global, unanswerable)? | E2 |
| RQ3 | How accurate is a **router** (rules, classifier, LLM, fallback), and how much cost does it save against the best static pipeline? | E3 |
| RQ4 | Does the **database** choice (Qdrant, Milvus, pgvector, FAISS; Neo4j, Memgraph, …) change recall, latency or build cost? | E4 |
| RQ5 | What does each **model factor** (embedder, reranker, extractor, generator size) change in accuracy and cost per 1,000 queries? | E5, E6 |
| RQ6 | What does graph RAG **cost**: indexing tokens and time, query tokens, storage? | every run |

**Fairness rule (the core of the design):** every retrieval path uses the same chunks, the same
embeddings, the same reranker, the same context-token budget and the same answer prompt. Only
the retrieval mode changes, so differences can be attributed to it.

```
 corpus ──► chunks ──┬─► BM25 index ─────────┐
                     ├─► dense vectors ──────┤   one shared index
                     └─► LLM extraction ─► knowledge graph (LightRAG)
                                             │
 question ─► router (R0–R4) ─► retrieval mode: V0–V4 | naive | local | global | hybrid | mix
                                             │
                           shared reranker + token budget + prompt ─► generator ─► answer
                                             │
                     metrics (EM, F1, recall, LLM judge) + bootstrap CIs + paired tests
```

## 2. What has been done, step by step

### Step 1: requirements and decisions
* Every requirement from the plan and the spec was extracted with a stable ID:
  [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md), mapped to code and tests in
  [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md).
* 35 design decisions are recorded with rationale and alternatives:
  [docs/DECISIONS.md](docs/DECISIONS.md) (D-01 to D-35).
* LightRAG 1.5.7's API was verified against the installed package before being used.

### Step 2: reproducible environment
* Python 3.12 and a uv lockfile (199 packages, frozen with `exclude-newer`).
* Every model is pinned to a Hugging Face revision.
* Each run is identified by a hash of everything that changes its outputs, and every LLM call is
  cached by content hash. Re-running a command therefore skips completed work and never pays
  twice.

### Step 3: datasets
Seven datasets are downloaded from pinned revisions, checksummed and converted to one schema, with
20 % dev / 80 % test splits. Tuning uses dev only; the test split is never used for tuning.

| Dataset | Docs | Questions | Why it is included |
|---|---:|---:|---|
| Natural Questions | 4,839 | 500 | single-hop control |
| HotpotQA (distractor, hard) | 9,801 | 1,000 | two-hop bridge and comparison |
| MuSiQue | 8,171 | 600 | multi-hop built to defeat shortcuts, plus 100 unanswerable |
| MultiHop-RAG | 609 | 2,556 | news, multi-document, temporal |
| GraphRAG-Bench medical / novel | 1 / 20 | 2,062 / 2,010 | the graph-RAG benchmark |
| RAGRouter-Bench | n/a | 7,727 | router training and evaluation only |

Real-data problems were found and handled, for example duplicate paragraphs, a duplicated upstream
question ID, and evidence that is paraphrased rather than quoted (see
[docs/PROGRESS.md](docs/PROGRESS.md)).

### Step 4: vector baselines (V0–V4)
* Sentence-aware chunking: 512 tokens with 64 overlap; ablations at 256, 1,024 and paragraph.
* Five baselines:
  * V0: closed-book, no retrieval;
  * V1: BM25;
  * V2: dense;
  * V3: hybrid with reciprocal-rank fusion;
  * V4: hybrid plus a cross-encoder reranker, the reference vector baseline.
* Answers cite numbered passages and refuse with "insufficient information" when the context does
  not support an answer.

### Step 5: graph RAG (LightRAG)
* The graph is built from the same chunks. Every entity and relation keeps its source chunk.
* Entity resolution combines a normalised name, which is spacing-insensitive, with embedding
  similarity.
* All five LightRAG modes are supported: naive, local, global, hybrid and mix. An optional
  community layer (Leiden clustering plus LLM reports) is available as an ablation.
* Graph quality is checked before any QA: answer-entity coverage (gate G3, at least 60 %), degree,
  duplicate rate, build time and tokens. A triple-audit sample is exported for human checking.

### Step 6: routers (R0–R4)
| Router | How it routes |
|---|---|
| R0 | static: always the same mode |
| R1 | keyword and pattern rules |
| R2 | TF-IDF + SVM classifier, trained on dev labels |
| R3 | an LLM classifies the question |
| R4 | tries the cheap path first and escalates to `mix` when confidence is low |

The oracle (the cheapest correct mode, computed offline) gives regret.

### Step 7: databases and model providers
* Vector store adapters: exact, FAISS, Qdrant, Milvus and pgvector.
* Graph store adapters: NetworkX, Neo4j, Memgraph and PostgreSQL.
* Provider clients: Anthropic, any OpenAI-compatible server (vLLM, Ollama),
  sentence-transformers and cross-encoders. All of them have retries, timeouts and bounded
  concurrency.
* A cost ledger enforces a hard spending cap before every paid call. The cap is $0 in every
  shipped config.

### Step 8: evaluation and statistics
* Metrics:
  * retrieval: recall@k, nDCG@10, evidence recall;
  * answers: EM, token F1, LLM-judge correctness, refusal accuracy, pairwise comparisons;
  * efficiency: latency p50/p95, tokens and $ per 1,000 queries.
* Statistics: 95 % bootstrap CIs, exact McNemar with Holm correction, and effect sizes.
* Human validation tooling: judge-agreement (κ) and triple-audit samples, and failure analysis.
  No human label has ever been fabricated.

### Step 9: experiments as config
* E1–E6, A1–A4, G1 and S0 are YAML specs. Each variant changes one factor.
* Runs are resumable, cost-estimated before they start, and shardable across machines
  (`--shard K/N`).
* `--pipeline`, `--datasets` and `--limit` re-target a spec without editing it.

### Step 10: quality
* Checks: Ruff, mypy, pytest (134 passed, 4 skipped, 86 % coverage) and an offline demo of the
  whole pipeline.
* CI runs lint, types, tests, shellcheck, a cluster dry run, and a Docker image build with the
  demo inside it.

### Step 11: model-free measurements on real data
BM25 retrieval and the R1/R2 routers were measured without any LLM
([results/model_free/](results/model_free/RESULTS.md)).

### Step 12: gate G1 with free local models (3B, then 7B/8B)
The whole pipeline ran end to end on 20 HotpotQA questions with real models on a laptop, at $0
([results/g1/](results/g1/G1_RESULTS.md), [results/g1_7b/](results/g1_7b/G1_7B_RESULTS.md)).

These runs found and fixed **11 bugs** that unit tests could not have found, including:
* FAISS and PyTorch crash together on macOS;
* Ollama silently truncates prompts at 4k tokens;
* a 3B extractor gets stuck in repetition loops;
* cached calls distorted latency;
* run IDs changed when the model server moved;
* judge-sample IDs collided;
* 7B models write 54 % of entity names without spaces ("HansenClarke"), which split entities and
  broke the coverage gate.

### Step 13: scale-out
Configs and scripts now cover three settings:
* GPU servers: `cluster_gpu.yaml`, `cluster_gpu_7b.yaml`;
* a SLURM cluster: [infrastructure/cluster/](infrastructure/cluster/);
* EC2 ([§6](#6-run-the-scaled-version-on-aws-ec2-gpu)).

They use only open models and vLLM, so there are no API fees.

## 3. Results so far

**Model-free, test splits** (BM25 recall@10): NQ 0.78, HotpotQA 0.82, **MuSiQue 0.53**, and
MultiHop-RAG 0.84. MuSiQue is hardest for lexical retrieval, as the plan expected.

The routers were also measured on real data:
* **R1 rules:** 97 % on NQ but only 32–54 % on multi-hop sets.
* **R2 classifier:** 93.6 % in-domain, falling to 65.9 % when a whole domain is held out.

**Gate G1** (20 HotpotQA questions, all local, $0):

| | 3B models | 7B/8B models |
|---|---|---|
| answers containing the gold answer | 30–45 % | 55–60 % |
| judge accuracy (V4 / best variant) | 0.45 / 0.70 (mix, inflated by a weak judge) | 0.50 / 0.65 (local) |
| judge accepted wrong answers | frequent (5 of 6 inspected for mix were wrong) | rare (1 clear case) |
| knowledge graph | 1,026 entities, 355 relations | 1,746 entities, 841 relations |
| significant difference from V4 | none | none |

**Reading:** the pipeline works end to end. With 20 questions no retrieval mode is significantly
better than the vector baseline. The 3B run's "mix wins" result was judge noise. **None of these
numbers answers a research question yet**; that needs the full datasets (E1/E2) and a judge
validated by humans. Details: [results/README.md](results/README.md).

## 4. What still needs to be done

| # | Task | Needs | How |
|---|---|---|---|
| 1 | **Scale tier 0** on a GPU machine: G1's 20 questions with vLLM models | EC2 GPU or the cluster | [§6](#6-run-the-scaled-version-on-aws-ec2-gpu) step 7 / `submit.sh smoke` |
| 2 | **Human judge validation:** two people label 100 answers; κ ≥ 0.6 required before any judge score counts | two annotators | `results/g1*/judge_sample.csv`, [docs/RUNNING.md §6](docs/RUNNING.md#6-human-validation-all-setups) |
| 3 | **Triple audit (G3):** two people label sampled triples | two annotators | `results/g1*/triple_audit_sample.csv` |
| 4 | **Tier 1:** one dataset (MultiHop-RAG), 200 questions, all modes, to measure throughput and extrapolate time | GPU | §6 step 8 |
| 5 | **E1 + E2** on all six datasets (RQ1, RQ2) | GPU, ~a day per block (measure in tier 1) | §6 step 9 |
| 6 | **E3** routers: dev labels → train R2 → test (RQ3) | GPU | §6 step 10 |
| 7 | **E5, E6, A1–A4**: model factors, generator size, ablations (RQ5) | GPU; some variants name API models and need a budget | §6 step 10 |
| 8 | **E4** databases: start Milvus, pgvector, Memgraph and run the DB tests (RQ4) | Docker on EC2 | §6 step 11 |
| 9 | Failure analysis: 50 failures per system, labelled by cause | annotators | `elevate-rag annotate failures` |
| 10 | Final report: `elevate-rag report` per block, then freeze results (gate G4) | – | – |
| 11 | Housekeeping: choose the LICENSE holder; restart Docker Desktop on the laptop | owner | – |

Deliberately left out or partial (with reasons in [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)):
* the LlamaIndex extractor fallback;
* the official GraphRAG-Bench scorers;
* runners for external systems (Microsoft GraphRAG, HippoRAG 2). Their outputs plug in through
  `elevate-rag experiment external`;
* Langfuse tracing.

## 5. Run it locally

Requires [uv](https://docs.astral.sh/uv/), which installs Python 3.12 if missing.

### 5a. Offline demo (any machine, about 20 seconds, no models)
```bash
git clone <your-repo-url> elevate-rag && cd elevate-rag
uv sync --locked --extra faiss
uv run elevate-rag demo                      # synthetic corpus → index → 13 variants → report
open demo_output/report/REPORT.md            # Linux: xdg-open. Every number is labelled DEMO
```

### 5b. Gate G1 with real models on a laptop (Ollama, $0)
You need 16 GB RAM and about 12 GB of free disk for the 7B/8B pair. The 3B pair needs 8 GB RAM.
```bash
uv sync --locked --extra faiss --extra local-models
uv run elevate-rag data prepare hotpotqa --n-questions 20 --as hotpotqa_g1
uv run python scripts/download_models.py configs/pipelines/local_7b.yaml   # embedder + reranker

# Ollama's context MUST be >= 8192; the default 4096 silently truncates prompts.
# On macOS with Homebrew, run `brew services stop ollama` first.
OLLAMA_CONTEXT_LENGTH=8192 ollama serve &
ollama pull qwen2.5:7b-instruct && ollama pull llama3.1:8b

scripts/run_pipeline.sh configs/pipelines/local_7b.yaml \
  configs/experiments/G1_smoke_local_7b.yaml hotpotqa_g1 results/g1_7b \
  --set vector_store.backend=exact --set lightrag.vector_storage=NanoVectorDBStorage \
  --set graph_store.backend=networkx --set lightrag.graph_storage=NetworkXStorage
```
* It takes about 4.3 hours on a 16 GB Apple-silicon laptop. Keep it plugged in;
  `run_pipeline.sh` keeps the Mac awake.
* For the 3B pair, use `local_free.yaml`, `G1_smoke_local.yaml`, `qwen2.5:3b-instruct` and
  `llama3.2:3b`.
* On macOS keep `exact` vectors: FAISS and PyTorch cannot share a process there.

To reproduce the model-free numbers on CPU only, in minutes:
`uv run elevate-rag data prepare all && uv run python scripts/model_free_measurements.py`.

### 5c. Local workstation with NVIDIA GPUs
The steps are the same as on EC2 below, without steps 1–3.

## 6. Run the scaled version on AWS EC2 (GPU)

The research tool is a batch CLI. It needs no open ports except SSH. Models are served by vLLM on
the same instance, so there are no API fees, only the instance cost. Everything below uses
in-process storage (FAISS + NetworkX), so **Docker is not needed** except for E4.

### Step 1: choose the instance
Start with option A. Move to option B only after tiers 0–1 work.

| Option | Config | Instance (GPUs) | GPU use | Data volume |
|---|---|---|---|---|
| **A: 7B generator** | `cluster_gpu_7b.yaml` | `g5.12xlarge` (4× A10G 24 GB) or `g6.12xlarge` (4× L4 24 GB) | GPU 0: Qwen2.5-7B; GPUs 1–2: phi-4 judge (TP 2) | 200 GB gp3 |
| **B: reference open models** | `cluster_gpu.yaml` | `g6e.48xlarge` (8× L40S 48 GB) or `p4de.24xlarge` (8× A100 80 GB) | Qwen3.6-35B-A3B on 4× L40S or 2× A100, plus phi-4 on one more GPU | 300 GB gp3 |

* **Prices change.** Check the current price in your region and use the tier 1 timing to
  estimate the full run before launching big blocks.
* **Spot instances** are much cheaper and safe here: runs resume after an interruption.
* **GPU quota:** new AWS accounts usually have **zero quota for GPU instances**. Request an
  increase first: Service Quotas → EC2 → "Running On-Demand G and VT instances" for g5/g6/g6e, or
  "… P instances" for p4de. The value is in vCPUs; `g5.12xlarge` needs 48.

### Step 2: launch
* **AMI:** an AWS *Deep Learning Base GPU AMI (Ubuntu)*. It ships the NVIDIA driver.
* **Root volume:** 100 GB gp3.
* **Data volume:** an extra gp3 volume as sized above.
* **Security group:** inbound SSH (22) from your IP only.
* Read [docs/DEPLOYMENT.md §2](docs/DEPLOYMENT.md#2-security) for secrets and the security
  group.

```bash
ssh -i <key.pem> ubuntu@<public-ip>
nvidia-smi                                  # must list the GPUs; if not, the AMI has no driver
sudo apt-get update && sudo apt-get install -y tmux git curl
tmux new -s erag                            # long jobs survive a dropped SSH (re-attach: tmux a -t erag)
```

### Step 3: mount the data volume (once)
```bash
lsblk                                       # find the new, empty volume (e.g. nvme1n1)
sudo mkfs -t xfs /dev/nvme1n1               # ONLY for a new, empty volume: this erases it
sudo mkdir -p /mnt/elevate && sudo mount /dev/nvme1n1 /mnt/elevate && sudo chown "$USER" /mnt/elevate
# After a reboot: sudo mount /dev/nvme1n1 /mnt/elevate (or add it to /etc/fstab by UUID, with nofail)
```

### Step 4: code, Python environment and storage locations
```bash
git clone <your-repo-url> elevate-rag && cd elevate-rag
curl -LsSf https://astral.sh/uv/0.11.8/install.sh | sh && source "$HOME/.local/bin/env"
uv sync --locked --extra faiss --extra local-models

cat > ~/elevate.env <<'EOF'
export ELEVATE_DATA_DIR=/mnt/elevate/data ELEVATE_INDEX_DIR=/mnt/elevate/indexes
export ELEVATE_CACHE_DIR=/mnt/elevate/cache ELEVATE_RUNS_DIR=/mnt/elevate/runs
export HF_HOME=/mnt/elevate/hf
export ELEVATE_GEN_URL=http://127.0.0.1:8000/v1 ELEVATE_JUDGE_URL=http://127.0.0.1:8001/v1
EOF
source ~/elevate.env                        # repeat in every new shell
mkdir -p "$ELEVATE_RUNS_DIR/logs"
```

### Step 5: data and model weights (once)
```bash
uv run elevate-rag data prepare all                                         # 7 datasets
uv run elevate-rag data prepare hotpotqa --n-questions 20 --as hotpotqa_g1  # tier 0 subset
uv run python scripts/download_models.py configs/pipelines/cluster_gpu_7b.yaml   # option A, ≈ 48 GB
# option B: configs/pipelines/cluster_gpu.yaml (≈ 105 GB)

uv venv .venv-vllm --python 3.12                     # vLLM in its own environment
uv pip install --python .venv-vllm/bin/python vllm==0.33.0
```

### Step 6: start the model servers
Do this after every instance start. The served names must equal the model IDs in the config.

**Option A** (`cluster_gpu_7b.yaml`):
```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 nohup .venv-vllm/bin/vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 32768 --seed 0 \
  > "$ELEVATE_RUNS_DIR/logs/vllm-gen.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1,2 HF_HUB_OFFLINE=1 nohup .venv-vllm/bin/vllm serve microsoft/phi-4 \
  --revision 2db69c1c3e91a05d2c64a3185acfbaf36f744e25 --served-model-name microsoft/phi-4 \
  --host 127.0.0.1 --port 8001 --tensor-parallel-size 2 --max-model-len 16384 --seed 0 \
  > "$ELEVATE_RUNS_DIR/logs/vllm-judge.log" 2>&1 &

# wait until both answer (a few minutes); on failure read the two logs above
until curl -sf localhost:8000/v1/models >/dev/null && curl -sf localhost:8001/v1/models >/dev/null; do sleep 15; done; echo ready
uv run elevate-rag env                       # versions and what is reachable
```

**Option B** (`cluster_gpu.yaml`) changes only the generator command:
* model `Qwen/Qwen3.6-35B-A3B`, `--revision 345d467b117d1fd5eda9b879a5988de9ae5f47a7`;
* `CUDA_VISIBLE_DEVICES=0,1,2,3` with `--tensor-parallel-size 4` on L40S, or `0,1` with
  `--tensor-parallel-size 2` on A100 80 GB;
* add `--reasoning-parser qwen3 --language-model-only`.

The judge runs alone on the next GPU, without `--tensor-parallel-size`. In every command below,
use `cluster_gpu.yaml` in place of `cluster_gpu_7b.yaml`.

### Step 7: tier 0, the smoke test (20 questions, about 1 hour)
```bash
scripts/run_pipeline.sh configs/pipelines/cluster_gpu_7b.yaml \
  configs/experiments/S0_cluster_smoke.yaml hotpotqa_g1 results/s0_ec2
```
* It runs index + graph → G3 check → 6 variants → report in `results/s0_ec2/report/REPORT.md`.
* Step logs are in `$ELEVATE_RUNS_DIR/logs/`. If a step fails, the script prints the end of its
  log.
* Compare with `results/g1_7b/`; the numbers will differ somewhat, because the embedder and judge
  are different.

### Step 8: tier 1, one dataset at moderate size (measures throughput)
```bash
P=configs/pipelines/cluster_gpu_7b.yaml
uv run elevate-rag index pilot    $P --set dataset=multihop_rag           # extrapolated build time
uv run elevate-rag index build    $P --set dataset=multihop_rag
uv run elevate-rag index validate $P --set dataset=multihop_rag --audit-out results/ec2/multihop_triples.csv
uv run elevate-rag experiment run configs/experiments/E2_retrieval_modes.yaml \
  --pipeline $P --datasets multihop_rag --limit 200 --yes
uv run elevate-rag report --experiment E2 --out results/ec2/E2_tier1 --reference V4
```
Use the time per question from this tier to estimate tier 2 before you start it.

### Step 9: tier 2, the main blocks E1 + E2 (RQ1, RQ2)
```bash
P=configs/pipelines/cluster_gpu_7b.yaml
for ds in nq hotpotqa musique multihop_rag graphrag_bench_medical graphrag_bench_novel; do
  uv run elevate-rag index build    $P --set dataset=$ds
  uv run elevate-rag index validate $P --set dataset=$ds --audit-out results/ec2/${ds}_triples.csv
done
uv run elevate-rag experiment estimate configs/experiments/E1_vector_baselines.yaml --pipeline $P
uv run elevate-rag experiment run configs/experiments/E1_vector_baselines.yaml --pipeline $P --yes
uv run elevate-rag experiment run configs/experiments/E2_retrieval_modes.yaml  --pipeline $P --yes
uv run elevate-rag report --experiment E1 --out results/ec2/E1 --reference V4
uv run elevate-rag report --experiment E2 --out results/ec2/E2 --reference V4
```
Every command is resumable. After an interruption (a spot reclaim or a reboot), redo step 6, then
re-run the same command; finished graph documents and completed runs are skipped.

### Step 10: tier 3, routers, factors and ablations (RQ3, RQ5)
```bash
P=configs/pipelines/cluster_gpu_7b.yaml
DS="--dataset nq --dataset hotpotqa --dataset musique --dataset multihop_rag --dataset graphrag_bench_medical --dataset graphrag_bench_novel"
uv run elevate-rag experiment run configs/experiments/E3_labels_dev.yaml --pipeline $P --yes   # dev split only
uv run elevate-rag router train runs/routers/r2-type    --scheme type $DS
uv run elevate-rag router train runs/routers/r2-outcome --scheme outcome --runs "$ELEVATE_RUNS_DIR/E3-labels" $DS
uv run elevate-rag experiment run configs/experiments/E3_routers.yaml  --pipeline $P --yes
uv run elevate-rag experiment run configs/experiments/A_ablations.yaml --pipeline $P --yes
uv run elevate-rag experiment run configs/experiments/A1_paragraph_chunks.yaml --pipeline $P --yes
uv run elevate-rag report --experiment E3 --out results/ec2/E3 --reference R0-mix
uv run elevate-rag report --experiment A  --out results/ec2/A  --reference reference
```
* Ablations that change chunking or entity merging need their own index. A run builds a missing
  index automatically on first use, which takes as long as a normal graph build.
* **E5 and E6** compare embedders, rerankers, extractors and generators. Their Claude variants
  (`gen-sonnet`, `ext-sonnet`) are refused while `budget.max_usd` is 0:
  `experiment validate configs/experiments/E5_model_factors.yaml --pipeline $P` lists them.
  Either add an API key and an agreed budget, or run the open variants one at a time with
  `--only` (it keeps runs whose id contains the text), e.g.
  `experiment run configs/experiments/E5_model_factors.yaml --pipeline $P --only emb-bge-small --yes`.

### Step 11: E4 databases (RQ4, needs Docker)
```bash
docker --version || { bash scripts/ec2_setup.sh && newgrp docker; }   # the GPU AMI usually has Docker
cp .env.example .env && chmod 600 .env && nano .env   # replace every change-me password
make db-up PROFILE=all && make db-ps           # wait for "healthy"
uv run elevate-rag bench vector --help         # Level A micro-benchmarks
uv run elevate-rag experiment run configs/experiments/E4_databases_levelB.yaml --pipeline $P --yes
```

### Step 12: copy results back and stop paying
```bash
# on your laptop:
rsync -av ubuntu@<public-ip>:elevate-rag/results/ec2/ results/ec2/       # reports (commit these)
rsync -av ubuntu@<public-ip>:/mnt/elevate/runs/ runs/                    # raw runs (git-ignored)
```
**Stop the instance when it is idle.** EBS volumes, and so your data and runs, persist. Snapshot
the data volume as a backup.

Then do the human steps (§4, rows 2, 3 and 9) on the exported CSVs. No judge-based number should
be reported until κ ≥ 0.6.

Running on CPU-only EC2 with API models, or using the Docker image instead of uv, is covered in
[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## 7. Run on the college cluster (SLURM)

```bash
nano infrastructure/cluster/env.sh            # the only file to edit: scratch path, partitions, account
uv sync --locked --extra faiss --extra local-models      # or build the Apptainer image
infrastructure/cluster/submit.sh prepare      # login node: datasets, model weights, vLLM image
infrastructure/cluster/submit.sh smoke        # tier 0 as a GPU job (the first real test)
infrastructure/cluster/submit.sh index multihop_rag
LIMIT=200 EXP_DATASETS=multihop_rag infrastructure/cluster/submit.sh experiment configs/experiments/E2_retrieval_modes.yaml 3
infrastructure/cluster/submit.sh status
```
* `submit.sh experiment SPEC N` submits a job array of N shards and a report job.
* For 24 GB GPUs, use `PIPELINE=configs/pipelines/cluster_gpu_7b.yaml` with the matching
  `GEN_MODEL`, `GEN_REVISION` and `GEN_TP=1`.
* `DRY_RUN=1` prints the jobs without submitting them (`make cluster-dry`).
* Full guide: [docs/RUNNING.md §4](docs/RUNNING.md#4-college--hpc-cluster-slurm).

## Configuration

* `configs/pipelines/`:

  | Config | Use |
  |---|---|
  | `reference.yaml` | the plan's reference (needs an API budget and a judge choice) |
  | `cluster_gpu.yaml`, `cluster_gpu_7b.yaml` | all-open models on vLLM, for GPU servers, EC2 or the cluster |
  | `local_free.yaml`, `local_7b.yaml` | Ollama on a laptop, $0 |
  | `local_dev.yaml`, `cpu_small.yaml` | reference variants without Docker / on a laptop CPU |
  | `demo_offline.yaml` | offline stand-ins |

  A config can `extends:` another. A model role that the child defines with its own `model:`
  replaces the parent's role entirely.
* `configs/experiments/`: E1–E6, A1–A4, G1 (the laptop gate) and S0 (the GPU smoke tier).
* `--set key=value` overrides any pipeline field from the command line.
* **Environment** (`.env`, see [`.env.example`](.env.example)): API keys, database passwords,
  model-server URLs and storage locations (`ELEVATE_DATA_DIR`, `ELEVATE_INDEX_DIR`,
  `ELEVATE_CACHE_DIR`, `ELEVATE_RUNS_DIR`).
* **Run IDs** depend only on what changes outputs, not on paths, server URLs, concurrency or
  timeouts. Runs therefore resume across disks, machines and cluster nodes.

## Repository layout

```
src/elevate_rag/
  data/          dataset loaders (pinned revisions), preparation, splits, synthetic corpus
  indexing/      chunking and the shared chunk store
  retrieval/     BM25 and rank fusion
  graph/         LightRAG engine, entity resolution, graph quality, communities
  storage/       vector (exact, FAISS, Qdrant, Milvus, pgvector) and graph (NetworkX, Neo4j,
                 Memgraph, Postgres) adapters
  routing/       routers R0–R4, fallback, oracle
  generation/    context assembly (shared token budget) and the answer generator
  providers/     LLM / embedding / reranker clients, cost ledger, retries
  evaluation/    metrics, LLM judge, harness, human-annotation tooling
  experiments/   experiment specs → resumable, shardable runs; external-system predictions
  reporting/     tables, paired tests, figures
  cli/           the `elevate-rag` command
configs/         pipeline and experiment YAML
infrastructure/
  docker/        docker-compose for the databases (Qdrant, Neo4j, Milvus, pgvector, Memgraph)
  cluster/       SLURM jobs, vLLM start-up, Apptainer image definition, site settings (env.sh)
scripts/         run_pipeline.sh, download_models.py, model-free measurements, EC2 setup
tests/           unit, integration (live DBs, skipped when absent), end-to-end
docs/            requirements, architecture, decisions, progress, running, deployment
results/         committed results with write-ups; raw runs are git-ignored
```

## Development

```bash
make check        # ruff + mypy + pytest (what CI runs)
make cov          # coverage report
make db-up        # live DBs, so the integration tests run instead of skipping
make cluster-dry  # the SLURM jobs `submit.sh smoke` would submit (no cluster needed)
```

Tests are offline. The DB conformance tests run when `ELEVATE_*` URLs point at live services and
are reported as skipped otherwise.

## Documentation

| Document | Contents |
|---|---|
| [docs/RUNNING.md](docs/RUNNING.md) | every setup in detail: laptop, GPU, cluster, API, human validation, troubleshooting |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | EC2 security, the Docker image, CPU-only / API deployments |
| [results/README.md](results/README.md) | every committed result, what it shows, how it was produced |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | requirements extracted from the plan and spec |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components, invariants, data flow |
| [docs/DECISIONS.md](docs/DECISIONS.md) | decisions with rationale and alternatives |
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | phases, gates and acceptance criteria |
| [docs/REQUIREMENTS_TRACEABILITY.md](docs/REQUIREMENTS_TRACEABILITY.md) | requirement → code → tests → status |
| [docs/PROGRESS.md](docs/PROGRESS.md) | what was verified, with evidence |
| [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md) | limitations, blockers, human-dependent tasks |

Dataset licences are listed by `elevate-rag data list`. Large data, indexes, caches and runs are
git-ignored.
