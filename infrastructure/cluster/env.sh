# Site settings for running Elevate-RAG on a SLURM cluster. Sourced by every job script.
# Edit this file (or export the variables before `submit.sh`) for your cluster; nothing else
# should need changing. See docs/RUNNING.md, "College / HPC cluster".

# --- where things live (use scratch / project storage, not $HOME, for data and models) --------
export ELEVATE_ROOT="${ELEVATE_ROOT:-${SCRATCH:-$HOME/scratch}/elevate-rag}"
export ELEVATE_DATA_DIR="${ELEVATE_DATA_DIR:-$ELEVATE_ROOT/data}"
export ELEVATE_INDEX_DIR="${ELEVATE_INDEX_DIR:-$ELEVATE_ROOT/indexes}"
export ELEVATE_RUNS_DIR="${ELEVATE_RUNS_DIR:-$ELEVATE_ROOT/runs}"
export ELEVATE_CACHE_DIR="${ELEVATE_CACHE_DIR:-$ELEVATE_ROOT/cache}"
export HF_HOME="${HF_HOME:-$ELEVATE_ROOT/hf}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$ELEVATE_ROOT/tiktoken}"
export ELEVATE_JSON_LOGS="${ELEVATE_JSON_LOGS:-1}"

# --- how to run the CLI: either a uv venv in the checkout or the Apptainer image ---------------
# venv : run `uv sync --locked --extra faiss --extra local-models` once on the login node.
# image: build infrastructure/cluster/elevate-rag.def into $ELEVATE_SIF (see docs/RUNNING.md).
export ELEVATE_SIF="${ELEVATE_SIF:-}"
if [[ -n "$ELEVATE_SIF" ]]; then
  ERAG="apptainer exec --nv --bind $ELEVATE_ROOT $ELEVATE_SIF elevate-rag"
  PY="apptainer exec --nv --bind $ELEVATE_ROOT $ELEVATE_SIF python"
  ERAG_ONLINE="$ERAG" PY_ONLINE="$PY"
else
  ERAG="uv run --offline elevate-rag"     # compute nodes: never touch the network for packages
  PY="uv run --offline python"
  ERAG_ONLINE="uv run elevate-rag" PY_ONLINE="uv run python"   # login node (prepare step)
fi
export ERAG PY ERAG_ONLINE PY_ONLINE

# --- model servers (vLLM) ---------------------------------------------------------------------
# VLLM_CMD: "vllm" from a separate venv, or the official image via Apptainer:
#   apptainer pull $ELEVATE_ROOT/vllm.sif docker://vllm/vllm-openai:v0.33.0
export VLLM_CMD="${VLLM_CMD:-apptainer exec --nv --bind $ELEVATE_ROOT $ELEVATE_ROOT/vllm.sif vllm}"
# Generator (also extractor / keywords / router) and judge, matching configs/pipelines/cluster_gpu.yaml.
export GEN_MODEL="${GEN_MODEL:-Qwen/Qwen3.6-35B-A3B}"
export GEN_REVISION="${GEN_REVISION:-345d467b117d1fd5eda9b879a5988de9ae5f47a7}"
export GEN_TP="${GEN_TP:-2}"            # tensor-parallel GPUs for the generator
export JUDGE_MODEL="${JUDGE_MODEL:-microsoft/phi-4}"
export JUDGE_REVISION="${JUDGE_REVISION:-2db69c1c3e91a05d2c64a3185acfbaf36f744e25}"
export JUDGE_TP="${JUDGE_TP:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"   # generator; Elevate-RAG prompts stay below 16k tokens
export JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"   # phi-4's context limit is 16k
export GEN_PORT="${GEN_PORT:-8000}" JUDGE_PORT="${JUDGE_PORT:-8001}"

# --- SLURM resources (passed to sbatch by submit.sh; adjust to your partitions) ----------------
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
export GPU_PARTITION="${GPU_PARTITION:-gpu}"
export CPU_PARTITION="${CPU_PARTITION:-cpu}"
export GPUS_PER_JOB="${GPUS_PER_JOB:-$((GEN_TP + JUDGE_TP))}"
export GPU_TIME="${GPU_TIME:-12:00:00}"
export CPU_TIME="${CPU_TIME:-02:00:00}"

# --- optional: site modules (uncomment what your cluster needs) --------------------------------
# module load apptainer
# module load cuda/12.4

mkdir -p "$ELEVATE_DATA_DIR" "$ELEVATE_INDEX_DIR" "$ELEVATE_RUNS_DIR/logs" "$ELEVATE_CACHE_DIR" \
  "$HF_HOME" "$TIKTOKEN_CACHE_DIR"
