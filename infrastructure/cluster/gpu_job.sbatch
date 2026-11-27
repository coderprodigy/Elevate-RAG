#!/usr/bin/env bash
# GPU job: start vLLM (generator + judge), then do one STEP. Submitted by submit.sh; resources
# (partition, GPUs, time, account) are passed on the sbatch command line from env.sh.
#
#   STEP=index       build index + graph for every dataset in $DATASETS, then G3 validation
#                    (INDEX_SETS="k=v k=v" adds --set overrides, e.g. for chunking ablations)
#   STEP=experiment  run shard $SLURM_ARRAY_TASK_ID/$SHARDS of $EXPERIMENT
#
#SBATCH --job-name=elevate-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck source=infrastructure/cluster/env.sh
source infrastructure/cluster/env.sh
export HF_HUB_OFFLINE=1   # compute nodes use the models pre-downloaded on the login node

: "${STEP:?set STEP=index or STEP=experiment}"
: "${PIPELINE:=configs/pipelines/cluster_gpu.yaml}"
read -r -a ERAG_CMD <<< "$ERAG"
echo "job ${SLURM_JOB_ID:-local} on $(hostname): STEP=$STEP PIPELINE=$PIPELINE GPUs=${CUDA_VISIBLE_DEVICES:-?}"

# shellcheck source=infrastructure/cluster/serve_models.sh
source infrastructure/cluster/serve_models.sh

case "$STEP" in
  index)
    : "${DATASETS:?set DATASETS, e.g. DATASETS='multihop_rag hotpotqa'}"
    sets=()
    for kv in ${INDEX_SETS:-}; do sets+=(--set "$kv"); done
    mkdir -p "$ELEVATE_ROOT/audit"
    for ds in $DATASETS; do
      echo "[$(date +%T)] index build: $ds ${INDEX_SETS:-}"
      "${ERAG_CMD[@]}" index build "$PIPELINE" --set "dataset=$ds" ${sets[@]+"${sets[@]}"}
      "${ERAG_CMD[@]}" index validate "$PIPELINE" --set "dataset=$ds" ${sets[@]+"${sets[@]}"} \
        --audit-out "$ELEVATE_ROOT/audit/${ds}_triples.csv"
    done
    ;;
  experiment)
    : "${EXPERIMENT:?set EXPERIMENT=configs/experiments/....yaml}"
    shard="${SLURM_ARRAY_TASK_ID:-0}/${SHARDS:-1}"
    # One SQLite cache per shard: SQLite locking is unreliable across nodes on shared storage.
    export ELEVATE_CACHE_DIR="$ELEVATE_CACHE_DIR/shard-${SLURM_ARRAY_TASK_ID:-0}"
    extra=()
    [[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
    [[ -n "${EXP_DATASETS:-}" ]] && extra+=(--datasets "$EXP_DATASETS")
    echo "[$(date +%T)] experiment $EXPERIMENT shard $shard ${extra[*]:-}"
    "${ERAG_CMD[@]}" experiment run "$EXPERIMENT" --pipeline "$PIPELINE" --shard "$shard" \
      --yes ${extra[@]+"${extra[@]}"}
    ;;
  *) echo "unknown STEP=$STEP" >&2; exit 2 ;;
esac
echo "[$(date +%T)] done"
