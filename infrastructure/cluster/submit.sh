#!/usr/bin/env bash
# Elevate-RAG on a SLURM cluster: one entry point. Run from the repository root on a login node.
#
#   infrastructure/cluster/submit.sh prepare [DATASET ...]       # login node: data, models, images
#   infrastructure/cluster/submit.sh smoke                       # tier 0: S0 on 20 questions
#   infrastructure/cluster/submit.sh index DATASET [DATASET ...]  # GPU: index + graph + G3 checks
#   infrastructure/cluster/submit.sh experiment SPEC.yaml [SHARDS]    # GPU array + report job
#   infrastructure/cluster/submit.sh status                      # your queued/running jobs
#
# Environment (see env.sh): LIMIT=N and EXP_DATASETS=a,b re-scope an experiment without editing
# it; PIPELINE=... picks another pipeline config; DRY_RUN=1 prints the sbatch commands only.
# Everything is resumable: resubmitting skips finished graph documents and completed runs.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck source=infrastructure/cluster/env.sh
source infrastructure/cluster/env.sh
read -r -a ERAG_CMD <<< "$ERAG"
read -r -a ERAG_NET <<< "$ERAG_ONLINE"
read -r -a PY_NET <<< "$PY_ONLINE"
export PIPELINE="${PIPELINE:-configs/pipelines/cluster_gpu.yaml}"
ALL_DATASETS=(nq hotpotqa musique multihop_rag graphrag_bench_medical graphrag_bench_novel ragrouter)

acct=(); [[ -n "$SLURM_ACCOUNT" ]] && acct=(--account "$SLURM_ACCOUNT")
# --export=ALL: job settings travel as environment variables (values may contain commas).
gpu_opts=(--export=ALL --partition "$GPU_PARTITION" --gres "gpu:${GPUS_PER_JOB}"
          --time "$GPU_TIME" --output "$ELEVATE_RUNS_DIR/logs/%x-%A_%a.out" ${acct[@]+"${acct[@]}"})
cpu_opts=(--export=ALL --partition "$CPU_PARTITION" --time "$CPU_TIME"
          --output "$ELEVATE_RUNS_DIR/logs/%x-%j.out" ${acct[@]+"${acct[@]}"})

sb() {  # sbatch wrapper: prints the job id (or the command, with DRY_RUN=1)
  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "DRY: STEP=${STEP:-} DATASETS=${DATASETS:-} EXPERIMENT=${EXPERIMENT:-} sbatch $*" >&2
    echo "dry$RANDOM"; return
  fi
  sbatch --parsable "$@"
}
spec_id() { sed -n 's/^id: *//p' "$1" | head -1; }

cmd="${1:-}"; shift || true
case "$cmd" in
  prepare)
    (( $# )) || set -- "${ALL_DATASETS[@]}"
    echo "== datasets (pinned revisions, verified checksums) -> $ELEVATE_DATA_DIR"
    for ds in "$@"; do "${ERAG_NET[@]}" data prepare "$ds"; done
    "${ERAG_NET[@]}" data prepare hotpotqa --n-questions 20 --as hotpotqa_g1
    echo "== model weights (pinned revisions) -> $HF_HOME"
    "${PY_NET[@]}" scripts/download_models.py "$PIPELINE"
    echo "== tokenizer vocabulary -> $TIKTOKEN_CACHE_DIR"
    "${PY_NET[@]}" -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
    if [[ "$VLLM_CMD" == *vllm.sif* && ! -f "$ELEVATE_ROOT/vllm.sif" ]]; then
      echo "== vLLM image -> $ELEVATE_ROOT/vllm.sif"
      apptainer pull "$ELEVATE_ROOT/vllm.sif" docker://vllm/vllm-openai:v0.33.0
    fi
    echo "prepared. Next: $0 smoke"
    ;;
  smoke)
    j1=$(STEP=index DATASETS=hotpotqa_g1 sb "${gpu_opts[@]}" --job-name elevate-index \
         infrastructure/cluster/gpu_job.sbatch)
    j2=$(STEP=experiment EXPERIMENT=configs/experiments/S0_cluster_smoke.yaml SHARDS=1 \
         sb "${gpu_opts[@]}" --job-name elevate-S0 --dependency "afterok:$j1" \
         infrastructure/cluster/gpu_job.sbatch)
    j3=$(STEP=report EXPERIMENT_ID=S0 sb "${cpu_opts[@]}" --job-name elevate-report \
         --dependency "afterok:$j2" infrastructure/cluster/cpu_job.sbatch)
    echo "submitted: index=$j1 -> S0=$j2 -> report=$j3"
    echo "result: $ELEVATE_RESULTS_DIR/S0/REPORT.md (run records: $ELEVATE_RESULTS_DIR/runs/)"
    ;;
  index)
    (( $# )) || { echo "usage: $0 index DATASET [DATASET ...]" >&2; exit 2; }
    j=$(STEP=index DATASETS="$*" INDEX_SETS="${INDEX_SETS:-}" sb "${gpu_opts[@]}" \
         --job-name elevate-index infrastructure/cluster/gpu_job.sbatch)
    echo "submitted index job $j for: $*"
    ;;
  experiment)
    (( $# )) || { echo "usage: $0 experiment SPEC.yaml [SHARDS]" >&2; exit 2; }
    spec="$1"; shards="${2:-4}"; id="$(spec_id "$spec")"
    extra=()
    [[ -n "${LIMIT:-}" ]] && extra+=(--limit "$LIMIT")
    [[ -n "${EXP_DATASETS:-}" ]] && extra+=(--datasets "$EXP_DATASETS")
    # Fail fast on the login node: spec, pipeline, judge independence, budget.
    ELEVATE_GEN_URL=pending ELEVATE_JUDGE_URL=pending ELEVATE_JUDGE2_URL=pending "${ERAG_CMD[@]}" experiment validate \
      "$spec" --pipeline "$PIPELINE" ${extra[@]+"${extra[@]}"}
    j=$(STEP=experiment EXPERIMENT="$spec" SHARDS="$shards" sb "${gpu_opts[@]}" \
         --job-name "elevate-$id" --array "0-$((shards - 1))" infrastructure/cluster/gpu_job.sbatch)
    r=$(STEP=report EXPERIMENT_ID="$id" sb "${cpu_opts[@]}" --job-name elevate-report \
         --dependency "afterany:$j" infrastructure/cluster/cpu_job.sbatch)
    echo "submitted $id: array job $j ($shards shards) -> report $r"
    echo "(the indexes for this spec's datasets must exist: \`$0 index ...\`)"
    ;;
  status) squeue --me --format "%.12i %.24j %.10T %.10M %R" ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
