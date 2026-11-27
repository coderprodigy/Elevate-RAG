# Start the generator and judge vLLM servers (plus the second judge unless JUDGE2_TP=0) on this
# job's GPUs and wait until they answer.
# Source it from a job script AFTER env.sh:   source infrastructure/cluster/serve_models.sh
# Exports ELEVATE_GEN_URL / ELEVATE_JUDGE_URL / ELEVATE_JUDGE2_URL (read by
# configs/pipelines/cluster_gpu.yaml) and stops the servers when the job exits. Logs: $ELEVATE_RUNS_DIR/logs/vllm-*-$SLURM_JOB_ID.log
set -euo pipefail

# Ports derived from the job id, so two jobs sharing a node never collide.
_base=$((20000 + (${SLURM_JOB_ID:-$$} % 10000) * 3))
GEN_PORT="${GEN_PORT_OVERRIDE:-$_base}"
JUDGE_PORT="${JUDGE_PORT_OVERRIDE:-$((_base + 1))}"
JUDGE2_PORT="${JUDGE2_PORT_OVERRIDE:-$((_base + 2))}"
JUDGE2_TP="${JUDGE2_TP:-0}"

# Split the allocated GPUs: the first GEN_TP for the generator, the next JUDGE_TP for the judge.
IFS=',' read -r -a _gpus <<< "${CUDA_VISIBLE_DEVICES:-0,1,2}"
_need=$((GEN_TP + JUDGE_TP + JUDGE2_TP))
if (( ${#_gpus[@]} < _need )); then
  echo "need $_need GPUs, job has ${#_gpus[@]} (${CUDA_VISIBLE_DEVICES:-unset})" >&2
  exit 1
fi
_gen_gpus="$(IFS=','; echo "${_gpus[*]:0:GEN_TP}")"
_judge_gpus="$(IFS=','; echo "${_gpus[*]:GEN_TP:JUDGE_TP}")"
_judge2_gpus="$(IFS=','; echo "${_gpus[*]:GEN_TP+JUDGE_TP:JUDGE2_TP}")"

# Model-specific flags. Qwen3.x: text-only (skip the vision encoder) + reasoning parser, per its
# model card; thinking itself is disabled per request in cluster_gpu.yaml.
GEN_EXTRA_ARGS="${GEN_EXTRA_ARGS-}"
if [[ -z "${GEN_EXTRA_ARGS}" && "$GEN_MODEL" == Qwen/Qwen3* ]]; then
  GEN_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only"
fi

_logdir="$ELEVATE_RUNS_DIR/logs"; mkdir -p "$_logdir"
_start() {  # name gpus model revision tp port max_len extra...
  local name="$1" gpus="$2" model="$3" rev="$4" tp="$5" port="$6" max_len="$7"; shift 7
  # shellcheck disable=SC2086  # VLLM_CMD and extra args are intentionally word-split
  CUDA_VISIBLE_DEVICES="$gpus" HF_HUB_OFFLINE=1 $VLLM_CMD serve "$model" \
    --revision "$rev" --served-model-name "$model" --host 127.0.0.1 --port "$port" \
    --tensor-parallel-size "$tp" --max-model-len "$max_len" --seed 0 \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" "$@" \
    > "$_logdir/vllm-$name-${SLURM_JOB_ID:-local}.log" 2>&1 &
  echo $!
}
# shellcheck disable=SC2086
GEN_PID="$(_start gen "$_gen_gpus" "$GEN_MODEL" "$GEN_REVISION" "$GEN_TP" "$GEN_PORT" "$MAX_MODEL_LEN" $GEN_EXTRA_ARGS)"
JUDGE_PID="$(_start judge "$_judge_gpus" "$JUDGE_MODEL" "$JUDGE_REVISION" "$JUDGE_TP" "$JUDGE_PORT" "$JUDGE_MAX_MODEL_LEN")"
JUDGE2_PID=""
if (( JUDGE2_TP > 0 )); then
  JUDGE2_PID="$(_start judge2 "$_judge2_gpus" "$JUDGE2_MODEL" "$JUDGE2_REVISION" "$JUDGE2_TP" \
    "$JUDGE2_PORT" "$JUDGE2_MAX_MODEL_LEN")"
fi
# shellcheck disable=SC2086  # JUDGE2_PID is empty when the second judge is disabled
trap 'kill "$GEN_PID" "$JUDGE_PID" $JUDGE2_PID 2>/dev/null || true' EXIT

export ELEVATE_GEN_URL="http://127.0.0.1:${GEN_PORT}/v1"
export ELEVATE_JUDGE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"
export ELEVATE_JUDGE2_URL="http://127.0.0.1:${JUDGE2_PORT}/v1"

_wait() {  # url pid name
  local deadline=$((SECONDS + ${VLLM_START_TIMEOUT:-2400}))
  until curl -sf -m 5 "$1/models" >/dev/null; do
    if ! kill -0 "$2" 2>/dev/null; then
      echo "vLLM $3 exited during start-up; see $_logdir/vllm-$3-${SLURM_JOB_ID:-local}.log" >&2
      tail -20 "$_logdir/vllm-$3-${SLURM_JOB_ID:-local}.log" >&2; exit 1
    fi
    (( SECONDS < deadline )) || { echo "vLLM $3 not ready after timeout" >&2; exit 1; }
    sleep 10
  done
  echo "vLLM $3 ready at $1"
}
_wait "$ELEVATE_GEN_URL" "$GEN_PID" gen
_wait "$ELEVATE_JUDGE_URL" "$JUDGE_PID" judge
if [[ -n "$JUDGE2_PID" ]]; then _wait "$ELEVATE_JUDGE2_URL" "$JUDGE2_PID" judge2; fi
