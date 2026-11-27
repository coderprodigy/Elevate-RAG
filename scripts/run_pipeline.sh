#!/usr/bin/env bash
# One command for a full block: index + graph build -> graph checks (G3) -> judge calibration
# (GroUSE) -> experiment (each run writes its record to results/runs/) -> report.
# Works on a laptop, a workstation or inside a cluster job (see docs/RUNNING.md).
#
#   scripts/run_pipeline.sh PIPELINE.yaml EXPERIMENT.yaml DATASET OUT_DIR [--set k=v ...]
#
# Example (gate G1 with mid-size local models, no Docker):
#   scripts/run_pipeline.sh configs/pipelines/local_7b.yaml \
#     configs/experiments/G1_smoke_local_7b.yaml hotpotqa_g1 results/g1_7b \
#     --set vector_store.backend=exact --set lightrag.vector_storage=NanoVectorDBStorage \
#     --set graph_store.backend=networkx --set lightrag.graph_storage=NetworkXStorage
#
# The --set overrides must match the experiment's variant overrides that affect the *index*
# (storage backends), so the experiment finds the index built here. Every step is resumable:
# re-running skips finished graph documents and completed runs.
#
# Environment: ERAG (CLI command, default "uv run --offline elevate-rag"), PY (python command),
# ELEVATE_RUNS_DIR (logs go to $ELEVATE_RUNS_DIR/logs), REFERENCE (report baseline, default V4),
# ELEVATE_NO_CAFFEINATE=1 to skip keeping a Mac awake.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  sed -n '2,19p' "$0"; exit 2
fi
PIPELINE="$1"; EXPERIMENT="$2"; DATASET="$3"; OUT="$4"; shift 4
SETS=(--set "dataset=${DATASET}" "$@")
read -r -a ERAG <<< "${ERAG:-uv run --offline elevate-rag}"
read -r -a PY <<< "${PY:-uv run --offline python}"
EXP_ID="$(sed -n 's/^id: *//p' "$EXPERIMENT" | head -1)"
LOGS="${ELEVATE_RUNS_DIR:-runs}/logs"
mkdir -p "$OUT" "$LOGS"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
# run_step NAME CMD...: full output to the log; abort with the log tail if the step fails.
run_step() {
  local name="$1"; shift
  local log="$LOGS/${EXP_ID}_${name}.log"
  if ! "$@" > "$log" 2>&1; then
    echo "[$(stamp)] step '$name' FAILED, last lines of $log:" >&2
    tail -30 "$log" >&2
    exit 1
  fi
}

if [[ "$(uname)" == "Darwin" && -z "${ELEVATE_NO_CAFFEINATE:-}" ]]; then
  caffeinate -i -s -w $$ &   # a sleeping laptop pauses long local runs
fi

echo "[$(stamp)] 0/4 checking model servers named in $PIPELINE"
"${PY[@]}" - "$PIPELINE" <<'EOF'
import sys, httpx
from elevate_rag.config import load_pipeline_config
cfg = load_pipeline_config(sys.argv[1])
urls = {m.base_url for m in vars(cfg.models).values()
        if getattr(m, "provider", None) == "openai-compatible" and getattr(m, "base_url", None)}
for u in sorted(urls):
    try:
        httpx.get(u.rstrip("/") + "/models", timeout=5).raise_for_status()
    except Exception as exc:
        sys.exit(f"model server not reachable at {u}: {exc}")
    print(f"  ok {u}")
EOF

echo "[$(stamp)] 1/4 index + graph build (log: $LOGS/${EXP_ID}_index_build.log)"
run_step index_build "${ERAG[@]}" index build "$PIPELINE" "${SETS[@]}"

echo "[$(stamp)] 2/4 graph validation (G3 coverage) and judge calibration (GroUSE)"
run_step validate "${ERAG[@]}" index validate "$PIPELINE" "${SETS[@]}"
sed -n '/^{/,$p' "$LOGS/${EXP_ID}_validate.log" > "$OUT/graph_validation.json"
run_step judge_calibrate "${ERAG[@]}" judge calibrate "$PIPELINE" \
  --results-dir "${ELEVATE_RESULTS_DIR:-results}"

echo "[$(stamp)] 3/4 experiment ${EXP_ID} (log: $LOGS/${EXP_ID}_experiment.log)"
# --pipeline: run the spec against the pipeline the index was built with (a spec names its own
# default pipeline, e.g. S0 -> cluster_gpu.yaml, which may differ from PIPELINE).
run_step experiment_validate "${ERAG[@]}" experiment validate "$EXPERIMENT" --pipeline "$PIPELINE"
run_step experiment "${ERAG[@]}" experiment run "$EXPERIMENT" --pipeline "$PIPELINE" --yes \
  --results-dir "${ELEVATE_RESULTS_DIR:-results}"
grep -E "✓|✗" "$LOGS/${EXP_ID}_experiment.log" || true

echo "[$(stamp)] 4/4 report"
run_step report "${ERAG[@]}" report --experiment "$EXP_ID" --out "$OUT/report" \
  --reference "${REFERENCE:-V4}"
echo "[$(stamp)] done: $OUT (run records: ${ELEVATE_RESULTS_DIR:-results}/runs/INDEX.csv)"
