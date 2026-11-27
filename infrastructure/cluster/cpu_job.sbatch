#!/usr/bin/env bash
# CPU job: STEP=report (tables, paired tests, figures for $EXPERIMENT_ID) or STEP=check (env).
# Data preparation and model downloads need internet: run those on the login node instead
# (`submit.sh prepare`), since compute nodes are often offline.
#
#SBATCH --job-name=elevate-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
# shellcheck source=infrastructure/cluster/env.sh
source infrastructure/cluster/env.sh
read -r -a ERAG_CMD <<< "$ERAG"

case "${STEP:?set STEP=report or STEP=check}" in
  report)
    : "${EXPERIMENT_ID:?set EXPERIMENT_ID to the id of the experiment spec, e.g. E2}"
    out="$ELEVATE_ROOT/results/${EXPERIMENT_ID}"
    "${ERAG_CMD[@]}" report --experiment "$EXPERIMENT_ID" --out "$out" \
      --reference "${REFERENCE:-V4}"
    echo "report: $out/REPORT.md"
    ;;
  check) "${ERAG_CMD[@]}" env ;;
  *) echo "unknown STEP" >&2; exit 2 ;;
esac
