#!/usr/bin/env bash
# Run under tmux; performs an Ours smoke before the full resumable batch.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/figure_segmentation_over_under}"
mkdir -p "${OUTPUT_ROOT}"
exec > >(tee -a "${OUTPUT_ROOT}/pipeline.log") 2>&1
trap 'result=$?; if (( result != 0 )); then echo "PIPELINE FAILED: exit ${result}; inspect the final stage log and resume after fixing"; fi' EXIT
echo "PIPELINE STARTED: $(date -u +%FT%TZ)"
LIMIT=1 bash scripts/mine_segmentation_figure_examples.sh ours
unset LIMIT
bash scripts/mine_segmentation_figure_examples.sh coda metaslot adaslot ours select
echo "PIPELINE COMPLETED: $(date -u +%FT%TZ)"
echo "Automatic candidates are ready. Manual visual review remains required."
