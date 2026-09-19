#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="${SMOKE_ROOT:-$(mktemp -d "${PROJECT_ROOT}/.smoke_gqa_deor_cache.XXXXXX")}" 
trap 'echo "Smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2' ERR
PREPARED_ROOT="${SMOKE_ROOT}/prepared" SPLITS=testdev MAX_IMAGES=2 \
  bash "${PROJECT_ROOT}/scripts/prepare_gqa_deor_cache.sh"
GPU="${GPU:-0}" PREPARED_ROOT="${SMOKE_ROOT}/prepared" \
  CACHE_ROOT="${SMOKE_ROOT}/cache" SPLIT=testdev MAX_IMAGES=2 BATCH_SIZE=2 \
  AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}" \
  bash "${PROJECT_ROOT}/scripts/cache_gqa_deor_objects.sh"
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}" \
  /home/jovyan/.conda/envs/scale_rae/bin/python -m pgot.eval.gqa_object_cache verify \
  --images_manifest "${SMOKE_ROOT}/prepared/manifests/images_testdev.jsonl" \
  --cache_root "${SMOKE_ROOT}/cache" --max_images 2 --require_complete
# A complete cache must resume without loading the model again.
GPU="${GPU:-0}" PREPARED_ROOT="${SMOKE_ROOT}/prepared" \
  CACHE_ROOT="${SMOKE_ROOT}/cache" SPLIT=testdev MAX_IMAGES=2 BATCH_SIZE=2 \
  AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}" \
  bash "${PROJECT_ROOT}/scripts/cache_gqa_deor_objects.sh"
if [[ "${KEEP_SMOKE:-0}" == 1 ]]; then
  echo "Smoke retained at ${SMOKE_ROOT}"
else
  case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_gqa_deor_cache."*) find "${SMOKE_ROOT}" -depth -delete ;;
    *) echo 'Custom smoke root retained; arbitrary paths are never auto-deleted.' ;;
  esac
fi
