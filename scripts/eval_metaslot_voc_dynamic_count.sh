#!/usr/bin/env bash
# Record the official MetaSlot prototype-pruning mask, then score VOC count MAE.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/baselines_voc_zeroshot_metaslot_dynamic_count}"

exec bash "${PROJECT_ROOT}/scripts/eval_baselines_voc_zeroshot.sh" metaslot
