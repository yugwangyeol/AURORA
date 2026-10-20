#!/usr/bin/env bash
# Usage: GPU=2 STAGE=coco bash scripts/eval_pgot_fixedquery_k7.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_fixedquery_k7_memory8_register32/checkpoint-5000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_fixedquery_k7_memory8_register32}"
STAGE="${STAGE:-coco}"
case "${STAGE}" in coco|voc) ;; *) echo "STAGE must be coco or voc" >&2; exit 2;; esac

"${PYTHON}" "${PROJECT_ROOT}/scripts/verify_fixedquery_checkpoint.py" "${MODEL_PATH}" --count 7
export MODEL_PATH OUTPUT_ROOT
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_memory8_full_coco_voc_ar.sh" "${STAGE}"
