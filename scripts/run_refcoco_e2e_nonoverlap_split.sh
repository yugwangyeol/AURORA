#!/usr/bin/env bash
# Generated-description object retrieval on one RefCOCO split, excluding DeOR
# train-image overlap. Intended for parallel testA/testB jobs.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPLIT="${SPLIT:?Set SPLIT=testA or testB}"
case "${SPLIT}" in testA|testB|val) ;; *) echo "SPLIT must be val/testA/testB" >&2; exit 2;; esac
export SPLITS="${SPLIT}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/refcoco_object_selection_memory8_register32_nonoverlap_${SPLIT}}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
export WITH_CTRLO="${WITH_CTRLO:-0}"
export EXCLUDE_TRAIN_OVERLAP=1
export PRESENCE_IOU=0.1
export CPU_THREADS="${CPU_THREADS:-3}"

exec bash "${PROJECT_ROOT}/scripts/run_refcoco_object_selection.sh"
