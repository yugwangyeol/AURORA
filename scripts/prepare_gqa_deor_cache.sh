#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
PREPARE_PYTHON="${PREPARE_PYTHON:-/home/jovyan/.conda/envs/metaquery/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-/home/jovyan/data/gqa/hf_balanced}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/gqa/prepared_balanced}"
read -r -a split_args <<< "${SPLITS:-train val testdev}"
extra=()
if [[ "${MAX_IMAGES:-0}" != 0 ]]; then extra+=(--max_images "${MAX_IMAGES}"); fi
PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  OMP_NUM_THREADS="${CPU_THREADS:-3}" MKL_NUM_THREADS="${CPU_THREADS:-3}" \
  "${PREPARE_PYTHON}" -u -m pgot.eval.gqa_object_cache prepare \
    --source_root "${SOURCE_ROOT}" --prepared_root "${PREPARED_ROOT}" \
    --splits "${split_args[@]}" --cpu_threads "${CPU_THREADS:-3}" "${extra[@]}"
