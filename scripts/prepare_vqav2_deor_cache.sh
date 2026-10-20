#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
VQAV2_ROOT="${VQAV2_ROOT:-/home/jovyan/data/vqav2}"
PREPARED_ROOT="${PREPARED_ROOT:-/home/jovyan/data/vqav2/prepared}"
read -r -a split_args <<< "${SPLITS:-train val}"
extra=()
if [[ "${MAX_IMAGES:-0}" != 0 ]]; then
  extra+=(--max_images "${MAX_IMAGES}")
fi

PYTHONNOUSERSITE=1 PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
  OMP_NUM_THREADS="${CPU_THREADS:-3}" MKL_NUM_THREADS="${CPU_THREADS:-3}" \
  "${PYTHON}" -u -m pgot.eval.vqav2_object_cache prepare \
    --vqav2_root "${VQAV2_ROOT}" --prepared_root "${PREPARED_ROOT}" \
    --splits "${split_args[@]}" "${extra[@]}"
