#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MEMORY_KEY_MODE="${MEMORY_KEY_MODE:-content}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_${MEMORY_KEY_MODE}/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memory4_${MEMORY_KEY_MODE}}"
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" "$@"
