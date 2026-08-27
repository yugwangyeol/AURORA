#!/usr/bin/env bash
# Full FP32 evaluation of E10-R / E10-A.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e10_r_raw_value/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e10_r_raw_value}"
export DTYPE=fp32

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
