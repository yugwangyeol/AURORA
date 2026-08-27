#!/usr/bin/env bash
# Full FP32 CODA-512 evaluation of E11 Dual-M4.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_dual_m4/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e11_dual_m4}"
export DTYPE=fp32

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
