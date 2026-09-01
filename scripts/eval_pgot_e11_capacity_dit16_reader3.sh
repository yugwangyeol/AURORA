#!/usr/bin/env bash
# Full FP32 CODA-512 evaluation of E11 Capacity + DiT-16 + Reader-3.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16_reader3/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16_reader3}"
export DTYPE=fp32

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
