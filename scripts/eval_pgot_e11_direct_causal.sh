#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_direct_causal/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e11_direct_causal}"
export DTYPE=fp32
exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh" "$@"
