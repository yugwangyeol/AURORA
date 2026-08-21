#!/usr/bin/env bash
# Full E9-A evaluation using the common E8/E9 ownership and reconstruction path.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e9_unified_gru/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e9_unified_gru}"
export DTYPE=fp32

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
