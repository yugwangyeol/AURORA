#!/usr/bin/env bash
# Evaluate the final E8.2 paired-causal checkpoint with the E8 bottleneck path.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e8_2_paired_causal/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/eval_pgot_e8_2_paired_causal}"
export DTYPE="${DTYPE:-fp32}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_e8_visual_memory.sh"
