#!/usr/bin/env bash
# Evaluate the complete validation set in TF and AR mode on one GPU each.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memorysoftmax_feature/checkpoint-${MAX_STEPS:-5000}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_oneshot_memorysoftmax_feature}"

exec bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_ownergrad_feature_parallel.sh"
