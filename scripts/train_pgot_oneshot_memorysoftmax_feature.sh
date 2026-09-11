#!/usr/bin/env bash
# Continue ownergrad-feature while allocating each patch across memories.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_ownergrad_feature/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memorysoftmax_feature}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memorysoftmax_feature}"
export ONE_SHOT_WRITER_SOFTMAX_AXIS="${ONE_SHOT_WRITER_SOFTMAX_AXIS:-memory}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_ownergrad_feature.sh"
