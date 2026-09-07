#!/usr/bin/env bash
# One-shot Reader with query-dependent raw K/V attention strictly inside one owner.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ONE_SHOT_READOUT_MODE=owner_masked
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_owner_masked}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_owner_masked}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_pooled.sh"
