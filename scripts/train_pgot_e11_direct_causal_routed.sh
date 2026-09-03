#!/usr/bin/env bash
# Direct-Causal DiT with Writer-predicted owner x memory soft routing.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PGOT_DIT_OVT_XATTN_ENABLE=True
export PGOT_DIT_OVT_XATTN_START_BLOCK=17
export PGOT_DIT_OVT_XATTN_EVERY_N_BLOCKS=1
export PGOT_DIT_SOFT_ROUTING_ENABLE=True
export PGOT_DIT_SOFT_ROUTING_SCALE="${PGOT_DIT_SOFT_ROUTING_SCALE:-1.0}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e11_direct_causal_routed}"
export WANDB_NAME="${WANDB_NAME:-pgot_e11_direct_causal_routed}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e11_causal_only.sh"
