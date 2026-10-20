#!/usr/bin/env bash
# Description-path ablation: OVT caption-average Init OFF, own-caption Attn ON.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch_noinit}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_scratch_noinit}"

# The standard O8/R32 scratch launcher fixes all other architecture and loss
# settings. These switches alter only the two description-to-OVT paths.
export PGOT_OVT_CAPTION_INIT=False
export PGOT_OVT_ISOLATED_ATTENTION=True
export PGOT_OVT_ATTENDS_OWN_CAPTION=True
export MASTER_PORT="${MASTER_PORT:-29695}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_scratch.sh"
