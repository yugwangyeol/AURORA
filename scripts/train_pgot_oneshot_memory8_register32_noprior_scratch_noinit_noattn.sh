#!/usr/bin/env bash
# Description-path ablation: OVT caption-average Init OFF, own-caption Attn OFF.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch_noinit_noattn}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_scratch_noinit_noattn}"

# Remove both description-to-OVT paths while retaining every other standard
# O8/R32 scratch setting. Isolated attention remains enabled so the OVT cannot
# recover its caption through the ordinary causal attention path.
export PGOT_OVT_CAPTION_INIT=False
export PGOT_OVT_ISOLATED_ATTENTION=True
export PGOT_OVT_ATTENDS_OWN_CAPTION=False
export MASTER_PORT="${MASTER_PORT:-29697}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_scratch.sh"
