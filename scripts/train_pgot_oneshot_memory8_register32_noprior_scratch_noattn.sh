#!/usr/bin/env bash
# Description-path ablation: OVT caption-average Init ON, own-caption Attn OFF.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_scratch_noattn}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_scratch_noattn}"

# Keep the description-average layer-0 OVT initialization, while preventing
# OVT rows from attending to their own caption span in subsequent LLM layers.
# Isolated attention stays enabled so the only remaining OVT inputs are image
# patches and the OVT itself; otherwise caption information could leak through
# the ordinary causal attention path.
export PGOT_OVT_CAPTION_INIT=True
export PGOT_OVT_ISOLATED_ATTENTION=True
export PGOT_OVT_ATTENDS_OWN_CAPTION=False
export MASTER_PORT="${MASTER_PORT:-29696}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_scratch.sh"
