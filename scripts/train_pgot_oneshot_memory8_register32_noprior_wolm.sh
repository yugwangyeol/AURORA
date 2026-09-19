#!/usr/bin/env bash
# Original 1.5B O8/R32: remove LM loss only; keep Writer and Reader losses.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS=2
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_wolm}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_wolm}"
export MASTER_PORT="${MASTER_PORT:-29674}"
export PGOT_LM_LOSS_WEIGHT=0.0
export PGOT_RECON_LOSS_WEIGHT=1.5
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT=0.5
export E8_READER_BACKGROUND_WEIGHT=0.25
export ONE_SHOT_DIRECT_RAE_QUERY=False
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=False
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=False
export ONE_SHOT_MEMORY_VALUE_SOURCE=siglip
exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior.sh"
