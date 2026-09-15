#!/usr/bin/env bash
# Direct-RAE adapter baseline plus CODA E_self - lambda * E_mixed objective.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_directrae_adapter_contrastive}"
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=True
export PGOT_CONTRASTIVE_TARGET_WEIGHT="${PGOT_CONTRASTIVE_TARGET_WEIGHT:-0.03}"
export PGOT_CONTRASTIVE_SAMPLING_RATE="${PGOT_CONTRASTIVE_SAMPLING_RATE:-0.5}"
export PGOT_CONTRASTIVE_WARMUP_STEPS="${PGOT_CONTRASTIVE_WARMUP_STEPS:-200}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_directrae_adapter.sh"
