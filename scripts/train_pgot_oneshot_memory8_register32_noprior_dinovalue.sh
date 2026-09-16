#!/usr/bin/env bash
# Object-8/Register-32 MLLM-query baseline with frozen DINOv2 patch values.
# SigLIP remains the MLLM image encoder, Writer key, and RAE target.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_dinovalue}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_dinovalue}"

export ONE_SHOT_DIRECT_RAE_QUERY=False
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=False
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=False
export ONE_SHOT_MEMORY_VALUE_SOURCE=dinov2
export ONE_SHOT_MEMORY_VALUE_DIM=768
export VISION_TOWER_AUX_LIST='["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224","facebook/dinov2-base-res518"]'
export VISION_TOWER_AUX_TOKEN_LEN_LIST='[1024,256,1024]'

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior.sh"
