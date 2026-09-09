#!/usr/bin/env bash
# Continue memory4-content with live ownership routing and direct latent features.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_content/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_ownergrad_feature}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_ownergrad_feature}"

export ONE_SHOT_READER_ENABLE=True
export ONE_SHOT_READOUT_MODE=memory_content
export ONE_SHOT_DETACH_OWNER_ROUTING=False
export ONE_SHOT_OWNER_GRADIENT_RAMP_STEPS="${ONE_SHOT_OWNER_GRADIENT_RAMP_STEPS:-500}"
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=False
export E11_MEMORIES_PER_OWNER=16
export E11_OBJECT_MEMORIES_PER_OWNER=4
export E11_REGISTER_MEMORIES_PER_OWNER=16
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_N_REGISTER=4
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT=1.0
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT=0.5
export E8_READER_BACKGROUND_WEIGHT=0.25
export E8_CAUSAL_ENABLE=False
export PGOT_DIT_OVT_XATTN_ENABLE=False
export PGOT_DIT_SOFT_ROUTING_ENABLE=False

# The auxiliary head predicts the frozen decoder-native SigLIP target from the
# exact Reader condition consumed by DiT. It is discarded during generation.
export PGOT_LATENT_DISTILL_ENABLE=True
export PGOT_LATENT_DISTILL_WEIGHT="${PGOT_LATENT_DISTILL_WEIGHT:-0.5}"
export PGOT_LATENT_DISTILL_RAMP_STEPS="${PGOT_LATENT_DISTILL_RAMP_STEPS:-500}"
export PGOT_LATENT_DISTILL_MSE_WEIGHT="${PGOT_LATENT_DISTILL_MSE_WEIGHT:-1.0}"
export PGOT_LATENT_DISTILL_COS_WEIGHT="${PGOT_LATENT_DISTILL_COS_WEIGHT:-1.0}"
export PGOT_LATENT_DISTILL_L1_WEIGHT=0.0

export DIT_UNFREEZE_LAST_N=16
export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export DIFF_HEAD_LR="${DIFF_HEAD_LR:-2e-5}"
export DIT_BODY_LR="${DIT_BODY_LR:-2e-5}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-2e-5}"
export REGISTER_LR="${REGISTER_LR:-2e-5}"
export RAE_QUERY_LR="${RAE_QUERY_LR:-2e-5}"
export LLM_LR="${LLM_LR:-5e-6}"
export LATENT_HEAD_LR="${LATENT_HEAD_LR:-1e-4}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
