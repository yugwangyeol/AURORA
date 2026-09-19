#!/usr/bin/env bash
# Cold-start PGOT O8/R32 on the official Qwen-7B + DiT-9.8B Scale-RAE.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/jovyan/.cache/huggingface/hub/models--nyu-visionx--Scale-RAE-Qwen7B_DiT9.8B/snapshots/0e27c3386041492e84bceac17c742efde039cc4c}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_bf16_scratch}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_7b_bf16_scratch}"

# Match the official 7B checkpoint architecture.  The 512px SigLIP tower is
# still used for PGOT input patches; the checkpoint's 224px tower supplies the
# native 256-token Scale-RAE diffusion target.
export DIFFUSION_MODEL_HIDDEN_SIZE=4096
export DIFFUSION_MODEL_CHANNELS=1152
export DIFFUSION_MODEL_Z_CHANNELS=4096
export DIFFUSION_MODEL_DEPTH=32
export DIFFUSION_MODEL_HEADS=32
export IMAGE_FEATURE_TOKEN_LEN=1024
export DIFFUSION_TARGET_TOKEN_LEN=256
export VISION_TOWER_AUX_LIST='["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224"]'
export VISION_TOWER_AUX_TOKEN_LEN_LIST='[1024,256]'

export MEMORY_KEY_MODE=content
export ONE_SHOT_READER_ENABLE=True
export ONE_SHOT_READOUT_MODE=memory_content
export ONE_SHOT_SCRATCH_INIT=True
export ONE_SHOT_DIRECT_RAE_QUERY=False
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=False
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=False
export ONE_SHOT_MEMORY_VALUE_SOURCE=siglip
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False
export ONE_SHOT_DETACH_OWNER_ROUTING=True
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=False
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=8
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_N_REGISTER=4
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT=1.0
export E8_READER_SUPERVISION_MODE="${E8_READER_SUPERVISION_MODE:-writer}"
export E8_READER_OBJECT_WEIGHT="${E8_READER_OBJECT_WEIGHT:-0.5}"
export E8_READER_BACKGROUND_WEIGHT="${E8_READER_BACKGROUND_WEIGHT:-0.25}"
export E8_CAUSAL_ENABLE=False
export PGOT_DIT_OVT_XATTN_ENABLE=False
export PGOT_DIT_SOFT_ROUTING_ENABLE=False
export DIT_UNFREEZE_LAST_N=16

export BF16="${BF16:-True}"
export FP16="${FP16:-False}"
export TF32="${TF32:-True}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
export PGOT_DIFFUSION_FORCE_FP32="${PGOT_DIFFUSION_FORCE_FP32:-False}"
export FSDP_CONFIG="${FSDP_CONFIG:-${PROJECT_ROOT}/configs/fsdp_7b_full_shard.json}"
export FSDP_MODE="${FSDP_MODE:-full_shard auto_wrap}"
export PGOT_TRAINABLE_FP32="${PGOT_TRAINABLE_FP32:-False}"
export PGOT_PARAMETER_UPDATE_AUDIT="${PGOT_PARAMETER_UPDATE_AUDIT:-False}"

export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
