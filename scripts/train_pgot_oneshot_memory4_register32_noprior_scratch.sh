#!/usr/bin/env bash
# Cold-start 1.5B DeOR O4/R32 ablation from the vanilla Scale-RAE checkpoint.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory4_register32_noprior_scratch}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory4_register32_noprior_scratch}"

# Match the O1/R32 scratch ablation and change only object visual-memory
# capacity from one to four. Register visual-memory capacity stays at 32.
export MEMORY_KEY_MODE=content
export ONE_SHOT_READER_ENABLE=True
export ONE_SHOT_READOUT_MODE=memory_content
export ONE_SHOT_SCRATCH_INIT=True
export ONE_SHOT_DIRECT_RAE_QUERY=False
export ONE_SHOT_RAE_QUERY_ADAPTER_ENABLE=False
export ONE_SHOT_MEMORY_CONTRASTIVE_ENABLE=False
export ONE_SHOT_MEMORY_VALUE_SOURCE=siglip
export PGOT_CONTRASTIVE_TARGET_WEIGHT=0.0
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False
export ONE_SHOT_DETACH_OWNER_ROUTING=True
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=False
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=4
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_N_REGISTER=4
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT=1.0
export E8_READER_SUPERVISION_MODE=writer
export E8_READER_OBJECT_WEIGHT=0.5
export E8_READER_BACKGROUND_WEIGHT=0.25
export PGOT_LM_LOSS_WEIGHT=1.0
export PGOT_RECON_LOSS_WEIGHT=1.5
export E8_CAUSAL_ENABLE=False
export PGOT_DIT_OVT_XATTN_ENABLE=False
export PGOT_DIT_SOFT_ROUTING_ENABLE=False
export DIT_UNFREEZE_LAST_N=16

export NUM_GPUS=2
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-24}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export MASTER_PORT="${MASTER_PORT:-29694}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-True}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory4.sh"
