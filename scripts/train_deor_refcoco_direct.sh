#!/usr/bin/env bash
# Selector-free RefCOCO fine-tuning: image + original expression -> one OVT mask.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/refcoco_direct}"
INSTANCES_JSON="${REFCOCO_INSTANCES_JSON:-/home/jovyan/data/coco/refcoco/instances.json}"

if [[ ! -s "${DATA_DIR}/train.jsonl" || ! -s "${DATA_DIR}/val.jsonl" ]]; then
    "${PYTHON}" "${PROJECT_ROOT}/preprocess/prepare_refcoco_direct_pgot.py" \
        --output-dir "${DATA_DIR}"
fi

export MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_refcoco_direct}"
export TRAIN_JSONL="${DATA_DIR}/train.jsonl"
export VAL_JSONL="${DATA_DIR}/val.jsonl"
export COCO_INSTANCES_JSON="${INSTANCES_JSON}"
export DATASET_FORMAT=coco_instance
export PGOT_REFCOCO_DIRECT_FINETUNE=True
export REBUILD_CAPTION=False
export IMAGE_PREPROCESS_MODE=default

# Keep the architecture/checkpoint route unchanged. Only the final ownership
# map is supervised; no selector, AR caption target, Reader loss, or DiT loss.
export MEMORY_KEY_MODE=content
export E11_MEMORIES_PER_OWNER=32
export E11_OBJECT_MEMORIES_PER_OWNER=8
export E11_REGISTER_MEMORIES_PER_OWNER=32
export E8_N_REGISTER=4
export ONE_SHOT_WRITER_SOFTMAX_AXIS=patch
export ONE_SHOT_WRITER_OWNER_PRIOR=False
export ONE_SHOT_READER_ENABLE=True
export ONE_SHOT_READOUT_MODE=memory_content
export ONE_SHOT_DETACH_OWNER_ROUTING=True
export E8_UPDATE_MODE=separate_memory
export E10_RAW_VALUE_ENABLE=True
export E11_DUAL_M4_ENABLE=False
export E11_QUERY_SEPARATION_ENABLE=False
export E12_CENTROID_READER_ENABLE=False
export E8_READER_LAYERS=1
export E8_OWNER_WEIGHT=1.0
export E8_READER_SUPERVISION_MODE=none
export E8_READER_OBJECT_WEIGHT=0.0
export E8_READER_BACKGROUND_WEIGHT=0.0
export E8_CAUSAL_ENABLE=False
export PGOT_LM_LOSS_WEIGHT=0.0
export PGOT_RECON_LOSS_WEIGHT=0.0
export PGOT_LATENT_DISTILL_ENABLE=False
export PGOT_DIT_OVT_XATTN_ENABLE=False
export PGOT_DIT_SOFT_ROUTING_ENABLE=False
export DIT_UNFREEZE_LAST_N=0

export NUM_GPUS="${NUM_GPUS:-2}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export MAX_STEPS="${MAX_STEPS:-5000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-False}"
export PGOT_SKIP_FINAL_SAVE=1
export PGOT_EVAL_LOG_RECON_IMAGES=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
export TOKENIZERS_PARALLELISM=false
export EVAL_STEPS="${EVAL_STEPS:-500}"
export EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-64}"
export LEARNING_RATE="${LEARNING_RATE:-2e-5}"
export LLM_LR="${LLM_LR:-2e-5}"
export MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-1e-5}"
export REGISTER_LR="${REGISTER_LR:-2e-5}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
export REPORT_TO="${REPORT_TO:-none}"
export WANDB_NAME="${WANDB_NAME:-deor_refcoco_direct}"

exec bash "${PROJECT_ROOT}/scripts/train_pgot_e8_visual_memory.sh"
