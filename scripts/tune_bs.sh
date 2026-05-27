#!/bin/bash
# =============================================================================
# PGOT — Sweep per_device_train_batch_size to find max safe value on a single GPU.
# Loads the model ONCE, then tries each batch size in a list.
# Set candidates via env:  CANDIDATES="4,8,12,16,20,24,32,40,48"
# =============================================================================
set -e

MODEL_PATH="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
TRAIN_JSONL="/home/jovyan/PGOT/data/pgot_train.jsonl"
VAL_JSONL="/home/jovyan/PGOT/data/pgot_val.jsonl"
OUTPUT_DIR="/tmp/pgot_bs_sweep"
mkdir -p "${OUTPUT_DIR}"

CANDIDATES="${CANDIDATES:-4,8,12,16,20,24,32}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

export WANDB_MODE=disabled
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PGOT_TUNE_BATCH="${CANDIDATES}"

echo "===== PGOT Batch-Size Sweep (single GPU=${CUDA_VISIBLE_DEVICES}) ====="
echo "Candidates: ${CANDIDATES}"
echo "==================================================="

"${PYTHON}" "${PROJECT_ROOT}/train.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --use_pgot True \
    --vision_tower_aux_list '["google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[256]' \
    --image_feature_token_len 256 \
    --diffusion_target_token_len 256 \
    --vision_loss diffusion-loss \
    --vision_loss_mode query \
    --vision_coef 1.0 \
    --diffusion_model_hidden_size 2048 \
    --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 \
    --diffusion_model_depth 32 \
    --diffusion_model_heads 32 \
    --dit_cls DiT \
    --pgot_n_register 64 \
    --pgot_n_ovt_per_object 2 \
    --pgot_max_objects 50 \
    --pgot_lm_loss_weight 1.0 \
    --pgot_mask_loss_weight 0.5 \
    --pgot_recon_loss_weight 1.0 \
    --pgot_contrastive_loss_target_weight 0.0 \
    --pgot_contrastive_warmup_steps 99999 \
    --pgot_attention_use_layer_norm True \
    --freeze_dit_body True \
    --freeze_vision_tower True \
    --pgot_lora_enable True \
    --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl   "${VAL_JSONL}" \
    --max_caption_tokens 2048 \
    --grid_size 16 --eval_num_images 4 \
    --image_aspect_ratio square \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --bf16 False --fp16 False --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --report_to none \
    --save_strategy "no" --evaluation_strategy "no" \
    --logging_steps 1 --dataloader_num_workers 0 \
    --remove_unused_columns False
