#!/bin/bash
# =============================================================================
# PGOT v8.1 smoke test.
# Verifies train, in-training eval, save, offline wandb scalar/table logging,
# and a tiny spatial/coco_instance eval.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_smoke_v8_1}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/home/jovyan/PGOT/outputs/smoke_pgot_v8_1_eval}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}/spatial"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_smoke_v8_1_llm_qk_outside}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-1}"

echo "===== PGOT v8.1 SMOKE TEST (BS=${PER_DEVICE_TRAIN_BATCH_SIZE}, steps=${MAX_STEPS}) ====="

"${PYTHON}" "${PROJECT_ROOT}/train.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --use_pgot True \
    --vision_tower_aux_list '["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[1024,256]' \
    --image_feature_token_len 1024 \
    --diffusion_target_token_len 256 \
    --diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH}" \
    --vision_loss diffusion-loss --vision_loss_mode query --vision_coef 1.0 \
    --diffusion_model_hidden_size 2048 --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 --diffusion_model_depth 32 \
    --diffusion_model_heads 32 --dit_cls DiT \
    --pgot_n_register 64 --pgot_n_null_bg 0 --pgot_n_ovt_per_object 2 --pgot_max_objects 50 \
    --pgot_lm_loss_weight 1.0 \
    --pgot_mask_ce_weight 0.0 \
    --pgot_mask_ce_temperature 1.0 \
    --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight 0.0 \
    --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 1.0 \
    --pgot_mask_llm_qk_outside_temperature 1.0 \
    --pgot_mask_llm_qk_outside_layers last4 \
    --pgot_recon_loss_weight 1.5 \
    --pgot_cfg_drop_rate 0.1 \
    --pgot_contrastive_loss_target_weight 0.0 \
    --pgot_contrastive_warmup_steps 0 \
    --pgot_attention_use_layer_norm True \
    --pgot_attention_temperature 0.5 \
    --pgot_rae_attends_caption False \
    --freeze_dit_body True --pgot_dit_unfreeze_last_n_blocks 8 --freeze_vision_tower True \
    --pgot_lora_enable True --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --train_jsonl "${TRAIN_JSONL}" --val_jsonl "${VAL_JSONL}" \
    --max_caption_tokens 2048 --grid_size 32 --eval_num_images 2 --image_aspect_ratio square \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" --num_train_epochs 100 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 --diff_head_lr 3e-5 --pgot_dit_body_lr 1e-5 \
    --pgot_register_lr 5e-5 --pgot_rae_query_lr 5e-5 --pgot_llm_lr 1e-4 \
    --weight_decay 0.01 --warmup_ratio 0.1 --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False --pgot_use_wsd_schedule False \
    --max_grad_norm 1.0 \
    --bf16 False --fp16 False --tf32 True --model_max_length 4096 \
    --gradient_checkpointing False \
    --save_strategy steps --save_steps 1 --save_total_limit 2 \
    --evaluation_strategy steps --eval_steps 1 --prediction_loss_only True \
    --pgot_eval_log_recon_images 1 \
    --logging_steps 1 --report_to wandb \
    --dataloader_num_workers 1 --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    2>&1 | tee "${OUTPUT_DIR}/smoke_train.log"

echo "===== PGOT v8.1 SMOKE EVAL (spatial/coco_instance) ====="
"${PYTHON}" -m pgot.eval.run_eval \
    --model_path "${OUTPUT_DIR}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${EVAL_OUTPUT_DIR}/spatial" \
    --batch_size 1 --num_workers 1 --max_samples 2 \
    --grid_size 32 --max_caption_tokens 2048 --n_ovt_per_object 2 --max_objects 50 \
    --eval_size 224 --readout spatial --eval_merge mean --spatial_temperature 1.0 \
    --gt_source coco_instance --dtype fp32 \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/spatial/smoke_eval.log"

echo "Smoke complete:"
echo "  train output: ${OUTPUT_DIR}"
echo "  eval output:  ${EVAL_OUTPUT_DIR}"
