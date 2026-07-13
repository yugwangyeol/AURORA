#!/bin/bash
# Verifies v17 train, in-training eval, save, W&B sigmoid/recon overlays,
# standalone threshold eval, and successful cleanup of smoke artifacts.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_smoke_v17}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-/home/jovyan/PGOT/outputs/smoke_pgot_v17_eval}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256}"

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: V17 smoke expects a V15 warm-start checkpoint at ${MODEL_PATH}" >&2
    exit 1
fi

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_smoke_v17}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "===== PGOT v17 GENERATIVE BINDING SMOKE ====="

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
    --pgot_n_register 0 --pgot_n_null_bg 1 --pgot_n_ovt_per_object 2 --pgot_max_objects 50 \
    --pgot_use_null_bg_competition False \
    --pgot_lm_loss_weight 1.0 \
    --pgot_mask_ce_weight 0.0 \
    --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight 1.0 \
    --pgot_mask_object_balanced_bce_weight 0.0 \
    --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 \
    --pgot_mask_spatial_outside_log_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 0.0 \
    --pgot_mask_llm_attention_outside_weight 0.0 \
    --pgot_mask_llm_patch_outside_weight 0.0 \
    --pgot_v12_enable False \
    --pgot_v14_enable True \
    --pgot_v14_route_temperature 1.0 \
    --pgot_v14_route_weight 0.0 \
    --pgot_v14_void_weight 0.5 \
    --pgot_v14_position_weight 1.0 \
    --pgot_v17_enable True \
    --pgot_v17_ownership_weight 0.1 \
    --pgot_v17_ownership_layers last4 \
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
    --max_caption_tokens 2048 --grid_size 32 --eval_num_images 1 --image_aspect_ratio square \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 1 --num_train_epochs 100 \
    --per_device_train_batch_size 1 \
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

test -f "${OUTPUT_DIR}/checkpoint-1/config.json"
test -f "${OUTPUT_DIR}/config.json"
grep -q '"pgot_n_register": 0' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_n_null_bg": 1' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_v12_enable": false' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_v14_enable": true' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_v14_route_weight": 0.0' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_v17_enable": true' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_v17_ownership_weight": 0.1' "${OUTPUT_DIR}/config.json"
grep -q '"pgot_mask_bce_weight": 1.0' "${OUTPUT_DIR}/config.json"
grep -q "loss_mask_bce" "${OUTPUT_DIR}/smoke_train.log"
grep -q "loss_v17_ownership" "${OUTPUT_DIR}/smoke_train.log"
grep -q "v17_ownership_mass" "${OUTPUT_DIR}/smoke_train.log"
grep -q "eval_loss" "${OUTPUT_DIR}/smoke_train.log"

WANDB_RUN_DIR="$(find "${OUTPUT_DIR}/wandb" -maxdepth 2 -type d -name 'offline-run-*' | head -n 1)"
test -n "${WANDB_RUN_DIR}"
WANDB_BINARY="$(find "${WANDB_RUN_DIR}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
test -n "${WANDB_BINARY}"
WANDB_TABLE="$(find "${WANDB_RUN_DIR}/files/media/table/eval" -type f -name '*.table.json' | head -n 1)"
test -n "${WANDB_TABLE}"
grep -q '"sigmoid_overlay"' "${WANDB_TABLE}"
grep -q '"our_recon"' "${WANDB_TABLE}"

echo "===== PGOT v17 STANDALONE THRESHOLD EVAL ====="
"${PYTHON}" -m pgot.eval.run_eval \
    --model_path "${OUTPUT_DIR}" \
    --val_jsonl "${VAL_JSONL}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --batch_size 1 \
    --num_workers 1 \
    --max_samples 2 \
    --grid_size 32 \
    --max_caption_tokens 2048 \
    --n_ovt_per_object 2 \
    --max_objects 50 \
    --eval_size 224 \
    --readout threshold \
    --eval_merge mean \
    --gt_source coco_instance \
    --coco_mask_cache "${COCO_MASK_CACHE}" \
    --image_preprocess_mode coda_center_crop \
    --dtype fp32 \
    2>&1 | tee "${EVAL_OUTPUT_DIR}/smoke_eval.log"

test -f "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"readout": "threshold"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"gt_source": "coco_instance"' "${EVAL_OUTPUT_DIR}/summary.json"
grep -q '"image_preprocess_mode": "coda_center_crop"' "${EVAL_OUTPUT_DIR}/summary.json"

rm -rf "${OUTPUT_DIR}" "${EVAL_OUTPUT_DIR}"
test ! -e "${OUTPUT_DIR}"
test ! -e "${EVAL_OUTPUT_DIR}"

echo "Smoke complete; smoke artifacts removed."
