#!/bin/bash
# PGOT v20: V15 router AdaLN condition + zero-init DiT OVT/void cross-attention.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_main_v15_bce_bottleneck}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_main_v20_dit_ovt_xattn}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-30}"
MAX_STEPS="${MAX_STEPS:-10000}"

LEARNING_RATE="${LEARNING_RATE:-5e-5}"
DIFF_HEAD_LR="${DIFF_HEAD_LR:-3e-5}"
DIT_BODY_LR="${DIT_BODY_LR:-1e-5}"
VOID_LR="${VOID_LR:-5e-5}"
RAE_QUERY_LR="${RAE_QUERY_LR:-5e-5}"
LLM_LR="${LLM_LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
EVAL_STEPS="${EVAL_STEPS:-500}"

PGOT_LM_LOSS_WEIGHT="${PGOT_LM_LOSS_WEIGHT:-1.0}"
PGOT_RECON_LOSS_WEIGHT="${PGOT_RECON_LOSS_WEIGHT:-1.5}"
PGOT_MASK_BCE_WEIGHT="${PGOT_MASK_BCE_WEIGHT:-1.0}"
PGOT_V14_ROUTE_WEIGHT="${PGOT_V14_ROUTE_WEIGHT:-0.0}"
PGOT_V14_VOID_WEIGHT="${PGOT_V14_VOID_WEIGHT:-0.5}"
PGOT_V14_ROUTE_TEMPERATURE="${PGOT_V14_ROUTE_TEMPERATURE:-1.0}"
PGOT_V14_POSITION_WEIGHT="${PGOT_V14_POSITION_WEIGHT:-1.0}"
PGOT_V14_ROUTER_DEPTH="${PGOT_V14_ROUTER_DEPTH:-1}"
PGOT_V14_ROUTER_MLP_RATIO="${PGOT_V14_ROUTER_MLP_RATIO:-4}"
PGOT_DIT_OVT_XATTN_START_BLOCK="${PGOT_DIT_OVT_XATTN_START_BLOCK:-25}"
PGOT_DIT_OVT_XATTN_EVERY_N_BLOCKS="${PGOT_DIT_OVT_XATTN_EVERY_N_BLOCKS:-1}"
PGOT_CFG_DROP_RATE="${PGOT_CFG_DROP_RATE:-0.1}"
PGOT_DIT_UNFREEZE_LAST_N_BLOCKS="${PGOT_DIT_UNFREEZE_LAST_N_BLOCKS:-8}"
PGOT_EVAL_LOG_RECON_IMAGES="${PGOT_EVAL_LOG_RECON_IMAGES:-4}"

export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_main_v20_dit_ovt_xattn}"
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    if [ "${RESUME_FROM_CHECKPOINT}" = "latest" ]; then
        RESUME_FROM_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi
    if [ -z "${RESUME_FROM_CHECKPOINT}" ] || [ ! -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json" ]; then
        echo "ERROR: invalid resume checkpoint: ${RESUME_FROM_CHECKPOINT:-<empty>}" >&2
        exit 1
    fi
    echo "Resume checkpoint: ${RESUME_FROM_CHECKPOINT}"
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

echo "===== PGOT v20 DiT OVT cross-attn - B200 x ${NUM_GPUS} ====="
echo "Init model:              ${MODEL_PATH}"
echo "Output:                  ${OUTPUT_DIR}"
echo "Effective batch:         $((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Register / void:         0 / 1"
echo "Grid size:               32"
echo "Router depth / MLP:      ${PGOT_V14_ROUTER_DEPTH} / ${PGOT_V14_ROUTER_MLP_RATIO}"
echo "DiT OVT xattn blocks:    start=${PGOT_DIT_OVT_XATTN_START_BLOCK}, every=${PGOT_DIT_OVT_XATTN_EVERY_N_BLOCKS}"
echo "Save policy:             steps=${SAVE_STEPS}, total_limit=${SAVE_TOTAL_LIMIT}"
echo "Loss weights:            LM=${PGOT_LM_LOSS_WEIGHT}, recon=${PGOT_RECON_LOSS_WEIGHT}, BCE=${PGOT_MASK_BCE_WEIGHT}, route=${PGOT_V14_ROUTE_WEIGHT}"
echo "============================================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT:-29533}" \
    "${PROJECT_ROOT}/train.py" \
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
    --pgot_lm_loss_weight "${PGOT_LM_LOSS_WEIGHT}" \
    --pgot_mask_ce_weight 0.0 \
    --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight "${PGOT_MASK_BCE_WEIGHT}" \
    --pgot_mask_object_balanced_bce_weight 0.0 \
    --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 \
    --pgot_mask_spatial_outside_log_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 0.0 \
    --pgot_mask_llm_attention_outside_weight 0.0 \
    --pgot_mask_llm_patch_outside_weight 0.0 \
    --pgot_v12_enable False \
    --pgot_v14_enable True \
    --pgot_v14_route_temperature "${PGOT_V14_ROUTE_TEMPERATURE}" \
    --pgot_v14_route_weight "${PGOT_V14_ROUTE_WEIGHT}" \
    --pgot_v14_void_weight "${PGOT_V14_VOID_WEIGHT}" \
    --pgot_v14_position_weight "${PGOT_V14_POSITION_WEIGHT}" \
    --pgot_v14_router_depth "${PGOT_V14_ROUTER_DEPTH}" \
    --pgot_v14_router_mlp_ratio "${PGOT_V14_ROUTER_MLP_RATIO}" \
    --pgot_dit_ovt_cross_attn_enable True \
    --pgot_dit_ovt_cross_attn_start_block "${PGOT_DIT_OVT_XATTN_START_BLOCK}" \
    --pgot_dit_ovt_cross_attn_every_n_blocks "${PGOT_DIT_OVT_XATTN_EVERY_N_BLOCKS}" \
    --pgot_recon_loss_weight "${PGOT_RECON_LOSS_WEIGHT}" \
    --pgot_latent_distill_enable False \
    --pgot_latent_distill_weight 0.0 \
    --pgot_cfg_drop_rate "${PGOT_CFG_DROP_RATE}" \
    --pgot_contrastive_loss_target_weight 0.0 \
    --pgot_contrastive_warmup_steps 0 \
    --pgot_attention_use_layer_norm True \
    --pgot_attention_temperature 0.5 \
    --pgot_rae_attends_caption False \
    --freeze_dit_body True --pgot_dit_unfreeze_last_n_blocks "${PGOT_DIT_UNFREEZE_LAST_N_BLOCKS}" --freeze_vision_tower True \
    --pgot_lora_enable True --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --train_jsonl "${TRAIN_JSONL}" --val_jsonl "${VAL_JSONL}" \
    --max_caption_tokens 2048 --grid_size 32 --eval_num_images 64 --image_aspect_ratio square \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" --diff_head_lr "${DIFF_HEAD_LR}" --pgot_dit_body_lr "${DIT_BODY_LR}" \
    --pgot_register_lr "${VOID_LR}" --pgot_rae_query_lr "${RAE_QUERY_LR}" --pgot_llm_lr "${LLM_LR}" \
    --weight_decay 0.01 --warmup_ratio "${WARMUP_RATIO}" --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False --pgot_use_wsd_schedule False \
    --max_grad_norm 1.0 \
    --bf16 False --fp16 False --tf32 True --model_max_length 4096 \
    --gradient_checkpointing False \
    --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --evaluation_strategy steps --eval_steps "${EVAL_STEPS}" --prediction_loss_only True \
    --pgot_eval_log_recon_images "${PGOT_EVAL_LOG_RECON_IMAGES}" \
    --logging_steps "${LOGGING_STEPS}" --report_to wandb \
    --dataloader_num_workers 4 --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    "${RESUME_ARGS[@]}"
