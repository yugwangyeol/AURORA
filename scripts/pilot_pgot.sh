#!/bin/bash
# =============================================================================
# PGOT — Pilot run (B200 x 2): 2000 steps for architecture sanity check
#  - Validates: <ovt> generation, attention mask, mask BCE, recon flow
#  - Same architecture as full run but small budget
# =============================================================================

MODEL_PATH="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"

TRAIN_JSONL="/home/jovyan/PGOT/data/pgot_train.jsonl"
VAL_JSONL="/home/jovyan/PGOT/data/pgot_val.jsonl"

OUTPUT_DIR="/home/jovyan/PGOT/checkpoints/pgot_pilot"
EXPERIMENT_NAME="pgot_pilot_b200x2"

NUM_GPUS=2
PER_DEVICE_TRAIN_BATCH_SIZE=8   # tuned: BS=16 OOMs, BS=12 = 140GB/GPU safe
PER_DEVICE_EVAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2    # effective = 12 * 2 * 2 = 48

MAX_STEPS=2000
NUM_TRAIN_EPOCHS=100   # huge -> max_steps controls

# Slightly higher LRs for pilot to surface signal fast
LEARNING_RATE=5e-5
DIFF_HEAD_LR=3e-5
REGISTER_LR=5e-5
RAE_QUERY_LR=5e-5
LLM_LR=1e-4
WARMUP_RATIO=0.05

LOGGING_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500

PGOT_N_REGISTER=64
PGOT_N_OVT_PER_OBJECT=2
PGOT_MAX_OBJECTS=50
PGOT_LM_LOSS_WEIGHT=1.0
PGOT_MASK_LOSS_WEIGHT=0.5
PGOT_RECON_LOSS_WEIGHT=1.0
# Disable contrastive during pilot (warmup never reached)
PGOT_CONTRASTIVE_TARGET_WEIGHT=0.0
PGOT_CONTRASTIVE_WARMUP_STEPS=99999

export WANDB_PROJECT="PGOT"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

echo "===== PGOT Pilot (B200 x 2) ====="
echo "Model:    ${MODEL_PATH}"
echo "GPUs:     ${NUM_GPUS}"
echo "Eff. bs:  $((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Max step: ${MAX_STEPS}"
echo "================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29503 \
    "${PROJECT_ROOT}/train.py" \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --use_pgot True \
    \
    --vision_tower_aux_list '["google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[256]' \
    --image_feature_token_len 256 \
    --diffusion_target_token_len 256 \
    \
    --vision_loss diffusion-loss \
    --vision_loss_mode query \
    --vision_coef 1.0 \
    --diffusion_model_hidden_size 2048 \
    --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 \
    --diffusion_model_depth 32 \
    --diffusion_model_heads 32 \
    --dit_cls DiT \
    \
    --pgot_n_register "${PGOT_N_REGISTER}" \
    --pgot_n_ovt_per_object "${PGOT_N_OVT_PER_OBJECT}" \
    --pgot_max_objects "${PGOT_MAX_OBJECTS}" \
    --pgot_lm_loss_weight "${PGOT_LM_LOSS_WEIGHT}" \
    --pgot_mask_loss_weight "${PGOT_MASK_LOSS_WEIGHT}" \
    --pgot_recon_loss_weight "${PGOT_RECON_LOSS_WEIGHT}" \
    --pgot_contrastive_loss_target_weight "${PGOT_CONTRASTIVE_TARGET_WEIGHT}" \
    --pgot_contrastive_warmup_steps "${PGOT_CONTRASTIVE_WARMUP_STEPS}" \
    --pgot_attention_use_layer_norm True \
    --pgot_attention_temperature 1.0 \
    --freeze_dit_body True \
    --freeze_vision_tower True \
    \
    --pgot_lora_enable True \
    --pgot_lora_r 16 \
    --pgot_lora_alpha 32 \
    --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl "${VAL_JSONL}" \
    --max_caption_tokens 2048 \
    --grid_size 16 \
    --eval_num_images 64 \
    --image_aspect_ratio square \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --diff_head_lr "${DIFF_HEAD_LR}" \
    --pgot_register_lr "${REGISTER_LR}" \
    --pgot_rae_query_lr "${RAE_QUERY_LR}" \
    --pgot_llm_lr "${LLM_LR}" \
    --weight_decay 0.01 \
    --warmup_ratio "${WARMUP_RATIO}" \
    --pgot_use_cosine_min_lr_schedule True \
    --pgot_min_lr_ratio 0.10 \
    --max_grad_norm 1.0 \
    --bf16 False \
    --fp16 False \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --prediction_loss_only True \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to wandb \
    --run_name "${EXPERIMENT_NAME}" \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    2>&1 | tee "${OUTPUT_DIR}/pilot.log"
