#!/bin/bash
# =============================================================================
# AURORA v2 training script
# =============================================================================

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"

RECON_IMAGE_FOLDER="${RECON_IMAGE_FOLDER:-/home/jovyan/data/coco/train2017}"
INPAINT_DATA_PATH="${INPAINT_DATA_PATH:-/home/jovyan/processed_coco/training_data_v4_patch/training_manifest_patch_v4_0_to_None.json}"
INPAINT_IMAGE_FOLDER="${INPAINT_IMAGE_FOLDER:-/home/jovyan/processed_coco/training_data_v4_patch}"
COCO_ANNOTATION="${COCO_ANNOTATION:-/home/jovyan/data/coco/annotations/instances_train2017.json}"

OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/aurora_phase1}"
NUM_GPUS="${NUM_GPUS:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-100}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
REPORT_TO="${REPORT_TO:-wandb}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DIFF_HEAD_LR="${DIFF_HEAD_LR:-5e-5}"
BF16="${BF16:-True}"
AURORA_TRAINING_STAGE="${AURORA_TRAINING_STAGE:-1}"
AURORA_MAX_SLOTS="${AURORA_MAX_SLOTS:-10}"
AURORA_N_REGISTER="${AURORA_N_REGISTER:-8}"
AURORA_CMD_LENGTH="${AURORA_CMD_LENGTH:-8}"
AURORA_MASK_LOSS_WEIGHT="${AURORA_MASK_LOSS_WEIGHT:-1.0}"
AURORA_DIVERSITY_LOSS_WEIGHT="${AURORA_DIVERSITY_LOSS_WEIGHT:-0.1}"
AURORA_INPAINT_WEIGHT="${AURORA_INPAINT_WEIGHT:-0.5}"
AURORA_TRAIN_DIFFUSION_CONDITION="${AURORA_TRAIN_DIFFUSION_CONDITION:-True}"
AURORA_INCLUDE_INPAINTING="${AURORA_INCLUDE_INPAINTING:-True}"
AURORA_INPAINT_WARMUP_STEPS="${AURORA_INPAINT_WARMUP_STEPS:-1000}"
AURORA_INPAINT_RAMP_STEPS="${AURORA_INPAINT_RAMP_STEPS:-4000}"
AURORA_EVAL_NUM_IMAGES="${AURORA_EVAL_NUM_IMAGES:-100}"
AURORA_EVAL_LOG_IMAGE_COUNT="${AURORA_EVAL_LOG_IMAGE_COUNT:-100}"
AURORA_EVAL_VISUAL_BATCH_SIZE="${AURORA_EVAL_VISUAL_BATCH_SIZE:-4}"
AURORA_EVAL_LOG_RECONSTRUCTIONS="${AURORA_EVAL_LOG_RECONSTRUCTIONS:-True}"
AURORA_EVAL_LOG_ATTENTION_OVERLAYS="${AURORA_EVAL_LOG_ATTENTION_OVERLAYS:-True}"
AURORA_EVAL_ATTENTION_OVERLAY_COUNT="${AURORA_EVAL_ATTENTION_OVERLAY_COUNT:-50}"
AURORA_EVAL_DECODER_REPO="${AURORA_EVAL_DECODER_REPO:-nyu-visionx/siglip2_decoder}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-}"

export WANDB_PROJECT="${WANDB_PROJECT:-AURORA}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-aurora_phase1}"

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
if [ -f "${SCALE_RAE_ENV}/bin/python" ]; then
    PYTHON="${SCALE_RAE_ENV}/bin/python"
else
    PYTHON="$(which python)"
fi

export PYTHONNOUSERSITE=1
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

PRETRAIN_ARGS=()
if [ -n "${PRETRAIN_CKPT}" ] && [ -f "${PRETRAIN_CKPT}" ]; then
    PRETRAIN_ARGS+=(--pretrain_adapter_and_vision_head "${PRETRAIN_CKPT}")
fi

COCO_ARGS=()
if [ -n "${COCO_ANNOTATION}" ]; then
    COCO_ARGS+=(--coco_annotation_path "${COCO_ANNOTATION}")
fi

INPAINT_ARGS=()
if [ -n "${INPAINT_DATA_PATH}" ]; then
    INPAINT_ARGS+=(--aurora_inpaint_data_path "${INPAINT_DATA_PATH}")
    INPAINT_ARGS+=(--aurora_inpaint_image_folder "${INPAINT_IMAGE_FOLDER}")
fi

NORM_STATS_ARGS=()
if [ -n "${DIFFUSION_NORM_STATS_PATH}" ] && [ -f "${DIFFUSION_NORM_STATS_PATH}" ]; then
    NORM_STATS_ARGS+=(--diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH}")
fi

echo "===== AURORA v2 Training ====="
echo "Model:  $MODEL_PATH"
echo "Recon:  $RECON_IMAGE_FOLDER"
echo "Inpaint manifest: ${INPAINT_DATA_PATH:-<disabled>}"
echo "Stage: ${AURORA_TRAINING_STAGE}"
echo "Output: $OUTPUT_DIR"
echo "GPUs:   $NUM_GPUS"
echo "Diffusion conditioner trainable: $AURORA_TRAIN_DIFFUSION_CONDITION"
echo "===================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29501 \
    -m scale_rae.train.aurora_trainer \
    \
    --model_name_or_path "${MODEL_PATH}" \
    "${PRETRAIN_ARGS[@]}" \
    --version qwen_2 \
    \
    --vision_tower_aux_list '["google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[256]' \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
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
    --use_aurora True \
    --aurora_max_slots "${AURORA_MAX_SLOTS}" \
    --aurora_n_register "${AURORA_N_REGISTER}" \
    --aurora_cmd_length "${AURORA_CMD_LENGTH}" \
    --aurora_mask_loss_weight "${AURORA_MASK_LOSS_WEIGHT}" \
    --aurora_diversity_loss_weight "${AURORA_DIVERSITY_LOSS_WEIGHT}" \
    --aurora_inpaint_weight "${AURORA_INPAINT_WEIGHT}" \
    --aurora_training_stage "${AURORA_TRAINING_STAGE}" \
    --aurora_fail_on_nan True \
    --aurora_train_diffusion_condition "${AURORA_TRAIN_DIFFUSION_CONDITION}" \
    \
    "${COCO_ARGS[@]}" \
    "${INPAINT_ARGS[@]}" \
    "${NORM_STATS_ARGS[@]}" \
    --aurora_include_inpainting "${AURORA_INCLUDE_INPAINTING}" \
    --aurora_reconstruction_image_folder "${RECON_IMAGE_FOLDER}" \
    --image_aspect_ratio square \
    --max_images_per_sample 1 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --diff_head_lr "${DIFF_HEAD_LR}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 "${BF16}" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --aurora_inpaint_warmup_steps "${AURORA_INPAINT_WARMUP_STEPS}" \
    --aurora_inpaint_ramp_steps "${AURORA_INPAINT_RAMP_STEPS}" \
    --aurora_eval_num_images "${AURORA_EVAL_NUM_IMAGES}" \
    --aurora_eval_log_image_count "${AURORA_EVAL_LOG_IMAGE_COUNT}" \
    --aurora_eval_visual_batch_size "${AURORA_EVAL_VISUAL_BATCH_SIZE}" \
    --aurora_eval_log_reconstructions "${AURORA_EVAL_LOG_RECONSTRUCTIONS}" \
    --aurora_eval_log_attention_overlays "${AURORA_EVAL_LOG_ATTENTION_OVERLAYS}" \
    --aurora_eval_attention_overlay_count "${AURORA_EVAL_ATTENTION_OVERLAY_COUNT}" \
    --aurora_eval_decoder_repo "${AURORA_EVAL_DECODER_REPO}" \
    --group_by_modality_length False \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to "${REPORT_TO}" \
    --run_name "${EXPERIMENT_NAME}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    2>&1 | tee "${OUTPUT_DIR}/train.log"
