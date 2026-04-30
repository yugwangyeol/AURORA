#!/bin/bash
# =============================================================================
# RefCOCO-family CaptionSlot training script
# =============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
OUTPUT_DIR="${OUTPUT_DIR:-./checkpoints/captionslot_stage1}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
MAX_STEPS="${MAX_STEPS:-40000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-16}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
REPORT_TO="${REPORT_TO:-wandb}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-False}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
DIFF_HEAD_LR="${DIFF_HEAD_LR:-3e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
BF16="${BF16:-False}"
TF32="${TF32:-True}"
CAPTIONSLOT_LATENT_QUERY_LR="${CAPTIONSLOT_LATENT_QUERY_LR:-}"
CAPTIONSLOT_LLM_LR="${CAPTIONSLOT_LLM_LR:-}"

VISION_TOWER_AUX_LIST="${VISION_TOWER_AUX_LIST:-[\"google/siglip2-so400m-patch16-512\",\"google/siglip2-so400m-patch14-224\"]}"
VISION_TOWER_AUX_TOKEN_LEN_LIST="${VISION_TOWER_AUX_TOKEN_LEN_LIST:-[1024,256]}"
IMAGE_FEATURE_TOKEN_LEN="${IMAGE_FEATURE_TOKEN_LEN:-1024}"
DIFFUSION_TARGET_TOKEN_LEN="${DIFFUSION_TARGET_TOKEN_LEN:-256}"

CAPTIONSLOT_MAX_SLOTS="${CAPTIONSLOT_MAX_SLOTS:-48}"
CAPTIONSLOT_SLOTS_PER_OBJECT="${CAPTIONSLOT_SLOTS_PER_OBJECT:-4}"
CAPTIONSLOT_N_REGISTER="${CAPTIONSLOT_N_REGISTER:-64}"
CAPTIONSLOT_CMD_LENGTH="${CAPTIONSLOT_CMD_LENGTH:-8}"
CAPTIONSLOT_RECON_LOSS_WEIGHT="${CAPTIONSLOT_RECON_LOSS_WEIGHT:-1.0}"
CAPTIONSLOT_MASK_BCE_LOSS_WEIGHT="${CAPTIONSLOT_MASK_BCE_LOSS_WEIGHT:-1.0}"
CAPTIONSLOT_MASK_TVERSKY_LOSS_WEIGHT="${CAPTIONSLOT_MASK_TVERSKY_LOSS_WEIGHT:-1.0}"
CAPTIONSLOT_MASK_BALANCED_BCE="${CAPTIONSLOT_MASK_BALANCED_BCE:-False}"
CAPTIONSLOT_MASK_MERGE_MODE="${CAPTIONSLOT_MASK_MERGE_MODE:-mean}"
CAPTIONSLOT_MASK_TVERSKY_ALPHA="${CAPTIONSLOT_MASK_TVERSKY_ALPHA:-0.5}"
CAPTIONSLOT_MASK_TVERSKY_BETA="${CAPTIONSLOT_MASK_TVERSKY_BETA:-0.5}"
CAPTIONSLOT_OBJECT_CAM_LOSS_WEIGHT="${CAPTIONSLOT_OBJECT_CAM_LOSS_WEIGHT:-1.0}"
CAPTIONSLOT_REGISTER_CAM_LOSS_WEIGHT="${CAPTIONSLOT_REGISTER_CAM_LOSS_WEIGHT:-0.3}"
CAPTIONSLOT_CAM_LAYERS="${CAPTIONSLOT_CAM_LAYERS:--1}"
CAPTIONSLOT_CAM_EPS="${CAPTIONSLOT_CAM_EPS:-1e-6}"
CAPTIONSLOT_CAPTION_LOSS_WEIGHT="${CAPTIONSLOT_CAPTION_LOSS_WEIGHT:-0.0}"
CAPTIONSLOT_DIVERSITY_LOSS_WEIGHT="${CAPTIONSLOT_DIVERSITY_LOSS_WEIGHT:-0.0}"
CAPTIONSLOT_TRAINING_STAGE="${CAPTIONSLOT_TRAINING_STAGE:-1}"
CAPTIONSLOT_TRAIN_LATENT_QUERIES="${CAPTIONSLOT_TRAIN_LATENT_QUERIES:-True}"
CAPTIONSLOT_UNFREEZE_DIFF_HEAD_BODY="${CAPTIONSLOT_UNFREEZE_DIFF_HEAD_BODY:-False}"
CAPTIONSLOT_UNFREEZE_LLM_LAST_N_LAYERS="${CAPTIONSLOT_UNFREEZE_LLM_LAST_N_LAYERS:-0}"
CAPTIONSLOT_UNFREEZE_LLM_ATTN_ONLY="${CAPTIONSLOT_UNFREEZE_LLM_ATTN_ONLY:-True}"
CAPTIONSLOT_ATTENTION_USE_LAYERNORM="${CAPTIONSLOT_ATTENTION_USE_LAYERNORM:-True}"
CAPTIONSLOT_ATTENTION_TEMPERATURE="${CAPTIONSLOT_ATTENTION_TEMPERATURE:-1.0}"
CAPTIONSLOT_PRIOR_BIAS_SCALE="${CAPTIONSLOT_PRIOR_BIAS_SCALE:-0.0}"  # legacy no-op
CAPTIONSLOT_CONTROL_MODE="${CAPTIONSLOT_CONTROL_MODE:-slots}"
CAPTIONSLOT_RAE_BIDIRECTIONAL="${CAPTIONSLOT_RAE_BIDIRECTIONAL:-False}"
CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION="${CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION:-False}"
CAPTIONSLOT_ADD_CROSS_ATTENTION="${CAPTIONSLOT_ADD_CROSS_ATTENTION:-True}"
CAPTIONSLOT_CROSS_ATTENTION_START_BLOCK="${CAPTIONSLOT_CROSS_ATTENTION_START_BLOCK:-8}"
CAPTIONSLOT_CROSS_ATTENTION_EVERY_N_BLOCKS="${CAPTIONSLOT_CROSS_ATTENTION_EVERY_N_BLOCKS:-4}"
CAPTIONSLOT_CROSS_ATTENTION_INCLUDE_REGISTERS="${CAPTIONSLOT_CROSS_ATTENTION_INCLUDE_REGISTERS:-True}"
CAPTIONSLOT_USE_WSD_SCHEDULE="${CAPTIONSLOT_USE_WSD_SCHEDULE:-False}"
CAPTIONSLOT_WSD_DECAY_FRACTION="${CAPTIONSLOT_WSD_DECAY_FRACTION:-0.10}"
CAPTIONSLOT_USE_COSINE_MIN_LR_SCHEDULE="${CAPTIONSLOT_USE_COSINE_MIN_LR_SCHEDULE:-True}"
CAPTIONSLOT_MIN_LR_RATIO="${CAPTIONSLOT_MIN_LR_RATIO:-0.10}"
CAPTIONSLOT_LORA_ENABLE="${CAPTIONSLOT_LORA_ENABLE:-False}"
CAPTIONSLOT_LORA_R="${CAPTIONSLOT_LORA_R:-16}"
CAPTIONSLOT_LORA_ALPHA="${CAPTIONSLOT_LORA_ALPHA:-32}"
CAPTIONSLOT_LORA_DROPOUT="${CAPTIONSLOT_LORA_DROPOUT:-0.05}"
CAPTIONSLOT_LORA_TARGET_MODULES="${CAPTIONSLOT_LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}"
CAPTIONSLOT_EVAL_NUM_IMAGES="${CAPTIONSLOT_EVAL_NUM_IMAGES:-128}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-192}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"

CAPTIONSLOT_COCO_ROOT="${CAPTIONSLOT_COCO_ROOT:-/home/jovyan/data/coco}"
CAPTIONSLOT_DATASETS="${CAPTIONSLOT_DATASETS:-refcoco,refcoco+,refcocog}"
CAPTIONSLOT_TRAIN_SPLITS="${CAPTIONSLOT_TRAIN_SPLITS:-train}"
CAPTIONSLOT_EVAL_SPLITS="${CAPTIONSLOT_EVAL_SPLITS:-val}"
CAPTIONSLOT_MIN_AREA="${CAPTIONSLOT_MIN_AREA:-0}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

export WANDB_PROJECT="${WANDB_PROJECT:-CaptionSlot}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-captionslot_stage1}"

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
if [ -f "${SCALE_RAE_ENV}/bin/python" ]; then
    PYTHON="${SCALE_RAE_ENV}/bin/python"
else
    PYTHON="$(which python)"
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"

find_latest_checkpoint() {
    find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

NORM_STATS_ARGS=()
if [ -n "${DIFFUSION_NORM_STATS_PATH}" ] && [ -f "${DIFFUSION_NORM_STATS_PATH}" ]; then
    NORM_STATS_ARGS+=(--diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH}")
fi

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    case "${RESUME_FROM_CHECKPOINT}" in
        auto|latest|True|true)
            RESUME_FROM_CHECKPOINT="$(find_latest_checkpoint)"
            if [ -z "${RESUME_FROM_CHECKPOINT}" ]; then
                echo "Resume requested, but no checkpoint was found under ${OUTPUT_DIR}. Starting from scratch."
            fi
            ;;
    esac

    if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
        if [ ! -d "${RESUME_FROM_CHECKPOINT}" ]; then
            echo "Requested resume checkpoint does not exist: ${RESUME_FROM_CHECKPOINT}" >&2
            exit 1
        fi
        RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    fi
fi

TEE_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TEE_ARGS+=(-a)
fi

echo "===== CaptionSlot Training ====="
echo "Model:           ${MODEL_PATH}"
echo "COCO root:       ${CAPTIONSLOT_COCO_ROOT}"
echo "Datasets:        ${CAPTIONSLOT_DATASETS}"
echo "Train splits:    ${CAPTIONSLOT_TRAIN_SPLITS}"
echo "Eval splits:     ${CAPTIONSLOT_EVAL_SPLITS}"
echo "Stage:           ${CAPTIONSLOT_TRAINING_STAGE}"
echo "Output:          ${OUTPUT_DIR}"
echo "Resume:          ${RESUME_FROM_CHECKPOINT:-<disabled>}"
echo "GPUs:            ${NUM_GPUS}"
echo "Train batch/GPU: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Grad accum:      ${GRADIENT_ACCUMULATION_STEPS}"
echo "Vision towers:   ${VISION_TOWER_AUX_LIST}"
echo "Vision tokens:   image=${IMAGE_FEATURE_TOKEN_LEN} target=${DIFFUSION_TARGET_TOKEN_LEN}"
echo "Model max len:   ${MODEL_MAX_LENGTH}"
echo "Slots/object:    ${CAPTIONSLOT_SLOTS_PER_OBJECT}"
echo "Latent queries:  ${CAPTIONSLOT_TRAIN_LATENT_QUERIES}"
echo "LLM train:       last_n=${CAPTIONSLOT_UNFREEZE_LLM_LAST_N_LAYERS} attn_only=${CAPTIONSLOT_UNFREEZE_LLM_ATTN_ONLY} lr=${CAPTIONSLOT_LLM_LR:-<main>}"
echo "Cross-attn:      ${CAPTIONSLOT_ADD_CROSS_ATTENTION} (start=${CAPTIONSLOT_CROSS_ATTENTION_START_BLOCK}, every=${CAPTIONSLOT_CROSS_ATTENTION_EVERY_N_BLOCKS}, regs=${CAPTIONSLOT_CROSS_ATTENTION_INCLUDE_REGISTERS})"
echo "Attn fix:        rae_bidir=${CAPTIONSLOT_RAE_BIDIRECTIONAL} same_object_slots=${CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION}"
echo "Loss weights:    recon=${CAPTIONSLOT_RECON_LOSS_WEIGHT} obj_cam=${CAPTIONSLOT_OBJECT_CAM_LOSS_WEIGHT} reg_cam=${CAPTIONSLOT_REGISTER_CAM_LOSS_WEIGHT} bce=${CAPTIONSLOT_MASK_BCE_LOSS_WEIGHT} tversky=${CAPTIONSLOT_MASK_TVERSKY_LOSS_WEIGHT}"
echo "CAM attention:   layers=${CAPTIONSLOT_CAM_LAYERS} eps=${CAPTIONSLOT_CAM_EPS}"
echo "Mask loss:       balanced_bce=${CAPTIONSLOT_MASK_BALANCED_BCE} tversky_alpha=${CAPTIONSLOT_MASK_TVERSKY_ALPHA} tversky_beta=${CAPTIONSLOT_MASK_TVERSKY_BETA}"
echo "Scheduler:       cosine-min=${CAPTIONSLOT_USE_COSINE_MIN_LR_SCHEDULE} min_ratio=${CAPTIONSLOT_MIN_LR_RATIO} warmup=${WARMUP_RATIO} type=${LR_SCHEDULER_TYPE}"
echo "LoRA:            enable=${CAPTIONSLOT_LORA_ENABLE} r=${CAPTIONSLOT_LORA_R} alpha=${CAPTIONSLOT_LORA_ALPHA} dropout=${CAPTIONSLOT_LORA_DROPOUT} targets=${CAPTIONSLOT_LORA_TARGET_MODULES}"
echo "================================"

"${PYTHON}" -u -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m scale_rae.train.captionslot_trainer \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --version qwen_2 \
    \
    --vision_tower_aux_list "${VISION_TOWER_AUX_LIST}" \
    --vision_tower_aux_token_len_list "${VISION_TOWER_AUX_TOKEN_LEN_LIST}" \
    --image_feature_token_len "${IMAGE_FEATURE_TOKEN_LEN}" \
    --diffusion_target_token_len "${DIFFUSION_TARGET_TOKEN_LEN}" \
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
    --use_captionslot True \
    --captionslot_max_slots "${CAPTIONSLOT_MAX_SLOTS}" \
    --captionslot_slots_per_object "${CAPTIONSLOT_SLOTS_PER_OBJECT}" \
    --captionslot_n_register "${CAPTIONSLOT_N_REGISTER}" \
    --captionslot_cmd_length "${CAPTIONSLOT_CMD_LENGTH}" \
    --captionslot_recon_loss_weight "${CAPTIONSLOT_RECON_LOSS_WEIGHT}" \
    --captionslot_mask_bce_loss_weight "${CAPTIONSLOT_MASK_BCE_LOSS_WEIGHT}" \
    --captionslot_mask_tversky_loss_weight "${CAPTIONSLOT_MASK_TVERSKY_LOSS_WEIGHT}" \
    --captionslot_mask_balanced_bce "${CAPTIONSLOT_MASK_BALANCED_BCE}" \
    --captionslot_mask_merge_mode "${CAPTIONSLOT_MASK_MERGE_MODE}" \
    --captionslot_mask_tversky_alpha "${CAPTIONSLOT_MASK_TVERSKY_ALPHA}" \
    --captionslot_mask_tversky_beta "${CAPTIONSLOT_MASK_TVERSKY_BETA}" \
    --captionslot_object_cam_loss_weight "${CAPTIONSLOT_OBJECT_CAM_LOSS_WEIGHT}" \
    --captionslot_register_cam_loss_weight "${CAPTIONSLOT_REGISTER_CAM_LOSS_WEIGHT}" \
    --captionslot_cam_layers "${CAPTIONSLOT_CAM_LAYERS}" \
    --captionslot_cam_eps "${CAPTIONSLOT_CAM_EPS}" \
    --captionslot_caption_loss_weight "${CAPTIONSLOT_CAPTION_LOSS_WEIGHT}" \
    --captionslot_diversity_loss_weight "${CAPTIONSLOT_DIVERSITY_LOSS_WEIGHT}" \
    --captionslot_training_stage "${CAPTIONSLOT_TRAINING_STAGE}" \
    --captionslot_train_latent_queries "${CAPTIONSLOT_TRAIN_LATENT_QUERIES}" \
    --captionslot_unfreeze_diff_head_body "${CAPTIONSLOT_UNFREEZE_DIFF_HEAD_BODY}" \
    --captionslot_unfreeze_llm_last_n_layers "${CAPTIONSLOT_UNFREEZE_LLM_LAST_N_LAYERS}" \
    --captionslot_unfreeze_llm_attn_only "${CAPTIONSLOT_UNFREEZE_LLM_ATTN_ONLY}" \
    --captionslot_attention_use_layer_norm "${CAPTIONSLOT_ATTENTION_USE_LAYERNORM}" \
    --captionslot_attention_temperature "${CAPTIONSLOT_ATTENTION_TEMPERATURE}" \
    --captionslot_prior_bias_scale "${CAPTIONSLOT_PRIOR_BIAS_SCALE}" \
    --captionslot_control_mode "${CAPTIONSLOT_CONTROL_MODE}" \
    --captionslot_rae_bidirectional "${CAPTIONSLOT_RAE_BIDIRECTIONAL}" \
    --captionslot_same_object_slot_attention "${CAPTIONSLOT_SAME_OBJECT_SLOT_ATTENTION}" \
    --captionslot_add_cross_attention "${CAPTIONSLOT_ADD_CROSS_ATTENTION}" \
    --captionslot_cross_attention_start_block "${CAPTIONSLOT_CROSS_ATTENTION_START_BLOCK}" \
    --captionslot_cross_attention_every_n_blocks "${CAPTIONSLOT_CROSS_ATTENTION_EVERY_N_BLOCKS}" \
    --captionslot_cross_attention_include_registers "${CAPTIONSLOT_CROSS_ATTENTION_INCLUDE_REGISTERS}" \
    --captionslot_use_wsd_schedule "${CAPTIONSLOT_USE_WSD_SCHEDULE}" \
    --captionslot_wsd_decay_fraction "${CAPTIONSLOT_WSD_DECAY_FRACTION}" \
    --captionslot_use_cosine_min_lr_schedule "${CAPTIONSLOT_USE_COSINE_MIN_LR_SCHEDULE}" \
    --captionslot_min_lr_ratio "${CAPTIONSLOT_MIN_LR_RATIO}" \
    --captionslot_lora_enable "${CAPTIONSLOT_LORA_ENABLE}" \
    --captionslot_lora_r "${CAPTIONSLOT_LORA_R}" \
    --captionslot_lora_alpha "${CAPTIONSLOT_LORA_ALPHA}" \
    --captionslot_lora_dropout "${CAPTIONSLOT_LORA_DROPOUT}" \
    --captionslot_lora_target_modules "${CAPTIONSLOT_LORA_TARGET_MODULES}" \
    \
    "${NORM_STATS_ARGS[@]}" \
    --captionslot_coco_root "${CAPTIONSLOT_COCO_ROOT}" \
    --captionslot_datasets "${CAPTIONSLOT_DATASETS}" \
    --captionslot_train_splits "${CAPTIONSLOT_TRAIN_SPLITS}" \
    --captionslot_eval_splits "${CAPTIONSLOT_EVAL_SPLITS}" \
    --captionslot_min_area "${CAPTIONSLOT_MIN_AREA}" \
    --max_caption_tokens "${MAX_CAPTION_TOKENS}" \
    --image_aspect_ratio square \
    --max_images_per_sample 1 \
    \
    "${RESUME_ARGS[@]}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --diff_head_lr "${DIFF_HEAD_LR}" \
    ${CAPTIONSLOT_LATENT_QUERY_LR:+--captionslot_latent_query_lr "${CAPTIONSLOT_LATENT_QUERY_LR}"} \
    ${CAPTIONSLOT_LLM_LR:+--captionslot_llm_lr "${CAPTIONSLOT_LLM_LR}"} \
    --weight_decay 0.01 \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --max_grad_norm 1.0 \
    --bf16 "${BF16}" \
    --tf32 "${TF32}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --captionslot_eval_num_images "${CAPTIONSLOT_EVAL_NUM_IMAGES}" \
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
    2>&1 | tee "${TEE_ARGS[@]}" "${OUTPUT_DIR}/train.log"
