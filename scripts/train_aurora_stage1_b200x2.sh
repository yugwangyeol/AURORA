#!/bin/bash
# =============================================================================
# AURORA v2 Stage 1 Training: Object-Centric Decomposition (B200 x 2)
# - L_total = L_reconstruction + λ_mask * L_mask + λ_div * L_diversity
# - Frozen: LLM backbone, SigLIP2, mm_projector, rae_query (latent_queries)
# - Trainable: cmd_embeddings, obj_embedding_pool, reg_embeddings, DiT AdaLN
# =============================================================================

MODEL_PATH="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"

# Data paths
RECON_IMAGE_FOLDER="/home/jovyan/data/coco/train2017"
COCO_ANNOTATION="/home/jovyan/data/coco/annotations/instances_train2017.json"

# Output
OUTPUT_DIR="./checkpoints/aurora_v2_stage1"
EXPERIMENT_NAME="aurora_v2_stage1_b200x2"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-must}"

# ---- Hyperparameters for B200 x 2 ----
NUM_GPUS=2
PER_DEVICE_TRAIN_BATCH_SIZE=32
PER_DEVICE_EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4

NUM_TRAIN_EPOCHS=100
MAX_STEPS=-1

LEARNING_RATE=1e-4
DIFF_HEAD_LR=5e-5
WARMUP_RATIO=0.05

LOGGING_STEPS=10
SAVE_STEPS=1000
EVAL_STEPS=500

# ---- AURORA v2 Stage 1 specific ----
AURORA_TRAINING_STAGE=1
AURORA_MAX_SLOTS=10
AURORA_N_REGISTER=8
AURORA_CMD_LENGTH=8
AURORA_MASK_LOSS_WEIGHT=1.0
AURORA_DIVERSITY_LOSS_WEIGHT=0.1
AURORA_GRAD_CLIP_MAX_NORM=1.0

export WANDB_PROJECT="AURORA_v2"
export PYTHONNOUSERSITE=1

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

CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

resolve_resume_checkpoint() {
    local resume_path="$1"
    if [ -z "${resume_path}" ]; then
        return 0
    fi
    if [ "${resume_path}" = "latest" ]; then
        local latest_ckpt
        latest_ckpt="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
        if [ -z "${latest_ckpt}" ]; then
            echo "No checkpoint-* directory found under ${OUTPUT_DIR}" >&2
            return 1
        fi
        echo "${latest_ckpt}"
        return 0
    fi
    echo "${resume_path}"
}

RESUME_ARGS=()
TEE_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    RESOLVED_RESUME_CHECKPOINT="$(resolve_resume_checkpoint "${RESUME_FROM_CHECKPOINT}")" || exit 1
    if [ ! -f "${RESOLVED_RESUME_CHECKPOINT}/trainer_state.json" ]; then
        echo "Checkpoint is missing trainer_state.json: ${RESOLVED_RESUME_CHECKPOINT}" >&2
        exit 1
    fi
    RESUME_ARGS+=(--resume_from_checkpoint "${RESOLVED_RESUME_CHECKPOINT}")
    TEE_ARGS+=(-a)
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

if [ -n "${WANDB_RUN_ID}" ]; then
    export WANDB_RUN_ID
    export WANDB_RESUME="${WANDB_RESUME_MODE}"
fi

echo "===== AURORA v2 Stage 1: Object-Centric Decomposition (B200 x 2) ====="
echo "Model:          ${MODEL_PATH}"
echo "GPUs:           ${NUM_GPUS}"
echo "Effective batch: $((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs:         ${NUM_TRAIN_EPOCHS}"
echo "LR:             ${LEARNING_RATE}"
echo "K_max:          ${AURORA_MAX_SLOTS}"
echo "λ_mask:         ${AURORA_MASK_LOSS_WEIGHT}"
echo "λ_div:          ${AURORA_DIVERSITY_LOSS_WEIGHT}"
echo "Output:         ${OUTPUT_DIR}"
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    echo "Resume:         ${RESOLVED_RESUME_CHECKPOINT}"
else
    echo "Resume:         disabled"
fi
echo "================================================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29501 \
    -m scale_rae.train.aurora_trainer \
    \
    --model_name_or_path "${MODEL_PATH}" \
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
    --aurora_training_stage "${AURORA_TRAINING_STAGE}" \
    --aurora_grad_clip_max_norm "${AURORA_GRAD_CLIP_MAX_NORM}" \
    --aurora_fail_on_nan True \
    \
    --coco_annotation_path "${COCO_ANNOTATION}" \
    --aurora_include_inpainting False \
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
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --aurora_eval_num_images 100 \
    --aurora_eval_log_image_count 50 \
    --aurora_eval_visual_batch_size 4 \
    --aurora_eval_log_reconstructions True \
    --aurora_eval_decoder_repo "nyu-visionx/siglip2_decoder" \
    --group_by_modality_length False \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to wandb \
    --run_name "${EXPERIMENT_NAME}" \
    "${RESUME_ARGS[@]}" \
    --dataloader_num_workers 16 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    2>&1 | tee "${TEE_ARGS[@]}" "${OUTPUT_DIR}/train_stage1.log"
