#!/bin/bash
# Clean PGOT core: full data, autoregressive <thing>/<stuff> caption + one OVT,
# exact all-layer/all-head outside binding, and optional residual VOID.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
PGOT_N_VOID="${PGOT_N_VOID:-1}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_core_alllayer_void${PGOT_N_VOID}_full}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MAX_STEPS="${MAX_STEPS:-10000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-100}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-32}"

CORE_OUTSIDE_WEIGHT="${CORE_OUTSIDE_WEIGHT:-1.0}"
CORE_OUTSIDE_LAYERS="${CORE_OUTSIDE_LAYERS:-all}"
CORE_OUTSIDE_TEMPERATURE="${CORE_OUTSIDE_TEMPERATURE:-1.0}"
CORE_VOID_WEIGHT="${CORE_VOID_WEIGHT:-1.0}"
CORE_TAIL_WEIGHT="${CORE_TAIL_WEIGHT:-0.0}"
CORE_TAIL_FRACTION="${CORE_TAIL_FRACTION:-0.1}"
PGOT_EVAL_LOG_RECON_IMAGES="${PGOT_EVAL_LOG_RECON_IMAGES:-4}"

if [ "${PGOT_N_VOID}" != "0" ] && [ "${PGOT_N_VOID}" != "1" ]; then
    echo "ERROR: PGOT_N_VOID must be 0 or 1, got ${PGOT_N_VOID}" >&2
    exit 1
fi
for manifest in "${TRAIN_JSONL}" "${VAL_JSONL}"; do
    if [ ! -f "${manifest}" ]; then
        echo "ERROR: missing manifest: ${manifest}" >&2
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_core_alllayer_void${PGOT_N_VOID}_full}"
mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    if [ "${RESUME_FROM_CHECKPOINT}" = "latest" ]; then
        RESUME_FROM_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi
    test -f "${RESUME_FROM_CHECKPOINT}/trainer_state.json"
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

echo "PGOT core full-data | 1024 grounding patches | one OVT | VOID=${PGOT_N_VOID}"
echo "outside=${CORE_OUTSIDE_WEIGHT}, layers=${CORE_OUTSIDE_LAYERS}, void_w=${CORE_VOID_WEIGHT}, tail=${CORE_TAIL_WEIGHT}"

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29531}" \
    "${PROJECT_ROOT}/train.py" \
    --model_name_or_path "${MODEL_PATH}" --use_pgot True \
    --vision_tower_aux_list '["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[1024,256]' \
    --image_feature_token_len 1024 --diffusion_target_token_len 256 \
    --diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH}" \
    --vision_loss diffusion-loss --vision_loss_mode query --vision_coef 1.0 \
    --diffusion_model_hidden_size 2048 --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 --diffusion_model_depth 32 \
    --diffusion_model_heads 32 --dit_cls DiT \
    --pgot_n_register 0 --pgot_n_null_bg "${PGOT_N_VOID}" --pgot_n_ovt_per_object 1 --pgot_max_objects 50 \
    --pgot_lm_loss_weight 1.0 --pgot_recon_loss_weight 1.5 \
    --pgot_mask_ce_weight 0.0 --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight 0.0 --pgot_mask_sigmoid_outside_weight 0.0 \
    --pgot_mask_object_balanced_bce_weight 0.0 --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 --pgot_mask_spatial_outside_log_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 0.0 --pgot_mask_llm_attention_outside_weight 0.0 \
    --pgot_mask_llm_patch_outside_weight 0.0 --pgot_mask_llm_image_use_weight 0.0 \
    --pgot_core_outside_weight "${CORE_OUTSIDE_WEIGHT}" \
    --pgot_core_outside_layers "${CORE_OUTSIDE_LAYERS}" \
    --pgot_core_outside_temperature "${CORE_OUTSIDE_TEMPERATURE}" \
    --pgot_core_void_weight "${CORE_VOID_WEIGHT}" \
    --pgot_core_tail_weight "${CORE_TAIL_WEIGHT}" --pgot_core_tail_fraction "${CORE_TAIL_FRACTION}" \
    --pgot_v12_enable False --pgot_v14_enable False --pgot_v17_enable False --pgot_v21_enable False \
    --pgot_v22_attention_competition_weight 0.0 \
    --pgot_cfg_drop_rate 0.1 --pgot_contrastive_loss_target_weight 0.0 \
    --pgot_rae_attends_caption False --pgot_rae_bidirectional False \
    --pgot_unfreeze_mm_projector True --freeze_vision_tower True \
    --freeze_dit_body True --pgot_dit_unfreeze_last_n_blocks 8 \
    --pgot_lora_enable True --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --train_jsonl "${TRAIN_JSONL}" --val_jsonl "${VAL_JSONL}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --max_caption_tokens 2048 --grid_size 32 --eval_num_images "${EVAL_NUM_IMAGES}" \
    --image_aspect_ratio square --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS}" --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 5e-5 --diff_head_lr 3e-5 --pgot_dit_body_lr 1e-5 \
    --pgot_mm_projector_lr 1e-5 --pgot_register_lr 5e-5 \
    --pgot_rae_query_lr 5e-5 --pgot_llm_lr 1e-4 \
    --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False --pgot_use_wsd_schedule False \
    --max_grad_norm 1.0 --bf16 False --fp16 False --tf32 True \
    --model_max_length 4096 --gradient_checkpointing False \
    --save_strategy steps --save_steps "${SAVE_STEPS}" --save_total_limit 3 \
    --evaluation_strategy steps --eval_steps "${EVAL_STEPS}" --prediction_loss_only True \
    --pgot_eval_log_recon_images "${PGOT_EVAL_LOG_RECON_IMAGES}" \
    --logging_steps "${LOGGING_STEPS}" --report_to wandb \
    --dataloader_num_workers 4 --remove_unused_columns False \
    --ddp_find_unused_parameters True "${RESUME_ARGS[@]}"
