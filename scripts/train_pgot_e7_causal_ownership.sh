#!/usr/bin/env bash
# E7: caption-defined OVTs -> competitive owner/write -> clean SD-v1.5 bottleneck.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/jovyan/PGOT/checkpoints/pgot_e2_pix_ovt_isolated/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/jovyan/PGOT/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/PGOT/checkpoints/pgot_e7_causal_ownership_20k}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NUM_GPUS="${NUM_GPUS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_GLOBAL_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS*PER_DEVICE_TRAIN_BATCH_SIZE=${MICRO_GLOBAL_BATCH}" >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))}"
EFFECTIVE_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))

for path in "${MODEL_PATH}" "${TRAIN_JSONL}" "${VAL_JSONL}"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_e7_causal_ownership_20k}"
mkdir -p "${OUTPUT_DIR}"

echo "E7 initialization: ${MODEL_PATH}"
echo "Train/val: ${TRAIN_JSONL} / ${VAL_JSONL}"
echo "Batch: ${PER_DEVICE_TRAIN_BATCH_SIZE}/GPU x ${NUM_GPUS} GPU x accum ${GRADIENT_ACCUMULATION_STEPS} = ${EFFECTIVE_BATCH}"
echo "Condition: predicted-owner visual-written OVTs + 4 background registers"
echo "Decoder: original SD-v1.5 U-Net; trainable K/V LoRA only; no positive null tokens"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    RESUME_PATH="${RESUME_FROM_CHECKPOINT}"
    if [[ "${RESUME_PATH}" == latest ]]; then
        RESUME_PATH="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi
    test -f "${RESUME_PATH}/trainer_state.json"
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_PATH}")
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29547}" \
    "${PROJECT_ROOT}/train.py" \
    --model_name_or_path "${MODEL_PATH}" --use_pgot True \
    --vision_tower_aux_list '["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[1024,256]' \
    --image_feature_token_len 1024 --diffusion_target_token_len 256 \
    --diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}" \
    --vision_loss diffusion-loss --vision_loss_mode query --vision_coef 1.0 \
    --diffusion_model_hidden_size 2048 --diffusion_model_channels 1152 \
    --diffusion_model_z_channels 2048 --diffusion_model_depth 32 \
    --diffusion_model_heads 32 --dit_cls DiT \
    --dataset_format coco_instance \
    --pgot_n_register 4 --pgot_n_null_bg 0 \
    --pgot_n_ovt_per_object 1 --pgot_max_objects 50 \
    --pgot_ovt_caption_init True --pgot_ovt_caption_init_scale 1.0 \
    --pgot_ovt_isolated_attention True --pgot_ovt_attends_own_caption True \
    --pgot_register_attends_caption False \
    --pgot_register_hard_gt_mask True --pgot_register_hard_gt_mask_eval False \
    --pgot_register_hard_gt_mask_threshold 0.0 \
    --pgot_e6_enable False \
    --pgot_e7_enable True \
    --pgot_e7_model_id stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --pgot_e7_context_dim 768 --pgot_e7_router_dim 512 \
    --pgot_e7_owner_num_heads 8 --pgot_e7_owner_temperature 1.0 \
    --pgot_e7_owner_weight "${E7_OWNER_WEIGHT:-1.0}" \
    --pgot_e7_owner_bg_weight "${E7_OWNER_BG_WEIGHT:-0.25}" \
    --pgot_e7_cfg_drop_rate 0.1 \
    --pgot_e7_unet_lora_rank "${E7_UNET_LORA_RANK:-8}" \
    --pgot_e7_unet_lora_alpha "${E7_UNET_LORA_ALPHA:-8.0}" \
    --pgot_e7_causal_weight "${E7_CAUSAL_WEIGHT:-0.1}" \
    --pgot_e7_invariance_weight "${E7_INVARIANCE_WEIGHT:-0.05}" \
    --pgot_e7_causal_min_timestep "${E7_CAUSAL_MIN_TIMESTEP:-700}" \
    --pgot_e7_causal_margin 0.05 --pgot_e7_causal_temperature 0.1 \
    --pgot_lm_loss_weight 1.0 --pgot_recon_loss_weight "${RECON_LOSS_WEIGHT:-1.0}" \
    --pgot_mask_ce_weight 0.0 --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight 0.0 --pgot_mask_sigmoid_outside_weight 0.0 \
    --pgot_register_foreground_suppression_weight 0.0 \
    --pgot_mask_object_balanced_bce_weight 0.0 --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 --pgot_mask_spatial_outside_log_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 0.0 --pgot_mask_llm_attention_outside_weight 0.0 \
    --pgot_mask_llm_patch_outside_weight 0.0 --pgot_mask_llm_image_use_weight 0.0 \
    --pgot_core_outside_weight "${CORE_OUTSIDE_WEIGHT:-1.0}" \
    --pgot_core_register_outside_weight 0.0 \
    --pgot_core_outside_layers all --pgot_core_outside_temperature 1.0 \
    --pgot_core_void_weight 0.0 --pgot_core_tail_weight 0.0 \
    --pgot_e3_attention_competition_weight 0.0 \
    --pgot_e4_rae_isolated False --pgot_e4_full_inside_weight 0.0 \
    --pgot_e4_rae_bind_weight 0.0 --pgot_e5_forcing_probability 0.0 \
    --pgot_fvw_enable False --pgot_fvw_write_outside_weight 0.0 \
    --pgot_v12_enable False --pgot_v14_enable False --pgot_v17_enable False \
    --pgot_v21_enable False --pgot_v22_attention_competition_weight 0.0 \
    --pgot_contrastive_loss_target_weight 0.0 \
    --pgot_rae_attends_caption False --pgot_rae_bidirectional False \
    --pgot_unfreeze_mm_projector False --freeze_vision_tower True \
    --freeze_dit_body True --pgot_dit_unfreeze_last_n_blocks 0 \
    --pgot_lora_enable True --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --train_jsonl "${TRAIN_JSONL}" --val_jsonl "${VAL_JSONL}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --max_caption_tokens 1024 --grid_size 32 --eval_num_images "${EVAL_NUM_IMAGES:-32}" \
    --image_aspect_ratio square --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS:-20000}" --num_train_epochs 200 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate 5e-5 --pgot_register_lr 5e-5 --pgot_llm_lr 1e-4 \
    --pgot_e7_owner_lr "${E7_OWNER_LR:-5e-5}" \
    --pgot_e7_projector_lr "${E7_PROJECTOR_LR:-5e-5}" \
    --pgot_e7_unet_lr "${E7_UNET_LR:-1e-6}" \
    --weight_decay 0.01 --warmup_ratio 0.03 --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False --pgot_use_wsd_schedule False \
    --pgot_e7_causal_warmup_start_steps "${E7_CAUSAL_WARMUP_START_STEPS:-2000}" \
    --pgot_e7_causal_ramp_steps "${E7_CAUSAL_RAMP_STEPS:-3000}" \
    --max_grad_norm 1.0 --bf16 True --fp16 False --tf32 True \
    --model_max_length 4096 --gradient_checkpointing False \
    --save_strategy steps --save_steps "${SAVE_STEPS:-10000}" --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
    --evaluation_strategy steps --eval_steps "${EVAL_STEPS:-500}" --prediction_loss_only True \
    --pgot_eval_log_recon_images "${PGOT_EVAL_LOG_RECON_IMAGES:-4}" \
    --pgot_e7_eval_inference_steps "${E7_EVAL_INFERENCE_STEPS:-4}" \
    --pgot_e7_eval_guidance_scale "${E7_EVAL_GUIDANCE_SCALE:-2.5}" \
    --pgot_log_metric_profile e7 \
    --logging_steps "${LOGGING_STEPS:-10}" --report_to "${REPORT_TO:-wandb}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" --remove_unused_columns False \
    --ddp_find_unused_parameters True --seed "${SEED:-42}" "${RESUME_ARGS[@]}"
