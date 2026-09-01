#!/usr/bin/env bash
# E8.1 clean semantic-key / image-only visual-memory bottleneck.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_e8_1_clean}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
DIFFUSION_NORM_STATS_PATH="${DIFFUSION_NORM_STATS_PATH:-/home/jovyan/data/siglip2_bn_stats.pt}"

NUM_GPUS="${NUM_GPUS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_GLOBAL_BATCH=$((NUM_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by ${MICRO_GLOBAL_BATCH}" >&2
    exit 1
fi
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))}"

for path in "${MODEL_PATH}" "${TRAIN_JSONL}" "${VAL_JSONL}" "${DIFFUSION_NORM_STATS_PATH}"; do
    test -e "${path}" || { echo "Missing required path: ${path}" >&2; exit 1; }
done

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="${WANDB_PROJECT:-PGOT}"
export WANDB_NAME="${WANDB_NAME:-pgot_e8_1_clean}"
mkdir -p "${OUTPUT_DIR}"

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

echo "E8 model: ${MODEL_PATH}"
echo "E8 output: ${OUTPUT_DIR}"
echo "E8 writer layers: ${E8_LAYERS:-21,24,27}"
echo "E8/E9 update mode: ${E8_UPDATE_MODE:-separate_memory}"
echo "E10 raw source-SigLIP values: ${E10_RAW_VALUE_ENABLE:-False}"
echo "E11 Dual-M4: ${E11_DUAL_M4_ENABLE:-False}; memories/owner=${E11_MEMORIES_PER_OWNER:-4}"
echo "E11 heterogeneous memories: object=${E11_OBJECT_MEMORIES_PER_OWNER:-0}; register=${E11_REGISTER_MEMORIES_PER_OWNER:-0}; query_separation=${E11_QUERY_SEPARATION_ENABLE:-False}"
echo "E12 centroid-aware Reader: ${E12_CENTROID_READER_ENABLE:-False}; gate_init=${E12_CENTROID_GATE_INIT:-0.0}"
if [[ "${E8_UPDATE_MODE:-separate_memory}" == final_ovt ]]; then
    echo "E9.1 update: final post-Qwen OVT/register states are Reader K/V and causal targets"
elif [[ "${E8_UPDATE_MODE:-separate_memory}" == unified_gru ]]; then
    echo "E9 update: OVT/register hidden states updated in-place; Reader Value=last visual residual"
else
    echo "E8 clean refinement: enabled; memory-to-Qwen injection disabled"
fi
echo "E8 supervision: writer_weight=${E8_OWNER_WEIGHT:-1.0}; reader_mode=${E8_READER_SUPERVISION_MODE:-gt}"
echo "E8.2 paired causal: ${E8_CAUSAL_ENABLE:-False}"
echo "E8 background: ${E8_N_REGISTER:-4} competitive registers; NULL_BG disabled"

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT:-29548}" \
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
    --dataset_format coco_instance \
    --pgot_n_register "${E8_N_REGISTER:-4}" --pgot_n_null_bg 0 \
    --pgot_n_ovt_per_object 1 --pgot_max_objects 50 \
    --pgot_use_null_bg_competition False \
    --pgot_register_attends_caption False \
    --pgot_ovt_caption_init True --pgot_ovt_caption_init_scale 1.0 \
    --pgot_ovt_isolated_attention True --pgot_ovt_attends_own_caption True \
    --pgot_e8_visual_memory_enable True \
    --pgot_e8_layers "${E8_LAYERS:-21,24,27}" \
    --pgot_e8_owner_temperature "${E8_OWNER_TEMPERATURE:-1.0}" \
    --pgot_e8_owner_weight "${E8_OWNER_WEIGHT:-1.0}" \
    --pgot_e8_owner_bg_weight "${E8_OWNER_BG_WEIGHT:-0.5}" \
    --pgot_e8_reader_num_heads "${E8_READER_HEADS:-8}" \
    --pgot_e8_reader_num_layers "${E8_READER_LAYERS:-1}" \
    --pgot_e8_reader_temperature "${E8_READER_TEMPERATURE:-1.0}" \
    --pgot_e8_clean_refinement True --pgot_e8_inject_memory False \
    --pgot_e8_update_mode "${E8_UPDATE_MODE:-separate_memory}" \
    --pgot_e10_raw_value_enable "${E10_RAW_VALUE_ENABLE:-False}" \
    --pgot_e11_dual_m4_enable "${E11_DUAL_M4_ENABLE:-False}" \
    --pgot_e11_memories_per_owner "${E11_MEMORIES_PER_OWNER:-4}" \
    --pgot_e11_object_memories_per_owner "${E11_OBJECT_MEMORIES_PER_OWNER:-0}" \
    --pgot_e11_register_memories_per_owner "${E11_REGISTER_MEMORIES_PER_OWNER:-0}" \
    --pgot_e11_query_separation_enable "${E11_QUERY_SEPARATION_ENABLE:-False}" \
    --pgot_e12_centroid_reader_enable "${E12_CENTROID_READER_ENABLE:-False}" \
    --pgot_e12_centroid_gate_init "${E12_CENTROID_GATE_INIT:-0.0}" \
    --pgot_e9_update_dim "${E9_UPDATE_DIM:-512}" \
    --pgot_e9_mlp_ratio "${E9_MLP_RATIO:-2.0}" \
    --pgot_e8_reader_supervision_mode "${E8_READER_SUPERVISION_MODE:-gt}" \
    --pgot_e8_reader_object_weight "${E8_READER_OBJECT_WEIGHT:-0.5}" \
    --pgot_e8_reader_background_weight "${E8_READER_BACKGROUND_WEIGHT:-0.25}" \
    --pgot_e8_causal_enable "${E8_CAUSAL_ENABLE:-False}" \
    --pgot_e8_causal_margin "${E8_CAUSAL_MARGIN:-0.05}" \
    --pgot_e8_register_margin "${E8_REGISTER_MARGIN:-0.05}" \
    --pgot_e8_need_weight "${E8_NEED_WEIGHT:-0.1}" \
    --pgot_e8_local_weight "${E8_LOCAL_WEIGHT:-0.05}" \
    --pgot_e8_register_bg_weight "${E8_REGISTER_BG_WEIGHT:-0.1}" \
    --pgot_e8_register_fg_weight "${E8_REGISTER_FG_WEIGHT:-0.1}" \
    --pgot_e8_causal_batch_probability "${E8_CAUSAL_BATCH_PROBABILITY:-0.25}" \
    --pgot_e8_causal_ramp_steps "${E8_CAUSAL_RAMP_STEPS:-1000}" \
    --pgot_lm_loss_weight "${PGOT_LM_LOSS_WEIGHT:-1.0}" \
    --pgot_recon_loss_weight "${PGOT_RECON_LOSS_WEIGHT:-1.5}" \
    --pgot_mask_ce_weight 0.0 --pgot_mask_aux_competition_weight 0.0 \
    --pgot_mask_bce_weight 0.0 --pgot_mask_sigmoid_outside_weight 0.0 \
    --pgot_register_foreground_suppression_weight 0.0 \
    --pgot_mask_object_balanced_bce_weight 0.0 --pgot_mask_tversky_weight 0.0 \
    --pgot_mask_spatial_outside_weight 0.0 --pgot_mask_spatial_outside_log_weight 0.0 \
    --pgot_mask_llm_qk_outside_weight 0.0 --pgot_mask_llm_attention_outside_weight 0.0 \
    --pgot_mask_llm_patch_outside_weight 0.0 --pgot_mask_llm_image_use_weight 0.0 \
    --pgot_core_outside_weight 0.0 --pgot_core_register_outside_weight 0.0 \
    --pgot_e3_attention_competition_weight 0.0 \
    --pgot_v12_enable False --pgot_v14_enable False --pgot_v17_enable False \
    --pgot_v21_enable False --pgot_v22_attention_competition_weight 0.0 \
    --pgot_fvw_enable False --pgot_e6_enable False --pgot_e7_enable False \
    --pgot_e4_rae_isolated True --pgot_rae_attends_caption False --pgot_rae_bidirectional False \
    --pgot_cfg_drop_rate "${PGOT_CFG_DROP_RATE:-0.0}" \
    --pgot_contrastive_loss_target_weight 0.0 --pgot_contrastive_warmup_steps 0 \
    --pgot_unfreeze_mm_projector True --freeze_vision_tower True \
    --freeze_dit_body True --pgot_dit_unfreeze_last_n_blocks "${DIT_UNFREEZE_LAST_N:-8}" \
    --pgot_lora_enable True --pgot_lora_r 16 --pgot_lora_alpha 32 --pgot_lora_dropout 0.05 \
    --pgot_lora_target_modules q_proj,k_proj,v_proj,o_proj \
    --train_jsonl "${TRAIN_JSONL}" --val_jsonl "${VAL_JSONL}" \
    --image_preprocess_mode coda_center_crop --coda_crop_size 512 \
    --max_caption_tokens 1024 --grid_size 32 --eval_num_images "${EVAL_NUM_IMAGES:-32}" \
    --image_aspect_ratio square --output_dir "${OUTPUT_DIR}" \
    --max_steps "${MAX_STEPS:-10000}" --num_train_epochs 100 \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE:-5e-5}" --diff_head_lr "${DIFF_HEAD_LR:-3e-5}" \
    --pgot_dit_body_lr "${DIT_BODY_LR:-1e-5}" --pgot_mm_projector_lr "${MM_PROJECTOR_LR:-1e-5}" \
    --pgot_register_lr "${REGISTER_LR:-5e-5}" --pgot_rae_query_lr "${RAE_QUERY_LR:-5e-5}" \
    --pgot_llm_lr "${LLM_LR:-1e-4}" --weight_decay 0.01 \
    --warmup_ratio "${WARMUP_RATIO:-0.03}" --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False --pgot_use_wsd_schedule False \
    --max_grad_norm 1.0 --bf16 False --fp16 False --tf32 False \
    --model_max_length 4096 --gradient_checkpointing False \
    --save_strategy steps --save_steps "${SAVE_STEPS:-500}" --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
    --evaluation_strategy steps --eval_steps "${EVAL_STEPS:-500}" --prediction_loss_only True \
    --pgot_eval_log_recon_images "${PGOT_EVAL_LOG_RECON_IMAGES:-4}" \
    --logging_steps "${LOGGING_STEPS:-10}" --report_to "${REPORT_TO:-wandb}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" --remove_unused_columns False \
    --ddp_find_unused_parameters True "${RESUME_ARGS[@]}"
