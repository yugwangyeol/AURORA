#!/bin/bash
# =============================================================================
# PGOT — FULL TRAINING v6a (representation-level competition via last-layer Q/K)
#
# Method change vs. v5 (which still landed at bimodal trade-off):
#  - Diagnosis: v3/v4/v5 all put competition (softmax over OVTs) only at the
#    LOSS surface. The LLM forward itself uses standard attention (softmax over
#    keys) -> no zero-sum between OVTs at representation formation -> trade-off.
#  - v6a (auxiliary, staged): re-use the LLM's own LAST attention layer Q/K
#    projection matrices to compute an alternative score = Q_proj(OVT) @ K_proj(img).
#    Apply the SAME object-level competition CE on this new score. Gradient flows
#    back through last layer's q_proj/k_proj (LoRA) -> the model's actual attention
#    projections are nudged toward competition-friendly Q/K. v5 main loss kept as
#    anchor for stability. If this opens the trade-off, v6b adds a gated residual
#    update to OVT_hidden; v6c (last resort) modifies the attention forward.
#
# Carried over from v3:
#  - SigLIP-512 (1024 patches, 32x32 grid) + SigLIP-224 (256) diffusion target
#  - OVT-swap contrastive (last-layer re-run, caption preserved)
#  - SigLIP latent BatchNorm-stats normalization for the diffusion target
#  - CFG dropping rate 0.1
#  - rae_query attends OVT + register only (caption text blocked -> bottleneck)
#  - Attention temperature 0.5; LR scheduler constant_with_warmup; fp32
# =============================================================================

MODEL_PATH="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B"
TRAIN_JSONL="/home/jovyan/PGOT/data/pgot_train.jsonl"
VAL_JSONL="/home/jovyan/PGOT/data/pgot_val.jsonl"

OUTPUT_DIR="/home/jovyan/PGOT/checkpoints/pgot_main_v6"
EXPERIMENT_NAME="pgot_main_v6a_b200x2_qk_compete_aux"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
DIFFUSION_NORM_STATS_PATH="/home/jovyan/data/siglip2_bn_stats.pt"

# ---- B200 x 2 hyperparameters ----
NUM_GPUS=2
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-6}"   # smoke sweep: BS=8 OK single-GPU, BS=10 OOM. DDP OOM 시 BS=6 fallback.
PER_DEVICE_EVAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-6}"   # effective = 8*2*4 = 64

NUM_TRAIN_EPOCHS=30
MAX_STEPS=10000

# LRs
LEARNING_RATE=5e-5
DIFF_HEAD_LR=3e-5
DIT_BODY_LR=1e-5
REGISTER_LR=5e-5
RAE_QUERY_LR=5e-5
LLM_LR=1e-4
WARMUP_RATIO=0.03

LOGGING_STEPS=10
SAVE_STEPS=2000
EVAL_STEPS=1000

# Loss weights
PGOT_LM_LOSS_WEIGHT=1.0
# Mask loss: v5 main (object-level CE + register bg) + v6a aux (competition on
# last-layer Q/K projections). BCE/Tversky off.
PGOT_MASK_CE_WEIGHT=1.0
PGOT_MASK_CE_TEMPERATURE=1.0
PGOT_MASK_AUX_COMPETITION_WEIGHT=0.5   # v6a: 0 = pure v5; ~0.5 = balanced anchor + aux
PGOT_MASK_BCE_WEIGHT=0.0
PGOT_MASK_TVERSKY_WEIGHT=0.0
PGOT_MASK_TVERSKY_ALPHA=0.5
PGOT_MASK_TVERSKY_BETA=0.5
PGOT_RECON_LOSS_WEIGHT=1.5
PGOT_CONTRASTIVE_TARGET_WEIGHT=0.03
PGOT_CONTRASTIVE_WARMUP_STEPS=2000   # 2K steps of stabilization before contrastive
PGOT_CFG_DROP_RATE=0.1
PGOT_ATTENTION_TEMPERATURE=0.5
PGOT_DIT_UNFREEZE_LAST_N_BLOCKS=8

export WANDB_PROJECT="PGOT"
export PYTHONNOUSERSITE=1

mkdir -p "${OUTPUT_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

SCALE_RAE_ENV="${HOME}/.conda/envs/scale_rae"
PYTHON="${SCALE_RAE_ENV}/bin/python"
CONDA_LIB="$(dirname "${PYTHON}")/../lib"
export LD_LIBRARY_PATH="${CONDA_LIB}:${LD_LIBRARY_PATH}"

RESUME_ARGS=()
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    if [ "${RESUME_FROM_CHECKPOINT}" = "latest" ]; then
        RESUME_FROM_CHECKPOINT="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    fi
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    # PyTorch 2.6 made torch.load default to weights_only=True, which rejects
    # the numpy-backed RNG state pickled by HF Trainer checkpoints.
    # AURORA uses the same flag — disables the weights-only restriction.
    export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
fi

echo "===== PGOT Main v3 (full Tversky+CFG+ovt-swap) — B200 x 2 ====="
echo "Model:                  ${MODEL_PATH}"
echo "Effective batch:        $((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION_STEPS))"
echo "Vision towers:          SigLIP-512 (1024) + SigLIP-224 (256)"
echo "Mask grid:              32x32"
echo "Attention temperature:  ${PGOT_ATTENTION_TEMPERATURE}"
echo "Mask loss:              v5 main CE (w=${PGOT_MASK_CE_WEIGHT}) + v6a aux Q/K-competition CE (w=${PGOT_MASK_AUX_COMPETITION_WEIGHT}); bce=${PGOT_MASK_BCE_WEIGHT}, tversky=${PGOT_MASK_TVERSKY_WEIGHT}; ce_temp=${PGOT_MASK_CE_TEMPERATURE}"
echo "Recon w:                ${PGOT_RECON_LOSS_WEIGHT}"
echo "CFG drop rate:          ${PGOT_CFG_DROP_RATE}"
echo "Contrastive (ovt-swap): target=${PGOT_CONTRASTIVE_TARGET_WEIGHT}, warmup=${PGOT_CONTRASTIVE_WARMUP_STEPS}"
echo "rae attends caption:    False (OVT-only bottleneck)"
echo "==============================================================="

"${PYTHON}" -m torch.distributed.run \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port=29501 \
    "${PROJECT_ROOT}/train.py" \
    \
    --model_name_or_path "${MODEL_PATH}" \
    --use_pgot True \
    \
    --vision_tower_aux_list '["google/siglip2-so400m-patch16-512","google/siglip2-so400m-patch14-224"]' \
    --vision_tower_aux_token_len_list '[1024,256]' \
    --image_feature_token_len 1024 \
    --diffusion_target_token_len 256 \
    --diffusion_norm_stats_path "${DIFFUSION_NORM_STATS_PATH}" \
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
    --pgot_n_register 64 \
    --pgot_n_ovt_per_object 2 \
    --pgot_max_objects 50 \
    --pgot_lm_loss_weight "${PGOT_LM_LOSS_WEIGHT}" \
    --pgot_mask_ce_weight "${PGOT_MASK_CE_WEIGHT}" \
    --pgot_mask_ce_temperature "${PGOT_MASK_CE_TEMPERATURE}" \
    --pgot_mask_aux_competition_weight "${PGOT_MASK_AUX_COMPETITION_WEIGHT}" \
    --pgot_mask_bce_weight "${PGOT_MASK_BCE_WEIGHT}" \
    --pgot_mask_tversky_weight "${PGOT_MASK_TVERSKY_WEIGHT}" \
    --pgot_mask_tversky_alpha "${PGOT_MASK_TVERSKY_ALPHA}" \
    --pgot_mask_tversky_beta "${PGOT_MASK_TVERSKY_BETA}" \
    --pgot_recon_loss_weight "${PGOT_RECON_LOSS_WEIGHT}" \
    --pgot_cfg_drop_rate "${PGOT_CFG_DROP_RATE}" \
    --pgot_contrastive_loss_target_weight "${PGOT_CONTRASTIVE_TARGET_WEIGHT}" \
    --pgot_contrastive_warmup_steps "${PGOT_CONTRASTIVE_WARMUP_STEPS}" \
    --pgot_attention_use_layer_norm True \
    --pgot_attention_temperature "${PGOT_ATTENTION_TEMPERATURE}" \
    --pgot_rae_attends_caption False \
    --freeze_dit_body True \
    --pgot_dit_unfreeze_last_n_blocks "${PGOT_DIT_UNFREEZE_LAST_N_BLOCKS}" \
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
    --grid_size 32 \
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
    --pgot_dit_body_lr "${DIT_BODY_LR}" \
    --pgot_register_lr "${REGISTER_LR}" \
    --pgot_rae_query_lr "${RAE_QUERY_LR}" \
    --pgot_llm_lr "${LLM_LR}" \
    --weight_decay 0.01 \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type constant_with_warmup \
    --pgot_use_cosine_min_lr_schedule False \
    --pgot_use_wsd_schedule False \
    --max_grad_norm 1.0 \
    --bf16 False \
    --fp16 False \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing False \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps "${EVAL_STEPS}" \
    --prediction_loss_only True \
    --pgot_eval_log_recon_images 4 \
    --logging_steps "${LOGGING_STEPS}" \
    --report_to wandb \
    --run_name "${EXPERIMENT_NAME}" \
    --dataloader_num_workers 6 \
    --dataloader_persistent_workers True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters False \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"
