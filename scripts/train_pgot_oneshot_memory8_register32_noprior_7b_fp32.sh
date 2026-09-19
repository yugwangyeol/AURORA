#!/usr/bin/env bash
# Cold-start PGOT O8/R32 on Qwen-7B + DiT-9.8B in full FP32.
# Four 90-GB GPUs are required for optimizer-state and activation headroom.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior_7b_fp32_scratch}"
export WANDB_NAME="${WANDB_NAME:-pgot_oneshot_memory8_register32_noprior_7b_fp32_scratch}"
export NUM_GPUS="${NUM_GPUS:-4}"
export BF16=False
export FP16=False
export TF32=True
export PGOT_DIFFUSION_FORCE_FP32=False
# Large variable-length PGOT batches fragment the native allocator when
# max_split_size_mb is forced. Expandable segments reduce fragmentation;
# per-microbatch gradient synchronization prevents unsharded accumulation.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PGOT_FSDP_SYNC_EACH_MICROBATCH=True
export PGOT_ACTIVATION_CPU_OFFLOAD="${PGOT_ACTIVATION_CPU_OFFLOAD:-False}"
# Legacy W&B image logging runs only on rank 0 and bypasses the root FSDP
# forward, so its child all-gathers are unsafe. Standalone TF/AR evaluation
# still reconstructs images; regular distributed validation loss is unchanged.
export PGOT_EVAL_LOG_RECON_IMAGES=0

exec bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_7b_bf16.sh"
