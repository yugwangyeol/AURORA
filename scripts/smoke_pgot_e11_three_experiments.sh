#!/usr/bin/env bash
# FP32 smoke for train, in-train eval, save, W&B, and standalone eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_e11_three.XXXXXX")"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-$((2 * SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE))}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_e11_three."*) ;;
    *) echo "Refusing unsafe smoke root: ${SMOKE_ROOT}" >&2; exit 1 ;;
esac
cleanup_smoke() {
    if [[ -d "${SMOKE_ROOT}" ]]; then
        find "${SMOKE_ROOT}" -depth -delete
    fi
}
preserve_failed_smoke() {
    local status=$?
    if (( status != 0 )); then
        echo "Three-experiment smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT

run_smoke() {
    local variant="$1"
    local train_script="$2"
    local eval_script="$3"
    local expect_direct="$4"
    local expect_routed="$5"
    local train_dir="${SMOKE_ROOT}/${variant}_checkpoint"
    local eval_dir="${SMOKE_ROOT}/${variant}_eval"
    local checkpoint="${train_dir}/checkpoint-1"
    mkdir -p "${train_dir}" "${eval_dir}"

    WANDB_NAME="smoke_${variant}" WANDB_DIR="${train_dir}/wandb" \
    MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${train_dir}" \
    NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE}" \
    PER_DEVICE_EVAL_BATCH_SIZE=1 MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 \
    EVAL_STEPS=1 LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
    DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
    E8_CAUSAL_BATCH_PROBABILITY=1.0 E8_CAUSAL_RAMP_STEPS=1 REPORT_TO=wandb \
        bash "${PROJECT_ROOT}/scripts/${train_script}" \
        2>&1 | tee "${train_dir}/smoke_train.log"

    test -f "${checkpoint}/config.json"
    test -f "${checkpoint}/trainer_state.json"
    test -f "${checkpoint}/training_args.bin"
    grep -q '"pgot_e8_causal_enable": true' "${checkpoint}/config.json"
    grep -q '"pgot_e8_need_weight": 0.0' "${checkpoint}/config.json"
    grep -q '"pgot_e8_register_fg_weight": 0.0' "${checkpoint}/config.json"
    grep -q "\"pgot_dit_ovt_cross_attn_enable\": ${expect_direct}" "${checkpoint}/config.json"
    grep -q "\"pgot_dit_soft_routing_enable\": ${expect_routed}" "${checkpoint}/config.json"
    grep -q 'loss_e8_local_weighted' "${train_dir}/smoke_train.log"
    grep -q 'loss_e8_register_bg_weighted' "${train_dir}/smoke_train.log"
    grep -q 'lm_first_caption_token_supervised' "${train_dir}/smoke_train.log"
    grep -q 'eval_loss' "${train_dir}/smoke_train.log"
    grep -q 'train_runtime' "${train_dir}/smoke_train.log"
    grep -q '\[PGOT/Freeze\] DiT last 16 blocks unfrozen' "${train_dir}/smoke_train.log"
    if [[ "${expect_direct}" == true ]]; then
        grep -q 'dit_direct_memory_enabled' "${train_dir}/smoke_train.log"
        grep -q 'dit_direct_context_tokens' "${train_dir}/smoke_train.log"
        grep -q '\[LightningDiT\] Sparse slot/register cross-attention blocks: \[17, 18' "${train_dir}/smoke_train.log"
    fi

    local wandb_run
    wandb_run="$(find "${train_dir}/wandb" -maxdepth 3 -type d -name 'offline-run-*' | head -n 1)"
    test -n "${wandb_run}"
    test -n "$(find "${wandb_run}" -maxdepth 1 -type f -name 'run-*.wandb' | head -n 1)"
    test -n "$(find "${wandb_run}/files/media/table" -type f -name '*.table.json' | head -n 1)"

    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MODEL_PATH="${checkpoint}" \
    VAL_JSONL="${VAL_JSONL}" OUTPUT_DIR="${eval_dir}" MAX_SAMPLES=2 \
    BATCH_SIZE=1 NUM_WORKERS=1 DTYPE=fp32 DIFFUSION_INFERENCE_STEPS=2 \
    COMPUTE_RFID=True bash "${PROJECT_ROOT}/scripts/${eval_script}" \
        2>&1 | tee "${eval_dir}/smoke_eval.log"

    test -f "${eval_dir}/summary.json"
    grep -q '"recon_mse"' "${eval_dir}/summary.json"
    grep -q '"e8_visual_memory_bottleneck": true' "${eval_dir}/summary.json"
    echo "${variant}: train/eval/save/W&B/standalone eval PASS"
    # Each checkpoint is large; remove a verified variant before starting the
    # next one so the smoke never holds three full checkpoints at once.
    find "${train_dir}" -depth -delete
    find "${eval_dir}" -depth -delete
}

for variant in ${SMOKE_VARIANTS:-causal_only direct_causal direct_causal_routed}; do
    case "${variant}" in
        causal_only)
            run_smoke pgot_e11_causal_only \
                train_pgot_e11_causal_only.sh eval_pgot_e11_causal_only.sh false false
            ;;
        direct_causal)
            run_smoke pgot_e11_direct_causal \
                train_pgot_e11_direct_causal.sh eval_pgot_e11_direct_causal.sh true false
            ;;
        direct_causal_routed)
            run_smoke pgot_e11_direct_causal_routed \
                train_pgot_e11_direct_causal_routed.sh eval_pgot_e11_direct_causal_routed.sh true true
            ;;
        *) echo "Unknown smoke variant: ${variant}" >&2; exit 2 ;;
    esac
done

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "All three E11 experiment smokes passed; smoke outputs were removed."
