#!/usr/bin/env bash
# Full-FP32, full-microbatch smoke for train/eval/save/W&B/standalone eval.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_oneshot.XXXXXX")"
BASE_MODEL="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
TRAIN_JSONL="${TRAIN_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_thing_train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE:-6}"
SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-24}"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_oneshot."*) ;;
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
        echo "One-shot smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE=offline
export WANDB_PROJECT=PGOT

run_smoke() {
    local mode="$1"
    local train_script="$2"
    local eval_script="$3"
    local port="$4"
    local train_dir="${SMOKE_ROOT}/${mode}_checkpoint"
    local eval_dir="${SMOKE_ROOT}/${mode}_eval"
    local checkpoint="${train_dir}/checkpoint-1"
    mkdir -p "${train_dir}" "${eval_dir}"

    WANDB_NAME="smoke_oneshot_${mode}" WANDB_DIR="${train_dir}/wandb" \
    MASTER_PORT="${port}" MODEL_PATH="${BASE_MODEL}" OUTPUT_DIR="${train_dir}" \
    NUM_GPUS=2 PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE}" PER_DEVICE_EVAL_BATCH_SIZE=1 \
    MAX_STEPS=1 SAVE_STEPS=1 SAVE_TOTAL_LIMIT=1 SAVE_ONLY_MODEL=True \
    EVAL_STEPS=1 LOGGING_STEPS=1 EVAL_NUM_IMAGES=1 PGOT_EVAL_LOG_RECON_IMAGES=1 \
    DATALOADER_NUM_WORKERS=1 TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" \
    REPORT_TO=wandb \
        bash "${PROJECT_ROOT}/scripts/${train_script}" \
        2>&1 | tee "${train_dir}/smoke_train.log"

    test -f "${checkpoint}/config.json"
    test -f "${checkpoint}/trainer_state.json"
    test -f "${checkpoint}/training_args.bin"
    grep -q '"pgot_one_shot_reader_enable": true' "${checkpoint}/config.json"
    grep -q "\"pgot_one_shot_readout_mode\": \"${mode}\"" "${checkpoint}/config.json"
    grep -q 'one_shot_reader_enabled' "${train_dir}/smoke_train.log"
    grep -q 'one_shot_patch_entropy' "${train_dir}/smoke_train.log"
    grep -q 'lm_first_caption_token_supervised' "${train_dir}/smoke_train.log"
    grep -q 'eval_loss' "${train_dir}/smoke_train.log"
    grep -q 'train_runtime' "${train_dir}/smoke_train.log"
    grep -q '\[PGOT/Freeze\] DiT last 16 blocks unfrozen' "${train_dir}/smoke_train.log"
    if [[ "${mode}" == owner_masked ]]; then
        grep -q 'one_shot_hard_outside_mass' "${train_dir}/smoke_train.log"
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
    grep -q '"one_shot_reader_enabled": true' "${eval_dir}/summary.json"
    grep -q "\"one_shot_readout_mode\": \"${mode}\"" "${eval_dir}/summary.json"
    grep -q '"recon_mse"' "${eval_dir}/summary.json"
    echo "one-shot ${mode}: full-batch train/eval/save/W&B/standalone eval PASS"

    find "${train_dir}" -depth -delete
    find "${eval_dir}" -depth -delete
}

run_smoke pooled train_pgot_oneshot_pooled.sh eval_pgot_oneshot_pooled.sh 29631
run_smoke owner_masked train_pgot_oneshot_owner_masked.sh eval_pgot_oneshot_owner_masked.sh 29632

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "Both one-shot smokes passed at per-device batch ${SMOKE_PER_DEVICE_TRAIN_BATCH_SIZE}; smoke outputs were removed."
