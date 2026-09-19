#!/usr/bin/env bash
# Full-condition 7B smoke: four GPUs, FP32, and all eight accumulation
# microsteps used by the production global batch of 32. Three optimizer steps
# cover both lazy Adam state creation and the next forward/backward with states
# resident. Reload the checkpoint for TF and AR inference before cleanup.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SMOKE_ROOT="$(mktemp -d "${PROJECT_ROOT}/.smoke_pgot_o8r32_7b_fp32.XXXXXX")"
TRAIN_OUTPUT="${SMOKE_ROOT}/checkpoint"
LOG_PATH="${SMOKE_ROOT}/train.log"

case "${SMOKE_ROOT}" in
    "${PROJECT_ROOT}/.smoke_pgot_o8r32_7b_fp32."*) ;;
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
        echo "7B FP32 smoke failed; outputs preserved at ${SMOKE_ROOT}" >&2
    fi
    return "${status}"
}
trap preserve_failed_smoke EXIT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=, read -r -a SMOKE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#SMOKE_DEVICES[@]} != 4 )); then
    echo "Set CUDA_VISIBLE_DEVICES to exactly four GPUs" >&2
    exit 2
fi

SMOKE_PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
SMOKE_BATCH="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
case "${SMOKE_BATCH}" in 1|2|4|8) ;; *) echo "Invalid smoke batch" >&2; exit 2 ;; esac
SMOKE_ACCUMULATION=$((8 / SMOKE_BATCH))
PGOT_PROBE_MICROBATCH="${SMOKE_BATCH}" \
PGOT_PROBE_TOKEN_MASK=1 PGOT_PARAMETER_UPDATE_AUDIT=True \
PGOT_TOKEN_MASK_RESULT="${SMOKE_ROOT}/token_mask_probe.json" \
LD_LIBRARY_PATH="$(dirname "${SMOKE_PYTHON}")/../lib:${LD_LIBRARY_PATH:-}" \
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
PYTHONNOUSERSITE=1 \
"${SMOKE_PYTHON}" -m torch.distributed.run \
    --nproc_per_node=4 --master_port="${SMOKE_PROBE_PORT:-29571}" \
    "${PROJECT_ROOT}/tests/test_pgot_fsdp_accumulation.py" \
    2>&1 | tee "${SMOKE_ROOT}/accumulation_probe.log"

mkdir -p "${TRAIN_OUTPUT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
OUTPUT_DIR="${TRAIN_OUTPUT}" \
WANDB_NAME=pgot_smoke_o8r32_7b_fp32 \
WANDB_MODE=offline \
REPORT_TO=none \
NUM_GPUS=4 \
PER_DEVICE_TRAIN_BATCH_SIZE="${SMOKE_BATCH}" \
GLOBAL_BATCH_SIZE=32 \
GRADIENT_ACCUMULATION_STEPS="${SMOKE_ACCUMULATION}" \
TRAIN_ENTRYPOINT="${PROJECT_ROOT}/train.py" PGOT_SKIP_FINAL_SAVE=0 \
MAX_STEPS=3 \
SAVE_STEPS=3 \
SAVE_TOTAL_LIMIT=1 \
EVAL_STEPS=3 \
EVAL_NUM_IMAGES=4 \
LOGGING_STEPS=1 \
DATALOADER_NUM_WORKERS=4 \
PGOT_PARAMETER_UPDATE_AUDIT=True \
PGOT_CUDA_MEMORY_DIAGNOSTICS=True \
bash "${PROJECT_ROOT}/scripts/train_pgot_oneshot_memory8_register32_noprior_7b_fp32.sh" \
    2>&1 | tee "${LOG_PATH}"

test -f "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_e11_object_memories_per_owner": 8' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_e11_register_memories_per_owner": 32' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_one_shot_direct_rae_query": false' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_one_shot_rae_query_adapter_enable": false' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_one_shot_memory_contrastive_enable": false' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q '"pgot_one_shot_memory_value_source": "siglip"' "${TRAIN_OUTPUT}/checkpoint-3/config.json"
grep -q "sync each microbatch; accumulation=${SMOKE_ACCUMULATION} unchanged" "${LOG_PATH}"
grep -q 'RAE=self-only MLLM' "${LOG_PATH}"
grep -q "'mllm_rae_query_tokens': 256.0" "${LOG_PATH}"
grep -q "'one_shot_direct_rae_query_enabled': 0.0" "${LOG_PATH}"
grep -q "'loss':" "${LOG_PATH}"
grep -q "'eval_loss':" "${LOG_PATH}"
if grep -Eiq 'Traceback|NVML_SUCCESS|out of memory|non-finite|nan' "${LOG_PATH}"; then
    echo "7B FP32 smoke log contains a fatal numerical or allocator error" >&2
    exit 1
fi

"${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}" - "${TRAIN_OUTPUT}" <<'PY'
import json
import pathlib
import sys

reports = sorted(pathlib.Path(sys.argv[1]).glob("parameter_update_audit.rank*.json"))
assert len(reports) == 4, f"Expected four rank audits, got {reports}"
groups = {}
for path in reports:
    report = json.loads(path.read_text())
    assert report["global_step"] == 3
    assert report["loss_finite_all_logged"], path
    for group, stats in report["groups"].items():
        assert stats["dtype"] == "float32", (path, group, stats)
        assert stats["gradient_finite_all_sampled"], (path, group, stats)
        groups.setdefault(group, []).append(stats)
expected = {"qwen_lora", "mm_projector", "dit_adaln", "dit_last16_body",
            "register", "rae_query", "writer", "reader"}
assert expected <= groups.keys(), f"Missing groups: {expected - groups.keys()}"
for group in expected:
    assert any(s["gradient_nonzero_any_sampled"] for s in groups[group]), group
    assert any(s["update_nonzero_any_sampled"] for s in groups[group]), group
print("FP32 audit passed: all eight parameter groups have finite gradients and actual updates.")
row_reports = sorted(pathlib.Path(sys.argv[1]).glob("token_row_mask_audit.rank*.json"))
assert len(row_reports) == 4
timings = []
for path in reports:
    report = json.loads(path.read_text())
    timings.append(report["steps"][-1]["seconds"])
for path in row_reports:
    rows = json.loads(path.read_text())
    assert len(rows) == 2, (path, rows)
    for name, stats in rows.items():
        assert stats["hook_calls"] > 0, (path, name, stats)
        assert stats["gradient_finite_all_sampled"], (path, name, stats)
        assert not stats["old_gradient_nonzero_any_sampled"], (path, name, stats)
        assert stats["new_gradient_nonzero_any_sampled"], (path, name, stats)
print("Token-row audit passed: embedding/head live hooks fire, old gradients zero, new gradients finite/nonzero.")
print(f"Third optimizer update (slowest rank): {max(timings):.3f}s; 5000-update compute estimate: {max(timings)*5000/3600:.2f}h.")
PY

for caption_mode in tf ar; do
    CUDA_VISIBLE_DEVICES="${SMOKE_DEVICES[0]}" \
    MODEL_PATH="${TRAIN_OUTPUT}/checkpoint-3" \
    OUTPUT_DIR="${SMOKE_ROOT}/${caption_mode}" \
    MAX_SAMPLES=2 BATCH_SIZE=1 AR_BATCH_SIZE=1 \
    COMPUTE_RFID=True COMPUTE_KID=True KID_SUBSETS=1 KID_SUBSET_SIZE=2 NUM_WORKERS=0 \
    bash "${PROJECT_ROOT}/scripts/eval_pgot_oneshot_memory8_register32_noprior_7b_fp32_${caption_mode}.sh" \
        2>&1 | tee "${SMOKE_ROOT}/${caption_mode}.log"
    test -f "${SMOKE_ROOT}/${caption_mode}/summary.json"
    "${SMOKE_PYTHON}" - "${SMOKE_ROOT}/${caption_mode}/summary.json" <<'PY'
import json
import math
import sys

summary = json.load(open(sys.argv[1]))
assert summary["num_samples"] == 2, summary["num_samples"]
for key in ("fARI", "mBO", "mIoU", "rFID", "KID_mean", "recon_mse", "recon_psnr"):
    assert key in summary and math.isfinite(summary[key]), (key, summary.get(key))
assert not any(key.endswith("_error") for key in summary), summary
print("Evaluation smoke passed: segmentation + reconstruction + rFID/KID are finite.")
PY
done

cleanup_smoke
test ! -e "${SMOKE_ROOT}"
trap - EXIT
echo "PGOT O8/R32 7B FP32 smoke passed: 3 optimizer steps + TF + AR; outputs removed."
