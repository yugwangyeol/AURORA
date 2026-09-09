#!/usr/bin/env bash
# Autoregressive (zero-shot) evaluation of E11 Capacity + DiT-16 on all four
# benchmarks in a single pass: COCO pix2cap val, VOC, MOVi-C, MOVi-E.
#
# COCO and the CODA datasets need different flags (JSONL + cached COCO instance
# masks + CODA centre crop versus the packaged dataset masks), so both call
# shapes live here rather than being forced into one.  Everything else is held
# identical to the per-dataset scripts this replaces, which keeps the numbers
# comparable with the runs already in outputs/.
#
# Usage:
#   GPU=0 bash scripts/eval_pgot_e11_capacity_dit16_all_ar.sh
#   GPU=0 MAX_SAMPLES=100 bash scripts/eval_pgot_e11_capacity_dit16_all_ar.sh
#   GPU=0 bash scripts/eval_pgot_e11_capacity_dit16_all_ar.sh voc movi-e
#
# `set -e` is deliberately NOT used: one dataset failing must not cancel the
# remaining hours of work.  Per-dataset status is collected and reported at the
# end, and the script exits non-zero if anything failed.
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16_ar_all}"
VAL_JSONL="${VAL_JSONL:-${PROJECT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-${PROJECT_ROOT}/data/coco_inst_mask_cache_coda512}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
AR_MAX_NEW_TOKENS="${AR_MAX_NEW_TOKENS:-512}"
DTYPE="${DTYPE:-fp32}"
# rFID needs the diffusion decoder and only has a reference distribution on
# COCO, so it stays a COCO-only metric, exactly as in the pilot script.
COMPUTE_RFID="${COMPUTE_RFID:-True}"
# Set to 1 to resume an interrupted sweep, skipping datasets already finished.
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if (( $# == 0 )); then
    datasets=(coco voc movi-c movi-e)
else
    datasets=("$@")
fi

# ---- Preflight ------------------------------------------------------------
# Everything is checked before the first model load so a missing file cannot
# strand the sweep hours in.
fail=0
check() { test -e "$1" || { echo "Missing required path: $1" >&2; fail=1; }; }
check "${PYTHON}"
check "${MODEL_PATH}"
for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        coco)
            check "${VAL_JSONL}"
            check "${COCO_MASK_CACHE}/meta.json"
            ;;
        voc|movi-c|movi-e)
            check "${DATA_ROOT}/${dataset}/.pgot_download_complete"
            ;;
        *)
            echo "Unknown dataset: ${dataset} (expected coco, voc, movi-c, movi-e)" >&2
            fail=1
            ;;
    esac
done
(( fail == 0 )) || exit 1

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

common_args=(
    --model_path "${MODEL_PATH}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --grid_size 32
    --eval_size 512
    --max_caption_tokens 1024
    --n_ovt_per_object 1
    --max_objects 50
    --caption_mode autoregressive
    --ar_max_new_tokens "${AR_MAX_NEW_TOKENS}"
    --readout ovt_owner
    --eval_merge mean
    --register_eval_route unrestricted
    --dtype "${DTYPE}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    common_args+=(--max_samples "${MAX_SAMPLES}")
fi

mkdir -p "${OUTPUT_ROOT}"
statuses=()
started_at="$(date +%s)"

for dataset in "${datasets[@]}"; do
    output_dir="${OUTPUT_ROOT}/${dataset}"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${output_dir}/summary.json" ]]; then
        echo "===== ${dataset}: summary.json exists, skipping ====="
        statuses+=("${dataset} SKIPPED")
        continue
    fi

    dataset_args=()
    case "${dataset}" in
        coco)
            dataset_args=(
                --dataset pix2cap
                --val_jsonl "${VAL_JSONL}"
                --gt_source coco_instance
                --coco_mask_cache "${COCO_MASK_CACHE}"
                --image_preprocess_mode coda_center_crop
                --coda_crop_size 512
                --guidance_scale "${GUIDANCE_SCALE:-1.0}"
                --diffusion_inference_steps "${DIFFUSION_INFERENCE_STEPS:-10}"
            )
            [[ "${COMPUTE_RFID}" == True ]] && dataset_args+=(--compute_rfid)
            ;;
        *)
            dataset_args=(
                --dataset "${dataset}"
                --data_root "${DATA_ROOT}/${dataset}"
                --gt_source dataset_mask
            )
            ;;
    esac

    mkdir -p "${output_dir}"
    echo "===== E11 Capacity DiT-16 | ${dataset} | autoregressive ====="
    dataset_started="$(date +%s)"
    "${PYTHON}" -m pgot.eval.run_eval \
        --output_dir "${output_dir}" \
        "${common_args[@]}" \
        "${dataset_args[@]}" \
        2>&1 | tee "${output_dir}/eval.log"
    rc="${PIPESTATUS[0]}"
    elapsed=$(( $(date +%s) - dataset_started ))
    if (( rc == 0 )); then
        statuses+=("${dataset} OK ${elapsed}s")
    else
        statuses+=("${dataset} FAILED(rc=${rc}) ${elapsed}s")
        echo "!!!!! ${dataset} failed with exit code ${rc}; continuing !!!!!" >&2
    fi
done

# ---- Combined report ------------------------------------------------------
echo
echo "===== sweep finished in $(( $(date +%s) - started_at ))s ====="
printf '%s\n' "${statuses[@]}"
echo
"${PYTHON}" - "${OUTPUT_ROOT}" "${datasets[@]}" <<'PY'
import json, os, sys

root, names = sys.argv[1], sys.argv[2:]
cols = ["fARI", "mBO", "mIoU", "ar_format_valid_rate",
        "ar_object_count_mae", "rFID"]
head = f"{'dataset':<9}{'n':>6}" + "".join(f"{c:>22}" for c in cols)
print(head)
print("-" * len(head))
for name in names:
    path = os.path.join(root, name, "summary.json")
    if not os.path.exists(path):
        print(f"{name:<9}{'-':>6}" + f"{'(no summary.json)':>22}")
        continue
    s = json.load(open(path))
    row = f"{name:<9}{s.get('num_samples', '-'):>6}"
    for c in cols:
        v = s.get(c)
        row += f"{'-' if v is None else format(v, '.4f'):>22}"
    print(row)
PY

for status in "${statuses[@]}"; do
    [[ "${status}" == *FAILED* ]] && exit 1
done
exit 0
