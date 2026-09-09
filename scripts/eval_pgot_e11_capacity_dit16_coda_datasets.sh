#!/usr/bin/env bash
# Zero-shot autoregressive evaluation on CODA's VOC, MOVi-C, and MOVi-E sets.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_e11_capacity_dit16/checkpoint-10000}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/data}"
GPU="${GPU:-0}"

if (( $# == 0 )); then
    datasets=(voc movi-c movi-e)
else
    datasets=("$@")
fi

test -d "${MODEL_PATH}" || { echo "Missing model: ${MODEL_PATH}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"

extra_args=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    extra_args+=(--max_samples "${MAX_SAMPLES}")
fi

for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        voc|movi-c|movi-e) ;;
        *) echo "Unknown dataset: ${dataset}" >&2; exit 2 ;;
    esac
    dataset_root="${DATA_ROOT}/${dataset}"
    output_dir="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/eval_pgot_e11_capacity_dit16}_${dataset}_ar"
    test -f "${dataset_root}/.pgot_download_complete" || {
        echo "Dataset is not complete: ${dataset_root}" >&2
        exit 1
    }
    mkdir -p "${output_dir}"
    echo "===== E11 Capacity DiT-16 | ${dataset} | autoregressive ====="
    "${PYTHON}" -m pgot.eval.run_eval \
        --model_path "${MODEL_PATH}" \
        --dataset "${dataset}" \
        --data_root "${dataset_root}" \
        --output_dir "${output_dir}" \
        --batch_size "${BATCH_SIZE:-1}" \
        --num_workers "${NUM_WORKERS:-2}" \
        --grid_size 32 \
        --eval_size 512 \
        --max_caption_tokens 1024 \
        --n_ovt_per_object 1 \
        --max_objects 50 \
        --caption_mode autoregressive \
        --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
        --readout ovt_owner \
        --eval_merge mean \
        --gt_source dataset_mask \
        --register_eval_route unrestricted \
        --dtype "${DTYPE:-fp32}" \
        "${extra_args[@]}" \
        2>&1 | tee "${output_dir}/eval.log"
done
