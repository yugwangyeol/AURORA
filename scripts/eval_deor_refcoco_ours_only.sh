#!/usr/bin/env bash
# Generic DeOR-only RefCOCO inference. The caller audits checkpoint settings.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TOKENIZERS_PARALLELISM=false
# EVAL_GPUS="0 1" shards the DeOR stage across devices; GPU alone keeps one device.
read -r -a EVAL_DEVICES <<< "${EVAL_GPUS:-${GPU:?Set GPU (or EVAL_GPUS) to available CUDA indices or MIG UUIDs}}"
export CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[0]}"

PYTHON="${PGOT_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the trained DeOR checkpoint}"
REF_DATASET="${REF_DATASET:-refcoco}"
if [[ "${SMOKE:-0}" == 1 ]]; then
    SPLITS="${SPLITS:-val}"
    MAX_REFS="${MAX_REFS:-3}"
    default_output="${PROJECT_ROOT}/outputs/smoke_deor_${REF_DATASET}_ours_only"
else
    default_output="${PROJECT_ROOT}/outputs/eval_deor_${REF_DATASET}_ours_only"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output}}"
mkdir -p "${OUTPUT_ROOT}"

args=(--output_dir "${OUTPUT_ROOT}" --dataset "${REF_DATASET}"
      --ours_only
      --data_root "${REF_DATA_ROOT:-/home/jovyan/data/coco}"
      --image_root "${REF_IMAGE_ROOT:-/home/jovyan/data/coco}"
      --model_path "${MODEL_PATH}"
      --ctrlo_repo "${CTRLO_REPO:-/home/jovyan/baselines/CTRL-O}"
      --checkpoint_dir "${CTRLO_CHECKPOINT_DIR:-/home/jovyan/baselines/ckpt/ctrlo}"
      --supervision_label "${SUPERVISION_LABEL:-frozen checkpoints, no additional RefCOCO training}"
      --max_refs "${MAX_REFS:-0}" --max_samples "${MAX_SAMPLES:-0}"
      --eval_size "${EVAL_SIZE:-512}" --dtype "${DTYPE:-fp32}" --seed "${SEED:-0}"
      --batch_size "${BATCH_SIZE:-1}")
if [[ -n "${SPLITS:-}" ]]; then
    read -r -a split_args <<< "${SPLITS}"
    args+=(--splits "${split_args[@]}")
fi
if [[ -n "${SPLIT_BY:-}" ]]; then args+=(--split_by "${SPLIT_BY}"); fi

run_stage() {
    local stage="$1"
    printf '\n[%s] stage=%s dataset=%s\n' "$(date -u +%FT%TZ)" "${stage}" "${REF_DATASET}" | tee -a "${OUTPUT_ROOT}/run.log"
    "${PYTHON}" -u -m pgot.eval.run_refcoco_eval --stage "${stage}" "${args[@]}" 2>&1 | tee -a "${OUTPUT_ROOT}/run.log"
}

run_sharded_ours() {
    local n="${#EVAL_DEVICES[@]}" pids=() i failed=0
    printf '\n[%s] stage=ours dataset=%s shards=%s devices=%s\n' "$(date -u +%FT%TZ)" "${REF_DATASET}" "${n}" "${EVAL_DEVICES[*]}" | tee -a "${OUTPUT_ROOT}/run.log"
    for i in "${!EVAL_DEVICES[@]}"; do
        CUDA_VISIBLE_DEVICES="${EVAL_DEVICES[$i]}" "${PYTHON}" -u -m pgot.eval.run_refcoco_eval --stage ours "${args[@]}" \
            --num_shards "${n}" --shard_index "${i}" > "${OUTPUT_ROOT}/run.shard${i}.log" 2>&1 &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "DeOR shard ${i} failed; see ${OUTPUT_ROOT}/run.shard${i}.log" | tee -a "${OUTPUT_ROOT}/run.log" >&2; failed=1; }
    done
    return "${failed}"
}

trap 'printf "Stopped at line %s; see %s/run.log. Completed queries can be resumed.\n" "${LINENO}" "${OUTPUT_ROOT}" >&2' ERR
run_stage prepare
if (( ${#EVAL_DEVICES[@]} > 1 )); then run_sharded_ours; else run_stage ours; fi
run_stage score_ours
printf '\nDone: %s/summary.json\n' "${OUTPUT_ROOT}"
