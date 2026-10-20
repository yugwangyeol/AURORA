#!/usr/bin/env bash
# Image-only AR: COCO val ALL 5000 + VOC val ALL 1449; retain objectless images.
# Defaults to the checkpoint used in the user's existing memory8/reg32 results.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000}"
default_output="${PROJECT_ROOT}/outputs/segeval_pgot_memory8_full_coco_voc_ar"
if [[ "${SMOKE:-0}" == 1 ]]; then default_output="${PROJECT_ROOT}/outputs/_smoke_pgot_full_coco_voc_ar"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${default_output}}"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/data}"
COCO_MASK_CACHE="${COCO_MASK_CACHE:-${PROJECT_ROOT}/data/coco_inst_mask_cache_coda512}"
COCO_CATEGORY_MASK_CACHE="${COCO_CATEGORY_MASK_CACHE:-${PROJECT_ROOT}/data/coco_cat_mask_cache_coda512}"
COMPUTE_RFID="${COMPUTE_RFID:-False}"
COMPUTE_KID="${COMPUTE_KID:-False}"
RECON_METRIC_RESOLUTION="${RECON_METRIC_RESOLUTION:-0}"
ALSO_RECON_METRIC_RESOLUTION="${ALSO_RECON_METRIC_RESOLUTION:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$(dirname "${PYTHON}")/../lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-3}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-3}"
case "${COMPUTE_RFID}" in True|False|true|false|1|0) ;; *) echo "COMPUTE_RFID must be a boolean" >&2; exit 2;; esac
case "${COMPUTE_KID}" in True|False|true|false|1|0) ;; *) echo "COMPUTE_KID must be a boolean" >&2; exit 2;; esac
[[ "${RECON_METRIC_RESOLUTION}" =~ ^[0-9]+$ ]] || { echo "RECON_METRIC_RESOLUTION must be non-negative" >&2; exit 2; }
[[ "${ALSO_RECON_METRIC_RESOLUTION}" =~ ^[0-9]+$ ]] || { echo "ALSO_RECON_METRIC_RESOLUTION must be non-negative" >&2; exit 2; }
if [[ "${COMPUTE_KID,,}" =~ ^(true|1)$ && ! "${COMPUTE_RFID,,}" =~ ^(true|1)$ ]]; then
    echo "COMPUTE_KID requires COMPUTE_RFID" >&2; exit 2
fi
if (( ALSO_RECON_METRIC_RESOLUTION > 0 )) && [[ ! "${COMPUTE_RFID,,}" =~ ^(true|1)$ ]]; then
    echo "ALSO_RECON_METRIC_RESOLUTION requires COMPUTE_RFID" >&2; exit 2
fi
if (( $# == 0 )); then stages=(coco voc); else stages=("$@"); fi
test -f "${MODEL_PATH}/config.json"
mkdir -p "${OUTPUT_ROOT}"
for stage in "${stages[@]}"; do
    extra_args=()
    case "${stage}" in
        coco)
            default_batch=4
            stage_args=(--gt_source coco_instance --coco_mask_cache "${COCO_MASK_CACHE}"
                        --compute_class_metrics --coco_category_mask_cache "${COCO_CATEGORY_MASK_CACHE}"
                        --image_preprocess_mode coda_center_crop --coda_crop_size 512)
            if [[ "${SMOKE:-0}" == 1 ]]; then
                # Deliberately include a REAL GT-empty image, not only the first N.
                empty_index="$(COCO_MASK_CACHE="${COCO_MASK_CACHE}" "${PYTHON}" - <<'PY'
import os, numpy as np
root=os.environ['COCO_MASK_CACHE']
m=np.load(root+'/masks.npy',mmap_mode='r');o=np.load(root+'/overlap_masks.npy',mmap_mode='r')
for index in range(1,len(m)):
    if not np.any((m[index]>0) & (o[index]==0)):
        print(index);break
else:
    raise RuntimeError('No GT-empty image found for smoke')
PY
)"
                extra_args+=(--sample_indices 0 "${empty_index}")
            fi
            ;;
        voc)
            default_batch=1
            stage_args=(--gt_source dataset_mask)
            if [[ "${SMOKE:-0}" == 1 ]]; then extra_args+=(--max_samples 2); fi
            ;;
        *) echo "Unknown stage: ${stage} (coco|voc)" >&2; exit 2 ;;
    esac
    if [[ -n "${MAX_SAMPLES:-}" ]]; then
        [[ "${SMOKE:-0}" != 1 ]] || { echo "SMOKE cannot be combined with MAX_SAMPLES" >&2; exit 2; }
        extra_args+=(--max_samples "${MAX_SAMPLES}")
    fi
    if [[ "${COMPUTE_RFID,,}" =~ ^(true|1)$ ]]; then extra_args+=(--compute_rfid); fi
    if [[ "${COMPUTE_KID,,}" =~ ^(true|1)$ ]]; then extra_args+=(--compute_kid); fi
    if (( RECON_METRIC_RESOLUTION > 0 )); then
        extra_args+=(--recon_metric_resolution "${RECON_METRIC_RESOLUTION}")
    fi
    if (( ALSO_RECON_METRIC_RESOLUTION > 0 )); then
        extra_args+=(--also_recon_metric_resolution "${ALSO_RECON_METRIC_RESOLUTION}")
    fi
    "${PYTHON}" -m pgot.eval.run_eval \
        --model_path "${MODEL_PATH}" --dataset "${stage}" --data_root "${DATA_ROOT}/${stage}" \
        --output_dir "${OUTPUT_ROOT}/${stage}_ar" \
        --batch_size "${BATCH_SIZE:-${default_batch}}" --num_workers "${NUM_WORKERS:-2}" \
        --grid_size 32 --eval_size 512 --max_caption_tokens 1024 --n_ovt_per_object 1 --max_objects 50 \
        --caption_mode autoregressive --ar_max_new_tokens "${AR_MAX_NEW_TOKENS:-512}" \
        --readout ovt_owner --eval_merge mean --register_eval_route unrestricted --dtype "${DTYPE:-fp32}" \
        "${stage_args[@]}" "${extra_args[@]}" 2>&1 | tee "${OUTPUT_ROOT}/${stage}_ar.log"
done
