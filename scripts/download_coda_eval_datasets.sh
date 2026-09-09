#!/usr/bin/env bash
# Download the CODA evaluation datasets into PGOT's shared data directory.
#
# Usage:
#   bash scripts/download_coda_eval_datasets.sh voc
#   bash scripts/download_coda_eval_datasets.sh movi-c movi-e
#   bash scripts/download_coda_eval_datasets.sh all
#
# Optional overrides:
#   DATA_ROOT=/path/to/data
#   BASE_PYTHON=/path/to/python
#   DATA_ENV=/path/to/download-only-venv

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/jovyan/data}"
BASE_PYTHON="${BASE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
DATA_ENV="${DATA_ENV:-/home/jovyan/.venvs/pgot_coda_data_clean}"
DOWNLOAD_ROOT="${DATA_ROOT}/.downloads/coda"

VOC_OFFICIAL_URL="https://thor.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

usage() {
    cat <<EOF
Usage: bash ${PROJECT_ROOT}/scripts/download_coda_eval_datasets.sh DATASET [DATASET ...]

DATASET may be: voc, movi-c, movi-e, all

Datasets are written to:
  ${DATA_ROOT}/voc
  ${DATA_ROOT}/movi-c
  ${DATA_ROOT}/movi-e
EOF
}

if (( $# == 0 )); then
    usage >&2
    exit 2
fi

datasets=()
for dataset in "$@"; do
    case "${dataset}" in
        all)
            datasets=(voc movi-c movi-e)
            break
            ;;
        voc|movi-c|movi-e)
            datasets+=("${dataset}")
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown dataset: ${dataset}" >&2
            usage >&2
            exit 2
            ;;
    esac
done

test -x "${BASE_PYTHON}" || {
    echo "Missing BASE_PYTHON: ${BASE_PYTHON}" >&2
    exit 1
}
command -v wget >/dev/null || {
    echo "wget is required but was not found." >&2
    exit 1
}
command -v tar >/dev/null || {
    echo "tar is required but was not found." >&2
    exit 1
}

mkdir -p "${DATA_ROOT}" "${DOWNLOAD_ROOT}"

need_movi=0
for dataset in "${datasets[@]}"; do
    if [[ "${dataset}" == movi-* ]]; then
        need_movi=1
    fi
done

ensure_data_env() {
    if [[ ! -x "${DATA_ENV}/bin/python" ]]; then
        echo "Creating download-only Python environment: ${DATA_ENV}"
        "${BASE_PYTHON}" -m venv "${DATA_ENV}"
    fi

    local data_python="${DATA_ENV}/bin/python"
    if (( need_movi )); then
        if ! "${data_python}" -c 'import tensorflow, tensorflow_datasets' >/dev/null 2>&1; then
            "${data_python}" -m pip install \
                'numpy>=1.26,<2' \
                'tensorflow-cpu==2.21.0' \
                'tensorflow-datasets==4.9.10' \
                pillow tqdm
        fi
    fi
}

copy_tree_contents() {
    local source_dir="$1"
    local target_dir="$2"
    test -d "${source_dir}" || {
        echo "Expected directory not found after extraction: ${source_dir}" >&2
        return 1
    }
    mkdir -p "${target_dir}"
    cp -a "${source_dir}/." "${target_dir}/"
}

download_voc() {
    local target="${DATA_ROOT}/voc"
    local marker="${target}/.pgot_download_complete"
    if [[ -f "${marker}" ]]; then
        echo "VOC already complete: ${target}"
        return
    fi

    local official_archive="${DOWNLOAD_ROOT}/VOCtrainval_11-May-2012.tar"
    local stage

    echo "Downloading official VOC 2012 images and segmentation masks..."
    wget --continue --output-document="${official_archive}" "${VOC_OFFICIAL_URL}"

    stage="$(mktemp -d "${DATA_ROOT}/.voc-stage.XXXXXX")"
    echo "Extracting VOC archives in ${stage}..."
    tar -xf "${official_archive}" -C "${stage}"

    local official_root="${stage}/VOCdevkit/VOC2012"
    copy_tree_contents "${official_root}/JPEGImages" "${target}/images"
    copy_tree_contents "${official_root}/SegmentationClass" "${target}/SegmentationClass"
    copy_tree_contents "${official_root}/SegmentationObject" "${target}/SegmentationObject"
    mkdir -p "${target}/sets"
    cp "${official_root}/ImageSets/Segmentation/val.txt" "${target}/sets/val.txt"

    test -s "${target}/sets/val.txt"
    test -n "$(find "${target}/images" -maxdepth 1 -type f -name '*.jpg' -print -quit)"
    test -n "$(find "${target}/SegmentationObject" -maxdepth 1 -type f -name '*.png' -print -quit)"
    touch "${marker}"
    rm -rf -- "${stage}"
    echo "VOC ready: ${target}"
}

download_movi() {
    local dataset="$1"
    local level="${dataset#movi-}"
    local target="${DATA_ROOT}/${dataset}"
    local marker="${target}/.pgot_download_complete"
    if [[ -f "${marker}" ]]; then
        echo "${dataset} already complete: ${target}"
        return
    fi

    mkdir -p "${target}"
    echo "Downloading and converting ${dataset} validation from the official Kubric TFDS bucket..."
    CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
    "${DATA_ENV}/bin/python" "${PROJECT_ROOT}/preprocess/download_movi_eval.py" \
        --out_path="${target}" \
        --level="${level}" \
        --image_size=256 \
        --split=validation

    test -d "${target}/validation" || {
        echo "Missing ${dataset} split after download: ${target}/validation" >&2
        return 1
    }
    test -n "$(find "${target}/validation" -type f -name '*.jpg' -print -quit)"
    test -n "$(find "${target}/validation" -type f -name '*_mask.png' -print -quit)"
    touch "${marker}"
    echo "${dataset} ready: ${target}"
}

ensure_data_env

for dataset in "${datasets[@]}"; do
    case "${dataset}" in
        voc) download_voc ;;
        movi-c|movi-e) download_movi "${dataset}" ;;
    esac
done

echo "All requested CODA evaluation datasets are ready under ${DATA_ROOT}."
