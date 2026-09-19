#!/usr/bin/env bash
# Inference-only dependencies: inherit the working modern CUDA torch without
# installing AdaSlot's obsolete Python 3.8 / Lightning training dependency set.
set -euo pipefail
BASE_PYTHON="${BASE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
VENV_DIR="${VENV_DIR:-/home/jovyan/.venvs/adaslot_eval}"
ADASLOT_ROOT="${ADASLOT_ROOT:-/home/jovyan/baselines/AdaSlot}"
ADASLOT_CKPT="${ADASLOT_CKPT:-/home/jovyan/baselines/ckpt/adaslot-coco.ckpt}"
REVISION=6a60387f4ee985e55b254274f41974e1aa5130e8
CHECKPOINT_SHA256=9ce54cdb7c8b0e73c46f2dd2bd9f490d76ead2cf5034929949dac0f5cf2bd7c8
export PYTHONNOUSERSITE=1
if [[ ! -e "${ADASLOT_ROOT}" ]]; then
    git clone https://github.com/amazon-science/AdaSlot.git "${ADASLOT_ROOT}"
    git -C "${ADASLOT_ROOT}" checkout --detach "${REVISION}"
fi
test "$(git -C "${ADASLOT_ROOT}" rev-parse HEAD)" = "${REVISION}" || {
    echo "AdaSlot revision mismatch; use a separate ADASLOT_ROOT for the pinned repository." >&2
    exit 1
}
if [[ ! -e "${VENV_DIR}" ]]; then
    "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi
"${VENV_DIR}/bin/python" -m pip install --no-deps \
    timm==0.6.7 torchtyping==0.1.5 typeguard==2.13.3 pluggy==1.0.0
mkdir -p "$(dirname "${ADASLOT_CKPT}")"
if [[ ! -f "${ADASLOT_CKPT}" ]]; then
    curl -L --fail --retry 3 \
        'https://drive.usercontent.google.com/download?id=1NaGR0n25Y3Sn6zQKESst3uvjuI0F443C&export=download&confirm=t' \
        -o "${ADASLOT_CKPT}.part"
    printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${ADASLOT_CKPT}.part" | sha256sum --check -
    mv "${ADASLOT_CKPT}.part" "${ADASLOT_CKPT}"
fi
printf '%s  %s\n' "${CHECKPOINT_SHA256}" "${ADASLOT_CKPT}" | sha256sum --check -
echo "AdaSlot ready. Smoke: GPU=0 bash scripts/smoke_baselines_voc_zeroshot.sh adaslot"
