#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${SCALE_RAE_PYTHON:-/home/jovyan/.conda/envs/scale_rae/bin/python}"
VENV="${SLOTDIFFUSION_VENV:-/home/jovyan/.venvs/slotdiffusion_eval}"
REPO=/home/jovyan/baselines/SlotDiffusion
NERV=/home/jovyan/baselines/nerv
export PYTHONNOUSERSITE=1
if [[ ! -d "${REPO}/.git" ]]; then
    git clone https://github.com/Wuziyi616/SlotDiffusion.git "${REPO}"
    git -C "${REPO}" checkout --detach eacfc9a83a3e1b20b31aa2213597a86fb067de39
fi
if [[ ! -d "${NERV}/.git" ]]; then
    git clone --branch v0.4.0 --depth 1 https://github.com/Wuziyi616/nerv.git "${NERV}"
fi
[[ "$(git -C "${REPO}" rev-parse HEAD)" == eacfc9a83a3e1b20b31aa2213597a86fb067de39 ]]
[[ "$(git -C "${NERV}" rev-parse HEAD)" == f204db59ecc40eaf7497430cf992068d5fccc864 ]]
if [[ ! -x "${VENV}/bin/python" ]]; then
    "${BASE_PYTHON}" -m venv --system-site-packages "${VENV}"
fi
# Training/video dependencies are deliberately not installed. This venv uses
# scale_rae's existing B200-compatible torch; don't let pip replace CUDA/torch.
"${VENV}/bin/python" -m pip install --no-deps -e "${NERV}" -e "${REPO}" \
    moviepy==1.0.3 lpips==0.1.4 decorator==4.4.2 proglog==0.1.12 \
    imageio==2.37.4 imageio_ffmpeg==0.6.0 scikit-image==0.24.0 lazy-loader==0.5 tifffile==2025.5.10
"${VENV}/bin/python" "${PROJECT_ROOT}/scripts/download_slotdiffusion_coco.py"
OMP_NUM_THREADS=3 "${VENV}/bin/python" -c 'import torch, slotdiffusion.img_based.models; print("SlotDiffusion imports OK", torch.__version__)'
