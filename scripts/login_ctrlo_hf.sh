#!/usr/bin/env bash
# Interactive only. Never put access tokens in command-line arguments/history.
set -euo pipefail
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/home/jovyan/.conda/envs/scale_rae/lib:${LD_LIBRARY_PATH:-}"
exec /home/jovyan/.venvs/ctrlo_eval/bin/python -c 'from huggingface_hub import login; login(add_to_git_credential=False)'
