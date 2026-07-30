#!/usr/bin/env bash
set -euo pipefail

PGOT_ROOT=/home/jovyan/PGOT
GEN_ROOT=/home/jovyan/data/pix2cap/generated_coco_val_5k
SCALE_PYTHON=/home/jovyan/.conda/envs/scale_rae/bin/python

while tmux has-session -t pix2cap_val5k_gpu0 2>/dev/null ||
      tmux has-session -t pix2cap_val5k_gpu1 2>/dev/null; do
  sleep 30
done

cd "${PGOT_ROOT}"

"${SCALE_PYTHON}" preprocess/merge_pix2cap_generated_shards.py \
  --shard_root "${GEN_ROOT}/shards" \
  --instances_json /home/jovyan/data/coco/annotations/instances_val2017.json \
  --panoptic_categories_json /home/jovyan/data/coco/annotations/panoptic_val2017.json \
  --mask_root "${GEN_ROOT}/masks" \
  --output "${GEN_ROOT}/pix2cap_coco_val_generated.json" \
  --summary_output "${GEN_ROOT}/generation_summary.json" \
  --expected_images 5000

"${SCALE_PYTHON}" preprocess/prepare_pix2cap_thing_pgot.py \
  --coco_root /home/jovyan/data/coco \
  --split val \
  --input "${GEN_ROOT}/pix2cap_coco_val_generated.json" \
  --panoptic_root "${GEN_ROOT}/masks" \
  --output "${PGOT_ROOT}/data/pgot_pix2cap_generated_val5k.jsonl" \
  --stats_output "${PGOT_ROOT}/data/pgot_pix2cap_generated_val5k.stats.json" \
  --crop_size 512 \
  --max_objects 50 \
  --max_caption_tokens 1024 \
  --n_ovt_per_object 1

echo "PIX2CAP_VAL5K_FINALIZE_COMPLETE"
