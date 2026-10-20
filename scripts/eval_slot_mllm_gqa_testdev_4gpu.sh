#!/usr/bin/env bash
set -euo pipefail

cd /home/jovyan/PGOT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TOKENIZERS_PARALLELISM=false
export SLOT_MLLM_ENCODER_ONLY=1

PYTHON=/home/jovyan/.venvs/slot_mllm_eval/bin/python
QUESTIONS=/home/jovyan/data/gqa/prepared_balanced/manifests/questions_testdev.jsonl
IMAGE_ROOT=/home/jovyan/data/gqa/prepared_balanced/images
VISUAL_TOKENIZER=/home/jovyan/data/slot_mllm/Slot_Q-Former
MODEL=/home/jovyan/data/slot_mllm/Slot-MLLM-7B-instruct
SLOT_CACHE=/home/jovyan/data/gqa/slot_mllm_public_v1_testdev_slots.pt
RESULT_DIR="${RESULT_DIR:-/home/jovyan/PGOT/outputs/eval_gqa_slot_mllm_7b_public_v1}"
NUM_BEAMS="${NUM_BEAMS:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"

if [[ ! -s "$SLOT_CACHE" ]]; then
  "$PYTHON" -u pgot/eval/gqa_slot_mllm.py cache \
    --questions "$QUESTIONS" \
    --image_root "$IMAGE_ROOT" \
    --visual_tokenizer "$VISUAL_TOKENIZER" \
    --slot_mllm_repo /home/jovyan/external/Slot-MLLM \
    --output "$SLOT_CACHE" \
    --batch_size 8 \
    --num_workers 4
fi

"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  pgot/eval/gqa_slot_mllm.py eval \
  --questions "$QUESTIONS" \
  --slot_cache "$SLOT_CACHE" \
  --model "$MODEL" \
  --result_dir "$RESULT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens 12 \
  --num_beams "$NUM_BEAMS"

test -s "$RESULT_DIR/summary.json"
echo "Slot-MLLM GQA evaluation complete: $RESULT_DIR/summary.json"
