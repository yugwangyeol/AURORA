# 7B Object8/Register32 without Reader supervision

This run keeps the successful 7B FP32 fast architecture and training schedule.
The Reader module remains active and receives reconstruction gradients. Only
the explicit Reader-to-Writer consistency terms are disabled:

- `pgot_e8_reader_supervision_mode=none`
- `pgot_e8_reader_object_weight=0.0`
- `pgot_e8_reader_background_weight=0.0`

Writer GT grounding remains 1.0; LM and reconstruction losses remain 1.0 and
1.5. Object8/Register32, four background registers, MLLM RAE queries, SigLIP
values, no owner prior, no direct-query adapter, and no contrastive loss remain
unchanged.

Training uses all four GPUs in FP32 with per-device batch 4, accumulation 2,
and global batch 32. After checkpoint-5000 is audited, COCO AR, VOC AR, and
compositional generation run concurrently on GPUs 0, 1, and 2. GPU 3 remains
free during evaluation.
