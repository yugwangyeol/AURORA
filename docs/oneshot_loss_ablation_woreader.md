# Original 1.5B: w/o Reader loss

This ablation keeps the Reader module and its reconstruction gradients. Only the
object/background Reader–Writer consistency supervision is removed:
`E8_READER_SUPERVISION_MODE=none`, `E8_READER_OBJECT_WEIGHT=0.0`,
`E8_READER_BACKGROUND_WEIGHT=0.0`.

Writer GT grounding remains at weight 1.0. LM loss (1.0), reconstruction loss
(1.5), Object8 / Register32 per register (4 registers), SigLIP memory values,
MLLM RAE query input and the no-owner-prior path remain the original configuration.
No direct-query adapter, contrastive loss, or DINO values are enabled.
The training initializer is the same E11 checkpoint used by the baseline;
this is not fine-tuning the baseline's completed checkpoint-5000.

```bash
cd /home/jovyan/PGOT
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_pgot_oneshot_memory8_register32_noprior_woreader.sh
```

Order: training on GPUs 2,3 (5000 steps), then autoregressive evaluation on GPU 3.
Teacher-forced evaluation is excluded. AR evaluation includes the original
segmentation/class metrics and rFID/KID. Checkpoints and evaluation results are
stored in separate `*_woreader` directories. Evaluation audits the trained
configuration before loading models. Any stage failure stops subsequent stages.
Existing training outputs are refused to prevent accidental overwriting.

To evaluate a completed training run without retraining:

```bash
CUDA_VISIBLE_DEVICES=2,3 bash scripts/eval_pgot_oneshot_memory8_register32_noprior_woreader.sh
```

`tests/test_woreader_ablation.py` checks the real launcher delegation and resolved
arguments using a fake model executable: this is not a real CUDA training smoke.
