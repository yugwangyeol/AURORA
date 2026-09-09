# One-shot memory4 with ownership gradient and direct feature reconstruction

This continuation starts from
`checkpoints/pgot_oneshot_memory4_content/checkpoint-5000`. It preserves GT
ownership supervision, four content-addressed visual memories per object, 16
memories per background register, the one-layer Reader, and the last 16
trainable DiT blocks.

Two training paths are added. First, Writer routing keeps the same forward
ownership probabilities but opens the reconstruction gradient to ownership.
Its gradient scale ramps from zero to one over the first 500 optimizer steps.
Second, the existing direct latent head predicts the frozen decoder-native
SigLIP target from the exact Reader condition consumed by DiT. Its weighted
MSE-plus-cosine loss ramps from zero to 0.5 over the same 500 steps. The head is
an auxiliary training module and is not used during diffusion generation.

The default learning rates are `2e-5` for the Writer, Reader, diffusion head,
unfrozen DiT blocks, projector, registers, and RAE queries; `5e-6` for Qwen
LoRA; and `1e-4` for the new latent head. Training uses two GPUs, FP32, global
batch 24, and 5,000 additional optimizer steps.

Run training followed by simultaneous full-set TF and AR evaluation:

```bash
cd /home/jovyan/PGOT
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_pgot_oneshot_ownergrad_feature.sh
```

Run training and evaluation separately:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_pgot_oneshot_ownergrad_feature.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_pgot_oneshot_ownergrad_feature_parallel.sh
```

The outputs are:

- `checkpoints/pgot_oneshot_ownergrad_feature/checkpoint-5000/`
- `outputs/eval_pgot_oneshot_ownergrad_feature/train.log`
- `outputs/eval_pgot_oneshot_ownergrad_feature/tf/summary.json`
- `outputs/eval_pgot_oneshot_ownergrad_feature/ar/summary.json`

The parallel evaluator assigns the complete TF validation set to GPU 0 and the
complete AR validation set to GPU 1. It does not split or average FID shards.

Smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=online \
  bash scripts/smoke_pgot_oneshot_ownergrad_feature.sh
```

Validated on 2026-09-09: nine one-shot unit tests passed. The online two-GPU
smoke completed two FP32 optimizer steps, in-training evaluation, checkpoint
save, W&B scalar and image-table verification, and simultaneous TF/AR reload
evaluation. The step-wise smoke ramps reached ownership gradient scales 0.5
and 1.0 and auxiliary loss weights 0.25 and 0.5. Inactive causal, contrastive,
gate, GRU, centroid, routing, and L1 metrics were absent from terminal and W&B
history. The temporary W&B run and all local smoke artifacts were deleted.
