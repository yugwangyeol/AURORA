# One-shot object visual memory (4 per object, 64 background)

This experiment preserves one caption OVT per object and RAE self-only attention
inside the MLLM. It reuses E11's Writer projections and memory-ID embeddings, but
performs one write from the final semantic owners. No old memory, update gate,
GRU, centroid path, or direct raw-patch Reader is used.

The final predicted ownership A is detached on the reconstruction path, as in
the existing one-shot runs. Each write query is `W_s LN(s_k) + e_j`; its patch
logits receive `log(A_kp + epsilon)` and are normalized over patches. Values
come from projected frozen raw SigLIP features. The resulting `Z[B,S,16,D]`
is padded: only the first 4 memories are valid for each object; all 16 are
valid for each of the 4 background registers. Valid capacity is `4K + 64`.
The soft prior discourages cross-owner reads; it is not a hard support mask.

The Reader computes soft semantic owner attention G, then within-owner memory
attention beta. Its output is `sum_k G_ik sum_j beta_ikj V(Z_kj)`. The Reader
interface accepts semantic owners, stored Z, validity masks, and RAE queries;
it accepts no source image or raw patches. The output conditions the existing
DiT. `memory_content` uses projected memory content as the inner key;
`memory_id` uses E11's learned memory-ID key. Both share the same one-shot
Writer and outer semantic routing. This is not the old E11 flat joint softmax.

Both variants default to E11 capacity_dit16/checkpoint-10000, FP32, 2 GPUs,
microbatch 2/GPU (6 accumulation steps), global batch 24, 5,000 additional optimizer steps, and the
last 16 DiT blocks trainable. Existing LM/reconstruction/owner/Reader losses
are retained. A new content-key projection initializes from the source
Reader's key weights; source Writer IDs are reused, not reset. Checkpoint
step numbers refer to the new stage, not cumulative pretraining steps.
Checkpoints default to model weights only (`SAVE_ONLY_MODEL=True`), retaining
the latest periodic checkpoint. Set `SAVE_ONLY_MODEL=False` before training
if optimizer/scheduler state is needed for exact interrupted-run resume.

## Run training and both evaluations

```bash
cd /home/jovyan/PGOT
CUDA_VISIBLE_DEVICES=0,1 MEMORY_KEY_MODE=content \
  bash scripts/run_pgot_oneshot_memory4.sh
```

After two-GPU training, TF evaluation runs on GPU 0 and AR evaluation on GPU 1.
Each evaluates the complete same validation set, so no shard-FID averaging is
needed. The combined runner refuses to overwrite an existing training output.

To run the ID control from the same E11 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0,1 MEMORY_KEY_MODE=id \
  bash scripts/run_pgot_oneshot_memory4.sh
```

Run these experiments sequentially; each training job uses both GPUs.

Default outputs:

- `checkpoints/pgot_oneshot_memory4_{content,id}/checkpoint-5000/`
- `outputs/eval_pgot_oneshot_memory4_{content,id}/train.log`
- `outputs/eval_pgot_oneshot_memory4_{content,id}/{tf,ar}/summary.json`
- `outputs/eval_pgot_oneshot_memory4_{content,id}/{tf,ar}.log`

Independent commands, if training/evaluation should be launched separately:

```bash
CUDA_VISIBLE_DEVICES=0,1 MEMORY_KEY_MODE=content \
  bash scripts/train_pgot_oneshot_memory4.sh

CUDA_VISIBLE_DEVICES=0,1 MEMORY_KEY_MODE=content \
  bash scripts/eval_pgot_oneshot_memory4_parallel.sh
```

`MODEL_PATH`, `OUTPUT_DIR`, `EVAL_ROOT`, `MAX_STEPS`, `WANDB_PROJECT`,
`WANDB_NAME`, `BATCH_SIZE`, `AR_BATCH_SIZE`, and `AR_MAX_NEW_TOKENS` are
overridable environment settings. TF/AR default to batch 4, ten diffusion
steps, guidance 1.0, and AR length 512. Full rFID comparisons require keeping
the validation set and inference settings fixed.
Each evaluation process defaults to four CPU threads to avoid CPU thread
oversubscription while the two checkpoints load concurrently.

## Logging

The new modes automatically use the `one_shot_memory` metric profile. Both
terminal and W&B history retain active LM/reconstruction/owner/Reader losses,
ownership/routing quality, memory norm, extraction entropy, within-owner
Reader entropy, object/register memory pair cosine, and valid token count.
Inactive causal/gate/centroid/contrastive diagnostics and constant feature
switches are omitted. An active metric reaching zero is still recorded.
Static architecture settings remain in the checkpoint/W&B configuration.
Training evaluation includes a W&B image table with source, decoded target,
reconstruction, and ownership overlay.

## Smoke

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/smoke_pgot_oneshot_memory4.sh
```

The default smoke exercises both modes with 2 optimizer steps at the configured
training microbatch, in-training evaluation, model checkpoint save, W&B
online metrics and images, and concurrent TF/AR reload evaluations. It checks
that Writer IDs changed during training. Tiny 2-image FIDs are only execution
checks and must not be interpreted as model quality.

On success, the script deletes only its own W&B smoke runs/artifacts and its
unique `.smoke_oneshot_memory4.*` directory. Failures are preserved for diagnosis.
`WANDB_MODE=offline` can verify local W&B capture without server verification.

Unit checks are in `tests/test_pgot_one_shot_memory.py`: patch normalization,
heterogeneous capacity/padding, detached ownership with live Writer gradients,
semantic-dependent extraction, stored-memory save/reload, content-addressed
permutation invariance, and metric filtering.

Validated on 2026-09-08: all 15 selected new/legacy unit tests passed. Both
content and ID modes completed two FP32 DDP optimizer steps with microbatch
2/GPU and global batch 24, in-training evaluation, checkpoint save, and
simultaneous two-image TF/AR reload evaluations. Writer IDs changed by a
maximum of approximately `5.126e-5` in each mode. W&B server history and
reconstruction/ownership image tables were verified. The final metric profile
also removed epoch/FLOPs and duplicate evaluation rows. The initial
microbatch-6 run failed during GPU memory allocation; the defaults above use
the successfully exercised microbatch 2. These checks establish execution
correctness, not reconstruction quality or a content-vs-ID quality ranking.
