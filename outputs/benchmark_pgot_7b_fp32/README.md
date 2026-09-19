# 7B FP32 throughput diagnostic — 2026-09-16

Original 4-GPU O8/R32 training stopped at completed update81, before its first
checkpoint. Its log remains in the original evaluation directory. No production
training was restarted and no original outputs/checkpoints were deleted.

Global batch32, FP32 storage/compute with TF32 enabled, native Qwen7B/DiT9.8B,
MLLM RAE256, SigLIP values, no adapter/contrastive/prior changes.

| Per-device batch / accumulation | Prefetch | Seconds/update | 5000-update compute estimate |
| --- | --- | ---: | ---: |
| 1 / 8 | none | ~66.6 | ~92.5h |
| 2 / 4 | none | 35.72 | 49.61h |
| 4 / 2 | none | 20.49 | 28.46h |
| 4 / 2 | backward_post | 19.30 | 26.80h |

Pilot estimates exclude loading, profiler collection/export, saving and
evaluation. Median of two warmed optimizer updates, using the slowest rank
per update. B4 timing includes parameter-update audits. Different microbatch
grouping preserves global batch but not necessarily identical variable-length
loss weighting, RNG/dropout or floating-point accumulation.

B2 completed five updates. B4 completed eight, including the final long-caption
stress update (804–964 tokens, up to38 objects). B4 peak allocated71.50GiB,
reserved86.18GiB on89GiB MIG devices. All eight audited training groups have
FP32 finite gradients and actual nonzero updates. This is runtime/update
verification, not convergence or proof that every configured maximum input fits.

Raw rank3 B2 CUDA timeline35.60s includes communication kernels active for
30.25s (interval union, not a claim that all of this time can be eliminated).
GPU peer access is false for all pairs; NCCL uses SHM/direct/direct, host-memory
staging rather than model CPU offload. Forced Simple collective protocol
microbenchmarks showed no improvement, and were not applied. AdamW GPU kernels
in this trace total only~0.06s, so fused AdamW is not the main opportunity.

## Correctness finding: original-Parameter token-row hooks

`freeze_for_pgot` registers embedding and LM-head gradient row masks on original
Parameters before FSDP wrapping. FSDP uses temporary Tensor views during
forward/backward. A miniature test with the same registration mechanism shows
zero hook invocations on all four ranks. With SGD and **no weight decay**,
two original embedding rows and six original head rows changed; a plain model
with the same masks changed none of these rows. See `token_mask_probe.json`.

This reproduces the mechanism failure; no full-model base-row audit was run.
Because PGOT uses this same mechanism, old vocabulary gradient protection is
not safe to assume, even though the eight-group numerical audit passes.
No production row-mask fix was applied in this diagnostic turn.

Suggested follow-up: attach the gradient mask to the *live* unsharded weight
Tensor via embedding/head module forward-pre-hooks, rather than only the saved
original Parameter. Ensure it runs before gradient clipping and FSDP reduction.
Verify against masked plain-model SGD and AdamW references, preserving the
existing weight-decay semantics; then audit full PGOT base/new-token rows and
repeat B4/accum2 + backward_post smoke before issuing a restart command.

Benchmark-only helpers are `scripts/benchmark_pgot_7b_entry.py` and
`scripts/benchmark_pgot_nccl.py`. Normal training entrypoint and optimizer
defaults remain unchanged; the shared launcher gained optional TRAIN_ENTRYPOINT
and OPTIM overrides solely to run these comparisons.

## Follow-up: row-mask fix and production-condition smoke

The originally reproduced hook bug above has now been fixed with live-weight
forward-pre-hooks. See `maskfix_validation.json` for the follow-up QA metadata;
the earlier failure report and pre-fix throughput measurements remain historical.
32 CPU tests pass, including real shell orchestration (train, TF, AR) with a
mocked heavy launcher and protection against overwriting an existing train log.
Four-GPU accumulated SGD and AdamW updates, and clipped gradient norms, match
independently masked plain-model references. Old rows stay bitwise unchanged
under SGD without decay and change only by existing decay under AdamW.

Full native 7B FP32 + TF32, batch4/accum2, BACKWARD_POST smoke completed three
updates, validation4, periodic and final full-weight saves, and TF2/AR2 reloads
with missing=0/unexpected=0. All eight training groups show finite FP32 gradients
and actual updates. Embedding/head live-mask hooks fire six times per rank;
sampled old gradients are zero and new gradients finite/nonzero. Temporary
smoke checkpoints, logs and evaluation outputs were deleted after success.
This is runtime/numerical validation, not convergence: both three-update AR
captions have invalid object format and exhaust the token budget.

Third warmed update, slowest rank:19.082s, or26.50h of compute for5000 updates.
Plan27–30h for training with startup/save/validation. Two-image TF/AR loops are
about3.20/15.09s per image at batch1 and10 diffusion steps. Linear5000-image
estimates are4.44h TF and20.96h AR, not representative full-dataset guarantees;
trained captions that stop earlier can shorten AR substantially.

Restart command (not launched automatically):

```bash
cd /home/jovyan/PGOT
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_pgot_oneshot_memory8_register32_noprior_7b_fp32_fast.sh
```

Fresh `_fast` checkpoint/evaluation roots preserve the original update81 log.
The fast wrapper forces the real train.py entrypoint and enables final save,
never the save-skipping benchmark entrypoint. Global batch32 and baseline
O8/R32, MLLM RAE256, SigLIP values, no adapter/contrastive/prior are preserved.
Different microbatch grouping still need not be bitwise identical to batch1/accum8
because variable-length loss normalization, dropout/RNG and accumulation differ.
