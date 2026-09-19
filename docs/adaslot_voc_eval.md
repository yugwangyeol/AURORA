# AdaSlot COCO → VOC zero-shot evaluation

The adapter uses the unmodified [official AdaSlot implementation](https://github.com/amazon-science/AdaSlot),
revision `6a60387f4ee985e55b254274f41974e1aa5130e8`, and the authors' `COCO.ckpt`.
No VOC training, GT caption, GT category, or GT object count enters inference.
The COCO model has 33 candidate slots; its official Gumbel selector decides
which slots to keep. The frozen DINO backbone is included in the checkpoint.
All model tensors, including DINO and the selector, load with `strict=True`.

## Setup

```bash
cd /home/jovyan/PGOT
bash scripts/setup_adaslot_eval.sh
```

This creates `/home/jovyan/.venvs/adaslot_eval` with system packages inherited
from the working `scale_rae` environment. Only timm, torchtyping, typeguard,
and pluggy are installed into the new venv; base torch/CUDA is not replaced.
User-site packages are disabled. This avoids the original Python 3.8,
torch 1.12, and Lightning training dependencies on modern B200 GPUs.
The checkpoint download is verified by SHA-256.

## Smoke and full evaluation

```bash
GPU=0 NUM_WORKERS=0 bash scripts/smoke_baselines_voc_zeroshot.sh adaslot
GPU=0 bash scripts/eval_baselines_voc_zeroshot.sh adaslot
```

Smoke evaluates the first 8 VOC images in batches of 4. It verifies CUDA use,
strict checkpoint loading, finite segmentation metrics, kept-slot/mask
consistency, and object-count group coverage. Full evaluation uses all 1,449
VOC validation images; default batch size is 4, seed is 0, and workers is 2.
Change these with `BATCH_SIZE`, `SEED`, `NUM_WORKERS`, `MAX_SAMPLES`, and `GPU`.

The selector and slot initialization are stochastic, as in the official code.
For reproducible replay keep the seed, batch size, sample order, device, and
dependency versions unchanged. Do not replace Gumbel sampling with argmax or
fix the slot count to the GT count.

## Protocol and outputs

Model images use the official tensor bicubic resize/min-side 224, clamp,
center crop, and ImageNet normalization. Tensor resize has `antialias=False`
to preserve torchvision 0.13 behavior. Readout uses the official decoder's
kept-slot-masked `masks_as_image` at 224, bilinear upsampling to 512, then argmax.
GT and metrics are shared with the existing PGOT/CODA/MetaSlot/SPOT VOC
comparison: 512 center-cropped instance/semantic masks, 255 ignore handling,
and per-image NaN-skipped PGOT metrics.

Full outputs:

```text
outputs/baselines_voc_zeroshot/adaslot/summary.json
outputs/baselines_voc_zeroshot/adaslot/per_image.jsonl
outputs/baselines_voc_zeroshot/adaslot.log
```

`summary.json` includes fARI, instance mBO/mIoU, semantic sMBO/sMIOU,
`by_object_count` (1–3, 4–6, 7–10, 11+), `by_exact_object_count`, RNG settings,
kept-slot statistics, checkpoint load report, repository revision, and timings.
Groups use GT instances **after crop and ignore filtering**, excluding
background. Semantic metrics use those same instance-count groups.

After full AdaSlot evaluation, aggregate all four saved baselines without GPU:

```bash
/home/jovyan/.conda/envs/scale_rae/bin/python -m pgot.eval.voc_object_counts \
  --output_root outputs/baselines_voc_zeroshot
```

This writes `object_count_comparison.json` and `object_count_comparison.csv`.
Metrics in JSON/CSV are on the 0–1 scale. Existing per-model files are not changed.
The aggregator rejects comparisons with different image IDs/GT counts.
For a subset, add `--models coda metaslot adaslot`.
VOC's 11+ group contains only 6 images, so treat it as a small-sample result.
