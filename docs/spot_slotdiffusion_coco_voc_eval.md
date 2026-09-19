# SPOT / SlotDiffusion: COCO-trained checkpoint evaluation

Run three evaluations, all with **COCO-trained** checkpoints:

```bash
cd /home/jovyan/PGOT
bash scripts/setup_slotdiffusion_eval.sh
GPU=0 BATCH_SIZE=8 bash scripts/eval_spot_slotdiffusion_coco_voc.sh
PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python \
  -m pgot.eval.summarize_coco_checkpoint_baselines
```

Individual stages: `spot_coco`, `slotdiffusion_coco`, `slotdiffusion_voc`.
The existing SPOT VOC evaluation is not rerun or overwritten.
Output root defaults to `outputs/baselines_coco_checkpoint`; override with `OUTPUT_ROOT`.
Each stage writes `summary.json`, `per_image.jsonl`, and a log. The summary includes
overall instance/category metrics and GT object-count buckets/exact counts.

Smoke (separate output, does not overwrite full evaluations):

```bash
GPU=0 MAX_SAMPLES=8 BATCH_SIZE=8 \
  OUTPUT_ROOT=/home/jovyan/PGOT/outputs/_smoke_spot_slotdiffusion \
  bash scripts/eval_spot_slotdiffusion_coco_voc.sh
```

## Weights and implementation

- SPOT: existing `baselines/ckpt/spot_coco_checkpoint.pt.tar`, 7 slots,
  DINO ViT-B/16, transformer decoder mask attention, all evaluation permutations.
- SlotDiffusion: official revision `eacfc9a83a3e1b20b31aa2213597a86fb067de39`,
  nerv v0.4.0 revision `f204db59ecc40eaf7497430cf992068d5fccc864`.
  Config `sa_ldm_dino_coco_params-res224.py`, DINO ViT-S/8, 7 slots.
- Download only `sa_ldm_dino_coco_params-res224.pth` and
  `vqvae_coco_params-res224.pth` from the official Google Drive ZIP using HTTP
  Range requests. ZIP CRC and pinned SHA256 validate the downloaded files.
  **Do not use** `sa_ldm_dino_voc_params-res224.pth` for the VOC run.
- Build the full unmodified official model and strictly load its entire state.
  Its official segmentation forward returns final slot-attention masks without
  diffusion sampling. Keep the official intermediate bilinear mask resize
  (28x28 -> 224x224), then resize to the common PGOT 512 metric grid.
- Isolated venv `~/.venvs/slotdiffusion_eval` inherits scale_rae's existing torch.
  Install only extra inference import dependencies; no training or GPU changes.

## Evaluation protocol and fair comparison

- COCO: all 5,000 COCO2017 validation images in existing cache `image_ids.npy`
  order. Use the exact PGOT CODA-style 512 instance and thing-category caches,
  preserving cache labels, with **separate** instance/category overlap maps.
- VOC: same 1,449 `sets/val.txt` images and 512 center-crop GT / 255-ignore
  handling as the existing CODA/SPOT/MetaSlot/AdaSlot zero-shot results.
- SPOT input: its official COCO/VOC PIL bilinear Resize(224), CenterCrop(224),
  ImageNet normalization. SlotDiffusion input: official dataset-specific OpenCV
  ResizeMinShape(224), CenterCrop(224), mean/std 0.5 normalization.
- Readout: bilinear soft-mask resize to 512, then argmax. Use existing
  `pgot_metrics` fARI/mBO/mIoU per image, NaN-skipped means. `mIoU` follows
  PGOT's existing Hungarian metric implementation (includes GT background).
  These are unified PGOT-protocol evaluations, **not** a claim to reproduce
  every paper's original mask resolution, averaging, or benchmark row.
- Object counts: visible GT instance IDs after instance overlap exclusion,
  background excluded. No GT count/labels are supplied to either model.
  PGOT AR's existing COCO object-count buckets may instead use manifest counts;
  do not assume its bucket membership is identical to these cropped GT counts.
- PGOT's generated-caption COCO evaluation covers 4,720 of the 5,000 images.
  The summarizer writes `summary_pgot_subset.json` for both COCO models using
  exactly `data/pgot_pix2cap_generated_val5k.jsonl` image IDs; compare these rows
  with PGOT, not the full 5,000-image means. No second inference is needed.
- `comparison.csv` / `comparison.json` collect full and image-matched results.
  `comparison_by_object_count.csv` collects their object-count bucket rows.
  Metrics in JSON/CSV use 0..1 scale; console table uses percentages.

Official sources:
[SlotDiffusion benchmark/weights](https://github.com/Wuziyi616/SlotDiffusion/blob/main/docs/benchmark.md),
[segmentation script](https://github.com/Wuziyi616/SlotDiffusion/blob/main/slotdiffusion/img_based/test_seg.py),
[SPOT](https://github.com/gkakogeorgiou/spot).

## Verified run (2026-09-16)

GPU 0, batch 8, seed 0, FP32. All three full evaluations completed.
COCO: identical 5,000 image IDs and visible GT count assignments for both models.
VOC: identical 1,449 image IDs / GT counts to the existing AdaSlot VOC result.
All values below are percentages under the unified PGOT protocol:

| Run | N | FG-ARI | mBO^i | mBO^c | mIoU^i | mIoU^c |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPOT COCO | 5000 | 37.28 | 34.57 | 44.40 | 35.24 | 45.73 |
| SlotDiffusion COCO | 5000 | 37.12 | 31.03 | 35.51 | 31.23 | 36.91 |
| SlotDiffusion VOC | 1449 | 20.99 | 46.56 | 47.40 | 43.91 | 45.33 |

Smoke: 8 images per stage passed. The SlotDiffusion GPU integration test checks
exact preprocessing equality to independently computed OpenCV resize/crop,
exact official-forward mask equality, finite normalized masks, and frozen DINO.
The cache/subset unit tests (3) and object-count aggregation unit tests (3) pass.
Use the same batch size for smoke/full replay: small floating-point differences
with another batch size can flip nearly tied pixel argmax labels.
