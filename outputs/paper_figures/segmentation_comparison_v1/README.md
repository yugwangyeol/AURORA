# Combined segmentation comparison figure

One 3444x2240 image, with four rows and six model/input columns.

Columns: Image / GT / AdaSlot / MetaSlot / CODA / Ours.
Header appears once. Large bold COCO/VOC labels sit to the left of each row,
with spanning Over-segmentation / Under-segmentation labels further left.
A thin divider separates the two error types.

Rows, using the exact user selections:

1. COCO over 01: 552371.
2. VOC over 05: 2007_005149.
3. COCO under 03: 559956.
4. VOC under 07: 2007_007795.

The 24 panels are copied pixel-for-pixel from the existing v2 panel PNGs;
no source image, GT, prediction mask, or overlay color has been changed.
Background remains muted blue-gray (128,142,160), opacity 0.65 for all models.
Source paths, hashes and panel boxes are recorded in `manifest.json`.
VOC under 07 remains a relaxed qualitative case, with some errors in Ours too;
this figure does not claim perfect segmentation in that row.

- `segmentation_comparison.png`: full-resolution lossless combined figure.
- `segmentation_comparison.pdf`: corresponding raster PDF at 300 DPI.
- `preview.png`: half-size preview.

Reproduce, CPU only:

```bash
cd /home/jovyan/PGOT
PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python \
  -m pgot.eval.compose_segmentation_paper_figure
```
