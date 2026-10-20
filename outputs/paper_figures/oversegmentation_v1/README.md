# Over-segmentation figure, first draft

Selected by the user:

- COCO over 01: image 552371 (two horses).
- VOC over 05: image 2007_005149 (two dogs).

Layout: Image / GT / AdaSlot / MetaSlot / CODA / Ours.
First row: COCO. Second row: PASCAL VOC.

`preview.png` is a smaller preview. `oversegmentation_figure.png` is the
full-size 3102x1184 PNG; `oversegmentation_figure.pdf` is the corresponding
raster PDF at 300 DPI. Separate COCO and VOC figures are also exported.

All models use the same 0.65 mask color opacity, versus 0.52 in the original
mining preview. Background is visibly colored, rather than dark gray.
Foreground colors are aligned one-to-one with GT by IoU for display only;
extra predicted regions retain distinct colors. No regions were merged,
removed, smoothed, or changed. GT-ignore pixels are consistently displayed
in neutral gray and excluded from color assignment.

Each operation uses already-cached masks, without GPU inference. `manifest.json`
records identities, per-label display colors, source hashes, overlay strength,
and the unchanged original metrics. Panel PNGs and colored maps are saved in
`coco_over/` and `voc_over/`.

The user's other selections are retained for a future under-segmentation figure:
COCO under 03 (559956), VOC under 07 (2007_007795). They were not rendered here.

Reproduce:

```bash
cd /home/jovyan/PGOT
PYTHONNOUSERSITE=1 /home/jovyan/.conda/envs/scale_rae/bin/python \
  -m pgot.eval.render_segmentation_paper_figure
```

For a different opacity, choose a new output directory, e.g.
`--alpha 0.75 --output_dir /home/jovyan/PGOT/outputs/paper_figures/oversegmentation_v2`.
