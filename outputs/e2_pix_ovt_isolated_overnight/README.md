# E2-Pix-OVT-Isolated overnight run

Completed on 2026-07-30 UTC.

## 1. Checkpoint-10000 evaluation

- Checkpoint: `/home/jovyan/PGOT/checkpoints/pgot_e2_pix_ovt_isolated/checkpoint-10000`
- Input: released Pix2Cap thing-only validation manifest (2,088 images)
- Protocol: teacher forcing, COCO instance GT, CODA 512 center crop,
  512 segmentation metrics, 32 x 32 / 1,024 image patches
- FG-ARI: **0.0874028**
- mBO: **0.2388551**
- mIoU: **0.3314100**
- rFID: **37.7701530**
- PSNR: **10.4637565**
- SSIM: **0.2261915**

The full evaluation output is:
`/home/jovyan/PGOT/outputs/eval_pgot_e2_pix_ovt_isolated_10k_tf_coda512_rfid`

## 2. Pix2Cap generation over COCO val2017

- Pix2Cap checkpoint:
  `/home/jovyan/data/pix2cap/checkpoints/pix2cap_bs32_dw0.1_epoch25.pt`
- Source images: all 5,000 COCO val2017 images
- Successful model inferences: **5,000 / 5,000**
- Prediction errors: **0**
- Images with at least one predicted thing: **4,948**
- Images without a predicted thing: **52**
- Predicted thing segments: **29,689**
- Predicted stuff segments: **17,061**
- Runtime: 2 B200 GPUs, batch size 4 per GPU, about 34 minutes

Every generated PNG was checked against its source image resolution, JSON
segment IDs, and pixel areas before the merged file was written.

Raw generated outputs:

- Root: `/home/jovyan/data/pix2cap/generated_coco_val_5k`
- Merged JSON:
  `/home/jovyan/data/pix2cap/generated_coco_val_5k/pix2cap_coco_val_generated.json`
- Masks: `/home/jovyan/data/pix2cap/generated_coco_val_5k/masks`
- Generation statistics:
  `/home/jovyan/data/pix2cap/generated_coco_val_5k/generation_summary.json`

## 3. Strict clean PGOT validation manifest

The strict policy keeps an image only when it has at least one crop-visible
predicted thing and every such thing has a usable Pix2Cap caption. Stuff is
removed from the OVT list and remains available only through the register
background path.

- Final images: **4,720**
- Final thing OVTs: **24,322**
- Excluded for no crop-visible thing: **62**
  - 52 have no predicted thing at all.
  - 10 have predicted things, but none survives the CODA 512 center crop.
- Excluded for at least one invalid crop-visible caption: **218**
  - object-not-visible refusal: 214
  - cannot-describe refusal: 4
- Category fallback used: **no**
- Maximum objects in one retained image: **36**
- Token/object budget violations in retained data: **0**
- Duplicate image IDs: **0**
- Missing image/mask paths: **0**
- OVT-to-segment invariant failures: **0**

Final manifest:
`/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl`

Manifest statistics:
`/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.stats.json`

## Interpretation

Pix2Cap inference genuinely covered all 5,000 source images. The directly
usable, clean teacher-forced PGOT validation set is 4,720 images because the
strict policy does not invent OVTs when Pix2Cap predicts no thing and does not
replace refusal captions with category-only fallback. Therefore results on
this manifest should be reported as a 4,720-image evaluation, not as a
5,000-image evaluation.

If an exactly 5,000-row PGOT manifest is required later, the remaining 280
images need an explicit policy decision: category fallback for invalid
captions and a defined register-only/zero-OVT behavior for images with no
crop-visible predicted thing. No new Pix2Cap inference is needed for that
policy experiment; the complete raw predictions are already saved.
