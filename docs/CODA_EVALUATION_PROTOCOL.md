# PGOT COCO/CODA Evaluation Protocol

Protocol version: 1.4 (2026-08-04)

This document is the fixed reference for PGOT segmentation evaluation. Any
result table must name the protocol version, evaluation set, checkpoint, and
readout. Do not compare rows that use different evaluation sets as if they
were directly comparable.

## 1. Ground truth and preprocessing

- Ground truth: COCO 2017 **instance** annotations (`things` only).
- Ignore `iscrowd` annotations.
- Non-instance regions, including COCO `stuff`, are background label 0.
- Input crop: CODA `ResizeMinShape(512) -> CenterCrop(512)`.
- Model image grid: 32 x 32 = 1,024 image patch tokens.
- Current native-resolution GT cache: `data/coco_inst_mask_cache_coda512`,
  built directly from the original COCO masks in `mode=coda` for all 5,000
  COCO val2017 image IDs at 512 x 512. It includes `overlap_masks.npy`.
- Legacy cache: `data/coco_inst_mask_cache_coda256`. Keep it only for
  reproducing historical PGOT numbers; do not use it for new E2/CODA tables.
- Instance-overlap pixels: use CODA's exact `preproc_masks_overlap` convention:
  assign GT overlap pixels to background and prediction overlap pixels to a
  fresh label before computing metrics. This prevents overlapping pixels from
  being credited to an object and matches CODA's implementation.

## 2. Metrics

- FG-ARI: ARI over GT foreground instances; GT background is ignored.
- mBO^i: mean, over GT instances, of the best IoU with any predicted region.
- mIoU^i: Hungarian matching including the background label, normalized by
  the number of GT regions.
- Metric definitions live in `pgot/eval/pgot_metrics.py` and are matched to
  `/home/jovyan/coda/src/metric/segmentation.py`.

Always report all three metrics together. Also report per-object-count and,
when available, per-object-size results.

## 3. Required evaluation sets

Report two PGOT evaluation sets separately:

### PGOT-CODA-2338

- The exact 2,338 image IDs in `data/pgot_val.jsonl`.
- Purpose: direct comparison with historical PGOT CODA-aligned results.
- New thing-only manifests must preserve these image IDs while using the
  caption format required by the evaluated model.

### PGOT-COCO-4943

- The 4,943 image IDs in `data/pgot_coco_instance_val.jsonl`.
- Purpose: broad current-model validation close to the full COCO val2017 set.
- This is not numerically interchangeable with PGOT-CODA-2338 or CODA's full
  5,000-image evaluation.

For an official CODA comparison, additionally evaluate all 5,000 COCO
val2017 images with CODA's native 512 x 512 metric pipeline. Images with no
valid foreground must follow CODA's own aggregation behavior.

## 4. Historical E1 evaluation configuration

The E1 launcher `scripts/eval_pgot_instance_e1.sh` currently fixes:

- `grid_size=32`
- `image_preprocess_mode=coda_center_crop`
- `coda_crop_size=512`
- `gt_source=coco_instance`
- `readout=llm_attention`
- `eval_merge=mean`
- `n_ovt_per_object=1`
- `max_objects=50`
- overlap cache enabled

The launcher passes `eval_size=224`, but `pgot/eval/run_eval.py` overrides that
argument with `coco_cache.size` whenever `gt_source=coco_instance`. The current
cache is 256 x 256, so the **effective metric resolution is 256 x 256**, not
224 x 224. This override has existed since the original evaluator commit, so
historical PGOT COCO-instance results made with this cache are also 256 x 256.
They are internally comparable to each other, but they are not the final native
CODA 512 x 512 protocol. Tables must label this as `PGOT-CODA-v1/256`; do not
label it `CODA-native-512`.

### Current E2 evaluation configuration

The E2 launcher `scripts/eval_pgot_lviscap_hardreg_e2.sh` fixes:

- `grid_size=32` (1,024 image patch tokens)
- `image_preprocess_mode=coda_center_crop`, `coda_crop_size=512`
- `gt_source=coco_instance`
- `readout=llm_attention`, `eval_merge=mean`
- `n_ovt_per_object=1`, `max_objects=50`
- `coco_mask_cache=data/coco_inst_mask_cache_coda512`
- effective segmentation metric resolution: 512 x 512
- instance-overlap pixels excluded using the matching 512 overlap cache
- GT hard register mask disabled in standalone evaluation

The default E2 validation manifest still contains 4,943 images, so call this
set `PGOT-COCO-4943/512`; it is native-resolution but not the full 5,000-image
CODA evaluation set.

### Resolution terminology

Do not conflate the following three resolutions:

- Model input: a 512 x 512 CODA crop encoded as a 32 x 32 grid (1,024 tokens).
- Historical E1 segmentation metric: effectively 256 x 256 because its cache
  overrides the launcher's requested 224. Current E2 uses the native 512 x 512
  cache; this change does not require retraining the DiT.
- Reconstruction output: natively 224 x 224 because the frozen Scale-RAE
  target is a 16 x 16 grid of 256 SigLIP-224 features and the pretrained RAE
  decoder predicts 14 x 14 pixels per feature token. Native 512 reconstruction
  requires a new high-resolution decoder or a changed diffusion target; merely
  resizing the 224 output to 512 does not add reconstruction information.

### E6 direct-CODA evaluation configuration

- Current teacher-forced manifest: `data/pgot_pix2cap_generated_val5k.jsonl`
  (4,720 images). Label this set `PGOT-Pix2Cap-4720/512`; it is not CODA's full
  5,000-image set.
- E6 reconstruction is generated natively at 512 x 512 by the CODA SD-v1.5
  decoder. It does not use the Scale-RAE 256-query/DiT/224px path.
- CODA segmentation is read from its **encoder slot-attention** map, not from
  U-Net decoder cross-attention. The corresponding primary E6 readout is
  `llm_attention` (`core` OVT-to-image attention plus register background).
  `e6_decoder` is retained only as a decoder-ownership diagnostic and must not
  be labelled a CODA-equivalent segmentation readout.
- The original CODA evaluation recipe uses FP32 evaluation, 100 DDIM steps and
  guidance 2.0. PGOT's historical reconstruction launcher uses BF16, 10 steps
  and guidance 2.5. Results using these recipes must be labelled separately.
- An E6 checkpoint is valid for evaluation only when PEFT/LoRA is detected and
  injected before loading `base_layer`/`lora_*` tensors. This applies to both
  sharded checkpoints and a single `model.safetensors` file. Summaries record
  `checkpoint_lora_detected`; a false value invalidates a LoRA-trained row.

## 5. Background rule for E1

- Object maps: exact post-RoPE OVT-to-image-patch attention, averaged using
  the checkpoint's configured layers/heads.
- Background map: mean register-to-image-patch attention when there is no
  VOID token.
- An empty VOID tensor must never be treated as a valid background map.

The empty-VOID fallback is implemented in `pgot/eval/run_eval.py`. Historical
E1 segmentation numbers produced before this fix are invalid; reconstruction
metrics from those runs are unaffected.

## 6. Mandatory result metadata

Every saved summary and reported table must include:

- protocol version and evaluation-set name
- exact number of evaluated samples
- checkpoint and global step
- prediction readout and attention layers
- crop size, patch grid, and metric resolution
- overlap handling and background source
- teacher-forced versus autoregressive inference
- random seed, diffusion steps, and guidance scale for reconstruction

For reconstruction-path experiments, also report Full, OVT-only,
Register-only, zero-OVT-input, and zero-register-input under the same fixed
diffusion noise.

## 7. Reference values

CODA's published COCO instance results are:

| Model | FG-ARI | mBO^i | mIoU^i | rFID |
|---|---:|---:|---:|---:|
| CODA | 0.4750 | 0.3630 | 0.3641 | 10.65 |

Segmentation and reconstruction comparisons must still state their respective
evaluation set and image-resolution protocol; sharing a table does not by
itself make mismatched protocols directly comparable.

The corrected E1 checkpoint-10000 result on PGOT-COCO-4943 with the current
effective 256-resolution PGOT evaluator is:

| Model | Set | Readout | FG-ARI | mBO^i | mIoU^i |
|---|---|---|---:|---:|---:|
| E1-10k | PGOT-COCO-4943 | all-layer LLM attention + register background | 0.2948 | 0.3301 | 0.4144 |

These rows are useful references but are not a fully controlled head-to-head
comparison until the evaluation set and native metric resolution are equal.
