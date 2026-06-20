# Outside Loss Failure Diagnostics

Run:
- Output dir: `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128`
- Dataset subset: first 128 validation samples plus presentation samples `0,1020,1407`
- GT: `coco_instance`
- CODA overlap pixels: excluded
- No checkpoint-by-checkpoint sweep was run.

## Metric Definitions

- `self_mass_frac`: for one OVT/object map, normalized map mass inside its own GT mask.
- `other_region_mass_frac`: normalized map mass inside other annotated regions.
- `own_gt_patch_win_frac`: among patches belonging to an object's GT mask, fraction where that object's map is the highest-scoring OVT map. This is closer to actual object ownership than self mass.
- `bg_miss_frac`: GT foreground pixels predicted as background.
- `split_frac`: GT foreground not assigned to each GT object's dominant predicted cluster.
- `merge_frac`: predicted foreground clusters mixing multiple GT objects.

## Subset Results

| Model | fARI | mBO | mIoU | bg miss | split | merge | thing self mass | thing ownership |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V3 | 37.76 | 37.86 | 47.64 | 28.50 | 17.30 | 7.37 | 60.85 | 66.94 |
| V8.2 outside spatial | 24.13 | 29.10 | 39.12 | 27.06 | 29.61 | 14.76 | 68.35 | 54.73 |
| V8.3 BCE + outside | 38.65 | 36.85 | 46.56 | 26.36 | 17.36 | 8.96 | 57.00 | 65.22 |
| V11 balanced BCE + CE | 39.74 | 34.23 | 43.61 | 19.18 | 16.52 | 11.90 | 41.09 | 69.81 |
| V8.5 internal outside | 21.70 | 28.57 | 39.89 | 42.98 | 31.71 | 11.36 | 67.98 | 56.59 |

## Full Eval Reference

| Model | fARI | mBO | mIoU | rFID |
| --- | ---: | ---: | ---: | ---: |
| V3 | 36.39 | 37.84 | 46.41 | 26.43 |
| V8.2 outside spatial | 22.16 | 28.97 | 37.31 | 28.66 |
| V8.3 BCE + outside | 37.33 | 37.19 | 45.51 | 28.81 |
| V11 balanced BCE + CE | 40.81 | 35.50 | 43.47 | 29.22 |
| V8.5 internal outside | 20.57 | 28.30 | 38.97 | 47.89 |

## Main Evidence

1. Outside-only does not fail because it cannot put mass inside the object.
   - V8.2 thing self mass is `68.35%`, higher than V3's `60.85%`.
   - V8.5 internal outside thing self mass is `67.98%`.
   - But their thing ownership is only `54.73%` and `56.59%`, far below V3/V11.

2. This means "do not look outside" is not the same as "own the object in the final partition."
   - V8.2/V8.5 maps can satisfy outside pressure while many patches are still won by other OVTs, stuff, void/background, or noisy competing regions.
   - FG-ARI penalizes this heavily because it evaluates foreground clustering/ownership, not only per-token inside mass.

3. The main observed error mode is split and foreground miss.
   - V8.2 split is `29.61%`, much worse than V3 `17.30%`.
   - V8.5 split is `31.71%`, and bg miss is `42.98%`.
   - V8.5 also has very poor full-eval rFID `47.89`, so the issue is not only the readout.

4. Large-object paradox:
   - V8.2/V8.5 large-object self mass is about `97%`, but large-object ownership is only `47.82%` / `57.02%`.
   - This is the cleanest sign that outside loss can look successful according to its own metric while failing segmentation ownership.

## Visual Files To Use

- Sample 1020 final pred overlays:
  `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128/sample_1020/comparison_pred_overlays.png`
- Sample 1020 all-region ownership before thing-only mapping:
  `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128/sample_1020/comparison_all_region_winners.png`
- Sample 1407 final pred overlays:
  `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128/sample_1407/comparison_pred_overlays.png`
- Sample 1407 all-region ownership before thing-only mapping:
  `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128/sample_1407/comparison_all_region_winners.png`
- Sample 0 simple-object sanity:
  `/home/jovyan/PGOT/outputs/outside_failure_diagnostics_128/sample_0000/comparison_pred_overlays.png`

## Interpretation

The evidence supports demoting outside loss from the main supervision to a weak auxiliary. The main loss still needs a positive object signal and/or explicit ownership competition:

`LM + recon + object-balanced BCE + competition CE + weak outside auxiliary`

Outside loss is still conceptually useful as a leakage regularizer, but it is not sufficient as the principal mask/attention supervision for CODA-style FG-ARI.
