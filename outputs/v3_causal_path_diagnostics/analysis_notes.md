# V3 causal path diagnostic notes

Compared with V13 using the same no-training causal diagnostic. Samples: 0, 1020, 1407.

## Files
- `summary.json`: all metrics.
- `V3/sample*.json`: per-sample detailed metrics.
- `V3/sample*_causal_recon_grid.png`: source + baseline/access/zero/swap recon grids.

## Reconstruction access ablation

| sample | baseline | OVT-only delta | register-only delta | self-only delta | zero OVT delta | zero register delta | swap delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5313 | +0.5809 | +0.0003 | +3.1441 | +0.0042 | +0.2298 | +7.3112 |
| 1020 | 0.6618 | +0.5058 | -0.0020 | +3.1334 | -0.0147 | +0.0673 | +7.9269 |
| 1407 | 0.8304 | +0.4533 | +0.0002 | +3.0717 | +0.0011 | +0.3301 | +7.8691 |

Interpretation: RAE reconstruction can almost fully preserve baseline when it is allowed to use only registers. OVT-only is much worse. Zeroing OVT inputs barely changes reconstruction.

## RAE query QK source mass, average over last 4 layers

| sample | OVT | register | RAE self | image |
|---:|---:|---:|---:|---:|
| 0 | 0.0031 | 0.6444 | 0.3525 | 0.0000 |
| 1020 | 0.0078 | 0.6312 | 0.3609 | 0.0000 |
| 1407 | 0.0205 | 0.6271 | 0.3524 | 0.0000 |

Interpretation: V3 RAE queries read registers far more than OVTs. OVT mass remains tiny.

## Gradient comparison: recon vs V3 full BCE

| sample | recon grad OVT | recon grad register | selected BCE grad OVT | BCE/recon OVT ratio | cosine OVT |
|---:|---:|---:|---:|---:|---:|
| 0 | 6.21e-02 | 1.69e+00 | 4.62e-01 | 7.43 | 0.007 |
| 1020 | 1.12e-01 | 1.56e+00 | 1.63e-01 | 1.46 | 0.081 |
| 1407 | 1.02e-01 | 1.96e+00 | 3.48e-02 | 0.34 | -0.019 |

Interpretation: V3 BCE supervision is active and strong, unlike V13 outside-only collapse. However, the BCE gradient and reconstruction gradient are nearly orthogonal on OVT tokens. Segmentation readout learning and reconstruction content learning are not well aligned.

## Main conclusion
V3 is a good segmentation/readout baseline, but this diagnostic does not support the claim that reconstruction is primarily using OVT object content. Reconstruction is register-dominated. V3's better segmentation score likely comes from direct BCE supervision on the OVT-patch dot-product map, not from OVT being a clean generative object bottleneck.
