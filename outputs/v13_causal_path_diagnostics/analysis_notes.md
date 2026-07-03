# V13 causal path diagnostic notes

## Files
- `summary.json`: all metrics for samples 0, 1020, 1407.
- `V13/sample*.json`: per-sample detailed metrics.
- `V13/sample*_causal_recon_grid.png`: source + baseline/access/zero/swap recon grids.

## Key numbers

| sample | baseline | OVT-only delta | register-only delta | self-only delta | zero OVT delta | zero register delta | swap delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5456 | +0.6309 | +0.0009 | +1.8190 | -0.0017 | +0.3007 | +1.9685 |
| 1020 | 0.6585 | +0.6248 | +0.0033 | +1.7461 | +0.0062 | +0.2872 | +2.0170 |
| 1407 | 0.8034 | +0.4893 | +0.0005 | +1.6122 | +0.0141 | +0.2913 | +1.9435 |

Interpretation: blocking OVT access barely changes reconstruction when register access remains, while blocking register access makes reconstruction much worse. Zeroing OVT inputs also has almost no effect.

## RAE query QK source mass, average over last 4 layers

| sample | OVT | register | RAE self | image |
|---:|---:|---:|---:|---:|
| 0 | 0.0011 | 0.2434 | 0.7555 | 0.0000 |
| 1020 | 0.0022 | 0.2555 | 0.7423 | 0.0000 |
| 1407 | 0.0057 | 0.2583 | 0.7360 | 0.0000 |

Interpretation: RAE queries are not reading OVTs in any meaningful amount. They mostly use self and registers.

## Gradient comparison

| sample | recon grad OVT | recon grad register | mask grad OVT | mask/recon OVT ratio | cosine OVT |
|---:|---:|---:|---:|---:|---:|
| 0 | 5.72e-02 | 4.14e-01 | 4.25e-04 | 7.42e-03 | 0.075 |
| 1020 | 6.00e-02 | 4.37e-01 | 2.43e-05 | 4.04e-04 | 0.167 |
| 1407 | 3.46e-02 | 3.40e-01 | 1.94e-06 | 5.61e-05 | 0.052 |

Interpretation: after V13 collapse, outside/register mask losses are nearly saturated and provide almost no useful gradient. Recon gradients are much stronger through register/RAE than through OVT. Recon and outside gradients are nearly orthogonal on OVT positions.

## Main conclusion
V13 does not fail merely because eval readout is bad. It fails because the reconstruction path is register/self dominated, while the supervised dot-product OVT map has collapsed. Reconstruction is not forcing OVTs to carry object-local visual content.
