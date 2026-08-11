# E8.1-A Writer-GT / Reader-Writer 분석

## 결론

E8.1-A는 기존 E8.1의 GT Reader supervision을 제거하고, Reader가 detached final
Writer ownership을 따라가도록 학습한 실험이다. 분석 결과는 다음과 같다.

- 기존 GT/GT E8.1의 object discovery, reconstruction, reader locality를 유지했다.
- Reader의 foreground register leakage는 27.9%에서 22.5%로 감소했다.
- same-category memory swap은 selected object error를 평균 5.27% 증가시켰으며,
  95% CI가 1을 넘는다. Object visual memory는 reconstruction에 causally 사용된다.
- visual-memory instance residual R2는 0.1162로 기존 0.1191과 사실상 동일하다.
  Reader GT를 제거해도 appearance 정보량은 유지되지만 저장 병목은 해결되지 않았다.
- 따라서 A는 다음 E8.2의 더 좋은 출발점이지만, Writer GT supervision까지 제거된
  mask-free 학습이나 object-complete appearance latent는 아직 달성하지 못했다.

## 분석 설정

- checkpoint: `checkpoints/pgot_e8_ablation_a_writer_gt_reader_writer/checkpoint-10000`
- A 학습: Writer GT ownership CE, Reader detached Writer ownership supervision
- FP32, 512 images, batch 4, CODA 512 center crop
- appearance probe: 2,635 objects, image-disjoint 80/20 split
- paired swap: 498 images with a previous-image same-category donor
- paired branches share semantic keys, registers, RF timestep, and noise
- qualitative grids use the same bear/person/bus pairs and seeds as GT/GT E8.1

## Full evaluation

| Metric | GT/GT E8.1 | A: GT/Writer | Change |
|---|---:|---:|---:|
| fARI | 0.6141 | 0.6131 | -0.0011 |
| mBO | 0.5289 | **0.5368** | +0.0079 |
| mIoU | 0.5760 | **0.5834** | +0.0074 |
| rFID | 19.3694 | **18.9945** | -0.3749 |
| PSNR | 11.4070 | **11.4394** | +0.0324 dB |
| SSIM | **0.2620** | 0.2597 | -0.0023 |
| MSE | 0.07875 | **0.07847** | -0.00028 |

작은 차이는 diffusion sampling 변동을 고려해 동률 수준으로 보는 것이 안전하다.
중요한 결과는 Reader GT supervision을 제거해도 성능이 나빠지지 않았다는 점이다.

## Writer와 Reader locality

| Reader diagnostic | GT/GT E8.1 | A: GT/Writer |
|---|---:|---:|
| Binding AUROC | 0.8959 | **0.8967** |
| Inside/outside ratio | 17.28 | **17.71** |
| Top-area precision | 0.3884 | **0.3951** |
| Register attention on GT FG | 0.2788 | **0.2246** |
| Object attention on GT BG | **0.1452** | 0.1585 |
| Single-zero delta inside/outside | **16.22** | 16.12 |

A는 foreground에서 register를 읽는 비율을 5.42 percentage point 줄였다. 대신
background에서 object를 읽는 비율은 1.33 point 증가했다. 전체적으로는 object 영역의
reader binding이 유지되거나 소폭 개선됐고, foreground shortcut은 줄었다.

| Writer layer | FG acc | BG acc | Register prob on FG | Object prob on BG |
|---|---:|---:|---:|---:|
| 21 | 0.8265 | 0.9207 | 0.0950 | 0.0798 |
| 24 | 0.8222 | 0.9263 | 0.0945 | 0.0733 |
| 27 | 0.8242 | 0.9262 | 0.0899 | 0.0729 |

세 layer 모두 안정적이다. Layer 21 이후 object accuracy는 증가하지 않고 entropy만
낮아지므로, 후반 Writer는 새 위치를 찾기보다 기존 assignment를 sharpen한다.

## Global memory ablation

동일 image와 noise에서 memory component만 제거한 diffusion training loss이다.

| Intervention | GT/GT delta | A delta | A relative to full |
|---|---:|---:|---:|
| Register zero: object memory only | +0.04978 | +0.04703 | +7.69% |
| Object zero: register memory only | +0.03219 | +0.03139 | +5.13% |
| All memory zero | +0.08915 | +0.08413 | +13.75% |

A에서도 object memory를 제거하면 reconstruction loss가 증가한다. Causal dependence는
GT/GT E8.1과 거의 동일하다. 다만 global loss에서는 register-zero 영향이 더 크며,
이는 background 면적과 register memory의 중요성이 여전히 크다는 뜻이다.

## Same-category paired swap

| Metric | GT/GT E8.1 | A: GT/Writer |
|---|---:|---:|
| Swap selected-error ratio mean | 1.0430 | **1.0527** |
| Swap selected-error ratio median | 1.0144 | **1.0164** |
| Swap ratio approximate 95% CI | [1.0249, 1.0611] | **[1.0341, 1.0714]** |
| Zero selected-error ratio mean | **1.0569** | 1.0544 |
| Swap prediction delta inside | 0.04173 | **0.04779** |
| Swap prediction delta outside | **0.00698** | 0.00788 |
| Swap delta localization ratio | 12.25 | **12.45** |

A는 same-category donor memory로 교체했을 때 selected-object reconstruction error가
평균 5.27% 증가한다. CI가 1을 넘으므로 instance memory가 causally 사용된다는 증거다.
기존 GT/GT보다 mean effect는 크지만 두 checkpoint 간 증가량 자체는 작아, 반복 seed
없이 A가 통계적으로 더 강하다고 단정하지 않는다. 핵심은 GT Reader 없이 causal
effect를 보존했다는 것이다.

## Appearance probe

Word/category prediction을 제거한 held-out instance residual R2이다.

| Representation | GT/GT E8.1 | A: GT/Writer |
|---|---:|---:|
| Semantic OVT | 0.0706 | 0.0705 |
| Object visual memory | **0.1191** | 0.1162 |
| LLM image stream | **0.1641** | 0.1631 |
| Raw SigLIP pooled ceiling | 0.3343 | 0.3343 |

모든 representation이 기존 E8.1과 사실상 같다. A는 appearance content를 잃지 않았지만
visual memory가 LLM image stream에 존재하는 정보조차 모두 전달하지 못한다. Reader
supervision 변경은 read 경로를 개선하지만 write-value capacity를 개선하지 않는다.

## Qualitative intervention

- Bear: full의 한 갈색 곰이 same-category swap에서 donor 계열의 두 곰 구도로 변한다.
- Person: category는 유지하면서 인물 구성, 복장, 배치가 변한다.
- Bus: full의 주황색 coach가 donor와 유사한 빨간 double-decker로 변한다.
- Difference map은 object 중심이지만 인접 background까지 변화해 완벽한 local editing은 아니다.

세 예시 모두 swap loss가 full보다 높았다. Bear와 bus에서 transfer가 특히 선명하고,
person은 정량 loss 변화가 작아 object별 causal strength 편차가 남아 있다.

## 목표별 판정

| Goal | Judgment |
|---|---|
| Reader GT mask 제거 | 달성 |
| Writer object discovery 유지 | 달성 |
| Reconstruction 유지 | 달성 |
| Object memory causal use 유지 | 달성 |
| Same-category appearance transfer | 부분 달성 |
| Appearance information 증가 | 미달성 |
| Writer GT mask 제거 | 미달성 |
| Object-complete latent | 미달성 |

## 다음 결정

A를 E8.1의 대표 checkpoint이자 E8.2의 시작점으로 사용하는 것이 합리적이다. A는
GT Reader mask 없이도 Writer가 만든 ownership을 Reader까지 일관되게 전달하며,
reconstruction과 causal swap 효과를 보존한다. 다음 E8.2에서는 paired causal loss로
object별 사용 강도의 편차를 줄여야 한다. Appearance residual R2가 0.16 부근에서
정체하면 Writer value source를 raw SigLIP/DINO feature로 바꾸는 다음 단계가 필요하다.

## Outputs

- `outputs/eval_pgot_e8_ablation_a_writer_gt_reader_writer/summary.json`
- `outputs/analysis_e8_ablation_a_causal_512/summary.json`
- `outputs/analysis_e8_ablation_a_appearance_probe_512/summary.json`
- `outputs/analysis_e8_ablation_a_paired_swap_512/summary.json`
- `outputs/analysis_e8_ablation_a_visual_examples/summary.json`
