# E8.1 Clean Visual Memory 결과 분석

## 결론

E8.1은 **중간 단계 목표에는 성공했고, 최종 목표에는 부분 성공했다.**

- layer 8/16의 잘못된 early ownership write를 제거하고 21/24/27에 동일한
  supervision을 준 결과, 세 layer 모두 object를 안정적으로 찾는다.
- reader supervision은 foreground의 register shortcut과 background의 object leakage를
  크게 줄였다.
- object memory를 제거하거나 같은 category의 다른 instance memory로 바꾸면 해당
  object 영역의 DiT reconstruction error가 실제로 증가한다. 기존 E8에는 없던
  non-gameable causal effect다.
- full reconstruction도 기존 E8보다 명확히 좋아졌다.
- 하지만 visual memory의 instance-appearance residual R2는 0.1190에서 0.1191로
  정체했다. 즉 **더 정확히 읽고 더 실제로 사용하지만, 담긴 appearance 양 자체는
  늘지 않았다.**
- 따라서 object-complete appearance latent라는 최종 목표는 아직 달성하지 못했고,
  E8.2 paired causal 학습이 필요하다.

## 평가 설정

- checkpoint: `checkpoints/pgot_e8_1_clean/checkpoint-10000`
- FP32, 2 GPU, 10,000 steps, global batch 32
- full eval: 4,720 images, COCO instance GT, CODA 512 center crop
- causal/appearance: 동일한 앞 512 images
- appearance probe: 2,635 objects, image-disjoint 80/20 split
- paired swap: 498 images에 대해 이전 이미지의 same-category donor 사용
- paired branch는 semantic key, register, timestep, noise를 고정하고 object visual
  memory 하나만 zero/swap

## Full evaluation

| Metric | E8 | E8.1 | 변화 |
|---|---:|---:|---:|
| fARI | 0.6023 | **0.6141** | +0.0119 |
| mBO | **0.5329** | 0.5289 | -0.0040 |
| mIoU | **0.5790** | 0.5760 | -0.0030 |
| rFID | 23.4459 | **19.3694** | -17.4% |
| PSNR | 10.7878 | **11.4070** | +0.619 dB |
| SSIM | 0.2330 | **0.2620** | +12.5% |
| MSE | 0.09066 | **0.07875** | -13.1% |

Segmentation은 전체적으로 유지 수준이다. fARI는 좋아졌지만 mBO/mIoU가 각각
0.4/0.3 percentage point 낮아졌다. 반대로 가장 취약했던 11+ object bucket은
fARI 0.6823 -> 0.7080, mBO 0.2955 -> 0.3091, mIoU 0.3008 -> 0.3148로 모두 좋아졌다.

Reconstruction은 같은 E8 계열 평가에서 명확히 개선됐다. 과거 E2/E3/E4는 guidance
2.5, E8 계열은 guidance 1.0이므로 이 표의 핵심 비교는 E8 대 E8.1이다.

## Training / train-time evaluation

- 10,000 steps, 약 19.42 epochs, 17.45시간을 정상 완료했다.
- OOM, NaN, traceback, DDP/save/W&B 실패는 없다.
- train loss: step 10의 15.249 -> step 10k의 2.600.
- train-time eval은 매번 32 images라 full eval보다 noisy하다.
- eval total loss 최저: 3.642 at 4k; final 4.227.
- eval reconstruction loss 최저: 0.519 at 7k; final 0.611.
- eval ownership CE 최저: 0.827 at 1.5k; final 1.394.
- eval reader CE 최저: 0.580 at 4k; final 0.780.
- owner FG accuracy: 0.783 at 500 -> 0.800 at 10k.
- reader matching mass on FG: 0.232 -> 0.541.
- reader register mass on FG: 0.305 -> 0.132.
- reader object mass on BG: 0.522 -> 0.238.

Accuracy와 routing mass는 유지/개선되는데 CE가 후반에 증가한다. 이는 object를 못
찾는 붕괴가 아니라 ambiguous boundary/overlap의 일부 오답에 지나치게 높은 confidence를
주는 calibration/generalization 문제다. 같은 설정으로 E8.1을 더 오래 학습하는 것은
권장하지 않는다.

## Writer / reader 진단

| Writer layer | FG acc | BG acc | Register prob on FG | Object prob on BG | Entropy |
|---|---:|---:|---:|---:|---:|
| 21 | 0.8298 | 0.9087 | 0.0845 | 0.0897 | 0.7017 |
| 24 | 0.8308 | 0.9165 | 0.0776 | 0.0822 | 0.5876 |
| 27 | 0.8319 | 0.9167 | 0.0719 | 0.0820 | 0.5314 |

기존 E8의 layer 8/16 FG accuracy는 0.027/0.156이었지만 E8.1은 첫 write인 layer
21부터 0.830이다. Early ownership 실패는 해결됐다. 21 -> 27에서 accuracy 증가는
거의 없고 entropy만 낮아져, 후반 write가 location을 새로 발견한다기보다 확신을
강화한다.

| Reader diagnostic | E8 | E8.1 |
|---|---:|---:|
| Object binding AUROC | 0.7318 | **0.8959** |
| Object binding inside/outside | 2.87 | **17.28** |
| Top-area precision | 0.1855 | **0.3884** |
| Register attention on GT FG | 0.5587 | **0.2788** |
| Object attention on GT BG | 0.2379 | **0.1452** |
| Single-zero delta inside/outside | 2.74 | **16.22** |

따라서 E8.1의 가장 확실한 성공은 reader locality와 object/register 역할 분리다.
Register foreground shortcut은 절반으로 줄었지만 27.9%가 남아 완전히 사라진 것은
아니다.

## Global causal memory ablation

동일 image/noise에서 memory component만 제거한 512-image diffusion training loss:

| Condition | E8 delta | E8.1 delta | E8.1 상대 변화 |
|---|---:|---:|---:|
| Register zero (object memory only) | +0.0444 | +0.0498 | +8.1% |
| Object zero (register memory only) | +0.0114 | **+0.0322** | **+5.3%** |
| All memory zero | +0.1035 | +0.0891 | +14.5% |

Object-zero 영향은 2.8배 커졌다. Object memory가 reconstruction에 사용되는 정도는
명확히 증가했다. 다만 global 기준에서는 register-zero 영향 8.1%가 object-zero
5.3%보다 아직 크다. Background가 이미지 대부분을 차지하는 효과도 있으므로 최종
판정은 아래 area-normalized single-object swap을 사용한다.

## Same-category paired swap

Semantic key는 원래 object 것으로 유지하고 visual memory만 같은 category의 다른
이미지 instance 것으로 교체했다. 498개 image에서 object-mask 면적으로 정규화한 결과:

| Metric | E8 | E8.1 |
|---|---:|---:|
| Swap selected-error ratio, mean | 0.9990 | **1.0430** |
| Swap selected-error ratio, median | 1.0018 | **1.0144** |
| Swap ratio mean 95% CI | [0.9847, 1.0133] | **[1.0249, 1.0611]** |
| Zero selected-error ratio, mean | 1.0071 | **1.0569** |
| Swap prediction delta inside | 0.01445 | **0.04173** |
| Swap prediction delta outside | 0.00423 | 0.00698 |
| Swap delta inside/outside | 4.29 | **12.25** |

기존 E8은 same-category swap ratio의 CI가 1을 포함하므로 사실상 swap-insensitive하다.
E8.1은 평균 4.3%의 유의한 selected-object error 증가가 있고 변화가 object 내부에
강하게 국소화된다. 이 지표는 zero 감지 shortcut으로 설명할 수 없으므로, E8.1
object visual memory의 instance 정보가 reconstruction에 **causally 사용되기 시작했다**는
증거다.

다만 median은 1.4% 증가에 불과하고 sample 간 편차가 크다. 일부 object에서는 강하게
사용하지만 모든 object가 균일하게 appearance-complete하지는 않다.

## Appearance probe

Word/category prediction을 제거한 held-out instance residual R2:

| Representation | E8 | E8.1 |
|---|---:|---:|
| Semantic OVT | 0.0685 | 0.0706 |
| Object visual memory | **0.1190** | **0.1191** |
| LLM image stream | 0.1444 | **0.1641** |
| Raw SigLIP pooled ceiling | 0.3343 | 0.3343 |

Visual memory는 semantic OVT보다 instance appearance를 더 많이 담지만, E8.1에서
정보량은 늘지 않았다. 오히려 LLM image stream에는 0.164까지 정보가 생겼는데 memory는
0.119에 머물러 writer가 available appearance를 모두 전달하지 못한다. E8.2가 개선할
수 있는 immediate headroom은 우선 0.119 -> 0.164이며, 그 이후 raw ceiling 0.334와의
차이는 raw-feature value를 쓰는 E8.3 문제다.

## 목표별 최종 판정

| 목표 | 판정 | 근거 |
|---|---|---|
| Early ownership 오염 제거 | 달성 | layer 21/24/27 FG acc 모두 약 0.83 |
| Reader object/register 역할 분리 | 크게 달성 | FG register mass 55.9% -> 27.9%, AUROC 0.896 |
| Reconstruction 유지/개선 | 달성 | rFID -17.4%, MSE -13.1% vs E8 |
| Object memory를 reconstruction이 사용 | 달성 | object-zero +5.3%, same-category swap +4.3% |
| Object memory appearance 정보 증가 | 미달성 | residual R2 0.1190 -> 0.1191 |
| 모든 object에 강한 exchangeable latent | 미달성 | swap median +1.4%, 큰 sample 편차 |
| 최종 연구 목표 | 부분 달성 | causal path는 생겼지만 object-complete하지 않음 |

## 다음 결정

E8.1을 다시 학습하거나 더 오래 연장할 필요는 없다. E8.2로 진행하는 것이 맞다.
E8.2의 역할은 새 path를 처음 만드는 것이 아니라, 이미 생긴 localized causal path를
모든 object에서 강하게 만드는 것이다.

다만 ownership/reader CE가 후반 과적합하므로 E8.2에서 동일 supervision을 같은 LR로
5k 더 강하게 미는 것은 피한다. 권장 방향은:

1. `checkpoint-10000`에서 E8.2 시작.
2. paired causal probability 0.25, 1k ramp 유지.
3. owner/reader loss 또는 해당 learning rate를 낮춰 이미 확보한 routing을 보존.
4. 500/1k/2.5k/5k에서 paired swap ratio, appearance residual R2, FG register mass를 확인.
5. swap은 증가하지만 R2가 0.16 부근에서 정체하면 E8.3 raw-feature value로 이동.

## Reproducibility outputs

- `outputs/eval_pgot_e8_1_clean/summary.json`
- `outputs/analysis_e8_1_clean_causal_512/summary.json`
- `outputs/analysis_e8_1_clean_appearance_probe_512/summary.json`
- `outputs/analysis_e8_1_clean_paired_swap_512/summary.json`
- `outputs/analysis_e8_visual_memory_paired_swap_512/summary.json`
- `checkpoints/pgot_e8_1_clean/trainer_state.json`
- `wandb/run-20260806_081226-1c9xsoum/files/output.log`
