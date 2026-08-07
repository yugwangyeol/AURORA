# E8 Visual Memory 결과 분석

## 결론

E8은 **ownership과 representation collapse 문제는 크게 개선했지만, object
memory를 reconstruction의 필수 병목으로 만드는 목표는 아직 달성하지 못했다.**

- 최종 object ownership은 이전 one-OVT 실험 중 가장 좋다.
- E8 object visual memory에는 이전 OVT보다 훨씬 많은 instance appearance가 있다.
- visual memory 전체는 reconstruction에 실제로 사용된다.
- 하지만 object memory를 제거해도 reconstruction loss가 1.8%만 증가한다.
- 반대로 register를 제거하면 7.1%, 모든 memory를 제거하면 16.7% 증가한다.
- 따라서 현재 reconstruction의 주 경로는 여전히 register이고, object memory는
  보조 경로다.

핵심 원인은 layer 8/16의 초기 write가 foreground를 register에 대량 기록한 뒤,
마지막 layer 27의 좋은 ownership map만 평가·감독한다는 데 있다. 누적 memory는
초기 오염을 유지하므로 최종 segmentation과 실제 memory 역할이 불일치한다.

## 평가 설정

- checkpoint: `checkpoints/pgot_e8_visual_memory/checkpoint-10000`
- FP32, two-GPU training, 10,000 steps
- full evaluation: 4,720 images, one OVT per object, teacher-forced captions
- COCO instance GT, CODA 512 center crop, overlap excluded
- causal/appearance diagnostics: 앞 512 images, 2,635 appearance-probe objects,
  2,742 causal binding objects

## Full evaluation

| Model | fARI | mBO | mIoU | rFID | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| E2 | 0.1425 | 0.2534 | 0.3350 | 20.8517 | 10.6554 | 0.2319 |
| E3 | 0.4917 | 0.4028 | 0.4681 | 27.3911 | 9.8965 | 0.1997 |
| E4 | 0.5455 | 0.4032 | 0.4662 | 22.4169 | 10.3714 | 0.2226 |
| FVW | 0.0922 | 0.2174 | 0.2948 | 23.5756 | 10.5472 | 0.2197 |
| E7 | 0.4978 | 0.4711 | 0.5199 | 421.6425 | 9.9861 | 0.3357 |
| **E8** | **0.6023** | **0.5329** | **0.5790** | 23.4459 | **10.7878** | **0.2330** |

E8은 모든 object-count bucket에서 E7/E4보다 segmentation이 좋다. 그러나
crowded scene에서는 fARI와 mBO가 다르게 움직인다. 11+ objects에서 fARI는
0.6823이지만 mBO는 0.2955다. 큰 object의 pixel partition은 잘 맞아도 작은
object를 놓치는 문제가 남았다는 뜻이다.

rFID는 E2보다 2.59 높아 나쁘고 E4보다 1.03 높다. E8 evaluation은 guidance
1.0, 이전 E2/E3/E4는 기본 guidance 2.5여서 완전히 동일한 generation protocol은
아니므로 작은 rFID 차이는 과해석하지 않는다. PSNR/MSE는 E8이 소폭 개선됐지만
overall reconstruction quality가 크게 도약한 결과는 아니다.

## Training/evaluation dynamics

- train은 10,000 steps, 약 19.42 epochs를 정상 완료했다.
- OOM, NaN, traceback은 없고 2-GPU DDP, FP32, save/eval/W&B가 정상 동작했다.
- eval set은 매번 32장뿐이므로 W&B validation curve는 noisy하다.
- eval LM loss: 1.2898 at 500 -> 1.1252 at 10k.
- eval reconstruction loss: 0.6084 at 500 -> 0.6247 at 10k; best 0.5225 at 7k.
  장기적으로 뚜렷하게 개선되지 않았다.
- eval owner FG accuracy: 0.7980 -> 0.8139.
- eval owner BG accuracy: 0.8156 -> 0.8795.
- register probability on FG: 0.1025 -> 0.0621.
- object probability on BG: 0.1508 -> 0.1169.
- owner entropy: 1.1908 -> 0.5955.
- owner loss: 0.9203 -> 1.6443으로 오히려 증가했다.

Accuracy는 유지되면서 entropy가 낮아지고 CE loss가 커진 것은 일부, 특히 작은
object의 오답에 점점 더 높은 confidence를 주는 현상과 일치한다. 마지막 train
batch의 owner loss 0.2204 대비 32-image eval owner loss 1.6443도 큰 generalization
gap이다. 현재 constant LR로 19 epochs를 더 학습하는 것은 해결책이 아니다.

## Appearance probe

Caption/word predictor가 설명하는 성분을 제거한 held-out instance residual R2:

| Representation | residual R2 | Effective dimension (PR) |
|---|---:|---:|
| E2 isolated OVT | 0.0147 | 12.55 |
| FVW OVT | 0.0495 | 4.65 |
| E8 OVT semantic state | 0.0685 | 72.81 |
| **E8 object visual memory** | **0.1190** | **92.99** |
| E8 LLM image stream | 0.1444 | 98.30 |
| Raw SigLIP pooled ceiling | 0.3343 | 86.20 |

E8 object memory는 E8 image stream이 보유한 residual appearance의 약 82%를
linear probe로 회수하고 raw SigLIP ceiling의 약 36%를 회수한다. 따라서
"object memory에 visual information이 없다"는 이전 실패는 고쳤다. 또한 FVW의
4.65-dimensional collapse도 사라졌다. 하지만 raw visual ceiling과의 차이가 커서
object-complete appearance representation이라고 부르기는 이르다.

## Causal reconstruction diagnostic

동일 image/동일 diffusion noise로 memory component만 제거한 512-image denoising
loss:

| Condition | Loss | Baseline 대비 | 상대 변화 |
|---|---:|---:|---:|
| Full memory | 0.6216 | - | - |
| Object memory only (register zero) | 0.6659 | +0.0444 | +7.1% |
| Register memory only (object zero) | 0.6330 | +0.0114 | +1.8% |
| All memory zero | 0.7251 | +0.1035 | +16.7% |

모든 memory를 지우면 loss가 충분히 증가하므로 typed visual-memory path 자체는
실제로 사용된다. 그러나 object memory를 모두 지운 register-only 조건이 baseline과
거의 같다는 것이 결정적인 실패 신호다. Object memory는 사용되지만 필수적이지
않고, register가 foreground reconstruction까지 대부분 담당한다.

RAE reader도 같은 결론을 보인다.

- GT foreground query에서 register attention mass: 0.5587
- GT background query에서 object attention mass: 0.2379
- Object reader binding: AUROC 0.7318, inside/outside 2.87
- 한 object memory를 zero했을 때 RAE delta localization:
  AUROC 0.7305, inside/outside 2.74

즉 object-to-spatial-query binding은 실제로 존재한다. 문제는 binding 부재가 아니라
reader가 foreground에서도 register를 과도하게 읽는 것이다.

## Layerwise writer diagnosis

| Writer layer | FG acc | BG acc | Register prob on FG | Object prob on BG |
|---|---:|---:|---:|---:|
| 8 | 0.0274 | 0.9205 | 0.8024 | 0.1766 |
| 16 | 0.1560 | 0.6152 | 0.5379 | 0.4490 |
| 24 | 0.6873 | 0.9576 | 0.2771 | 0.0528 |
| 27 | 0.8304 | 0.9256 | 0.0925 | 0.0721 |

Visual memory는 네 layer의 write를 누적하지만 owner loss는 마지막 layer output에만
직접 적용된다. 따라서 layer 8에서는 foreground visual evidence의 80%가 register로,
layer 16에서는 54%가 register로 기록된다. Layer 27의 segmentation만 좋아져도 이미
오염된 register memory는 지워지지 않는다.

기존 D1 결과도 이 shortcut을 강화한다. Final SigLIP background patch 자체가 removed
foreground에 대해 far-background relative L2 0.3334만큼 변한다. 즉 spatial owner가
background patch를 선택해도 그 value가 foreground appearance-free라는 보장이 없다.

## 목표별 판정

| 목표 | 판정 | 근거 |
|---|---|---|
| Object ownership 발견 | 달성 | fARI/mBO/mIoU 모두 큰 개선 |
| Object memory에 visual information 저장 | 부분 달성 | residual R2 0.1190, collapse 해소 |
| Visual memory 전체를 reconstruction이 사용 | 달성 | all-zero loss +16.7% |
| Object memory를 reconstruction이 필수 사용 | 미달성 | object-zero loss +1.8% |
| Object/register 역할 분리 | 미달성 | early write leakage, FG register reader mass 55.9% |
| Object-complete/exchangeable appearance latent | 미확인/미달성 | raw ceiling과 큰 차이, swap transfer 검증 전 |

## 다음 실험의 최소 수정안

E8을 폐기할 이유는 없다. Ownership과 non-collapsed visual memory라는 두 기반은
확보했다. 다음 실험은 구조를 크게 늘리기보다 현재 드러난 두 shortcut만 막아야 한다.

1. **Early write 제거 또는 전 layer owner supervision**
   - 우선 가장 싼 ablation은 writer layer를 `24,27`로 줄이는 것이다.
   - 네 write를 유지한다면 8/16/24/27 모두 owner loss를 걸어야 한다.
   - 마지막 reliable write가 이전 memory를 overwrite/clean하도록 만들어야 한다.

2. **Reader ownership supervision**
   - 16x16 RAE query에서 object slot은 해당 object 영역, register는 background를
     읽도록 reader attention에도 loss를 건다.
   - 현재 writer segmentation만 감독하고 reader route는 자유라 FG에서 register
     mass가 55.9%까지 올라간다.

3. **Phase-2 paired causal loss 추가**
   - 같은 sample에서 full/object-zero/register-zero를 함께 계산한다.
   - own object memory를 지우면 해당 foreground denoising loss가 증가해야 한다.
   - register-only 조건은 foreground를 복원하지 못하도록 한다.
   - random batch별 route가 아니라 동일 noise의 paired counterfactual이어야 한다.

4. **Register value leakage 차단**
   - register write에는 early/local SigLIP feature를 쓰거나 foreground-invariant value
     constraint를 추가한다.
   - final globally contextualized SigLIP background value만으로는 foreground-free
     register를 보장할 수 없다.

추천 순서는 `writes=24,27 + 두 layer owner loss + reader route loss`로 E8.1을 먼저
확인한 뒤, object-zero가 여전히 약하면 paired causal loss를 추가하는 것이다. 현재
checkpoint를 더 오래 학습하는 것은 권장하지 않는다.

## Reproducibility outputs

- `outputs/eval_pgot_e8_visual_memory/summary.json`
- `outputs/analysis_e8_visual_memory_appearance_probe_512/summary.json`
- `outputs/analysis_e8_visual_memory_appearance_probe_512/spectra.json`
- `outputs/analysis_e8_visual_memory_causal_512/summary.json`
- `checkpoints/pgot_e8_visual_memory/trainer_state.json`
- `wandb/run-20260805_065744-z45hua51/files/output.log`
