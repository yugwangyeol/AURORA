# E11 / pooled / owner_masked 연구 검토

2026-09-08. 저장된 평가·진단 JSON, 학습 스크립트, checkpoint config, 현재 모델 코드와 bear swap 예시를 검토했다. 새 학습이나 모델 평가는 실행하지 않았다. 아래 실험 제안의 성능은 아직 검증되지 않았다.

## 결론

object visual latent를 유지하려면 **최종 semantic owner로 object별 visual token 집합을 한 번 만들고, Reader가 선택된 owner의 압축 token 안에서 content-dependent attention을 수행**하는 방향을 우선 검토한다. 반복 Writer를 없애는 것과 object visual bottleneck을 없애는 것은 별개 선택이다. owner_masked는 압축을 풀었을 때의 reference/teacher로 보존할 가치가 있다.

현재 자료는 압축 용량과 읽기 방식 모두를 의심하게 한다. 단순히 “1024 patches를 76 tokens로 줄여서 rFID 14가 필연적”이라고 결론내릴 수 없다. 특히 E11 Reader에는 아래와 같은 구체적인 주소 지정 제약이 있다.

## 평가 결과

각 행의 출처는 `outputs/<평가 폴더>/summary.json`. 모두 4,720 images, CODA center crop, instance 평가. AR 표기 외에는 teacher-forced captions. checkpoint 번호는 해당 stage의 step이며 전체 학습량을 뜻하지 않는다.

| 평가 폴더 | rFID ↓ | PSNR ↑ | fARI ↑ | mIoU ↑ |
|---|---:|---:|---:|---:|
| eval_pgot_e8_1_clean | 19.369 | 11.407 | .6141 | .5760 |
| eval_pgot_e9_1_final_ovt | 18.438 | 11.751 | .6161 | .5923 |
| eval_pgot_e10_r_raw_value | 17.668 | 12.039 | .5961 | .5920 |
| eval_pgot_e11_dual_m4 | 15.303 | 12.916 | .5905 | .5922 |
| eval_pgot_e11_capacity | 14.312 | 13.562 | .5819 | .5928 |
| eval_pgot_e11_capacity_dit16 | 14.102 | 13.802 | .5789 | .5925 |
| eval_pgot_e11_capacity_dit16_reader3 | 14.294 | 13.881 | .5764 | .5915 |
| eval_pgot_e11_capacity_query_separation | 14.564 | 13.662 | .5936 | .5858 |
| eval_pgot_e12_centroid_reader | 15.315 | 13.107 | .5861 | .5919 |
| eval_pgot_oneshot_pooled | 16.076 | 12.305 | .6115 | .5849 |
| eval_pgot_oneshot_owner_masked | 12.736 | 14.443 | .6115 | .5840 |
| eval_pgot_oneshot_owner_masked_15k | 11.420 | 15.180 | .5936 | .5884 |
| eval_pgot_oneshot_owner_masked_15k_ar_full | 11.709 | 15.043 | .5694 | .5703 |

pooled와 owner_masked 5k는 스크립트상 같은 E11 DiT16 checkpoint에서 시작하는 대응 비교다. 다만 변경점은 압축 유무 하나가 아니다: soft owner mixture 대 head별 hard owner 선택, raw content key 사용, within-owner attention까지 동시에 달라진다. E11과 one-shot은 Writer/ownership 추출 방식과 추가 학습량도 달라 직접 원인 분리가 안 된다. pooled 15k의 완료 평가 summary는 검토 시점에 없으며 학습 로그만 존재했다.

## 실제 정보 경로와 token 수

`pgot/model/visual_memory.py`의 `PGOTOneShotOwnerReader` 참조.

- E11 capacity/DiT16: object당 8 memories, register당 16 memories × 4 registers. 유효 visual memory 수는 `8K + 64`; K=3이면 88이다. 76은 `4K + 64` 설정에서의 수다. padded tensor 크기와 유효 memory 수는 다르다.
- pooled: owner patch distribution A와 query-to-owner attention G를 합성한다. 각 head에서 `context = (G A) V = G (A V)`. 따라서 object당 1개, register당 1개의 pooled visual representation이 **대수적으로 존재**한다. K=3이면 7 owner vectors다. 독립 Writer state가 없어졌어도 visual bottleneck은 남아 있다.
- owner_masked: 각 query/head가 semantic owner를 top-1로 선택한 뒤 해당 owner의 원본 raw SigLIP patches를 content attention으로 읽는다. 별도 고정 길이 object visual code는 reconstruction의 필수 경로에 없다. 반환되는 `visual_memory=valid_owner_values`는 진단용이고 reconstruction에서 사용하지 않는다(1133행 부근).
- owner_masked의 object representation을 가변 길이 patch set으로 정의할 수는 있다. 그러나 그것은 원본 patch 수에 비례하는 저장량을 갖는다. 고정 길이 object compression을 달성한 것은 아니다.
- 위 수치는 visual value token 예산이다. semantic states, query states, ownership/geometry 등 전달되는 부가 정보를 포함한 엄밀한 전체 압축률과 구분해야 한다. 채널 차원도 함께 보고해야 한다.

## E11 Reader의 구조적 제약: owner 안에서 visual content를 보고 memory를 고르지 못함

`visual_memory.py:825`에서 key는 `W_s LN(s_k) + e_j`이다. E11 DiT16 checkpoint에서는 centroid 기능이 꺼져 있다. 각 head의 첫 Reader attention logit은

`l(q,k,j) = q·W_s LN(s_k) + q·e_j = a(q,k) + b(q,j)`.

같은 유효 memory 개수를 가진 object들에서는

`P(j | q,k) = softmax_j b(q,j)`.

즉 **고정된 query/head에 대해 object 내부 memory-ID 선택 비율이 object identity/content와 독립**이다. object와 register의 유효 ID 집합(8 대 16) 차이는 정규화에 영향을 준다. 전체 owner 선택은 semantic state에 의존하지만, 같은 owner 내부에서는 해당 이미지에서 어떤 memory가 어떤 디테일을 담았는지 직접 key로 판별할 수 없다. 이 대수적 제약은 코드에서 확정할 수 있지만, rFID 14 정체의 주원인이라는 주장은 아직 ablation이 필요하다.

Writer 또한 단순 weighted sum만은 아니다. 320행 이후 raw value pooling에 normalization, nonlinear fuse, gate, 이전 memory refinement가 이어지고, 이전 memory가 다음 ownership/memory query에도 영향을 준다. 따라서 E11 전체와 one-shot 합성이 항상 동치인 것은 아니다.

E12 centroid 실험은 이미 존재하며 rFID가 개선되지 않았다. 그러므로 centroid 추가를 새로운 해결책처럼 반복 제안하지 않는다. **owner를 먼저 선택하고, 선택된 owner 안에서 content key로 읽는 계층적 Reader**가 여기서 제안하는 구별되는 변경이다.

## 기존 진단이 말해 주는 것

1. `analysis_e11_capacity_dit16_followup_512/summary.json`: object memory를 utilization 상위 4개만 남길 때 foreground diffusion MSE 증가는 .00865, 1개만 남길 때 .03318. register당 4개만 남길 때 background 증가는 .03916, 1개만 남길 때 .08424. 이는 inference-time memory-value ablation이며 해당 수로 재학습한 성능이나 rFID가 아니다. 무조건 object capacity만 늘릴 근거는 약하고 배경도 중요하다.
2. 같은 진단의 64 donor swaps에서 donor 유사도가 원본 복원보다 증가한 비율은 87.5%, 평균 cosine gain .0983이다. `donor_closer`는 swap이 target보다 donor에 더 가까워졌다는 뜻이 아니다. E11 object memories가 appearance 전달에 실제 기여한다는 증거다.
3. `analysis_e11_capacity_dit16_priority_visible_512/summary.json`: instance residual probe R²는 direct_self .2295, direct_other_mean .0270, direct_register_flat .0557. object-specific 정보가 존재한다. 단, 이는 제한된 probe의 설명력이며 총 정보량은 아니다.
4. `analysis_pgot_oneshot_{pooled,owner_masked}_512/summary.json`: owner-correct patch top1은 .8389/.8419, reader-correct object query top1은 .8110/.8140으로 비슷하다. 반면 same-object attention mass는 .6518/.7779, effective patch count는 417.4/64.8이다. 성능 개선은 더 선택적인 patch readout과 일관된다. effective count는 실제 저장 token 수가 아니다.
5. GT owner routing 교체는 pooled의 diffusion loss를 .6073→.6170, masked를 .5425→.5579로 오히려 높인다. 따라서 inference-only GT 대입을 routing 품질의 엄밀한 상한으로 해석하지 않는다. 훈련된 경로의 분포 변화가 섞인다.
6. `analysis_pgot_oneshot_owner_masked_swap64/summary.json`: donor appearance similarity가 증가한 비율 85.94%, 평균 gain .0965. 하지만 raw owner crop을 target region에 warp한 patch intervention이다. 압축 object latent swap의 증명은 아니다. selected-region change energy fraction 평균 .2645, median .1384이므로 “변화가 object 안에 완전히 국한된다”는 주장도 불가하다. E11 diffusion prediction localization ratio와 이 generated-image ratio는 직접 비교하지 않는다.
7. `eval_pgot_e11_capacity_latent_splice_{object,background}_gt/summary.json`: object latent를 GT로 교체하면 rFID 9.700, background를 교체하면 9.689. 양쪽 모두 개선 여지가 있다. 비선형 diffusion 경로의 splice 결과이므로 두 효과의 합산이나 원인 비중 분해는 하지 않는다.
8. `pooled_gt_oracle_e11_obj4_bg16/summary.json`의 높은 rFID 119.64/166.13은 E11 memory 예산의 불가능성 증거가 아니다. 이 실험은 **256 target latent**의 object당 4 codes + **배경 전체 16 codes**, 평균 총 27.76 codes를 복원 grid에 채워 frozen decoder에 직접 넣는 stress test다. E11의 1024 source patches→8K+64 memory→learned Reader/DiT와 예산·경로가 다르며 off-manifold 효과도 있다.

hard owner mask는 Reader의 patch-index 접근 제한이다. SigLIP feature 자체가 다른 영역의 정보를 이미 포함할 수 있으므로 pixel-level object independence를 보장하지 않는다. 기존 `probe_encoder_leakage_siglip_dino/summary.json`에도 encoder feature의 원거리 변화가 기록되어 있다.

## 다음 실험: 원인 분리부터

### A. 같은 E11 memory 예산에서 Reader key ablation

동일 E11 DiT16 checkpoint와 8K+64 capacity를 사용해 대조군을 같은 추가 step만큼 학습한다. frozen Writer로 시작해 Reader 변경의 영향을 분리한다.

- 대조군: 기존 semantic + memory-ID key.
- 변경군: `query -> semantic owner -> selected owner's memories`; owner 내부 key를 `W_k LN(m_kj)`로 만들어 content-dependent readout. value도 memory에서만 가져온다.
- owner hard gating 자체의 효과를 분리하려면 동일 계층 구조에서 ID-only inner key 대 content inner key를 비교한다.

이 실험은 반복 Writer를 유지하므로 최종 제안 모델이라기보다 **memory를 못 읽는 문제인지** 확인하는 진단이다. routing의 gradient 처리와 기존 owner supervision은 통제해야 한다.

### B. 최종 owner로 만드는 one-shot object visual tokenizer

`final semantic owners -> owner별 patch 묶음 -> query-independent object visual tokens -> owner-routed content Reader -> DiT`.

- 첫 baseline은 owner 영역을 좌표 기반으로 분할한 masked spatial pooling. 작은 object에는 빈 token을 억지로 배정하지 않는다.
- 이후 owner당 learned resampler queries로 소수의 token을 한 번 생성한다. output query마다 raw patch를 다시 읽지 않는다.
- 저장되는 object 표현은 `semantic state + visual tokens + 필요한 geometry`. Reader의 K/V는 저장된 tokens에서만 만든다. image input과 raw patches를 제거해도 저장된 표현으로 재복원할 수 있어야 한다.
- global visual budget은 우선 128/256 두 점에서 비교하고, 여유가 있으면 64를 추가한다. 배경에도 예산을 배정한다. token 차원과 geometry 비용을 함께 고정·기록한다.
- 처음부터 정교한 adaptive allocation을 추가하지 않는다. 고정 예산 baseline이 작동하면 예측 면적/feature variance에 따른 object별 배분을 같은 총 예산에서 검증한다.

attention resampler 자체는 선행 아이디어다. Flamingo의 Perceiver Resampler도 고정 개수 visual tokens를 만든다: https://arxiv.org/abs/2204.14198 . 연구 기여는 단순 모듈 채택보다 semantic ownership을 보존하는 압축, object별 교환 가능성, 동일 예산의 reconstruction/editing trade-off에서 입증해야 한다.

### C. owner_masked teacher의 활용

우선 128/256-token 학생이 동일 query의 teacher readout을 근사하도록 학습한다. `||C_student - stopgrad(C_teacher)||²`와 기존 reconstruction/owner loss를 사용하고, foreground/background를 분리 모니터링한다. raw-patch teacher 경로는 training target 계산에만 두며 student decoder의 입력에 더하지 않는다. 필요시 같은 noise/timestep의 diffusion prediction distillation을 추가한다.

teacher distillation은 원본 detail을 작은 예산에 모두 보존할 수 있다는 보장이 아니다. readout 근사는 좋은데 rFID가 나쁘면 decoder adaptation을, readout 근사부터 나쁘면 tokenizer/예산/geometry를 우선 의심할 수 있다.

### D. 필수 비교와 판단 기준

- owner_masked reference, same-owner compressed model, 동일 총 token 수의 owner-agnostic spatial tokenizer를 비교한다. owner 구조 자체의 기여를 확인한다.
- 같은 초기화/추가 학습량/데이터/추론 설정을 맞춘다. 512개는 pilot 진단에 쓰고 최종 rFID는 동일 4,720개에서 평가한다. 100-image AR rFID와 4,720-image rFID를 비교하지 않는다.
- rFID, PSNR, fARI/mIoU 외에 foreground/background error, visual token 수·차원·부가 geometry, latency를 기록한다.
- latent만 저장한 재복원, same-category visual-token swap, donor similarity gain, area-normalized inside/outside change와 total inside energy fraction을 같이 본다. semantic/layout은 고정하고 appearance code만 교체한다.
- raw patch zeroing은 강한 분포 밖 intervention일 수 있다. 이것만으로 semantic object removal이나 latent disentanglement를 주장하지 않는다.
- content Reader가 같은 8K+64에서 개선되면 읽기 제약을 지지한다. 128/256 budget 증가가 필요하면 용량의 영향을 지지한다. 압축해도 개선이 없으면 one-shot latent 생성/훈련 objective를 재검토한다.

## 연구 목표의 선택

독립 저장·조합 가능한 object visual latent가 목표라면 owner_masked의 낮은 rFID만으로 완료했다고 할 수 없다. 반대로 object ownership을 통한 controllable readout이 목표라면 가변 patch-set 표현도 유효하지만, 압축 표현이라는 주장을 낮추고 routing의 필요성을 검증해야 한다.

추천은 전자다. owner_masked를 충분한 시각 정보가 있는 reference로 유지하고, 반복 Writer를 없애되 **한 번 생성해 저장하는 object visual token 집합**을 되살린다. 이는 교수님의 모듈 단순화 제안과 연구의 object representation 목표를 함께 만족시키는 검증 가능한 방향이다.

작은 token 수만으로 불가능성을 단정하지 않는 참고 사례로 TiTok이 있다(32-token image representation): https://arxiv.org/abs/2406.07550 . 학습·데이터·해상도·decoder가 다르므로 PGOT의 rFID 목표치나 object disentanglement의 증거로 사용하지 않는다.
