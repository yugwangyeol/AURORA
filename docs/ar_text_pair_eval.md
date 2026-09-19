# 동일한 생성 AR Text: DeOR / CTRL-O Class–Mask Pair 평가

## 실행 전 필수 조건

기본값은 **클래스명 + 생성 설명 전체**를 공식 CTRL-O LLM2Vec로 인코딩한다.
클래스명만 넣었던 이전 Figure 실행과 다르며, 설명을 자동으로 클래스명으로
축약하거나 다른 인코더로 대체하지 않는다.

공식 MNTP / unsupervised-SimCSE 어댑터는 공개되었지만 기반 모델
[`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)는
접근 승인이 필요한 gated 모델이다. 이 계정의 접근 승인을 받은 뒤 서버에서 로그인한다.
토큰을 채팅, 셸 명령 인자, 코드에 넣지 않는다.

```bash
cd /home/jovyan/PGOT
bash scripts/login_ctrlo_hf.sh
```

현재 conda 환경의 `hf` CLI는 패키지 버전 혼재로 실행 오류가 있어,
위 스크립트는 실제 import 가능한 `huggingface_hub.login` API를 사용한다.
로그인은 사용자가 직접 해야 하며, 단순 로그인만으로 계정의 모델 접근 승인이 생기는 것은 아니다.

권한과 프로토콜만 먼저 확인하려면:

```bash
GPU=0 PREFLIGHT_ONLY=1 \
  bash scripts/eval_deor_ctrlo_ar_text_pairs.sh
```

`Preflight OK`가 확인되어도 모든 가중치/실제 full-text encoder forward가 검증된 것은
아니다. 처음 실행이면 다음 full-description smoke를 권장한다:

```bash
GPU=0 SMOKE=1 \
  bash scripts/eval_deor_ctrlo_ar_text_pairs.sh
```

GPU는 작업 시점에 **비어 있는 장치**로 선택한다. 2026-09-18 서버 구성은
GPU 0의 45GB MIG 한 대이며, smoke 시작 시 약 44GiB가 비어 있었다.
9월 17일의 GPU 2 / 90GB MIG UUID는 현재 서버에서 사용할 수 없다.

## 전체 평가 및 접속 종료

tmux 안에서는 그대로 실행한다:

```bash
cd /home/jovyan/PGOT
GPU=0 bash scripts/eval_deor_ctrlo_ar_text_pairs.sh
```

`Ctrl+b` 후 `d`로 tmux에서 detach하면 서버 프로세스는 계속 실행된다.
tmux 밖에서 실행해야 한다면 아래처럼 nohup을 사용할 수 있다:

```bash
cd /home/jovyan/PGOT
mkdir -p outputs/eval_deor_ctrlo_ar_text_pairs_description
nohup env GPU=0 \
  bash scripts/eval_deor_ctrlo_ar_text_pairs.sh \
  >> outputs/eval_deor_ctrlo_ar_text_pairs_description/launch.log 2>&1 < /dev/null &
```

서버/컨테이너가 유지되는 동안 터미널 연결 종료로 인한 SIGHUP을 피하는 명령이다.
서버 정지나 프로세스 종료까지 방지하지는 않는다.

```bash
tail -f outputs/eval_deor_ctrlo_ar_text_pairs_description/run.log
```

중단됐다면 동일한 설정/OUTPUT_ROOT로 재실행한다. 완료한 이미지 예측과
Text embedding은 건너뛰고, 점수 집계는 예측 캐시에서 다시 생성한다.
다른 모델/설명/seed/프로토콜이면 새 OUTPUT_ROOT를 사용해야 한다.

## 기본 설정

- **DeOR checkpoint**: `checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000`.
  object memory 8, register memory 32 x 4, writer owner prior 없음. 새 7B 모델이 아니다.
- COCO: 공식 val2017 **5,000장**, 객체 없는 이미지 포함.
- VOC: `sets/val.txt` **1,449장**. VOC에는 평가용 5,000장 split이 없으므로 복제하지 않는다.
- 생성 Text 출처: 기존 전체 AR 결과
  `outputs/segeval_pgot_memory8_full_coco_voc_ar/{coco,voc}_ar/generated_captions.jsonl`.
  checkpoint / 전체 coverage / 중복 ID를 검증한다. GT 성능으로 이미지를 골라내지 않는다.
- RGB는 두 모델 모두 CODA식 PIL min-side resize + 512 center crop에서 시작한다.
  GT 이진 mask도 동일한 PIL crop에 nearest interpolation을 적용한다.
  기존 OpenCV instance cache를 재사용하는 것이 아니라 annotation에서 공통 GT를 만든다.
- DeOR는 저장된 image-only AR caption을 원형 그대로 재토큰화하여 native owner
  readout을 재계산한다. AR 생성은 다시 하지 않으며, 최초 평가 때의 토큰/mask 캐시를
  그대로 로드하는 것은 아니다. Registers는 unrestricted이고 diffusion/RFID/KID는 실행하지 않는다.
- CTRL-O는 공식 공개 checkpoint step 71000을 strict load한다. 해당 release config의
  training shards는 `vg_coco/train`이며 COCO-only 학습이라고 주장하지 않는다.
  공식 PIL 224 resize +
  ImageNet normalize 후 `object_decoder.masks_as_image`를 bilinear 512 upsample,
  **7-slot 전체 argmax**한다. 각 입력 Text의 대응 slot만 그 Text의 예측으로 채점한다.
  GT에 가장 잘 맞는 다른 slot을 고르지 않는다.
- CTRL-O 입력은 생성 category와 **전체 description**, 번호/스키마 토큰만 제거한다.
  첫 문장만 자르거나 GT 클래스명, GT 좌표, GT 개수, GT mask를 넣지 않는다.
  예: `<thing> Dog 1: A brown dog. It is sitting. <ovt>` → `Dog: A brown dog. It is sitting.`
- 공식 7 slots 중 최대 **6개 Text + 최소 1개 unbound other slot**을 사용한다.
  넘는 설명은 원래 순서대로 여러 pass로 처리하여 **모든 쌍을 포함**한다.
  이 방식은 7개 이상 slot을 가진 단일-pass CTRL-O 모델도, 논문 그대로의 실험도 아니다.
  각 chunk 사이에는 경쟁하는 Text 집합이 달라질 수 있음을 비교 시 밝혀야 한다.
- DeOR FP32 batch 1, CTRL-O FP32 batch 1, LLM2Vec BF16 embedding batch 4.
  모델들은 서로 다른 프로세스에서 순차 실행되어 GPU에 동시에 올리지 않는다.
- LLM2Vec는 공식 MNTP + unsupervised-SimCSE 어댑터를 그대로 사용한다.
  `max_length`와 `doc_max_length`를 기본 1024로 설정하고, 길이를 넘는 설명은
  자동 축약하지 않고 오류로 중단한다. 모델/길이 설정은 결과에 저장한다.
- COCO crowd/ignore 영역과 GT mask overlap은 ignore 처리한다.
  VOC instance/semantic 255는 ignore, GT instance의 class는 유효 semantic pixels의
  최빈값이다. ignore 제외 후 영역이 남는 **모든** GT instance가 Recall 분모다.

## 지표 정의 및 해석

주 지표는 **IoU ≥ 0.50**에서 Class–Mask Pair Precision / Recall / F1이다.
0.25, 0.75도 함께 보고한다. 수치는 JSON/CSV에 **0..1**로 저장한다.

Text에서 파싱한 generated class와 GT class가 같고 mask IoU가 임계값 이상이면
매칭 후보가 된다. 고정 lexical synonym normalization을 적용한다.
유효한 매칭 개수가 최대가 되도록 일대일 Hungarian matching하고, 같은 개수일 때
총 IoU가 큰 것을 선택한다. 매칭 성공은 TP, 나머지 생성 쌍은 FP,
나머지 GT instance는 FN이다. 배경 mask는 TP 후보/분모에 넣지 않는다.

- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2TP / (2TP + FP + FN)

클래스명 미지/파싱 실패, 빈 mask, 중복 쌍도 버리지 않는다.
명시적 STUFF는 thing-instance 평가에서 제외하지만, 형식이 망가진 OVT owner는
임의로 제거하지 않는다. OVT 없는 실제 trailing THING description도 빈 DeOR
mask를 가진 생성 쌍으로 포함한다. 빈 장면의 구조적 `<thing>` seed만은 객체가 아니다.
동일 Text를 CTRL-O에도 제공한다. GT가 없는데 생성한 쌍은 FP이고,
GT는 있는데 아무것도 생성하지 않았으면 모두 FN이다.
분모 0인 precision/recall은 임의로 1이 아닌 `null`로 저장한다.
주 점수는 전체 이미지에 대해 TP/FP/FN을 합산한 micro 값이다.

추가 진단: 같은 Text에 연결된 두 mask의 IoU, 같은 query가 두 모델에서 같은 GT
instance에 매칭되었는지/다른 instance에 매칭되었는지의 수.
GT 개수 구간 `0`, `1–3`, `4–6`, `7–10`, `11+`별 점수도 제공한다.

**주의:** Class–Mask Pair 지표는 class + 영역의 정확도를 측정한다. 색/속성/행동/
관계 등 생성 description의 모든 문장이 맞는지 검사하지 않는다. 같은 class의
두 객체 설명을 맞바꿔도 class 기반 Precision/Recall이 같을 수 있다.
추가 alignment 진단 역시 GT referring expression 정답을 대신하지 않는다.
따라서 이 결과만으로 “설명 전체의 의미가 정확히 grounding된다”고 주장하면 안 된다.
DeOR가 만든 표현을 두 모델이 쓰는 조건이라는 편향, 입력 조건/모델 구조 차이,
overflow multi-pass 조건을 밝혀야 한다. **Binding Hit 71.2%와 직접 비교하지 않는다.**

## 결과 파일

`outputs/eval_deor_ctrlo_ar_text_pairs_description/`:

- `summary.json`: 데이터셋별 / 객체 개수별 두 모델 점수, protocol, provenance.
- `comparison.csv`: 모델/데이터셋/IoU 기준별 P/R/F1 표.
- `per_image.jsonl`: 모든 생성 쌍, GT instance identity, TP/FP/FN 및 매칭 IoU.
- `manifest.json`: checkpoint, source/annotation SHA256, 실제 공유 Text와 원래 OVT index.
- `ours/`, `ctrlo/`: 재현 가능한 원본 hard prediction 캐시.
- `embeddings/`, `embedding_info.json`: 공유 Text SHA256별 공식 embedding 캐시 및 인코더 정보.
- `run.log`: 단계별 로그. 권한 부족 시 `preflight`에서 큰 모델을 올리기 전에 중단한다.

## 별도 category-only 조건 / 확인 내역

`TEXT_MODE=category`는 **별개 실험**으로만 지원한다. 이름은 생성 AR category이며
GT category가 아니다. 공식 배포된 category embedding을 쓴다. 전체 데이터의 일부
생성 이름/잘못된 Text는 이 asset에 없으므로 전체 category-only 실행도 preflight에서
중단될 수 있다. 누락 이름을 몰래 other로 바꾸거나 쌍을 제외하지 않는다.

2026-09-17 검증:

- 11 CPU 단위 테스트 통과: label compaction/OVT identity, 중복/미지 클래스,
  malformed/trailing Text, empty seed, TP/FP/FN, threshold 최대-cardinality matching,
  ignore pixels, overflow query identity, 공통 crop.
- 실제 GPU category-only smoke 통과: COCO 139(13쌍, 3 CTRL-O passes),
  COCO 450686(2쌍), VOC 2010_004165(2쌍), **17쌍 모두 포함**, 모델별 cache/summary/CSV 출력.
  smoke는 실행 확인용 고정 작은 subset이고 대표 성능이 아니다.
- 당시 default full-description preflight가 기반 Llama 모델 **HTTP 401**로 중단되었다.
  아래 2026-09-18 접근 승인 후 full-description smoke로 해결/검증했다.
  로컬 모델은 `LLM2VEC_BASE` / `LLM2VEC_ADAPTER`로 설정하고 새 OUTPUT_ROOT를 사용한다.
  base 경로는 공식 MNTP와 동일한 가중치/토크나이저를 가진 모델이어야 하며,
  임의 Llama/Qwen 원본이나 다른 embedding 모델로 바꾸면 동일한 CTRL-O 조건이 아니다.

2026-09-18 추가 검증:

- 접근 승인 후 공식 LLM2Vec 전체 설명 인코딩 및 두 모델의 실제 GPU 추론 통과.
  현재 GPU 0인 B200 MIG 45GB에서 메모리 부족 오류 없이 실행했다.
- COCO 2장 + VOC 1장, 생성 쌍 17개 모두 유지했다. 고유 Text 15개의
  4096차원 embedding이 유한한 값인지 확인했고, CTRL-O 총 5 passes를 검증했다.
- DeOR object memory 8 / register memory 32 / noprior 원본 checkpoint 및
  공식 CTRL-O checkpoint strict loading(global step 71000)을 확인했다.
- 단위/Binding 관련 회귀 테스트 총 24개 통과. 이미지별 TP+FP=생성 쌍 수,
  TP+FN=GT 수를 검증하고 summary JSON / comparison CSV 저장을 확인했다.
- 결과: `outputs/smoke_deor_ctrlo_ar_text_pairs_description/`.
  smoke는 실행 확인용이며 대표 성능이 아니다. 전체 COCO 5,000장 + VOC 1,449장
  평가는 아직 실행하지 않았다. 처음 받은 Llama 가중치는 이후 실행에서 캐시를 재사용한다.
