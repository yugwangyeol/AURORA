# AURORA v2: Object-Centric Visual Decomposition via Attention Bottleneck in MLLMs

---

## 1. Overview

### 1.1 한 줄 요약

Frozen MLLM (Scale-RAE)에 소수의 learnable token(cmd, object prompt, register)과 attention mask만 추가하여,
이미지를 **K개의 object-centric representation + register(잔여 정보)**로 분해하고,
이 분해된 정보만으로 이미지를 reconstruct하는 method.

### 1.2 핵심 원리: Information Bottleneck

이미지 생성을 담당하는 `rae_query`가 원본 이미지 패치를 **직접 볼 수 없고**,
**반드시 object prompt + register를 경유**해야 한다.
이 병목(bottleneck) 때문에:

1. 각 object prompt에 의미 있는 object 정보가 담길 수밖에 없음
2. K개의 prompt가 서로 다른 영역을 담당하게 됨 (중복 시 register 부담 증가 → reconstruction 악화)
3. Register는 object prompt가 커버하지 못한 잔여 정보(다른 objects + background)를 흡수

### 1.3 Random K의 핵심 가치

매 training iteration마다 K를 랜덤하게 다르게 줌:
- 이미지에 object가 7개인데 K=3이면 → 3개 object만 명시적 분해, 나머지 4개 + background는 register로
- K=7이면 → 7개 전부 명시적 분해, background만 register로
- 이를 통해 **과분할(over-segmentation) 없이** 유연한 granularity의 decomposition 학습
- **Additive Compositionality**: K를 바꿔도 총 정보량(obj slots + register)은 일정해야 하므로,
  representation의 compositionality가 암묵적으로 강제됨

---

## 2. Architecture

### 2.1 전체 구조도

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Input Sequence (매 iteration K가 다름)        │
│                                                                      │
│  [img₁...img₂₅₆]  [cmd₁...cmd_C]  [obj₁...objₖ]  [reg₁...reg_R]  [rae₁...rae₂₅₆]  │
│      (256)            (C)             (K)            (R)             (256)            │
│                                                                      │
│  img: SigLIP2 → mm_projector 출력 (frozen)                          │
│  cmd: learnable embeddings (trainable) — task instruction            │
│  obj: learnable embedding pool에서 K개 선택 (trainable)              │
│  reg: learnable embeddings (trainable) — 잔여 정보 흡수              │
│  rae: 기존 Scale-RAE의 latent_queries (frozen)                       │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │   Frozen LLM       │
                     │   (Qwen2 1.5B)     │
                     │                    │
                     │  Custom Attention   │
                     │  Mask 적용 (§2.4)  │
                     └────────────────────┘
                                │
                                ▼
               ┌────────────────┼────────────────┐
               │                │                │
               ▼                ▼                ▼
         obj hidden        reg hidden       rae hidden
         states            states           states
         (K × d)           (R × d)          (256 × d)
               │                                 │
               ▼                                 ▼
         Attention Map                    DiT Diffusion Head
         추출 (§2.6)                     (AdaLN만 fine-tune)
               │                                 │
               ▼                                 ▼
         K개의 Object Masks             Reconstructed Image
         (K × 256)
```

### 2.2 각 Token 역할

| Token | 개수 | 역할 | 학습 여부 |
|-------|------|------|----------|
| `img` | 256 (16×16) | SigLIP2로 인코딩된 image patch features | **frozen** (SigLIP2 + mm_projector 출력) |
| `cmd` | C (e.g., 8) | Frozen LLM에게 "object decomposition task"를 수행하라는 context 제공. Frozen backbone이 학습하지 못한 새로운 행동 패턴을 cmd가 유도 | **trainable** |
| `obj_k` | K (매 iteration 랜덤, 1 ≤ K ≤ min(N_obj, K_max)) | 각각 하나의 object 정보를 인코딩하는 slot. obj_k의 LLM output hidden state가 곧 해당 object의 representation | **trainable** (pool에서 K개 선택) |
| `reg` | R (e.g., 8) | Object prompt가 커버하지 못한 잔여 정보 흡수 (나머지 objects + background + texture 등) | **trainable** |
| `rae_query` | 256 | DiT에 넘길 reconstruction용 query. **obj+reg만 볼 수 있음** (Information Bottleneck) | **frozen** (기존 Scale-RAE의 학습된 latent_queries를 가져옴) |

### 2.3 Frozen / Trainable 분류

```
┌─────────────────────────────────────────────────────────────────┐
│ Frozen (gradient 차단)                                          │
│   - SigLIP2 vision encoder                                      │
│   - mm_projector (MLP 2x GELU)                                 │
│   - Qwen2 1.5B LLM backbone (전체)                              │
│   - DiT body (attention layers, FFN 등)                         │
│   - rae_query (기존 Scale-RAE의 학습된 latent_queries)           │
├─────────────────────────────────────────────────────────────────┤
│ Trainable                                                       │
│   - cmd_embeddings:  nn.Parameter(C, d)         ~12K params     │
│   - obj_embedding_pool: nn.Parameter(K_max, d)  ~15K params     │
│   - reg_embeddings:  nn.Parameter(R, d)         ~12K params     │
│   - DiT AdaLN modulation layers                ~수M params      │
│     (y_embedder + 각 block의 adaLN_modulation)                  │
├─────────────────────────────────────────────────────────────────┤
│ 총 trainable: ~수M params (전체 모델의 <1%)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 Attention Mask 설계 (핵심)

#### Position Ranges (K개 object prompt가 들어간 경우)

```
positions [0, 255]                    → img tokens
positions [256, 256+C-1]              → cmd tokens
positions [256+C, 256+C+K-1]          → object prompt tokens (K개만)
positions [256+C+K, 256+C+K+R-1]      → register tokens
positions [256+C+K+R, 256+C+K+R+255]  → rae_query tokens
```

#### Attention Rules (K=3 예시)

```
              │  img(256)  │  cmd(C)  │  obj₁  │  obj₂  │  obj₃  │  reg(R)  │  rae(256)  │
──────────────┼────────────┼──────────┼────────┼────────┼────────┼──────────┼────────────┤
img(256)      │     BI     │    BI    │   ✗    │   ✗    │   ✗    │    ✗     │     ✗      │
cmd(C)        │     BI     │    BI    │   ✗    │   ✗    │   ✗    │    ✗     │     ✗      │
obj₁          │     ✓      │    ✓    │   ✓    │   ✗    │   ✗    │    ✗     │     ✗      │
obj₂          │     ✓      │    ✓    │   ✓    │   ✓    │   ✗    │    ✗     │     ✗      │
obj₃          │     ✓      │    ✓    │   ✓    │   ✓    │   ✓    │    ✗     │     ✗      │
reg(R)        │     ✓      │    ✓    │   ✓    │   ✓    │   ✓    │    BI    │     ✗      │
rae(256)      │     ✗      │    ✗    │   ✓    │   ✓    │   ✓    │    ✓     │     BI     │
```

- `BI` = 양방향 (bidirectional, 서로 attend 가능)
- `✓` = attend 가능 (row가 column을 봄)
- `✗` = attend 불가 (-inf bias)

#### 각 규칙의 이유

| # | 규칙 | 설명 |
|---|------|------|
| 1 | img ↔ img: BI | 이미지 패치끼리 양방향 self-attention → 풍부한 visual feature |
| 2 | img ↔ cmd: BI | cmd가 이미지 global context를 흡수 + img가 task context를 받음 |
| 3 | img/cmd → obj,reg,rae: ✗ | img와 cmd의 hidden state는 decomposition의 영향을 받지 않아야 함 ("순수한 이미지 정보" 유지) |
| 4 | obj_k → img, cmd: ✓ | object prompt가 이미지+지시를 읽어서 object 정보 추출 |
| 5 | obj_k → obj_{<k}: ✓ (causal) | 후순위 prompt가 선순위 hidden state를 봄 → "이미 인코딩된 영역" 정보 전달 → 다른 영역 인코딩 유도 |
| 6 | reg → img,cmd,obj: ✓ | register는 모든 선행 정보를 보고 "아직 커버 안 된 것"을 흡수 |
| 7 | reg ↔ reg: BI | register끼리 정보 교환하여 잔여 정보 분산 |
| 8 | **rae → img,cmd: ✗** | **Information Bottleneck (핵심!)** — rae_query가 원본 이미지를 직접 못 봄 |
| 9 | rae → obj,reg: ✓ | rae_query는 분해된 정보(obj+reg)만으로 reconstruction해야 함 |
| 10 | rae ↔ rae: BI | rae_query끼리 정보 교환하여 reconstruction 준비 |

#### Object Prompt 간 분화가 일어나는 이유 (별도 모듈 없이)

명시적 suppression mask나 competition module 없이, **causal attention + reconstruction bottleneck**만으로
각 object prompt가 서로 다른 object를 인코딩하게 되는 논리:

1. obj₁이 frozen LLM의 attention을 통해 특정 img patches에 강하게 attend → hidden state에 해당 object 정보 인코딩
2. obj₂가 img + obj₁에 attend → obj₁의 hidden state를 통해 "이미 인코딩된 정보" 간접 인식
3. Reconstruction loss: 만약 obj₁과 obj₂가 같은 object를 중복 인코딩하면 → register가 더 많은 정보를 혼자 감당 → register capacity 부족 → reconstruction 악화 → gradient가 중복 해소 방향으로 작용
4. 정보 보존 등식: `I(image) = I(obj₁) + I(obj₂) + ... + I(objₖ) + I(register)` — 중복 없이 분할해야 reconstruction 최적

따라서 explicit diversity loss는 **optional accelerator**일 뿐, 필수가 아님.

### 2.5 Object Prompt Embedding Pool

K_max개(e.g., 10)의 서로 다른 learned embedding을 미리 준비:

```python
self.obj_embedding_pool = nn.Parameter(torch.randn(K_max, d) * embed_std)
# e.g., shape: (10, 1536)
```

매 iteration에서 K개만 선택하여 시퀀스에 삽입:

```python
K = sample_k(...)  # e.g., K=3
obj_embeds = self.obj_embedding_pool[:K]  # (3, d) — 앞에서 K개 선택
# 나머지 7개(obj₄~obj₁₀)는 이 iteration에서 아예 사용되지 않음 (시퀀스에 포함 안 됨)
```

**서로 다른 초기화를 사용하는 이유:**
- 동일 초기화 시 초기 몇 step에서 모든 obj prompt가 동일한 attention pattern → 분화가 느림
- 서로 다른 초기 embedding은 처음부터 다른 방향의 attention 유도 → 빠른 분화
- DETR의 learned object queries도 동일한 원리로 서로 다른 초기화 사용

**항상 앞에서부터 K개를 선택하는 이유:**
- Causal ordering에서 obj₁이 항상 "첫 번째로 보는" prompt
- obj₁은 모든 K 값에서 학습됨 (가장 안정적), objₖ_max는 K=K_max일 때만 학습됨
- 자연스러운 saliency 기반 계층: obj₁ → 가장 salient한 object, obj₂ → 다음, ...

### 2.6 Attention Map 추출

LLM forward pass 후, 각 object prompt의 "어디를 보고 있는가"를 측정.
추가 파라미터 없이, output hidden state 간 dot-product로 계산:

```python
# LLM output에서 추출
h_obj_k = lm_output[:, obj_k_position, :]       # (B, d)
H_img   = lm_output[:, img_start:img_end, :]    # (B, 256, d)
```

여기서:
- `h_obj_k`: obj_k 위치에서의 **LLM output hidden state** — 모든 layer를 거쳐 최종적으로 인코딩된 representation
- `H_img`: 이미지 패치 위치들의 **LLM output hidden states** — img ↔ img bidirectional attention으로 context가 반영된 고수준 features

```python
# Attention map 계산
logits = torch.einsum('bd,bnd->bn', h_obj_k, H_img) / math.sqrt(d)  # (B, 256)
attn_map_k = torch.sigmoid(logits)  # (B, 256)
```

**sigmoid를 사용하는 이유:**
- Softmax: 합이 1 → 작은 object도 전체 이미지에 분산 → 부적절
- Sigmoid: 각 patch가 독립적으로 0~1 → object 크기에 무관하게 해당 영역만 활성화
- GT mask도 binary (0/1)이므로 sigmoid + BCE가 자연스러움

**왜 LLM 내부 attention이 아닌 output dot-product인가:**
- 내부 attention: 28개 layer × 다수 head → "어떤 layer/head를 쓸지" 추가 hyperparameter 필요
- Output dot-product: 모든 layer의 정보가 종합된 최종 representation 사용 → 더 안정적
- 구현 간단 (hook 불필요), gradient 경로도 깔끔
- 내부 attention 추출은 추후 ablation 및 시각화에 활용 가능

---

## 3. Training

### 3.1 Stage 1: Object-Centric Decomposition 학습

#### 데이터
- **COCO train2017**: ~118K images (`/home/jovyan/data/coco/train2017`)
- **Instance annotations**: `/home/jovyan/data/coco/annotations/instances_train2017.json`
  - 각 이미지별 object별 segmentation mask → 16×16 patch grid로 변환하여 GT mask supervision에 사용
- **Precomputed DINOv2 features** (선택적): `/home/jovyan/processed_coco/training_data_v4_patch_dino/`

#### Random K Sampling

```python
def sample_k_for_batch(n_objects_list: list[int], k_max: int) -> int:
    """배치 내 모든 sample에 동일한 K를 적용.
    
    Args:
        n_objects_list: 배치 내 각 sample의 GT object 수 리스트
        k_max: object prompt pool 크기 (e.g., 10)
    Returns:
        K: 이 배치에서 사용할 object prompt 수
    """
    # 배치 내 최소 object 수를 upper bound로 사용
    min_n_objects = min(n_objects_list)
    upper = min(min_n_objects, k_max)
    if upper < 1:
        return 1  # 최소 1개
    return random.randint(1, upper)
```

- **같은 배치 내에서는 K가 동일** → sequence length 통일, padding 불필요
- **배치마다 K가 다름** → 다양한 decomposition granularity 학습
- K ~ Uniform(1, min(batch 내 최소 N_obj, K_max))

#### GT Mask Assignment — Hungarian Matching

```python
from scipy.optimize import linear_sum_assignment

def hungarian_match(pred_maps: torch.Tensor, gt_masks: torch.Tensor) -> list[tuple[int, int]]:
    """예측된 attention map과 GT mask 간 최적 1:1 매칭.
    
    Args:
        pred_maps: (K, 256) — K개 object prompt의 predicted attention maps
        gt_masks:  (N, 256) — N개 GT instance masks (N >= K)
    Returns:
        matches: list of (pred_idx, gt_idx) pairs, 길이 K
    """
    K, N = pred_maps.shape[0], gt_masks.shape[0]
    
    # Cost matrix: (K, N) — BCE loss 기반
    cost = torch.zeros(K, N)
    for i in range(K):
        for j in range(N):
            cost[i, j] = F.binary_cross_entropy(
                pred_maps[i], gt_masks[j], reduction='mean'
            )
    
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    return list(zip(row_ind.tolist(), col_ind.tolist()))
```

- N > K인 경우: K개의 prompt에 K개의 GT만 매칭, 나머지 N-K개 GT object 정보는 register가 흡수
- GT mask는 COCO instance segmentation을 16×16 grid로 변환 (기존 코드의 `build_gt_patch_masks` 활용)
- GT mask는 area 순 정렬 (큰 object부터)

#### Loss Functions

```
L_total = L_reconstruction + λ_mask × L_mask + λ_div × L_diversity
```

**L_reconstruction — Diffusion Reconstruction Loss:**
```python
# rae_query의 LLM output hidden states → DiT
rae_hidden = lm_output[:, rae_start:rae_end, :]  # (B, 256, d)

# 기존 Scale-RAE의 diffusion training loss (DiT)
# rae_hidden이 DiT의 conditioning이 되어 원본 이미지를 reconstruct
# target: 원본 이미지의 SigLIP features
target_features = img_features.detach()
L_reconstruction = dit_head.training_loss(z=rae_hidden, x=target_features)
```

**L_mask — Mask Supervision Loss (Full supervision):**
```python
def compute_mask_loss(lm_output, obj_positions, img_start, img_end, gt_masks, matching):
    """
    Args:
        lm_output: (B, L, d) — frozen LLM output
        obj_positions: list of int — K개 obj prompt의 position indices
        img_start, img_end: img patch 위치 범위 (0, 256)
        gt_masks: (B, N, 256) — GT instance masks (16x16 grid)
        matching: list of (pred_idx, gt_idx) — Hungarian matching 결과
    """
    d = lm_output.shape[-1]
    H_img = lm_output[:, img_start:img_end, :]  # (B, 256, d)
    
    total_loss = 0.0
    for pred_idx, gt_idx in matching:
        h_obj = lm_output[:, obj_positions[pred_idx], :]  # (B, d)
        
        logits = torch.einsum('bd,bnd->bn', h_obj, H_img) / math.sqrt(d)
        pred_map = torch.sigmoid(logits)        # (B, 256)
        gt_map = gt_masks[:, gt_idx, :]          # (B, 256)
        
        total_loss += F.binary_cross_entropy(pred_map, gt_map, reduction='mean')
    
    return total_loss / max(len(matching), 1)
```

**L_diversity — Object Prompt Diversity Loss (optional accelerator):**
```python
def compute_diversity_loss(lm_output, obj_positions):
    """Object prompt hidden states 간 cosine similarity 패널티.
    
    중복 인코딩 방지를 가속하는 보조 loss.
    Reconstruction bottleneck만으로도 분화가 일어나지만, 이 loss가 수렴을 빠르게 함.
    """
    obj_hidden = lm_output[:, obj_positions, :]  # (B, K, d)
    normed = F.normalize(obj_hidden, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)
    
    K = len(obj_positions)
    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    off_diag = (sim * (1 - eye)).sum() / max(K * (K - 1) * obj_hidden.shape[0], 1)
    
    return off_diag
```

**Loss 계수:**
- `λ_mask = 1.0`
- `λ_div = 0.1`

### 3.2 Stage 2: Object-Level Editing 학습 (Inpainting)

#### 추가 데이터
- **Inpainting 데이터**: `/home/jovyan/processed_coco/training_data_v4_patch/`
  - Manifest: `training_manifest_patch_v4_0_to_None.json`
  - 각 sample: `{source_path, target_path, mask_path}`
    - `source_path`: 원본 이미지 (object 있는 상태)
    - `target_path`: 목표 이미지 (object 제거 후)
    - `mask_path`: 제거할 영역의 mask

#### Stage 2 학습 방식

Stage 1에서 학습된 모든 파라미터를 유지하면서:
1. Source image로 forward → K개 object slot 생성
2. 제거할 object에 해당하는 slot을 식별 (mask_path와 attention map의 IoU로 매칭)
3. 해당 slot을 **제거**하고 나머지 slots + register로만 rae_query를 conditioning
4. DiT가 target image (object 없는 이미지)를 reconstruct하도록 학습

추가로 학습 가능한 파라미터:
- Stage 1의 모든 trainable params (계속 학습)
- 선택적으로 rae_query를 unfreeze할 수 있음 (ablation)
- 선택적으로 mm_projector를 unfreeze할 수 있음 (ablation)

---

## 4. Forward Pass 전체 흐름

```python
def forward_aurora_v2(self, images, gt_masks, n_objects):
    B = images.shape[0]
    
    # ============ 1. Image Encoding (Frozen) ============
    with torch.no_grad():
        img_features = self.vision_encoder(images)           # (B, 256, d_vis)
        img_embeds = self.mm_projector(img_features)          # (B, 256, d)
    
    # ============ 2. Random K Sampling (배치 단위) ============
    n_objects_list = n_objects.tolist()  # 배치 내 각 sample의 GT object 수
    K = sample_k_for_batch(n_objects_list, self.k_max)
    
    # ============ 3. Build Input Sequence ============
    cmd_embeds = self.cmd_embeddings.unsqueeze(0).expand(B, -1, -1)           # (B, C, d)
    obj_embeds = self.obj_embedding_pool[:K].unsqueeze(0).expand(B, -1, -1)   # (B, K, d)
    reg_embeds = self.reg_embeddings.unsqueeze(0).expand(B, -1, -1)           # (B, R, d)
    rae_embeds = self.rae_query.unsqueeze(0).expand(B, -1, -1)                # (B, 256, d) — frozen
    
    inputs_embeds = torch.cat([
        img_embeds,    # [0, 256)
        cmd_embeds,    # [256, 256+C)
        obj_embeds,    # [256+C, 256+C+K)
        reg_embeds,    # [256+C+K, 256+C+K+R)
        rae_embeds,    # [256+C+K+R, 256+C+K+R+256)
    ], dim=1)          # (B, 256+C+K+R+256, d)
    
    # ============ 4. Build Attention Mask ============
    attn_bias = build_aurora_v2_attention_mask(
        n_img=256, n_cmd=self.C, n_obj=K, n_reg=self.R, n_rae=256,
        device=inputs_embeds.device
    )  # (1, 1, L, L)
    
    # ============ 5. Frozen LLM Forward ============
    lm_output = self.frozen_llm(
        inputs_embeds=inputs_embeds,
        attention_bias=attn_bias,
    ).last_hidden_state  # (B, L, d)
    
    # ============ 6. Position Indices ============
    img_s, img_e = 0, 256
    cmd_s = 256
    obj_s, obj_e = 256 + self.C, 256 + self.C + K
    reg_s, reg_e = obj_e, obj_e + self.R
    rae_s, rae_e = reg_e, reg_e + 256
    
    obj_positions = list(range(obj_s, obj_e))  # K개 위치
    
    # ============ 7. Attention Map 추출 ============
    H_img = lm_output[:, img_s:img_e, :]  # (B, 256, d)
    d = H_img.shape[-1]
    
    pred_maps = []
    for k_idx in range(K):
        h_obj = lm_output[:, obj_positions[k_idx], :]  # (B, d)
        logits = torch.einsum('bd,bnd->bn', h_obj, H_img) / math.sqrt(d)
        pred_maps.append(torch.sigmoid(logits))  # (B, 256)
    pred_maps = torch.stack(pred_maps, dim=1)  # (B, K, 256)
    
    # ============ 8. Hungarian Matching (per sample) ============
    all_matchings = []
    for b in range(B):
        n_obj_b = min(int(n_objects[b].item()), gt_masks.shape[1])
        matching_b = hungarian_match(
            pred_maps[b, :K],         # (K, 256)
            gt_masks[b, :n_obj_b],    # (N_b, 256)
        )
        all_matchings.append(matching_b)
    
    # ============ 9. Losses ============
    
    # 9a. Reconstruction Loss
    rae_hidden = lm_output[:, rae_s:rae_e, :]  # (B, 256, d)
    target_features = img_features.detach()
    L_recon = self.dit_head.training_loss(z=rae_hidden, x=target_features)
    
    # 9b. Mask Supervision Loss
    L_mask = 0.0
    for b in range(B):
        for pred_idx, gt_idx in all_matchings[b]:
            h_obj = lm_output[b:b+1, obj_positions[pred_idx], :]  # (1, d)
            logits = torch.einsum('bd,bnd->bn', h_obj, H_img[b:b+1]) / math.sqrt(d)
            pred = torch.sigmoid(logits)
            gt = gt_masks[b:b+1, gt_idx, :]
            L_mask += F.binary_cross_entropy(pred, gt, reduction='mean')
    L_mask = L_mask / max(sum(len(m) for m in all_matchings), 1)
    
    # 9c. Diversity Loss (optional)
    if K > 1:
        L_div = compute_diversity_loss(lm_output, obj_positions)
    else:
        L_div = torch.tensor(0.0, device=lm_output.device)
    
    # 9d. Total
    L_total = L_recon + 1.0 * L_mask + 0.1 * L_div
    
    return L_total, pred_maps, all_matchings
```

---

## 5. Attention Mask 구현

```python
def build_aurora_v2_attention_mask(n_img, n_cmd, n_obj, n_reg, n_rae, device):
    """AURORA v2의 custom attention mask 생성.
    
    Args:
        n_img: img patch 수 (256)
        n_cmd: cmd token 수 (e.g., 8)
        n_obj: 이 iteration의 active object prompt 수 (K)
        n_reg: register token 수 (e.g., 8)
        n_rae: rae_query 수 (256)
    
    Returns:
        (1, 1, L, L) additive attention bias.
        0 = attend 가능, -inf = attend 불가.
        Bidirectional 영역에서 causal mask를 상쇄하려면 large positive value 사용.
    """
    L = n_img + n_cmd + n_obj + n_reg + n_rae
    NEG_INF = float('-inf')
    
    bias = torch.full((L, L), NEG_INF, device=device)
    
    # Segment boundaries
    img_s, img_e = 0, n_img
    cmd_s, cmd_e = n_img, n_img + n_cmd
    obj_s, obj_e = cmd_e, cmd_e + n_obj
    reg_s, reg_e = obj_e, obj_e + n_reg
    rae_s, rae_e = reg_e, reg_e + n_rae
    
    # (1) img ↔ img: bidirectional
    bias[img_s:img_e, img_s:img_e] = 0
    
    # (2) img ↔ cmd: bidirectional
    bias[img_s:img_e, cmd_s:cmd_e] = 0
    bias[cmd_s:cmd_e, img_s:img_e] = 0
    bias[cmd_s:cmd_e, cmd_s:cmd_e] = 0
    
    # (3) obj → img, cmd: can attend
    bias[obj_s:obj_e, img_s:img_e] = 0
    bias[obj_s:obj_e, cmd_s:cmd_e] = 0
    
    # (4) obj → obj: causal (lower triangle including diagonal)
    for i in range(n_obj):
        for j in range(i + 1):
            bias[obj_s + i, obj_s + j] = 0
    
    # (5) reg → img, cmd, obj: can attend
    bias[reg_s:reg_e, img_s:img_e] = 0
    bias[reg_s:reg_e, cmd_s:cmd_e] = 0
    bias[reg_s:reg_e, obj_s:obj_e] = 0
    # (6) reg ↔ reg: bidirectional
    bias[reg_s:reg_e, reg_s:reg_e] = 0
    
    # (7) rae → obj, reg ONLY (NOT img, NOT cmd) — Information Bottleneck!
    bias[rae_s:rae_e, obj_s:obj_e] = 0
    bias[rae_s:rae_e, reg_s:reg_e] = 0
    # (8) rae ↔ rae: bidirectional
    bias[rae_s:rae_e, rae_s:rae_e] = 0
    
    return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
```

**XLA SPMD 환경 참고:**
기존 Scale-RAE의 `build_block_attention_bias`처럼, causal mask와 결합 시
bidirectional 영역에는 large positive value로 upper-triangle -inf를 상쇄:

```python
try:
    from torch_xla.experimental.custom_kernel import FlashAttention
    large_positive = -float(FlashAttention.DEFAULT_MASK_VALUE)
except Exception:
    large_positive = 0.7 * float(torch.finfo(torch.float32).max)
```

---

## 6. Inference (재학습 없이)

### 6.1 기본 Reconstruction

```python
# 사용자가 K 지정 (e.g., K=5)
sequence = [img(256)] + [cmd(C)] + [obj₁...obj₅] + [reg(R)] + [rae(256)]
→ frozen LLM → DiT → reconstructed image
```

### 6.2 Object Removal

```python
# Step 1: K=5로 forward → 5개 object의 attention map 추출
pred_maps = forward(image, K=5)  # 5개 attention maps

# Step 2: 제거할 object의 attention map과 user-provided mask를 비교
#          가장 IoU가 높은 slot 선택 (e.g., obj₃)

# Step 3: obj₃를 제거하고 K=4로 re-forward
sequence = [img(256)] + [cmd(C)] + [obj₁][obj₂][obj₄][obj₅] + [reg(R)] + [rae(256)]
→ frozen LLM → DiT → object가 제거된 이미지
```

### 6.3 Object Transfer (Compositional Editing)

```python
# Image A에서 forward → obj₂가 "강아지"를 인코딩
h_dog = lm_output_A[:, obj₂_position, :]  # 강아지 representation

# Image B에서 forward → obj₁ 자리에 h_dog를 injection
# (LLM output level에서 hidden state를 교체)
lm_output_B[:, obj₁_position, :] = h_dog  # 강아지 정보 주입
rae_hidden = lm_output_B[:, rae_start:rae_end, :]
→ DiT → Image B 배경 + Image A 강아지
```

---

## 7. Resolution 전략

### 7.1 현재: 16×16 (256 patches)

- SigLIP2 (patch14-224): 224×224 → 16×16 = 256 patches
- 장점: 빠름, LLM sequence 짧음
- 단점: 작은 object의 mask가 뭉개질 수 있음

### 7.2 확장 계획: 32×32 (1024 patches)

- SigLIP2에 448×448 이미지 입력 → 448/14 = 32 → 32×32 = 1024 patches
- 시퀀스 길이: 1024 + C + K + R + 1024 ≈ 2070 (Qwen2 max 32K 이내)
- rae_query도 1024개로 확장 필요 (또는 256 유지 후 upsample)
- FlashAttention 사용 시 메모리 관리 가능

### 7.3 권장 접근

1. **16×16 (256)로 먼저 proof-of-concept** → 아이디어 검증, 빠른 iteration
2. **검증 후 32×32 (1024)로 scale-up** → 세밀한 mask, 더 나은 segmentation quality
3. 64×64 (4096)은 시퀀스 길이 문제로 비추천

---

## 8. 실험 계획

### 8.1 Main Results

| 실험 | 평가 지표 | 의미 |
|------|----------|------|
| Reconstruction quality | FID, LPIPS (COCO val) | Bottleneck이 정보를 보존하는가 |
| Object segmentation | AP, mIoU (attention map → mask) | Object decomposition 품질 |
| Object removal | FID of inpainted images | Slot 단위 편집 가능성 |

### 8.2 핵심 Ablation

| # | 실험 | 변수 | 확인 사항 |
|---|------|------|----------|
| A1 | Bottleneck 제거 | rae가 img도 볼 수 있게 | Decomposition 붕괴 여부 |
| A2 | K 고정 vs 랜덤 | K=N_obj(고정) vs random K | Random K의 이점 |
| A3 | Causal vs Bidirectional | obj prompt 간 mask 변경 | 순서 정보의 중요성 |
| A4 | cmd 유무 | cmd 토큰 제거 | cmd의 task context 기여 |
| A5 | Register 수 | R=0,4,8,16 | 잔여 정보 용량의 영향 |
| A6 | Mask supervision 강도 | λ_mask=0, 0.1, 1.0 | Supervision 없이도 분해 가능한가 |
| A7 | Diversity loss 유무 | λ_div=0 vs 0.1 | Reconstruction bottleneck만으로 충분한가 |

### 8.3 Additive Compositionality 검증

고정된 이미지에서:
- K=1: reconstruction quality = Q₁ (1개 분해 + 나머지 register)
- K=3: reconstruction quality = Q₃ (3개 분해 + 나머지 register)
- K=7: reconstruction quality = Q₇ (7개 분해 + 나머지 register)
- Q₁ ≈ Q₃ ≈ Q₇이면 → 정보 보존 성공, K에 무관하게 총 정보량 동일

---

## 9. 기존 연구 대비 차별점

| | Slot Attention | DETR | AURORA v1 (이전) | **AURORA v2 (제안)** |
|---|---|---|---|---|
| LLM 활용 | ✗ | ✗ | ✓ (일부 학습) | ✓ (**frozen**) |
| 추가 모듈 | Slot module | Decoder head | SlotHead + AR loop | **Learnable tokens만** |
| 분해 메커니즘 | Softmax 경쟁 (iterative) | Cross-attention | 명시적 suppression mask | **Reconstruction bottleneck** |
| K 유연성 | 고정 | 고정 (100) | AR로 가변 | **Random K** |
| 이미지 생성 | ✗ | ✗ | ✓ (복잡) | ✓ |
| 과분할 문제 | 있음 (고정 K) | 있음 | AR로 회피 | **Random K로 회피** |
| 복잡도 | 별도 모듈+iteration | 별도 decoder | 복잡한 AR loop | **Simple (mask+tokens)** |

---

## 10. 구현 체크리스트

- [ ] `nn.Parameter`로 `cmd_embeddings(C,d)`, `obj_embedding_pool(K_max,d)`, `reg_embeddings(R,d)` 정의
- [ ] 기존 Scale-RAE의 학습된 `latent_queries`를 `rae_query`로 로드 후 freeze
- [ ] `build_aurora_v2_attention_mask()` 구현 (K에 따라 동적 생성)
- [ ] LLM forward에 attention bias 전달 경로 구현 (기존 `build_block_attention_bias` 패턴 활용)
- [ ] DiT의 AdaLN layer만 `requires_grad=True`로 설정
- [ ] `sample_k_for_batch()` — 배치 단위 random K sampling
- [ ] Attention map 추출: output dot-product 방식
- [ ] `hungarian_match()` 구현 (scipy.optimize.linear_sum_assignment)
- [ ] GT mask 전처리: COCO annotation → 16×16 patch mask (기존 `build_gt_patch_masks` 활용)
- [ ] `L_reconstruction`, `L_mask`, `L_diversity` 구현
- [ ] Stage 1 training loop (COCO reconstruction + mask supervision)
- [ ] Stage 2 training loop (inpainting 데이터 추가)
- [ ] Inference: reconstruction, object removal, compositional editing
- [ ] 32×32 해상도 확장 (Stage 1 검증 후)
