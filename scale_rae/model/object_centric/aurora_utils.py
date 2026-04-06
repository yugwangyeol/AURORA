"""
AURORA v2 utility functions — attention masks, random K sampling,
Hungarian matching, and loss functions.
"""

import math
import random
from typing import List, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# Attention Mask
# ──────────────────────────────────────────────────────────────────

def build_aurora_v2_attention_mask(
    n_img: int,
    n_cmd: int,
    n_obj: int,
    n_reg: int,
    n_rae: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build the AURORA v2 custom attention mask.

    Token layout: [img | cmd | obj₁…objₖ | reg | rae_query]

    Returns (1, 1, L, L) additive attention bias.
    0 = attend, -inf = blocked.
    """
    L = n_img + n_cmd + n_obj + n_reg + n_rae
    NEG_INF = float("-inf")

    bias = torch.full((L, L), NEG_INF, device=device, dtype=dtype)

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
    if n_obj > 0:
        obj_indices = torch.arange(n_obj, device=device)
        causal = obj_indices.unsqueeze(0) <= obj_indices.unsqueeze(1)  # (n_obj, n_obj)
        bias[obj_s:obj_e, obj_s:obj_e] = torch.where(
            causal,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(NEG_INF, device=device, dtype=dtype),
        )

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


# ──────────────────────────────────────────────────────────────────
# Random K Sampling
# ──────────────────────────────────────────────────────────────────

def sample_k_for_batch(n_objects_list: List[int], k_max: int) -> int:
    """Sample a single K for the entire batch.

    K ~ Uniform(1, min(batch_min_n_objects, k_max)).
    Same K for all samples in the batch → no padding needed.
    """
    min_n_objects = min(n_objects_list) if n_objects_list else 1
    upper = min(min_n_objects, k_max)
    if upper < 1:
        return 1
    return random.randint(1, upper)


# ──────────────────────────────────────────────────────────────────
# Hungarian Matching
# ──────────────────────────────────────────────────────────────────

def hungarian_match(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
) -> List[Tuple[int, int]]:
    """Optimal 1:1 matching between predicted attention maps and GT masks.

    Args:
        pred_logits: (K, P) — K predicted raw attention logits.
        gt_masks:  (N, P) — N GT binary masks (N >= K).

    Returns:
        List of (pred_idx, gt_idx) pairs, length K.
    """
    from scipy.optimize import linear_sum_assignment

    K, N = pred_logits.shape[0], gt_masks.shape[0]
    if K == 0:
        return []

    # Build cost matrix using BCE on raw logits to stay autocast-safe.
    pred_logits = pred_logits.detach().float()
    gt_float = gt_masks.detach().float()

    cost = torch.zeros(K, N, device=pred_logits.device, dtype=torch.float32)
    for i in range(K):
        for j in range(N):
            cost[i, j] = F.binary_cross_entropy_with_logits(
                pred_logits[i], gt_float[j], reduction="mean"
            )

    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
    return list(zip(row_ind.tolist(), col_ind.tolist()))


# ──────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────

def compute_mask_loss(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    all_matchings: List[List[Tuple[int, int]]],
) -> torch.Tensor:
    """Mask supervision loss computed directly on raw attention logits.

    Args:
        pred_logits:    (B, K, P) — raw attention logits before sigmoid.
        gt_masks:        (B, N, P) — GT instance masks (16×16 grid).
        all_matchings:   Per-sample list of (pred_idx, gt_idx) from Hungarian matching.

    Returns:
        Scalar BCE loss averaged over all matched pairs.
    """
    B = pred_logits.shape[0]

    total_loss = torch.tensor(0.0, device=pred_logits.device, dtype=torch.float32)
    n_pairs = 0

    for b in range(B):
        for pred_idx, gt_idx in all_matchings[b]:
            logits = pred_logits[b, pred_idx].float()
            gt_map = gt_masks[b, gt_idx, :]  # (P,)
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                logits, gt_map.float(), reduction="mean"
            )
            n_pairs += 1

    return total_loss / max(n_pairs, 1)


def compute_diversity_loss(
    lm_output: torch.Tensor,
    obj_positions: List[int],
) -> torch.Tensor:
    """Object prompt diversity loss — cosine similarity penalty.

    Penalises off-diagonal cosine similarity between object prompt
    hidden states to accelerate differentiation.
    """
    K = len(obj_positions)
    if K <= 1:
        return torch.tensor(0.0, device=lm_output.device, dtype=torch.float32)

    obj_hidden = lm_output[:, obj_positions, :]  # (B, K, d)
    normed = F.normalize(obj_hidden.float(), dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))  # (B, K, K)

    eye = torch.eye(K, device=sim.device).unsqueeze(0)
    off_diag = (sim * (1 - eye)).sum() / (K * (K - 1) * obj_hidden.shape[0])

    return off_diag


def extract_attention_logits(
    lm_output: torch.Tensor,
    obj_positions: List[int],
    img_start: int,
    img_end: int,
) -> torch.Tensor:
    """Extract raw attention logits from LLM output via dot-product.

    Returns:
        (B, K, P) raw attention logits.
    """
    d = lm_output.shape[-1]
    H_img = lm_output[:, img_start:img_end, :]  # (B, P, d)
    K = len(obj_positions)
    if K == 0:
        return H_img.new_zeros((lm_output.shape[0], 0, H_img.shape[1]), dtype=torch.float32)

    logits = []
    for k in range(K):
        h_obj = lm_output[:, obj_positions[k], :]  # (B, d)
        logits.append(torch.einsum("bd,bnd->bn", h_obj.float(), H_img.float()) / math.sqrt(d))

    return torch.stack(logits, dim=1)  # (B, K, P)


def extract_attention_maps(
    lm_output: torch.Tensor,
    obj_positions: List[int],
    img_start: int,
    img_end: int,
) -> torch.Tensor:
    """Extract attention maps from LLM output via dot-product + sigmoid.

    Returns:
        (B, K, P) sigmoid attention maps.
    """
    return torch.sigmoid(
        extract_attention_logits(lm_output, obj_positions, img_start, img_end)
    )
