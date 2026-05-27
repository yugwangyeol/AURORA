"""
AURORA v2 utility functions — attention masks, random K sampling,
Hungarian matching, and loss functions.
"""

import math
import random
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# Attention Mask
# ──────────────────────────────────────────────────────────────────

def build_active_slot_mask(
    n_obj: int,
    device: torch.device,
    active_k_per_sample: Optional[List[int]] = None,
    active_slot_mask: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Build a boolean [B, K] mask for active object slots."""
    if active_slot_mask is not None:
        mask = active_slot_mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.shape[-1] != n_obj:
            raise ValueError(f"Expected active_slot_mask last dim {n_obj}, got {tuple(mask.shape)}")
        if batch_size is not None:
            if mask.shape[0] == 1 and batch_size > 1:
                mask = mask.expand(batch_size, -1)
            elif mask.shape[0] != batch_size:
                raise ValueError(
                    f"Expected active_slot_mask batch {batch_size}, got {mask.shape[0]}"
                )
        return mask

    if active_k_per_sample is None:
        batch_size = 1 if batch_size is None else batch_size
        return torch.ones(batch_size, n_obj, device=device, dtype=torch.bool)

    mask = torch.zeros(len(active_k_per_sample), n_obj, device=device, dtype=torch.bool)
    for b, k_i in enumerate(active_k_per_sample):
        if k_i > 0:
            mask[b, :min(int(k_i), n_obj)] = True
    return mask


def _build_visible_img_mask(
    n_img: int,
    device: torch.device,
    visible_img_mask: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Build a boolean [B, P] mask for image tokens visible to slot discovery."""
    if visible_img_mask is None:
        batch_size = 1 if batch_size is None else batch_size
        return torch.ones(batch_size, n_img, device=device, dtype=torch.bool)

    mask = visible_img_mask.to(device=device, dtype=torch.bool)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[-1] != n_img:
        raise ValueError(f"Expected visible_img_mask last dim {n_img}, got {tuple(mask.shape)}")
    if batch_size is not None:
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1)
        elif mask.shape[0] != batch_size:
            raise ValueError(
                f"Expected visible_img_mask batch {batch_size}, got {mask.shape[0]}"
            )
    return mask

def build_aurora_v2_attention_mask(
    n_img: int,
    n_cmd: int,
    n_obj: int,
    n_reg: int,
    n_rae: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    active_k_per_sample: List[int] = None,
    active_slot_mask: Optional[torch.Tensor] = None,
    visible_img_mask: Optional[torch.Tensor] = None,
    n_rae_anchor: int = 0,
) -> torch.Tensor:
    """Build the AURORA v2 custom attention mask.

    Token layout: [cmd | img | obj₁…obj_K_max | reg | im_start_anchor | rae_query]

    When active_k_per_sample/active_slot_mask is provided, builds per-sample masks
    (B, 1, L, L) where inactive obj slots are fully blocked in all directions.

    When visible_img_mask is provided, only visible image patches participate in
    the attention graph. Hidden image patches are fully blocked so visible patches
    cannot indirectly leak their information through image-image attention.

    When active_k_per_sample is None, builds a shared mask (1, 1, L, L) where
    all n_obj slots are active (backward-compatible).

    Returns additive attention bias. 0 = attend, -inf = blocked.
    """
    L = n_img + n_cmd + n_obj + n_reg + n_rae_anchor + n_rae
    NEG_INF = float("-inf")

    # Segment boundaries
    cmd_s, cmd_e = 0, n_cmd
    img_s, img_e = cmd_e, cmd_e + n_img
    obj_s, obj_e = img_e, img_e + n_obj
    reg_s, reg_e = obj_e, obj_e + n_reg
    anchor_s, anchor_e = reg_e, reg_e + n_rae_anchor
    rae_s, rae_e = anchor_e, anchor_e + n_rae

    if active_k_per_sample is None and active_slot_mask is None and visible_img_mask is None:
        # ── Shared mask (original behavior) ──
        bias = torch.full((L, L), NEG_INF, device=device, dtype=dtype)

        # (1) img <-> img: bidirectional
        bias[img_s:img_e, img_s:img_e] = 0
        # (2) img <-> cmd: bidirectional
        bias[img_s:img_e, cmd_s:cmd_e] = 0
        bias[cmd_s:cmd_e, img_s:img_e] = 0
        bias[cmd_s:cmd_e, cmd_s:cmd_e] = 0
        # (3) obj -> img, cmd
        bias[obj_s:obj_e, img_s:img_e] = 0
        bias[obj_s:obj_e, cmd_s:cmd_e] = 0
        # (4) obj -> obj: causal
        if n_obj > 0:
            obj_idx = torch.arange(n_obj, device=device)
            causal = obj_idx.unsqueeze(0) <= obj_idx.unsqueeze(1)
            bias[obj_s:obj_e, obj_s:obj_e] = torch.where(
                causal,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(NEG_INF, device=device, dtype=dtype),
            )
        # (5) reg -> img, cmd, obj; reg <-> reg
        bias[reg_s:reg_e, img_s:img_e] = 0
        bias[reg_s:reg_e, cmd_s:cmd_e] = 0
        bias[reg_s:reg_e, obj_s:obj_e] = 0
        bias[reg_s:reg_e, reg_s:reg_e] = 0
        # (6) im_start anchor stays content-free and only carries a fixed modality prior.
        if n_rae_anchor > 0:
            bias[anchor_s:anchor_e, anchor_s:anchor_e] = 0
        # (7) rae -> obj, reg, anchor; rae <-> rae (Information Bottleneck!)
        bias[rae_s:rae_e, obj_s:obj_e] = 0
        bias[rae_s:rae_e, reg_s:reg_e] = 0
        if n_rae_anchor > 0:
            bias[rae_s:rae_e, anchor_s:anchor_e] = 0
        bias[rae_s:rae_e, rae_s:rae_e] = 0

        return bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

    # ── Per-sample mask ──
    if active_slot_mask is not None:
        batch_size = active_slot_mask.shape[0] if active_slot_mask.dim() > 1 else 1
    elif visible_img_mask is not None:
        batch_size = visible_img_mask.shape[0] if visible_img_mask.dim() > 1 else 1
    elif active_k_per_sample is not None:
        batch_size = len(active_k_per_sample)
    else:
        batch_size = 1

    active_mask = build_active_slot_mask(
        n_obj=n_obj,
        device=device,
        active_k_per_sample=active_k_per_sample,
        active_slot_mask=active_slot_mask,
        batch_size=batch_size,
    )
    B = active_mask.shape[0]
    visible_mask = _build_visible_img_mask(
        n_img=n_img,
        device=device,
        visible_img_mask=visible_img_mask,
        batch_size=B,
    )
    bias = torch.full((B, L, L), NEG_INF, device=device, dtype=dtype)
    bias[:, cmd_s:cmd_e, cmd_s:cmd_e] = 0
    bias[:, reg_s:reg_e, cmd_s:cmd_e] = 0
    bias[:, reg_s:reg_e, reg_s:reg_e] = 0
    if n_rae_anchor > 0:
        bias[:, anchor_s:anchor_e, anchor_s:anchor_e] = 0
        bias[:, rae_s:rae_e, anchor_s:anchor_e] = 0
    bias[:, rae_s:rae_e, reg_s:reg_e] = 0
    bias[:, rae_s:rae_e, rae_s:rae_e] = 0

    for b in range(B):
        img_idx = torch.nonzero(visible_mask[b], as_tuple=False).flatten()
        obj_idx = torch.nonzero(active_mask[b], as_tuple=False).flatten()

        if img_idx.numel() > 0:
            img_rows = img_s + img_idx
            bias[b, img_rows[:, None], img_rows] = 0
            bias[b, img_rows[:, None], cmd_s:cmd_e] = 0
            bias[b, cmd_s:cmd_e, img_rows] = 0
            bias[b, reg_s:reg_e, img_rows] = 0

        if obj_idx.numel() > 0:
            obj_rows = obj_s + obj_idx
            bias[b, obj_rows[:, None], cmd_s:cmd_e] = 0
            if img_idx.numel() > 0:
                img_cols = img_s + img_idx
                bias[b, obj_rows[:, None], img_cols] = 0

            causal = obj_idx.unsqueeze(1) >= obj_idx.unsqueeze(0)
            obj_bias = torch.full(
                (obj_idx.numel(), obj_idx.numel()),
                NEG_INF,
                device=device,
                dtype=dtype,
            )
            obj_bias[causal] = 0.0
            bias[b, obj_rows[:, None], obj_rows] = obj_bias

            bias[b, reg_s:reg_e, obj_rows] = 0
            bias[b, rae_s:rae_e, obj_rows] = 0

    return bias.unsqueeze(1)  # (B, 1, L, L)


# ──────────────────────────────────────────────────────────────────
# Random K Sampling
# ──────────────────────────────────────────────────────────────────

def sample_k_for_batch(n_objects_list: List[int], k_max: int) -> int:
    """Sample a single K for the entire batch (legacy).

    K ~ Uniform(1, min(batch_min_n_objects, k_max)).
    Same K for all samples in the batch.
    """
    min_n_objects = min(n_objects_list) if n_objects_list else 1
    upper = min(min_n_objects, k_max)
    if upper < 1:
        return 1
    return random.randint(1, upper)


def sample_k_per_sample(n_objects_list: List[int], k_max: int) -> List[int]:
    """Sample per-sample K_i values.

    For each sample: K_i ~ Uniform(1, min(n_objects_i, k_max)).
    Different samples can have different active slot counts.
    """
    result = []
    for n_obj in n_objects_list:
        upper = min(n_obj, k_max)
        if upper < 1:
            result.append(1)
        else:
            result.append(random.randint(1, upper))
    return result


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
        pred_logits:    (B, K_max, P) — raw attention logits (inactive slots ignored via matchings).
        gt_masks:        (B, N, P) — GT instance masks (16x16 grid).
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
    pred_maps: torch.Tensor,
    active_k_per_sample: List[int] = None,
    active_slot_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Object slot diversity loss from attention-map overlap.

    Args:
        pred_maps: (B, K_max, P) — sigmoid attention maps in patch space.
        active_k_per_sample: optional prefix-active counts.
        active_slot_mask: optional explicit [B, K_max] active mask.
    """
    B, K_max, _ = pred_maps.shape
    if K_max <= 1:
        return torch.tensor(0.0, device=pred_maps.device, dtype=torch.float32)

    slot_mask = build_active_slot_mask(
        n_obj=K_max,
        device=pred_maps.device,
        active_k_per_sample=active_k_per_sample,
        active_slot_mask=active_slot_mask,
        batch_size=B,
    )
    pred_maps = pred_maps.float().clamp(0.0, 1.0)

    total = torch.tensor(0.0, device=pred_maps.device, dtype=torch.float32)
    n_pairs_total = 0

    for b in range(B):
        active_idx = torch.nonzero(slot_mask[b], as_tuple=False).flatten()
        if active_idx.numel() <= 1:
            continue
        maps = pred_maps[b, active_idx]  # (k_i, P)
        intersection = maps @ maps.T
        mass = maps.sum(dim=-1, keepdim=True)
        union = mass + mass.T - intersection
        overlap = intersection / union.clamp_min(1e-6)
        eye = torch.eye(active_idx.numel(), device=overlap.device, dtype=torch.bool)
        off_diag_sum = overlap.masked_select(~eye).sum()
        n_pairs = active_idx.numel() * (active_idx.numel() - 1)
        total = total + off_diag_sum
        n_pairs_total += n_pairs

    return total / max(n_pairs_total, 1)


def extract_attention_logits(
    lm_output: torch.Tensor,
    obj_positions: List[int],
    img_start: int,
    img_end: int,
    normalize_tokens: bool = False,
    temperature: float = 1.0,
    active_k_per_sample: List[int] = None,
    active_slot_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract raw attention logits from LLM output via dot-product.

    Returns:
        (B, K_max, P) raw attention logits.
        Inactive slots (index >= K_i) are filled with 0.
    """
    d = lm_output.shape[-1]
    H_img = lm_output[:, img_start:img_end, :].float()  # (B, P, d)
    if normalize_tokens:
        H_img = F.layer_norm(H_img, (d,))
    K = len(obj_positions)
    if K == 0:
        return H_img.new_zeros((lm_output.shape[0], 0, H_img.shape[1]), dtype=torch.float32)

    temp = max(float(temperature), 1e-6)
    logits = []
    for k in range(K):
        h_obj = lm_output[:, obj_positions[k], :].float()  # (B, d)
        if normalize_tokens:
            h_obj = F.layer_norm(h_obj, (d,))
        logits.append(torch.einsum("bd,bnd->bn", h_obj, H_img) / (math.sqrt(d) * temp))

    result = torch.stack(logits, dim=1)  # (B, K_max, P)

    # Zero out inactive slots so they don't affect downstream losses
    if active_k_per_sample is not None or active_slot_mask is not None:
        slot_mask = build_active_slot_mask(
            n_obj=K,
            device=result.device,
            active_k_per_sample=active_k_per_sample,
            active_slot_mask=active_slot_mask,
            batch_size=result.shape[0],
        )
        result = result.masked_fill(~slot_mask.unsqueeze(-1), 0.0)

    return result


def extract_attention_maps(
    lm_output: torch.Tensor,
    obj_positions: List[int],
    img_start: int,
    img_end: int,
    normalize_tokens: bool = False,
    temperature: float = 1.0,
    active_k_per_sample: List[int] = None,
    active_slot_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract attention maps from LLM output via dot-product + sigmoid.

    Returns:
        (B, K_max, P) sigmoid attention maps.
    """
    maps = torch.sigmoid(
        extract_attention_logits(
            lm_output,
            obj_positions,
            img_start,
            img_end,
            normalize_tokens=normalize_tokens,
            temperature=temperature,
            active_k_per_sample=active_k_per_sample,
            active_slot_mask=active_slot_mask,
        )
    )
    if active_k_per_sample is not None or active_slot_mask is not None:
        slot_mask = build_active_slot_mask(
            n_obj=maps.shape[1],
            device=maps.device,
            active_k_per_sample=active_k_per_sample,
            active_slot_mask=active_slot_mask,
            batch_size=maps.shape[0],
        )
        maps = maps * slot_mask.unsqueeze(-1).to(dtype=maps.dtype)
    return maps
