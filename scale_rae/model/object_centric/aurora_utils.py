"""
AURORA utility functions — attention masks, stopping criteria, and losses.
"""

import math
import torch
import torch.nn.functional as F
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────
# Attention Masks
# ──────────────────────────────────────────────────────────────────

def build_bidirectional_mask(length: int, device: torch.device) -> torch.Tensor:
    """Fully bidirectional mask for Phase 1 (img ↔ cmd)."""
    return torch.zeros(1, 1, length, length, device=device)


def build_phase3_mask(
    n_reg: int,
    n_rae: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Phase 3 mask for [register(n_reg) | rae_query(n_rae)].
    - register ↔ register: bidirectional
    - rae_query ↔ rae_query: bidirectional
    - rae_query → register: can attend
    - register → rae_query: blocked
    """
    L = n_reg + n_rae
    m = torch.full((1, 1, L, L), float("-inf"), device=device)
    m[:, :, :n_reg, :n_reg] = 0.0
    m[:, :, n_reg:, :n_reg] = 0.0
    m[:, :, n_reg:, n_reg:] = 0.0
    return m


def build_phase3_cache_attention_bias(
    prefix_len: int,
    n_reg: int,
    n_rae: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Additive attention bias for Phase 3 when a KV-cache prefix already exists.
    """
    current_len = n_reg + n_rae
    local_mask = build_phase3_mask(n_reg=n_reg, n_rae=n_rae, device=device).to(dtype=dtype)
    bias = torch.zeros(
        batch_size,
        1,
        current_len,
        prefix_len + current_len,
        device=device,
        dtype=dtype,
    )
    bias[:, :, :, prefix_len:] = local_mask.expand(batch_size, -1, -1, -1)
    return bias


def build_full_attention_mask(
    n_img: int,
    n_cmd: int,
    n_slots: int,
    n_reg: int,
    n_rae: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Single-forward mask for editing / inpainting.
    [img(n_img) | cmd(n_cmd) | slots(n_slots) | register(n_reg) | rae_query(n_rae)]
    """
    base = n_img + n_cmd
    L = base + n_slots + n_reg + n_rae
    m = torch.full((1, 1, L, L), float("-inf"), device=device)

    m[:, :, :base, :base] = 0.0

    ss = base
    for i in range(n_slots):
        idx = ss + i
        m[:, :, idx, :base] = 0.0
        m[:, :, idx, ss : idx + 1] = 0.0

    rs = ss + n_slots
    m[:, :, rs : rs + n_reg, :rs] = 0.0
    m[:, :, rs : rs + n_reg, rs : rs + n_reg] = 0.0

    qs = rs + n_reg
    m[:, :, qs:, :qs] = 0.0
    m[:, :, qs:, qs:] = 0.0

    return m


# ──────────────────────────────────────────────────────────────────
# Stopping Criteria
# ──────────────────────────────────────────────────────────────────

def check_stopping(
    alpha_t: torch.Tensor,
    slot_t: torch.Tensor,
    prev_slot: Optional[torch.Tensor],
    n_patches: int = 256,
    entropy_threshold: float = 0.92,
    similarity_threshold: float = 0.95,
) -> torch.Tensor:
    """
    Dual stopping criterion.
    Returns (B,) bool — True means stop for that sample.
    """
    entropy = -(alpha_t * torch.log(alpha_t + 1e-8)).sum(dim=-1)
    max_ent = math.log(n_patches)
    norm_entropy = entropy / max_ent
    entropy_stop = norm_entropy > entropy_threshold

    if prev_slot is not None:
        s_cur = slot_t.squeeze(1) if slot_t.dim() == 3 else slot_t
        s_prev = prev_slot.squeeze(1) if prev_slot.dim() == 3 else prev_slot
        sim = F.cosine_similarity(s_cur, s_prev, dim=-1)
        sim_stop = sim > similarity_threshold
    else:
        sim_stop = torch.zeros_like(entropy_stop)

    return entropy_stop | sim_stop


# ──────────────────────────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────────────────────────

def compute_diversity_loss(
    slots: List[torch.Tensor],
    n_objects: torch.Tensor,
) -> torch.Tensor:
    """
    Off-diagonal cosine-similarity penalty between valid slot pairs.
    slots: list of (B, 1, D) tensors.
    n_objects: (B,) int — number of valid objects per sample.
    """
    if len(slots) == 0:
        return n_objects.new_zeros((), dtype=torch.float32)
    if len(slots) == 1:
        return slots[0].new_zeros(())

    all_slots = torch.cat(slots, dim=1)                 # (B, K, D)
    B, K, D = all_slots.shape
    device = all_slots.device

    normed = F.normalize(all_slots, dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))     # (B, K, K)
    eye = torch.eye(K, device=device).unsqueeze(0)

    valid = torch.zeros(B, K, dtype=torch.bool, device=device)
    for b in range(B):
        k = min(int(n_objects[b].item()), K)
        valid[b, :k] = True

    pair = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair = pair & ~eye.bool()

    n_pairs = pair.sum().clamp(min=1)
    return (sim * pair.float()).sum() / n_pairs


def match_slot_to_mask(
    attn_maps: List[torch.Tensor],
    inpaint_mask: torch.Tensor,
    n_objects: torch.Tensor,
) -> torch.Tensor:
    """
    Match each sample's inpainting mask to the slot with highest IoU.
    Returns (B,) int — index of the best-matching slot per sample.
    """
    B = inpaint_mask.shape[0]
    K = len(attn_maps)
    device = inpaint_mask.device
    best_idx = torch.zeros(B, dtype=torch.long, device=device)

    for b in range(B):
        best_iou = -1.0
        mask_b = inpaint_mask[b]
        for t in range(min(K, int(n_objects[b].item()))):
            alpha_b = attn_maps[t][b]
            intersection = (alpha_b * mask_b).sum()
            union = alpha_b.sum() + mask_b.sum() - intersection + 1e-8
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_idx[b] = t

    return best_idx
