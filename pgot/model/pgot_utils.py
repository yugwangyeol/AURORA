"""PGOT model utilities.

Differences from AURORA captionslot:
1. No fixed slot-embedding block. Object tokens (`<ovt>`) live INSIDE the caption
   at variable positions.
2. rae_query is **blocked from attending image tokens** — information must flow
   image -> caption/ovt -> rae_query.
3. Variable K per sample handled via per-sample ovt_positions + ovt_valid_mask.
"""

from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sequence layout
# ---------------------------------------------------------------------------
def pgot_positions(
    caption_len: int,
    system_prefix_len: int,
    system_suffix_len: int,
    user_prefix_len: int,
    user_suffix_len: int,
    assistant_prefix_len: int,
    assistant_suffix_len: int,
    num_image_tokens: int,
    n_register: int,
    n_rae_query: int,
    n_null_bg: int = 0,
) -> Dict[str, int]:
    """Compute absolute positions for every block.

    Layout (after concat):
        [ sys_prefix | sys_suffix
        | user_prefix | image | user_suffix
        | assistant_prefix | caption(with <ovt>) | assistant_suffix
        | null_bg | register | rae_query ]

    Note: no CMD block, no separate slot block, no im_start/im_end. The OVT
    tokens are inside `caption`; their per-sample offsets are tracked externally
    in `ovt_positions_in_caption`.
    """
    cursor = 0

    sys_s = cursor
    cursor += system_prefix_len + system_suffix_len
    sys_e = cursor

    user_prefix_s = cursor
    cursor += user_prefix_len
    user_prefix_e = cursor

    img_s = cursor
    cursor += num_image_tokens
    img_e = cursor

    user_suffix_s = cursor
    cursor += user_suffix_len
    user_suffix_e = cursor

    assistant_prefix_s = cursor
    cursor += assistant_prefix_len
    assistant_prefix_e = cursor

    cap_s = cursor
    cursor += caption_len
    cap_e = cursor

    assistant_suffix_s = cursor
    cursor += assistant_suffix_len
    assistant_suffix_e = cursor

    null_bg_s = cursor
    cursor += n_null_bg
    null_bg_e = cursor

    reg_s = cursor
    cursor += n_register
    reg_e = cursor

    rae_s = cursor
    cursor += n_rae_query
    rae_e = cursor

    return {
        "sys_s": sys_s, "sys_e": sys_e,
        "user_prefix_s": user_prefix_s, "user_prefix_e": user_prefix_e,
        "img_s": img_s, "img_e": img_e,
        "user_suffix_s": user_suffix_s, "user_suffix_e": user_suffix_e,
        "assistant_prefix_s": assistant_prefix_s, "assistant_prefix_e": assistant_prefix_e,
        "cap_s": cap_s, "cap_e": cap_e,
        "assistant_suffix_s": assistant_suffix_s, "assistant_suffix_e": assistant_suffix_e,
        "null_bg_s": null_bg_s, "null_bg_e": null_bg_e,
        "reg_s": reg_s, "reg_e": reg_e,
        "rae_s": rae_s, "rae_e": rae_e,
        "total_len": rae_e,
    }


# ---------------------------------------------------------------------------
# Attention bias
# ---------------------------------------------------------------------------
def build_pgot_attention_mask(
    positions: Dict[str, int],
    caption_padding_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    rae_bidirectional: bool = False,
    rae_attends_caption: bool = False,
    ovt_absolute_positions: Optional[torch.Tensor] = None,
    ovt_valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build additive attention bias for PGOT.

    Rules:
      - image: bidirectional within image + sees sys prefix
      - caption tokens (incl. ovt): causal up to self + attends image bidirectionally
      - assistant_suffix: causal
      - register: attends image + valid caption + self
      - rae_query: attends OVT(if rae_attends_caption=False, via OVT_only_positions
                   passed by model) + register + self  (IMAGE + CAPTION BOTH BLOCKED).
                   `rae_attends_caption=True` restores the legacy mode (caption visible).
    """
    batch_size = int(caption_padding_mask.shape[0])
    total_len = int(positions["total_len"])
    neg_inf = float("-inf")

    bias = torch.full((batch_size, total_len, total_len), neg_inf, device=device, dtype=dtype)
    diag = torch.arange(total_len, device=device)
    bias[:, diag, diag] = 0.0

    def allow_cols(b_idx: int, row_idx: int, start: int, end: int) -> None:
        if end > start:
            bias[b_idx, row_idx, start:end] = 0.0

    def allow_rows_causal(start: int, end: int) -> None:
        for row_idx in range(start, end):
            bias[:, row_idx, : row_idx + 1] = 0.0

    sys_s, sys_e = positions["sys_s"], positions["sys_e"]
    user_prefix_s, user_prefix_e = positions["user_prefix_s"], positions["user_prefix_e"]
    img_s, img_e = positions["img_s"], positions["img_e"]
    user_suffix_s, user_suffix_e = positions["user_suffix_s"], positions["user_suffix_e"]
    assistant_prefix_s, assistant_prefix_e = positions["assistant_prefix_s"], positions["assistant_prefix_e"]
    cap_s, cap_e = positions["cap_s"], positions["cap_e"]
    assistant_suffix_s, assistant_suffix_e = positions["assistant_suffix_s"], positions["assistant_suffix_e"]
    null_bg_s, null_bg_e = positions.get("null_bg_s", 0), positions.get("null_bg_e", 0)
    reg_s, reg_e = positions["reg_s"], positions["reg_e"]
    rae_s, rae_e = positions["rae_s"], positions["rae_e"]

    # Causal text blocks
    allow_rows_causal(sys_s, sys_e)
    allow_rows_causal(user_prefix_s, user_prefix_e)
    allow_rows_causal(user_suffix_s, user_suffix_e)
    allow_rows_causal(assistant_prefix_s, assistant_prefix_e)
    allow_rows_causal(assistant_suffix_s, assistant_suffix_e)

    for b_idx in range(batch_size):
        valid_cap_idx = caption_padding_mask[b_idx].nonzero(as_tuple=False).flatten()
        valid_cap_positions = cap_s + valid_cap_idx

        # Image rows: see sys + image bidirectional
        for row_idx in range(img_s, img_e):
            allow_cols(b_idx, row_idx, sys_s, user_prefix_e)
            allow_cols(b_idx, row_idx, img_s, img_e)

        # Caption rows (including <ovt>): causal up to self + see full image
        for local_idx in range(cap_e - cap_s):
            row_idx = cap_s + local_idx
            if not bool(caption_padding_mask[b_idx, local_idx]):
                continue
            # Sees sys + user_prefix + image + user_suffix + assistant_prefix
            allow_cols(b_idx, row_idx, sys_s, assistant_prefix_e)
            # Causal over caption
            allow_cols(b_idx, row_idx, cap_s, row_idx + 1)

        # Assistant suffix: causal over previous text + caption
        for row_idx in range(assistant_suffix_s, assistant_suffix_e):
            allow_cols(b_idx, row_idx, sys_s, assistant_prefix_e)
            if valid_cap_positions.numel() > 0:
                bias[b_idx, row_idx, valid_cap_positions] = 0.0
            allow_cols(b_idx, row_idx, assistant_suffix_s, row_idx + 1)

        # Null-bg row: image + caption + self. It is a segmentation owner for
        # non-thing pixels, not a generic max-over-register background score.
        for row_idx in range(null_bg_s, null_bg_e):
            allow_cols(b_idx, row_idx, img_s, img_e)
            if valid_cap_positions.numel() > 0:
                bias[b_idx, row_idx, valid_cap_positions] = 0.0
            allow_cols(b_idx, row_idx, null_bg_s, null_bg_e)

        # Register rows: image + caption + self (DOES NOT see rae_query)
        for row_idx in range(reg_s, reg_e):
            allow_cols(b_idx, row_idx, img_s, img_e)
            if valid_cap_positions.numel() > 0:
                bias[b_idx, row_idx, valid_cap_positions] = 0.0
            allow_cols(b_idx, row_idx, null_bg_s, null_bg_e)
            allow_cols(b_idx, row_idx, reg_s, reg_e)

        # rae_query rows: OVT (positions inside caption) + register + self
        # IMAGE and full caption text are BLOCKED — only the OVT positions inside
        # the caption are visible. This forces OVT to be the unique bottleneck
        # for image reconstruction so that swapping an OVT directly steers DiT.
        if rae_attends_caption and valid_cap_positions.numel() > 0:
            # Legacy path (kept for ablation): rae attends entire valid caption.
            cap_target_positions = valid_cap_positions
        elif ovt_absolute_positions is not None and ovt_valid_mask is not None:
            ovt_pos_b = ovt_absolute_positions[b_idx]
            ovt_valid_b = ovt_valid_mask[b_idx]
            keep = ovt_valid_b.nonzero(as_tuple=False).flatten()
            cap_target_positions = ovt_pos_b[keep].to(torch.long) if keep.numel() > 0 else valid_cap_positions[:0]
        else:
            cap_target_positions = valid_cap_positions[:0]  # nothing from caption

        for row_idx in range(rae_s, rae_e):
            if cap_target_positions.numel() > 0:
                bias[b_idx, row_idx, cap_target_positions] = 0.0
            allow_cols(b_idx, row_idx, null_bg_s, null_bg_e)
            allow_cols(b_idx, row_idx, reg_s, reg_e)
            if rae_bidirectional:
                allow_cols(b_idx, row_idx, rae_s, rae_e)
            else:
                allow_cols(b_idx, row_idx, rae_s, row_idx + 1)

    return bias.unsqueeze(1)  # (B, 1, L, L)


# ---------------------------------------------------------------------------
# Extracting OVT hidden states
# ---------------------------------------------------------------------------
def gather_ovt_hidden_states(
    lm_hidden: torch.Tensor,
    ovt_absolute_positions: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Pull out the OVT hidden states from the full LLM sequence.

    Args:
        lm_hidden: (B, L, D) full LLM output.
        ovt_absolute_positions: (B, M_max) long, absolute positions in the
            global sequence (already shifted by cap_s).
        ovt_valid_mask: (B, M_max) bool, True for real OVT tokens.

    Returns:
        ovt_hidden: (B, M_max, D), zero-filled at padded positions.
    """
    B, M_max = ovt_absolute_positions.shape
    D = lm_hidden.shape[-1]
    # Clamp positions to a valid index (we will zero-out padded entries).
    safe_positions = ovt_absolute_positions.clamp(min=0, max=lm_hidden.shape[1] - 1)
    gathered = lm_hidden.gather(
        dim=1,
        index=safe_positions.unsqueeze(-1).expand(-1, -1, D),
    )
    mask = ovt_valid_mask.unsqueeze(-1).to(gathered.dtype)
    return gathered * mask


# ---------------------------------------------------------------------------
# Per-OVT mask BCE supervision  (L2)
# ---------------------------------------------------------------------------
def compute_per_ovt_mask_logits(
    ovt_hidden: torch.Tensor,
    img_hidden: torch.Tensor,
    temperature: float = 1.0,
    normalize_tokens: bool = True,
) -> torch.Tensor:
    """Dot-product between every OVT token and every image patch token.

    Args:
        ovt_hidden:  (B, M, D)
        img_hidden:  (B, P, D)

    Returns:
        logits: (B, M, P) — raw scaled dot-product, used for BCE & sigmoid map.
    """
    d = ovt_hidden.shape[-1]
    ov = ovt_hidden.float()
    img = img_hidden.float()
    if normalize_tokens:
        ov = F.layer_norm(ov, (d,))
        img = F.layer_norm(img, (d,))
    temp = max(float(temperature), 1e-6)
    logits = torch.einsum("bmd,bpd->bmp", ov, img) / (math.sqrt(d) * temp)
    return logits


def compute_mask_bce_loss(
    ovt_logits: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
) -> torch.Tensor:
    """BCE loss over valid OVT tokens.

    Args:
        ovt_logits:        (B, M, P) raw logits.
        gt_masks_per_ovt:  (B, M, P) GT patch masks, broadcast-replicated for
                           the multiple OVT tokens that belong to the same object.
        ovt_valid_mask:    (B, M) bool.

    Returns:
        scalar BCE averaged over valid OVT positions.
    """
    B, M, P = ovt_logits.shape
    logits_f = ovt_logits.float()
    targets_f = gt_masks_per_ovt.float()

    bce = F.binary_cross_entropy_with_logits(
        logits_f, targets_f, reduction="none"
    )  # (B, M, P)
    bce = bce.mean(dim=-1)  # (B, M)
    valid = ovt_valid_mask.float()
    denom = valid.sum().clamp_min(1.0)
    return (bce * valid).sum() / denom


def compute_spatial_outside_attention_loss(
    ovt_logits: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    n_ovt_per_object: int,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """V8 outside-attention loss over all valid thing/stuff regions.

    For each OVT, normalize its patch scores with a spatial softmax. The loss is
    the attention mass assigned to every other annotated region. Unannotated or
    void patches are neutral because they are not part of the valid-region union.
    """
    B, M, P = ovt_logits.shape
    n = max(int(n_ovt_per_object), 1)
    K = M // n
    if K <= 0:
        z = ovt_logits.new_zeros((), dtype=torch.float32)
        return {"loss": z, "self_mass": z, "other_mass": z, "neutral_mass": z}

    logits = ovt_logits[:, : K * n].reshape(B, K, n, P).float()
    masks = gt_masks_per_ovt[:, : K * n].reshape(B, K, n, P).float().amax(dim=2)
    obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)

    masks = masks.clamp(0.0, 1.0) * obj_valid.unsqueeze(-1).float()
    region_union = masks.amax(dim=1, keepdim=True)  # (B,1,P)
    other_regions = []
    for k in range(K):
        if K == 1:
            other_regions.append(torch.zeros_like(masks[:, k]))
        else:
            other_regions.append(torch.cat([masks[:, :k], masks[:, k + 1:]], dim=1).amax(dim=1))
    forbidden = torch.stack(other_regions, dim=1).clamp(0.0, 1.0)  # (B,K,P)
    self_region = masks
    neutral = (1.0 - region_union).clamp(0.0, 1.0)

    temp = max(float(temperature), eps)
    attn = F.softmax(logits / temp, dim=-1)  # (B,K,n,P), spatial over patches

    valid = obj_valid.unsqueeze(-1).float()  # (B,K,1)
    denom = (valid.sum() * n).clamp_min(1.0)
    other_mass = (attn * forbidden.unsqueeze(2)).sum(dim=-1)
    loss = (other_mass * valid).sum() / denom

    self_mass = (attn * self_region.unsqueeze(2)).sum(dim=-1)
    neutral_mass = (attn * neutral.unsqueeze(2)).sum(dim=-1)
    self_mean = (self_mass * valid).sum() / denom
    other_mean = (other_mass * valid).sum() / denom
    neutral_mean = (neutral_mass * valid).sum() / denom

    for val in (loss, self_mean, other_mean, neutral_mean):
        if not torch.isfinite(val):
            z = ovt_logits.new_zeros((), dtype=torch.float32)
            return {"loss": z, "self_mass": z, "other_mass": z, "neutral_mass": z}

    return {
        "loss": loss,
        "self_mass": self_mean.detach(),
        "other_mass": other_mean.detach(),
        "neutral_mass": neutral_mean.detach(),
    }


def compute_spatial_outside_log_attention_loss(
    ovt_logits: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """V8.2 outside-only log penalty on per-OVT spatial-softmax attention.

    For each valid OVT, normalize patch scores with a patch-axis softmax. The
    target object's GT mask is ignored; every patch outside that mask is treated
    as a negative and is penalized with -log(1 - attention). The outside terms
    are summed over patches, so for small attention values this is close to the
    outside attention mass but keeps the intended log penalty.
    """
    logits = ovt_logits.float()
    masks = gt_masks_per_ovt.float().clamp(0.0, 1.0)
    valid = ovt_valid_mask.float()
    temp = max(float(temperature), eps)

    attn = F.softmax(logits / temp, dim=-1)
    outside = (1.0 - masks).clamp(0.0, 1.0)

    per_ovt_loss = (outside * (-torch.log((1.0 - attn).clamp_min(eps)))).sum(dim=-1)

    valid_denom = valid.sum().clamp_min(1.0)
    loss = (per_ovt_loss * valid).sum() / valid_denom

    self_mass = (attn * masks).sum(dim=-1)
    outside_mass = (attn * outside).sum(dim=-1)
    self_mean = (self_mass * valid).sum() / valid_denom
    outside_mean = (outside_mass * valid).sum() / valid_denom
    outside_log_mean = (per_ovt_loss * valid).sum() / valid_denom

    for val in (loss, self_mean, outside_mean, outside_log_mean):
        if not torch.isfinite(val):
            z = ovt_logits.new_zeros((), dtype=torch.float32)
            return {"loss": z, "self_mass": z, "outside_mass": z, "outside_log_mean": z}

    return {
        "loss": loss,
        "self_mass": self_mean.detach(),
        "outside_mass": outside_mean.detach(),
        "outside_log_mean": outside_log_mean.detach(),
    }


def compute_per_patch_ce_loss(
    ovt_logits: torch.Tensor,        # (B, M, P) raw logits
    gt_masks_per_ovt: torch.Tensor,  # (B, M, P) soft patch masks
    ovt_valid_mask: torch.Tensor,    # (B, M) bool — valid OVTs (thing + stuff)
    temperature: float = 1.0,
) -> torch.Tensor:
    """CODA-style per-patch competition loss.

    Each patch p is assigned (softmax over OVTs) to exactly ONE OVT. The GT label
    for patch p is the OVT with the highest GT mask coverage at that patch.
    Patches not covered by any valid OVT are ignored (-100).

    This enforces an EXCLUSIVE object partition (good for fARI) while remaining
    fully supervised (good for mBO/mIoU). thing + stuff OVTs both compete, so
    stuff OVTs naturally absorb background pixels.
    """
    B, M, P = ovt_logits.shape
    temp = max(float(temperature), 1e-6)
    logits = ovt_logits.float() / temp
    neg = torch.finfo(logits.dtype).min
    valid = ovt_valid_mask.unsqueeze(-1)               # (B, M, 1)
    logits = logits.masked_fill(~valid, neg)           # invalid OVTs can't win

    # GT patch label = argmax over valid OVTs' coverage
    gt = gt_masks_per_ovt.float().masked_fill(~valid, 0.0)
    patch_label = gt.argmax(dim=1)                     # (B, P)
    max_cover = gt.amax(dim=1)                          # (B, P)
    ignore = max_cover <= 0.0
    patch_label = patch_label.masked_fill(ignore, -100)

    # CE over OVT dimension per patch:  logits (B, M, P) -> (B*P, M)
    logits_t = logits.permute(0, 2, 1).reshape(B * P, M)
    loss = F.cross_entropy(logits_t, patch_label.reshape(B * P), ignore_index=-100)
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=ovt_logits.device, dtype=torch.float32)
    return loss


def compute_competition_ce_loss(
    ovt_logits: torch.Tensor,        # (B, M, P) OVT-vs-patch logits
    reg_logits: torch.Tensor,        # (B, R, P) register-vs-patch logits
    gt_masks_per_ovt: torch.Tensor,  # (B, M, P) soft coverage (2 OVTs of an obj share mask)
    ovt_valid_mask: torch.Tensor,    # (B, M) bool
    n_ovt_per_object: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Object-level CODA-style competition CE with a register BACKGROUND class.

    Each patch competes over {K objects, background}:
      - object k logit = max over its n OVTs (the n OVTs cooperate, not compete)
      - background logit = max over the R register tokens (register = scene/non-object)
    GT label per patch:
      - the object with the highest GT coverage, if any object covers it
      - else the BACKGROUND class (index K)  -> uncovered patches are now supervised
        (no -100 ignore), so the model learns an explicit background.

    Fixes the two failure modes of the per-OVT ignore-CE: (1) the n OVTs of one
    object no longer compete against each other; (2) background patches have a real
    target instead of being ignored & then force-assigned to some object at eval.
    """
    B, M, P = ovt_logits.shape
    n = n_ovt_per_object
    K = M // n
    temp = max(float(temperature), 1e-6)
    neg = torch.finfo(torch.float32).min

    # object logits: max over the n OVTs per object
    obj_logits = ovt_logits[:, : K * n].reshape(B, K, n, P).float().amax(dim=2) / temp  # (B,K,P)
    obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)                   # (B,K)
    obj_logits = obj_logits.masked_fill(~obj_valid.unsqueeze(-1), neg)

    # background logit: max over registers
    bg_logit = reg_logits.float().amax(dim=1, keepdim=True) / temp                       # (B,1,P)

    all_logits = torch.cat([obj_logits, bg_logit], dim=1)                                # (B, K+1, P)

    # GT patch label = best-covering valid object, else background class (K)
    obj_cover = gt_masks_per_ovt[:, : K * n].reshape(B, K, n, P).float().amax(dim=2)     # (B,K,P)
    obj_cover = obj_cover.masked_fill(~obj_valid.unsqueeze(-1), 0.0)
    best_cover, best_idx = obj_cover.max(dim=1)                                          # (B,P)
    bg_index = torch.full_like(best_idx, K)
    label = torch.where(best_cover > 0.0, best_idx, bg_index)                            # (B,P)

    logits_t = all_logits.permute(0, 2, 1).reshape(B * P, K + 1)
    loss = F.cross_entropy(logits_t, label.reshape(B * P))
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=ovt_logits.device, dtype=torch.float32)
    return loss


def compute_null_bg_competition_losses(
    ovt_logits: torch.Tensor,          # (B, M, P) OVT-vs-patch logits
    null_bg_logits: torch.Tensor,      # (B, 1, P) null-bg-vs-patch logits
    gt_masks_per_ovt: torch.Tensor,    # (B, M, P) soft coverage
    ovt_valid_mask: torch.Tensor,      # (B, M) bool
    ovt_is_thing: torch.Tensor,        # (B, M) bool
    n_ovt_per_object: int,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """V7 ownership losses over {thing objects, null-bg}.

    Stuff/background/register tokens are not segmentation classes. Patches covered
    by a GT thing object target that thing object; every other patch targets the
    single null-bg owner.
    """
    B, M, P = ovt_logits.shape
    n = max(int(n_ovt_per_object), 1)
    K = M // n
    temp = max(float(temperature), 1e-6)
    neg = torch.finfo(torch.float32).min

    obj_logits = ovt_logits[:, : K * n].reshape(B, K, n, P).float().amax(dim=2) / temp
    obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)
    obj_thing = ovt_is_thing[:, : K * n].reshape(B, K, n).any(dim=2)
    thing_valid = obj_valid & obj_thing
    obj_logits = obj_logits.masked_fill(~thing_valid.unsqueeze(-1), neg)

    bg_logit = null_bg_logits.float() / temp
    if bg_logit.ndim == 2:
        bg_logit = bg_logit.unsqueeze(1)
    all_logits = torch.cat([obj_logits, bg_logit], dim=1)  # (B, K+1, P)

    obj_cover = gt_masks_per_ovt[:, : K * n].reshape(B, K, n, P).float().amax(dim=2)
    thing_cover = obj_cover.masked_fill(~thing_valid.unsqueeze(-1), 0.0)
    best_cover, best_idx = thing_cover.max(dim=1)
    bg_index = torch.full_like(best_idx, K)
    label = torch.where(best_cover > 0.0, best_idx, bg_index)

    logits_t = all_logits.permute(0, 2, 1).reshape(B * P, K + 1)
    loss_owner = F.cross_entropy(logits_t, label.reshape(B * P))

    probs = F.softmax(all_logits, dim=1)
    thing_probs = probs[:, :K]
    bg_prob = probs[:, K]
    fg_prob = thing_probs.sum(dim=1).clamp_min(eps)

    fg_weight = best_cover.clamp(0.0, 1.0)
    fg_denom = fg_weight.sum().clamp_min(1.0)
    loss_fg = -(fg_weight * torch.log(fg_prob)).sum() / fg_denom

    outside_weight = (1.0 - thing_cover.clamp(0.0, 1.0)) * thing_valid.unsqueeze(-1).float()
    out_denom = outside_weight.sum().clamp_min(1.0)
    loss_outside = (outside_weight * thing_probs).sum() / out_denom

    bg_weight = (1.0 - fg_weight).clamp(0.0, 1.0)
    bg_denom = bg_weight.sum().clamp_min(1.0)
    thing_on_bg = (bg_weight * fg_prob).sum() / bg_denom
    bg_on_fg = (fg_weight * bg_prob).sum() / fg_denom

    losses = {
        "loss_owner": loss_owner,
        "loss_fg": loss_fg,
        "loss_outside": loss_outside,
        "bg_prob_on_fg": bg_on_fg.detach(),
        "thing_prob_on_bg": thing_on_bg.detach(),
        "thing_object_count": thing_valid.float().sum(dim=1).mean().detach(),
    }
    for key in ("loss_owner", "loss_fg", "loss_outside"):
        if not bool(torch.isfinite(losses[key]).all()):
            losses[key] = torch.zeros((), device=ovt_logits.device, dtype=torch.float32)
    return losses


def build_pred_mask_null_bg_eval(
    ovt_logits: torch.Tensor,        # (B, M, P) raw logits
    null_bg_logits: torch.Tensor,    # (B, 1, P) null-bg class logits
    ovt_valid_mask: torch.Tensor,    # (B, M) bool
    ovt_is_thing: torch.Tensor,      # (B, M) bool
    target_size: int,
    n_ovt_per_object: int,
    patch_grid: int = 32,
    merge: str = "max",
) -> torch.Tensor:
    """V7 eval readout: argmax over {thing objects, null-bg}.

    null-bg wins -> background 0. Stuff OVTs and registers are excluded from
    segmentation ownership.
    """
    B, M, P = ovt_logits.shape
    n = max(int(n_ovt_per_object), 1)
    K = M // n
    logits = ovt_logits[:, : K * n].reshape(B, K, n, P).float()
    obj_logits = logits.amax(dim=2) if merge == "max" else logits.mean(dim=2)
    obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)
    obj_thing = ovt_is_thing[:, : K * n].reshape(B, K, n).any(dim=2)
    thing_valid = obj_valid & obj_thing

    bg_logit = null_bg_logits.float()
    if bg_logit.ndim == 2:
        bg_logit = bg_logit.unsqueeze(1)

    obj_2d = obj_logits.reshape(B, K, patch_grid, patch_grid)
    bg_2d = bg_logit.reshape(B, 1, patch_grid, patch_grid)
    all_2d = torch.cat([obj_2d, bg_2d], dim=1)
    up = F.interpolate(all_2d, size=(target_size, target_size), mode="bilinear", align_corners=False)
    neg = torch.finfo(up.dtype).min
    valid_full = torch.cat(
        [thing_valid, torch.ones(B, 1, dtype=torch.bool, device=thing_valid.device)], dim=1
    )
    up = up.masked_fill(~valid_full.view(B, K + 1, 1, 1), neg)
    assign = up.argmax(dim=1)

    pred = torch.zeros((B, target_size, target_size), dtype=torch.int64, device=ovt_logits.device)
    for b in range(B):
        rank = 0
        for k in range(K):
            if not bool(thing_valid[b, k]):
                continue
            rank += 1
            pred[b][assign[b] == k] = rank
    return pred


def build_pred_mask_competition_eval(
    ovt_logits: torch.Tensor,        # (B, M, P) raw logits
    reg_logits: torch.Tensor,        # (B, R, P) register-vs-patch logits (background class)
    ovt_valid_mask: torch.Tensor,    # (B, M) bool (thing + stuff valid)
    ovt_is_thing: torch.Tensor,      # (B, M) bool — True if OVT belongs to a thing object
    target_size: int,
    n_ovt_per_object: int,
    patch_grid: int = 32,
    merge: str = "max",
) -> torch.Tensor:
    """Eval readout matching compute_competition_ce_loss: argmax over
    {K objects, register-background}. A patch wins:
      - a THING object  -> that thing's index (1..K_thing)
      - a STUFF object  -> background (0)
      - the BACKGROUND class (register) -> background (0)

    Returns integer mask (B, H, W): 0 = background, 1..K = thing objects.
    """
    B, M, P = ovt_logits.shape
    K = M // n_ovt_per_object
    logits = ovt_logits[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object, P).float()
    obj_logits = logits.amax(dim=2) if merge == "max" else logits.mean(dim=2)   # (B, K, P)
    obj_valid = ovt_valid_mask.reshape(B, K, n_ovt_per_object).any(dim=2)        # (B, K)
    obj_thing = ovt_is_thing.reshape(B, K, n_ovt_per_object).any(dim=2)          # (B, K)
    bg_logit = reg_logits.float().amax(dim=1, keepdim=True)                       # (B, 1, P)

    # Build (B, K+1, grid, grid), upsample, then mask invalid objects before argmax.
    obj_2d = obj_logits.reshape(B, K, patch_grid, patch_grid)
    bg_2d = bg_logit.reshape(B, 1, patch_grid, patch_grid)
    all_2d = torch.cat([obj_2d, bg_2d], dim=1)                                   # (B, K+1, g, g)
    up = F.interpolate(all_2d, size=(target_size, target_size), mode="bilinear", align_corners=False)
    neg = torch.finfo(up.dtype).min
    valid_full = torch.cat(
        [obj_valid, torch.ones(B, 1, dtype=torch.bool, device=obj_valid.device)], dim=1
    )                                                                            # (B, K+1)
    up = up.masked_fill(~valid_full.view(B, K + 1, 1, 1), neg)
    assign = up.argmax(dim=1)   # (B, H, W) → 0..K  (K = register background)

    pred = torch.zeros((B, target_size, target_size), dtype=torch.int64, device=ovt_logits.device)
    for b in range(B):
        a = assign[b]
        rank = 0
        for k in range(K):
            if not bool(obj_valid[b, k]):
                continue
            if bool(obj_thing[b, k]):
                rank += 1
                pred[b][a == k] = rank
            # stuff object → leave as 0 (background)
        # assign == K (register background) → stays 0
    return pred


def compute_anti_overlap_loss(
    ovt_logits: torch.Tensor,        # (B, M, P) raw logits
    ovt_valid_mask: torch.Tensor,    # (B, M) bool
    n_ovt_per_object: int,
) -> torch.Tensor:
    """OBJECT-LEVEL pairwise anti-overlap penalty (for v6.5).

    A single object owns n OVTs that share the same GT mask, so the n OVTs
    should co-activate on the object's patches (BCE drives this). We must
    NOT penalize within-object overlap; we only penalize CROSS-object overlap
    (two different objects firing on the same patch).

    Algorithm:
      1) sigmoid(ovt_logits) -> per-OVT mask prob p (B, M, P)
      2) zero-out invalid OVTs
      3) mean-pool the n OVTs of each object into an object-level prob
         p_obj (B, K, P), K = M // n
      4) pairwise cross-object activation product (per patch):
           overlap[b, p] = Σ_{k<l, both valid} p_obj[b, k, p] · p_obj[b, l, p]
                        = ½ · ( S² − Σ_k p_obj_k² )      (algebraic identity)
      5) normalize by num valid object pairs per sample so the loss scale is
         comparable across batches with different object counts.

    Range: ∈ [0, 0.25] per sample (max when all objects at p=0.5).
    Cooperates with BCE/Tversky: BCE pulls each OVT toward its GT (which is
    panoptic-exclusive across objects), anti-overlap erodes any residual
    inter-object overlap. They target the same exclusive structure.
    """
    B, M, P = ovt_logits.shape
    n = n_ovt_per_object
    K = M // n

    probs = torch.sigmoid(ovt_logits.float())                              # (B, M, P)
    valid = ovt_valid_mask.float().unsqueeze(-1)                            # (B, M, 1)
    probs = probs * valid                                                   # invalid → 0

    # Pool to object-level (mean of n OVTs per object) -- key correctness step
    p_obj = probs[:, : K * n].reshape(B, K, n, P).mean(dim=2)               # (B, K, P)

    # Object-level validity (object valid if any of its n OVTs are valid)
    obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)      # (B, K)
    p_obj = p_obj * obj_valid.float().unsqueeze(-1)                         # (B, K, P), invalid → 0

    # Per-patch cross-object overlap via algebraic identity
    S  = p_obj.sum(dim=1)                                                   # (B, P)
    S2 = (p_obj * p_obj).sum(dim=1)                                         # (B, P)
    overlap_per_patch = 0.5 * (S * S - S2).clamp_min(0.0)                   # (B, P)

    # Normalize by num valid object pairs per sample
    n_valid = obj_valid.float().sum(dim=1)                                  # (B,)
    n_pairs = (n_valid * (n_valid - 1) / 2).clamp_min(1.0)                  # (B,)

    loss = (overlap_per_patch.mean(dim=1) / n_pairs).mean()
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=ovt_logits.device, dtype=torch.float32)
    return loss


def compute_mask_tversky_loss(
    ovt_logits: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss on per-OVT mask logits.

    Tversky = TP / (TP + α·FP + β·FN)
    α = β = 0.5 reduces to Dice (matches AURORA's captionslot setting).
    Higher α penalises false-positives more (sharper boundaries) — good for fARI.
    """
    probs = torch.sigmoid(ovt_logits.float())
    targets = gt_masks_per_ovt.float()
    valid = ovt_valid_mask.float().unsqueeze(-1)
    tp = (probs * targets * valid).sum(dim=-1)
    fp = (probs * (1.0 - targets) * valid).sum(dim=-1)
    fn = ((1.0 - probs) * targets * valid).sum(dim=-1)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    valid_per_ovt = ovt_valid_mask.float()
    denom = valid_per_ovt.sum().clamp_min(1.0)
    return ((1.0 - tversky) * valid_per_ovt).sum() / denom


def compute_ovt_shuffle_contrastive_loss(
    ovt_logits: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    sampling_rate: float = 0.5,
    margin: float = 0.2,
) -> torch.Tensor:
    """Hinge loss that makes OVT maps worse on shuffled object masks.

    The positive mask BCE is handled by L2. This term samples valid OVT slots,
    pairs them with same-index masks from another batch item, and pushes the
    shuffled-mask BCE to be at least `margin` worse than the positive BCE.
    """
    B, M, _ = ovt_logits.shape
    if B < 2:
        return ovt_logits.new_zeros(())

    shift = B // 2
    neg_masks = torch.cat([gt_masks_per_ovt[shift:], gt_masks_per_ovt[:shift]], dim=0)
    neg_valid = torch.cat([ovt_valid_mask[shift:], ovt_valid_mask[:shift]], dim=0)

    valid = ovt_valid_mask & neg_valid
    if sampling_rate < 1.0:
        sampled = torch.rand(B, M, device=ovt_logits.device) <= float(sampling_rate)
        valid = valid & sampled

    if not bool(valid.any()):
        return ovt_logits.new_zeros(())

    logits_f = ovt_logits.float()
    pos_targets = gt_masks_per_ovt.float()
    neg_targets = neg_masks.float()

    pos_bce = F.binary_cross_entropy_with_logits(
        logits_f, pos_targets, reduction="none"
    ).mean(dim=-1)
    neg_bce = F.binary_cross_entropy_with_logits(
        logits_f, neg_targets, reduction="none"
    ).mean(dim=-1)

    hinge = F.relu(float(margin) + pos_bce.detach() - neg_bce)
    valid_f = valid.float()
    return (hinge * valid_f).sum() / valid_f.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Soft-merge OVT attention map for object-level segmentation metric
# ---------------------------------------------------------------------------
def merge_ovt_maps_to_object(
    ovt_maps: torch.Tensor,
    n_ovt_per_object: int,
    object_valid_mask: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Aggregate per-OVT maps into per-object maps.

    Args:
        ovt_maps:    (B, M_max, P) sigmoid maps.
        n_ovt_per_object: e.g., 2.
        object_valid_mask: (B, K_max) bool.
        mode: "mean" | "max"

    Returns:
        (B, K_max, P) maps. Padded with zeros for inactive objects.
    """
    B, M_max, P = ovt_maps.shape
    K_max = M_max // n_ovt_per_object
    maps = ovt_maps[:, : K_max * n_ovt_per_object].reshape(B, K_max, n_ovt_per_object, P)
    if mode == "max":
        merged = maps.amax(dim=2)
    else:
        merged = maps.mean(dim=2)
    merged = merged * object_valid_mask.unsqueeze(-1).to(merged.dtype)
    return merged


# ---------------------------------------------------------------------------
# Contrastive shuffle  (L4)
# ---------------------------------------------------------------------------
def shuffle_ovt_for_contrastive(
    ovt_hidden: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    sampling_rate: float = 0.5,
) -> torch.Tensor:
    """CODA-style negative: roll the batch by half and randomly swap OVT slots.

    Args:
        ovt_hidden:     (B, M, D)
        ovt_valid_mask: (B, M)
        sampling_rate:  probability that an OVT slot is REPLACED by the shifted batch's OVT.

    Returns:
        shifted_hidden: (B, M, D) — partially shuffled.
    """
    B = ovt_hidden.shape[0]
    if B < 2:
        return ovt_hidden
    shift = B // 2
    shifted = torch.cat([ovt_hidden[shift:], ovt_hidden[:shift]], dim=0)
    swap = (torch.rand(B, ovt_hidden.shape[1], 1, device=ovt_hidden.device) <= sampling_rate).to(ovt_hidden.dtype)
    mixed = swap * shifted + (1.0 - swap) * ovt_hidden
    return mixed * ovt_valid_mask.unsqueeze(-1).to(mixed.dtype)
