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
) -> Dict[str, int]:
    """Compute absolute positions for every block.

    Layout (after concat):
        [ sys_prefix | sys_suffix
        | user_prefix | image | user_suffix
        | assistant_prefix | caption(with <ovt>) | assistant_suffix
        | register | rae_query ]

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

        # Register rows: image + caption + self (DOES NOT see rae_query)
        for row_idx in range(reg_s, reg_e):
            allow_cols(b_idx, row_idx, img_s, img_e)
            if valid_cap_positions.numel() > 0:
                bias[b_idx, row_idx, valid_cap_positions] = 0.0
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
