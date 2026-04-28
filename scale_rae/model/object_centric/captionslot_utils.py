"""
CaptionSlot attention utilities.
"""

from typing import Dict, Optional

import torch


def build_captionslot_attention_mask(
    positions: Dict[str, int],
    active_slot_mask: torch.Tensor,
    caption_padding_mask: torch.Tensor,
    device: torch.device,
    ref_spans: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    noun_chunk_spans: Optional[torch.Tensor] = None,
    slot_prior_maps: Optional[torch.Tensor] = None,
    slot_prior_valid_mask: Optional[torch.Tensor] = None,
    prior_bias_scale: float = 0.0,
) -> torch.Tensor:
    """Build additive attention bias for CaptionSlot.

    Args:
        positions: Segment boundary dictionary from `_captionslot_positions`.
        ref_spans: `[B, K_max, 2]` tensor of per-slot referring-expression spans.
        active_slot_mask: `[B, K_max]` bool tensor.
        caption_padding_mask: `[B, T]` bool tensor.
        noun_chunk_spans / slot_prior_* / prior_bias_scale:
            Legacy compatibility arguments kept as no-ops for older callers.

    Returns:
        `[B, 1, L, L]` additive attention bias.
    """
    if ref_spans is None:
        ref_spans = noun_chunk_spans
    if ref_spans is None:
        raise ValueError("build_captionslot_attention_mask requires ref_spans.")

    batch_size = int(active_slot_mask.shape[0])
    total_len = int(positions["total_len"])
    neg_inf = float("-inf")

    bias = torch.full((batch_size, total_len, total_len), neg_inf, device=device, dtype=dtype)
    diag = torch.arange(total_len, device=device)
    bias[:, diag, diag] = 0.0

    def allow_cols(batch_idx: int, row_idx: int, start: int, end: int) -> None:
        if end > start:
            bias[batch_idx, row_idx, start:end] = 0.0

    def allow_rows_causal(start: int, end: int) -> None:
        for row_idx in range(start, end):
            bias[:, row_idx, : row_idx + 1] = 0.0

    sys_s, sys_e = positions["sys_s"], positions["sys_e"]
    user_prefix_s, user_prefix_e = positions["user_prefix_s"], positions["user_prefix_e"]
    img_s, img_e = positions["img_s"], positions["img_e"]
    user_suffix_s, user_suffix_e = positions["user_suffix_s"], positions["user_suffix_e"]
    assistant_prefix_s, assistant_prefix_e = positions["assistant_prefix_s"], positions["assistant_prefix_e"]
    cap_s, cap_e = positions["cap_s"], positions["cap_e"]
    slot_s, slot_e = positions["slot_s"], positions["slot_e"]
    reg_s, reg_e = positions["reg_s"], positions["reg_e"]
    im_start_idx = positions["im_start_idx"]
    rae_s, rae_e = positions["rae_s"], positions["rae_e"]
    im_end_idx = positions["im_end_idx"]
    assistant_suffix_s, assistant_suffix_e = positions["assistant_suffix_s"], positions["assistant_suffix_e"]

    allow_rows_causal(sys_s, sys_e)
    allow_rows_causal(user_prefix_s, user_prefix_e)
    allow_rows_causal(user_suffix_s, user_suffix_e)
    allow_rows_causal(assistant_prefix_s, assistant_prefix_e)
    allow_rows_causal(assistant_suffix_s, assistant_suffix_e)

    prefix_visible_end = user_prefix_e
    caption_prefix_visible_end = assistant_prefix_e

    for batch_idx in range(batch_size):
        active_idx = active_slot_mask[batch_idx].nonzero(as_tuple=False).flatten()
        active_slot_positions = slot_s + active_idx
        valid_caption_idx = caption_padding_mask[batch_idx].nonzero(as_tuple=False).flatten()
        valid_caption_positions = cap_s + valid_caption_idx

        for row_idx in range(img_s, img_e):
            allow_cols(batch_idx, row_idx, sys_s, prefix_visible_end)
            allow_cols(batch_idx, row_idx, img_s, img_e)

        for local_idx in range(cap_e - cap_s):
            row_idx = cap_s + local_idx
            if not bool(caption_padding_mask[batch_idx, local_idx]):
                continue
            allow_cols(batch_idx, row_idx, sys_s, caption_prefix_visible_end)
            allow_cols(batch_idx, row_idx, cap_s, row_idx + 1)

        # Each active slot reads the full image grid and only its own referring-expression span.
        for slot_local_idx in active_idx.tolist():
            row_idx = slot_s + slot_local_idx
            allow_cols(batch_idx, row_idx, img_s, img_e)
            span_start = int(ref_spans[batch_idx, slot_local_idx, 0].item())
            span_end = int(ref_spans[batch_idx, slot_local_idx, 1].item())
            if span_start >= 0 and span_end > span_start:
                allow_cols(
                    batch_idx,
                    row_idx,
                    cap_s + span_start,
                    cap_s + min(span_end, cap_e - cap_s),
                )

        # Registers aggregate residual/global information.
        for row_idx in range(reg_s, reg_e):
            allow_cols(batch_idx, row_idx, img_s, img_e)
            if valid_caption_positions.numel() > 0:
                bias[batch_idx, row_idx, valid_caption_positions] = 0.0
            if active_slot_positions.numel() > 0:
                bias[batch_idx, row_idx, active_slot_positions] = 0.0
            allow_cols(batch_idx, row_idx, reg_s, reg_e)

        if valid_caption_positions.numel() > 0:
            bias[batch_idx, im_start_idx, valid_caption_positions] = 0.0
        if active_slot_positions.numel() > 0:
            bias[batch_idx, im_start_idx, active_slot_positions] = 0.0
        allow_cols(batch_idx, im_start_idx, reg_s, reg_e)
        bias[batch_idx, im_start_idx, im_start_idx] = 0.0

        # Frozen RAE queries only read caption/slot/register summaries.
        for row_idx in range(rae_s, rae_e):
            if valid_caption_positions.numel() > 0:
                bias[batch_idx, row_idx, valid_caption_positions] = 0.0
            if active_slot_positions.numel() > 0:
                bias[batch_idx, row_idx, active_slot_positions] = 0.0
            allow_cols(batch_idx, row_idx, reg_s, reg_e)
            bias[batch_idx, row_idx, im_start_idx] = 0.0
            allow_cols(batch_idx, row_idx, rae_s, row_idx + 1)

        if valid_caption_positions.numel() > 0:
            bias[batch_idx, im_end_idx, valid_caption_positions] = 0.0
        if active_slot_positions.numel() > 0:
            bias[batch_idx, im_end_idx, active_slot_positions] = 0.0
        allow_cols(batch_idx, im_end_idx, reg_s, reg_e)
        bias[batch_idx, im_end_idx, im_start_idx] = 0.0
        allow_cols(batch_idx, im_end_idx, rae_s, rae_e)

    return bias.unsqueeze(1)


def build_captionslot_caption_only_attention_mask(
    positions: Dict[str, int],
    text_padding_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build additive attention bias for caption-only reconstruction control."""
    batch_size = int(text_padding_mask.shape[0])
    total_len = int(positions["total_len"])
    neg_inf = float("-inf")

    bias = torch.full((batch_size, total_len, total_len), neg_inf, device=device, dtype=dtype)
    diag = torch.arange(total_len, device=device)
    bias[:, diag, diag] = 0.0

    def allow_cols(batch_idx: int, row_idx: int, start: int, end: int) -> None:
        if end > start:
            bias[batch_idx, row_idx, start:end] = 0.0

    def allow_rows_causal(start: int, end: int) -> None:
        for row_idx in range(start, end):
            bias[:, row_idx, : row_idx + 1] = 0.0

    sys_s, sys_e = positions["sys_s"], positions["sys_e"]
    sys_suffix_s, sys_suffix_e = positions["sys_suffix_s"], positions["sys_suffix_e"]
    user_prefix_s, user_prefix_e = positions["user_prefix_s"], positions["user_prefix_e"]
    text_s, text_e = positions["text_s"], positions["text_e"]
    user_suffix_s, user_suffix_e = positions["user_suffix_s"], positions["user_suffix_e"]
    assistant_prefix_s, assistant_prefix_e = positions["assistant_prefix_s"], positions["assistant_prefix_e"]
    im_start_idx = positions["im_start_idx"]
    rae_s, rae_e = positions["rae_s"], positions["rae_e"]
    im_end_idx = positions["im_end_idx"]
    assistant_suffix_s, assistant_suffix_e = positions["assistant_suffix_s"], positions["assistant_suffix_e"]

    allow_rows_causal(sys_s, sys_e)
    allow_rows_causal(sys_suffix_s, sys_suffix_e)
    allow_rows_causal(user_prefix_s, user_prefix_e)
    allow_rows_causal(user_suffix_s, user_suffix_e)
    allow_rows_causal(assistant_prefix_s, assistant_prefix_e)
    allow_rows_causal(assistant_suffix_s, assistant_suffix_e)

    text_prefix_visible_end = user_prefix_e

    for batch_idx in range(batch_size):
        valid_text_idx = text_padding_mask[batch_idx].nonzero(as_tuple=False).flatten()
        valid_text_positions = text_s + valid_text_idx

        for local_idx in range(text_e - text_s):
            row_idx = text_s + local_idx
            if not bool(text_padding_mask[batch_idx, local_idx]):
                continue
            allow_cols(batch_idx, row_idx, sys_s, text_prefix_visible_end)
            allow_cols(batch_idx, row_idx, text_s, row_idx + 1)

        if valid_text_positions.numel() > 0:
            bias[batch_idx, im_start_idx, valid_text_positions] = 0.0
        bias[batch_idx, im_start_idx, im_start_idx] = 0.0

        for row_idx in range(rae_s, rae_e):
            if valid_text_positions.numel() > 0:
                bias[batch_idx, row_idx, valid_text_positions] = 0.0
            bias[batch_idx, row_idx, im_start_idx] = 0.0
            allow_cols(batch_idx, row_idx, rae_s, row_idx + 1)

        if valid_text_positions.numel() > 0:
            bias[batch_idx, im_end_idx, valid_text_positions] = 0.0
        bias[batch_idx, im_end_idx, im_start_idx] = 0.0
        allow_cols(batch_idx, im_end_idx, rae_s, rae_e)

    return bias.unsqueeze(1)
