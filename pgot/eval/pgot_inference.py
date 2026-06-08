"""Eval-time forward for PGOT — recomposes the same sequence as `_forward_pgot`
but skips loss computation and returns the intermediate tensors needed for
metric evaluation:
    ovt_logits   : (B, M_max, P)   raw mask logits per OVT
    rae_hidden   : (B, K_q, D)     fed into DiT for image reconstruction
    img_features : (B, P, D)       LLM-side image hidden states
    gt_siglip    : (B, P, C)       diffusion target features (for rFID later)
"""

from typing import Dict
import torch

from pgot.model.pgot_utils import (
    pgot_positions,
    build_pgot_attention_mask,
    gather_ovt_hidden_states,
    compute_per_ovt_mask_logits,
)


@torch.no_grad()
def pgot_forward_eval(
    model,
    *,
    images: torch.Tensor,
    target_images: torch.Tensor,
    caption_input_ids: torch.LongTensor,
    caption_attention_mask: torch.Tensor,
    ovt_positions_in_caption: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    return_llm_qk_maps: bool = False,
) -> Dict[str, torch.Tensor]:
    model.eval()

    device = model.pgot_register_embeddings.device
    images = images.to(device)
    target_images = target_images.to(device)
    caption_input_ids = caption_input_ids.to(device)
    caption_attention_mask = caption_attention_mask.to(device, dtype=torch.bool)
    ovt_positions_in_caption = ovt_positions_in_caption.to(device, dtype=torch.long)
    ovt_valid_mask = ovt_valid_mask.to(device, dtype=torch.bool)
    B, caption_len = caption_input_ids.shape

    # 1) Vision tower
    _, img_features, gt_siglip = model._encode_images_aurora(images, target_images=target_images)
    dtype = model._aurora_model_dtype()

    # 2) Template + caption embeds (no grad)
    sys_p = model._pgot_embed_frozen_tokens(model.pgot_system_prefix_ids, B, device, dtype)
    sys_s = model._pgot_embed_frozen_tokens(model.pgot_system_suffix_ids, B, device, dtype)
    user_p = model._pgot_embed_frozen_tokens(model.pgot_user_prefix_ids, B, device, dtype)
    user_s = model._pgot_embed_frozen_tokens(model.pgot_user_suffix_ids, B, device, dtype)
    asst_p = model._pgot_embed_frozen_tokens(model.pgot_assistant_prefix_ids, B, device, dtype)
    asst_s = model._pgot_embed_frozen_tokens(model.pgot_assistant_suffix_ids, B, device, dtype)
    caption_embeds = model._pgot_embed_caption(caption_input_ids, device, dtype)
    null_bg_embeds = model._pgot_embed_null_bg(B, device, dtype)
    register_embeds = model.pgot_register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
    n_rae = model.get_model().latent_queries.shape[0]
    rae_embeds = model.get_model().latent_queries.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)

    # 3) Positions
    positions = pgot_positions(
        caption_len=caption_len,
        system_prefix_len=sys_p.shape[1],
        system_suffix_len=sys_s.shape[1],
        user_prefix_len=user_p.shape[1],
        user_suffix_len=user_s.shape[1],
        assistant_prefix_len=asst_p.shape[1],
        assistant_suffix_len=asst_s.shape[1],
        num_image_tokens=img_features.shape[1],
        n_register=model.pgot_n_register,
        n_rae_query=n_rae,
        n_null_bg=null_bg_embeds.shape[1],
    )

    # 4) Concat sequence
    inputs_embeds = torch.cat(
        [sys_p, sys_s,
         user_p, img_features.to(dtype=dtype), user_s,
         asst_p, caption_embeds, asst_s,
         null_bg_embeds,
         register_embeds,
         rae_embeds],
        dim=1,
    )

    # 5) Attention bias
    # rae_query attends ONLY to [OVT positions inside caption + register + self]
    # Caption text + image are BLOCKED — matches the v3 training setup.
    cap_s_pos = positions["cap_s"]
    ovt_abs_positions = cap_s_pos + ovt_positions_in_caption
    attn_bias = build_pgot_attention_mask(
        positions=positions,
        caption_padding_mask=caption_attention_mask,
        device=device,
        dtype=inputs_embeds.dtype,
        rae_bidirectional=bool(getattr(model.config, "pgot_rae_bidirectional", False)),
        rae_attends_caption=bool(getattr(model.config, "pgot_rae_attends_caption", False)),
        ovt_absolute_positions=ovt_abs_positions,
        ovt_valid_mask=ovt_valid_mask,
    )

    # 6) LLM forward
    need_llm_qk_maps = bool(return_llm_qk_maps) and (
        float(getattr(model.config, "pgot_mask_llm_qk_outside_weight", 0.0)) > 0.0
    )
    out = model.model(
        inputs_embeds=inputs_embeds,
        attention_bias=attn_bias,
        use_cache=False,
        output_hidden_states=need_llm_qk_maps,
        return_dict=True,
    )
    hidden = out.last_hidden_state

    img_hidden = hidden[:, positions["img_s"]:positions["img_e"], :]
    rae_hidden = hidden[:, positions["rae_s"]:positions["rae_e"], :]
    ovt_hidden = gather_ovt_hidden_states(hidden, ovt_abs_positions, ovt_valid_mask)

    # 7) OVT-image attention logits (+ register-image logits = background class)
    _attn_temp = float(getattr(model.config, "pgot_attention_temperature", 1.0))
    _attn_ln = bool(getattr(model.config, "pgot_attention_use_layer_norm", True))
    ovt_logits = compute_per_ovt_mask_logits(
        ovt_hidden=ovt_hidden,
        img_hidden=img_hidden,
        temperature=_attn_temp,
        normalize_tokens=_attn_ln,
    )
    register_hidden = hidden[:, positions["reg_s"]:positions["reg_e"], :]
    reg_logits = compute_per_ovt_mask_logits(
        ovt_hidden=register_hidden,
        img_hidden=img_hidden,
        temperature=_attn_temp,
        normalize_tokens=_attn_ln,
    )
    null_bg_hidden = hidden[:, positions["null_bg_s"]:positions["null_bg_e"], :]
    null_bg_logits = None
    if null_bg_hidden.shape[1] > 0:
        null_bg_logits = compute_per_ovt_mask_logits(
            ovt_hidden=null_bg_hidden,
            img_hidden=img_hidden,
            temperature=_attn_temp,
            normalize_tokens=_attn_ln,
        )

    llm_qk_attn_maps = None
    if need_llm_qk_maps and hasattr(model, "_compute_llm_qk_attention_maps"):
        llm_qk_attn_maps = model._compute_llm_qk_attention_maps(
            hidden_states=out.hidden_states,
            positions=positions,
            ovt_abs_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            layers_spec=str(getattr(model.config, "pgot_mask_llm_qk_outside_layers", "last4")),
            temperature=float(getattr(model.config, "pgot_mask_llm_qk_outside_temperature", 1.0)),
        )

    result = {
        "ovt_logits": ovt_logits,
        "reg_logits": reg_logits,
        "null_bg_logits": null_bg_logits,
        "llm_qk_attn_maps": llm_qk_attn_maps,
        "ovt_valid_mask": ovt_valid_mask,
        "rae_hidden": rae_hidden,
        "img_hidden": img_hidden,
        "gt_siglip": gt_siglip,
    }
    # Expose internals needed for ovt-swap editing
    result["hidden"] = hidden
    result["attn_bias"] = attn_bias
    result["positions"] = positions
    result["ovt_abs_positions"] = ovt_abs_positions
    return result


@torch.no_grad()
def ovt_swap_inference(
    model,
    out_A: Dict,
    out_B: Dict,
    swap_pairs: list,
    n_ovt_per_object: int = 2,
):
    """OVT-swap editing inference (matches training-time negative-branch recipe).

    Given two evaluated samples A and B, replace selected OVT positions in A's
    LLM hidden state with B's corresponding OVT positions, re-run only the
    LAST LLM layer (so rae_query reflects the swap), and return rae_hidden_mixed.

    Args:
        out_A, out_B: dicts returned by pgot_forward_eval (must include
                       'hidden', 'attn_bias', 'positions', 'ovt_abs_positions').
        swap_pairs: list of (obj_idx_in_A, obj_idx_in_B). For each pair, we
                       overwrite the n_ovt_per_object OVT tokens of object obj_idx_in_A
                       in A with the OVT tokens of object obj_idx_in_B in B.

    Returns:
        rae_hidden_mixed: (1, K_q, D)  — fed to diff_head for image gen
        hidden_mixed:     (1, L, D)
    """
    model.eval()
    hidden_A = out_A["hidden"]
    hidden_B = out_B["hidden"]
    ovt_abs_A = out_A["ovt_abs_positions"]   # (1, M_max)
    ovt_abs_B = out_B["ovt_abs_positions"]
    positions_A = out_A["positions"]
    attn_bias_A = out_A["attn_bias"]
    D = hidden_A.shape[-1]

    hidden_mixed = hidden_A.clone()
    for (idx_A, idx_B) in swap_pairs:
        a_start = idx_A * n_ovt_per_object
        b_start = idx_B * n_ovt_per_object
        for k in range(n_ovt_per_object):
            pos_a = int(ovt_abs_A[0, a_start + k].item())
            pos_b = int(ovt_abs_B[0, b_start + k].item())
            hidden_mixed[:, pos_a, :] = hidden_B[:, pos_b, :]

    # Re-run only the last layer with the mixed hidden so rae_query attention reflects the swap.
    try:
        last_layer = model.model.layers[-1]
    except AttributeError:
        return hidden_mixed[:, positions_A["rae_s"]:positions_A["rae_e"], :], hidden_mixed

    L = hidden_mixed.shape[1]
    position_ids = torch.arange(L, device=hidden_mixed.device).unsqueeze(0)
    try:
        layer_out = last_layer(
            hidden_mixed,
            attention_mask=attn_bias_A,
            position_ids=position_ids,
            use_cache=False,
        )
        updated = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    except TypeError:
        updated = last_layer(hidden_mixed, attention_mask=attn_bias_A)[0]

    rae_hidden_mixed = updated[:, positions_A["rae_s"]:positions_A["rae_e"], :]
    return rae_hidden_mixed, updated


@torch.no_grad()
def generate_siglip_latent(model, rae_hidden: torch.Tensor, guidance_level: float = 1.0) -> torch.Tensor:
    """Run diff_head.infer to denoise SigLIP latent from rae_hidden.

    Returns: (B, P, C)  — SigLIP target-space latent (P = diffusion_target_token_len).
    """
    cond = model._captionslot_prepare_diffusion_condition(rae_hidden).float()
    device = cond.device
    model.diff_head = model.diff_head.to(device)
    model.set_diff_fp32()

    diff_head = model.diff_head
    P = int(diff_head.diffusion_tokens)
    C = int(diff_head.diffusion_channels)
    side = int(round(P ** 0.5))
    B = cond.shape[0]
    x_end = torch.randn((B, C, side, side), device=device, dtype=torch.float32)

    generated = diff_head.infer(z=cond, x_end=x_end, guidance_level=guidance_level)
    return generated  # (B, P, C)
