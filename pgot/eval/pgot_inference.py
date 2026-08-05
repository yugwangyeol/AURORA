"""Eval-time forward for PGOT — recomposes the same sequence as `_forward_pgot`
but skips loss computation and returns the intermediate tensors needed for
metric evaluation:
    ovt_logits   : (B, M_max, P)   raw mask logits per OVT
    rae_hidden   : (B, K_q, D)     fed into DiT for image reconstruction
    img_features : (B, P, D)       LLM-side image hidden states
    gt_siglip    : (B, P, C)       diffusion target features (for rFID later)
"""

from typing import Dict, Optional
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
    return_llm_attention_maps: bool = False,
    llm_attention_readout: str = "auto",
    rae_access_mode: str = "baseline",
    return_hidden_states: bool = False,
    return_v12_block_maps: bool = False,
    rae_block_ovt_indices: tuple[int, ...] = (),
    zero_ovt_inputs: bool = False,
    zero_register_inputs: bool = False,
    register_image_block_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()

    if llm_attention_readout not in {"auto", "fvw", "core"}:
        raise ValueError(
            "llm_attention_readout must be one of auto/fvw/core, got "
            f"{llm_attention_readout}"
        )

    device = model._pgot_model_device() if hasattr(model, "_pgot_model_device") else model.pgot_register_embeddings.device
    images = images.to(device)
    target_images = target_images.to(device)
    caption_input_ids = caption_input_ids.to(device)
    caption_attention_mask = caption_attention_mask.to(device, dtype=torch.bool)
    ovt_positions_in_caption = ovt_positions_in_caption.to(device, dtype=torch.long)
    ovt_valid_mask = ovt_valid_mask.to(device, dtype=torch.bool)
    B, caption_len = caption_input_ids.shape

    if bool(getattr(model.config, "pgot_v14_enable", False)):
        if rae_access_mode != "baseline":
            raise ValueError("V14 eval currently supports only rae_access_mode='baseline'.")
        out = model._pgot_v14_forward_features(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
            output_hidden_states=bool(return_hidden_states),
        )
        hidden = out["hidden"]
        img_hidden = out["img_hidden"]
        attn_temp = float(getattr(model.config, "pgot_attention_temperature", 1.0))
        attn_ln = bool(getattr(model.config, "pgot_attention_use_layer_norm", True))
        ovt_hidden = gather_ovt_hidden_states(hidden, out["ovt_abs_positions"], out["ovt_valid_mask"])
        ovt_logits = compute_per_ovt_mask_logits(
            ovt_hidden=ovt_hidden,
            img_hidden=img_hidden,
            temperature=attn_temp,
            normalize_tokens=attn_ln,
        )
        P = img_hidden.shape[1]
        reg_logits = img_hidden.new_empty(B, 0, P)
        null_bg_logits = None
        if out["void_ovts"].shape[1] > 0:
            null_bg_logits = compute_per_ovt_mask_logits(
                ovt_hidden=out["void_ovts"],
                img_hidden=img_hidden,
                temperature=attn_temp,
                normalize_tokens=attn_ln,
            )
        slot_context = None
        slot_mask = None
        if (
            hasattr(model, "_pgot_prepare_dit_ovt_context")
            and bool(getattr(model.config, "pgot_dit_ovt_cross_attn_enable", False))
        ):
            slot_context, slot_mask = model._pgot_prepare_dit_ovt_context(
                out["ovt_states"],
                out["ovt_valid"],
            )
        result = {
            "ovt_logits": ovt_logits,
            "reg_logits": reg_logits,
            "null_bg_logits": null_bg_logits,
            "llm_qk_attn_maps": None,
            "llm_attention_maps": None,
            "llm_attention_void_maps": None,
            "llm_attention_source": None,
            "ovt_valid_mask": out["ovt_valid_mask"],
            "rae_hidden": out["condition_hidden"],
            "raw_rae_hidden": out["rae_hidden"],
            "slot_context": slot_context,
            "slot_mask": slot_mask,
            "img_hidden": img_hidden,
            "gt_siglip": out["gt_siglip"],
            "rae_access_mode": rae_access_mode,
            "hidden": hidden,
            "attn_bias": out["attn_bias"],
            "positions": out["positions"],
            "ovt_abs_positions": out["ovt_abs_positions"],
            "ovt_owner_logits": out["owner_logits"],
            "ovt_object_probs": out["object_probs"],
            "ovt_void_probs": out["void_probs"],
            "ovt_object_valid": out["object_valid"],
            "v12_block_owner_records": None,
        }
        if getattr(model, "pgot_latent_head", None) is not None and hasattr(model, "pgot_predict_direct_latent"):
            result["direct_latent"] = model.pgot_predict_direct_latent(out["condition_hidden"])
        if return_hidden_states:
            result["hidden_states"] = out["hidden_states"]
            result["inputs_embeds"] = out["inputs_embeds"].detach()
        return result

    if bool(getattr(model.config, "pgot_v12_enable", False)):
        if rae_access_mode != "baseline":
            raise ValueError("V12 eval currently supports only rae_access_mode='baseline'.")
        out = model._pgot_v12_forward_features(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
            output_hidden_states=bool(return_hidden_states),
            return_v12_block_maps=bool(return_v12_block_maps),
            rae_block_ovt_indices=tuple(rae_block_ovt_indices),
        )
        hidden = out["hidden"]
        img_hidden = out["img_hidden"]
        attn_temp = float(getattr(model.config, "pgot_attention_temperature", 1.0))
        attn_ln = bool(getattr(model.config, "pgot_attention_use_layer_norm", True))
        ovt_hidden = gather_ovt_hidden_states(hidden, out["ovt_abs_positions"], out["ovt_valid_mask"])
        ovt_logits = compute_per_ovt_mask_logits(
            ovt_hidden=ovt_hidden,
            img_hidden=img_hidden,
            temperature=attn_temp,
            normalize_tokens=attn_ln,
        )
        P = img_hidden.shape[1]
        reg_logits = img_hidden.new_empty(B, 0, P)
        null_bg_logits = None
        if out["void_ovts"].shape[1] > 0:
            null_bg_logits = compute_per_ovt_mask_logits(
                ovt_hidden=out["void_ovts"],
                img_hidden=img_hidden,
                temperature=attn_temp,
                normalize_tokens=attn_ln,
            )
        result = {
            "ovt_logits": ovt_logits,
            "reg_logits": reg_logits,
            "null_bg_logits": null_bg_logits,
            "llm_qk_attn_maps": None,
            "llm_attention_maps": None,
            "llm_attention_void_maps": None,
            "llm_attention_source": None,
            "ovt_valid_mask": out["ovt_valid_mask"],
            "rae_hidden": out["rae_hidden"],
            "img_hidden": img_hidden,
            "gt_siglip": out["gt_siglip"],
            "rae_access_mode": rae_access_mode,
            "hidden": hidden,
            "attn_bias": out["attn_bias"],
            "positions": out["positions"],
            "ovt_abs_positions": out["ovt_abs_positions"],
            "ovt_owner_logits": out["owner_logits"],
            "ovt_object_probs": out["object_probs"],
            "ovt_void_probs": out["void_probs"],
            "ovt_object_valid": out["object_valid"],
            "v12_block_owner_records": out.get("v12_block_owner_records"),
        }
        if return_hidden_states:
            result["hidden_states"] = out["hidden_states"]
            result["inputs_embeds"] = out["inputs_embeds"].detach()
        return result

    # 1) Vision tower. Direct-SD experiments have no Scale-RAE target-latent
    # path, so they avoid the second (224px) vision-tower forward entirely.
    e6_enabled = bool(getattr(model.config, "pgot_e6_enable", False))
    e7_enabled = bool(getattr(model.config, "pgot_e7_enable", False))
    direct_sd_enabled = e6_enabled or e7_enabled
    _, img_features, gt_siglip = model._encode_images_aurora(
        images, target_images=None if direct_sd_enabled else target_images
    )
    dtype = model._aurora_model_dtype()

    # 2) Template + caption embeds (no grad)
    sys_p = model._pgot_embed_frozen_tokens(model.pgot_system_prefix_ids, B, device, dtype)
    sys_s = model._pgot_embed_frozen_tokens(model.pgot_system_suffix_ids, B, device, dtype)
    user_p = model._pgot_embed_frozen_tokens(model.pgot_user_prefix_ids, B, device, dtype)
    user_s = model._pgot_embed_frozen_tokens(model.pgot_user_suffix_ids, B, device, dtype)
    asst_p = model._pgot_embed_frozen_tokens(model.pgot_assistant_prefix_ids, B, device, dtype)
    asst_s = model._pgot_embed_frozen_tokens(model.pgot_assistant_suffix_ids, B, device, dtype)
    caption_embeds = model._pgot_embed_caption(caption_input_ids, device, dtype)
    caption_embeds = model._pgot_apply_caption_conditioned_ovt_init(
        caption_embeds, caption_input_ids, ovt_positions_in_caption, ovt_valid_mask
    )
    null_bg_embeds = model._pgot_embed_null_bg(B, device, dtype)
    register_embeds = model.pgot_register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
    if direct_sd_enabled:
        n_rae = 0
        rae_embeds = torch.empty(
            B, 0, model.config.hidden_size, device=device, dtype=dtype
        )
    else:
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
    if zero_ovt_inputs or zero_register_inputs:
        inputs_embeds = inputs_embeds.clone()
        if zero_ovt_inputs:
            for b_idx in range(B):
                valid_pos = ovt_abs_positions[b_idx][ovt_valid_mask[b_idx]]
                if valid_pos.numel() > 0:
                    inputs_embeds[b_idx, valid_pos] = 0
        if zero_register_inputs and positions["reg_e"] > positions["reg_s"]:
            inputs_embeds[:, positions["reg_s"]:positions["reg_e"]] = 0

    attn_bias = build_pgot_attention_mask(
        positions=positions,
        caption_padding_mask=caption_attention_mask,
        device=device,
        dtype=inputs_embeds.dtype,
        rae_bidirectional=bool(getattr(model.config, "pgot_rae_bidirectional", False)),
        rae_isolated=bool(getattr(model.config, "pgot_e4_rae_isolated", False)),
        rae_attends_caption=bool(getattr(model.config, "pgot_rae_attends_caption", False)),
        ovt_absolute_positions=ovt_abs_positions,
        ovt_valid_mask=ovt_valid_mask,
        register_attends_caption=bool(
            getattr(model.config, "pgot_register_attends_caption", True)
        ),
        ovt_isolated=bool(
            getattr(model.config, "pgot_ovt_isolated_attention", False)
        ),
        ovt_attends_own_caption=bool(
            getattr(model.config, "pgot_ovt_attends_own_caption", False)
        ),
    )
    register_blocked_patch_fraction = inputs_embeds.new_zeros((), dtype=torch.float32)
    register_blocked_patch_count = inputs_embeds.new_zeros((), dtype=torch.float32)
    if register_image_block_mask is not None:
        blocked = register_image_block_mask.to(device=device, dtype=torch.bool)
        n_patches = int(positions["img_e"] - positions["img_s"])
        if blocked.ndim != 2 or blocked.shape != (B, n_patches):
            raise ValueError(
                "register_image_block_mask must have shape [B,P]="
                f"[{B},{n_patches}], got {tuple(blocked.shape)}"
            )
        if positions["reg_e"] > positions["reg_s"]:
            attn_bias = attn_bias.clone()
            register_to_image = attn_bias[
                :, :, positions["reg_s"]:positions["reg_e"],
                positions["img_s"]:positions["img_e"],
            ]
            register_to_image.masked_fill_(
                blocked[:, None, None, :], float("-inf")
            )
        register_blocked_patch_fraction = blocked.float().mean().detach()
        register_blocked_patch_count = blocked.float().sum(dim=-1).mean().detach()
    rae_access_mode = str(rae_access_mode).lower()
    if rae_access_mode not in {"baseline", "ovt_only", "register_only", "self_only"}:
        raise ValueError(f"Unknown rae_access_mode={rae_access_mode}")
    if rae_access_mode != "baseline":
        attn_bias = attn_bias.clone()
        rae_s, rae_e = positions["rae_s"], positions["rae_e"]
        if rae_access_mode in {"ovt_only", "self_only"}:
            attn_bias[:, :, rae_s:rae_e, positions["null_bg_s"]:positions["reg_e"]] = float("-inf")
        if rae_access_mode in {"register_only", "self_only"}:
            for b_idx in range(B):
                valid_pos = ovt_abs_positions[b_idx][ovt_valid_mask[b_idx]]
                if valid_pos.numel() > 0:
                    attn_bias[b_idx, :, rae_s:rae_e, valid_pos] = float("-inf")
    if rae_block_ovt_indices:
        attn_bias = attn_bias.clone()
        rae_s, rae_e = positions["rae_s"], positions["rae_e"]
        for b_idx in range(B):
            for ovt_idx in rae_block_ovt_indices:
                if 0 <= ovt_idx < ovt_abs_positions.shape[1] and bool(ovt_valid_mask[b_idx, ovt_idx]):
                    pos = int(ovt_abs_positions[b_idx, ovt_idx].item())
                    attn_bias[b_idx, :, rae_s:rae_e, pos] = float("-inf")

    # 6) LLM forward
    need_llm_qk_maps = bool(return_llm_qk_maps) and (
        float(getattr(model.config, "pgot_mask_llm_qk_outside_weight", 0.0)) > 0.0
    )
    llm_patch_attention_enabled = (
        float(getattr(model.config, "pgot_mask_llm_patch_outside_weight", 0.0)) > 0.0
        or float(getattr(model.config, "pgot_mask_llm_image_use_weight", 0.0)) > 0.0
        or float(getattr(model.config, "pgot_core_outside_weight", 0.0)) > 0.0
        or float(getattr(model.config, "pgot_core_tail_weight", 0.0)) > 0.0
        or float(getattr(model.config, "pgot_core_register_outside_weight", 0.0)) > 0.0
    )
    need_llm_attention_maps = bool(return_llm_attention_maps) and (
        float(getattr(model.config, "pgot_mask_llm_attention_outside_weight", 0.0)) > 0.0
        or llm_patch_attention_enabled
    )
    need_hidden_states = (
        need_llm_qk_maps or need_llm_attention_maps or bool(return_hidden_states)
    )
    if bool(getattr(model.config, "pgot_fvw_enable", False)):
        out, fvw_records = model._pgot_forward_with_fvw(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            positions=positions,
            ovt_abs_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            output_hidden_states=need_hidden_states,
        )
    else:
        out = model.model(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            use_cache=False,
            output_hidden_states=need_hidden_states,
            return_dict=True,
        )
        fvw_records = []
    hidden = out.last_hidden_state

    fvw_attention_maps = None
    fvw_last_attention_maps = None
    if fvw_records:
        fvw_per_write = [record["weights"].float().mean(dim=1) for record in fvw_records]
        fvw_attention_maps = torch.stack(fvw_per_write, dim=0).mean(dim=0)
        fvw_last_attention_maps = fvw_per_write[-1]
        valid_float = ovt_valid_mask.unsqueeze(-1).to(fvw_attention_maps.dtype)
        fvw_attention_maps = fvw_attention_maps * valid_float
        fvw_last_attention_maps = fvw_last_attention_maps * valid_float

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

    llm_attention_maps = None
    llm_attention_void_maps = None
    llm_attention_register_maps = None
    llm_attention_source = None
    e3_competition_object_probs = None
    e3_competition_background_probs = None
    if (
        need_llm_attention_maps
        and llm_patch_attention_enabled
        and hasattr(model, "_compute_llm_patch_attention_maps")
    ):
        use_fvw_attention = (
            fvw_attention_maps is not None
            and llm_attention_readout in {"auto", "fvw"}
        )
        if llm_attention_readout == "fvw" and fvw_attention_maps is None:
            raise ValueError(
                "llm_attention_readout='fvw' requested but the checkpoint did not "
                "produce FVW maps"
            )
        if use_fvw_attention:
            # FVW's map is the attention that actually writes the exported OVT
            # value, so it is the primary segmentation/routing readout.
            llm_attention_maps = fvw_attention_maps
            llm_attention_source = "fvw_image_only_write_softmax"
        else:
            llm_attention_maps, llm_attention_void_maps = (
                model._compute_llm_patch_attention_maps(
                    hidden_states=out.hidden_states,
                    attention_bias=attn_bias,
                    positions=positions,
                    ovt_abs_positions=ovt_abs_positions,
                    ovt_valid_mask=ovt_valid_mask,
                    layers_spec=str(
                        getattr(model.config, "pgot_core_outside_layers", "all")
                        if float(getattr(model.config, "pgot_core_outside_weight", 0.0)) > 0.0
                        or float(getattr(model.config, "pgot_core_tail_weight", 0.0)) > 0.0
                        else getattr(
                            model.config, "pgot_mask_llm_patch_outside_layers", "last4"
                        )
                    ),
                    temperature=float(
                        getattr(model.config, "pgot_core_outside_temperature", 1.0)
                        if float(getattr(model.config, "pgot_core_outside_weight", 0.0)) > 0.0
                        or float(getattr(model.config, "pgot_core_tail_weight", 0.0)) > 0.0
                        else getattr(
                            model.config, "pgot_mask_llm_patch_outside_temperature", 1.0
                        )
                    ),
                )
            )
            llm_attention_source = "post_rope_image_patch_softmax"
        # Register maps are an eval readout, not a training-loss diagnostic.
        # E2 uses a hard train-time register route and intentionally sets the
        # soft register-loss weight to zero, so gating this map on that weight
        # silently removed the background class at evaluation time.
        if (
            int(positions.get("reg_e", 0)) > int(positions.get("reg_s", 0))
            and hasattr(model, "_compute_llm_register_patch_attention_maps")
        ):
            llm_attention_register_maps = model._compute_llm_register_patch_attention_maps(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                layers_spec=str(getattr(model.config, "pgot_core_outside_layers", "all")),
                temperature=float(
                    getattr(model.config, "pgot_core_outside_temperature", 1.0)
                ),
            )
        if (
            float(
                getattr(
                    model.config,
                    "pgot_e3_attention_competition_weight",
                    0.0,
                )
            )
            > 0.0
            and hasattr(model, "_compute_e3_competition_owner_maps")
        ):
            (
                e3_competition_object_probs,
                e3_competition_background_probs,
            ) = model._compute_e3_competition_owner_maps(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                layers_spec=str(
                    getattr(
                        model.config,
                        "pgot_e3_attention_competition_layers",
                        "all",
                    )
                ),
                temperature=float(
                    getattr(
                        model.config,
                        "pgot_e3_attention_competition_temperature",
                        1.0,
                    )
                ),
            )
    elif need_llm_attention_maps and hasattr(model, "_compute_exact_llm_attention_maps"):
        llm_attention_maps, llm_attention_void_maps = model._compute_exact_llm_attention_maps(
            hidden_states=out.hidden_states,
            attention_bias=attn_bias,
            positions=positions,
            ovt_abs_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            layers_spec=str(
                getattr(model.config, "pgot_mask_llm_attention_outside_layers", "last4")
            ),
        )
        llm_attention_source = "post_rope_full_key_softmax"

    result = {
        "ovt_logits": ovt_logits,
        "reg_logits": reg_logits,
        "null_bg_logits": null_bg_logits,
        "llm_qk_attn_maps": llm_qk_attn_maps,
        "llm_attention_maps": llm_attention_maps,
        "llm_attention_void_maps": llm_attention_void_maps,
        "llm_attention_register_maps": llm_attention_register_maps,
        "llm_attention_source": llm_attention_source,
        "fvw_attention_maps": fvw_attention_maps,
        "fvw_last_attention_maps": fvw_last_attention_maps,
        "fvw_write_layers": [int(record["layer"]) for record in fvw_records],
        "e3_competition_object_probs": e3_competition_object_probs,
        "e3_competition_background_probs": e3_competition_background_probs,
        "ovt_valid_mask": ovt_valid_mask,
        "rae_hidden": rae_hidden,
        "ovt_hidden": ovt_hidden,
        "register_hidden": register_hidden,
        "img_hidden": img_hidden,
        "gt_siglip": gt_siglip,
        "rae_access_mode": rae_access_mode,
        "register_image_block_mask": (
            register_image_block_mask.detach()
            if register_image_block_mask is not None
            else None
        ),
        "register_blocked_patch_fraction": register_blocked_patch_fraction,
        "register_blocked_patch_count": register_blocked_patch_count,
    }
    # Expose internals needed for ovt-swap editing
    result["hidden"] = hidden
    result["attn_bias"] = attn_bias
    result["positions"] = positions
    result["ovt_abs_positions"] = ovt_abs_positions
    if return_hidden_states:
        result["hidden_states"] = out.hidden_states
        result["inputs_embeds"] = inputs_embeds.detach()
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
def ovt_swap_all_layers_inference(
    model,
    out_A: Dict,
    out_B: Dict,
    swap_pairs: list,
    n_ovt_per_object: int = 2,
):
    """Causally replace selected OVT states throughout the LLM stack.

    ``ovt_swap_inference`` is the legacy last-layer approximation.  This
    variant uses the cached layer-input trajectory from image B and overwrites
    the selected OVT positions in image A before *every* transformer layer.
    Consequently A's RAE queries read the donor OVT at every depth while A's
    image, other OVTs, registers, attention mask, and spatial queries remain
    unchanged.

    Both inputs must come from ``pgot_forward_eval(...,
    return_hidden_states=True)``.
    """
    hidden_states_A = out_A.get("hidden_states")
    hidden_states_B = out_B.get("hidden_states")
    if hidden_states_A is None or hidden_states_B is None:
        raise ValueError(
            "all-layer OVT swap requires return_hidden_states=True for A and B"
        )

    layers = model.model.layers
    if len(hidden_states_A) < len(layers) + 1:
        raise ValueError(
            "unexpected hidden-state trajectory length for A: "
            f"{len(hidden_states_A)} for {len(layers)} layers"
        )
    if len(hidden_states_B) < len(layers) + 1:
        raise ValueError(
            "unexpected hidden-state trajectory length for B: "
            f"{len(hidden_states_B)} for {len(layers)} layers"
        )

    hidden_mixed = hidden_states_A[0].clone()
    ovt_abs_A = out_A["ovt_abs_positions"]
    ovt_abs_B = out_B["ovt_abs_positions"]
    positions_A = out_A["positions"]
    attn_bias_A = out_A["attn_bias"]
    seq_len = hidden_mixed.shape[1]
    position_ids = torch.arange(
        seq_len,
        device=hidden_mixed.device,
        dtype=torch.long,
    ).unsqueeze(0)

    for layer_idx, layer in enumerate(layers):
        donor_layer_input = hidden_states_B[layer_idx]
        for idx_A, idx_B in swap_pairs:
            a_start = int(idx_A) * int(n_ovt_per_object)
            b_start = int(idx_B) * int(n_ovt_per_object)
            for offset in range(int(n_ovt_per_object)):
                pos_a = int(ovt_abs_A[0, a_start + offset].item())
                pos_b = int(ovt_abs_B[0, b_start + offset].item())
                hidden_mixed[:, pos_a, :] = donor_layer_input[:, pos_b, :]

        try:
            layer_out = layer(
                hidden_mixed,
                attention_mask=attn_bias_A,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden_mixed = (
                layer_out[0] if isinstance(layer_out, tuple) else layer_out
            )
        except TypeError:
            hidden_mixed = layer(
                hidden_mixed,
                attention_mask=attn_bias_A,
            )[0]

        # Match the normal E2-Pix-FVW forward exactly.  Without this step an
        # all-layer swap silently skips the hard visual overwrite and compares
        # a non-FVW intervention against an FVW baseline.
        if (
            bool(getattr(model.config, "pgot_fvw_enable", False))
            and getattr(model, "pgot_fvw_block", None) is not None
            and int(layer_idx) in set(int(x) for x in model.pgot_fvw_layers)
        ):
            current_ovts = gather_ovt_hidden_states(
                hidden_mixed,
                ovt_abs_A,
                out_A["ovt_valid_mask"],
            )
            image_states = hidden_mixed[
                :, positions_A["img_s"]:positions_A["img_e"], :
            ]
            visual_ovts, _ = model.pgot_fvw_block(
                ovt_states=current_ovts,
                image_states=image_states,
                ovt_valid_mask=out_A["ovt_valid_mask"],
            )
            hidden_mixed = model._pgot_fvw_scatter_overwrite(
                hidden_states=hidden_mixed,
                ovt_abs_positions=ovt_abs_A,
                ovt_valid_mask=out_A["ovt_valid_mask"],
                visual_ovts=visual_ovts,
            )

    if hasattr(model.model, "norm"):
        hidden_mixed = model.model.norm(hidden_mixed)
    rae_hidden_mixed = hidden_mixed[
        :, positions_A["rae_s"]:positions_A["rae_e"], :
    ]
    return rae_hidden_mixed, hidden_mixed


@torch.no_grad()
def generate_siglip_latent(
    model,
    rae_hidden: torch.Tensor,
    guidance_level: float = 1.0,
    slot_context: torch.Tensor | None = None,
    slot_mask: torch.Tensor | None = None,
) -> torch.Tensor:
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

    generated = diff_head.infer(
        z=cond,
        x_end=x_end,
        guidance_level=guidance_level,
        slot_context=slot_context,
        slot_mask=slot_mask,
    )
    return generated  # (B, P, C)
