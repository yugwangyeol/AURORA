"""PGOT model: Qwen2 MLLM + caption-grounded object visual tokens.

Inherits ScaleRAEQwenForCausalLM and adds a `use_pgot` forward path:
  1. tokenize caption (already done by dataloader) — caption contains <ovt>'s inline
  2. assemble sequence: [sys | user_prefix | image | user_suffix | assistant_prefix
                         | caption(with <ovt>) | assistant_suffix
                         | register | rae_query]
  3. run LLM with PGOT attention bias (rae_query CANNOT see image)
  4. extract OVT hidden states + image hidden states + rae_hidden
  5. compute L1 (LM) + L2 (per-OVT mask BCE) + L3 (rectified-flow recon)
     + L4 (contrastive ovt shuffle) [warmup-gated]
"""

from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from scale_rae.model.language_model.scale_rae_qwen2 import (
    ScaleRAEQwenForCausalLM,
    ScaleRAEQwenConfig,
)
from pgot.model.pgot_utils import (
    pgot_positions,
    build_pgot_attention_mask,
    gather_ovt_hidden_states,
    compute_per_ovt_mask_logits,
    compute_mask_bce_loss,
    compute_spatial_outside_attention_loss,
    compute_spatial_outside_log_attention_loss,
    compute_mask_tversky_loss,
    compute_per_patch_ce_loss,
    compute_competition_ce_loss,
    compute_null_bg_competition_losses,
    compute_anti_overlap_loss,
    compute_ovt_shuffle_contrastive_loss,
)


class PGOTQwen2Config(ScaleRAEQwenConfig):
    """Config for PGOT — extends ScaleRAE with PGOT-specific fields."""
    model_type = "pgot_qwen2"


class PGOTQwen2ForCausalLM(ScaleRAEQwenForCausalLM):
    """PGOT model: caption-grounded variable-K object tokens."""

    config_class = PGOTQwen2Config

    def __init__(self, config):
        super().__init__(config)
        # PGOT-specific init runs LAST so register/template ids are set up.
        if bool(getattr(config, "use_pgot", False)):
            self._init_pgot()

    # ------------------------------------------------------------------
    # PGOT module init
    # ------------------------------------------------------------------
    def _init_pgot(self) -> None:
        D = self.config.hidden_size
        embed_std = 1.0 / math.sqrt(D)

        self.pgot_n_register = int(getattr(self.config, "pgot_n_register", 64))
        self.pgot_n_ovt_per_object = int(getattr(self.config, "pgot_n_ovt_per_object", 2))
        self.pgot_use_null_bg_competition = bool(
            getattr(self.config, "pgot_use_null_bg_competition", False)
        )
        self.pgot_n_null_bg = int(
            getattr(self.config, "pgot_n_null_bg", 1 if self.pgot_use_null_bg_competition else 0)
        )

        # Register: 새로 만든다 (사용자 답변: fine-tune 불가, 새 임베딩)
        self.pgot_register_embeddings = nn.Parameter(
            torch.randn(self.pgot_n_register, D) * embed_std
        )
        if self.pgot_n_null_bg > 0:
            self.pgot_null_bg_embeddings = nn.Parameter(
                torch.randn(self.pgot_n_null_bg, D) * embed_std
            )

        # rae_query는 부모 클래스의 self.get_model().latent_queries에 이미 있음
        # → scale-rae checkpoint에서 fine-tune

        # PGOT bookkeeping
        self.pgot_loss_lm = None
        self.pgot_loss_mask = None
        self.pgot_loss_recon = None
        self.pgot_loss_contrastive = None
        self.pgot_loss_details = {}
        self.pgot_n_objects_mean = None

        # Will be set after tokenizer registration
        self.pgot_ovt_token_id = None
        self.pgot_scene_end_token_id = None

        # Frozen-template token id sequences (set by trainer setup hook)
        self.pgot_system_prefix_ids: List[int] = []
        self.pgot_system_suffix_ids: List[int] = []
        self.pgot_user_prefix_ids: List[int] = []
        self.pgot_user_suffix_ids: List[int] = []
        self.pgot_assistant_prefix_ids: List[int] = []
        self.pgot_assistant_suffix_ids: List[int] = []

        print(
            f"[PGOT] Initialised — D={D}, N_register={self.pgot_n_register}, "
            f"n_ovt_per_object={self.pgot_n_ovt_per_object}, "
            f"N_null_bg={self.pgot_n_null_bg}"
        )

    # ------------------------------------------------------------------
    # forward dispatcher
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        decoding: Optional[bool] = False,
        # PGOT-specific inputs
        caption_input_ids: Optional[torch.LongTensor] = None,
        caption_attention_mask: Optional[torch.Tensor] = None,
        caption_labels: Optional[torch.LongTensor] = None,
        ovt_positions_in_caption: Optional[torch.Tensor] = None,
        ovt_valid_mask: Optional[torch.Tensor] = None,
        ovt_is_thing: Optional[torch.Tensor] = None,
        gt_masks_per_ovt: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        pgot_contrastive_weight: Optional[float] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if (
            bool(getattr(self.config, "use_pgot", False))
            and not decoding
            and images is not None
            and caption_input_ids is not None
        ):
            pgot_loss, _ = self._forward_pgot(
                images=images,
                caption_input_ids=caption_input_ids,
                caption_attention_mask=caption_attention_mask,
                caption_labels=caption_labels,
                ovt_positions_in_caption=ovt_positions_in_caption,
                ovt_valid_mask=ovt_valid_mask,
                ovt_is_thing=ovt_is_thing,
                gt_masks_per_ovt=gt_masks_per_ovt,
                target_images=target_images,
                pgot_contrastive_weight=pgot_contrastive_weight,
            )
            return CausalLMOutputWithPast(
                loss=pgot_loss,
                logits=torch.zeros(1, device=images.device),
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict,
            decoding=decoding,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _pgot_embed_frozen_tokens(
        self,
        token_ids: List[int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Embed deterministic template tokens with NO gradient — matches AURORA's approach."""
        if len(token_ids) == 0:
            return torch.empty(batch_size, 0, self.config.hidden_size, device=device, dtype=dtype)
        ids = torch.tensor(
            token_ids,
            device=self.model.embed_tokens.weight.device,
            dtype=torch.long,
        ).unsqueeze(0)
        with torch.no_grad():
            embeds = self.model.embed_tokens(ids).detach()
        return embeds.expand(batch_size, -1, -1).to(device=device, dtype=dtype)

    def _pgot_embed_caption(
        self,
        caption_input_ids: torch.LongTensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Embed caption tokens WITH gradient (LM next-token loss flows through)."""
        embeds = self.model.embed_tokens(
            caption_input_ids.to(device=self.model.embed_tokens.weight.device)
        )
        return embeds.to(device=device, dtype=dtype)

    def _pgot_embed_null_bg(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        n_null = int(getattr(self, "pgot_n_null_bg", 0))
        if n_null <= 0 or not hasattr(self, "pgot_null_bg_embeddings"):
            return torch.empty(batch_size, 0, self.config.hidden_size, device=device, dtype=dtype)
        return self.pgot_null_bg_embeddings.unsqueeze(0).expand(batch_size, -1, -1).to(
            device=device, dtype=dtype
        )

    def _resolve_llm_qk_outside_layers(self, spec: str) -> List[int]:
        """Resolve a layer spec such as 'last4', 'all', '20:28', or '20,24,27'."""
        n_layers = len(getattr(self.model, "layers", []))
        if n_layers <= 0:
            return []
        spec = (spec or "last4").strip().lower()
        if spec in {"all", "*"}:
            return list(range(n_layers))
        if spec.startswith("last"):
            suffix = spec[4:]
            k = int(suffix) if suffix else 4
            return list(range(max(0, n_layers - max(k, 1)), n_layers))

        layers = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                start_s, end_s = part.split(":", 1)
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else n_layers
                if start < 0:
                    start = n_layers + start
                if end < 0:
                    end = n_layers + end
                layers.extend(range(max(0, start), min(n_layers, end)))
            else:
                idx = int(part)
                if idx < 0:
                    idx = n_layers + idx
                if 0 <= idx < n_layers:
                    layers.append(idx)
        deduped = []
        seen = set()
        for idx in layers:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
        return deduped

    def _compute_llm_qk_outside_attention_loss(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Outside loss from the LLM's own Q/K projections at selected layers.

        For every selected layer, OVT tokens are queries and image tokens are keys.
        The softmax axis is image patches, per OVT and per head. The loss is the
        attention mass assigned to other annotated regions. This is loss-only:
        it does not insert a new attention block or alter the next-layer hidden.
        """
        if hidden_states is None:
            z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
            return {"loss": z, "self_mass": z, "other_mass": z, "neutral_mass": z}

        B, M, P = gt_masks_per_ovt.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M // n
        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if K <= 0 or len(layers) == 0:
            z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
            return {"loss": z, "self_mass": z, "other_mass": z, "neutral_mass": z}

        masks = gt_masks_per_ovt[:, : K * n].reshape(B, K, n, P).float().amax(dim=2)
        obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)
        masks = masks.clamp(0.0, 1.0) * obj_valid.unsqueeze(-1).float()
        region_union = masks.amax(dim=1, keepdim=True)

        other_regions = []
        for k in range(K):
            if K == 1:
                other_regions.append(torch.zeros_like(masks[:, k]))
            else:
                other_regions.append(torch.cat([masks[:, :k], masks[:, k + 1:]], dim=1).amax(dim=1))
        forbidden = torch.stack(other_regions, dim=1).clamp(0.0, 1.0)
        neutral = (1.0 - region_union).clamp(0.0, 1.0)

        temp = max(float(temperature), 1e-6)
        loss_sum = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        self_sum = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        other_sum = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        neutral_sum = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        valid_layer_count = 0

        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            layer_input = hidden_states[layer_idx]
            block = self.model.layers[layer_idx]
            attn = block.self_attn
            attn_input = block.input_layernorm(layer_input)

            img_input = attn_input[:, positions["img_s"]:positions["img_e"], :]
            if img_input.shape[1] != P:
                continue
            ovt_input = gather_ovt_hidden_states(attn_input, ovt_abs_positions, ovt_valid_mask)
            ovt_input = ovt_input[:, : K * n]

            q_proj = attn.q_proj(ovt_input)
            k_proj = attn.k_proj(img_input)
            num_heads = int(getattr(attn, "num_heads", getattr(self.config, "num_attention_heads", 1)))
            num_kv_heads = int(
                getattr(attn, "num_key_value_heads", getattr(self.config, "num_key_value_heads", num_heads))
            )
            head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // max(num_heads, 1)))
            if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
                continue

            q = q_proj.reshape(B, K * n, num_heads, head_dim)
            k_img = k_proj.reshape(B, P, num_kv_heads, head_dim)
            if num_kv_heads != num_heads:
                repeat = max(num_heads // num_kv_heads, 1)
                k_img = k_img.repeat_interleave(repeat, dim=2)
                if k_img.shape[2] < num_heads:
                    pad = num_heads - k_img.shape[2]
                    k_img = torch.cat([k_img, k_img[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                elif k_img.shape[2] > num_heads:
                    k_img = k_img[:, :, :num_heads]

            scores = torch.einsum("bmhd,bphd->bmhp", q.float(), k_img.float())
            scores = scores / math.sqrt(float(head_dim))
            scores = scores.reshape(B, K, n, num_heads, P)
            attn_probs = F.softmax(scores / temp, dim=-1)

            valid = obj_valid.view(B, K, 1, 1).float()
            denom = (valid.sum() * n * num_heads).clamp_min(1.0)
            other_mass = (attn_probs * forbidden[:, :, None, None, :]).sum(dim=-1)
            self_mass = (attn_probs * masks[:, :, None, None, :]).sum(dim=-1)
            neutral_mass = (attn_probs * neutral[:, :, None, None, :]).sum(dim=-1)

            layer_loss = (other_mass * valid).sum() / denom
            layer_self = (self_mass * valid).sum() / denom
            layer_other = (other_mass * valid).sum() / denom
            layer_neutral = (neutral_mass * valid).sum() / denom
            if not torch.isfinite(layer_loss):
                continue

            loss_sum = loss_sum + layer_loss
            self_sum = self_sum + layer_self
            other_sum = other_sum + layer_other
            neutral_sum = neutral_sum + layer_neutral
            valid_layer_count += 1

        if valid_layer_count <= 0:
            z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
            return {"loss": z, "self_mass": z, "other_mass": z, "neutral_mass": z}

        scale = float(valid_layer_count)
        return {
            "loss": loss_sum / scale,
            "self_mass": (self_sum / scale).detach(),
            "other_mass": (other_sum / scale).detach(),
            "neutral_mass": (neutral_sum / scale).detach(),
        }

    def _compute_llm_qk_attention_maps(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        layers_spec: str,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Average selected-layer LLM Q/K OVT->image attention maps for viz."""
        if hidden_states is None:
            return None
        B, M = ovt_valid_mask.shape
        P = int(positions["img_e"] - positions["img_s"])
        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if len(layers) == 0 or P <= 0:
            return None

        temp = max(float(temperature), 1e-6)
        acc = None
        count = 0
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            layer_input = hidden_states[layer_idx]
            block = self.model.layers[layer_idx]
            attn = block.self_attn
            attn_input = block.input_layernorm(layer_input)
            img_input = attn_input[:, positions["img_s"]:positions["img_e"], :]
            ovt_input = gather_ovt_hidden_states(attn_input, ovt_abs_positions, ovt_valid_mask)

            q_proj = attn.q_proj(ovt_input)
            k_proj = attn.k_proj(img_input)
            num_heads = int(getattr(attn, "num_heads", getattr(self.config, "num_attention_heads", 1)))
            num_kv_heads = int(
                getattr(attn, "num_key_value_heads", getattr(self.config, "num_key_value_heads", num_heads))
            )
            head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // max(num_heads, 1)))
            if num_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
                continue

            q = q_proj.reshape(B, M, num_heads, head_dim)
            k_img = k_proj.reshape(B, P, num_kv_heads, head_dim)
            if num_kv_heads != num_heads:
                repeat = max(num_heads // num_kv_heads, 1)
                k_img = k_img.repeat_interleave(repeat, dim=2)
                if k_img.shape[2] < num_heads:
                    pad = num_heads - k_img.shape[2]
                    k_img = torch.cat([k_img, k_img[:, :, -1:, :].expand(-1, -1, pad, -1)], dim=2)
                elif k_img.shape[2] > num_heads:
                    k_img = k_img[:, :, :num_heads]

            scores = torch.einsum("bmhd,bphd->bmhp", q.float(), k_img.float())
            scores = scores / math.sqrt(float(head_dim))
            probs = F.softmax(scores / temp, dim=-1).mean(dim=2)
            probs = probs * ovt_valid_mask.unsqueeze(-1).float()
            acc = probs if acc is None else acc + probs
            count += 1

        if count <= 0 or acc is None:
            return None
        return (acc / float(count)).detach()

    # ------------------------------------------------------------------
    # PGOT forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def pgot_get_rae_hidden(
        self,
        images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        target_images: torch.Tensor,
        ovt_positions_in_caption: Optional[torch.Tensor] = None,
        ovt_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run the LLM forward (no loss) and return rae_hidden + gt_siglip.

        Used for eval-time image reconstruction sampling. Mirrors the first 7
        steps of `_forward_pgot` (sections 1–7) but stops before any loss compute.
        """
        model_device = self.pgot_register_embeddings.device
        images = images.to(model_device)
        target_images = target_images.to(model_device)
        caption_input_ids = caption_input_ids.to(model_device)
        caption_attention_mask = caption_attention_mask.to(model_device, dtype=torch.bool)
        if ovt_positions_in_caption is not None:
            ovt_positions_in_caption = ovt_positions_in_caption.to(model_device, dtype=torch.long)
        if ovt_valid_mask is not None:
            ovt_valid_mask = ovt_valid_mask.to(model_device, dtype=torch.bool)
        B, caption_len = caption_input_ids.shape

        _, img_features, gt_siglip = self._encode_images_aurora(images, target_images=target_images)
        if gt_siglip is None:
            raise ValueError("PGOT recon sampling requires target_images.")
        dtype = self._aurora_model_dtype()

        sys_p = self._pgot_embed_frozen_tokens(self.pgot_system_prefix_ids, B, model_device, dtype)
        sys_s = self._pgot_embed_frozen_tokens(self.pgot_system_suffix_ids, B, model_device, dtype)
        user_p = self._pgot_embed_frozen_tokens(self.pgot_user_prefix_ids, B, model_device, dtype)
        user_s = self._pgot_embed_frozen_tokens(self.pgot_user_suffix_ids, B, model_device, dtype)
        asst_p = self._pgot_embed_frozen_tokens(self.pgot_assistant_prefix_ids, B, model_device, dtype)
        asst_s = self._pgot_embed_frozen_tokens(self.pgot_assistant_suffix_ids, B, model_device, dtype)
        caption_embeds = self._pgot_embed_caption(caption_input_ids, model_device, dtype)
        null_bg_embeds = self._pgot_embed_null_bg(B, model_device, dtype)
        register_embeds = self.pgot_register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
        n_rae = self.get_model().latent_queries.shape[0]
        rae_embeds = self.get_model().latent_queries.unsqueeze(0).expand(B, -1, -1).to(
            device=model_device, dtype=dtype
        )

        positions = pgot_positions(
            caption_len=caption_len,
            system_prefix_len=sys_p.shape[1],
            system_suffix_len=sys_s.shape[1],
            user_prefix_len=user_p.shape[1],
            user_suffix_len=user_s.shape[1],
            assistant_prefix_len=asst_p.shape[1],
            assistant_suffix_len=asst_s.shape[1],
            num_image_tokens=img_features.shape[1],
            n_register=self.pgot_n_register,
            n_rae_query=n_rae,
            n_null_bg=null_bg_embeds.shape[1],
        )

        cap_s_pos = positions["cap_s"]
        ovt_abs_positions = (
            cap_s_pos + ovt_positions_in_caption
            if ovt_positions_in_caption is not None
            else None
        )

        inputs_embeds = torch.cat(
            [
                sys_p, sys_s,
                user_p, img_features.to(dtype=dtype), user_s,
                asst_p, caption_embeds, asst_s,
                null_bg_embeds,
                register_embeds,
                rae_embeds,
            ],
            dim=1,
        )
        attn_bias = build_pgot_attention_mask(
            positions=positions,
            caption_padding_mask=caption_attention_mask,
            device=model_device,
            dtype=inputs_embeds.dtype,
            rae_bidirectional=bool(getattr(self.config, "pgot_rae_bidirectional", False)),
            rae_attends_caption=bool(getattr(self.config, "pgot_rae_attends_caption", False)),
            ovt_absolute_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
        )
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        rae_hidden = hidden[:, positions["rae_s"]:positions["rae_e"], :]
        return {"rae_hidden": rae_hidden, "gt_siglip": gt_siglip}

    @torch.no_grad()
    def pgot_sample_recon_latents(
        self,
        images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        target_images: torch.Tensor,
        ovt_positions_in_caption: Optional[torch.Tensor] = None,
        ovt_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Get rae_hidden, project to DiT cond, run rectified-flow inference.

        Returns dict with:
          - 'pred_latent': (B, T, D) sampled latents from diff_head.infer
          - 'gt_siglip':   (B, T, D) target siglip features (for sanity decoding)
        """
        feats = self.pgot_get_rae_hidden(
            images=images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            target_images=target_images,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
        )
        rae_hidden = feats["rae_hidden"]
        cond = self._captionslot_prepare_diffusion_condition(rae_hidden).float()
        self.diff_head = self.diff_head.to(cond.device)
        pred_latent = self.diff_head.infer(z=cond)
        return {"pred_latent": pred_latent, "gt_siglip": feats["gt_siglip"]}

    def _forward_pgot(
        self,
        images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        caption_labels: Optional[torch.LongTensor],
        ovt_positions_in_caption: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        ovt_is_thing: Optional[torch.Tensor],
        gt_masks_per_ovt: torch.Tensor,
        target_images: torch.Tensor,
        pgot_contrastive_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        model_device = self.pgot_register_embeddings.device
        images = images.to(model_device)
        target_images = target_images.to(model_device)
        caption_input_ids = caption_input_ids.to(model_device)
        caption_attention_mask = caption_attention_mask.to(model_device, dtype=torch.bool)
        ovt_positions_in_caption = ovt_positions_in_caption.to(model_device, dtype=torch.long)
        ovt_valid_mask = ovt_valid_mask.to(model_device, dtype=torch.bool)
        if ovt_is_thing is None:
            ovt_is_thing = ovt_valid_mask
        else:
            ovt_is_thing = ovt_is_thing.to(model_device, dtype=torch.bool)
        gt_masks_per_ovt = gt_masks_per_ovt.to(model_device).float()
        if caption_labels is not None:
            caption_labels = caption_labels.to(model_device, dtype=torch.long)

        B, caption_len = caption_input_ids.shape

        # 1) Encode image (AURORA helper returns: feature for LLM, target for diffusion)
        _, img_features, gt_siglip = self._encode_images_aurora(images, target_images=target_images)
        if gt_siglip is None:
            raise ValueError("PGOT training requires target_images.")
        dtype = self._aurora_model_dtype()

        # 2) Embed template + caption
        sys_p = self._pgot_embed_frozen_tokens(self.pgot_system_prefix_ids, B, model_device, dtype)
        sys_s = self._pgot_embed_frozen_tokens(self.pgot_system_suffix_ids, B, model_device, dtype)
        user_p = self._pgot_embed_frozen_tokens(self.pgot_user_prefix_ids, B, model_device, dtype)
        user_s = self._pgot_embed_frozen_tokens(self.pgot_user_suffix_ids, B, model_device, dtype)
        asst_p = self._pgot_embed_frozen_tokens(self.pgot_assistant_prefix_ids, B, model_device, dtype)
        asst_s = self._pgot_embed_frozen_tokens(self.pgot_assistant_suffix_ids, B, model_device, dtype)
        caption_embeds = self._pgot_embed_caption(caption_input_ids, model_device, dtype)
        null_bg_embeds = self._pgot_embed_null_bg(B, model_device, dtype)
        register_embeds = self.pgot_register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype=dtype)
        n_rae = self.get_model().latent_queries.shape[0]
        rae_embeds = self.get_model().latent_queries.unsqueeze(0).expand(B, -1, -1).to(
            device=model_device, dtype=dtype
        )

        # 3) Compute positions
        positions = pgot_positions(
            caption_len=caption_len,
            system_prefix_len=sys_p.shape[1],
            system_suffix_len=sys_s.shape[1],
            user_prefix_len=user_p.shape[1],
            user_suffix_len=user_s.shape[1],
            assistant_prefix_len=asst_p.shape[1],
            assistant_suffix_len=asst_s.shape[1],
            num_image_tokens=img_features.shape[1],
            n_register=self.pgot_n_register,
            n_rae_query=n_rae,
            n_null_bg=null_bg_embeds.shape[1],
        )

        # 4) Concat sequence
        inputs_embeds = torch.cat(
            [
                sys_p, sys_s,
                user_p, img_features.to(dtype=dtype), user_s,
                asst_p, caption_embeds, asst_s,
                null_bg_embeds,
                register_embeds,
                rae_embeds,
            ],
            dim=1,
        )

        # 5) Attention bias — rae_query attends ONLY [OVT positions inside caption + register + self]
        # Caption text (non-OVT) and Image are both BLOCKED from rae_query, making OVT
        # the unique bottleneck for reconstruction (enables clean ovt-swap editing).
        cap_s_pos = positions["cap_s"]
        ovt_abs_positions = cap_s_pos + ovt_positions_in_caption  # (B, M_max)
        attn_bias = build_pgot_attention_mask(
            positions=positions,
            caption_padding_mask=caption_attention_mask,
            device=model_device,
            dtype=inputs_embeds.dtype,
            rae_bidirectional=bool(getattr(self.config, "pgot_rae_bidirectional", False)),
            rae_attends_caption=bool(getattr(self.config, "pgot_rae_attends_caption", False)),
            ovt_absolute_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
        )

        # 6) LLM forward
        mask_llm_qk_outside_w = float(getattr(self.config, "pgot_mask_llm_qk_outside_weight", 0.0))
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            use_cache=False,
            output_hidden_states=mask_llm_qk_outside_w > 0.0,
            return_dict=True,
        )
        hidden = out.last_hidden_state  # (B, L, D)

        # 7) Slice out chunks of interest
        img_hidden = hidden[:, positions["img_s"]:positions["img_e"], :]
        rae_hidden = hidden[:, positions["rae_s"]:positions["rae_e"], :]
        ovt_hidden = gather_ovt_hidden_states(hidden, ovt_abs_positions, ovt_valid_mask)

        # 8) L1: LM next-token CE on the caption span
        loss_lm = self._compute_lm_loss(
            hidden=hidden,
            positions=positions,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            caption_labels=caption_labels,
        )

        # 9) L2: per-OVT mask supervision.
        #   CODA-style per-patch softmax CE is the primary signal (exclusive
        #   competition over thing + stuff OVTs -> good fARI). BCE/Tversky are
        #   kept available but default to 0 weight.
        attn_temp = float(getattr(self.config, "pgot_attention_temperature", 1.0))
        attn_ln = bool(getattr(self.config, "pgot_attention_use_layer_norm", True))
        ovt_logits = compute_per_ovt_mask_logits(
            ovt_hidden=ovt_hidden,
            img_hidden=img_hidden,
            temperature=attn_temp,
            normalize_tokens=attn_ln,
        )
        # Register-vs-patch logits = the BACKGROUND class for the competition CE.
        register_hidden = hidden[:, positions["reg_s"]:positions["reg_e"], :]
        reg_logits = compute_per_ovt_mask_logits(
            ovt_hidden=register_hidden,
            img_hidden=img_hidden,
            temperature=attn_temp,
            normalize_tokens=attn_ln,
        )
        null_bg_hidden = hidden[:, positions["null_bg_s"]:positions["null_bg_e"], :]
        null_bg_logits = None
        if null_bg_hidden.shape[1] > 0:
            null_bg_logits = compute_per_ovt_mask_logits(
                ovt_hidden=null_bg_hidden,
                img_hidden=img_hidden,
                temperature=attn_temp,
                normalize_tokens=attn_ln,
            )
        ce_temp = float(getattr(self.config, "pgot_mask_ce_temperature", 1.0))
        use_null_bg = bool(getattr(self.config, "pgot_use_null_bg_competition", False))
        mask_ce_w = float(getattr(self.config, "pgot_mask_ce_weight", 1.0))
        mask_fg_w = float(getattr(self.config, "pgot_mask_fg_weight", 0.0))
        mask_outside_w = float(getattr(self.config, "pgot_mask_outside_weight", 0.0))
        mask_ce_aux_w = float(getattr(self.config, "pgot_mask_aux_competition_weight", 0.0))
        mask_bce_w = float(getattr(self.config, "pgot_mask_bce_weight", 0.0))
        mask_tversky_w = float(getattr(self.config, "pgot_mask_tversky_weight", 0.0))
        mask_spatial_outside_w = float(getattr(self.config, "pgot_mask_spatial_outside_weight", 0.0))
        mask_spatial_outside_log_w = float(getattr(self.config, "pgot_mask_spatial_outside_log_weight", 0.0))

        zero = ovt_logits.new_zeros(())
        loss_mask_ce = zero
        loss_mask_fg = ovt_logits.new_zeros(())
        loss_mask_outside = ovt_logits.new_zeros(())
        null_bg_prob_on_fg = ovt_logits.new_zeros(())
        thing_prob_on_bg = ovt_logits.new_zeros(())
        thing_objects_mean = ovt_logits.new_zeros(())
        need_owner_loss = (mask_ce_w > 0.0) or (mask_fg_w > 0.0) or (mask_outside_w > 0.0)
        if use_null_bg and need_owner_loss:
            if null_bg_logits is None:
                raise ValueError("pgot_use_null_bg_competition=True requires pgot_n_null_bg > 0.")
            null_losses = compute_null_bg_competition_losses(
                ovt_logits=ovt_logits,
                null_bg_logits=null_bg_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                ovt_is_thing=ovt_is_thing,
                n_ovt_per_object=self.pgot_n_ovt_per_object,
                temperature=ce_temp,
            )
            loss_mask_ce = null_losses["loss_owner"]
            loss_mask_fg = null_losses["loss_fg"]
            loss_mask_outside = null_losses["loss_outside"]
            null_bg_prob_on_fg = null_losses["bg_prob_on_fg"]
            thing_prob_on_bg = null_losses["thing_prob_on_bg"]
            thing_objects_mean = null_losses["thing_object_count"]
        elif (not use_null_bg) and mask_ce_w > 0.0:
            # Object-level competition CE over {K objects, register-background}.
            loss_mask_ce = compute_competition_ce_loss(
                ovt_logits=ovt_logits,
                reg_logits=reg_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                n_ovt_per_object=self.pgot_n_ovt_per_object,
                temperature=ce_temp,
            )

        # ───── v6a: auxiliary competition CE on last-layer Q/K projections ────
        # Diagnosis of v5: competition only at the *loss* (CE) on top of
        # LN(OVT)·LN(patch). LLM attention itself stays standard (softmax over
        # keys), so OVT_hidden never develops representational exclusivity. v6a
        # re-uses the LLM's own last attention layer Q/K projections (the same
        # matrices that compute the model's actual attention) to form an
        # alternative score matrix, then puts a competitive CE on it. Gradient
        # flows back into q_proj/k_proj LoRA -> the LLM's attention itself is
        # nudged toward competition-friendly projections.
        loss_mask_ce_aux = loss_mask_ce.new_zeros(())
        if mask_ce_aux_w > 0.0 and not use_null_bg:
            try:
                last_attn = self.model.layers[-1].self_attn
                head_dim = int(getattr(last_attn, "head_dim", ovt_hidden.shape[-1] // 16))
                # Take Q-head 0 / K-head 0 of the last attention layer
                # (avoids GQA expansion + multi-head averaging; single-head signal).
                Q_ovt_a = last_attn.q_proj(ovt_hidden)[..., :head_dim]
                K_img_a = last_attn.k_proj(img_hidden)[..., :head_dim]
                Q_reg_a = last_attn.q_proj(register_hidden)[..., :head_dim]
                scale = math.sqrt(float(head_dim))
                aux_ovt_logits = torch.einsum("bmd,bpd->bmp", Q_ovt_a.float(), K_img_a.float()) / scale
                aux_reg_logits = torch.einsum("brd,bpd->brp", Q_reg_a.float(), K_img_a.float()) / scale
                loss_mask_ce_aux = compute_competition_ce_loss(
                    ovt_logits=aux_ovt_logits,
                    reg_logits=aux_reg_logits,
                    gt_masks_per_ovt=gt_masks_per_ovt,
                    ovt_valid_mask=ovt_valid_mask,
                    n_ovt_per_object=self.pgot_n_ovt_per_object,
                    temperature=ce_temp,
                )
            except Exception as _e:
                # Fail-soft: if anything goes wrong (model structure mismatch,
                # head_dim missing, dtype issue), drop aux silently this step.
                loss_mask_ce_aux = loss_mask_ce.new_zeros(())

        loss_mask_bce = zero
        if mask_bce_w > 0.0:
            loss_mask_bce = compute_mask_bce_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
            )

        loss_mask_spatial_outside = zero
        spatial_self_mass = zero
        spatial_other_mass = zero
        spatial_neutral_mass = zero
        if mask_spatial_outside_w > 0.0:
            spatial_temp = float(getattr(self.config, "pgot_mask_spatial_temperature", 1.0))
            spatial_losses = compute_spatial_outside_attention_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                n_ovt_per_object=self.pgot_n_ovt_per_object,
                temperature=spatial_temp,
            )
            loss_mask_spatial_outside = spatial_losses["loss"]
            spatial_self_mass = spatial_losses["self_mass"]
            spatial_other_mass = spatial_losses["other_mass"]
            spatial_neutral_mass = spatial_losses["neutral_mass"]

        loss_mask_spatial_outside_log = zero
        spatial_log_self_mass = zero
        spatial_log_outside_mass = zero
        spatial_log_mean = zero
        if mask_spatial_outside_log_w > 0.0:
            spatial_log_temp = float(getattr(self.config, "pgot_mask_spatial_outside_log_temperature", 1.0))
            spatial_log_losses = compute_spatial_outside_log_attention_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                temperature=spatial_log_temp,
            )
            loss_mask_spatial_outside_log = spatial_log_losses["loss"]
            spatial_log_self_mass = spatial_log_losses["self_mass"]
            spatial_log_outside_mass = spatial_log_losses["outside_mass"]
            spatial_log_mean = spatial_log_losses["outside_log_mean"]

        loss_mask_llm_qk_outside = zero
        llm_qk_self_mass = zero
        llm_qk_other_mass = zero
        llm_qk_neutral_mass = zero
        if mask_llm_qk_outside_w > 0.0:
            llm_qk_temp = float(getattr(self.config, "pgot_mask_llm_qk_outside_temperature", 1.0))
            llm_qk_layers = str(getattr(self.config, "pgot_mask_llm_qk_outside_layers", "last4"))
            llm_qk_losses = self._compute_llm_qk_outside_attention_loss(
                hidden_states=out.hidden_states,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=llm_qk_layers,
                temperature=llm_qk_temp,
            )
            loss_mask_llm_qk_outside = llm_qk_losses["loss"]
            llm_qk_self_mass = llm_qk_losses["self_mass"]
            llm_qk_other_mass = llm_qk_losses["other_mass"]
            llm_qk_neutral_mass = llm_qk_losses["neutral_mass"]

        loss_mask_tversky = zero
        if mask_tversky_w > 0.0:
            tversky_alpha = float(getattr(self.config, "pgot_mask_tversky_alpha", 0.5))
            tversky_beta = float(getattr(self.config, "pgot_mask_tversky_beta", 0.5))
            loss_mask_tversky = compute_mask_tversky_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                alpha=tversky_alpha,
                beta=tversky_beta,
            )

        # 10) L3: rectified-flow reconstruction (rae_hidden -> diff_head) with CFG dropping
        cfg_drop_rate = float(getattr(self.config, "pgot_cfg_drop_rate", 0.0))
        rae_hidden_for_diff = rae_hidden
        if self.training and cfg_drop_rate > 0.0:
            # Per-sample independent Bernoulli drop. When dropped, rae_hidden -> zeros
            # so the diffusion head learns an unconditional path (used for CFG at inference).
            drop_mask = (torch.rand(B, device=rae_hidden.device) < cfg_drop_rate).view(B, 1, 1)
            rae_hidden_for_diff = rae_hidden * (~drop_mask).to(rae_hidden.dtype)
        loss_recon = self._captionslot_compute_diffusion_loss(
            hidden=rae_hidden_for_diff,
            target_features=gt_siglip,
            slot_context=None,
            slot_mask=None,
        )

        # 11) L4: OVT-swap contrastive (CODA-style, last-layer re-run for editing alignment)
        loss_contrastive = loss_recon.new_zeros(())
        contrastive_w = float(getattr(self.config, "pgot_contrastive_loss_weight", 0.0))
        if pgot_contrastive_weight is not None:
            contrastive_w = float(pgot_contrastive_weight)
        if contrastive_w > 0.0 and B >= 2:
            loss_contrastive = self._compute_ovt_swap_contrastive(
                hidden=hidden,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                attn_bias=attn_bias,
                gt_siglip=gt_siglip,
                pos_recon_error=loss_recon,   # detach inside _compute
            )

        # 12) Combine
        lm_w = float(getattr(self.config, "pgot_lm_loss_weight", 1.0))
        recon_w = float(getattr(self.config, "pgot_recon_loss_weight", 1.0))

        loss_mask = (
            mask_ce_w * loss_mask_ce
            + mask_fg_w * loss_mask_fg
            + mask_outside_w * loss_mask_outside
            + mask_ce_aux_w * loss_mask_ce_aux
            + mask_bce_w * loss_mask_bce
            + mask_tversky_w * loss_mask_tversky
            + mask_spatial_outside_w * loss_mask_spatial_outside
            + mask_spatial_outside_log_w * loss_mask_spatial_outside_log
            + mask_llm_qk_outside_w * loss_mask_llm_qk_outside
        )
        total_loss = (
            lm_w * loss_lm
            + loss_mask
            + recon_w * loss_recon
            + contrastive_w * loss_contrastive
        )

        if not torch.isfinite(total_loss).all():
            total_loss = torch.nan_to_num(total_loss, nan=1e4, posinf=1e4, neginf=1e4)

        # Bookkeeping for trainer logging
        self.pgot_loss_lm = loss_lm.detach()
        self.pgot_loss_mask = loss_mask.detach()
        self.pgot_loss_recon = loss_recon.detach()
        self.pgot_loss_contrastive = loss_contrastive.detach() if contrastive_w > 0.0 else None
        self.pgot_n_objects_mean = ovt_valid_mask.float().sum(dim=-1).mean().detach() / max(
            self.pgot_n_ovt_per_object, 1
        )
        details = {}
        if mask_ce_w > 0.0:
            details["loss_mask_ce"] = loss_mask_ce.detach()
        if mask_fg_w > 0.0:
            details["loss_mask_fg"] = loss_mask_fg.detach()
        if mask_outside_w > 0.0:
            details["loss_mask_outside"] = loss_mask_outside.detach()
        if mask_ce_aux_w > 0.0:
            details["loss_mask_ce_aux"] = loss_mask_ce_aux.detach()
        if mask_bce_w > 0.0:
            details["loss_mask_bce"] = loss_mask_bce.detach()
        if mask_tversky_w > 0.0:
            details["loss_mask_tversky"] = loss_mask_tversky.detach()
        if mask_spatial_outside_w > 0.0:
            details["loss_mask_spatial_outside"] = loss_mask_spatial_outside.detach()
            details["spatial_self_mass"] = spatial_self_mass.detach()
            details["spatial_other_mass"] = spatial_other_mass.detach()
            details["spatial_neutral_mass"] = spatial_neutral_mass.detach()
        if mask_spatial_outside_log_w > 0.0:
            details["loss_mask_spatial_outside_log"] = loss_mask_spatial_outside_log.detach()
            details["spatial_log_self_mass"] = spatial_log_self_mass.detach()
            details["spatial_log_outside_mass"] = spatial_log_outside_mass.detach()
            details["spatial_log_mean"] = spatial_log_mean.detach()
        if mask_llm_qk_outside_w > 0.0:
            details["loss_mask_llm_qk_outside"] = loss_mask_llm_qk_outside.detach()
            details["llm_qk_self_mass"] = llm_qk_self_mass.detach()
            details["llm_qk_other_mass"] = llm_qk_other_mass.detach()
            details["llm_qk_neutral_mass"] = llm_qk_neutral_mass.detach()
        if use_null_bg and need_owner_loss:
            details["null_bg_prob_on_fg"] = null_bg_prob_on_fg.detach()
            details["thing_prob_on_bg"] = thing_prob_on_bg.detach()
            details["thing_objects_mean"] = thing_objects_mean.detach()
        self.pgot_loss_details = details

        info = {
            "loss_lm": loss_lm.detach(),
            "loss_mask": loss_mask.detach(),
            "loss_recon": loss_recon.detach(),
            "loss_contrastive": loss_contrastive.detach(),
        }
        info.update(details)
        return total_loss, info

    # ------------------------------------------------------------------
    # OVT-swap contrastive (Q3-style editing-aligned negative)
    # ------------------------------------------------------------------
    def _compute_ovt_swap_contrastive(
        self,
        hidden: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        attn_bias: torch.Tensor,
        gt_siglip: torch.Tensor,
        pos_recon_error: torch.Tensor,
    ) -> torch.Tensor:
        """OVT-only swap: keep caption + image + register fixed, replace OVT hidden
        with a shifted batch sample, re-run the LLM's LAST layer (so rae_query
        sees the swapped OVT), then ask diff_head to reconstruct the ORIGINAL image
        — it should FAIL (high error).

        Loss is a BOUNDED HINGE so optimization stays stable:
            loss_neg = relu(pos_error.detach() + margin - error_neg)
        - If error_neg already exceeds pos_error + margin -> loss_neg = 0 (saturated).
        - Otherwise pushes error_neg upward, but loss_neg is non-negative and bounded
          by `(pos_error + margin)`, so total loss cannot run to -inf.

        Training/inference path are consistent: at editing time we follow the
        same recipe (swap a target OVT, re-run last layer, decode via DiT).
        """
        B = hidden.shape[0]
        if B < 2:
            return hidden.new_zeros(())

        shift = B // 2
        shifted_idx = torch.cat([
            torch.arange(shift, B, device=hidden.device),
            torch.arange(0, shift, device=hidden.device),
        ])

        # Take OVT hidden values from positive pass at their absolute positions
        D = hidden.shape[-1]
        ovt_hidden_pos = hidden.gather(
            dim=1,
            index=ovt_abs_positions.clamp(0, hidden.shape[1] - 1).unsqueeze(-1).expand(-1, -1, D),
        )  # (B, M, D)
        ovt_hidden_neg = ovt_hidden_pos[shifted_idx]  # swap with mate
        # Mask invalid positions
        valid_pos = ovt_valid_mask.unsqueeze(-1).to(ovt_hidden_neg.dtype)
        valid_shifted = ovt_valid_mask[shifted_idx].unsqueeze(-1).to(ovt_hidden_neg.dtype)
        # Effective swap only where BOTH sides are valid
        swap_gate = valid_pos * valid_shifted
        ovt_hidden_neg = swap_gate * ovt_hidden_neg + (1 - swap_gate) * ovt_hidden_pos

        # Build hidden_mixed: positive hidden with OVT positions overwritten
        hidden_mixed = hidden.clone()
        scatter_idx = ovt_abs_positions.clamp(0, hidden.shape[1] - 1).unsqueeze(-1).expand(-1, -1, D)
        hidden_mixed = hidden_mixed.scatter(dim=1, index=scatter_idx, src=ovt_hidden_neg)

        # Re-run only the LAST decoder layer with the mixed hidden so rae_query
        # attention reflects the swap. Other layers are untouched.
        try:
            last_layer = self.model.layers[-1]
        except AttributeError:
            return hidden.new_zeros(())

        # Build position_ids for the layer
        L = hidden_mixed.shape[1]
        position_ids = torch.arange(L, device=hidden_mixed.device).unsqueeze(0).expand(B, -1)
        try:
            layer_out = last_layer(
                hidden_mixed,
                attention_mask=attn_bias,
                position_ids=position_ids,
                use_cache=False,
            )
            updated = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        except TypeError:
            # Fallback for layers with simpler signature
            updated = last_layer(hidden_mixed, attention_mask=attn_bias)[0]

        rae_hidden_neg = updated[:, positions["rae_s"]:positions["rae_e"], :]
        error_neg = self._captionslot_compute_diffusion_loss(
            hidden=rae_hidden_neg,
            target_features=gt_siglip,
            slot_context=None,
            slot_mask=None,
        )
        # Sanity clamp: cap error_neg to a finite range to defuse any blowup.
        error_neg = torch.nan_to_num(error_neg, nan=10.0, posinf=10.0, neginf=0.0).clamp(min=0.0, max=20.0)

        # Bounded hinge: push error_neg above (pos_error + margin), stop after.
        margin = float(getattr(self.config, "pgot_contrastive_margin", 1.0))
        pos_target = pos_recon_error.detach().clamp(min=0.0, max=10.0) + margin
        loss_hinge = F.relu(pos_target - error_neg)   # ∈ [0, pos_target] ⊂ [0, 11]
        return loss_hinge

    # ------------------------------------------------------------------
    # LM loss (next-token CE on caption span)
    # ------------------------------------------------------------------
    def _compute_lm_loss(
        self,
        hidden: torch.Tensor,
        positions: Dict[str, int],
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        caption_labels: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Next-token cross-entropy over assistant caption tokens.

        We supervise positions [cap_s : cap_e] of the LLM hidden states so that
        predicted token at position i matches caption_input_ids[i+1].
        """
        cap_s = positions["cap_s"]
        cap_e = positions["cap_e"]
        # Logits from the lm_head over caption hidden states (causal LM head)
        cap_hidden = hidden[:, cap_s:cap_e, :]
        logits = self.lm_head(cap_hidden.float())  # (B, T, V)

        if caption_labels is None:
            caption_labels = caption_input_ids.clone()
            caption_labels = caption_labels.masked_fill(~caption_attention_mask.bool(), -100)

        # Shift: logits[:, :-1] predicts ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = caption_labels[:, 1:].contiguous()
        # Mask padded positions as -100
        if caption_attention_mask is not None:
            pad_mask = caption_attention_mask[:, 1:].bool()
            shift_labels = shift_labels.masked_fill(~pad_mask, -100)

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        return loss
