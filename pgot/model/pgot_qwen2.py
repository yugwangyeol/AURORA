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
    compute_object_balanced_mask_bce_loss,
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


def _resolve_layer_spec(spec: str, n_layers: int) -> List[int]:
    """Resolve a layer spec such as 'last4', 'all', '20:28', or '20,24,27'."""
    if n_layers <= 0:
        return []
    spec = (spec or "last4").strip().lower()
    if spec in {"all", "*"}:
        return list(range(n_layers))
    if spec.startswith("last"):
        suffix = spec[4:]
        k = int(suffix) if suffix else 4
        return list(range(max(0, n_layers - max(k, 1)), n_layers))

    layers: List[int] = []
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

    deduped: List[int] = []
    seen = set()
    for idx in layers:
        if idx not in seen:
            deduped.append(idx)
            seen.add(idx)
    return deduped


class PGOTOVTOwnerHead(nn.Module):
    """OVT-owner logits over image patches."""

    def __init__(self, dim: int, temperature: float = 1.0):
        super().__init__()
        # Legacy parameter name kept for checkpoint compatibility.
        self.slot_ln = nn.LayerNorm(dim)
        self.img_ln = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.temperature = float(temperature)

    def forward(
        self,
        ovt_states: torch.Tensor,
        img_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = max(
            float(self.temperature if temperature is None else temperature),
            1e-6,
        )
        q = self.q_proj(self.slot_ln(ovt_states)).float()
        k = self.k_proj(self.img_ln(img_hidden)).float()
        logits = torch.einsum("bsd,bpd->bsp", q, k) / math.sqrt(float(q.shape[-1]))
        logits = logits / temp
        neg = -1e4
        logits = logits.masked_fill(~ovt_valid_mask.unsqueeze(-1), neg)
        probs = F.softmax(logits, dim=1)
        probs = probs * ovt_valid_mask.unsqueeze(-1).float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return logits, probs


class PGOTOVTUpdateBlock(nn.Module):
    """OVT update block with owner competition and gated residual updates.

    The GRUCell was replaced by a LayerNorm'd linear fusion of (OVT, aggregated
    patch update) because the fused GRUCell kernel produced NaN from finite inputs
    on this setup. ReZero gates (g1, g2) init at 0 so the block is an EXACT
    identity at step 0 and turns on gradually.
    """

    def __init__(self, dim: int, temperature: float = 1.0, mlp_ratio: int = 4):
        super().__init__()
        self.owner_head = PGOTOVTOwnerHead(dim=dim, temperature=temperature)
        self.value_ln = nn.LayerNorm(dim)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        # Legacy parameter name kept for checkpoint compatibility.
        self.slot_ln = nn.LayerNorm(dim)
        self.upd_ln = nn.LayerNorm(dim)
        self.fuse = nn.Linear(2 * dim, dim, bias=False)
        self.mlp_ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        # ReZero gates: init 0 -> exact identity at step 0, turns on gradually.
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        ovt_states: torch.Tensor,
        img_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        owner_logits, owner_probs = self.owner_head(
            ovt_states=ovt_states,
            img_hidden=img_hidden,
            ovt_valid_mask=ovt_valid_mask,
        )

        # Owner softmax over OVTs first, then per-OVT renormalization over
        # image patches for the update.
        weights = owner_probs / owner_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        values = self.v_proj(self.value_ln(img_hidden))
        updates = torch.einsum("bsp,bpd->bsd", weights, values)

        # Gated residual update (numerically stable, identity at init).
        fused = self.fuse(
            torch.cat([self.slot_ln(ovt_states), self.upd_ln(updates)], dim=-1)
        )
        o_mid = ovt_states + self.g1 * fused
        o_new = o_mid + self.g2 * self.mlp(self.mlp_ln(o_mid))
        o_new = torch.where(torch.isfinite(o_new), o_new, ovt_states)
        updated = torch.where(
            ovt_valid_mask.unsqueeze(-1),
            o_new,
            ovt_states,
        )
        return updated, owner_logits, owner_probs


class PGOTQwen2ForCausalLM(ScaleRAEQwenForCausalLM):
    """PGOT model: caption-grounded variable-K object tokens."""

    config_class = PGOTQwen2Config

    def __init__(self, config):
        super().__init__(config)
        # PGOT-specific init runs LAST so register/template ids are set up.
        if bool(getattr(config, "use_pgot", False)):
            self._init_pgot()

    def _pgot_enforce_stable_train_modes(self) -> None:
        """Keep frozen/high-risk submodules in eval mode during V12 training."""
        if not self.training or not bool(getattr(self.config, "pgot_v12_enable", False)):
            return

        # Eval mode removes train-time stochasticity but still allows gradients
        # on LoRA / PGOT parameters.
        self.model.eval()

        try:
            vt_list = self.get_vision_tower_aux_list()
        except Exception:
            vt_list = None
        if vt_list is not None:
            for vt in vt_list:
                if vt is not None:
                    vt.eval()

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

        self.pgot_v12_enable = bool(getattr(self.config, "pgot_v12_enable", False))
        self.pgot_v12_layers = _resolve_layer_spec(
            str(getattr(self.config, "pgot_v12_layers", "12,16,20,24")),
            int(getattr(self.config, "num_hidden_layers", 0)),
        )
        if self.pgot_v12_enable:
            ovt_temp = float(
                getattr(
                    self.config,
                    "pgot_v12_ovt_temperature",
                    getattr(self.config, "pgot_v12_slot_temperature", 1.0),
                )
            )
            owner_temp = float(getattr(self.config, "pgot_v12_owner_temperature", 1.0))
            # Legacy attribute name kept for checkpoint compatibility.
            self.pgot_v12_slot_update_blocks = nn.ModuleList(
                [
                    PGOTOVTUpdateBlock(dim=D, temperature=ovt_temp)
                    for _ in self.pgot_v12_layers
                ]
            )
            self.pgot_v12_owner_head = PGOTOVTOwnerHead(dim=D, temperature=owner_temp)
        else:
            self.pgot_v12_slot_update_blocks = nn.ModuleList()
            self.pgot_v12_owner_head = None

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
            f"N_null_bg={self.pgot_n_null_bg}, "
            f"v12={self.pgot_v12_enable}, v12_layers={self.pgot_v12_layers}"
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
        self._pgot_enforce_stable_train_modes()
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

    def _pgot_model_device(self) -> torch.device:
        if hasattr(self, "pgot_register_embeddings"):
            return self.pgot_register_embeddings.device
        return next(self.parameters()).device

    def _pgot_build_sequence_inputs(
        self,
        *,
        images: torch.Tensor,
        target_images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        ovt_positions_in_caption: Optional[torch.Tensor] = None,
        ovt_valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        model_device = self._pgot_model_device()
        images = images.to(model_device)
        target_images = target_images.to(model_device)
        caption_input_ids = caption_input_ids.to(model_device)
        caption_attention_mask = caption_attention_mask.to(model_device, dtype=torch.bool)
        if ovt_positions_in_caption is not None:
            ovt_positions_in_caption = ovt_positions_in_caption.to(model_device, dtype=torch.long)
        if ovt_valid_mask is not None:
            ovt_valid_mask = ovt_valid_mask.to(model_device, dtype=torch.bool)
        B, caption_len = caption_input_ids.shape

        _, img_features, gt_siglip = self._encode_images_aurora(
            images,
            target_images=target_images,
        )
        if gt_siglip is None:
            raise ValueError("PGOT forward requires target_images.")
        dtype = self._aurora_model_dtype()

        sys_p = self._pgot_embed_frozen_tokens(
            self.pgot_system_prefix_ids, B, model_device, dtype
        )
        sys_s = self._pgot_embed_frozen_tokens(
            self.pgot_system_suffix_ids, B, model_device, dtype
        )
        user_p = self._pgot_embed_frozen_tokens(
            self.pgot_user_prefix_ids, B, model_device, dtype
        )
        user_s = self._pgot_embed_frozen_tokens(
            self.pgot_user_suffix_ids, B, model_device, dtype
        )
        asst_p = self._pgot_embed_frozen_tokens(
            self.pgot_assistant_prefix_ids, B, model_device, dtype
        )
        asst_s = self._pgot_embed_frozen_tokens(
            self.pgot_assistant_suffix_ids, B, model_device, dtype
        )
        caption_embeds = self._pgot_embed_caption(caption_input_ids, model_device, dtype)
        null_bg_embeds = self._pgot_embed_null_bg(B, model_device, dtype)
        register_embeds = self.pgot_register_embeddings.unsqueeze(0).expand(B, -1, -1).to(
            device=model_device, dtype=dtype
        )
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

        inputs_embeds = torch.cat(
            [
                sys_p,
                sys_s,
                user_p,
                img_features.to(dtype=dtype),
                user_s,
                asst_p,
                caption_embeds,
                asst_s,
                null_bg_embeds,
                register_embeds,
                rae_embeds,
            ],
            dim=1,
        )

        cap_s_pos = positions["cap_s"]
        ovt_abs_positions = (
            cap_s_pos + ovt_positions_in_caption
            if ovt_positions_in_caption is not None
            else None
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
        return {
            "device": model_device,
            "images": images,
            "target_images": target_images,
            "caption_input_ids": caption_input_ids,
            "caption_attention_mask": caption_attention_mask,
            "ovt_positions_in_caption": ovt_positions_in_caption,
            "ovt_valid_mask": ovt_valid_mask,
            "img_features": img_features,
            "gt_siglip": gt_siglip,
            "positions": positions,
            "inputs_embeds": inputs_embeds,
            "attn_bias": attn_bias,
            "ovt_abs_positions": ovt_abs_positions,
        }

    def _pgot_merge_object_ovts(
        self,
        hidden_states: torch.Tensor,
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ovt_hidden = gather_ovt_hidden_states(hidden_states, ovt_abs_positions, ovt_valid_mask)
        B, M, D = ovt_hidden.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M // n
        if K <= 0:
            return (
                hidden_states.new_empty(B, 0, D),
                hidden_states.new_zeros(B, 0, dtype=torch.bool),
            )

        ovt_hidden = ovt_hidden[:, : K * n].reshape(B, K, n, D)
        ovt_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n)
        obj_valid = ovt_valid.any(dim=2)
        denom = ovt_valid.float().sum(dim=2, keepdim=True).clamp_min(1.0)
        object_ovts = (ovt_hidden * ovt_valid.unsqueeze(-1).float()).sum(dim=2) / denom
        return object_ovts, obj_valid

    def _pgot_repeat_object_ovts_to_tokens(
        self,
        object_ovts: torch.Tensor,
        total_ovt_tokens: int,
    ) -> torch.Tensor:
        B, K, D = object_ovts.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        repeated = object_ovts.unsqueeze(2).expand(B, K, n, D).reshape(B, K * n, D)
        if repeated.shape[1] >= total_ovt_tokens:
            return repeated[:, :total_ovt_tokens]
        pad = total_ovt_tokens - repeated.shape[1]
        zeros = repeated.new_zeros(B, pad, D)
        return torch.cat([repeated, zeros], dim=1)

    def _pgot_v12_build_ovt_states(
        self,
        hidden_states: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        object_ovts, object_valid = self._pgot_merge_object_ovts(
            hidden_states=hidden_states,
            ovt_abs_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
        )
        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        void_ovts = hidden_states[:, void_s:void_e, :]
        if void_ovts.shape[1] > 0:
            void_valid = torch.ones(
                hidden_states.shape[0],
                void_ovts.shape[1],
                device=hidden_states.device,
                dtype=torch.bool,
            )
        else:
            void_valid = hidden_states.new_zeros(
                hidden_states.shape[0], 0, dtype=torch.bool
            )
        ovt_states = torch.cat([object_ovts, void_ovts], dim=1)
        ovt_valid = torch.cat([object_valid, void_valid], dim=1)
        return {
            "object_ovts": object_ovts,
            "object_valid": object_valid,
            "void_ovts": void_ovts,
            "void_valid": void_valid,
            "ovt_states": ovt_states,
            "ovt_valid": ovt_valid,
        }

    def _pgot_v12_scatter_ovts_to_hidden(
        self,
        hidden_states: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        object_delta: torch.Tensor,
        void_delta: torch.Tensor,
    ) -> torch.Tensor:
        """Add per-OVT DELTAs onto the residual stream, out-of-place (autograd-safe).

        Builds a full-sequence delta tensor (zeros except OVT/void positions) and
        returns hidden + delta. No in-place writes on tensors that autograd needs.
        """
        B, L, D = hidden_states.shape
        safe_positions = ovt_abs_positions.clamp(min=0, max=L - 1)
        repeated_delta = self._pgot_repeat_object_ovts_to_tokens(
            object_ovts=object_delta,
            total_ovt_tokens=ovt_abs_positions.shape[1],
        )
        # Zero the delta at padded OVT positions so only real OVTs are updated.
        repeated_delta = repeated_delta * ovt_valid_mask.unsqueeze(-1).to(repeated_delta.dtype)
        ovt_idx = safe_positions.unsqueeze(-1).expand(-1, -1, D)
        delta_full = torch.zeros_like(hidden_states).scatter(1, ovt_idx, repeated_delta)
        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        if void_e > void_s and void_delta.shape[1] > 0:
            void_idx = torch.arange(void_s, void_e, device=hidden_states.device)
            void_idx = void_idx.view(1, -1, 1).expand(B, void_e - void_s, D)
            delta_full = delta_full.scatter(1, void_idx, void_delta)
        return hidden_states + delta_full

    def _pgot_v12_compute_owner_outputs(
        self,
        hidden_states: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ovt_pack = self._pgot_v12_build_ovt_states(
            hidden_states=hidden_states,
            positions=positions,
            ovt_abs_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
        )
        img_hidden = hidden_states[:, positions["img_s"]:positions["img_e"], :]
        owner_logits, owner_probs = self.pgot_v12_owner_head(
            ovt_states=ovt_pack["ovt_states"],
            img_hidden=img_hidden,
            ovt_valid_mask=ovt_pack["ovt_valid"],
            temperature=float(getattr(self.config, "pgot_v12_owner_temperature", 1.0)),
        )
        K = ovt_pack["object_ovts"].shape[1]
        return {
            "img_hidden": img_hidden,
            "owner_logits": owner_logits,
            "owner_probs": owner_probs,
            "object_probs": owner_probs[:, :K],
            "void_probs": owner_probs[:, K:],
            "object_valid": ovt_pack["object_valid"],
            "ovt_valid": ovt_pack["ovt_valid"],
            "object_ovts": ovt_pack["object_ovts"],
            "void_ovts": ovt_pack["void_ovts"],
        }

    def _pgot_v12_compute_owner_loss(
        self,
        owner_logits: torch.Tensor,
        owner_probs: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, S, P = owner_logits.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        M = gt_masks_per_ovt.shape[1]
        K = M // n
        V = max(S - K, 0)

        if K <= 0:
            z = owner_logits.new_zeros(())
            return {
                "loss": z,
                "fg_acc": z,
                "bg_acc": z,
                "void_prob_on_fg": z,
                "object_prob_on_bg": z,
                "object_count": z,
            }

        obj_cover = gt_masks_per_ovt[:, : K * n].reshape(B, K, n, P).float().amax(dim=2)
        obj_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).any(dim=2)
        obj_cover = obj_cover.masked_fill(~obj_valid.unsqueeze(-1), 0.0)
        best_cover, best_idx = obj_cover.max(dim=1)

        if V > 0:
            bg_index = torch.full_like(best_idx, K)
            label = torch.where(best_cover > 0.0, best_idx, bg_index)
            loss = F.cross_entropy(
                owner_logits.permute(0, 2, 1).reshape(B * P, S),
                label.reshape(B * P),
            )
        else:
            label = best_idx.masked_fill(best_cover <= 0.0, -100)
            loss = F.cross_entropy(
                owner_logits.permute(0, 2, 1).reshape(B * P, S),
                label.reshape(B * P),
                ignore_index=-100,
            )

        if not torch.isfinite(loss):
            loss = owner_logits.new_zeros(())

        pred = owner_logits.argmax(dim=1)
        fg_mask = best_cover > 0.0
        bg_mask = ~fg_mask
        fg_acc = (
            (pred[fg_mask] == label[fg_mask]).float().mean()
            if bool(fg_mask.any())
            else owner_logits.new_zeros(())
        )
        if V > 0 and bool(bg_mask.any()):
            bg_acc = (pred[bg_mask] >= K).float().mean()
        else:
            bg_acc = owner_logits.new_zeros(())

        void_prob = (
            owner_probs[:, K:].sum(dim=1)
            if V > 0
            else owner_probs.new_zeros(B, P)
        )
        object_prob = (
            owner_probs[:, :K].sum(dim=1)
            if K > 0
            else owner_probs.new_zeros(B, P)
        )
        void_prob_on_fg = (
            void_prob[fg_mask].mean()
            if bool(fg_mask.any())
            else owner_logits.new_zeros(())
        )
        object_prob_on_bg = (
            object_prob[bg_mask].mean()
            if bool(bg_mask.any())
            else owner_logits.new_zeros(())
        )
        return {
            "loss": loss,
            "fg_acc": fg_acc.detach(),
            "bg_acc": bg_acc.detach(),
            "void_prob_on_fg": void_prob_on_fg.detach(),
            "object_prob_on_bg": object_prob_on_bg.detach(),
            "object_count": obj_valid.float().sum(dim=1).mean().detach(),
        }

    def _pgot_v12_forward_features(
        self,
        *,
        images: torch.Tensor,
        target_images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        ovt_positions_in_caption: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        output_hidden_states: bool = False,
        return_v12_block_maps: bool = False,
        rae_block_ovt_indices: Tuple[int, ...] = (),
    ) -> Dict[str, torch.Tensor]:
        seq = self._pgot_build_sequence_inputs(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
        )
        if rae_block_ovt_indices:
            seq["attn_bias"] = seq["attn_bias"].clone()
            rae_s, rae_e = seq["positions"]["rae_s"], seq["positions"]["rae_e"]
            for b_idx in range(seq["ovt_abs_positions"].shape[0]):
                for ovt_idx in rae_block_ovt_indices:
                    if 0 <= ovt_idx < seq["ovt_abs_positions"].shape[1] and bool(seq["ovt_valid_mask"][b_idx, ovt_idx]):
                        pos = int(seq["ovt_abs_positions"][b_idx, ovt_idx].item())
                        seq["attn_bias"][b_idx, :, rae_s:rae_e, pos] = float("-inf")
        layer_to_block = {
            int(layer_idx): block_idx
            for block_idx, layer_idx in enumerate(self.pgot_v12_layers)
        }
        v12_block_records = []

        def hidden_state_postprocess_fn(hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
            block_idx = layer_to_block.get(int(layer_idx))
            if block_idx is None:
                return hidden_states
            ovt_pack = self._pgot_v12_build_ovt_states(
                hidden_states=hidden_states,
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
            )
            updated_ovts, owner_logits, owner_probs = self.pgot_v12_slot_update_blocks[block_idx](
                ovt_states=ovt_pack["ovt_states"],
                img_hidden=hidden_states[:, seq["positions"]["img_s"]:seq["positions"]["img_e"], :],
                ovt_valid_mask=ovt_pack["ovt_valid"],
            )
            K = ovt_pack["object_ovts"].shape[1]
            if return_v12_block_maps:
                v12_block_records.append(
                    {
                        "layer": int(layer_idx),
                        "block_index": int(block_idx),
                        "owner_logits": owner_logits.detach(),
                        "owner_probs": owner_probs.detach(),
                        "object_probs": owner_probs[:, :K].detach(),
                        "void_probs": owner_probs[:, K:].detach(),
                        "object_valid": ovt_pack["object_valid"].detach(),
                        "ovt_valid": ovt_pack["ovt_valid"].detach(),
                    }
                )
            # Apply the OVT DELTA back to the residual stream (not an overwrite),
            # so the block is an exact identity at init (delta = 0) and recon stays
            # on-distribution. Object delta is shared across that object's OVTs.
            delta = updated_ovts - ovt_pack["ovt_states"]
            return self._pgot_v12_scatter_ovts_to_hidden(
                hidden_states=hidden_states,
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
                object_delta=delta[:, :K],
                void_delta=delta[:, K:],
            )

        out = self.model(
            inputs_embeds=seq["inputs_embeds"],
            attention_bias=seq["attn_bias"],
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            hidden_state_postprocess_fn=hidden_state_postprocess_fn,
        )
        hidden = out.last_hidden_state
        owner = self._pgot_v12_compute_owner_outputs(
            hidden_states=hidden,
            positions=seq["positions"],
            ovt_abs_positions=seq["ovt_abs_positions"],
            ovt_valid_mask=seq["ovt_valid_mask"],
        )
        seq.update(
            {
                "hidden": hidden,
                "rae_hidden": hidden[:, seq["positions"]["rae_s"]:seq["positions"]["rae_e"], :],
                "outputs": out,
                "hidden_states": out.hidden_states if output_hidden_states else None,
                "v12_block_owner_records": v12_block_records if return_v12_block_maps else None,
            }
        )
        seq.update(owner)
        return seq

    def _resolve_llm_qk_outside_layers(self, spec: str) -> List[int]:
        """Resolve a layer spec such as 'last4', 'all', '20:28', or '20,24,27'."""
        return _resolve_layer_spec(spec, len(getattr(self.model, "layers", [])))

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

    @staticmethod
    def _pgot_rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _compute_exact_llm_attention_components_for_layer(
        self,
        *,
        layer_input: torch.Tensor,
        layer_idx: int,
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        patch_temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return full-key and image-conditional attention for one LLM layer.

        Both paths use the layer's real normalized input, Q/K projections, RoPE,
        GQA expansion, and attention bias. The full path matches transformer
        attention. The conditional path re-normalizes the same image-key scores
        over image patches only and is used solely for v8.5 supervision/readout.
        """
        block = self.model.layers[layer_idx]
        attn = block.self_attn
        attn_input = block.input_layernorm(layer_input)
        B, L, _ = attn_input.shape
        M = ovt_abs_positions.shape[1]
        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        V = max(void_e - void_s, 0)

        if V > 0:
            void_positions = torch.arange(
                void_s, void_e, device=attn_input.device, dtype=torch.long
            ).view(1, V).expand(B, -1)
            query_positions = torch.cat([ovt_abs_positions, void_positions], dim=1)
            query_valid = torch.cat(
                [
                    ovt_valid_mask,
                    torch.ones(B, V, device=attn_input.device, dtype=torch.bool),
                ],
                dim=1,
            )
        else:
            query_positions = ovt_abs_positions
            query_valid = ovt_valid_mask

        safe_positions = query_positions.clamp(min=0, max=L - 1)
        q_input = attn_input.gather(
            dim=1,
            index=safe_positions.unsqueeze(-1).expand(-1, -1, attn_input.shape[-1]),
        )
        q_input = q_input * query_valid.unsqueeze(-1).to(q_input.dtype)

        q_proj = attn.q_proj(q_input)
        k_proj = attn.k_proj(attn_input)
        num_heads = int(getattr(attn, "num_heads", self.config.num_attention_heads))
        num_kv_heads = int(
            getattr(attn, "num_key_value_heads", self.config.num_key_value_heads)
        )
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // num_heads))

        q = q_proj.view(B, -1, num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(B, L, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(k, seq_len=L)
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        all_positions = torch.arange(L, device=q.device, dtype=torch.long)
        k_cos = cos[all_positions].view(1, 1, L, head_dim)
        k_sin = sin[all_positions].view(1, 1, L, head_dim)
        q_cos = cos[safe_positions].unsqueeze(1)
        q_sin = sin[safe_positions].unsqueeze(1)
        q = (q * q_cos) + (self._pgot_rotate_half(q) * q_sin)
        k = (k * k_cos) + (self._pgot_rotate_half(k) * k_sin)

        if num_kv_heads != num_heads:
            groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
            k = k.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(2, 3))
        scores = scores / math.sqrt(float(head_dim))
        bias_rows = attention_bias.gather(
            dim=2,
            index=safe_positions[:, None, :, None].expand(B, 1, safe_positions.shape[1], L),
        )
        scores = scores + bias_rows.float()
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        probs = probs * query_valid[:, None, :, None].float()
        image_probs = probs[..., positions["img_s"]:positions["img_e"]]
        image_scores = scores[..., positions["img_s"]:positions["img_e"]]
        patch_probs = F.softmax(
            image_scores / max(float(patch_temperature), 1e-6),
            dim=-1,
            dtype=torch.float32,
        )
        patch_probs = patch_probs * query_valid[:, None, :, None].float()
        return (
            image_probs[:, :, :M],
            image_probs[:, :, M:M + V],
            patch_probs[:, :, :M],
            patch_probs[:, :, M:M + V],
        )

    def _compute_exact_llm_image_attention_for_layer(
        self,
        *,
        layer_input: torch.Tensor,
        layer_idx: int,
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reproduce v8.4's full-key-softmax OVT/void image attention."""
        object_full, void_full, _, _ = (
            self._compute_exact_llm_attention_components_for_layer(
                layer_input=layer_input,
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
            )
        )
        return object_full, void_full

    def _compute_exact_llm_attention_outside_loss(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str,
        void_weight: float = 1.0,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Outside-only log loss on the LLM's actual OVT/void attention.

        Object OVTs ignore their own region and penalize every image patch
        outside it. The always-present void token uses the complement target:
        annotated thing/stuff patches are forbidden and unassigned patches are
        ignored. Object and void terms are averaged independently so one void
        token is not drowned by a variable number of object OVTs.
        """
        z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        if hidden_states is None:
            return {
                "loss": z, "object_loss": z, "void_loss": z,
                "object_self_mass": z, "object_outside_mass": z,
                "object_image_mass": z, "object_nonimage_mass": z,
                "void_self_mass": z, "void_outside_mass": z,
                "void_image_mass": z, "void_nonimage_mass": z,
            }

        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if not layers:
            return {
                "loss": z, "object_loss": z, "void_loss": z,
                "object_self_mass": z, "object_outside_mass": z,
                "object_image_mass": z, "object_nonimage_mass": z,
                "void_self_mass": z, "void_outside_mass": z,
                "void_image_mass": z, "void_nonimage_mass": z,
            }

        masks = gt_masks_per_ovt.float().clamp(0.0, 1.0)
        valid = ovt_valid_mask.float()
        annotated_union = (
            masks * valid.unsqueeze(-1)
        ).amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        void_target = (1.0 - annotated_union).clamp(0.0, 1.0)

        sums = {
            "object_loss": z, "void_loss": z,
            "object_self_mass": z, "object_outside_mass": z,
            "object_image_mass": z, "object_nonimage_mass": z,
            "void_self_mass": z, "void_outside_mass": z,
            "void_image_mass": z, "void_nonimage_mass": z,
        }
        count = 0
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            object_probs, void_probs = self._compute_exact_llm_image_attention_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
            )
            H = object_probs.shape[1]
            object_outside = (1.0 - masks).clamp(0.0, 1.0)
            object_log = -torch.log((1.0 - object_probs).clamp_min(eps))
            object_per_head = (object_log * object_outside[:, None]).sum(dim=-1)
            object_denom = (valid.sum() * H).clamp_min(1.0)
            object_loss = (object_per_head * valid[:, None]).sum() / object_denom

            object_self = (object_probs * masks[:, None]).sum(dim=-1)
            object_out = (object_probs * object_outside[:, None]).sum(dim=-1)
            object_image = object_probs.sum(dim=-1)
            sums["object_loss"] = sums["object_loss"] + object_loss
            sums["object_self_mass"] = sums["object_self_mass"] + (
                (object_self * valid[:, None]).sum() / object_denom
            )
            sums["object_outside_mass"] = sums["object_outside_mass"] + (
                (object_out * valid[:, None]).sum() / object_denom
            )
            object_image_mean = (object_image * valid[:, None]).sum() / object_denom
            sums["object_image_mass"] = sums["object_image_mass"] + object_image_mean
            sums["object_nonimage_mass"] = sums["object_nonimage_mass"] + (1.0 - object_image_mean)

            if void_probs.shape[2] > 0:
                V = void_probs.shape[2]
                void_outside = annotated_union
                void_log = -torch.log((1.0 - void_probs).clamp_min(eps))
                void_loss = (void_log * void_outside[:, None]).sum(dim=-1).mean()
                void_self = (void_probs * void_target[:, None]).sum(dim=-1).mean()
                void_out = (void_probs * void_outside[:, None]).sum(dim=-1).mean()
                void_image = void_probs.sum(dim=-1).mean()
                sums["void_loss"] = sums["void_loss"] + void_loss
                sums["void_self_mass"] = sums["void_self_mass"] + void_self
                sums["void_outside_mass"] = sums["void_outside_mass"] + void_out
                sums["void_image_mass"] = sums["void_image_mass"] + void_image
                sums["void_nonimage_mass"] = sums["void_nonimage_mass"] + (1.0 - void_image)
            count += 1

        if count <= 0:
            return {
                "loss": z, "object_loss": z, "void_loss": z,
                "object_self_mass": z, "object_outside_mass": z,
                "object_image_mass": z, "object_nonimage_mass": z,
                "void_self_mass": z, "void_outside_mass": z,
                "void_image_mass": z, "void_nonimage_mass": z,
            }

        scale = float(count)
        averaged = {key: value / scale for key, value in sums.items()}
        averaged["loss"] = (
            averaged["object_loss"] + float(void_weight) * averaged["void_loss"]
        )
        for key in averaged:
            if key != "loss":
                averaged[key] = averaged[key].detach()
        return averaged

    def _compute_llm_patch_outside_and_image_use_loss(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str,
        temperature: float = 1.0,
        void_weight: float = 1.0,
        image_use_margin: float = 0.05,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """V8.5 spatial outside loss plus a weak real-attention image-use hinge.

        The location map is softmax-normalized over image patches only. Thus,
        reducing outside probability necessarily moves probability inside the
        target region. The separate image-use hinge observes the transformer's
        real full-key softmax and prevents OVTs from satisfying the spatial loss
        while assigning negligible total attention to image keys.
        """
        z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "outside_loss": z,
            "image_use_loss": z,
            "object_outside_loss": z,
            "void_outside_loss": z,
            "object_image_use_loss": z,
            "void_image_use_loss": z,
            "object_self_mass": z,
            "object_outside_mass": z,
            "object_full_image_mass": z,
            "void_self_mass": z,
            "void_outside_mass": z,
            "void_full_image_mass": z,
            "void_valid_fraction": z,
        }
        if hidden_states is None:
            return empty

        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if not layers:
            return empty

        masks = gt_masks_per_ovt.float().clamp(0.0, 1.0)
        valid = ovt_valid_mask.float()
        annotated_union = (
            masks * valid.unsqueeze(-1)
        ).amax(dim=1, keepdim=True).clamp(0.0, 1.0)
        void_target = (1.0 - annotated_union).clamp(0.0, 1.0)
        # A conditional image softmax cannot satisfy a void target when no
        # residual patch exists. Keep the token present, but skip its spatial
        # and image-use terms for those samples.
        void_sample_valid = (void_target.sum(dim=-1) > eps).float()

        sums = {key: z for key in empty if key != "void_valid_fraction"}
        count = 0
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            (
                object_full,
                void_full,
                object_patch,
                void_patch,
            ) = self._compute_exact_llm_attention_components_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                patch_temperature=temperature,
            )

            H = object_patch.shape[1]
            object_outside = (1.0 - masks).clamp(0.0, 1.0)
            object_log = -torch.log((1.0 - object_patch).clamp_min(eps))
            object_per_head = (object_log * object_outside[:, None]).sum(dim=-1)
            object_denom = (valid.sum() * H).clamp_min(1.0)
            object_outside_loss = (
                object_per_head * valid[:, None]
            ).sum() / object_denom

            object_self = (object_patch * masks[:, None]).sum(dim=-1)
            object_out = (object_patch * object_outside[:, None]).sum(dim=-1)
            object_full_image = object_full.sum(dim=-1)
            object_image_use = F.relu(
                float(image_use_margin) - object_full_image
            )

            sums["object_outside_loss"] = (
                sums["object_outside_loss"] + object_outside_loss
            )
            sums["object_image_use_loss"] = sums["object_image_use_loss"] + (
                (object_image_use * valid[:, None]).sum() / object_denom
            )
            sums["object_self_mass"] = sums["object_self_mass"] + (
                (object_self * valid[:, None]).sum() / object_denom
            )
            sums["object_outside_mass"] = sums["object_outside_mass"] + (
                (object_out * valid[:, None]).sum() / object_denom
            )
            sums["object_full_image_mass"] = sums["object_full_image_mass"] + (
                (object_full_image * valid[:, None]).sum() / object_denom
            )

            void_outside_loss = z
            void_image_use_loss = z
            void_self_mass = z
            void_outside_mass = z
            void_full_image_mass = z
            if void_patch.shape[2] > 0:
                V = void_patch.shape[2]
                void_valid = void_sample_valid.expand(-1, V)
                void_denom = (void_valid.sum() * H).clamp_min(1.0)
                void_outside = annotated_union
                void_log = -torch.log((1.0 - void_patch).clamp_min(eps))
                void_per_head = (
                    void_log * void_outside[:, None]
                ).sum(dim=-1)
                void_outside_loss = (
                    void_per_head * void_valid[:, None]
                ).sum() / void_denom

                void_self = (void_patch * void_target[:, None]).sum(dim=-1)
                void_out = (void_patch * void_outside[:, None]).sum(dim=-1)
                void_full_image = void_full.sum(dim=-1)
                void_image_use = F.relu(
                    float(image_use_margin) - void_full_image
                )
                void_image_use_loss = (
                    void_image_use * void_valid[:, None]
                ).sum() / void_denom
                void_self_mass = (
                    void_self * void_valid[:, None]
                ).sum() / void_denom
                void_outside_mass = (
                    void_out * void_valid[:, None]
                ).sum() / void_denom
                void_full_image_mass = (
                    void_full_image * void_valid[:, None]
                ).sum() / void_denom

            sums["void_outside_loss"] = (
                sums["void_outside_loss"] + void_outside_loss
            )
            sums["void_image_use_loss"] = (
                sums["void_image_use_loss"] + void_image_use_loss
            )
            sums["void_self_mass"] = sums["void_self_mass"] + void_self_mass
            sums["void_outside_mass"] = (
                sums["void_outside_mass"] + void_outside_mass
            )
            sums["void_full_image_mass"] = (
                sums["void_full_image_mass"] + void_full_image_mass
            )
            count += 1

        if count <= 0:
            return empty

        averaged = {key: value / float(count) for key, value in sums.items()}
        averaged["outside_loss"] = (
            averaged["object_outside_loss"]
            + float(void_weight) * averaged["void_outside_loss"]
        )
        averaged["image_use_loss"] = (
            averaged["object_image_use_loss"]
            + float(void_weight) * averaged["void_image_use_loss"]
        )
        averaged["void_valid_fraction"] = void_sample_valid.mean().detach()
        for key in averaged:
            if key not in {"outside_loss", "image_use_loss"}:
                averaged[key] = averaged[key].detach()
        return averaged

    def _compute_llm_patch_attention_maps(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        layers_spec: str,
        temperature: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return layer/head-mean image-conditional maps used by v8.5."""
        if hidden_states is None:
            return None, None
        object_acc = None
        void_acc = None
        count = 0
        for layer_idx in self._resolve_llm_qk_outside_layers(layers_spec):
            if layer_idx >= len(hidden_states) - 1:
                continue
            _, _, object_patch, void_patch = (
                self._compute_exact_llm_attention_components_for_layer(
                    layer_input=hidden_states[layer_idx],
                    layer_idx=layer_idx,
                    attention_bias=attention_bias,
                    positions=positions,
                    ovt_abs_positions=ovt_abs_positions,
                    ovt_valid_mask=ovt_valid_mask,
                    patch_temperature=temperature,
                )
            )
            object_map = object_patch.mean(dim=1)
            void_map = void_patch.mean(dim=1)
            object_acc = object_map if object_acc is None else object_acc + object_map
            void_acc = void_map if void_acc is None else void_acc + void_map
            count += 1
        if count <= 0:
            return None, None
        return (
            (object_acc / float(count)).detach(),
            (void_acc / float(count)).detach() if void_acc is not None else None,
        )

    def _compute_exact_llm_attention_maps(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        layers_spec: str,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return head/layer-mean maps from the same attention used by v8.4."""
        if hidden_states is None:
            return None, None
        object_acc = None
        void_acc = None
        count = 0
        for layer_idx in self._resolve_llm_qk_outside_layers(layers_spec):
            if layer_idx >= len(hidden_states) - 1:
                continue
            object_probs, void_probs = self._compute_exact_llm_image_attention_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
            )
            object_map = object_probs.mean(dim=1)
            void_map = void_probs.mean(dim=1)
            object_acc = object_map if object_acc is None else object_acc + object_map
            void_acc = void_map if void_acc is None else void_acc + void_map
            count += 1
        if count <= 0:
            return None, None
        return (
            (object_acc / float(count)).detach(),
            (void_acc / float(count)).detach() if void_acc is not None else None,
        )

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
        if bool(getattr(self.config, "pgot_v12_enable", False)):
            feats = self._pgot_v12_forward_features(
                images=images,
                target_images=target_images,
                caption_input_ids=caption_input_ids,
                caption_attention_mask=caption_attention_mask,
                ovt_positions_in_caption=ovt_positions_in_caption,
                ovt_valid_mask=ovt_valid_mask,
                output_hidden_states=False,
            )
            return {"rae_hidden": feats["rae_hidden"], "gt_siglip": feats["gt_siglip"]}

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

    def _forward_pgot_v12(
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
        del ovt_is_thing, pgot_contrastive_weight

        seq = self._pgot_v12_forward_features(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
            output_hidden_states=False,
        )
        hidden = seq["hidden"]
        rae_hidden = seq["rae_hidden"]
        gt_siglip = seq["gt_siglip"]
        B = hidden.shape[0]

        if caption_labels is not None:
            caption_labels = caption_labels.to(hidden.device, dtype=torch.long)
        gt_masks_per_ovt = gt_masks_per_ovt.to(hidden.device).float()
        ovt_valid_mask = ovt_valid_mask.to(hidden.device, dtype=torch.bool)
        caption_input_ids = caption_input_ids.to(hidden.device)
        caption_attention_mask = caption_attention_mask.to(hidden.device, dtype=torch.bool)

        loss_lm = self._compute_lm_loss(
            hidden=hidden,
            positions=seq["positions"],
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            caption_labels=caption_labels,
        )

        owner_stats = self._pgot_v12_compute_owner_loss(
            owner_logits=seq["owner_logits"],
            owner_probs=seq["owner_probs"],
            gt_masks_per_ovt=gt_masks_per_ovt,
            ovt_valid_mask=ovt_valid_mask,
        )
        loss_owner = owner_stats["loss"]

        cfg_drop_rate = float(getattr(self.config, "pgot_cfg_drop_rate", 0.0))
        rae_hidden_for_diff = rae_hidden
        if self.training and cfg_drop_rate > 0.0:
            drop_mask = (torch.rand(B, device=rae_hidden.device) < cfg_drop_rate).view(B, 1, 1)
            rae_hidden_for_diff = rae_hidden * (~drop_mask).to(rae_hidden.dtype)
        loss_recon = self._captionslot_compute_diffusion_loss(
            hidden=rae_hidden_for_diff,
            target_features=gt_siglip,
            slot_context=None,
            slot_mask=None,
        )

        lm_w = float(getattr(self.config, "pgot_lm_loss_weight", 1.0))
        recon_w = float(getattr(self.config, "pgot_recon_loss_weight", 1.0))
        owner_w = float(getattr(self.config, "pgot_v12_owner_weight", 1.0))
        loss_mask = owner_w * loss_owner
        total_loss = lm_w * loss_lm + recon_w * loss_recon + loss_mask

        B_mask = ovt_valid_mask.shape[0]
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = ovt_valid_mask.shape[1] // n
        n_objects_mean = (
            ovt_valid_mask[:, : K * n].reshape(B_mask, K, n).any(dim=2).float().sum(dim=1).mean()
            if K > 0
            else hidden.new_zeros(())
        )

        self.pgot_loss_lm = loss_lm.detach()
        self.pgot_loss_mask = loss_mask.detach()
        self.pgot_loss_recon = loss_recon.detach()
        self.pgot_loss_contrastive = hidden.new_zeros(())
        self.pgot_n_objects_mean = n_objects_mean.detach()
        self.pgot_loss_details = {
            "loss_ovt_owner": loss_owner.detach(),
            "ovt_fg_acc": owner_stats["fg_acc"],
            "ovt_bg_acc": owner_stats["bg_acc"],
            "ovt_void_prob_on_fg": owner_stats["void_prob_on_fg"],
            "ovt_object_prob_on_bg": owner_stats["object_prob_on_bg"],
            "ovt_object_count": owner_stats["object_count"],
        }
        return total_loss, {
            "owner_logits": seq["owner_logits"].detach(),
            "owner_probs": seq["owner_probs"].detach(),
            "object_probs": seq["object_probs"].detach(),
            "void_probs": seq["void_probs"].detach(),
        }

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
        if bool(getattr(self.config, "pgot_v12_enable", False)):
            return self._forward_pgot_v12(
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
        mask_llm_attention_outside_w = float(
            getattr(self.config, "pgot_mask_llm_attention_outside_weight", 0.0)
        )
        mask_llm_patch_outside_w = float(
            getattr(self.config, "pgot_mask_llm_patch_outside_weight", 0.0)
        )
        mask_llm_image_use_w = float(
            getattr(self.config, "pgot_mask_llm_image_use_weight", 0.0)
        )
        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_bias=attn_bias,
            use_cache=False,
            output_hidden_states=(
                mask_llm_qk_outside_w > 0.0
                or mask_llm_attention_outside_w > 0.0
                or mask_llm_patch_outside_w > 0.0
                or mask_llm_image_use_w > 0.0
            ),
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
        ce_merge = str(getattr(self.config, "pgot_mask_ce_merge", "max")).lower()
        use_null_bg = bool(getattr(self.config, "pgot_use_null_bg_competition", False))
        mask_ce_w = float(getattr(self.config, "pgot_mask_ce_weight", 1.0))
        mask_fg_w = float(getattr(self.config, "pgot_mask_fg_weight", 0.0))
        mask_outside_w = float(getattr(self.config, "pgot_mask_outside_weight", 0.0))
        mask_ce_aux_w = float(getattr(self.config, "pgot_mask_aux_competition_weight", 0.0))
        mask_bce_w = float(getattr(self.config, "pgot_mask_bce_weight", 0.0))
        mask_object_balanced_bce_w = float(getattr(self.config, "pgot_mask_object_balanced_bce_weight", 0.0))
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
                merge=ce_merge,
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
                    merge=ce_merge,
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

        loss_mask_object_balanced_bce = zero
        object_balanced_bce_pos = zero
        object_balanced_bce_neg = zero
        if mask_object_balanced_bce_w > 0.0:
            object_balanced_bce = compute_object_balanced_mask_bce_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                n_ovt_per_object=self.pgot_n_ovt_per_object,
                merge=ce_merge,
            )
            loss_mask_object_balanced_bce = object_balanced_bce["loss"]
            object_balanced_bce_pos = object_balanced_bce["pos_loss"]
            object_balanced_bce_neg = object_balanced_bce["neg_loss"]

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

        loss_mask_llm_attention_outside = zero
        llm_attention_object_loss = zero
        llm_attention_void_loss = zero
        llm_attention_object_self_mass = zero
        llm_attention_object_outside_mass = zero
        llm_attention_object_image_mass = zero
        llm_attention_object_nonimage_mass = zero
        llm_attention_void_self_mass = zero
        llm_attention_void_outside_mass = zero
        llm_attention_void_image_mass = zero
        llm_attention_void_nonimage_mass = zero
        if mask_llm_attention_outside_w > 0.0:
            if null_bg_hidden.shape[1] <= 0:
                raise ValueError(
                    "Exact LLM-attention outside loss requires pgot_n_null_bg > 0 "
                    "for the always-present void token."
                )
            llm_attention_layers = str(
                getattr(self.config, "pgot_mask_llm_attention_outside_layers", "last4")
            )
            llm_attention_void_weight = float(
                getattr(self.config, "pgot_mask_llm_attention_void_weight", 1.0)
            )
            llm_attention_losses = self._compute_exact_llm_attention_outside_loss(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=llm_attention_layers,
                void_weight=llm_attention_void_weight,
            )
            loss_mask_llm_attention_outside = llm_attention_losses["loss"]
            llm_attention_object_loss = llm_attention_losses["object_loss"]
            llm_attention_void_loss = llm_attention_losses["void_loss"]
            llm_attention_object_self_mass = llm_attention_losses["object_self_mass"]
            llm_attention_object_outside_mass = llm_attention_losses["object_outside_mass"]
            llm_attention_object_image_mass = llm_attention_losses["object_image_mass"]
            llm_attention_object_nonimage_mass = llm_attention_losses["object_nonimage_mass"]
            llm_attention_void_self_mass = llm_attention_losses["void_self_mass"]
            llm_attention_void_outside_mass = llm_attention_losses["void_outside_mass"]
            llm_attention_void_image_mass = llm_attention_losses["void_image_mass"]
            llm_attention_void_nonimage_mass = llm_attention_losses["void_nonimage_mass"]

        loss_mask_llm_patch_outside = zero
        loss_mask_llm_image_use = zero
        llm_patch_object_outside_loss = zero
        llm_patch_void_outside_loss = zero
        llm_patch_object_image_use_loss = zero
        llm_patch_void_image_use_loss = zero
        llm_patch_object_self_mass = zero
        llm_patch_object_outside_mass = zero
        llm_patch_object_full_image_mass = zero
        llm_patch_void_self_mass = zero
        llm_patch_void_outside_mass = zero
        llm_patch_void_full_image_mass = zero
        llm_patch_void_valid_fraction = zero
        if mask_llm_patch_outside_w > 0.0 or mask_llm_image_use_w > 0.0:
            if null_bg_hidden.shape[1] <= 0:
                raise ValueError(
                    "LLM patch outside loss requires pgot_n_null_bg > 0 "
                    "for the always-present void token."
                )
            llm_patch_layers = str(
                getattr(self.config, "pgot_mask_llm_patch_outside_layers", "last4")
            )
            llm_patch_temperature = float(
                getattr(
                    self.config,
                    "pgot_mask_llm_patch_outside_temperature",
                    1.0,
                )
            )
            llm_patch_void_weight = float(
                getattr(self.config, "pgot_mask_llm_patch_void_weight", 1.0)
            )
            llm_image_use_margin = float(
                getattr(self.config, "pgot_mask_llm_image_use_margin", 0.05)
            )
            llm_patch_losses = (
                self._compute_llm_patch_outside_and_image_use_loss(
                    hidden_states=out.hidden_states,
                    attention_bias=attn_bias,
                    positions=positions,
                    ovt_abs_positions=ovt_abs_positions,
                    ovt_valid_mask=ovt_valid_mask,
                    gt_masks_per_ovt=gt_masks_per_ovt,
                    layers_spec=llm_patch_layers,
                    temperature=llm_patch_temperature,
                    void_weight=llm_patch_void_weight,
                    image_use_margin=llm_image_use_margin,
                )
            )
            loss_mask_llm_patch_outside = llm_patch_losses["outside_loss"]
            loss_mask_llm_image_use = llm_patch_losses["image_use_loss"]
            llm_patch_object_outside_loss = llm_patch_losses[
                "object_outside_loss"
            ]
            llm_patch_void_outside_loss = llm_patch_losses["void_outside_loss"]
            llm_patch_object_image_use_loss = llm_patch_losses[
                "object_image_use_loss"
            ]
            llm_patch_void_image_use_loss = llm_patch_losses[
                "void_image_use_loss"
            ]
            llm_patch_object_self_mass = llm_patch_losses["object_self_mass"]
            llm_patch_object_outside_mass = llm_patch_losses[
                "object_outside_mass"
            ]
            llm_patch_object_full_image_mass = llm_patch_losses[
                "object_full_image_mass"
            ]
            llm_patch_void_self_mass = llm_patch_losses["void_self_mass"]
            llm_patch_void_outside_mass = llm_patch_losses[
                "void_outside_mass"
            ]
            llm_patch_void_full_image_mass = llm_patch_losses[
                "void_full_image_mass"
            ]
            llm_patch_void_valid_fraction = llm_patch_losses[
                "void_valid_fraction"
            ]

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
            + mask_object_balanced_bce_w * loss_mask_object_balanced_bce
            + mask_tversky_w * loss_mask_tversky
            + mask_spatial_outside_w * loss_mask_spatial_outside
            + mask_spatial_outside_log_w * loss_mask_spatial_outside_log
            + mask_llm_qk_outside_w * loss_mask_llm_qk_outside
            + mask_llm_attention_outside_w * loss_mask_llm_attention_outside
            + mask_llm_patch_outside_w * loss_mask_llm_patch_outside
            + mask_llm_image_use_w * loss_mask_llm_image_use
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
        if mask_object_balanced_bce_w > 0.0:
            details["loss_mask_object_balanced_bce"] = loss_mask_object_balanced_bce.detach()
            details["object_balanced_bce_pos"] = object_balanced_bce_pos.detach()
            details["object_balanced_bce_neg"] = object_balanced_bce_neg.detach()
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
        if mask_llm_attention_outside_w > 0.0:
            details["loss_mask_llm_attention_outside"] = (
                loss_mask_llm_attention_outside.detach()
            )
            details["llm_attention_object_loss"] = llm_attention_object_loss.detach()
            details["llm_attention_void_loss"] = llm_attention_void_loss.detach()
            details["llm_attention_object_self_mass"] = (
                llm_attention_object_self_mass.detach()
            )
            details["llm_attention_object_outside_mass"] = (
                llm_attention_object_outside_mass.detach()
            )
            details["llm_attention_object_image_mass"] = (
                llm_attention_object_image_mass.detach()
            )
            details["llm_attention_object_nonimage_mass"] = (
                llm_attention_object_nonimage_mass.detach()
            )
            details["llm_attention_void_self_mass"] = llm_attention_void_self_mass.detach()
            details["llm_attention_void_outside_mass"] = (
                llm_attention_void_outside_mass.detach()
            )
            details["llm_attention_void_image_mass"] = llm_attention_void_image_mass.detach()
            details["llm_attention_void_nonimage_mass"] = (
                llm_attention_void_nonimage_mass.detach()
            )
        if mask_llm_patch_outside_w > 0.0 or mask_llm_image_use_w > 0.0:
            details["loss_mask_llm_patch_outside"] = (
                loss_mask_llm_patch_outside.detach()
            )
            details["loss_mask_llm_image_use"] = (
                loss_mask_llm_image_use.detach()
            )
            details["llm_patch_object_outside_loss"] = (
                llm_patch_object_outside_loss.detach()
            )
            details["llm_patch_void_outside_loss"] = (
                llm_patch_void_outside_loss.detach()
            )
            details["llm_patch_object_image_use_loss"] = (
                llm_patch_object_image_use_loss.detach()
            )
            details["llm_patch_void_image_use_loss"] = (
                llm_patch_void_image_use_loss.detach()
            )
            details["llm_patch_object_self_mass"] = (
                llm_patch_object_self_mass.detach()
            )
            details["llm_patch_object_outside_mass"] = (
                llm_patch_object_outside_mass.detach()
            )
            details["llm_patch_object_full_image_mass"] = (
                llm_patch_object_full_image_mass.detach()
            )
            details["llm_patch_void_self_mass"] = (
                llm_patch_void_self_mass.detach()
            )
            details["llm_patch_void_outside_mass"] = (
                llm_patch_void_outside_mass.detach()
            )
            details["llm_patch_void_full_image_mass"] = (
                llm_patch_void_full_image_mass.detach()
            )
            details["llm_patch_void_valid_fraction"] = (
                llm_patch_void_valid_fraction.detach()
            )
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
