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
    compute_sigmoid_outside_bce_loss,
    compute_register_foreground_suppression_loss,
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


def apply_register_hard_gt_mask(
    attention_bias: torch.Tensor,
    *,
    positions: Dict[str, int],
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Hard-block register queries from every annotated foreground patch.

    ``attention_bias`` is shared by all transformer layers and has a singleton
    head axis, so writing ``-inf`` here applies to every layer and every head.
    Only register-query/image-key edges change; OVT and RAE-query paths are
    untouched.  Fractional 32x32 GT masks are unioned over valid OVTs and a
    patch is blocked whenever its coverage is greater than ``threshold``.
    """
    if attention_bias.ndim != 4 or attention_bias.shape[1] != 1:
        raise ValueError(
            "PGOT attention bias must have shape [B,1,L,L], got "
            f"{tuple(attention_bias.shape)}"
        )
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"register hard-mask threshold must be in [0,1], got {threshold}")

    reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
    img_s, img_e = int(positions["img_s"]), int(positions["img_e"])
    n_register = max(reg_e - reg_s, 0)
    n_patches = max(img_e - img_s, 0)
    zero = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
    if n_register == 0 or n_patches == 0:
        return {
            "attention_bias": attention_bias,
            "blocked_patch_fraction": zero,
            "blocked_patch_count": zero,
        }

    masks = gt_masks_per_ovt.to(device=attention_bias.device, dtype=torch.float32)
    valid = ovt_valid_mask.to(device=attention_bias.device, dtype=torch.bool)
    if masks.ndim != 3 or valid.ndim != 2 or masks.shape[:2] != valid.shape:
        raise ValueError(
            "GT masks/valid flags must be [B,M,P]/[B,M], got "
            f"{tuple(masks.shape)}/{tuple(valid.shape)}"
        )
    if masks.shape[0] != attention_bias.shape[0]:
        raise ValueError("GT mask batch does not match attention bias batch")
    if masks.shape[-1] != n_patches:
        raise ValueError(
            f"GT mask has {masks.shape[-1]} patches but image sequence has {n_patches}"
        )

    foreground = (
        masks.clamp(0.0, 1.0) * valid.unsqueeze(-1).to(masks.dtype)
    ).amax(dim=1)
    blocked = foreground > float(threshold)
    register_to_image = attention_bias[:, :, reg_s:reg_e, img_s:img_e]
    register_to_image.masked_fill_(blocked[:, None, None, :], float("-inf"))
    return {
        "attention_bias": attention_bias,
        "blocked_patch_fraction": blocked.float().mean().detach(),
        "blocked_patch_count": blocked.float().sum(dim=-1).mean().detach(),
    }


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


class PGOTOVTBottleneckRouterRefineBlock(nn.Module):
    """Zero-init residual OVT cross-attention block for deeper V14 routing."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = max(int(dim) * max(int(mlp_ratio), 1), int(dim))
        self.query_ln = nn.LayerNorm(dim)
        self.ovt_ln = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.mlp_ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.mlp_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        condition: torch.Tensor,
        ovt_states: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        q = self.q_proj(self.query_ln(condition)).float()
        k = self.k_proj(self.ovt_ln(ovt_states)).float()
        logits = torch.einsum("bqd,bsd->bqs", q, k) / math.sqrt(float(q.shape[-1]))
        logits = logits / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~ovt_valid_mask.unsqueeze(1), -1e4)
        attn = F.softmax(logits, dim=-1)
        attn = attn * ovt_valid_mask.unsqueeze(1).float()
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        values = self.v_proj(self.ovt_ln(ovt_states))
        context = torch.einsum("bqs,bsd->bqd", attn.to(values.dtype), values)
        attn_out = self.out_proj(context).to(condition.dtype)
        x = condition + self.attn_gate.to(condition.dtype) * attn_out
        mlp_out = self.mlp(self.mlp_ln(x)).to(x.dtype)
        x = x + self.mlp_gate.to(x.dtype) * mlp_out
        return torch.where(torch.isfinite(x), x, condition)


class PGOTOVTBottleneckRouter(nn.Module):
    """Build diffusion condition tokens from OVT/void states only.

    The learned latent queries are used as positional queries. They no longer
    carry image content from the LLM residual stream; image/object content must
    flow through the OVT/void states selected by the route softmax.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        position_weight: float = 1.0,
        depth: int = 1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        self.query_ln = nn.LayerNorm(dim)
        self.ovt_ln = nn.LayerNorm(dim)
        self.context_ln = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.pos_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.temperature = float(temperature)
        self.position_weight = float(position_weight)
        self.depth = max(int(depth), 1)
        self.refine_blocks = nn.ModuleList(
            [
                PGOTOVTBottleneckRouterRefineBlock(dim=dim, mlp_ratio=mlp_ratio)
                for _ in range(self.depth - 1)
            ]
        )

    def forward(
        self,
        query_base: torch.Tensor,
        ovt_states: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        temperature: Optional[float] = None,
        position_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        temp = max(float(self.temperature if temperature is None else temperature), 1e-6)
        pos_w = float(self.position_weight if position_weight is None else position_weight)
        q = self.q_proj(self.query_ln(query_base)).float()
        k = self.k_proj(self.ovt_ln(ovt_states)).float()
        logits_qs = torch.einsum("bqd,bsd->bqs", q, k) / math.sqrt(float(q.shape[-1]))
        logits_qs = logits_qs / temp
        logits_qs = logits_qs.masked_fill(~ovt_valid_mask.unsqueeze(1), -1e4)
        route_qs = F.softmax(logits_qs, dim=-1)
        route_qs = route_qs * ovt_valid_mask.unsqueeze(1).float()
        route_qs = route_qs / route_qs.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        values = self.v_proj(self.ovt_ln(ovt_states))
        context = torch.einsum("bqs,bsd->bqd", route_qs.to(values.dtype), values)
        pos = self.pos_proj(query_base)
        condition = self.out_proj(self.context_ln(context + pos_w * pos))
        for block in self.refine_blocks:
            condition = block(
                condition=condition,
                ovt_states=ovt_states,
                ovt_valid_mask=ovt_valid_mask,
                temperature=temp,
            )
        condition = torch.where(torch.isfinite(condition), condition, torch.zeros_like(condition))
        return {
            "condition_hidden": condition,
            "owner_logits": logits_qs.transpose(1, 2).contiguous(),
            "owner_probs": route_qs.transpose(1, 2).contiguous(),
        }


class PGOTGroupGroundedRouter(nn.Module):
    """Group-level generative responsibility router for V21.

    Unlike V14, object OVTs are not averaged before routing. The same per-object
    responsibility map is used both for grounding loss and for composing the
    per-patch diffusion condition.
    """

    def __init__(
        self,
        dim: int,
        code_dim: int = 0,
        temperature: float = 1.0,
        position_weight: float = 1.0,
    ):
        super().__init__()
        code_dim = int(code_dim) if int(code_dim or 0) > 0 else int(dim)
        self.query_ln = nn.LayerNorm(dim)
        self.ovt_ln = nn.LayerNorm(dim)
        self.context_ln = nn.LayerNorm(dim)
        self.code_proj = nn.Linear(dim, code_dim, bias=False)
        self.q_proj = nn.Linear(dim, code_dim, bias=False)
        self.k_proj = nn.Linear(code_dim, code_dim, bias=False)
        self.v_proj = nn.Linear(code_dim, dim, bias=False)
        self.pos_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.temperature = float(temperature)
        self.position_weight = float(position_weight)

    def forward(
        self,
        query_base: torch.Tensor,          # (B, Q, D)
        object_ovts: torch.Tensor,         # (B, K, n, D)
        object_token_valid: torch.Tensor,  # (B, K, n)
        void_ovts: torch.Tensor,           # (B, V, D)
        void_valid: torch.Tensor,          # (B, V)
        temperature: Optional[float] = None,
        position_weight: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        B, Q, D = query_base.shape
        K = int(object_ovts.shape[1])
        n = int(object_ovts.shape[2]) if object_ovts.dim() >= 3 else 0
        V = int(void_ovts.shape[1])
        temp = max(float(self.temperature if temperature is None else temperature), 1e-6)
        pos_w = float(self.position_weight if position_weight is None else position_weight)
        neg = -1e4

        q = self.q_proj(self.query_ln(query_base)).float()                 # (B,Q,C)
        obj_flat = object_ovts.reshape(B, max(K * n, 0), D)
        obj_code = self.code_proj(self.ovt_ln(obj_flat)).float() if K > 0 else q.new_empty(B, 0, q.shape[-1])
        obj_key = self.k_proj(obj_code)
        obj_val = self.v_proj(obj_code.to(next(self.v_proj.parameters()).dtype)).to(query_base.dtype)
        obj_valid_flat = object_token_valid.reshape(B, max(K * n, 0)).to(device=query_base.device, dtype=torch.bool)

        if K > 0:
            obj_logits = torch.einsum("bqc,bsc->bqs", q, obj_key) / math.sqrt(float(q.shape[-1]))
            obj_logits = (obj_logits / temp).masked_fill(~obj_valid_flat.unsqueeze(1), neg)
            obj_logits_kn = obj_logits.reshape(B, Q, K, n)
            obj_scores = torch.logsumexp(obj_logits_kn.float(), dim=-1)     # (B,Q,K)
            obj_valid = object_token_valid.to(device=query_base.device, dtype=torch.bool).any(dim=2)
            obj_scores = obj_scores.masked_fill(~obj_valid.unsqueeze(1), neg)

            beta_obj = F.softmax(obj_logits_kn.float(), dim=-1)
            beta_obj = beta_obj * object_token_valid.to(beta_obj.device).unsqueeze(1).float()
            beta_obj = beta_obj / beta_obj.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            obj_val_kn = obj_val.reshape(B, K, n, D)
            obj_content = torch.einsum("bqkn,bknd->bqkd", beta_obj.to(obj_val_kn.dtype), obj_val_kn)
        else:
            obj_scores = q.new_empty(B, Q, 0)
            obj_content = query_base.new_empty(B, Q, 0, D)
            obj_valid = torch.zeros(B, 0, device=query_base.device, dtype=torch.bool)

        bg_scores = q.new_empty(B, Q, 0)
        bg_content = query_base.new_empty(B, Q, 0, D)
        if V > 0:
            void_code = self.code_proj(self.ovt_ln(void_ovts)).float()
            void_key = self.k_proj(void_code)
            void_val = self.v_proj(void_code.to(next(self.v_proj.parameters()).dtype)).to(query_base.dtype)
            void_valid = void_valid.to(device=query_base.device, dtype=torch.bool)
            void_logits = torch.einsum("bqc,bvc->bqv", q, void_key) / math.sqrt(float(q.shape[-1]))
            void_logits = (void_logits / temp).masked_fill(~void_valid.unsqueeze(1), neg)
            bg_score = torch.logsumexp(void_logits.float(), dim=-1, keepdim=True)
            bg_is_valid = void_valid.any(dim=1, keepdim=True)
            bg_scores = bg_score.masked_fill(~bg_is_valid.unsqueeze(1), neg)
            beta_bg = F.softmax(void_logits.float(), dim=-1)
            beta_bg = beta_bg * void_valid.unsqueeze(1).float()
            beta_bg = beta_bg / beta_bg.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            bg_content = torch.einsum("bqv,bvd->bqd", beta_bg.to(void_val.dtype), void_val).unsqueeze(2)

        group_scores = torch.cat([obj_scores, bg_scores], dim=-1)          # (B,Q,K+bg)
        if group_scores.shape[-1] == 0:
            group_scores = query_base.new_zeros(B, Q, 1).float()
            group_content = query_base.new_zeros(B, Q, 1, D)
        else:
            group_content = torch.cat([obj_content, bg_content], dim=2)
        resp = F.softmax(group_scores.float(), dim=-1)
        resp = torch.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0)
        context = torch.einsum("bqg,bqgd->bqd", resp.to(group_content.dtype), group_content)
        pos = self.pos_proj(query_base)
        condition = self.out_proj(self.context_ln(context + pos_w * pos))
        condition = torch.where(torch.isfinite(condition), condition, torch.zeros_like(condition))
        return {
            "condition_hidden": condition,
            "owner_logits": group_scores.transpose(1, 2).contiguous(),
            "owner_probs": resp.transpose(1, 2).contiguous(),
            "object_probs": resp[:, :, :K].transpose(1, 2).contiguous(),
            "void_probs": resp[:, :, K:].transpose(1, 2).contiguous(),
            "object_valid": obj_valid,
        }


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
        if not self.training or not (
            bool(getattr(self.config, "pgot_v12_enable", False))
            or bool(getattr(self.config, "pgot_v14_enable", False))
        ):
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

        self.pgot_ovt_caption_init_enabled = bool(
            getattr(self.config, "pgot_ovt_caption_init", False)
        )
        if self.pgot_ovt_caption_init_enabled:
            self.pgot_ovt_caption_norm = nn.LayerNorm(D)
            self.pgot_ovt_caption_projector = nn.Linear(D, D, bias=False)
            nn.init.eye_(self.pgot_ovt_caption_projector.weight)
        else:
            self.pgot_ovt_caption_norm = None
            self.pgot_ovt_caption_projector = None

        self.pgot_v12_enable = bool(getattr(self.config, "pgot_v12_enable", False))
        self.pgot_v14_enable = bool(getattr(self.config, "pgot_v14_enable", False))
        self.pgot_v21_enable = bool(getattr(self.config, "pgot_v21_enable", False))
        self.pgot_v17_enable = bool(getattr(self.config, "pgot_v17_enable", False))
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

        if self.pgot_v14_enable and self.pgot_v21_enable:
            route_temp = float(getattr(self.config, "pgot_v21_temperature", getattr(self.config, "pgot_v14_route_temperature", 1.0)))
            position_weight = float(getattr(self.config, "pgot_v21_position_weight", getattr(self.config, "pgot_v14_position_weight", 1.0)))
            code_dim = int(getattr(self.config, "pgot_v21_code_dim", 0))
            self.pgot_v21_router = PGOTGroupGroundedRouter(
                dim=D,
                code_dim=code_dim,
                temperature=route_temp,
                position_weight=position_weight,
            )
            self.pgot_v14_router = None
        elif self.pgot_v14_enable and not self.pgot_v17_enable:
            route_temp = float(getattr(self.config, "pgot_v14_route_temperature", 1.0))
            position_weight = float(getattr(self.config, "pgot_v14_position_weight", 1.0))
            router_depth = int(getattr(self.config, "pgot_v14_router_depth", 1))
            router_mlp_ratio = int(getattr(self.config, "pgot_v14_router_mlp_ratio", 4))
            self.pgot_v14_router = PGOTOVTBottleneckRouter(
                dim=D,
                temperature=route_temp,
                position_weight=position_weight,
                depth=router_depth,
                mlp_ratio=router_mlp_ratio,
            )
        else:
            self.pgot_v14_router = None
        if not (self.pgot_v14_enable and self.pgot_v21_enable):
            self.pgot_v21_router = None

        self.pgot_latent_distill_enable = bool(
            getattr(self.config, "pgot_latent_distill_enable", False)
        )
        latent_distill_w = float(getattr(self.config, "pgot_latent_distill_weight", 0.0))
        if self.pgot_latent_distill_enable or latent_distill_w > 0.0:
            latent_in = int(getattr(self.config, "diffusion_model_z_channels", D) or D)
            latent_out = int(getattr(self.config, "diffusion_model_channels", 1152) or 1152)
            self.pgot_latent_head = nn.Sequential(
                nn.LayerNorm(latent_in),
                nn.Linear(latent_in, latent_in),
                nn.GELU(),
                nn.Linear(latent_in, latent_out),
            )
            final = self.pgot_latent_head[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        else:
            self.pgot_latent_head = None

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
        self.pgot_thing_token_id = None
        self.pgot_stuff_token_id = None

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
            f"v12={self.pgot_v12_enable}, v12_layers={self.pgot_v12_layers}, "
            f"v14={self.pgot_v14_enable}, "
            f"v14_router_depth={getattr(self.config, 'pgot_v14_router_depth', 1)}, "
            f"v21={self.pgot_v21_enable}, "
            f"latent_distill={self.pgot_latent_head is not None}"
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
        gt_rae_masks_per_ovt: Optional[torch.Tensor] = None,
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
                gt_rae_masks_per_ovt=gt_rae_masks_per_ovt,
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

    def _pgot_apply_caption_conditioned_ovt_init(
        self,
        caption_embeds: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        ovt_positions_in_caption: Optional[torch.Tensor],
        ovt_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Add each object's own caption representation to its layer-0 OVT.

        For every valid OVT, the span starts after the most recent ``<thing>``
        (or ``<stuff>`` for legacy manifests) marker and ends immediately before
        the OVT.  No mask or extra image feature enters this path: at inference
        it uses the caption tokens already generated autoregressively.
        """
        if (
            not self.pgot_ovt_caption_init_enabled
            or self.pgot_ovt_caption_projector is None
            or ovt_positions_in_caption is None
            or ovt_valid_mask is None
        ):
            return caption_embeds

        ids = caption_input_ids.to(caption_embeds.device)
        positions = ovt_positions_in_caption.to(caption_embeds.device)
        valid = ovt_valid_mask.to(caption_embeds.device, dtype=torch.bool)
        out = caption_embeds.clone()
        marker_ids = {
            int(x)
            for x in (self.pgot_thing_token_id, self.pgot_stuff_token_id)
            if x is not None and int(x) >= 0
        }
        scale = float(getattr(self.config, "pgot_ovt_caption_init_scale", 1.0))
        for b in range(out.shape[0]):
            for slot_idx in valid[b].nonzero(as_tuple=False).flatten().tolist():
                end = int(positions[b, slot_idx].item())
                if end <= 0 or end >= out.shape[1]:
                    continue
                start = 0
                if marker_ids:
                    prefix = ids[b, :end]
                    marker_mask = torch.zeros_like(prefix, dtype=torch.bool)
                    for marker_id in marker_ids:
                        marker_mask |= prefix == marker_id
                    marker_pos = marker_mask.nonzero(as_tuple=False).flatten()
                    if marker_pos.numel() > 0:
                        start = int(marker_pos[-1].item()) + 1
                if start >= end:
                    continue
                pooled = caption_embeds[b, start:end].float().mean(dim=0)
                conditioned = self.pgot_ovt_caption_projector(
                    self.pgot_ovt_caption_norm(pooled.to(caption_embeds.dtype))
                )
                out[b, end] = out[b, end] + scale * conditioned.to(out.dtype)
        return out

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
        caption_embeds = self._pgot_apply_caption_conditioned_ovt_init(
            caption_embeds, caption_input_ids, ovt_positions_in_caption, ovt_valid_mask
        )
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
            rae_isolated=bool(getattr(self.config, "pgot_e4_rae_isolated", False)),
            rae_attends_caption=bool(getattr(self.config, "pgot_rae_attends_caption", False)),
            ovt_absolute_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            register_attends_caption=bool(
                getattr(self.config, "pgot_register_attends_caption", True)
            ),
            ovt_isolated=bool(
                getattr(self.config, "pgot_ovt_isolated_attention", False)
            ),
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

    def _pgot_v21_build_ovt_states(
        self,
        hidden_states: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ovt_hidden = gather_ovt_hidden_states(hidden_states, ovt_abs_positions, ovt_valid_mask)
        B, M, D = ovt_hidden.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M // n
        if K > 0:
            object_ovts = ovt_hidden[:, : K * n].reshape(B, K, n, D)
            object_token_valid = ovt_valid_mask[:, : K * n].reshape(B, K, n).to(dtype=torch.bool)
            object_valid = object_token_valid.any(dim=2)
            object_flat = object_ovts.reshape(B, K * n, D)
            object_flat_valid = object_token_valid.reshape(B, K * n)
        else:
            object_ovts = hidden_states.new_empty(B, 0, n, D)
            object_token_valid = hidden_states.new_zeros(B, 0, n, dtype=torch.bool)
            object_valid = hidden_states.new_zeros(B, 0, dtype=torch.bool)
            object_flat = hidden_states.new_empty(B, 0, D)
            object_flat_valid = hidden_states.new_zeros(B, 0, dtype=torch.bool)

        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        void_ovts = hidden_states[:, void_s:void_e, :]
        if void_ovts.shape[1] > 0:
            void_valid = torch.ones(B, void_ovts.shape[1], device=hidden_states.device, dtype=torch.bool)
        else:
            void_valid = hidden_states.new_zeros(B, 0, dtype=torch.bool)
        ovt_states = torch.cat([object_flat, void_ovts], dim=1)
        ovt_valid = torch.cat([object_flat_valid, void_valid], dim=1)
        return {
            "object_ovts_grouped": object_ovts,
            "object_token_valid": object_token_valid,
            "object_valid": object_valid,
            "object_ovts": object_flat,
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


    def _pgot_v14_compute_route_loss(
        self,
        owner_logits: torch.Tensor,
        owner_probs: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, S, P = owner_logits.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        M = gt_masks_per_ovt.shape[1]
        K = min(M // n, S)
        V = max(S - K, 0)
        zero = owner_logits.new_zeros(())
        if K <= 0:
            return {
                "loss": zero,
                "object_loss": zero,
                "void_loss": zero,
                "fg_acc": zero,
                "bg_acc": zero,
                "void_prob_on_fg": zero,
                "object_prob_on_bg": zero,
                "entropy": zero,
                "object_count": zero,
            }

        gt_masks = gt_masks_per_ovt[:, : K * n].to(owner_logits.device).float()
        if gt_masks.shape[-1] != P:
            src_side = int(round(float(gt_masks.shape[-1]) ** 0.5))
            dst_side = int(round(float(P) ** 0.5))
            if src_side * src_side != gt_masks.shape[-1] or dst_side * dst_side != P:
                raise ValueError(
                    f"V14 route loss expects square masks, got gt={gt_masks.shape[-1]} route={P}."
                )
            gt_masks = F.interpolate(
                gt_masks.reshape(B * K * n, 1, src_side, src_side),
                size=(dst_side, dst_side),
                mode="area",
            ).reshape(B, K * n, P).clamp(0.0, 1.0)
        obj_cover = gt_masks.reshape(B, K, n, P).amax(dim=2).clamp(0.0, 1.0)
        obj_valid = ovt_valid_mask[:, : K * n].to(owner_logits.device, dtype=torch.bool)
        obj_valid = obj_valid.reshape(B, K, n).any(dim=2)
        obj_cover = obj_cover.masked_fill(~obj_valid.unsqueeze(-1), 0.0)

        log_probs = F.log_softmax(owner_logits.float(), dim=1)
        obj_area = obj_cover.sum(dim=-1)
        valid_obj = obj_valid & (obj_area > 0.0)
        per_obj = -(obj_cover * log_probs[:, :K]).sum(dim=-1) / obj_area.clamp_min(1.0)
        object_loss = (
            per_obj[valid_obj].mean()
            if bool(valid_obj.any())
            else zero
        )

        fg_union = obj_cover.amax(dim=1)
        bg_mask = (1.0 - fg_union).clamp(0.0, 1.0)
        void_loss = zero
        void_log_prob = None
        if V > 0:
            void_log_prob = torch.logsumexp(owner_logits[:, K:].float(), dim=1) - torch.logsumexp(owner_logits.float(), dim=1)
            bg_area = bg_mask.sum(dim=-1)
            valid_bg = bg_area > 0.0
            per_bg = -(bg_mask * void_log_prob).sum(dim=-1) / bg_area.clamp_min(1.0)
            void_loss = per_bg[valid_bg].mean() if bool(valid_bg.any()) else zero

        void_w = float(getattr(self.config, "pgot_v14_void_weight", 0.5))
        loss = object_loss + void_w * void_loss
        if not torch.isfinite(loss):
            loss = zero

        pred = owner_logits.argmax(dim=1)
        best_cover, best_idx = obj_cover.max(dim=1)
        fg_mask = best_cover > 0.0
        hard_bg = ~fg_mask
        fg_acc = (
            (pred[fg_mask] == best_idx[fg_mask]).float().mean()
            if bool(fg_mask.any())
            else zero
        )
        bg_acc = (
            (pred[hard_bg] >= K).float().mean()
            if V > 0 and bool(hard_bg.any())
            else zero
        )
        void_prob = owner_probs[:, K:].sum(dim=1) if V > 0 else owner_probs.new_zeros(B, P)
        object_prob = owner_probs[:, :K].sum(dim=1)
        void_prob_on_fg = void_prob[fg_mask].mean() if bool(fg_mask.any()) else zero
        object_prob_on_bg = object_prob[hard_bg].mean() if bool(hard_bg.any()) else zero
        prob = owner_probs.float().clamp_min(1e-8)
        entropy = (-(prob * prob.log()).sum(dim=1)).mean()
        return {
            "loss": loss,
            "object_loss": object_loss.detach(),
            "void_loss": void_loss.detach(),
            "fg_acc": fg_acc.detach(),
            "bg_acc": bg_acc.detach(),
            "void_prob_on_fg": void_prob_on_fg.detach(),
            "object_prob_on_bg": object_prob_on_bg.detach(),
            "entropy": entropy.detach(),
            "object_count": valid_obj.float().sum(dim=1).mean().detach(),
        }

    def _pgot_v14_forward_features(
        self,
        *,
        images: torch.Tensor,
        target_images: torch.Tensor,
        caption_input_ids: torch.LongTensor,
        caption_attention_mask: torch.Tensor,
        ovt_positions_in_caption: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        seq = self._pgot_build_sequence_inputs(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
        )
        out = self.model(
            inputs_embeds=seq["inputs_embeds"],
            attention_bias=seq["attn_bias"],
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        if bool(getattr(self.config, "pgot_v21_enable", False)):
            ovt_pack = self._pgot_v21_build_ovt_states(
                hidden_states=hidden,
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
            )
        else:
            ovt_pack = self._pgot_v12_build_ovt_states(
                hidden_states=hidden,
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
            )
        B = hidden.shape[0]
        K = ovt_pack["object_valid"].shape[1]
        Q = int(seq["positions"]["rae_e"] - seq["positions"]["rae_s"])
        rae_hidden = hidden[:, seq["positions"]["rae_s"]:seq["positions"]["rae_e"], :]
        if bool(getattr(self.config, "pgot_v17_enable", False)):
            condition_hidden = rae_hidden
            owner_logits = hidden.new_zeros(B, K + ovt_pack["void_ovts"].shape[1], Q)
            owner_probs = owner_logits
        elif bool(getattr(self.config, "pgot_v21_enable", False)):
            query_base = self.get_model().latent_queries.unsqueeze(0).expand(
                B, -1, -1
            ).to(device=hidden.device, dtype=hidden.dtype)
            route = self.pgot_v21_router(
                query_base=query_base,
                object_ovts=ovt_pack["object_ovts_grouped"],
                object_token_valid=ovt_pack["object_token_valid"],
                void_ovts=ovt_pack["void_ovts"],
                void_valid=ovt_pack["void_valid"],
                temperature=float(getattr(self.config, "pgot_v21_temperature", getattr(self.config, "pgot_v14_route_temperature", 1.0))),
                position_weight=float(getattr(self.config, "pgot_v21_position_weight", getattr(self.config, "pgot_v14_position_weight", 1.0))),
            )
            condition_hidden = route["condition_hidden"]
            owner_logits = route["owner_logits"]
            owner_probs = route["owner_probs"]
            ovt_pack["object_valid"] = route["object_valid"]
        else:
            query_base = self.get_model().latent_queries.unsqueeze(0).expand(
                B, -1, -1
            ).to(device=hidden.device, dtype=hidden.dtype)
            route = self.pgot_v14_router(
                query_base=query_base,
                ovt_states=ovt_pack["ovt_states"],
                ovt_valid_mask=ovt_pack["ovt_valid"],
                temperature=float(getattr(self.config, "pgot_v14_route_temperature", 1.0)),
                position_weight=float(getattr(self.config, "pgot_v14_position_weight", 1.0)),
            )
            condition_hidden = route["condition_hidden"]
            owner_logits = route["owner_logits"]
            owner_probs = route["owner_probs"]
        seq.update(
            {
                "hidden": hidden,
                "rae_hidden": rae_hidden,
                "condition_hidden": condition_hidden,
                "outputs": out,
                "hidden_states": out.hidden_states if output_hidden_states else None,
                "img_hidden": hidden[:, seq["positions"]["img_s"]:seq["positions"]["img_e"], :],
                "owner_logits": owner_logits,
                "owner_probs": owner_probs,
                "object_probs": owner_probs[:, :K],
                "void_probs": owner_probs[:, K:],
                "object_valid": ovt_pack["object_valid"],
                "ovt_valid": ovt_pack["ovt_valid"],
                "ovt_states": ovt_pack["ovt_states"],
                "object_ovts": ovt_pack["object_ovts"],
                "void_ovts": ovt_pack["void_ovts"],
            }
        )
        return seq

    def _pgot_dit_ovt_cross_attn_enabled(self) -> bool:
        return bool(getattr(self.config, "pgot_dit_ovt_cross_attn_enable", False))

    def _pgot_prepare_dit_ovt_context(
        self,
        ovt_states: torch.Tensor,
        ovt_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project OVT/void hidden states into the DiT slot-context space."""
        if self.use_diff_head_projector:
            proj_dtype = next(self.diff_head_projector.parameters()).dtype
            context = self.diff_head_projector(ovt_states.to(dtype=proj_dtype))
        else:
            context = ovt_states
        context = torch.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        mask = ovt_valid.to(device=context.device, dtype=torch.bool)
        context = context.masked_fill(~mask.unsqueeze(-1), 0.0)
        return context, mask

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

    def _compute_core_all_layer_outside_loss(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str = "all",
        temperature: float = 1.0,
        void_weight: float = 1.0,
        tail_fraction: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Core PGOT loss for object OVTs and the optional residual VOID.

        Attention is the layer's exact post-LN, post-RoPE, GQA-expanded Q/K
        attention, re-normalized over image patches only.  We retain every head
        separately, then average uniformly over valid (layer, head, OVT)
        tuples.  No positive/inside distribution target is imposed: lowering
        outside mass is sufficient, and reconstruction is free to decide which
        patches inside the object are useful.  When present, VOID uses the
        complement of the union of valid OVT masks as its allowed region.
        """
        zero = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "loss": zero,
            "tail_loss": zero,
            "inside_mass": zero,
            "outside_mass": zero,
            "full_image_mass": zero,
            "worst_head_outside_mass": zero,
            "void_loss": zero,
            "void_inside_mass": zero,
            "void_outside_mass": zero,
            "void_full_image_mass": zero,
            "void_worst_head_outside_mass": zero,
            "void_valid_fraction": zero,
            "valid_ovt_count": zero,
            "num_layers": zero,
            "layer_metrics": {},
        }
        if hidden_states is None:
            return empty

        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        masks = gt_masks_per_ovt.float().clamp(0.0, 1.0)
        valid = ovt_valid_mask.to(device=masks.device, dtype=torch.bool)
        if not layers or not bool(valid.any()):
            return empty

        layer_losses = []
        layer_inside = []
        layer_full_image = []
        layer_worst_heads = []
        layer_void_losses = []
        layer_void_inside = []
        layer_void_full_image = []
        layer_void_worst_heads = []
        all_valid_outside = []
        layer_metrics = {}
        annotated_union = (masks * valid.unsqueeze(-1).float()).amax(
            dim=1, keepdim=True
        ).clamp(0.0, 1.0)
        void_target = (1.0 - annotated_union).clamp(0.0, 1.0)
        void_sample_valid = void_target.sum(dim=-1).squeeze(1) > 1e-6
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            object_full, void_full, object_patch, void_patch = (
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
            # [B,H,M]. Fractional boundary patches are weighted by their GT
            # overlap, rather than being forced to a hard binary decision.
            outside = (object_patch * (1.0 - masks)[:, None]).sum(dim=-1)
            inside = (object_patch * masks[:, None]).sum(dim=-1)
            full_image = object_full.sum(dim=-1)
            valid_bhm = valid[:, None].expand(-1, object_patch.shape[1], -1)
            outside_valid = outside[valid_bhm]
            inside_valid = inside[valid_bhm]
            full_valid = full_image[valid_bhm]
            if outside_valid.numel() == 0:
                continue

            layer_loss = outside_valid.mean()
            layer_in = inside_valid.mean()
            layer_full = full_valid.mean()
            per_head_denom = valid.float().sum().clamp_min(1.0)
            per_head = (
                outside * valid[:, None].float()
            ).sum(dim=(0, 2)) / per_head_denom
            layer_worst = per_head.max()

            layer_losses.append(layer_loss)
            layer_inside.append(layer_in)
            layer_full_image.append(layer_full)
            layer_worst_heads.append(layer_worst)
            all_valid_outside.append(outside_valid)
            layer_metrics[f"core_outside_layer_{layer_idx:02d}"] = layer_loss.detach()
            layer_metrics[f"core_inside_layer_{layer_idx:02d}"] = layer_in.detach()

            if void_patch.shape[2] > 0 and bool(void_sample_valid.any()):
                void_outside = (void_patch * annotated_union[:, None]).sum(dim=-1)
                void_inside = (void_patch * void_target[:, None]).sum(dim=-1)
                void_full_mass = void_full.sum(dim=-1)
                valid_bhv = void_sample_valid[:, None, None].expand(
                    -1, void_patch.shape[1], void_patch.shape[2]
                )
                void_outside_valid = void_outside[valid_bhv]
                void_inside_valid = void_inside[valid_bhv]
                void_full_valid = void_full_mass[valid_bhv]
                void_layer_loss = void_outside_valid.mean()
                void_layer_inside = void_inside_valid.mean()
                void_layer_full = void_full_valid.mean()
                void_per_head_denom = (
                    void_sample_valid.float().sum() * void_patch.shape[2]
                ).clamp_min(1.0)
                void_per_head = (
                    void_outside * void_sample_valid[:, None, None].float()
                ).sum(dim=(0, 2)) / void_per_head_denom
                layer_void_losses.append(void_layer_loss)
                layer_void_inside.append(void_layer_inside)
                layer_void_full_image.append(void_layer_full)
                layer_void_worst_heads.append(void_per_head.max())
                all_valid_outside.append(void_outside_valid)
                layer_metrics[f"core_void_outside_layer_{layer_idx:02d}"] = (
                    void_layer_loss.detach()
                )
                layer_metrics[f"core_void_inside_layer_{layer_idx:02d}"] = (
                    void_layer_inside.detach()
                )

        if not layer_losses:
            return empty

        object_raw_loss = torch.stack(layer_losses).mean()
        void_raw_loss = (
            torch.stack(layer_void_losses).mean() if layer_void_losses else zero
        )
        raw_loss = object_raw_loss + float(void_weight) * void_raw_loss
        outside_values = torch.cat(all_valid_outside)
        fraction = min(max(float(tail_fraction), 0.0), 1.0)
        if fraction > 0.0:
            k = max(1, int(math.ceil(outside_values.numel() * fraction)))
            tail_loss = torch.topk(outside_values, k=k, largest=True).values.mean()
        else:
            tail_loss = zero
        return {
            "loss": raw_loss,
            "tail_loss": tail_loss,
            "inside_mass": torch.stack(layer_inside).mean().detach(),
            "outside_mass": object_raw_loss.detach(),
            "full_image_mass": torch.stack(layer_full_image).mean().detach(),
            "worst_head_outside_mass": torch.stack(layer_worst_heads).mean().detach(),
            "void_loss": void_raw_loss,
            "void_inside_mass": (
                torch.stack(layer_void_inside).mean().detach()
                if layer_void_inside else zero
            ),
            "void_outside_mass": void_raw_loss.detach(),
            "void_full_image_mass": (
                torch.stack(layer_void_full_image).mean().detach()
                if layer_void_full_image else zero
            ),
            "void_worst_head_outside_mass": (
                torch.stack(layer_void_worst_heads).mean().detach()
                if layer_void_worst_heads else zero
            ),
            "void_valid_fraction": void_sample_valid.float().mean().detach(),
            "valid_ovt_count": valid.float().sum(dim=1).mean().detach(),
            "num_layers": raw_loss.new_tensor(float(len(layer_losses))).detach(),
            "layer_metrics": layer_metrics,
        }

    def _compute_core_register_foreground_loss(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        gt_masks_per_ovt: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        layers_spec: str = "all",
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Keep every background register off the union of object instances.

        This reconstructs each selected layer/head's exact post-RoPE register
        query -> image-patch attention and renormalizes over image patches.  It
        is therefore the register analogue of the core OVT outside-mass loss.
        """
        zero = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "loss": zero,
            "foreground_mass": zero,
            "background_mass": zero,
            "full_image_mass": zero,
            "worst_head_foreground_mass": zero,
            "num_layers": zero,
            "layer_metrics": {},
        }
        reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
        n_register = max(reg_e - reg_s, 0)
        if hidden_states is None or n_register <= 0:
            return empty
        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if not layers:
            return empty

        B = gt_masks_per_ovt.shape[0]
        register_positions = torch.arange(
            reg_s, reg_e, device=gt_masks_per_ovt.device, dtype=torch.long
        ).view(1, n_register).expand(B, -1)
        register_valid = torch.ones(
            B, n_register, device=gt_masks_per_ovt.device, dtype=torch.bool
        )
        masks = gt_masks_per_ovt.float().clamp(0.0, 1.0)
        valid = ovt_valid_mask.to(masks.device, dtype=torch.bool)
        foreground = (masks * valid.unsqueeze(-1).float()).amax(
            dim=1, keepdim=True
        ).clamp(0.0, 1.0)
        background = (1.0 - foreground).clamp(0.0, 1.0)

        fg_values, bg_values, full_values, worst_values = [], [], [], []
        layer_metrics = {}
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            register_full, _, register_patch, _ = (
                self._compute_exact_llm_attention_components_for_layer(
                    layer_input=hidden_states[layer_idx],
                    layer_idx=layer_idx,
                    attention_bias=attention_bias,
                    positions=positions,
                    ovt_abs_positions=register_positions,
                    ovt_valid_mask=register_valid,
                    patch_temperature=temperature,
                )
            )
            fg_mass = (register_patch * foreground[:, None]).sum(dim=-1)
            bg_mass = (register_patch * background[:, None]).sum(dim=-1)
            full_mass = register_full.sum(dim=-1)
            layer_fg = fg_mass.mean()
            layer_bg = bg_mass.mean()
            layer_full = full_mass.mean()
            layer_worst = fg_mass.mean(dim=(0, 2)).max()
            fg_values.append(layer_fg)
            bg_values.append(layer_bg)
            full_values.append(layer_full)
            worst_values.append(layer_worst)
            layer_metrics[f"core_register_fg_layer_{layer_idx:02d}"] = layer_fg.detach()
            layer_metrics[f"core_register_bg_layer_{layer_idx:02d}"] = layer_bg.detach()

        if not fg_values:
            return empty
        loss = torch.stack(fg_values).mean()
        return {
            "loss": loss,
            "foreground_mass": loss.detach(),
            "background_mass": torch.stack(bg_values).mean().detach(),
            "full_image_mass": torch.stack(full_values).mean().detach(),
            "worst_head_foreground_mass": torch.stack(worst_values).mean().detach(),
            "num_layers": loss.new_tensor(float(len(fg_values))).detach(),
            "layer_metrics": layer_metrics,
        }

    def _compute_e3_joint_attention_losses(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str = "all",
        temperature: float = 1.0,
        bg_weight: float = 0.25,
        full_inside_target: float = 0.0,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """E3's joint all-layer OVT/register supervision.

        One exact post-LN/post-RoPE Q/K reconstruction per layer yields both
        OVT and register image-patch maps.  We retain the core outside-only
        objectives for every head/query, then add a head-averaged patch-owner
        CE over K object groups plus one residual-background register class.
        No target distribution is imposed *within* an object mask.
        """
        zero = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "object_loss": zero,
            "object_inside_mass": zero,
            "object_outside_mass": zero,
            "object_full_image_mass": zero,
            "object_full_inside_floor_loss": zero,
            "object_full_inside_mass": zero,
            "object_full_outside_mass": zero,
            "object_full_inside_satisfied_fraction": zero,
            "object_worst_head_outside_mass": zero,
            "register_loss": zero,
            "register_foreground_mass": zero,
            "register_background_mass": zero,
            "register_full_image_mass": zero,
            "register_worst_head_foreground_mass": zero,
            "competition_loss": zero,
            "competition_fg_loss": zero,
            "competition_bg_loss": zero,
            "competition_fg_acc": zero,
            "competition_bg_acc": zero,
            "competition_entropy": zero,
            "competition_register_prob_on_fg": zero,
            "competition_object_prob_on_bg": zero,
            "competition_fg_fraction": zero,
            "valid_ovt_count": zero,
            "num_layers": zero,
            "layer_metrics": {},
        }
        if hidden_states is None:
            return empty

        B, M_total, P = gt_masks_per_ovt.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M_total // n
        reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
        R = max(reg_e - reg_s, 0)
        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if K <= 0 or R <= 0 or not layers:
            return empty

        M = K * n
        masks = gt_masks_per_ovt[:, :M].float().clamp(0.0, 1.0)
        valid = ovt_valid_mask[:, :M].to(device=masks.device, dtype=torch.bool)
        object_valid = valid.reshape(B, K, n).any(dim=2)
        if not bool(object_valid.any()):
            return empty

        grouped_masks = masks.reshape(B, K, n, P).amax(dim=2)
        grouped_masks = grouped_masks * object_valid.unsqueeze(-1).float()
        # Panoptic instances are disjoint in pixels, but bilinear patch
        # resampling can yield fractional contributions from adjacent objects.
        # Sum-and-clamp is therefore the true soft union; max would leave such
        # foreground boundary mass available to the background register.
        foreground = grouped_masks.sum(dim=1, keepdim=True).clamp(0.0, 1.0)
        background = (1.0 - foreground).clamp(0.0, 1.0)
        target_mass = grouped_masks.sum(dim=1)
        fg_patch = target_mass > eps
        bg_patch = ~fg_patch
        target_fg = grouped_masks / target_mass[:, None, :].clamp_min(eps)
        target_class = target_fg.argmax(dim=1)

        register_positions = torch.arange(
            reg_s, reg_e, device=masks.device, dtype=torch.long
        ).view(1, R).expand(B, -1)
        register_valid = torch.ones(B, R, device=masks.device, dtype=torch.bool)
        joint_positions = torch.cat([ovt_abs_positions[:, :M], register_positions], dim=1)
        joint_valid = torch.cat([valid, register_valid], dim=1)

        object_losses, object_inside_values, object_full_values = [], [], []
        full_inside_floor_values, full_inside_values = [], []
        full_outside_values, full_inside_satisfied_values = [], []
        object_worst_values, register_fg_values, register_bg_values = [], [], []
        register_full_values, register_worst_values = [], []
        competition_values, comp_fg_values, comp_bg_values = [], [], []
        fg_acc_values, bg_acc_values, entropy_values = [], [], []
        register_on_fg_values, object_on_bg_values = [], []
        layer_metrics: Dict[str, torch.Tensor] = {}
        bg_w = max(float(bg_weight), 0.0)

        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            joint_full, _, joint_patch, _ = (
                self._compute_exact_llm_attention_components_for_layer(
                    layer_input=hidden_states[layer_idx],
                    layer_idx=layer_idx,
                    attention_bias=attention_bias,
                    positions=positions,
                    ovt_abs_positions=joint_positions,
                    ovt_valid_mask=joint_valid,
                    patch_temperature=temperature,
                )
            )
            object_full, register_full = joint_full[:, :, :M], joint_full[:, :, M:M + R]
            object_patch, register_patch = joint_patch[:, :, :M], joint_patch[:, :, M:M + R]
            H = object_patch.shape[1]

            # Original core outside-only loss: every layer/head/OVT counts.
            object_outside = (object_patch * (1.0 - masks)[:, None]).sum(dim=-1)
            object_inside = (object_patch * masks[:, None]).sum(dim=-1)
            object_full_mass = object_full.sum(dim=-1)
            object_full_inside = (object_full * masks[:, None]).sum(dim=-1)
            object_full_outside = (
                object_full * (1.0 - masks)[:, None].clamp(0.0, 1.0)
            ).sum(dim=-1)
            valid_bhm = valid[:, None].expand(-1, H, -1)
            outside_valid = object_outside[valid_bhm]
            inside_valid = object_inside[valid_bhm]
            full_valid = object_full_mass[valid_bhm]
            if outside_valid.numel() == 0:
                continue
            layer_object_loss = outside_valid.mean()
            object_losses.append(layer_object_loss)
            object_inside_values.append(inside_valid.mean())
            object_full_values.append(full_valid.mean())
            full_inside_valid = object_full_inside[valid_bhm]
            full_outside_valid = object_full_outside[valid_bhm]
            full_inside_values.append(full_inside_valid.mean())
            full_outside_values.append(full_outside_valid.mean())
            if float(full_inside_target) > 0.0:
                target_log = math.log(max(float(full_inside_target), eps))
                floor = F.relu(
                    object_full_inside.clamp_min(eps).log().new_tensor(target_log)
                    - object_full_inside.clamp_min(eps).log()
                )
                full_inside_floor_values.append(floor[valid_bhm].mean())
                full_inside_satisfied_values.append(
                    (full_inside_valid >= float(full_inside_target)).float().mean()
                )
            per_head_denom = valid.float().sum().clamp_min(1.0)
            object_worst_values.append(
                (object_outside * valid[:, None].float()).sum(dim=(0, 2)).div(per_head_denom).max()
            )

            # Register analogue: every layer/head/register avoids object union.
            register_fg = (register_patch * foreground[:, None]).sum(dim=-1)
            register_bg = (register_patch * background[:, None]).sum(dim=-1)
            register_full_mass = register_full.sum(dim=-1)
            layer_register_loss = register_fg.mean()
            register_fg_values.append(layer_register_loss)
            register_bg_values.append(register_bg.mean())
            register_full_values.append(register_full_mass.mean())
            register_worst_values.append(register_fg.mean(dim=(0, 2)).max())

            # Patch ownership uses head-mean maps; heads remain individually
            # supervised by the two outside objectives above.
            object_maps = object_patch.reshape(B, H, K, n, P).mean(dim=3).mean(dim=1)
            register_map = register_patch.mean(dim=(1, 2))
            object_scores = object_maps.clamp_min(eps).log()
            object_scores = object_scores.masked_fill(~object_valid.unsqueeze(-1), -1e4)
            owner_scores = torch.cat([object_scores, register_map.clamp_min(eps).log().unsqueeze(1)], dim=1)
            owner_probs = F.softmax(owner_scores.transpose(1, 2), dim=-1)
            object_probs = owner_probs[..., :K]
            register_probs = owner_probs[..., K]

            fg_loss = zero
            bg_loss = zero
            if bool(fg_patch.any()):
                fg_ce = -(target_fg.transpose(1, 2) * object_probs.clamp_min(eps).log()).sum(dim=-1)
                fg_loss = fg_ce[fg_patch].mean()
                fg_acc_values.append(
                    (object_probs.argmax(dim=-1)[fg_patch] == target_class[fg_patch]).float().mean()
                )
                register_on_fg_values.append(register_probs[fg_patch].mean())
                entropy_values.append(
                    (-(owner_probs.clamp_min(eps) * owner_probs.clamp_min(eps).log()).sum(dim=-1))[fg_patch].mean()
                )
            if bool(bg_patch.any()):
                bg_loss = -register_probs[bg_patch].clamp_min(eps).log().mean()
                bg_acc_values.append((owner_probs.argmax(dim=-1)[bg_patch] == K).float().mean())
                object_on_bg_values.append(object_probs.sum(dim=-1)[bg_patch].mean())
            if bool(fg_patch.any()) and bool(bg_patch.any()):
                comp_loss = (fg_loss + bg_w * bg_loss) / (1.0 + bg_w)
            elif bool(fg_patch.any()):
                comp_loss = fg_loss
            else:
                comp_loss = bg_loss
            competition_values.append(comp_loss)
            comp_fg_values.append(fg_loss)
            comp_bg_values.append(bg_loss)

            layer_metrics[f"core_outside_layer_{layer_idx:02d}"] = layer_object_loss.detach()
            layer_metrics[f"core_inside_layer_{layer_idx:02d}"] = object_inside_values[-1].detach()
            layer_metrics[f"core_register_fg_layer_{layer_idx:02d}"] = layer_register_loss.detach()
            layer_metrics[f"core_register_bg_layer_{layer_idx:02d}"] = register_bg_values[-1].detach()
            layer_metrics[f"e3_competition_layer_{layer_idx:02d}"] = comp_loss.detach()

        if not object_losses:
            return empty

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else zero

        object_loss = mean_or_zero(object_losses)
        register_loss = mean_or_zero(register_fg_values)
        return {
            "object_loss": object_loss,
            "object_inside_mass": mean_or_zero(object_inside_values).detach(),
            "object_outside_mass": object_loss.detach(),
            "object_full_image_mass": mean_or_zero(object_full_values).detach(),
            "object_full_inside_floor_loss": mean_or_zero(
                full_inside_floor_values
            ),
            "object_full_inside_mass": mean_or_zero(full_inside_values).detach(),
            "object_full_outside_mass": mean_or_zero(full_outside_values).detach(),
            "object_full_inside_satisfied_fraction": mean_or_zero(
                full_inside_satisfied_values
            ).detach(),
            "object_worst_head_outside_mass": mean_or_zero(object_worst_values).detach(),
            "register_loss": register_loss,
            "register_foreground_mass": register_loss.detach(),
            "register_background_mass": mean_or_zero(register_bg_values).detach(),
            "register_full_image_mass": mean_or_zero(register_full_values).detach(),
            "register_worst_head_foreground_mass": mean_or_zero(register_worst_values).detach(),
            "competition_loss": mean_or_zero(competition_values),
            "competition_fg_loss": mean_or_zero(comp_fg_values).detach(),
            "competition_bg_loss": mean_or_zero(comp_bg_values).detach(),
            "competition_fg_acc": mean_or_zero(fg_acc_values).detach(),
            "competition_bg_acc": mean_or_zero(bg_acc_values).detach(),
            "competition_entropy": mean_or_zero(entropy_values).detach(),
            "competition_register_prob_on_fg": mean_or_zero(register_on_fg_values).detach(),
            "competition_object_prob_on_bg": mean_or_zero(object_on_bg_values).detach(),
            "competition_fg_fraction": fg_patch.float().mean().detach(),
            "valid_ovt_count": valid.float().sum(dim=1).mean().detach(),
            "num_layers": object_loss.new_tensor(float(len(object_losses))).detach(),
            "layer_metrics": layer_metrics,
        }

    def _compute_exact_e4_rae_owner_attention_for_layer(
        self,
        *,
        layer_input: torch.Tensor,
        layer_idx: int,
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct E4 RAE-query attention without owner renormalization.

        Probabilities come from the selected transformer layer's actual
        post-LN/post-RoPE full-key softmax. OVT/register probabilities therefore
        remain small when a query escapes through its own state or another RAE
        query; this is essential for detecting and penalizing the shortcut.
        """
        block = self.model.layers[layer_idx]
        attn = block.self_attn
        attn_input = block.input_layernorm(layer_input)
        B, L, _ = attn_input.shape
        rae_s, rae_e = int(positions["rae_s"]), int(positions["rae_e"])
        reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
        Q = max(rae_e - rae_s, 0)
        M = int(ovt_abs_positions.shape[1])
        R = max(reg_e - reg_s, 0)

        q_proj = attn.q_proj(attn_input[:, rae_s:rae_e])
        k_proj = attn.k_proj(attn_input)
        num_heads = int(getattr(attn, "num_heads", self.config.num_attention_heads))
        num_kv_heads = int(
            getattr(attn, "num_key_value_heads", self.config.num_key_value_heads)
        )
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // num_heads))
        q = q_proj.view(B, Q, num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(B, L, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(k, seq_len=L)
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        all_positions = torch.arange(L, device=q.device, dtype=torch.long)
        query_positions = torch.arange(rae_s, rae_e, device=q.device, dtype=torch.long)
        k_cos = cos[all_positions].view(1, 1, L, head_dim)
        k_sin = sin[all_positions].view(1, 1, L, head_dim)
        q_cos = cos[query_positions].view(1, 1, Q, head_dim)
        q_sin = sin[query_positions].view(1, 1, Q, head_dim)
        q = (q * q_cos) + (self._pgot_rotate_half(q) * q_sin)
        k = (k * k_cos) + (self._pgot_rotate_half(k) * k_sin)
        if num_kv_heads != num_heads:
            groups = int(
                getattr(attn, "num_key_value_groups", num_heads // num_kv_heads)
            )
            k = k.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(2, 3))
        scores = scores / math.sqrt(float(head_dim))
        scores = scores + attention_bias[:, :, rae_s:rae_e, :].float()
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)

        safe_ovt = ovt_abs_positions.clamp(min=0, max=L - 1)
        object_probs = probs.gather(
            dim=-1,
            index=safe_ovt[:, None, None, :].expand(B, num_heads, Q, M),
        )
        object_probs = object_probs * ovt_valid_mask[:, None, None, :].float()
        register_probs = probs[..., reg_s:reg_e]
        rae_probs = probs[..., rae_s:rae_e]
        self_probs = torch.diagonal(rae_probs, dim1=-2, dim2=-1)
        other_rae_probs = (rae_probs.sum(dim=-1) - self_probs).clamp_min(0.0)
        return {
            "object_probs": object_probs,
            "register_probs": register_probs,
            "self_probs": self_probs,
            "other_rae_probs": other_rae_probs,
        }

    def _compute_e4_rae_binding_loss(
        self,
        *,
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str = "last8",
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Bind each 16x16 RAE query to its GT object OVT or registers.

        The loss uses actual full-key probabilities. It never converts a
        predicted 32x32 ownership map into a DiT condition and never
        renormalizes away RAE-self attention.
        """
        z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "loss": z,
            "correct_owner_mass": z,
            "owner_total_mass": z,
            "object_mass_on_fg": z,
            "register_mass_on_fg": z,
            "register_mass_on_bg": z,
            "self_mass": z,
            "other_rae_mass": z,
            "fg_acc": z,
            "bg_acc": z,
            "entropy": z,
            "fg_fraction": z,
            "num_layers": z,
            "layer_metrics": {},
        }
        if hidden_states is None:
            return empty

        B, M_total, P = gt_masks_per_ovt.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M_total // n
        M = K * n
        Q = int(positions["rae_e"] - positions["rae_s"])
        R = int(positions["reg_e"] - positions["reg_s"])
        if K <= 0 or Q <= 0 or R <= 0:
            return empty

        masks = gt_masks_per_ovt[:, :M].float().clamp(0.0, 1.0)
        if P != Q:
            src_side = int(math.isqrt(P))
            dst_side = int(math.isqrt(Q))
            if src_side * src_side != P or dst_side * dst_side != Q:
                raise ValueError(
                    "E4 RAE binding expects square mask/query grids, got "
                    f"P={P}, Q={Q}"
                )
            masks = F.interpolate(
                masks.reshape(B * M, 1, src_side, src_side),
                size=(dst_side, dst_side),
                mode="area",
            ).reshape(B, M, Q)

        token_valid = ovt_valid_mask[:, :M].to(dtype=torch.bool)
        object_valid = token_valid.reshape(B, K, n).any(dim=2)
        object_target = masks.reshape(B, K, n, Q).amax(dim=2)
        object_target = object_target * object_valid.unsqueeze(-1).float()
        foreground = object_target.sum(dim=1).clamp(0.0, 1.0)
        background = (1.0 - foreground).clamp(0.0, 1.0)
        target = torch.cat([object_target, background.unsqueeze(1)], dim=1)
        target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)
        target_q = target.transpose(1, 2)
        hard_target = target_q.argmax(dim=-1)
        fg_weight = foreground
        bg_weight = background

        losses: List[torch.Tensor] = []
        correct_values, owner_values = [], []
        object_fg_values, register_fg_values, register_bg_values = [], [], []
        self_values, other_rae_values = [], []
        fg_acc_values, bg_acc_values, entropy_values = [], [], []
        layer_metrics: Dict[str, torch.Tensor] = {}
        layers = _resolve_layer_spec(
            layers_spec,
            int(getattr(self.config, "num_hidden_layers", 0)),
        )

        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(hidden_states) - 1:
                continue
            attn = self._compute_exact_e4_rae_owner_attention_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions[:, :M],
                ovt_valid_mask=token_valid,
            )
            object_probs = (
                attn["object_probs"]
                .reshape(B, -1, Q, K, n)
                .sum(dim=-1)
                .mean(dim=1)
            )
            register_probs = attn["register_probs"].sum(dim=-1).mean(dim=1)
            owner_probs = torch.cat(
                [object_probs, register_probs.unsqueeze(-1)],
                dim=-1,
            )
            per_query_loss = -(
                target_q * owner_probs.clamp_min(eps).log()
            ).sum(dim=-1)
            losses.append(per_query_loss.mean())

            correct = (target_q * owner_probs).sum(dim=-1)
            owner_total = owner_probs.sum(dim=-1)
            object_mass = object_probs.sum(dim=-1)
            self_mass = attn["self_probs"].mean(dim=1)
            other_rae_mass = attn["other_rae_probs"].mean(dim=1)
            fg_denom = fg_weight.sum().clamp_min(1.0)
            bg_denom = bg_weight.sum().clamp_min(1.0)
            correct_values.append(correct.mean())
            owner_values.append(owner_total.mean())
            object_fg_values.append((object_mass * fg_weight).sum() / fg_denom)
            register_fg_values.append(
                (register_probs * fg_weight).sum() / fg_denom
            )
            register_bg_values.append(
                (register_probs * bg_weight).sum() / bg_denom
            )
            self_values.append(self_mass.mean())
            other_rae_values.append(other_rae_mass.mean())

            prediction = owner_probs.argmax(dim=-1)
            fg_cells = foreground > 0.5
            bg_cells = ~fg_cells
            if bool(fg_cells.any()):
                fg_acc_values.append(
                    (prediction[fg_cells] == hard_target[fg_cells]).float().mean()
                )
            if bool(bg_cells.any()):
                bg_acc_values.append(
                    (prediction[bg_cells] == K).float().mean()
                )
            owner_norm = owner_probs / owner_total.unsqueeze(-1).clamp_min(eps)
            entropy_values.append(
                -(
                    owner_norm.clamp_min(eps)
                    * owner_norm.clamp_min(eps).log()
                ).sum(dim=-1).mean()
            )
            layer_metrics[f"e4_rae_bind_layer_{layer_idx:02d}"] = (
                per_query_loss.mean().detach()
            )
            layer_metrics[f"e4_rae_ovt_mass_layer_{layer_idx:02d}"] = (
                object_mass.mean().detach()
            )
            layer_metrics[f"e4_rae_register_mass_layer_{layer_idx:02d}"] = (
                register_probs.mean().detach()
            )
            layer_metrics[f"e4_rae_self_mass_layer_{layer_idx:02d}"] = (
                self_mass.mean().detach()
            )

        if not losses:
            return empty

        def mean_or_zero(values):
            return torch.stack(values).mean() if values else z

        return {
            "loss": mean_or_zero(losses),
            "correct_owner_mass": mean_or_zero(correct_values).detach(),
            "owner_total_mass": mean_or_zero(owner_values).detach(),
            "object_mass_on_fg": mean_or_zero(object_fg_values).detach(),
            "register_mass_on_fg": mean_or_zero(register_fg_values).detach(),
            "register_mass_on_bg": mean_or_zero(register_bg_values).detach(),
            "self_mass": mean_or_zero(self_values).detach(),
            "other_rae_mass": mean_or_zero(other_rae_values).detach(),
            "fg_acc": mean_or_zero(fg_acc_values).detach(),
            "bg_acc": mean_or_zero(bg_acc_values).detach(),
            "entropy": mean_or_zero(entropy_values).detach(),
            "fg_fraction": foreground.mean().detach(),
            "num_layers": z.new_tensor(float(len(losses))).detach(),
            "layer_metrics": layer_metrics,
        }

    def _compute_exact_rae_to_ovtvoid_attention_for_layer(
        self,
        *,
        layer_input: torch.Tensor,
        layer_idx: int,
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        eps: float = 1e-6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return RAE query attention renormalized over OVT+void keys.

        The full-key softmax is reconstructed from the selected layer's real
        Q/K/RoPE/bias path. We then gather the OVT and void columns and
        renormalize over those columns only for the ownership CE target.
        """
        block = self.model.layers[layer_idx]
        attn = block.self_attn
        attn_input = block.input_layernorm(layer_input)
        B, L, _ = attn_input.shape
        rae_s = int(positions["rae_s"])
        rae_e = int(positions["rae_e"])
        Q = max(rae_e - rae_s, 0)
        M = int(ovt_abs_positions.shape[1])
        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        V = max(void_e - void_s, 0)

        q_input = attn_input[:, rae_s:rae_e, :]
        q_proj = attn.q_proj(q_input)
        k_proj = attn.k_proj(attn_input)
        num_heads = int(getattr(attn, "num_heads", self.config.num_attention_heads))
        num_kv_heads = int(
            getattr(attn, "num_key_value_heads", self.config.num_key_value_heads)
        )
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // num_heads))

        q = q_proj.view(B, Q, num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(B, L, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(k, seq_len=L)
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        all_positions = torch.arange(L, device=q.device, dtype=torch.long)
        query_positions = torch.arange(rae_s, rae_e, device=q.device, dtype=torch.long)
        k_cos = cos[all_positions].view(1, 1, L, head_dim)
        k_sin = sin[all_positions].view(1, 1, L, head_dim)
        q_cos = cos[query_positions].view(1, 1, Q, head_dim)
        q_sin = sin[query_positions].view(1, 1, Q, head_dim)
        q = (q * q_cos) + (self._pgot_rotate_half(q) * q_sin)
        k = (k * k_cos) + (self._pgot_rotate_half(k) * k_sin)

        if num_kv_heads != num_heads:
            groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
            k = k.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(2, 3))
        scores = scores / math.sqrt(float(head_dim))
        scores = scores + attention_bias[:, :, rae_s:rae_e, :].float()
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)

        if V > 0:
            void_positions = torch.arange(
                void_s, void_e, device=attn_input.device, dtype=torch.long
            ).view(1, V).expand(B, -1)
            key_positions = torch.cat([ovt_abs_positions, void_positions], dim=1)
            key_valid = torch.cat(
                [
                    ovt_valid_mask,
                    torch.ones(B, V, device=attn_input.device, dtype=torch.bool),
                ],
                dim=1,
            )
        else:
            key_positions = ovt_abs_positions
            key_valid = ovt_valid_mask

        safe_keys = key_positions.clamp(min=0, max=L - 1)
        selected = probs.gather(
            dim=-1,
            index=safe_keys[:, None, None, :].expand(B, num_heads, Q, M + V),
        )
        selected = selected * key_valid[:, None, None, :].float()
        selected_mass = selected.sum(dim=-1)
        selected_norm = selected / selected_mass.unsqueeze(-1).clamp_min(eps)
        return selected_norm, selected_mass

    def _compute_v17_ownership_loss(
        self,
        *,
        hidden_states: Optional[Tuple[torch.Tensor, ...]],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        layers_spec: str,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Supervise RAE-query -> OVT ownership with soft GT patch masks."""
        z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        if hidden_states is None:
            return {"loss": z, "mass": z, "ovt_mass": z, "void_mass": z, "entropy": z, "acc": z, "valid_fraction": z}

        B, M_total, P = gt_masks_per_ovt.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M_total // n
        Q = int(positions["rae_e"] - positions["rae_s"])
        void_s = int(positions.get("null_bg_s", 0))
        void_e = int(positions.get("null_bg_e", void_s))
        V = max(void_e - void_s, 0)
        if K <= 0 or Q <= 0:
            return {"loss": z, "mass": z, "ovt_mass": z, "void_mass": z, "entropy": z, "acc": z, "valid_fraction": z}

        M = K * n
        obj_valid = ovt_valid_mask[:, :M].reshape(B, K, n).any(dim=2)
        obj_masks = gt_masks_per_ovt[:, :M].reshape(B, K, n, P).amax(dim=2)
        obj_masks = obj_masks * obj_valid.unsqueeze(-1).to(obj_masks.dtype)

        if P != Q:
            src_side = int(math.isqrt(P))
            dst_side = int(math.isqrt(Q))
            if src_side * src_side != P or dst_side * dst_side != Q:
                raise ValueError(f"V17 ownership expects square masks/query grid, got P={P}, Q={Q}")
            obj_masks = F.interpolate(
                obj_masks.reshape(B * K, 1, src_side, src_side),
                size=(dst_side, dst_side),
                mode="area",
            ).reshape(B, K, Q)

        target_mass = obj_masks.sum(dim=1)
        valid_patch = target_mass > eps
        target = (obj_masks / target_mass[:, None, :].clamp_min(eps)).transpose(1, 2)
        valid_fraction = valid_patch.float().mean()

        n_layers = int(getattr(self.config, "num_hidden_layers", 0))
        layers = _resolve_layer_spec(layers_spec, n_layers)
        losses: List[torch.Tensor] = []
        masses: List[torch.Tensor] = []
        ovt_masses: List[torch.Tensor] = []
        void_masses: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        accs: List[torch.Tensor] = []
        target_argmax = target.argmax(dim=-1)

        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= len(hidden_states) - 1:
                continue
            selected_probs, selected_mass = self._compute_exact_rae_to_ovtvoid_attention_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions[:, :M],
                ovt_valid_mask=ovt_valid_mask[:, :M],
                eps=eps,
            )
            H = selected_probs.shape[1]
            ovt_probs = selected_probs[..., :M].reshape(B, H, Q, K, n).sum(dim=-1)
            if V > 0:
                void_probs = selected_probs[..., M:M + V].sum(dim=-1)
            else:
                void_probs = selected_probs.new_zeros(B, H, Q)
            valid_bhq = valid_patch[:, None, :].expand(B, H, Q)
            if valid_bhq.any():
                ce = -(target[:, None, :, :] * ovt_probs.clamp_min(eps).log()).sum(dim=-1)
                losses.append(ce[valid_bhq].mean())
                probs_with_void = torch.cat([ovt_probs, void_probs.unsqueeze(-1)], dim=-1)
                entropy = -(probs_with_void.clamp_min(eps) * probs_with_void.clamp_min(eps).log()).sum(dim=-1)
                entropies.append(entropy[valid_bhq].mean())
                pred = ovt_probs.argmax(dim=-1)
                accs.append((pred[valid_bhq] == target_argmax[:, None, :].expand(B, H, Q)[valid_bhq]).float().mean())
                masses.append(selected_mass[valid_bhq].mean())
                raw_ovt_mass = selected_mass.new_zeros(selected_mass.shape)
                raw_void_mass = selected_mass.new_zeros(selected_mass.shape)
                raw_ovt_mass = selected_probs[..., :M].sum(dim=-1) * selected_mass
                if V > 0:
                    raw_void_mass = selected_probs[..., M:M + V].sum(dim=-1) * selected_mass
                ovt_masses.append(raw_ovt_mass[valid_bhq].mean())
                void_masses.append(raw_void_mass[valid_bhq].mean())

        if not losses:
            return {"loss": z, "mass": z, "ovt_mass": z, "void_mass": z, "entropy": z, "acc": z, "valid_fraction": valid_fraction.detach()}

        return {
            "loss": torch.stack(losses).mean(),
            "mass": torch.stack(masses).mean().detach(),
            "ovt_mass": torch.stack(ovt_masses).mean().detach(),
            "void_mass": torch.stack(void_masses).mean().detach(),
            "entropy": torch.stack(entropies).mean().detach(),
            "acc": torch.stack(accs).mean().detach(),
            "valid_fraction": valid_fraction.detach(),
        }

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

    def _compute_v22_attention_competition_loss(
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
        include_void: bool = False,
        bg_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        """Weak patch->object CE from LLM-internal OVT image attention maps.

        The selected LLM attention maps are first normalized over image patches
        per OVT token. For each object group, we average over its OVTs and heads,
        then run a per-patch softmax across object groups. GT masks supervise
        only which object owns foreground patches; the object appearance/content
        inside that region remains governed by reconstruction.
        """
        z = gt_masks_per_ovt.new_zeros((), dtype=torch.float32)
        empty = {
            "loss": z,
            "fg_acc": z,
            "bg_acc": z,
            "entropy": z,
            "valid_fraction": z,
            "object_inside_mass": z,
            "object_outside_mass": z,
            "void_bg_mass": z,
        }
        if hidden_states is None:
            return empty

        B, M_total, P = gt_masks_per_ovt.shape
        n = max(int(self.pgot_n_ovt_per_object), 1)
        K = M_total // n
        if K <= 0:
            return empty

        M = K * n
        masks = gt_masks_per_ovt[:, :M].float().clamp(0.0, 1.0).reshape(B, K, n, P).amax(dim=2)
        object_valid = ovt_valid_mask[:, :M].reshape(B, K, n).any(dim=2)
        masks = masks * object_valid.unsqueeze(-1).float()
        target_mass = masks.sum(dim=1)
        fg_patch = target_mass > eps
        bg_patch = target_mass <= eps
        valid_fraction = fg_patch.float().mean()
        if not fg_patch.any() and not (include_void and bg_patch.any()):
            empty["valid_fraction"] = valid_fraction.detach()
            return empty

        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if not layers:
            empty["valid_fraction"] = valid_fraction.detach()
            return empty

        temp = max(float(temperature), 1e-6)
        bg_w = max(float(bg_weight), 0.0)
        losses: List[torch.Tensor] = []
        fg_accs: List[torch.Tensor] = []
        bg_accs: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []
        inside_masses: List[torch.Tensor] = []
        outside_masses: List[torch.Tensor] = []
        void_bg_masses: List[torch.Tensor] = []
        target_fg = (masks / target_mass[:, None, :].clamp_min(eps)).transpose(1, 2)
        target_argmax = target_fg.argmax(dim=-1)

        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            (
                _object_full,
                _void_full,
                object_patch,
                void_patch,
            ) = self._compute_exact_llm_attention_components_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions[:, :M],
                ovt_valid_mask=ovt_valid_mask[:, :M],
                patch_temperature=temp,
            )
            H = object_patch.shape[1]
            object_maps = object_patch.reshape(B, H, K, n, P).mean(dim=3)
            object_maps = object_maps * object_valid[:, None, :, None].float()
            class_scores = object_maps.clamp_min(eps).log().mean(dim=1)

            if include_void and void_patch.shape[2] > 0:
                void_map = void_patch.mean(dim=(1, 2)).clamp_min(eps).log().unsqueeze(1)
                class_scores = torch.cat([class_scores, void_map], dim=1)

            probs = F.softmax(class_scores.transpose(1, 2), dim=-1)
            obj_probs = probs[..., :K]
            if fg_patch.any():
                ce = -(target_fg * obj_probs.clamp_min(eps).log()).sum(dim=-1)
                losses.append(ce[fg_patch].mean())
                pred = obj_probs.argmax(dim=-1)
                fg_accs.append((pred[fg_patch] == target_argmax[fg_patch]).float().mean())
                entropy = -(probs.clamp_min(eps) * probs.clamp_min(eps).log()).sum(dim=-1)
                entropies.append(entropy[fg_patch].mean())

                object_group_maps = object_maps.mean(dim=1)
                self_mass = (object_group_maps * masks).sum(dim=-1)
                out_mass = (object_group_maps * (1.0 - masks).clamp(0.0, 1.0)).sum(dim=-1)
                denom = object_valid.float().sum().clamp_min(1.0)
                inside_masses.append((self_mass * object_valid.float()).sum() / denom)
                outside_masses.append((out_mass * object_valid.float()).sum() / denom)

            if include_void and bg_patch.any() and probs.shape[-1] == K + 1:
                bg_ce = -probs[..., -1].clamp_min(eps).log()
                if bg_w > 0.0:
                    losses.append(bg_w * bg_ce[bg_patch].mean())
                bg_accs.append((probs.argmax(dim=-1)[bg_patch] == K).float().mean())
                void_bg_masses.append(probs[..., -1][bg_patch].mean())

        if not losses:
            empty["valid_fraction"] = valid_fraction.detach()
            return empty

        return {
            "loss": torch.stack(losses).mean(),
            "fg_acc": torch.stack(fg_accs).mean().detach() if fg_accs else z,
            "bg_acc": torch.stack(bg_accs).mean().detach() if bg_accs else z,
            "entropy": torch.stack(entropies).mean().detach() if entropies else z,
            "valid_fraction": valid_fraction.detach(),
            "object_inside_mass": torch.stack(inside_masses).mean().detach() if inside_masses else z,
            "object_outside_mass": torch.stack(outside_masses).mean().detach() if outside_masses else z,
            "void_bg_mass": torch.stack(void_bg_masses).mean().detach() if void_bg_masses else z,
        }

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
            # Eval/viz must never hand NaN/Inf to PIL or W&B. The training loss
            # remains unsanitized and therefore still exposes numerical faults.
            object_map = torch.nan_to_num(object_map, nan=0.0, posinf=0.0, neginf=0.0)
            object_map = object_map / object_map.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            if void_map.numel() > 0:
                void_map = torch.nan_to_num(void_map, nan=0.0, posinf=0.0, neginf=0.0)
                void_map = void_map / void_map.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            object_acc = object_map if object_acc is None else object_acc + object_map
            void_acc = void_map if void_acc is None else void_acc + void_map
            count += 1
        if count <= 0:
            return None, None
        return (
            (object_acc / float(count)).detach(),
            (void_acc / float(count)).detach() if void_acc is not None else None,
        )

    def _compute_llm_register_patch_attention_maps(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        layers_spec: str,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Return layer/head-mean register -> image maps for E1 background."""
        reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
        R = max(reg_e - reg_s, 0)
        if hidden_states is None or R <= 0:
            return None
        B = hidden_states[0].shape[0]
        query_positions = torch.arange(
            reg_s, reg_e, device=hidden_states[0].device, dtype=torch.long
        ).view(1, R).expand(B, -1)
        query_valid = torch.ones(B, R, device=query_positions.device, dtype=torch.bool)
        acc = None
        count = 0
        for layer_idx in self._resolve_llm_qk_outside_layers(layers_spec):
            if layer_idx >= len(hidden_states) - 1:
                continue
            _, _, register_patch, _ = (
                self._compute_exact_llm_attention_components_for_layer(
                    layer_input=hidden_states[layer_idx],
                    layer_idx=layer_idx,
                    attention_bias=attention_bias,
                    positions=positions,
                    ovt_abs_positions=query_positions,
                    ovt_valid_mask=query_valid,
                    patch_temperature=temperature,
                )
            )
            register_map = register_patch.mean(dim=1)
            register_map = torch.nan_to_num(
                register_map, nan=0.0, posinf=0.0, neginf=0.0
            )
            denom = register_map.sum(dim=-1, keepdim=True)
            uniform = torch.full_like(register_map, 1.0 / max(register_map.shape[-1], 1))
            register_map = torch.where(
                denom > 1e-8,
                register_map / denom.clamp_min(1e-8),
                uniform,
            )
            acc = register_map if acc is None else acc + register_map
            count += 1
        return (acc / float(count)).detach() if count > 0 and acc is not None else None

    def _compute_e3_competition_owner_maps(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_bias: torch.Tensor,
        positions: Dict[str, int],
        ovt_abs_positions: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        layers_spec: str = "all",
        temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return E3's exact per-patch owner probabilities for evaluation.

        This mirrors the probability path used by
        ``_compute_e3_joint_attention_losses``:
          1. reconstruct exact post-LN/post-RoPE image-patch attention;
          2. average heads (and OVTs belonging to the same object);
          3. average registers into one residual-background class;
          4. softmax across ``K objects + background`` for every patch.

        The returned maps average those per-layer owner probabilities.  Unlike
        the independent patch-axis attention maps, classes sum to one at every
        image patch and therefore directly visualize the Competition CE
        decision used during E3 training.
        """
        if hidden_states is None:
            return None, None
        reg_s, reg_e = int(positions["reg_s"]), int(positions["reg_e"])
        R = max(reg_e - reg_s, 0)
        n = max(int(self.pgot_n_ovt_per_object), 1)
        B, M_total = ovt_abs_positions.shape
        K = M_total // n
        M = K * n
        layers = self._resolve_llm_qk_outside_layers(layers_spec)
        if K <= 0 or R <= 0 or not layers:
            return None, None

        object_positions = ovt_abs_positions[:, :M]
        object_valid_tokens = ovt_valid_mask[:, :M].to(dtype=torch.bool)
        object_valid = object_valid_tokens.reshape(B, K, n).any(dim=2)
        register_positions = torch.arange(
            reg_s,
            reg_e,
            device=object_positions.device,
            dtype=torch.long,
        ).view(1, R).expand(B, -1)
        register_valid = torch.ones(
            B,
            R,
            device=object_positions.device,
            dtype=torch.bool,
        )
        joint_positions = torch.cat([object_positions, register_positions], dim=1)
        joint_valid = torch.cat([object_valid_tokens, register_valid], dim=1)

        object_acc = None
        background_acc = None
        count = 0
        for layer_idx in layers:
            if layer_idx >= len(hidden_states) - 1:
                continue
            _, _, joint_patch, _ = self._compute_exact_llm_attention_components_for_layer(
                layer_input=hidden_states[layer_idx],
                layer_idx=layer_idx,
                attention_bias=attention_bias,
                positions=positions,
                ovt_abs_positions=joint_positions,
                ovt_valid_mask=joint_valid,
                patch_temperature=temperature,
            )
            H, P = joint_patch.shape[1], joint_patch.shape[-1]
            object_patch = joint_patch[:, :, :M]
            register_patch = joint_patch[:, :, M:M + R]
            object_maps = object_patch.reshape(B, H, K, n, P).mean(dim=3).mean(dim=1)
            background_map = register_patch.mean(dim=(1, 2))

            object_scores = object_maps.clamp_min(eps).log()
            object_scores = object_scores.masked_fill(
                ~object_valid.unsqueeze(-1),
                -1e4,
            )
            owner_scores = torch.cat(
                [object_scores, background_map.clamp_min(eps).log().unsqueeze(1)],
                dim=1,
            )
            owner_probs = F.softmax(owner_scores, dim=1, dtype=torch.float32)
            object_probs = owner_probs[:, :K]
            background_probs = owner_probs[:, K:K + 1]
            object_acc = object_probs if object_acc is None else object_acc + object_probs
            background_acc = (
                background_probs
                if background_acc is None
                else background_acc + background_probs
            )
            count += 1

        if count <= 0 or object_acc is None or background_acc is None:
            return None, None
        return (
            (object_acc / float(count)).detach(),
            (background_acc / float(count)).detach(),
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
        if bool(getattr(self.config, "pgot_v14_enable", False)):
            feats = self._pgot_v14_forward_features(
                images=images,
                target_images=target_images,
                caption_input_ids=caption_input_ids,
                caption_attention_mask=caption_attention_mask,
                ovt_positions_in_caption=ovt_positions_in_caption,
                ovt_valid_mask=ovt_valid_mask,
                output_hidden_states=False,
            )
            result = {"rae_hidden": feats["condition_hidden"], "gt_siglip": feats["gt_siglip"]}
            if self._pgot_dit_ovt_cross_attn_enabled():
                slot_context, slot_mask = self._pgot_prepare_dit_ovt_context(
                    feats["ovt_states"],
                    feats["ovt_valid"],
                )
                result["slot_context"] = slot_context
                result["slot_mask"] = slot_mask
            return result

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
        caption_embeds = self._pgot_apply_caption_conditioned_ovt_init(
            caption_embeds, caption_input_ids, ovt_positions_in_caption, ovt_valid_mask
        )
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
            rae_isolated=bool(getattr(self.config, "pgot_e4_rae_isolated", False)),
            rae_attends_caption=bool(getattr(self.config, "pgot_rae_attends_caption", False)),
            ovt_absolute_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            register_attends_caption=bool(
                getattr(self.config, "pgot_register_attends_caption", True)
            ),
            ovt_isolated=bool(
                getattr(self.config, "pgot_ovt_isolated_attention", False)
            ),
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

    def pgot_predict_direct_latent(self, condition_hidden: torch.Tensor) -> Optional[torch.Tensor]:
        if self.pgot_latent_head is None:
            return None
        cond = self._captionslot_prepare_diffusion_condition(condition_hidden)
        head_param = next(self.pgot_latent_head.parameters())
        pred = self.pgot_latent_head(cond.to(device=head_param.device, dtype=head_param.dtype))
        return pred

    def _pgot_compute_latent_distill_loss(
        self,
        *,
        condition_hidden: torch.Tensor,
        target_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred = self.pgot_predict_direct_latent(condition_hidden)
        if pred is None:
            zero = condition_hidden.new_zeros(())
            return {
                "loss": zero,
                "mse": zero,
                "cos": zero,
                "l1": zero,
                "pred": None,
                "pred_norm": zero,
                "target_norm": zero,
            }
        pred_f = pred.float()
        target_f = target_features.to(device=pred.device).float()
        mse = F.mse_loss(pred_f, target_f)
        l1 = F.l1_loss(pred_f, target_f)
        cos = 1.0 - F.cosine_similarity(pred_f, target_f, dim=-1).mean()
        mse_w = float(getattr(self.config, "pgot_latent_distill_mse_weight", 1.0))
        cos_w = float(getattr(self.config, "pgot_latent_distill_cos_weight", 1.0))
        l1_w = float(getattr(self.config, "pgot_latent_distill_l1_weight", 0.0))
        loss = mse_w * mse + cos_w * cos + l1_w * l1
        return {
            "loss": loss,
            "mse": mse,
            "cos": cos,
            "l1": l1,
            "pred": pred,
            "pred_norm": pred_f.norm(dim=-1).mean(),
            "target_norm": target_f.norm(dim=-1).mean(),
        }

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
        pred_latent = self.diff_head.infer(
            z=cond,
            slot_context=feats.get("slot_context"),
            slot_mask=feats.get("slot_mask"),
        )
        direct_latent = self.pgot_predict_direct_latent(rae_hidden)
        result = {"pred_latent": pred_latent, "gt_siglip": feats["gt_siglip"]}
        if direct_latent is not None:
            result["direct_latent"] = direct_latent.detach()
        return result

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


    def _forward_pgot_v14(
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
        v17_enabled = bool(getattr(self.config, "pgot_v17_enable", False))
        v17_w = float(getattr(self.config, "pgot_v17_ownership_weight", 0.0))
        v21_enabled = bool(getattr(self.config, "pgot_v21_enable", False))
        mask_llm_patch_outside_w = float(
            getattr(self.config, "pgot_mask_llm_patch_outside_weight", 0.0)
        )
        mask_llm_image_use_w = float(
            getattr(self.config, "pgot_mask_llm_image_use_weight", 0.0)
        )
        v22_attention_comp_w = float(
            getattr(self.config, "pgot_v22_attention_competition_weight", 0.0)
        )
        needs_exact_attention = (
            (v17_enabled and v17_w > 0.0)
            or mask_llm_patch_outside_w > 0.0
            or mask_llm_image_use_w > 0.0
            or v22_attention_comp_w > 0.0
        )

        seq = self._pgot_v14_forward_features(
            images=images,
            target_images=target_images,
            caption_input_ids=caption_input_ids,
            caption_attention_mask=caption_attention_mask,
            ovt_positions_in_caption=ovt_positions_in_caption,
            ovt_valid_mask=ovt_valid_mask,
            output_hidden_states=needs_exact_attention,
        )
        hidden = seq["hidden"]
        condition_hidden = seq["condition_hidden"]
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
        route_w = float(getattr(self.config, "pgot_v14_route_weight", 1.0))
        v21_ground_w = float(
            getattr(
                self.config,
                "pgot_v21_ground_weight_effective",
                getattr(self.config, "pgot_v21_ground_weight", 0.0),
            )
        )
        mask_bce_w = float(getattr(self.config, "pgot_mask_bce_weight", 0.0))
        zero = hidden.new_zeros(())

        route_stats = None
        loss_route = zero
        if ((route_w > 0.0 and not v21_enabled) or (v21_ground_w > 0.0 and v21_enabled)) and not v17_enabled:
            route_stats = self._pgot_v14_compute_route_loss(
                owner_logits=seq["owner_logits"],
                owner_probs=seq["owner_probs"],
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
            )
            loss_route = route_stats["loss"]

        loss_mask_bce = zero
        if mask_bce_w > 0.0:
            attn_temp = float(getattr(self.config, "pgot_attention_temperature", 1.0))
            attn_ln = bool(getattr(self.config, "pgot_attention_use_layer_norm", True))
            ovt_hidden = gather_ovt_hidden_states(
                hidden,
                seq["ovt_abs_positions"],
                seq["ovt_valid_mask"],
            )
            ovt_logits = compute_per_ovt_mask_logits(
                ovt_hidden=ovt_hidden,
                img_hidden=seq["img_hidden"],
                temperature=attn_temp,
                normalize_tokens=attn_ln,
            )
            loss_mask_bce = compute_mask_bce_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
            )

        ownership_stats = None
        loss_v17_ownership = zero
        if v17_enabled and v17_w > 0.0:
            ownership_stats = self._compute_v17_ownership_loss(
                hidden_states=seq["hidden_states"],
                attention_bias=seq["attn_bias"],
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=str(getattr(self.config, "pgot_v17_ownership_layers", "last4")),
            )
            loss_v17_ownership = ownership_stats["loss"]

        llm_patch_stats = None
        loss_mask_llm_patch_outside = zero
        loss_mask_llm_image_use = zero
        if mask_llm_patch_outside_w > 0.0 or mask_llm_image_use_w > 0.0:
            llm_patch_layers = str(
                getattr(self.config, "pgot_mask_llm_patch_outside_layers", "last4")
            )
            llm_patch_temperature = float(
                getattr(self.config, "pgot_mask_llm_patch_outside_temperature", 1.0)
            )
            llm_patch_void_weight = float(
                getattr(self.config, "pgot_mask_llm_patch_void_weight", 1.0)
            )
            llm_image_use_margin = float(
                getattr(self.config, "pgot_mask_llm_image_use_margin", 0.05)
            )
            llm_patch_stats = self._compute_llm_patch_outside_and_image_use_loss(
                hidden_states=seq["hidden_states"],
                attention_bias=seq["attn_bias"],
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=llm_patch_layers,
                temperature=llm_patch_temperature,
                void_weight=llm_patch_void_weight,
                image_use_margin=llm_image_use_margin,
            )
            loss_mask_llm_patch_outside = llm_patch_stats["outside_loss"]
            loss_mask_llm_image_use = llm_patch_stats["image_use_loss"]

        v22_attention_comp_stats = None
        loss_v22_attention_competition = zero
        if v22_attention_comp_w > 0.0:
            v22_attention_comp_stats = self._compute_v22_attention_competition_loss(
                hidden_states=seq["hidden_states"],
                attention_bias=seq["attn_bias"],
                positions=seq["positions"],
                ovt_abs_positions=seq["ovt_abs_positions"],
                ovt_valid_mask=seq["ovt_valid_mask"],
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=str(
                    getattr(self.config, "pgot_v22_attention_competition_layers", "26,27")
                ),
                temperature=float(
                    getattr(self.config, "pgot_v22_attention_competition_temperature", 1.0)
                ),
                include_void=bool(
                    getattr(self.config, "pgot_v22_attention_competition_include_void", False)
                ),
                bg_weight=float(
                    getattr(self.config, "pgot_v22_attention_competition_bg_weight", 0.25)
                ),
            )
            loss_v22_attention_competition = v22_attention_comp_stats["loss"]

        cfg_drop_rate = float(getattr(self.config, "pgot_cfg_drop_rate", 0.0))
        condition_for_diff = condition_hidden
        slot_context_for_diff = None
        slot_mask_for_diff = None
        if self._pgot_dit_ovt_cross_attn_enabled():
            slot_context_for_diff, slot_mask_for_diff = self._pgot_prepare_dit_ovt_context(
                seq["ovt_states"],
                seq["ovt_valid"],
            )
        if self.training and cfg_drop_rate > 0.0:
            drop_mask = (torch.rand(B, device=condition_hidden.device) < cfg_drop_rate).view(B, 1, 1)
            condition_for_diff = condition_hidden * (~drop_mask).to(condition_hidden.dtype)
            if slot_context_for_diff is not None:
                slot_keep = (~drop_mask).to(slot_context_for_diff.dtype)
                slot_context_for_diff = slot_context_for_diff * slot_keep
                slot_mask_for_diff = slot_mask_for_diff & (~drop_mask.squeeze(-1).to(slot_mask_for_diff.device))
        loss_recon = self._captionslot_compute_diffusion_loss(
            hidden=condition_for_diff,
            target_features=gt_siglip,
            slot_context=slot_context_for_diff,
            slot_mask=slot_mask_for_diff,
        )

        latent_distill_w = float(getattr(self.config, "pgot_latent_distill_weight", 0.0))
        latent_stats = None
        loss_latent_distill = zero
        if latent_distill_w > 0.0 and self.pgot_latent_head is not None:
            latent_stats = self._pgot_compute_latent_distill_loss(
                condition_hidden=condition_hidden,
                target_features=gt_siglip,
            )
            loss_latent_distill = latent_stats["loss"]

        lm_w = float(getattr(self.config, "pgot_lm_loss_weight", 1.0))
        recon_w = float(getattr(self.config, "pgot_recon_loss_weight", 1.0))
        route_loss_weight = v21_ground_w if v21_enabled else route_w
        loss_mask = (
            route_loss_weight * loss_route
            + mask_bce_w * loss_mask_bce
            + v17_w * loss_v17_ownership
            + mask_llm_patch_outside_w * loss_mask_llm_patch_outside
            + mask_llm_image_use_w * loss_mask_llm_image_use
            + v22_attention_comp_w * loss_v22_attention_competition
        )
        total_loss = (
            lm_w * loss_lm
            + recon_w * loss_recon
            + loss_mask
            + latent_distill_w * loss_latent_distill
        )

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
        details = {}
        if (not v21_enabled) and route_w > 0.0 and route_stats is not None:
            details.update(
                {
                    "loss_v14_route": loss_route.detach(),
                    "v14_route_object_loss": route_stats["object_loss"],
                    "v14_route_void_loss": route_stats["void_loss"],
                    "v14_route_fg_acc": route_stats["fg_acc"],
                    "v14_route_bg_acc": route_stats["bg_acc"],
                    "v14_route_void_prob_on_fg": route_stats["void_prob_on_fg"],
                    "v14_route_object_prob_on_bg": route_stats["object_prob_on_bg"],
                    "v14_route_entropy": route_stats["entropy"],
                    "v14_route_object_count": route_stats["object_count"],
                }
            )
        if v21_enabled and v21_ground_w > 0.0 and route_stats is not None:
            details.update(
                {
                    "loss_v21_ground": loss_route.detach(),
                    "v21_ground_weight": hidden.new_tensor(v21_ground_w).detach(),
                    "v21_ground_object_loss": route_stats["object_loss"],
                    "v21_ground_void_loss": route_stats["void_loss"],
                    "v21_ground_fg_acc": route_stats["fg_acc"],
                    "v21_ground_bg_acc": route_stats["bg_acc"],
                    "v21_ground_bg_prob_on_fg": route_stats["void_prob_on_fg"],
                    "v21_ground_object_prob_on_bg": route_stats["object_prob_on_bg"],
                    "v21_ground_entropy": route_stats["entropy"],
                    "v21_ground_object_count": route_stats["object_count"],
                }
            )
        if mask_bce_w > 0.0:
            details["loss_mask_bce"] = loss_mask_bce.detach()
        if v17_enabled and v17_w > 0.0 and ownership_stats is not None:
            details.update(
                {
                    "loss_v17_ownership": loss_v17_ownership.detach(),
                    "v17_ownership_mass": ownership_stats["mass"],
                    "v17_ownership_ovt_mass": ownership_stats["ovt_mass"],
                    "v17_ownership_void_mass": ownership_stats["void_mass"],
                    "v17_ownership_entropy": ownership_stats["entropy"],
                    "v17_ownership_acc": ownership_stats["acc"],
                    "v17_ownership_valid_fraction": ownership_stats["valid_fraction"],
                }
            )
        if mask_llm_patch_outside_w > 0.0 or mask_llm_image_use_w > 0.0:
            details.update(
                {
                    "loss_mask_llm_patch_outside": loss_mask_llm_patch_outside.detach(),
                    "loss_mask_llm_image_use": loss_mask_llm_image_use.detach(),
                    "llm_patch_object_outside_loss": llm_patch_stats["object_outside_loss"],
                    "llm_patch_void_outside_loss": llm_patch_stats["void_outside_loss"],
                    "llm_patch_object_image_use_loss": llm_patch_stats["object_image_use_loss"],
                    "llm_patch_void_image_use_loss": llm_patch_stats["void_image_use_loss"],
                    "llm_patch_object_self_mass": llm_patch_stats["object_self_mass"],
                    "llm_patch_object_outside_mass": llm_patch_stats["object_outside_mass"],
                    "llm_patch_object_full_image_mass": llm_patch_stats["object_full_image_mass"],
                    "llm_patch_void_self_mass": llm_patch_stats["void_self_mass"],
                    "llm_patch_void_outside_mass": llm_patch_stats["void_outside_mass"],
                    "llm_patch_void_full_image_mass": llm_patch_stats["void_full_image_mass"],
                    "llm_patch_void_valid_fraction": llm_patch_stats["void_valid_fraction"],
                }
            )
        if v22_attention_comp_w > 0.0 and v22_attention_comp_stats is not None:
            details.update(
                {
                    "loss_v22_attention_competition": loss_v22_attention_competition.detach(),
                    "v22_attention_competition_fg_acc": v22_attention_comp_stats["fg_acc"],
                    "v22_attention_competition_bg_acc": v22_attention_comp_stats["bg_acc"],
                    "v22_attention_competition_entropy": v22_attention_comp_stats["entropy"],
                    "v22_attention_competition_valid_fraction": v22_attention_comp_stats["valid_fraction"],
                    "v22_attention_object_inside_mass": v22_attention_comp_stats["object_inside_mass"],
                    "v22_attention_object_outside_mass": v22_attention_comp_stats["object_outside_mass"],
                    "v22_attention_void_bg_mass": v22_attention_comp_stats["void_bg_mass"],
                }
            )
        if latent_distill_w > 0.0 and latent_stats is not None:
            details.update(
                {
                    "loss_latent_distill": loss_latent_distill.detach(),
                    "latent_distill_mse": latent_stats["mse"].detach(),
                    "latent_distill_cos": latent_stats["cos"].detach(),
                    "latent_distill_l1": latent_stats["l1"].detach(),
                    "latent_pred_norm": latent_stats["pred_norm"].detach(),
                    "latent_target_norm": latent_stats["target_norm"].detach(),
                }
            )
        if self._pgot_dit_ovt_cross_attn_enabled():
            details["dit_ovt_xattn_context_tokens"] = seq["ovt_valid"].float().sum(dim=1).mean().detach()
        self.pgot_loss_details = details
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
        gt_rae_masks_per_ovt: Optional[torch.Tensor],
        target_images: torch.Tensor,
        pgot_contrastive_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if bool(getattr(self.config, "pgot_register_hard_gt_mask", False)) and (
            bool(getattr(self.config, "pgot_v14_enable", False))
            or bool(getattr(self.config, "pgot_v12_enable", False))
        ):
            raise ValueError(
                "pgot_register_hard_gt_mask currently supports the core PGOT path only "
                "(pgot_v12_enable=False, pgot_v14_enable=False)."
            )
        if bool(getattr(self.config, "pgot_v14_enable", False)):
            return self._forward_pgot_v14(
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
        if gt_rae_masks_per_ovt is not None:
            gt_rae_masks_per_ovt = gt_rae_masks_per_ovt.to(model_device).float()
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
        caption_embeds = self._pgot_apply_caption_conditioned_ovt_init(
            caption_embeds, caption_input_ids, ovt_positions_in_caption, ovt_valid_mask
        )
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
            rae_isolated=bool(getattr(self.config, "pgot_e4_rae_isolated", False)),
            rae_attends_caption=bool(getattr(self.config, "pgot_rae_attends_caption", False)),
            ovt_absolute_positions=ovt_abs_positions,
            ovt_valid_mask=ovt_valid_mask,
            register_attends_caption=bool(
                getattr(self.config, "pgot_register_attends_caption", True)
            ),
            ovt_isolated=bool(
                getattr(self.config, "pgot_ovt_isolated_attention", False)
            ),
        )

        # E2: the train-time GT union is an oracle routing constraint for
        # residual registers.  It modifies only register-query/image-key edges
        # in the shared bias, hence all heads in all layers receive the same
        # structural prohibition.  Standard validation/inference remains
        # GT-free unless the explicitly oracle-only eval flag is requested.
        register_hard_configured = bool(
            getattr(self.config, "pgot_register_hard_gt_mask", False)
        )
        register_hard_active = register_hard_configured and (
            self.training
            or bool(getattr(self.config, "pgot_register_hard_gt_mask_eval", False))
        )
        register_hard_blocked_fraction = gt_masks_per_ovt.new_zeros(())
        register_hard_blocked_count = gt_masks_per_ovt.new_zeros(())
        if register_hard_active:
            hard_stats = apply_register_hard_gt_mask(
                attn_bias,
                positions=positions,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                threshold=float(
                    getattr(self.config, "pgot_register_hard_gt_mask_threshold", 0.0)
                ),
            )
            attn_bias = hard_stats["attention_bias"]
            register_hard_blocked_fraction = hard_stats["blocked_patch_fraction"]
            register_hard_blocked_count = hard_stats["blocked_patch_count"]

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
        core_outside_w = float(
            getattr(self.config, "pgot_core_outside_weight", 0.0)
        )
        core_tail_w = float(getattr(self.config, "pgot_core_tail_weight", 0.0))
        core_register_w = float(
            getattr(self.config, "pgot_core_register_outside_weight", 0.0)
        )
        e3_competition_w = float(
            getattr(self.config, "pgot_e3_attention_competition_weight", 0.0)
        )
        e4_full_inside_w = float(
            getattr(
                self.config,
                "pgot_e4_full_inside_weight_effective",
                getattr(self.config, "pgot_e4_full_inside_weight", 0.0),
            )
        )
        e4_rae_bind_w = float(
            getattr(
                self.config,
                "pgot_e4_rae_bind_weight_effective",
                getattr(self.config, "pgot_e4_rae_bind_weight", 0.0),
            )
        )
        if e4_full_inside_w > 0.0 and e3_competition_w <= 0.0:
            raise ValueError(
                "E4 full-inside floor reuses E3's exact all-layer attention "
                "pass and requires pgot_e3_attention_competition_weight > 0."
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
                or core_outside_w > 0.0
                or core_tail_w > 0.0
                or core_register_w > 0.0
                or e3_competition_w > 0.0
                or e4_full_inside_w > 0.0
                or e4_rae_bind_w > 0.0
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
        mask_sigmoid_outside_w = float(getattr(self.config, "pgot_mask_sigmoid_outside_weight", 0.0))
        register_fg_suppression_w = float(
            getattr(self.config, "pgot_register_foreground_suppression_weight", 0.0)
        )
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

        loss_mask_sigmoid_outside = zero
        sigmoid_outside_inside_prob = zero
        sigmoid_outside_outside_prob = zero
        if mask_sigmoid_outside_w > 0.0:
            sigmoid_outside = compute_sigmoid_outside_bce_loss(
                ovt_logits=ovt_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
            )
            loss_mask_sigmoid_outside = sigmoid_outside["loss"]
            sigmoid_outside_inside_prob = sigmoid_outside["inside_prob"]
            sigmoid_outside_outside_prob = sigmoid_outside["outside_prob"]

        loss_register_foreground_suppression = zero
        register_fg_prob = zero
        register_bg_prob = zero
        register_fg_fraction = zero
        if register_fg_suppression_w > 0.0:
            register_suppression = compute_register_foreground_suppression_loss(
                reg_logits=reg_logits,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
            )
            loss_register_foreground_suppression = register_suppression["loss"]
            register_fg_prob = register_suppression["fg_prob"]
            register_bg_prob = register_suppression["bg_prob"]
            register_fg_fraction = register_suppression["fg_fraction"]

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

        loss_core_outside = zero
        loss_core_tail = zero
        core_inside_mass = zero
        core_outside_mass = zero
        core_full_image_mass = zero
        core_worst_head_outside_mass = zero
        core_void_loss = zero
        core_void_inside_mass = zero
        core_void_outside_mass = zero
        core_void_full_image_mass = zero
        core_void_worst_head_outside_mass = zero
        core_void_valid_fraction = zero
        core_valid_ovt_count = zero
        core_num_layers = zero
        core_layer_metrics = {}
        if (core_outside_w > 0.0 or core_tail_w > 0.0) and e3_competition_w <= 0.0:
            core_losses = self._compute_core_all_layer_outside_loss(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=str(getattr(self.config, "pgot_core_outside_layers", "all")),
                temperature=float(
                    getattr(self.config, "pgot_core_outside_temperature", 1.0)
                ),
                void_weight=float(
                    getattr(self.config, "pgot_core_void_weight", 1.0)
                ),
                tail_fraction=float(
                    getattr(self.config, "pgot_core_tail_fraction", 0.1)
                ),
            )
            loss_core_outside = core_losses["loss"]
            loss_core_tail = core_losses["tail_loss"]
            core_inside_mass = core_losses["inside_mass"]
            core_outside_mass = core_losses["outside_mass"]
            core_full_image_mass = core_losses["full_image_mass"]
            core_worst_head_outside_mass = core_losses[
                "worst_head_outside_mass"
            ]
            core_void_loss = core_losses["void_loss"]
            core_void_inside_mass = core_losses["void_inside_mass"]
            core_void_outside_mass = core_losses["void_outside_mass"]
            core_void_full_image_mass = core_losses["void_full_image_mass"]
            core_void_worst_head_outside_mass = core_losses[
                "void_worst_head_outside_mass"
            ]
            core_void_valid_fraction = core_losses["void_valid_fraction"]
            core_valid_ovt_count = core_losses["valid_ovt_count"]
            core_num_layers = core_losses["num_layers"]
            core_layer_metrics = core_losses["layer_metrics"]

        loss_core_register = zero
        core_register_fg_mass = zero
        core_register_bg_mass = zero
        core_register_full_image_mass = zero
        core_register_worst_head_fg_mass = zero
        core_register_num_layers = zero
        core_register_layer_metrics = {}
        loss_e3_competition = zero
        e3_competition_fg_loss = zero
        e3_competition_bg_loss = zero
        e3_competition_fg_acc = zero
        e3_competition_bg_acc = zero
        e3_competition_entropy = zero
        e3_register_prob_on_fg = zero
        e3_object_prob_on_bg = zero
        e3_competition_fg_fraction = zero
        e3_layer_metrics = {}
        loss_e4_full_inside = zero
        e4_full_inside_mass = zero
        e4_full_outside_mass = zero
        e4_full_inside_satisfied_fraction = zero

        # E3 computes OVT outside, register outside, and ownership competition
        # jointly, avoiding four separate all-layer Q/K reconstructions.
        if e3_competition_w > 0.0:
            if register_hidden.shape[1] <= 0:
                raise ValueError(
                    "pgot_e3_attention_competition_weight > 0 requires pgot_n_register > 0."
                )
            e3_stats = self._compute_e3_joint_attention_losses(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                gt_masks_per_ovt=gt_masks_per_ovt,
                layers_spec=str(
                    getattr(self.config, "pgot_e3_attention_competition_layers", "all")
                ),
                temperature=float(
                    getattr(self.config, "pgot_e3_attention_competition_temperature", 1.0)
                ),
                bg_weight=float(
                    getattr(self.config, "pgot_e3_attention_competition_bg_weight", 0.25)
                ),
                full_inside_target=float(
                    getattr(self.config, "pgot_e4_full_inside_target", 0.30)
                )
                if e4_full_inside_w > 0.0
                else 0.0,
            )
            loss_core_outside = e3_stats["object_loss"]
            core_inside_mass = e3_stats["object_inside_mass"]
            core_outside_mass = e3_stats["object_outside_mass"]
            core_full_image_mass = e3_stats["object_full_image_mass"]
            core_worst_head_outside_mass = e3_stats["object_worst_head_outside_mass"]
            core_valid_ovt_count = e3_stats["valid_ovt_count"]
            core_num_layers = e3_stats["num_layers"]
            loss_core_register = e3_stats["register_loss"]
            core_register_fg_mass = e3_stats["register_foreground_mass"]
            core_register_bg_mass = e3_stats["register_background_mass"]
            core_register_full_image_mass = e3_stats["register_full_image_mass"]
            core_register_worst_head_fg_mass = e3_stats[
                "register_worst_head_foreground_mass"
            ]
            core_register_num_layers = e3_stats["num_layers"]
            loss_e3_competition = e3_stats["competition_loss"]
            e3_competition_fg_loss = e3_stats["competition_fg_loss"]
            e3_competition_bg_loss = e3_stats["competition_bg_loss"]
            e3_competition_fg_acc = e3_stats["competition_fg_acc"]
            e3_competition_bg_acc = e3_stats["competition_bg_acc"]
            e3_competition_entropy = e3_stats["competition_entropy"]
            e3_register_prob_on_fg = e3_stats["competition_register_prob_on_fg"]
            e3_object_prob_on_bg = e3_stats["competition_object_prob_on_bg"]
            e3_competition_fg_fraction = e3_stats["competition_fg_fraction"]
            e3_layer_metrics = e3_stats["layer_metrics"]
            loss_e4_full_inside = e3_stats["object_full_inside_floor_loss"]
            e4_full_inside_mass = e3_stats["object_full_inside_mass"]
            e4_full_outside_mass = e3_stats["object_full_outside_mass"]
            e4_full_inside_satisfied_fraction = e3_stats[
                "object_full_inside_satisfied_fraction"
            ]
        elif core_register_w > 0.0:
            if register_hidden.shape[1] <= 0:
                raise ValueError(
                    "pgot_core_register_outside_weight > 0 requires pgot_n_register > 0."
                )
            register_stats = self._compute_core_register_foreground_loss(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                gt_masks_per_ovt=gt_masks_per_ovt,
                ovt_valid_mask=ovt_valid_mask,
                layers_spec=str(getattr(self.config, "pgot_core_outside_layers", "all")),
                temperature=float(
                    getattr(self.config, "pgot_core_outside_temperature", 1.0)
                ),
            )
            loss_core_register = register_stats["loss"]
            core_register_fg_mass = register_stats["foreground_mass"]
            core_register_bg_mass = register_stats["background_mass"]
            core_register_full_image_mass = register_stats["full_image_mass"]
            core_register_worst_head_fg_mass = register_stats[
                "worst_head_foreground_mass"
            ]
            core_register_num_layers = register_stats["num_layers"]
            core_register_layer_metrics = register_stats["layer_metrics"]

        e4_rae_stats = None
        loss_e4_rae_bind = zero
        if e4_rae_bind_w > 0.0:
            if not bool(getattr(self.config, "pgot_e4_rae_isolated", False)):
                raise ValueError(
                    "E4 RAE binding requires pgot_e4_rae_isolated=True so "
                    "cross-query propagation cannot bypass the object bottleneck."
                )
            binding_masks = (
                gt_rae_masks_per_ovt
                if gt_rae_masks_per_ovt is not None
                else gt_masks_per_ovt
            )
            e4_rae_stats = self._compute_e4_rae_binding_loss(
                hidden_states=out.hidden_states,
                attention_bias=attn_bias,
                positions=positions,
                ovt_abs_positions=ovt_abs_positions,
                ovt_valid_mask=ovt_valid_mask,
                gt_masks_per_ovt=binding_masks,
                layers_spec=str(
                    getattr(self.config, "pgot_e4_rae_bind_layers", "last8")
                ),
            )
            loss_e4_rae_bind = e4_rae_stats["loss"]

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
            + mask_sigmoid_outside_w * loss_mask_sigmoid_outside
            + register_fg_suppression_w * loss_register_foreground_suppression
            + mask_object_balanced_bce_w * loss_mask_object_balanced_bce
            + mask_tversky_w * loss_mask_tversky
            + mask_spatial_outside_w * loss_mask_spatial_outside
            + mask_spatial_outside_log_w * loss_mask_spatial_outside_log
            + mask_llm_qk_outside_w * loss_mask_llm_qk_outside
            + mask_llm_attention_outside_w * loss_mask_llm_attention_outside
            + mask_llm_patch_outside_w * loss_mask_llm_patch_outside
            + mask_llm_image_use_w * loss_mask_llm_image_use
            + core_outside_w * loss_core_outside
            + core_tail_w * loss_core_tail
            + core_register_w * loss_core_register
            + e3_competition_w * loss_e3_competition
            + e4_full_inside_w * loss_e4_full_inside
            + e4_rae_bind_w * loss_e4_rae_bind
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
        if register_hard_configured:
            details["register_hard_mask_active"] = gt_masks_per_ovt.new_tensor(
                float(register_hard_active)
            )
            details["register_hard_mask_blocked_patch_fraction"] = (
                register_hard_blocked_fraction.detach()
            )
            details["register_hard_mask_blocked_patch_count"] = (
                register_hard_blocked_count.detach()
            )
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
        if mask_sigmoid_outside_w > 0.0:
            details["loss_mask_sigmoid_outside"] = loss_mask_sigmoid_outside.detach()
            details["sigmoid_outside_inside_prob"] = sigmoid_outside_inside_prob.detach()
            details["sigmoid_outside_outside_prob"] = sigmoid_outside_outside_prob.detach()
        if register_fg_suppression_w > 0.0:
            details["loss_register_foreground_suppression"] = (
                loss_register_foreground_suppression.detach()
            )
            details["register_fg_prob"] = register_fg_prob.detach()
            details["register_bg_prob"] = register_bg_prob.detach()
            details["register_fg_fraction"] = register_fg_fraction.detach()
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
        if core_outside_w > 0.0 or core_tail_w > 0.0:
            details["loss_core_outside"] = loss_core_outside.detach()
            details["loss_core_tail"] = loss_core_tail.detach()
            details["core_inside_mass"] = core_inside_mass.detach()
            details["core_outside_mass"] = core_outside_mass.detach()
            details["core_full_image_mass"] = core_full_image_mass.detach()
            details["core_worst_head_outside_mass"] = (
                core_worst_head_outside_mass.detach()
            )
            details["core_void_loss"] = core_void_loss.detach()
            details["core_void_inside_mass"] = core_void_inside_mass.detach()
            details["core_void_outside_mass"] = core_void_outside_mass.detach()
            details["core_void_full_image_mass"] = (
                core_void_full_image_mass.detach()
            )
            details["core_void_worst_head_outside_mass"] = (
                core_void_worst_head_outside_mass.detach()
            )
            details["core_void_valid_fraction"] = (
                core_void_valid_fraction.detach()
            )
            details["core_valid_ovt_count"] = core_valid_ovt_count.detach()
            details["core_num_layers"] = core_num_layers.detach()
            details.update(core_layer_metrics)
        if core_register_w > 0.0:
            details["loss_core_register"] = loss_core_register.detach()
            details["core_register_fg_mass"] = core_register_fg_mass.detach()
            details["core_register_bg_mass"] = core_register_bg_mass.detach()
            details["core_register_full_image_mass"] = (
                core_register_full_image_mass.detach()
            )
            details["core_register_worst_head_fg_mass"] = (
                core_register_worst_head_fg_mass.detach()
            )
            details["core_register_num_layers"] = core_register_num_layers.detach()
            details.update(core_register_layer_metrics)
        if e3_competition_w > 0.0:
            details["loss_e3_attention_competition"] = loss_e3_competition.detach()
            details["e3_competition_fg_loss"] = e3_competition_fg_loss.detach()
            details["e3_competition_bg_loss"] = e3_competition_bg_loss.detach()
            details["e3_competition_fg_acc"] = e3_competition_fg_acc.detach()
            details["e3_competition_bg_acc"] = e3_competition_bg_acc.detach()
            details["e3_competition_entropy"] = e3_competition_entropy.detach()
            details["e3_register_prob_on_fg"] = e3_register_prob_on_fg.detach()
            details["e3_object_prob_on_bg"] = e3_object_prob_on_bg.detach()
            details["e3_competition_fg_fraction"] = e3_competition_fg_fraction.detach()
            details.update(e3_layer_metrics)
        if e4_full_inside_w > 0.0:
            details["loss_e4_full_inside"] = loss_e4_full_inside.detach()
            details["e4_full_inside_weight_effective"] = zero.new_tensor(
                e4_full_inside_w
            )
            details["e4_full_inside_mass"] = e4_full_inside_mass.detach()
            details["e4_full_outside_mass"] = e4_full_outside_mass.detach()
            details["e4_full_inside_satisfied_fraction"] = (
                e4_full_inside_satisfied_fraction.detach()
            )
        if e4_rae_bind_w > 0.0 and e4_rae_stats is not None:
            details["loss_e4_rae_bind"] = loss_e4_rae_bind.detach()
            details["e4_rae_bind_weight_effective"] = zero.new_tensor(
                e4_rae_bind_w
            )
            details["e4_rae_correct_owner_mass"] = e4_rae_stats[
                "correct_owner_mass"
            ]
            details["e4_rae_owner_total_mass"] = e4_rae_stats["owner_total_mass"]
            details["e4_rae_object_mass_on_fg"] = e4_rae_stats[
                "object_mass_on_fg"
            ]
            details["e4_rae_register_mass_on_fg"] = e4_rae_stats[
                "register_mass_on_fg"
            ]
            details["e4_rae_register_mass_on_bg"] = e4_rae_stats[
                "register_mass_on_bg"
            ]
            details["e4_rae_self_mass"] = e4_rae_stats["self_mass"]
            details["e4_rae_other_query_mass"] = e4_rae_stats["other_rae_mass"]
            details["e4_rae_fg_acc"] = e4_rae_stats["fg_acc"]
            details["e4_rae_bg_acc"] = e4_rae_stats["bg_acc"]
            details["e4_rae_entropy"] = e4_rae_stats["entropy"]
            details["e4_rae_fg_fraction"] = e4_rae_stats["fg_fraction"]
            details["e4_rae_num_layers"] = e4_rae_stats["num_layers"]
            details.update(e4_rae_stats["layer_metrics"])
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
