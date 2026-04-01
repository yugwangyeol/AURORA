"""
AURORA Slot Head — Cross-attention based object grounding.

Takes an object prompt (MLLM hidden) as query and image features as key/value,
producing a grounded slot representation and an attention map indicating which
image patches the slot covers.

Accumulated suppression mask prevents re-attending to previously discovered regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotHead(nn.Module):
    """
    Cross-attention head that grounds an object prompt in image features.

    Input:
        query:     (B, 1, D) — MLLM hidden at the current AR step
        key_value: (B, N, D) — MLLM-processed image patch features (N=256)
        attn_bias: (B, 1, 1, N) — accumulated suppression mask (additive)

    Output:
        slot:     (B, 1, D) — grounded object representation (MLLM space)
        attn_map: (B, N)    — normalised attention over image patches
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_bias: torch.Tensor = None,
    ):
        B, _, D = query.shape
        N = key_value.shape[1]
        H, d = self.n_heads, self.head_dim

        proj_dtype = self.q_proj.weight.dtype
        query_proj = query.to(dtype=proj_dtype)
        key_value_proj = key_value.to(dtype=proj_dtype)

        Q = self.q_proj(query_proj).view(B, 1, H, d).transpose(1, 2)       # (B, H, 1, d)
        K = self.k_proj(key_value_proj).view(B, N, H, d).transpose(1, 2)   # (B, H, N, d)
        V = self.v_proj(key_value_proj).view(B, N, H, d).transpose(1, 2)   # (B, H, N, d)

        # Compute attention in fp32 for numerical stability under bf16/fp16 training.
        logits = (Q.float() @ K.float().transpose(-2, -1)) * self.scale  # (B, H, 1, N)

        if attn_bias is not None:
            logits = logits + attn_bias.float()

        logits = logits - logits.amax(dim=-1, keepdim=True)
        weights = F.softmax(logits, dim=-1).to(dtype=V.dtype)          # (B, H, 1, N)

        attended = (weights.to(dtype=V.dtype) @ V).transpose(1, 2).reshape(B, 1, D)     # (B, 1, D)
        projected = self.out_proj(attended)

        slot = self.norm(query_proj + projected)

        attn_map = weights.mean(dim=1).squeeze(1)                     # (B, N)
        return slot, attn_map


class SlotSupervisionProjector(nn.Module):
    """Projects slot from MLLM space to a target feature space (e.g. DINO/SigLIP)."""

    def __init__(self, d_model: int, d_target: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_target),
        )

    def forward(self, slot: torch.Tensor) -> torch.Tensor:
        if slot.dim() == 3:
            slot = slot.squeeze(1)
        return self.proj(slot)
