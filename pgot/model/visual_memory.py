"""Object-semantic / image-appearance memory blocks for PGOT E8.

The writer uses object OVT states and background-register states only as
queries.  Values are projected exclusively from the MLLM image-token stream.
The reader then uses the final semantic states as keys and the accumulated
image-only memories as values, so semantic text cannot become reconstruction
content through this path.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class PGOTE8VisualMemoryWriter(nn.Module):
    """Competitive patch ownership followed by a gated image-only write."""

    def __init__(self, dim: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)

        self.semantic_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.image_norm = nn.LayerNorm(self.dim)
        self.update_norm = nn.LayerNorm(self.dim)

        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.memory_to_query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.fuse = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.gate = nn.Linear(3 * self.dim, self.dim)
        self.inject = nn.Linear(self.dim, self.dim, bias=False)

        # Start close to the pretrained MLLM while keeping a non-zero gradient
        # path through the writer on the first optimization step.
        self.write_logit = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid=0.1
        self.inject_logit = nn.Parameter(torch.tensor(-2.9444390))  # sigmoid=0.05

        for layer in (
            self.query,
            self.memory_to_query,
            self.key,
            self.value,
            self.inject,
        ):
            nn.init.eye_(layer.weight)
        # A newly-created clean E8.1 writer starts as an identity write from
        # the current image update.  Legacy E8 checkpoints overwrite this
        # tensor when loaded, so their behaviour is preserved.
        nn.init.zeros_(self.fuse.weight)
        with torch.no_grad():
            self.fuse.weight[:, self.dim :].copy_(torch.eye(self.dim))
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        *,
        semantic_slots: torch.Tensor,
        visual_memory: torch.Tensor,
        image_states: torch.Tensor,
        slot_valid: torch.Tensor,
        clean_refinement: bool = False,
        initialize_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if semantic_slots.shape != visual_memory.shape:
            raise ValueError(
                "E8 semantic/memory shape mismatch: "
                f"{tuple(semantic_slots.shape)} vs {tuple(visual_memory.shape)}"
            )
        if semantic_slots.ndim != 3 or image_states.ndim != 3:
            raise ValueError("E8 writer expects [B,S,D] slots and [B,P,D] image states")
        if semantic_slots.shape[0] != image_states.shape[0]:
            raise ValueError("E8 writer batch mismatch")
        if semantic_slots.shape[-1] != self.dim or image_states.shape[-1] != self.dim:
            raise ValueError(f"E8 writer expects hidden dim {self.dim}")
        if slot_valid.shape != semantic_slots.shape[:2]:
            raise ValueError("E8 slot_valid must have shape [B,S]")

        stream_dtype = semantic_slots.dtype
        module_dtype = self.semantic_norm.weight.dtype
        semantic_work = semantic_slots.to(dtype=module_dtype)
        memory_work = visual_memory.to(dtype=module_dtype)
        image_work = image_states.to(dtype=module_dtype)

        semantic_n = self.semantic_norm(semantic_work)
        memory_n = self.memory_norm(memory_work)
        query = self.query(semantic_n) + self.memory_to_query(memory_n)
        image_n = self.image_norm(image_work)
        key = self.key(image_n)

        logits = torch.einsum("bsd,bpd->bsp", query.float(), key.float())
        logits = logits / math.sqrt(float(self.dim))
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits.masked_fill(~slot_valid.unsqueeze(-1), -1e4)

        # Every patch chooses exactly one valid object/register owner.
        owner_probs = F.softmax(logits, dim=1)
        owner_probs = owner_probs * slot_valid.unsqueeze(-1).float()
        owner_probs = owner_probs / owner_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Convert patch ownership into a normalized write for each slot.
        write_weights = owner_probs / owner_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        values = self.value(image_n)
        update = torch.einsum(
            "bsp,bpd->bsd", write_weights.to(values.dtype), values
        )

        update_n = self.update_norm(update)
        candidate = self.fuse(torch.cat([memory_n, update_n], dim=-1))
        gate = torch.sigmoid(
            self.gate(torch.cat([semantic_n, memory_n, update_n], dim=-1))
        )
        if clean_refinement:
            # E8.1 keeps visual memory as a separate state.  The first writer
            # establishes it from the current image-only update; later writers
            # refine/overwrite that state instead of additively accumulating
            # early ownership mistakes.
            if initialize_memory:
                new_memory = candidate
            else:
                new_memory = (1.0 - gate) * memory_work + gate * candidate
            write_strength = candidate.new_ones(())
        else:
            # Legacy E8 path retained for loading and evaluating the original
            # E8 checkpoint.
            write_strength = torch.sigmoid(self.write_logit).to(candidate.dtype)
            new_memory = memory_work + write_strength * gate * candidate
        new_memory = torch.where(
            slot_valid.unsqueeze(-1), new_memory, torch.zeros_like(new_memory)
        )
        new_memory = torch.where(torch.isfinite(new_memory), new_memory, memory_work)
        new_memory = new_memory.to(dtype=stream_dtype)

        return {
            "visual_memory": new_memory,
            "owner_logits": logits,
            "owner_probs": owner_probs,
            "write_weights": write_weights,
            "write_update": update,
            "write_gate_mean": gate.mean().detach(),
            "write_strength": write_strength.detach(),
        }

    def injection_delta(self, visual_memory: torch.Tensor) -> torch.Tensor:
        stream_dtype = visual_memory.dtype
        module_dtype = self.memory_norm.weight.dtype
        memory_work = visual_memory.to(dtype=module_dtype)
        strength = torch.sigmoid(self.inject_logit).to(module_dtype)
        return (
            strength * self.inject(self.memory_norm(memory_work))
        ).to(dtype=stream_dtype)


class PGOTE9UnifiedSlotWriter(nn.Module):
    """Slot-style visual update applied directly to in-MLLM OVT states.

    Object OVTs and background registers compete for every image patch.  The
    resulting per-slot visual update is recurrently fused into the *same* slot
    hidden state with an explicitly-FP32 GRU.  There is no separately carried
    visual-memory tensor in this writer.

    ``update_dim`` keeps the recurrent update affordable for a wide Qwen
    hidden state while preserving a full-width OVT in the language model.
    """

    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        update_dim: int = 512,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.update_dim = int(update_dim)
        if self.update_dim <= 0:
            raise ValueError("E9 update_dim must be positive")
        self.temperature = float(temperature)

        U = self.update_dim
        hidden = max(int(round(U * float(mlp_ratio))), U)
        self.slot_norm = nn.LayerNorm(self.dim)
        self.image_norm = nn.LayerNorm(self.dim)
        self.query = nn.Linear(self.dim, U, bias=False)
        self.key = nn.Linear(self.dim, U, bias=False)
        self.value = nn.Linear(self.dim, U, bias=False)
        self.slot_down = nn.Linear(self.dim, U, bias=False)
        self.update_norm = nn.LayerNorm(U)

        # An explicit GRU cell is used instead of nn.GRUCell so all gate math
        # is visibly performed in FP32 and does not enter a fused low-precision
        # kernel.  z follows the PyTorch convention: z=1 retains the old state.
        self.gru_x_gates = nn.Linear(U, 2 * U)
        self.gru_h_gates = nn.Linear(U, 2 * U, bias=False)
        self.gru_x_candidate = nn.Linear(U, U)
        self.gru_h_candidate = nn.Linear(U, U, bias=False)
        self.slot_up = nn.Linear(U, self.dim, bias=False)
        # Kept as a standalone parameter so Hugging Face's missing-Linear-key
        # initialization cannot erase the intended retention prior when E9 is
        # bootstrapped from an E8 checkpoint.
        self.retain_bias = nn.Parameter(torch.tensor(1.3862944))

        self.mlp_norm = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )
        self.write_logit = nn.Parameter(torch.tensor(-2.1972246))  # sigmoid=0.1
        self.mlp_logit = nn.Parameter(torch.tensor(-2.9444390))  # sigmoid=0.05

        # The attention path is active from step one, but starts conservatively.
        nn.init.zeros_(self.gru_x_gates.bias)

    def forward(
        self,
        *,
        slot_states: torch.Tensor,
        image_states: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if slot_states.ndim != 3 or image_states.ndim != 3:
            raise ValueError("E9 writer expects [B,S,D] slots and [B,P,D] images")
        if slot_states.shape[0] != image_states.shape[0]:
            raise ValueError("E9 writer batch mismatch")
        if slot_states.shape[-1] != self.dim or image_states.shape[-1] != self.dim:
            raise ValueError(f"E9 writer expects hidden dim {self.dim}")
        if slot_valid.shape != slot_states.shape[:2]:
            raise ValueError("E9 slot_valid must have shape [B,S]")

        stream_dtype = slot_states.dtype
        # Whole E9 experiments are launched in FP32.  These casts additionally
        # protect the recurrent cell if the module is inspected under autocast.
        slots = slot_states.float()
        images = image_states.float()
        slot_n = self.slot_norm(slots)
        image_n = self.image_norm(images)
        query = self.query(slot_n)
        key = self.key(image_n)
        logits = torch.einsum("bsu,bpu->bsp", query, key)
        logits = logits / math.sqrt(float(self.update_dim))
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits.masked_fill(~slot_valid.unsqueeze(-1), -1e4)

        # Slot Attention normalization: patches first choose a slot, then each
        # slot receives a normalized weighted mean of its selected values.
        owner_probs = F.softmax(logits, dim=1, dtype=torch.float32)
        owner_probs = owner_probs * slot_valid.unsqueeze(-1).float()
        owner_probs = owner_probs / owner_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        write_weights = owner_probs / owner_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        values = self.value(image_n)
        visual_update = torch.einsum("bsp,bpu->bsu", write_weights, values)
        visual_update = self.update_norm(visual_update)

        old_low = self.slot_down(slot_n)
        x_reset, x_update = self.gru_x_gates(visual_update).chunk(2, dim=-1)
        h_reset, h_update = self.gru_h_gates(old_low).chunk(2, dim=-1)
        reset_gate = torch.sigmoid(x_reset + h_reset)
        update_gate = torch.sigmoid(x_update + h_update + self.retain_bias.float())
        candidate = torch.tanh(
            self.gru_x_candidate(visual_update)
            + reset_gate * self.gru_h_candidate(old_low)
        )
        recurrent = (1.0 - update_gate) * candidate + update_gate * old_low

        write_strength = torch.sigmoid(self.write_logit)
        recurrent_delta = self.slot_up(recurrent - old_low)
        slot_mid = slots + write_strength * recurrent_delta
        mlp_strength = torch.sigmoid(self.mlp_logit)
        updated = slot_mid + mlp_strength * self.mlp(self.mlp_norm(slot_mid))
        updated = torch.where(torch.isfinite(updated), updated, slots)
        updated = torch.where(slot_valid.unsqueeze(-1), updated, slots)
        write_delta = updated - slots

        return {
            "updated_slots": updated.to(dtype=stream_dtype),
            "write_delta": write_delta.to(dtype=stream_dtype),
            "owner_logits": logits,
            "owner_probs": owner_probs,
            "write_weights": write_weights,
            "write_update": visual_update,
            "write_gate_mean": (1.0 - update_gate).mean().detach(),
            "retain_gate_mean": update_gate.mean().detach(),
            "reset_gate_mean": reset_gate.mean().detach(),
            "write_strength": write_strength.detach(),
            "mlp_strength": mlp_strength.detach(),
        }


class PGOTE8TypedRAEReader(nn.Module):
    """RAE reader with semantic keys and image-only visual-memory values."""

    def __init__(self, dim: int, num_heads: int = 8, temperature: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"E8 reader dim={self.dim} must be divisible by heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        self.temperature = float(temperature)

        self.query_norm = nn.LayerNorm(self.dim)
        self.semantic_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        self.output_norm = nn.LayerNorm(self.dim)

        for layer in (self.query, self.key, self.value, self.output):
            nn.init.eye_(layer.weight)

    def forward(
        self,
        *,
        rae_queries: torch.Tensor,
        semantic_slots: torch.Tensor,
        visual_memory: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if semantic_slots.shape != visual_memory.shape:
            raise ValueError("E8 reader semantic/memory states must have identical shape")
        if slot_valid.shape != semantic_slots.shape[:2]:
            raise ValueError("E8 reader slot_valid must have shape [B,S]")
        B, Q, D = rae_queries.shape
        if D != self.dim or semantic_slots.shape[0] != B or semantic_slots.shape[-1] != D:
            raise ValueError("E8 reader shape mismatch")
        S = semantic_slots.shape[1]
        H, Dh = self.num_heads, self.head_dim

        stream_dtype = rae_queries.dtype
        module_dtype = self.query_norm.weight.dtype
        query_work = rae_queries.to(dtype=module_dtype)
        semantic_work = semantic_slots.to(dtype=module_dtype)
        memory_work = visual_memory.to(dtype=module_dtype)

        query = self.query(self.query_norm(query_work)).reshape(B, Q, H, Dh)
        key = self.key(self.semantic_norm(semantic_work)).reshape(B, S, H, Dh)
        logits = torch.einsum("bqhd,bshd->bhqs", query.float(), key.float())
        logits = logits / math.sqrt(float(Dh))
        logits = logits / max(float(self.temperature), 1e-6)
        logits = logits.masked_fill(~slot_valid[:, None, None, :], -1e4)

        attention = F.softmax(logits, dim=-1)
        attention = attention * slot_valid[:, None, None, :].float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Crucially, values originate only from visual_memory.  The semantic
        # OVT/register states are never available as reconstruction values.
        value = self.value(self.memory_norm(memory_work)).reshape(B, S, H, Dh)
        context = torch.einsum(
            "bhqs,bshd->bqhd", attention.to(value.dtype), value
        ).reshape(B, Q, D)
        condition = self.output_norm(self.output(context)).to(dtype=stream_dtype)
        condition = torch.where(torch.isfinite(condition), condition, torch.zeros_like(condition))
        return {
            "condition_hidden": condition,
            "reader_attention": attention.mean(dim=1),
            "reader_attention_heads": attention,
            "reader_entropy": (
                -(attention.float().clamp_min(1e-8).log() * attention.float()).sum(dim=-1)
            ).mean().detach(),
        }
