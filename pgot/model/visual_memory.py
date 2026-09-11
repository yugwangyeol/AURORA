"""Object-semantic / image-appearance memory blocks for PGOT E8/E9.

The writer uses object OVT states and background-register states only as
queries.  Values are projected exclusively from the MLLM image-token stream.
The reader then uses the final semantic states as keys and the accumulated
image-only memories as values, so semantic text cannot become reconstruction
content through this path.  E9.1 instead supplies the same final unified OVT
state as both key and value, making that token the explicit bottleneck.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_memory_valid_mask(
    *,
    slot_valid: torch.Tensor,
    object_count: int,
    object_memories_per_owner: int,
    register_memories_per_owner: int,
    max_memories_per_owner: int,
) -> torch.Tensor:
    """Return [B,S,Jmax] validity for heterogeneous object/register memory."""
    if slot_valid.ndim != 2:
        raise ValueError("memory validity expects slot_valid [B,S]")
    B, S = slot_valid.shape
    K = int(object_count)
    if not 0 <= K <= S:
        raise ValueError(f"object_count must be in [0,{S}], got {K}")
    owner_is_object = torch.arange(S, device=slot_valid.device) < K
    counts = torch.where(
        owner_is_object,
        torch.full(
            (S,),
            int(object_memories_per_owner),
            device=slot_valid.device,
            dtype=torch.long,
        ),
        torch.full(
            (S,),
            int(register_memories_per_owner),
            device=slot_valid.device,
            dtype=torch.long,
        ),
    )
    memory_index = torch.arange(
        int(max_memories_per_owner), device=slot_valid.device
    )
    return (
        slot_valid[:, :, None]
        & (memory_index[None, None, :] < counts[None, :, None])
    ).expand(B, -1, -1)


class PGOTE8VisualMemoryWriter(nn.Module):
    """Competitive patch ownership followed by a gated image-only write."""

    def __init__(
        self,
        dim: int,
        temperature: float = 1.0,
        raw_value_dim: int | None = None,
        memories_per_owner: int = 1,
        object_memories_per_owner: int | None = None,
        register_memories_per_owner: int | None = None,
        query_separation: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.temperature = float(temperature)
        self.object_memories_per_owner = int(
            memories_per_owner
            if object_memories_per_owner is None
            else object_memories_per_owner
        )
        self.register_memories_per_owner = int(
            memories_per_owner
            if register_memories_per_owner is None
            else register_memories_per_owner
        )
        self.memories_per_owner = max(
            self.object_memories_per_owner,
            self.register_memories_per_owner,
        )
        self.query_separation = bool(query_separation)
        if self.memories_per_owner <= 0:
            raise ValueError("memories_per_owner must be positive")
        if (
            self.object_memories_per_owner <= 0
            or self.register_memories_per_owner <= 0
        ):
            raise ValueError("object/register memory counts must be positive")
        self.raw_value_dim = (
            int(raw_value_dim) if raw_value_dim is not None else None
        )

        self.semantic_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.image_norm = nn.LayerNorm(self.dim)
        self.update_norm = nn.LayerNorm(self.dim)

        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.memory_to_query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        if self.raw_value_dim is not None:
            self.raw_value_norm = nn.LayerNorm(self.raw_value_dim)
            self.raw_value = nn.Linear(self.raw_value_dim, self.dim, bias=False)
            nn.init.xavier_uniform_(self.raw_value.weight)
        else:
            self.raw_value_norm = None
            self.raw_value = None
        self.fuse = nn.Linear(2 * self.dim, self.dim, bias=False)
        self.gate = nn.Linear(3 * self.dim, self.dim)
        self.inject = nn.Linear(self.dim, self.dim, bias=False)
        # E11 Dual-M4 keeps one semantic owner but lets four visual memories
        # compete inside that owner.  This identity embedding is the only
        # symmetry breaker; no positional/global-part prior is introduced.
        self.memory_id_embeddings = nn.Parameter(
            torch.zeros(self.memories_per_owner, self.dim)
        )
        if self.memories_per_owner > 1:
            nn.init.normal_(
                self.memory_id_embeddings,
                mean=0.0,
                # Added after the owner query projection, so unit variance
                # gives an O(1) identity contribution after /sqrt(D).
                std=1.0,
            )

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
        raw_value_states: torch.Tensor | None = None,
        slot_valid: torch.Tensor,
        object_count: int | None = None,
        clean_refinement: bool = False,
        initialize_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if semantic_slots.ndim != 3 or image_states.ndim != 3:
            raise ValueError("E8 writer expects [B,S,D] slots and [B,P,D] image states")
        memory_was_3d = visual_memory.ndim == 3
        if memory_was_3d:
            visual_memory = visual_memory.unsqueeze(2)
        if visual_memory.ndim != 4:
            raise ValueError("E8/E11 visual memory must have shape [B,S,J,D]")
        if (
            visual_memory.shape[:2] != semantic_slots.shape[:2]
            or visual_memory.shape[-1] != semantic_slots.shape[-1]
            or visual_memory.shape[2] != self.memories_per_owner
        ):
            raise ValueError(
                "E8/E11 semantic/memory shape mismatch: "
                f"semantic={tuple(semantic_slots.shape)} "
                f"memory={tuple(visual_memory.shape)} "
                f"expected_J={self.memories_per_owner}"
            )
        if semantic_slots.shape[0] != image_states.shape[0]:
            raise ValueError("E8 writer batch mismatch")
        if semantic_slots.shape[-1] != self.dim or image_states.shape[-1] != self.dim:
            raise ValueError(f"E8 writer expects hidden dim {self.dim}")
        if slot_valid.shape != semantic_slots.shape[:2]:
            raise ValueError("E8 slot_valid must have shape [B,S]")
        if object_count is None:
            object_count = semantic_slots.shape[1]
        memory_valid = _build_memory_valid_mask(
            slot_valid=slot_valid,
            object_count=int(object_count),
            object_memories_per_owner=self.object_memories_per_owner,
            register_memories_per_owner=self.register_memories_per_owner,
            max_memories_per_owner=self.memories_per_owner,
        )
        if raw_value_states is not None:
            if self.raw_value is None or self.raw_value_norm is None:
                raise ValueError(
                    "E10 raw visual values were supplied to a writer without "
                    "a raw-value projector"
                )
            if raw_value_states.ndim != 3:
                raise ValueError("E10 raw value states must have shape [B,P,C]")
            if raw_value_states.shape[:2] != image_states.shape[:2]:
                raise ValueError(
                    "E10 raw/image patch shape mismatch: "
                    f"raw={tuple(raw_value_states.shape)} "
                    f"image={tuple(image_states.shape)}"
                )
            if raw_value_states.shape[-1] != self.raw_value_dim:
                raise ValueError(
                    f"E10 writer expects raw value dim {self.raw_value_dim}, "
                    f"got {raw_value_states.shape[-1]}"
                )

        stream_dtype = semantic_slots.dtype
        module_dtype = self.semantic_norm.weight.dtype
        semantic_work = semantic_slots.to(dtype=module_dtype)
        memory_work = visual_memory.to(dtype=module_dtype)
        image_work = image_states.to(dtype=module_dtype)

        semantic_n = self.semantic_norm(semantic_work)
        memory_n = self.memory_norm(memory_work)
        # Owner routing remains exactly one query per object/register.  The
        # mean memory only carries the previous visual state into that query;
        # it does not create additional semantic owners.
        owner_memory_n = (
            memory_n * memory_valid.unsqueeze(-1).to(memory_n.dtype)
        ).sum(dim=2) / memory_valid.sum(dim=2, keepdim=True).clamp_min(1).to(
            memory_n.dtype
        )
        query = self.query(semantic_n)
        if not self.query_separation:
            query = query + self.memory_to_query(owner_memory_n)
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

        # Inside each semantic owner, J visual memories compete for its owned
        # patches.  The softmax is over J, while the owner softmax above stays
        # over semantic owners.  For J=1 this reduces exactly to E10-R.
        memory_query = self.memory_to_query(memory_n)
        if not self.query_separation:
            memory_query = memory_query + self.query(semantic_n).unsqueeze(2)
        memory_query = memory_query + self.memory_id_embeddings[
            None, None
        ].to(memory_n.dtype)
        memory_logits = torch.einsum(
            "bsjd,bpd->bsjp", memory_query.float(), key.float()
        )
        memory_logits = memory_logits / math.sqrt(float(self.dim))
        memory_logits = memory_logits / max(float(self.temperature), 1e-6)
        memory_logits = memory_logits.masked_fill(
            ~memory_valid.unsqueeze(-1), -1e4
        )
        memory_probs = F.softmax(memory_logits, dim=2)
        memory_probs = memory_probs * memory_valid.unsqueeze(-1).float()
        memory_probs = memory_probs / memory_probs.sum(
            dim=2, keepdim=True
        ).clamp_min(1e-6)

        joint_mass = owner_probs.unsqueeze(2) * memory_probs
        # Normalize over patches independently for every visual memory.  This
        # is the same attention-weighted write used by E10-R, now performed J
        # times under one owner rather than once.
        write_weights = joint_mass / joint_mass.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        patch_count = write_weights.shape[-1]
        patch_side = int(round(math.sqrt(float(patch_count))))
        if patch_side * patch_side != patch_count:
            raise ValueError(
                "E8/E11 visual-memory centroids require a square patch grid, "
                f"got P={patch_count}"
            )
        coordinate_axis = torch.linspace(
            -1.0,
            1.0,
            patch_side,
            device=write_weights.device,
            dtype=torch.float32,
        )
        coordinate_y, coordinate_x = torch.meshgrid(
            coordinate_axis, coordinate_axis, indexing="ij"
        )
        patch_coordinates = torch.stack(
            [coordinate_x.flatten(), coordinate_y.flatten()], dim=-1
        )
        # E12 uses the differentiable center of the patches that each visual
        # memory actually wrote.  No GT mask, fixed quadrant, or part label is
        # involved; the final Reader receives the Writer's own dynamic route.
        memory_centroids = torch.einsum(
            "bsjp,pd->bsjd", write_weights.float(), patch_coordinates
        ).to(dtype=module_dtype)
        if raw_value_states is None:
            values = self.value(image_n)
        else:
            raw_work = raw_value_states.to(
                device=image_work.device,
                dtype=self.raw_value_norm.weight.dtype,
            )
            values = self.raw_value(self.raw_value_norm(raw_work))
        update = torch.einsum(
            "bsjp,bpd->bsjd", write_weights.to(values.dtype), values
        )

        update_n = self.update_norm(update)
        candidate = self.fuse(torch.cat([memory_n, update_n], dim=-1))
        semantic_for_memory = semantic_n.unsqueeze(2).expand(
            -1, -1, self.memories_per_owner, -1
        )
        gate = torch.sigmoid(
            self.gate(
                torch.cat([semantic_for_memory, memory_n, update_n], dim=-1)
            )
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
            memory_valid.unsqueeze(-1), new_memory, torch.zeros_like(new_memory)
        )
        new_memory = torch.where(torch.isfinite(new_memory), new_memory, memory_work)
        new_memory = new_memory.to(dtype=stream_dtype)

        owner_mass = owner_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        memory_utilization = joint_mass.sum(dim=-1) / owner_mass
        utilization_entropy = -(
            memory_utilization
            * memory_utilization.clamp_min(1e-8).log()
        ).sum(dim=-1)
        valid_counts = memory_valid.sum(dim=-1).clamp_min(1).float()
        entropy_denom = valid_counts.log().clamp_min(1.0)
        utilization_entropy = torch.where(
            valid_counts > 1,
            utilization_entropy / entropy_denom,
            torch.zeros_like(utilization_entropy),
        )
        valid_utilization = memory_utilization[memory_valid]
        valid_entropy = utilization_entropy[slot_valid]
        assignment_entropy = -(
            memory_probs
            * memory_probs.clamp_min(1e-8).log()
        ).sum(dim=2)
        assignment_entropy = torch.where(
            valid_counts.unsqueeze(-1) > 1,
            assignment_entropy / entropy_denom.unsqueeze(-1),
            torch.zeros_like(assignment_entropy),
        )
        assignment_entropy = (
            (assignment_entropy * owner_probs).sum()
            / owner_probs.sum().clamp_min(1e-6)
        )

        if memory_was_3d:
            new_memory_out = new_memory.squeeze(2)
            update_out = update.squeeze(2)
        else:
            new_memory_out = new_memory
            update_out = update

        return {
            "visual_memory": new_memory_out,
            "owner_logits": logits,
            "owner_probs": owner_probs,
            "memory_logits": memory_logits,
            "memory_probs": memory_probs,
            "memory_valid": memory_valid,
            "write_weights": write_weights,
            "memory_centroids": memory_centroids,
            "write_update": update_out,
            "memory_utilization": memory_utilization,
            "memory_utilization_entropy": (
                valid_entropy.mean().detach()
                if valid_entropy.numel()
                else update.new_zeros(())
            ),
            "memory_assignment_entropy": assignment_entropy.detach(),
            "memory_utilization_min": (
                valid_utilization.min().detach()
                if valid_utilization.numel()
                else update.new_zeros(())
            ),
            "memory_utilization_max": (
                valid_utilization.max().detach()
                if valid_utilization.numel()
                else update.new_zeros(())
            ),
            "write_gate_mean": gate.mean().detach(),
            "write_strength": write_strength.detach(),
            "raw_value_enabled": update.new_tensor(
                float(raw_value_states is not None)
            ).detach(),
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


class _PGOTE8ReaderRefinementBlock(nn.Module):
    """Pre-norm cross-attention + FFN refinement for an existing Reader state.

    The residual output projections are zero-initialized so adding refinement
    layers exactly preserves a trained one-layer Reader at initialization.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        layer_index: int,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.layer_index = int(layer_index)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"Reader refinement dim={self.dim} must be divisible by "
                f"heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        hidden_dim = max(self.dim, int(round(self.dim * float(mlp_ratio))))

        self.query_norm = nn.LayerNorm(self.dim)
        self.key_norm = nn.LayerNorm(self.dim)
        self.value_norm = nn.LayerNorm(self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn_in = nn.Linear(self.dim, hidden_dim, bias=False)
        self.ffn_out = nn.Linear(hidden_dim, self.dim, bias=False)

        self.reset_as_identity()

    @torch.no_grad()
    def reset_as_identity(self) -> None:
        """Deterministically initialize this block as an exact residual no-op."""
        for norm in (
            self.query_norm,
            self.key_norm,
            self.value_norm,
            self.ffn_norm,
        ):
            norm.weight.fill_(1.0)
            norm.bias.zero_()
        for projection in (self.query, self.key, self.value):
            nn.init.eye_(projection.weight)
        nn.init.zeros_(self.output.weight)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1300 + self.layer_index)
        fan_out, fan_in = self.ffn_in.weight.shape
        bound = math.sqrt(6.0 / float(fan_in + fan_out))
        initialized = torch.empty(
            tuple(self.ffn_in.weight.shape), dtype=torch.float32, device="cpu"
        ).uniform_(-bound, bound, generator=generator)
        self.ffn_in.weight.copy_(
            initialized.to(
                device=self.ffn_in.weight.device,
                dtype=self.ffn_in.weight.dtype,
            )
        )
        nn.init.zeros_(self.ffn_out.weight)

    def forward(
        self,
        *,
        hidden: torch.Tensor,
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
        memory_valid: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, Q, D = hidden.shape
        T = key_tokens.shape[1]
        H, Dh = self.num_heads, self.head_dim
        query = self.query(self.query_norm(hidden)).reshape(B, Q, H, Dh)
        key = self.key(self.key_norm(key_tokens)).reshape(B, T, H, Dh)
        value = self.value(self.value_norm(value_tokens)).reshape(B, T, H, Dh)
        logits = torch.einsum("bqhd,bthd->bhqt", query.float(), key.float())
        logits = logits / math.sqrt(float(Dh))
        logits = logits / max(float(temperature), 1e-6)
        logits = logits.masked_fill(~memory_valid[:, None, None, :], -1e4)
        attention = F.softmax(logits, dim=-1)
        attention = attention * memory_valid[:, None, None, :].float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = torch.einsum(
            "bhqt,bthd->bqhd", attention.to(value.dtype), value
        ).reshape(B, Q, D)
        refined = hidden + self.output(context)
        refined = refined + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(refined))))
        refined = torch.where(torch.isfinite(refined), refined, hidden)
        return refined, attention


class PGOTE8TypedRAEReader(nn.Module):
    """RAE reader with semantic keys and image-only visual-memory values."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        temperature: float = 1.0,
        memories_per_owner: int = 1,
        object_memories_per_owner: int | None = None,
        register_memories_per_owner: int | None = None,
        centroid_position_enable: bool = False,
        centroid_gate_init: float = 0.0,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"E8 reader dim={self.dim} must be divisible by heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads
        self.temperature = float(temperature)
        self.object_memories_per_owner = int(
            memories_per_owner
            if object_memories_per_owner is None
            else object_memories_per_owner
        )
        self.register_memories_per_owner = int(
            memories_per_owner
            if register_memories_per_owner is None
            else register_memories_per_owner
        )
        self.memories_per_owner = max(
            self.object_memories_per_owner,
            self.register_memories_per_owner,
        )
        self.centroid_position_enable = bool(centroid_position_enable)
        self.num_layers = int(num_layers)
        if self.num_layers <= 0:
            raise ValueError("E8 Reader num_layers must be positive")
        if self.memories_per_owner <= 0:
            raise ValueError("memories_per_owner must be positive")
        if (
            self.object_memories_per_owner <= 0
            or self.register_memories_per_owner <= 0
        ):
            raise ValueError("object/register memory counts must be positive")

        self.query_norm = nn.LayerNorm(self.dim)
        self.semantic_norm = nn.LayerNorm(self.dim)
        self.memory_norm = nn.LayerNorm(self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.value = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        self.output_norm = nn.LayerNorm(self.dim)
        # Reader keys remain semantic-owner keys, with only a learned memory
        # identity offset distinguishing the J visual values under each owner.
        self.memory_key_embeddings = nn.Parameter(
            torch.zeros(self.memories_per_owner, self.dim)
        )
        if self.memories_per_owner > 1:
            nn.init.normal_(
                self.memory_key_embeddings,
                mean=0.0,
                # This offset is added after key projection; match the
                # projected semantic-key scale to avoid an all-uniform read.
                std=1.0,
            )

        # E12: expose where each visual memory actually read from to the typed
        # Reader.  Fourier features give a smooth 2-D code; separate object and
        # register gates let the two typed paths adopt position independently.
        # Both gates start at zero, so loading E11 initially reproduces its key
        # computation exactly while retaining a gradient into each gate.
        self.centroid_feature_dim = 14
        if self.centroid_position_enable:
            self.centroid_position_projector = nn.Linear(
                self.centroid_feature_dim, self.dim, bias=False
            )
            self.centroid_position_norm = nn.LayerNorm(self.dim)
            self.centroid_object_gate = nn.Parameter(
                torch.tensor(float(centroid_gate_init))
            )
            self.centroid_register_gate = nn.Parameter(
                torch.tensor(float(centroid_gate_init))
            )
            nn.init.xavier_uniform_(self.centroid_position_projector.weight)
        else:
            self.centroid_position_projector = None
            self.centroid_position_norm = None
            self.register_parameter("centroid_object_gate", None)
            self.register_parameter("centroid_register_gate", None)

        for layer in (self.query, self.key, self.value, self.output):
            nn.init.eye_(layer.weight)
        self.refinement_layers = nn.ModuleList(
            [
                _PGOTE8ReaderRefinementBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    layer_index=layer_index,
                )
                for layer_index in range(1, self.num_layers)
            ]
        )

    def forward(
        self,
        *,
        rae_queries: torch.Tensor,
        semantic_slots: torch.Tensor,
        visual_memory: torch.Tensor,
        slot_valid: torch.Tensor,
        memory_centroids: torch.Tensor | None = None,
        object_count: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        if slot_valid.shape != semantic_slots.shape[:2]:
            raise ValueError("E8 reader slot_valid must have shape [B,S]")
        if visual_memory.ndim == 3:
            visual_memory = visual_memory.unsqueeze(2)
        if visual_memory.ndim != 4:
            raise ValueError("E8/E11 Reader visual memory must be [B,S,J,D]")
        B, Q, D = rae_queries.shape
        if (
            D != self.dim
            or semantic_slots.shape[0] != B
            or semantic_slots.shape[-1] != D
            or visual_memory.shape[:2] != semantic_slots.shape[:2]
            or visual_memory.shape[-1] != D
            or visual_memory.shape[2] != self.memories_per_owner
        ):
            raise ValueError("E8 reader shape mismatch")
        S = semantic_slots.shape[1]
        J = visual_memory.shape[2]
        H, Dh = self.num_heads, self.head_dim
        if object_count is None:
            object_count = S
        memory_valid_3d = _build_memory_valid_mask(
            slot_valid=slot_valid,
            object_count=int(object_count),
            object_memories_per_owner=self.object_memories_per_owner,
            register_memories_per_owner=self.register_memories_per_owner,
            max_memories_per_owner=self.memories_per_owner,
        )

        stream_dtype = rae_queries.dtype
        module_dtype = self.query_norm.weight.dtype
        query_work = rae_queries.to(dtype=module_dtype)
        semantic_work = semantic_slots.to(dtype=module_dtype)
        memory_work = visual_memory.to(dtype=module_dtype)

        query = self.query(self.query_norm(query_work)).reshape(B, Q, H, Dh)
        semantic_key = self.key(self.semantic_norm(semantic_work)).unsqueeze(2)
        key = semantic_key + self.memory_key_embeddings[None, None].to(
            semantic_key.dtype
        )
        zero = semantic_key.new_zeros(())
        centroid_position_rms = zero
        centroid_mean_radius = zero
        object_gate_value = zero
        register_gate_value = zero
        if self.centroid_position_enable:
            if memory_centroids is None:
                raise ValueError(
                    "E12 centroid-aware Reader requires memory_centroids"
                )
            if memory_centroids.shape != (B, S, J, 2):
                raise ValueError(
                    "E12 memory centroids must have shape [B,S,J,2], got "
                    f"{tuple(memory_centroids.shape)}"
                )
            if object_count is None or not 0 <= int(object_count) <= S:
                raise ValueError(
                    f"E12 Reader requires object_count in [0,{S}], got {object_count}"
                )
            coordinates = memory_centroids.to(
                device=semantic_key.device, dtype=module_dtype
            ).clamp(-1.0, 1.0)
            features = [coordinates]
            for frequency in (1.0, 2.0, 4.0):
                phase = math.pi * frequency * coordinates
                features.extend([phase.sin(), phase.cos()])
            centroid_features = torch.cat(features, dim=-1)
            centroid_position = self.centroid_position_norm(
                self.centroid_position_projector(centroid_features)
            )
            object_gate_value = torch.tanh(self.centroid_object_gate).to(
                dtype=module_dtype
            )
            register_gate_value = torch.tanh(self.centroid_register_gate).to(
                dtype=module_dtype
            )
            owner_is_object = (
                torch.arange(S, device=semantic_key.device) < int(object_count)
            ).view(1, S, 1, 1)
            owner_gate = torch.where(
                owner_is_object, object_gate_value, register_gate_value
            )
            centroid_delta = owner_gate * centroid_position
            key = key + centroid_delta
            valid_position = slot_valid[:, :, None, None].to(
                centroid_delta.dtype
            )
            centroid_position_rms = (
                (centroid_delta.float().square() * valid_position.float()).sum()
                / valid_position.float().expand_as(centroid_delta).sum().clamp_min(1.0)
            ).sqrt()
            valid_centroid = slot_valid[:, :, None].to(coordinates.dtype)
            centroid_mean_radius = (
                coordinates.float().square().sum(dim=-1).sqrt()
                * valid_centroid.float()
            ).sum() / valid_centroid.float().expand(B, S, J).sum().clamp_min(1.0)
        key_tokens = key.reshape(B, S * J, D)
        key = key_tokens.reshape(B, S * J, H, Dh)
        logits = torch.einsum("bqhd,bthd->bhqt", query.float(), key.float())
        logits = logits / math.sqrt(float(Dh))
        logits = logits / max(float(self.temperature), 1e-6)
        memory_valid = memory_valid_3d.reshape(B, S * J)
        logits = logits.masked_fill(~memory_valid[:, None, None, :], -1e4)

        attention = F.softmax(logits, dim=-1)
        attention = attention * memory_valid[:, None, None, :].float()
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        # Crucially, values originate only from visual_memory.  The semantic
        # OVT/register states are never available as reconstruction values.
        value_tokens = self.value(self.memory_norm(memory_work)).reshape(B, S * J, D)
        value = value_tokens.reshape(B, S * J, H, Dh)
        context = torch.einsum(
            "bhqt,bthd->bqhd", attention.to(value.dtype), value
        ).reshape(B, Q, D)
        condition_work = self.output_norm(self.output(context))
        # The trained first layer remains the explicit owner-routing layer and
        # therefore continues to receive Reader supervision.  Deeper blocks
        # refine the visual value readout without replacing that routing target
        # with randomly initialized attention maps.
        routing_attention = attention
        for refinement in self.refinement_layers:
            condition_work, _ = refinement(
                hidden=condition_work,
                key_tokens=key_tokens,
                value_tokens=value_tokens,
                memory_valid=memory_valid,
                temperature=self.temperature,
            )
        attention = routing_attention
        condition = condition_work.to(dtype=stream_dtype)
        condition = torch.where(torch.isfinite(condition), condition, torch.zeros_like(condition))
        return {
            "condition_hidden": condition,
            "reader_attention": attention.mean(dim=1),
            "reader_owner_attention": attention.mean(dim=1).reshape(
                B, Q, S, J
            ).sum(dim=-1),
            "reader_attention_heads": attention,
            "centroid_position_enabled": zero.new_tensor(
                float(self.centroid_position_enable)
            ),
            "centroid_object_gate": object_gate_value.detach(),
            "centroid_register_gate": register_gate_value.detach(),
            "centroid_position_rms": centroid_position_rms.detach(),
            "centroid_mean_radius": centroid_mean_radius.detach(),
            "reader_num_layers": zero.new_tensor(float(self.num_layers)),
            "reader_entropy": (
                -(attention.float().clamp_min(1e-8).log() * attention.float()).sum(dim=-1)
            ).mean().detach(),
        }


class PGOTOneShotMemoryWriter(nn.Module):
    """One write: (final semantic query + E11 ID) attends patches under ownership.

    Names intentionally match E11 so its projections and IDs load unchanged.
    There is no previous-memory input, update gate, or recurrent state.
    """

    def __init__(self, *, dim, raw_value_dim, object_memories_per_owner=4,
                 register_memories_per_owner=16, temperature=1.0,
                 detach_owner_routing=True, softmax_axis="patch"):
        super().__init__()
        self.dim = int(dim)
        self.object_memories_per_owner = int(object_memories_per_owner)
        self.register_memories_per_owner = int(register_memories_per_owner)
        if min(self.object_memories_per_owner, self.register_memories_per_owner) < 1:
            raise ValueError("one-shot memory counts must be positive")
        self.memories_per_owner = max(self.object_memories_per_owner,
                                     self.register_memories_per_owner)
        self.temperature = float(temperature)
        self.detach_owner_routing = bool(detach_owner_routing)
        self.softmax_axis = str(softmax_axis).strip().lower()
        if self.softmax_axis not in {"patch", "memory"}:
            raise ValueError(
                "one-shot Writer softmax_axis must be 'patch' or 'memory', "
                f"got {softmax_axis!r}"
            )
        self.semantic_norm = nn.LayerNorm(dim)
        self.image_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.raw_value_norm = nn.LayerNorm(raw_value_dim)
        self.raw_value = nn.Linear(raw_value_dim, dim, bias=False)
        self.memory_id_embeddings = nn.Parameter(torch.empty(self.memories_per_owner, dim))
        nn.init.normal_(self.memory_id_embeddings)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        nn.init.xavier_uniform_(self.raw_value.weight)

    def forward(self, *, semantic_slots, image_states, raw_value_states,
                owner_probs, slot_valid, object_count,
                owner_gradient_scale=1.0):
        B, S, D = semantic_slots.shape
        P = image_states.shape[1]
        if image_states.shape != (B, P, D) or owner_probs.shape != (B, S, P):
            raise ValueError("one-shot Writer image/ownership shape mismatch")
        if raw_value_states.shape[:2] != (B, P) or not 0 <= object_count <= S:
            raise ValueError("one-shot Writer raw patches/object count mismatch")
        valid = _build_memory_valid_mask(
            slot_valid=slot_valid, object_count=object_count,
            object_memories_per_owner=self.object_memories_per_owner,
            register_memories_per_owner=self.register_memories_per_owner,
            max_memories_per_owner=self.memories_per_owner,
        )
        dtype = self.query.weight.dtype
        query = self.query(self.semantic_norm(semantic_slots.to(dtype)))[:, :, None]
        query = query + self.memory_id_embeddings[None, None]
        key = self.key(self.image_norm(image_states.to(dtype)))
        logits = torch.einsum("bsjd,bpd->bsjp", query.float(), key.float())
        logits = logits / math.sqrt(D) / max(self.temperature, 1e-6)
        routing = owner_probs.float()
        if self.detach_owner_routing:
            routing = routing.detach()
        else:
            # Preserve the forward routing while ramping only the reconstruction
            # gradient that reaches semantic ownership.
            scale = min(max(float(owner_gradient_scale), 0.0), 1.0)
            detached = routing.detach()
            routing = detached + scale * (routing - detached)
        routing = routing * slot_valid[..., None].float()
        routing = routing / routing.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logits = logits + routing.clamp_min(1e-8).log()[:, :, None]
        allocation_mass = None
        if self.softmax_axis == "patch":
            # Each memory independently selects source patches. This preserves
            # the behavior of existing one-shot memory checkpoints.
            weights = F.softmax(logits, dim=-1) * valid[..., None].float()
        else:
            # Each patch is allocated competitively among the J memories under
            # its semantic owner. The owner prior is shared across J, while the
            # content score decides which memory receives the patch. Normalize
            # once more over P so every valid memory remains a weighted average.
            allocation_logits = logits.masked_fill(
                ~valid[..., None], -1e4
            )
            allocation = F.softmax(allocation_logits, dim=2)
            allocation = allocation * valid[..., None].float()
            allocation = allocation / allocation.sum(
                dim=2, keepdim=True
            ).clamp_min(1e-8)
            joint_mass = allocation * routing[:, :, None]
            allocation_mass = joint_mass.sum(dim=-1)
            weights = joint_mass / allocation_mass[..., None].clamp_min(1e-8)
            weights = weights * valid[..., None].float()
        raw = raw_value_states.to(self.raw_value.weight.dtype)
        values = self.raw_value(self.raw_value_norm(raw))
        memory = torch.einsum("bsjp,bpd->bsjd", weights.to(values.dtype), values)
        memory = memory.to(semantic_slots.dtype)
        result = {
            "visual_memory": memory,
            "memory_valid": valid,
            "write_weights": weights,
        }
        if allocation_mass is not None:
            result["memory_allocation_mass"] = allocation_mass
        return result


class PGOTOneShotMemoryReader(nn.Module):
    """G(q, semantic owner) * beta(q, within-owner memory), then visual values.

    The interface accepts stored memories only: no source-patch bypass.
    memory_content and memory_id differ only in the within-owner keys.
    """

    def __init__(self, *, dim, num_heads=8, memories_per_owner=16,
                 readout_mode="memory_content", temperature=1.0):
        super().__init__()
        if readout_mode not in {"memory_content", "memory_id"}:
            raise ValueError("expected memory_content or memory_id")
        if dim % num_heads:
            raise ValueError("Reader dimension must be divisible by heads")
        self.dim, self.num_heads = int(dim), int(num_heads)
        self.memories_per_owner = int(memories_per_owner)
        self.readout_mode, self.temperature = readout_mode, float(temperature)
        self.query_norm = nn.LayerNorm(dim)
        self.semantic_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.output_norm = nn.LayerNorm(dim)
        if readout_mode == "memory_content":
            self.content_key = nn.Linear(dim, dim, bias=False)
            nn.init.eye_(self.content_key.weight)
        else:
            self.memory_key_embeddings = nn.Parameter(torch.empty(memories_per_owner, dim))
            nn.init.normal_(self.memory_key_embeddings)
        for layer in (self.query, self.key, self.value, self.output):
            nn.init.eye_(layer.weight)

    def forward(self, *, rae_queries, semantic_slots, visual_memory,
                slot_valid, memory_valid):
        B, Q, D = rae_queries.shape
        S, J = visual_memory.shape[1:3]
        if visual_memory.shape != (B, S, J, D) or J != self.memories_per_owner:
            raise ValueError("one-shot Reader memory shape mismatch")
        if semantic_slots.shape != (B, S, D) or memory_valid.shape != (B, S, J):
            raise ValueError("one-shot Reader owner/validity shape mismatch")
        if slot_valid.shape != (B, S):
            raise ValueError("one-shot Reader slot_valid shape mismatch")
        H, Dh = self.num_heads, D // self.num_heads
        dtype = self.query.weight.dtype
        query = self.query(self.query_norm(rae_queries.to(dtype))).reshape(B, Q, H, Dh)
        semantic_key = self.key(self.semantic_norm(semantic_slots.to(dtype))).reshape(B, S, H, Dh)
        scale = math.sqrt(Dh) * max(self.temperature, 1e-6)
        owner_logits = torch.einsum("bqhd,bshd->bhqs", query.float(), semantic_key.float()) / scale
        owner_valid = slot_valid & memory_valid.any(dim=-1)
        owner_attention = F.softmax(owner_logits.masked_fill(~owner_valid[:, None, None], -1e4), -1)
        owner_attention = owner_attention * owner_valid[:, None, None].float()
        owner_attention = owner_attention / owner_attention.sum(-1, keepdim=True).clamp_min(1e-8)
        memory_n = self.memory_norm(visual_memory.to(dtype))
        if self.readout_mode == "memory_content":
            keys = self.content_key(memory_n).reshape(B, S, J, H, Dh)
        else:
            keys = self.memory_key_embeddings[None, None].expand(B, S, J, D).reshape(B, S, J, H, Dh)
        inner_logits = torch.einsum("bqhd,bsjhd->bhqsj", query.float(), keys.float()) / scale
        beta = F.softmax(inner_logits.masked_fill(~memory_valid[:, None, None], -1e4), -1)
        beta = beta * memory_valid[:, None, None].float()
        beta = beta / beta.sum(-1, keepdim=True).clamp_min(1e-8)
        joint = owner_attention[..., None] * beta
        values = self.value(memory_n).reshape(B, S, J, H, Dh)
        context = torch.einsum("bhqsj,bsjhd->bqhd", joint.to(values.dtype), values).reshape(B, Q, D)
        condition = self.output_norm(self.output(context)).to(rae_queries.dtype)
        inner_entropy = -(beta * beta.clamp_min(1e-8).log()).sum(-1)
        return {
            "condition_hidden": condition,
            "reader_owner_attention": owner_attention.mean(1),
            "reader_attention_heads": owner_attention,
            "reader_memory_attention": joint.mean(1).flatten(2),
            "reader_inner_attention_heads": beta,
            "reader_entropy": -(owner_attention * owner_attention.clamp_min(1e-8).log()).sum(-1).mean().detach(),
            "memory_reader_entropy": (inner_entropy * owner_attention).sum(-1).mean().detach(),
        }


class PGOTOneShotOwnerReader(nn.Module):
    """One-shot object-routed readout from frozen raw SigLIP patches.

    The semantic owner attention ``G(q_i, s_k)`` is shared by both modes.
    ``pooled`` composes it with the semantic-only patch ownership map and is
    therefore the clean owner-vector bottleneck baseline.  ``owner_masked``
    lets every attention head select exactly one owner and performs a second,
    query-dependent attention only over patches hard-assigned to that owner.
    Raw SigLIP features are the sole reconstruction values in both modes.
    """

    VALID_MODES = {"pooled", "owner_masked"}

    def __init__(
        self,
        *,
        dim: int,
        raw_value_dim: int,
        num_heads: int = 8,
        temperature: float = 1.0,
        readout_mode: str = "pooled",
        detach_owner_routing: bool = True,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.raw_value_dim = int(raw_value_dim)
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)
        self.readout_mode = str(readout_mode).strip().lower()
        self.detach_owner_routing = bool(detach_owner_routing)
        if self.readout_mode not in self.VALID_MODES:
            raise ValueError(
                f"one-shot readout_mode must be one of {sorted(self.VALID_MODES)}, "
                f"got {readout_mode!r}"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"one-shot reader dim={self.dim} must be divisible by "
                f"heads={self.num_heads}"
            )
        self.head_dim = self.dim // self.num_heads

        # Keep these names compatible with the trained E11 typed Reader so its
        # query-to-owner routing projections initialise from the checkpoint.
        self.query_norm = nn.LayerNorm(self.dim)
        self.semantic_norm = nn.LayerNorm(self.dim)
        self.query = nn.Linear(self.dim, self.dim, bias=False)
        self.key = nn.Linear(self.dim, self.dim, bias=False)
        self.output = nn.Linear(self.dim, self.dim, bias=False)
        self.output_norm = nn.LayerNorm(self.dim)

        # Raw SigLIP is never exposed outside the owner route.  In pooled mode
        # only raw_value is used; raw_key is active only for the hard-masked
        # within-owner detail readout.
        self.raw_norm = nn.LayerNorm(self.raw_value_dim)
        self.raw_key = nn.Linear(self.raw_value_dim, self.dim, bias=False)
        self.raw_value = nn.Linear(self.raw_value_dim, self.dim, bias=False)

        for projection in (self.query, self.key, self.output):
            nn.init.eye_(projection.weight)
        nn.init.xavier_uniform_(self.raw_key.weight)
        nn.init.xavier_uniform_(self.raw_value.weight)

    def forward(
        self,
        *,
        rae_queries: torch.Tensor,
        semantic_slots: torch.Tensor,
        raw_patches: torch.Tensor,
        owner_probs: torch.Tensor,
        slot_valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, Q, D = rae_queries.shape
        S = semantic_slots.shape[1]
        P = raw_patches.shape[1]
        if D != self.dim or semantic_slots.shape != (B, S, D):
            raise ValueError("one-shot Reader semantic/query shape mismatch")
        if raw_patches.shape != (B, P, self.raw_value_dim):
            raise ValueError("one-shot Reader raw SigLIP shape mismatch")
        if owner_probs.shape != (B, S, P):
            raise ValueError(
                "one-shot Reader owner_probs must be [B,S,P], got "
                f"{tuple(owner_probs.shape)}"
            )
        if slot_valid.shape != (B, S):
            raise ValueError("one-shot Reader slot_valid must be [B,S]")

        stream_dtype = rae_queries.dtype
        module_dtype = self.query_norm.weight.dtype
        H, Dh = self.num_heads, self.head_dim
        q_work = rae_queries.to(dtype=module_dtype)
        s_work = semantic_slots.to(dtype=module_dtype)
        raw_work = raw_patches.to(dtype=self.raw_norm.weight.dtype)

        query = self.query(self.query_norm(q_work)).reshape(B, Q, H, Dh)
        semantic_key = self.key(self.semantic_norm(s_work)).reshape(B, S, H, Dh)
        owner_logits = torch.einsum(
            "bqhd,bshd->bhqs", query.float(), semantic_key.float()
        )
        owner_logits = owner_logits / math.sqrt(float(Dh))
        owner_logits = owner_logits / max(self.temperature, 1e-6)
        owner_logits = owner_logits.masked_fill(
            ~slot_valid[:, None, None, :], -1e4
        )
        owner_attention = F.softmax(owner_logits, dim=-1)
        owner_attention = owner_attention * slot_valid[:, None, None, :].float()
        owner_attention = owner_attention / owner_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        routing = owner_probs.float()
        if self.detach_owner_routing:
            routing = routing.detach()
        routing = routing * slot_valid.unsqueeze(-1).float()
        routing = routing / routing.sum(dim=1, keepdim=True).clamp_min(1e-6)

        raw_n = self.raw_norm(raw_work)
        value = self.raw_value(raw_n).reshape(B, P, H, Dh)
        zero = owner_attention.new_zeros(())
        hard_outside_mass = zero

        if self.readout_mode == "pooled":
            patch_distribution = routing / routing.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            patch_attention = torch.einsum(
                "bhqs,bsp->bhqp", owner_attention, patch_distribution
            )
            patch_attention = patch_attention / patch_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
        else:
            # Each head/query chooses exactly one semantic owner.  Each patch
            # also has exactly one predicted owner.  Equality of those two
            # hard assignments is a strict support mask: content logits can
            # never cross an owner boundary.
            selected_owner = owner_attention.argmax(dim=-1)  # [B,H,Q]
            patch_owner = routing.argmax(dim=1)  # [B,P]
            allowed = selected_owner.unsqueeze(-1) == patch_owner[:, None, None]

            # An early or padded owner can receive no top-1 patch.  Give such a
            # query its owner's highest-probability patch so softmax never sees
            # a fully masked row.
            expanded_routing = routing[:, None].expand(B, H, S, P)
            selected_maps = expanded_routing.gather(
                2,
                selected_owner.unsqueeze(-1).expand(B, H, Q, P),
            )
            fallback_patch = selected_maps.argmax(dim=-1, keepdim=True)
            has_allowed = allowed.any(dim=-1, keepdim=True)
            fallback = torch.zeros_like(allowed).scatter(
                -1, fallback_patch, True
            )
            allowed = allowed | ((~has_allowed) & fallback)

            raw_key = self.raw_key(raw_n).reshape(B, P, H, Dh)
            content_logits = torch.einsum(
                "bqhd,bphd->bhqp", query.float(), raw_key.float()
            ) / math.sqrt(float(Dh))
            content_logits = content_logits / max(self.temperature, 1e-6)
            content_logits = content_logits.masked_fill(~allowed, -1e4)
            patch_attention = F.softmax(content_logits, dim=-1)
            patch_attention = patch_attention * allowed.float()
            patch_attention = patch_attention / patch_attention.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            hard_outside_mass = (
                patch_attention * (~allowed).float()
            ).sum(dim=-1).mean()

            # Forward value is exactly one.  The straight-through scale lets
            # reconstruction gradients still improve q->owner confidence;
            # the explicit Reader loss remains the primary routing teacher.
            selected_prob = owner_attention.gather(
                -1, selected_owner.unsqueeze(-1)
            ).squeeze(-1)
            straight_through_owner = (
                torch.ones_like(selected_prob)
                + selected_prob
                - selected_prob.detach()
            )
            patch_attention = patch_attention * straight_through_owner.unsqueeze(-1)

        context = torch.einsum(
            "bhqp,bphd->bqhd", patch_attention.to(value.dtype), value
        ).reshape(B, Q, D)
        condition = self.output_norm(self.output(context)).to(dtype=stream_dtype)
        condition = torch.where(
            torch.isfinite(condition), condition, torch.zeros_like(condition)
        )

        # Owner-pooled values are diagnostics and provide a common intervention
        # representation; owner_masked reconstruction does not consume them.
        owner_patch_distribution = routing / routing.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        owner_values = torch.einsum(
            "bsp,bpd->bsd",
            owner_patch_distribution.to(raw_n.dtype),
            self.raw_value(raw_n),
        ).to(dtype=stream_dtype)
        valid_owner_values = owner_values * slot_valid.unsqueeze(-1).to(
            owner_values.dtype
        )

        return {
            "condition_hidden": condition,
            "reader_owner_attention": owner_attention.mean(dim=1),
            "reader_attention_heads": owner_attention,
            "reader_patch_attention": patch_attention.mean(dim=1),
            "reader_patch_attention_heads": patch_attention,
            "visual_memory": valid_owner_values,
            "reader_entropy": (
                -(
                    owner_attention.float().clamp_min(1e-8).log()
                    * owner_attention.float()
                ).sum(dim=-1)
            ).mean().detach(),
            "patch_entropy": (
                -(
                    patch_attention.float().clamp_min(1e-8).log()
                    * patch_attention.float()
                ).sum(dim=-1)
            ).mean().detach(),
            "hard_outside_mass": hard_outside_mass.detach(),
            "hard_owner_fraction": zero.new_tensor(
                float(self.readout_mode == "owner_masked")
            ),
        }
