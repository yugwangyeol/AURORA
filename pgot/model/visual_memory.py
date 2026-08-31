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
        key = key.reshape(B, S * J, H, Dh)
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
        value = self.value(self.memory_norm(memory_work)).reshape(B, S * J, H, Dh)
        context = torch.einsum(
            "bhqt,bthd->bqhd", attention.to(value.dtype), value
        ).reshape(B, Q, D)
        condition = self.output_norm(self.output(context)).to(dtype=stream_dtype)
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
            "reader_entropy": (
                -(attention.float().clamp_min(1e-8).log() * attention.float()).sum(dim=-1)
            ).mean().detach(),
        }
