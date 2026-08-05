"""E7 PGOT-SD causal ownership bottleneck.

The decoder accepts only a variable set of caption-defined OVT states and a
small fixed set of background registers.  A patch-to-owner router both defines
the segmentation readout and performs the image-only visual write used to
construct the Stable-Diffusion condition.  No spatial ownership map is passed
to the U-Net.
"""

from __future__ import annotations

import contextlib
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer


class _LowRankResidualLinear(nn.Module):
    """Frozen linear layer plus a zero-initialized trainable low-rank update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base_layer = base
        self.base_layer.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_down = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_up = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(hidden_states)
        update_input = hidden_states.to(dtype=self.lora_down.weight.dtype)
        update = self.lora_up(self.lora_down(update_input)) * self.scale
        return base + update.to(dtype=base.dtype)


class PGOTE7OwnerVisualWrite(nn.Module):
    """Competitive patch ownership followed by a predicted visual write.

    Every image patch normalizes over valid OVTs and all registers.  The same
    per-head probabilities are then normalized over patches to aggregate visual
    values for each owner.  This makes the ownership map causal for the exported
    decoder condition without feeding a high-bandwidth spatial map to SD.
    """

    def __init__(
        self,
        hidden_dim: int,
        router_dim: int = 512,
        num_heads: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if router_dim % num_heads != 0:
            raise ValueError(
                f"router_dim={router_dim} must be divisible by num_heads={num_heads}"
            )
        self.hidden_dim = int(hidden_dim)
        self.router_dim = int(router_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.router_dim // self.num_heads
        self.temperature = float(temperature)

        self.owner_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        self.owner_query = nn.Linear(hidden_dim, router_dim, bias=False)
        self.image_key = nn.Linear(hidden_dim, router_dim, bias=False)
        self.image_value = nn.Linear(hidden_dim, router_dim, bias=False)
        self.visual_out = nn.Linear(router_dim, hidden_dim, bias=False)
        self.mix_norm = nn.LayerNorm(hidden_dim * 2)
        self.mix_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        owners = torch.cat([ovt_hidden, register_hidden], dim=1)
        batch_size, n_owner, _ = owners.shape
        n_ovt = ovt_hidden.shape[1]
        n_register = register_hidden.shape[1]
        register_valid = torch.ones(
            batch_size,
            n_register,
            device=owners.device,
            dtype=torch.bool,
        )
        owner_valid = torch.cat([ovt_valid_mask.bool(), register_valid], dim=1)
        if not owner_valid.any(dim=1).all():
            raise ValueError("E7 requires at least one valid owner for every sample")

        owner_input = owners.to(dtype=self.owner_norm.weight.dtype)
        image_input = image_hidden.to(dtype=self.image_norm.weight.dtype)
        queries = self.owner_query(self.owner_norm(owner_input))
        keys = self.image_key(self.image_norm(image_input))
        values = self.image_value(self.image_norm(image_input))
        queries = queries.reshape(
            batch_size, n_owner, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        keys = keys.reshape(
            batch_size, image_hidden.shape[1], self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        values = values.reshape(
            batch_size, image_hidden.shape[1], self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)

        scale = math.sqrt(float(self.head_dim)) * max(self.temperature, 1e-6)
        logits = torch.einsum("bhpd,bhod->bhpo", keys.float(), queries.float()) / scale
        logits = logits.masked_fill(
            ~owner_valid[:, None, None, :], torch.finfo(logits.dtype).min
        )
        owner_probs_heads = torch.softmax(logits, dim=-1).to(values.dtype)
        owner_probs = owner_probs_heads.float().mean(dim=1).transpose(1, 2)

        # Normalize each owner's predicted responsibility over patches for the
        # actual image-only write. Invalid OVTs receive exactly zero.
        numerator = torch.einsum("bhpo,bhpd->bhod", owner_probs_heads, values)
        denominator = owner_probs_heads.sum(dim=2).unsqueeze(-1).clamp(min=1e-6)
        visual = numerator / denominator
        visual = visual.permute(0, 2, 1, 3).reshape(batch_size, n_owner, self.router_dim)
        visual = self.visual_out(visual)
        mixed = self.mix_mlp(self.mix_norm(torch.cat([owner_input, visual], dim=-1)))
        output = self.output_norm(mixed)
        output = output * owner_valid.unsqueeze(-1).to(output.dtype)

        return {
            "ovt_hidden": output[:, :n_ovt],
            "register_hidden": output[:, n_ovt:],
            "owner_logits_heads": logits,
            "owner_probs": owner_probs,
            "owner_probs_heads": owner_probs_heads,
            "owner_valid_mask": owner_valid,
        }


class PGOTE7SDCausalOwnership(nn.Module):
    """SD-v1.5 decoder with competitive ownership and same-category swaps."""

    def __init__(
        self,
        llm_dim: int,
        model_id: str,
        context_dim: int = 768,
        router_dim: int = 512,
        owner_num_heads: int = 8,
        owner_temperature: float = 1.0,
        owner_bg_weight: float = 0.25,
        cfg_drop_rate: float = 0.1,
        unet_lora_rank: int = 8,
        unet_lora_alpha: float = 8.0,
        causal_min_timestep: int = 700,
        causal_margin: float = 0.05,
        causal_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.context_dim = int(context_dim)
        self.owner_bg_weight = float(owner_bg_weight)
        self.cfg_drop_rate = float(cfg_drop_rate)
        self.causal_min_timestep = int(causal_min_timestep)
        self.causal_margin = float(causal_margin)
        self.causal_temperature = float(causal_temperature)

        self.owner_write = PGOTE7OwnerVisualWrite(
            hidden_dim=llm_dim,
            router_dim=router_dim,
            num_heads=owner_num_heads,
            temperature=owner_temperature,
        )
        self.context_norm = nn.LayerNorm(llm_dim)
        self.context_projector = nn.Linear(llm_dim, context_dim, bias=False)
        self.context_out_norm = nn.LayerNorm(context_dim)

        self.scheduler_train = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.scheduler_test = DDIMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(self.model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        )
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.eval()
        self.unet.eval()
        self._install_unet_lora(
            rank=int(unet_lora_rank), alpha=float(unet_lora_alpha)
        )

        # The 77 empty-prompt tokens exist only on the unconditional CFG branch.
        # They are never appended to the positive/sample-specific condition.
        tokenizer = CLIPTokenizer.from_pretrained(self.model_id, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        )
        text_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            empty_prompt = text_encoder(text_inputs.input_ids)[0]
        self.register_buffer(
            "empty_prompt_embedding", empty_prompt.float(), persistent=False
        )
        del text_encoder, tokenizer

    def _install_unet_lora(self, rank: int, alpha: float) -> None:
        wrapped = 0
        for _, module in self.unet.named_modules():
            if not hasattr(module, "to_k") or not hasattr(module, "to_v"):
                continue
            # Self-attention has no cross_attention_dim; E7 adapts attn2 only.
            if getattr(module, "is_cross_attention", False) is not True:
                continue
            if isinstance(module.to_k, nn.Linear):
                module.to_k = _LowRankResidualLinear(module.to_k, rank, alpha)
                wrapped += 1
            if isinstance(module.to_v, nn.Linear):
                module.to_v = _LowRankResidualLinear(module.to_v, rank, alpha)
                wrapped += 1
        if wrapped == 0:
            raise RuntimeError("No SD U-Net cross-attention K/V layers were wrapped")
        self.unet_lora_layer_count = wrapped

    def enable_trainable_parameters(self) -> Tuple[int, int, int]:
        owner_count = 0
        for parameter in self.owner_write.parameters():
            parameter.requires_grad_(True)
            owner_count += parameter.numel()
        context_count = 0
        for module in (self.context_norm, self.context_projector, self.context_out_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                context_count += parameter.numel()
        unet_count = 0
        for name, parameter in self.unet.named_parameters():
            trainable = ".lora_down." in name or ".lora_up." in name
            parameter.requires_grad_(trainable)
            if trainable:
                unet_count += parameter.numel()
        return owner_count, context_count, unet_count

    def train(self, mode: bool = True):
        super().train(mode)
        self.vae.eval()
        self.unet.eval()
        return self

    def route_and_write(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.owner_write(
            ovt_hidden=ovt_hidden,
            register_hidden=register_hidden,
            image_hidden=image_hidden,
            ovt_valid_mask=ovt_valid_mask,
        )

    def _project_context(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat([ovt_hidden, register_hidden], dim=1)
        hidden = hidden.to(dtype=self.context_norm.weight.dtype)
        register_valid = torch.ones(
            register_hidden.shape[:2], device=hidden.device, dtype=torch.bool
        )
        valid = torch.cat([ovt_valid_mask.bool(), register_valid], dim=1)
        context = self.context_out_norm(
            self.context_projector(self.context_norm(hidden))
        )
        context = context * valid.unsqueeze(-1).to(context.dtype)
        return context, valid

    @staticmethod
    def _pad_context(
        context: torch.Tensor, mask: torch.Tensor, length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if context.shape[1] < length:
            context = F.pad(context, (0, 0, 0, length - context.shape[1]))
            mask = F.pad(mask, (0, length - mask.shape[1]), value=False)
        return context[:, :length], mask[:, :length]

    def _negative_context(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        negative = self.empty_prompt_embedding.to(device=device, dtype=dtype).expand(
            batch_size, -1, -1
        )
        mask = torch.ones(
            negative.shape[:2], device=device, dtype=torch.bool
        )
        return negative, mask

    def _prepare_training_context(
        self, context: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dropped = torch.zeros(
            context.shape[0], device=context.device, dtype=torch.bool
        )
        if not self.training or self.cfg_drop_rate <= 0.0:
            return context, mask, dropped
        negative, negative_mask = self._negative_context(
            context.shape[0], context.device, context.dtype
        )
        length = max(context.shape[1], negative.shape[1])
        context, mask = self._pad_context(context, mask, length)
        negative, negative_mask = self._pad_context(negative, negative_mask, length)
        dropped = torch.rand(context.shape[0], device=context.device) < self.cfg_drop_rate
        context = torch.where(dropped[:, None, None], negative, context)
        mask = torch.where(dropped[:, None], negative_mask, mask)
        return context, mask, dropped

    def _ownership_loss(
        self,
        owner_probs: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        n_ovt_per_object: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, n_ovt, n_patch = gt_masks_per_ovt.shape
        n_per = max(int(n_ovt_per_object), 1)
        n_object = n_ovt // n_per
        if n_object <= 0:
            zero = owner_probs.new_zeros(())
            return zero, {
                "e7_owner_fg_acc": zero,
                "e7_owner_bg_acc": zero,
                "e7_owner_bg_iou": zero,
                "e7_owner_entropy": zero,
                "e7_owner_supervised_fg_fraction": zero,
                "e7_owner_ambiguous_fraction": zero,
                "e7_register_prob_on_fg": zero,
                "e7_object_prob_on_bg": zero,
            }

        object_probs = owner_probs[:, :n_ovt].reshape(
            batch_size, n_object, n_per, n_patch
        ).sum(dim=2)
        object_valid = ovt_valid_mask[:, : n_object * n_per].reshape(
            batch_size, n_object, n_per
        ).any(dim=2)
        object_gt = gt_masks_per_ovt[:, : n_object * n_per].reshape(
            batch_size, n_object, n_per, n_patch
        ).amax(dim=2)
        object_gt = object_gt * object_valid.unsqueeze(-1).to(object_gt.dtype)
        register_probs = owner_probs[:, n_ovt:].sum(dim=1)

        active = object_gt > 0.5
        active_count = active.sum(dim=1)
        fg_mask = active_count == 1
        bg_mask = object_gt.amax(dim=1) <= 1e-6
        ambiguous = ~(fg_mask | bg_mask)
        fg_target = active.float().argmax(dim=1)
        fg_prob = object_probs.gather(1, fg_target[:, None]).squeeze(1)
        eps = 1e-7
        fg_loss = -torch.log(fg_prob.clamp(min=eps))[fg_mask].mean() if fg_mask.any() else owner_probs.new_zeros(())
        bg_loss = -torch.log(register_probs.clamp(min=eps))[bg_mask].mean() if bg_mask.any() else owner_probs.new_zeros(())
        loss = fg_loss + self.owner_bg_weight * bg_loss

        predicted_owner = owner_probs.argmax(dim=1)
        predicted_object = predicted_owner < n_ovt
        predicted_group = torch.div(predicted_owner.clamp(max=max(n_ovt - 1, 0)), n_per, rounding_mode="floor")
        fg_acc = (
            (predicted_group[fg_mask] == fg_target[fg_mask]).float().mean()
            if fg_mask.any()
            else owner_probs.new_zeros(())
        )
        predicted_bg = ~predicted_object
        bg_acc = (
            predicted_bg[bg_mask].float().mean()
            if bg_mask.any()
            else owner_probs.new_zeros(())
        )
        supervised = fg_mask | bg_mask
        bg_intersection = (predicted_bg & bg_mask & supervised).sum().float()
        bg_union = ((predicted_bg | bg_mask) & supervised).sum().float()
        bg_iou = bg_intersection / bg_union.clamp(min=1.0)

        valid_owner_count = (
            ovt_valid_mask.float().sum(dim=1) + (owner_probs.shape[1] - n_ovt)
        ).clamp(min=2.0)
        entropy = -(owner_probs.clamp(min=eps) * torch.log(owner_probs.clamp(min=eps))).sum(dim=1)
        entropy = (entropy / torch.log(valid_owner_count)[:, None]).mean()
        register_on_fg = (
            register_probs[fg_mask].mean()
            if fg_mask.any()
            else owner_probs.new_zeros(())
        )
        object_on_bg = (
            object_probs.sum(dim=1)[bg_mask].mean()
            if bg_mask.any()
            else owner_probs.new_zeros(())
        )
        stats = {
            "e7_owner_fg_loss": fg_loss.detach(),
            "e7_owner_bg_loss": bg_loss.detach(),
            "e7_owner_fg_acc": fg_acc.detach(),
            "e7_owner_bg_acc": bg_acc.detach(),
            "e7_owner_bg_iou": bg_iou.detach(),
            "e7_owner_entropy": entropy.detach(),
            "e7_owner_supervised_fg_fraction": fg_mask.float().mean().detach(),
            "e7_owner_ambiguous_fraction": ambiguous.float().mean().detach(),
            "e7_register_prob_on_fg": register_on_fg.detach(),
            "e7_object_prob_on_bg": object_on_bg.detach(),
        }
        return loss, stats

    @staticmethod
    def _find_same_category_swaps(
        category_ids: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        eligible_samples: torch.Tensor,
    ) -> List[Tuple[int, int, int, int]]:
        matches: List[Tuple[int, int, int, int]] = []
        batch_size, n_ovt = category_ids.shape
        for batch_idx in range(batch_size):
            if not bool(eligible_samples[batch_idx]):
                continue
            anchors = [
                idx for idx in range(n_ovt)
                if bool(ovt_valid_mask[batch_idx, idx])
                and int(category_ids[batch_idx, idx]) >= 0
            ]
            if anchors:
                order = torch.randperm(len(anchors), device=category_ids.device).tolist()
                anchors = [anchors[idx] for idx in order]
            found = None
            for anchor_idx in anchors:
                category = int(category_ids[batch_idx, anchor_idx])
                for offset in range(1, batch_size):
                    source_batch = (batch_idx + offset) % batch_size
                    candidates = (
                        (category_ids[source_batch] == category)
                        & ovt_valid_mask[source_batch]
                    ).nonzero(as_tuple=False).flatten()
                    if candidates.numel() > 0:
                        source_idx = int(candidates[0])
                        found = (batch_idx, anchor_idx, source_batch, source_idx)
                        break
                if found is not None:
                    break
            if found is not None:
                matches.append(found)
        return matches

    @contextlib.contextmanager
    def _freeze_unet_adapters_for_counterfactual(self):
        parameters = [
            parameter
            for name, parameter in self.unet.named_parameters()
            if ".lora_down." in name or ".lora_up." in name
        ]
        states = [parameter.requires_grad for parameter in parameters]
        try:
            for parameter in parameters:
                parameter.requires_grad_(False)
            yield
        finally:
            for parameter, state in zip(parameters, states):
                parameter.requires_grad_(state)

    def diffusion_loss(
        self,
        images: torch.Tensor,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        gt_masks_per_ovt: torch.Tensor,
        ovt_category_ids: Optional[torch.Tensor],
        n_ovt_per_object: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        routed = self.route_and_write(
            ovt_hidden, register_hidden, image_hidden, ovt_valid_mask
        )
        final_ovt = routed["ovt_hidden"]
        final_register = routed["register_hidden"]
        owner_probs = routed["owner_probs"]
        owner_loss, owner_stats = self._ownership_loss(
            owner_probs,
            gt_masks_per_ovt,
            ovt_valid_mask,
            n_ovt_per_object=n_ovt_per_object,
        )

        context, context_mask = self._project_context(
            final_ovt, final_register, ovt_valid_mask
        )
        context, context_mask, dropped = self._prepare_training_context(
            context, context_mask
        )
        vae_dtype = next(self.vae.parameters()).dtype
        images = images.to(device=context.device, dtype=vae_dtype).clamp(-1.0, 1.0)
        with torch.no_grad():
            latent = self.vae.encode(images).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
        noise = torch.randn_like(latent)
        timesteps = torch.randint(
            0,
            self.scheduler_train.config.num_train_timesteps,
            (latent.shape[0],),
            device=latent.device,
            dtype=torch.long,
        )
        noisy_latent = self.scheduler_train.add_noise(latent, noise, timesteps)
        prediction_type = self.scheduler_train.config.prediction_type
        if prediction_type == "epsilon":
            target = noise
        elif prediction_type == "v_prediction":
            target = self.scheduler_train.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(f"Unsupported SD prediction_type={prediction_type}")
        prediction = self.unet(
            noisy_latent,
            timesteps,
            encoder_hidden_states=context,
            encoder_attention_mask=context_mask,
            return_dict=False,
        )[0]
        diffusion_loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")

        causal_loss = diffusion_loss.new_zeros(())
        invariance_loss = diffusion_loss.new_zeros(())
        causal_gap = diffusion_loss.new_zeros(())
        causal_pos_error = diffusion_loss.new_zeros(())
        causal_swap_error = diffusion_loss.new_zeros(())
        matches: List[Tuple[int, int, int, int]] = []
        if self.training and ovt_category_ids is not None:
            eligible = (timesteps >= self.causal_min_timestep) & ~dropped
            matches = self._find_same_category_swaps(
                ovt_category_ids.to(device=ovt_valid_mask.device, dtype=torch.long),
                ovt_valid_mask,
                eligible,
            )
        if matches:
            swapped_ovt = final_ovt.clone()
            anchor_batches = []
            anchor_masks = []
            for batch_idx, anchor_idx, source_batch, source_idx in matches:
                swapped_ovt[batch_idx, anchor_idx] = final_ovt[source_batch, source_idx]
                anchor_batches.append(batch_idx)
                anchor_masks.append(gt_masks_per_ovt[batch_idx, anchor_idx])
            anchor_index = torch.tensor(
                anchor_batches, device=latent.device, dtype=torch.long
            )
            swapped_context, swapped_mask = self._project_context(
                swapped_ovt, final_register, ovt_valid_mask
            )
            with self._freeze_unet_adapters_for_counterfactual():
                swapped_prediction = self.unet(
                    noisy_latent.index_select(0, anchor_index),
                    timesteps.index_select(0, anchor_index),
                    encoder_hidden_states=swapped_context.index_select(0, anchor_index),
                    encoder_attention_mask=swapped_mask.index_select(0, anchor_index),
                    return_dict=False,
                )[0]
            positive_prediction = prediction.index_select(0, anchor_index)
            target_subset = target.index_select(0, anchor_index)
            positive_error_map = (positive_prediction.float() - target_subset.float()).square().mean(dim=1)
            swapped_error_map = (swapped_prediction.float() - target_subset.float()).square().mean(dim=1)
            mask = torch.stack(anchor_masks).to(device=latent.device, dtype=torch.float32)
            side = int(math.isqrt(mask.shape[-1]))
            if side * side != mask.shape[-1]:
                raise ValueError(f"E7 causal mask is not square: P={mask.shape[-1]}")
            mask = F.interpolate(
                mask.reshape(mask.shape[0], 1, side, side),
                size=positive_error_map.shape[-2:],
                mode="nearest",
            ).squeeze(1).clamp(0.0, 1.0)
            inside_denom = mask.sum(dim=(-2, -1)).clamp(min=1.0)
            pos_per_sample = (positive_error_map * mask).sum(dim=(-2, -1)) / inside_denom
            swap_per_sample = (swapped_error_map * mask).sum(dim=(-2, -1)) / inside_denom
            causal_loss = F.softplus(
                (
                    self.causal_margin
                    + pos_per_sample
                    - swap_per_sample
                ) / max(self.causal_temperature, 1e-6)
            ).mean()
            outside = 1.0 - mask
            outside_denom = outside.sum(dim=(-2, -1)).clamp(min=1.0)
            change_map = (positive_prediction.float() - swapped_prediction.float()).square().mean(dim=1)
            invariance_loss = (
                (change_map * outside).sum(dim=(-2, -1)) / outside_denom
            ).mean()
            causal_pos_error = pos_per_sample.mean().detach()
            causal_swap_error = swap_per_sample.mean().detach()
            causal_gap = (swap_per_sample - pos_per_sample).mean().detach()

        details = {
            "loss_e7_diffusion": diffusion_loss.detach(),
            "loss_e7_owner": owner_loss.detach(),
            "loss_e7_causal": causal_loss.detach(),
            "loss_e7_invariance": invariance_loss.detach(),
            "e7_cfg_drop_fraction": dropped.float().mean().detach(),
            "e7_causal_active_fraction": diffusion_loss.new_tensor(
                float(len(matches)) / float(max(images.shape[0], 1))
            ),
            "e7_causal_pos_error": causal_pos_error,
            "e7_causal_swap_error": causal_swap_error,
            "e7_causal_error_gap": causal_gap,
        }
        details.update(owner_stats)
        return diffusion_loss, owner_loss, causal_loss, invariance_loss, details

    @torch.no_grad()
    def sample(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        image_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        resolution: int = 512,
        inference_steps: int = 10,
        guidance_scale: float = 2.5,
        seed: int = 1234,
    ) -> Dict[str, torch.Tensor]:
        routed = self.route_and_write(
            ovt_hidden, register_hidden, image_hidden, ovt_valid_mask
        )
        context, context_mask = self._project_context(
            routed["ovt_hidden"], routed["register_hidden"], ovt_valid_mask
        )
        batch_size = context.shape[0]
        do_cfg = float(guidance_scale) > 1.0
        if do_cfg:
            negative, negative_mask = self._negative_context(
                batch_size, context.device, context.dtype
            )
            length = max(context.shape[1], negative.shape[1])
            context, context_mask = self._pad_context(context, context_mask, length)
            negative, negative_mask = self._pad_context(
                negative, negative_mask, length
            )
            model_context = torch.cat([negative, context], dim=0)
            model_mask = torch.cat([negative_mask, context_mask], dim=0)
        else:
            model_context, model_mask = context, context_mask

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        latent = torch.randn(
            (
                batch_size,
                self.unet.config.in_channels,
                int(resolution) // 8,
                int(resolution) // 8,
            ),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(device=context.device, dtype=next(self.unet.parameters()).dtype)
        self.scheduler_test.set_timesteps(int(inference_steps), device=context.device)
        latent = latent * self.scheduler_test.init_noise_sigma
        for timestep in self.scheduler_test.timesteps:
            latent_input = torch.cat([latent, latent], dim=0) if do_cfg else latent
            latent_input = self.scheduler_test.scale_model_input(latent_input, timestep)
            noise_prediction = self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=model_context,
                encoder_attention_mask=model_mask,
                return_dict=False,
            )[0]
            if do_cfg:
                unconditional, conditional = noise_prediction.chunk(2)
                noise_prediction = unconditional + float(guidance_scale) * (
                    conditional - unconditional
                )
            latent = self.scheduler_test.step(
                noise_prediction, timestep, latent, return_dict=False
            )[0]
        decoded = self.vae.decode(
            latent / self.vae.config.scaling_factor, return_dict=False
        )[0]
        images = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        return {
            "images": images,
            "owner_probs": routed["owner_probs"],
            "context_mask": context_mask,
            "ovt_hidden": routed["ovt_hidden"],
            "register_hidden": routed["register_hidden"],
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save E7 heads and U-Net adapters, never frozen SD base weights."""
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        learned_prefixes = (
            f"{prefix}owner_write.",
            f"{prefix}context_norm.",
            f"{prefix}context_projector.",
            f"{prefix}context_out_norm.",
        )
        for key in list(state.keys()):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix):]
            keep = key.startswith(learned_prefixes) or (
                local.startswith("unet.")
                and (".lora_down." in local or ".lora_up." in local)
            )
            if not keep:
                del state[key]
        return state
