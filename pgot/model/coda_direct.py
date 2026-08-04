"""CODA Stable-Diffusion decoder conditioned directly on PGOT OVT/register states.

E6 intentionally removes Scale-RAE queries and the rectified-flow DiT from the
reconstruction path.  The only sample-specific conditioning accepted by the
decoder is the final Qwen hidden state of valid OVTs and background registers.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file


_CODA_TRAINABLE_PARTS = (".attn2.to_k.", ".attn2.to_v.", ".attn2.to_out.")


def _is_coda_cross_attention_parameter(name: str) -> bool:
    dotted = f".{name}"
    return any(part in dotted for part in _CODA_TRAINABLE_PARTS)


class _RecordingCrossAttentionProcessor:
    """Diffusers attention processor that records 32x32 cross-attention.

    At all other resolutions, or while recording is disabled, the original
    processor is called unchanged.  The recorded map is a true context-token
    ownership distribution: softmax is over OVT/register keys for every U-Net
    spatial query.
    """

    def __init__(self, base_processor, owner: "PGOTCODADirectBottleneck"):
        self.base_processor = base_processor
        self.owner = owner

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        query_length = (
            int(hidden_states.shape[-2] * hidden_states.shape[-1])
            if hidden_states.ndim == 4
            else int(hidden_states.shape[1])
        )
        if (
            not self.owner._record_attention
            or encoder_hidden_states is None
            or query_length != self.owner.attention_query_tokens
        ):
            return self.base_processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                temb=temb,
                *args,
                **kwargs,
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            batch_size = hidden_states.shape[0]

        key_length = encoder_hidden_states.shape[1]
        prepared_mask = None
        if attention_mask is not None:
            prepared_mask = attn.prepare_attention_mask(
                attention_mask, key_length, batch_size
            )
            prepared_mask = prepared_mask.view(
                batch_size, attn.heads, -1, prepared_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
        scores = scores / math.sqrt(float(head_dim))
        if prepared_mask is not None:
            scores = scores + prepared_mask.float()
        probs = torch.softmax(scores, dim=-1).to(value.dtype)

        # Keep only the conditional half when classifier-free guidance doubles B.
        record_probs = probs
        expected = self.owner._record_expected_batch
        if expected is not None and record_probs.shape[0] == 2 * expected:
            record_probs = record_probs[expected:]
        self.owner._attention_records.append(
            record_probs.float().mean(dim=1).transpose(1, 2).detach()
        )  # (B, context, 1024)

        hidden_states = torch.matmul(probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class PGOTCODADirectBottleneck(nn.Module):
    """Direct OVT/register bottleneck backed by CODA's SD-v1.5 decoder."""

    def __init__(
        self,
        llm_dim: int,
        model_id: str,
        coda_unet_checkpoint: str,
        context_dim: int = 768,
        cfg_drop_rate: float = 0.1,
        attention_grid_size: int = 32,
    ) -> None:
        super().__init__()
        self.model_id = str(model_id)
        self.coda_unet_checkpoint = str(coda_unet_checkpoint)
        self.context_dim = int(context_dim)
        self.cfg_drop_rate = float(cfg_drop_rate)
        self.attention_grid_size = int(attention_grid_size)
        self.attention_query_tokens = self.attention_grid_size ** 2

        self.context_norm = nn.LayerNorm(llm_dim)
        self.context_projector = nn.Linear(llm_dim, self.context_dim, bias=False)
        self.context_out_norm = nn.LayerNorm(self.context_dim)

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

        if self.coda_unet_checkpoint:
            if not os.path.isfile(self.coda_unet_checkpoint):
                raise FileNotFoundError(
                    f"CODA U-Net checkpoint not found: {self.coda_unet_checkpoint}"
                )
            coda_state = load_file(self.coda_unet_checkpoint, device="cpu")
            missing, unexpected = self.unet.load_state_dict(coda_state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected CODA U-Net keys ({len(unexpected)}): {unexpected[:5]}"
                )
            loaded = sum(1 for key in coda_state if _is_coda_cross_attention_parameter(key))
            if loaded != len(coda_state):
                raise RuntimeError(
                    "CODA checkpoint contains non-cross-attention parameters: "
                    f"loaded={loaded}, total={len(coda_state)}"
                )

        # Reuse CODA's exact CLIP empty-prompt embedding for CFG.  It is loaded
        # from the adjacent encoder checkpoint and is reproducible, so it need
        # not be duplicated in PGOT checkpoints.
        negative_prompt = self._load_coda_negative_prompt(coda_unet_checkpoint)
        self.register_buffer(
            "coda_negative_prompt", negative_prompt.float(), persistent=False
        )

        self._record_attention = False
        self._record_expected_batch: Optional[int] = None
        self._attention_records = []
        self._install_attention_recorders()

    @staticmethod
    def _load_coda_negative_prompt(coda_unet_checkpoint: str) -> torch.Tensor:
        encoder_path = os.path.join(
            os.path.dirname(os.path.dirname(coda_unet_checkpoint)),
            "encoder",
            "diffusion_pytorch_model.safetensors",
        )
        if not os.path.isfile(encoder_path):
            raise FileNotFoundError(
                "CODA negative-prompt embedding was not found next to the U-Net: "
                f"{encoder_path}"
            )
        encoder_state = load_file(encoder_path, device="cpu")
        if "null_embedding" not in encoder_state:
            raise KeyError(f"null_embedding is missing from {encoder_path}")
        return encoder_state["null_embedding"]

    def _install_attention_recorders(self) -> None:
        processors = {}
        for name, processor in self.unet.attn_processors.items():
            processors[name] = (
                _RecordingCrossAttentionProcessor(processor, self)
                if ".attn2.processor" in name
                else processor
            )
        self.unet.set_attn_processor(processors)

    def enable_trainable_parameters(self) -> Tuple[int, int]:
        """Enable only the OVT projector and CODA-selected U-Net x-attention."""
        projector_count = 0
        for module in (self.context_norm, self.context_projector, self.context_out_norm):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                projector_count += parameter.numel()
        unet_count = 0
        for name, parameter in self.unet.named_parameters():
            trainable = _is_coda_cross_attention_parameter(name)
            parameter.requires_grad_(trainable)
            if trainable:
                unet_count += parameter.numel()
        return projector_count, unet_count

    def train(self, mode: bool = True):
        super().train(mode)
        # CODA also keeps these modules in eval mode while allowing gradients
        # through the selected U-Net cross-attention projections.
        self.vae.eval()
        self.unet.eval()
        return self

    def _project_context(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.cat([ovt_hidden, register_hidden], dim=1)
        # Trainer's scalar eval runs under autocast, while media sampling is
        # intentionally outside it. Newly initialized E6 modules can therefore
        # remain fp32 even when Qwen hidden states are bf16.
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

    def _prepare_cfg_training_context(
        self, context: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = context.shape[0]
        dropped = torch.zeros(batch_size, device=context.device, dtype=torch.bool)
        if self.cfg_drop_rate <= 0.0:
            return context, mask, dropped
        negative = self.coda_negative_prompt.to(
            device=context.device, dtype=context.dtype
        ).expand(batch_size, -1, -1)
        negative_mask = torch.ones(
            negative.shape[:2], device=context.device, dtype=torch.bool
        )
        length = max(context.shape[1], negative.shape[1])
        context, mask = self._pad_context(context, mask, length)
        negative, negative_mask = self._pad_context(negative, negative_mask, length)
        if self.training:
            dropped = torch.rand(batch_size, device=context.device) < self.cfg_drop_rate
            context = torch.where(dropped[:, None, None], negative, context)
            mask = torch.where(dropped[:, None], negative_mask, mask)
        return context, mask, dropped

    def diffusion_loss(
        self,
        images: torch.Tensor,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context, context_mask = self._project_context(
            ovt_hidden, register_hidden, ovt_valid_mask
        )
        context, context_mask, dropped = self._prepare_cfg_training_context(
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
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")

        m = ovt_hidden.shape[1]
        projected_unpadded, _ = self._project_context(
            ovt_hidden, register_hidden, ovt_valid_mask
        )
        valid_float = ovt_valid_mask.unsqueeze(-1).to(projected_unpadded.dtype)
        ovt_norm = (
            projected_unpadded[:, :m].float().norm(dim=-1) * valid_float.squeeze(-1)
        ).sum() / valid_float.sum().clamp(min=1.0)
        if register_hidden.shape[1] > 0:
            bg_norm = projected_unpadded[:, m:].float().norm(dim=-1).mean()
        else:
            bg_norm = loss.new_zeros(())
        details = {
            "loss_e6_diffusion": loss.detach(),
            "e6_ovt_context_norm": ovt_norm.detach(),
            "e6_bg_context_norm": bg_norm.detach(),
            "e6_cfg_drop_fraction": dropped.float().mean().detach(),
        }
        return loss, details

    @torch.no_grad()
    def sample(
        self,
        ovt_hidden: torch.Tensor,
        register_hidden: torch.Tensor,
        ovt_valid_mask: torch.Tensor,
        resolution: int = 512,
        inference_steps: int = 10,
        guidance_scale: float = 2.5,
        seed: int = 1234,
        record_attention: bool = True,
    ) -> Dict[str, Optional[torch.Tensor]]:
        context, context_mask = self._project_context(
            ovt_hidden, register_hidden, ovt_valid_mask
        )
        direct_context_length = context.shape[1]
        batch_size = context.shape[0]
        do_cfg = float(guidance_scale) > 1.0
        if do_cfg:
            negative = self.coda_negative_prompt.to(
                device=context.device, dtype=context.dtype
            ).expand(batch_size, -1, -1)
            negative_mask = torch.ones(
                negative.shape[:2], device=context.device, dtype=torch.bool
            )
            length = max(context.shape[1], negative.shape[1])
            context, context_mask = self._pad_context(context, context_mask, length)
            negative, negative_mask = self._pad_context(negative, negative_mask, length)
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
        self._attention_records = []
        self._record_attention = bool(record_attention)
        self._record_expected_batch = batch_size
        try:
            for timestep in self.scheduler_test.timesteps:
                latent_input = torch.cat([latent, latent], dim=0) if do_cfg else latent
                latent_input = self.scheduler_test.scale_model_input(
                    latent_input, timestep
                )
                noise_prediction = self.unet(
                    latent_input,
                    timestep,
                    encoder_hidden_states=model_context,
                    encoder_attention_mask=model_mask,
                    return_dict=False,
                )[0]
                if do_cfg:
                    uncond, cond = noise_prediction.chunk(2)
                    noise_prediction = uncond + float(guidance_scale) * (cond - uncond)
                latent = self.scheduler_test.step(
                    noise_prediction, timestep, latent, return_dict=False
                )[0]
        finally:
            self._record_attention = False
            self._record_expected_batch = None

        decoded = self.vae.decode(
            latent / self.vae.config.scaling_factor, return_dict=False
        )[0]
        images = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)

        attention_maps = None
        if self._attention_records:
            attention_maps = torch.stack(self._attention_records, dim=0).mean(dim=0)
            # CFG pads the context to 77; only OVT/register tokens are relevant.
            attention_maps = attention_maps[:, :direct_context_length]
        self._attention_records = []
        return {
            "images": images,
            "attention_maps": attention_maps,
            "context_mask": context_mask,
        }

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Save projector + trainable CODA x-attention, not frozen SD weights."""
        state = super().state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        keep_prefixes = (
            f"{prefix}context_norm.",
            f"{prefix}context_projector.",
            f"{prefix}context_out_norm.",
        )
        for key in list(state.keys()):
            if not key.startswith(prefix):
                continue
            local = key[len(prefix):]
            keep = key.startswith(keep_prefixes) or (
                local.startswith("unet.")
                and _is_coda_cross_attention_parameter(local[len("unet."):])
            )
            if not keep:
                del state[key]
        return state
