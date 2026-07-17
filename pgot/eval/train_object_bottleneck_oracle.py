"""Trainable reconstruction ceilings for PGOT object bottlenecks.

Unlike ``eval_recon_oracles.py``, this experiment trains both a condition
adapter and a controlled subset of the DiT.  The three sources isolate the
point at which visual information is lost:

``c_full``
    Full decoder-native SigLIP patch features.
``c_gtobj``
    GT-mask-restricted object/background resampler tokens.
``c_ovt``
    Raw final OVT/void states from the PGOT checkpoint.

Each invocation trains exactly one source so its DiT adaptation is independent
from the other ceilings.
"""

import argparse
import json
import logging
import math
import os
import random
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from pgot.eval.eval_recon_oracles import (
    build_loader,
    infer_from_z,
    load_model_and_tokenizer,
)
from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.object_bottleneck_oracle")

ORACLES = {"c_current", "c_full", "c_gtobj", "c_ovt"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resize_object_masks(
    masks_per_ovt: torch.Tensor,
    valid_per_ovt: torch.Tensor,
    *,
    n_ovt_per_object: int,
    target_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge repeated OVT masks and resize them to the target feature grid."""
    masks = masks_per_ovt.float().clamp(0.0, 1.0)
    valid = valid_per_ovt.bool()
    batch, total_ovt, source_tokens = masks.shape
    slots_per_object = max(int(n_ovt_per_object), 1)
    objects = total_ovt // slots_per_object
    masks = masks[:, : objects * slots_per_object].reshape(
        batch, objects, slots_per_object, source_tokens
    ).amax(dim=2)
    valid = valid[:, : objects * slots_per_object].reshape(
        batch, objects, slots_per_object
    ).any(dim=2)

    if source_tokens != target_tokens:
        source_side = int(round(math.sqrt(source_tokens)))
        target_side = int(round(math.sqrt(target_tokens)))
        if source_side * source_side != source_tokens or target_side * target_side != target_tokens:
            raise ValueError(
                f"Mask resize requires square grids, got {source_tokens} -> {target_tokens}."
            )
        masks = F.interpolate(
            masks.reshape(batch * objects, 1, source_side, source_side),
            size=(target_side, target_side),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, objects, target_tokens)
    masks = masks * valid.unsqueeze(-1).float()
    return masks, valid


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.source_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        source_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        key_padding_mask = None if source_valid is None else ~source_valid.bool()
        update, _ = self.attn(
            self.query_norm(query),
            self.source_norm(source),
            self.source_norm(source),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + update
        return query + self.mlp(self.mlp_norm(query))


class ConditionResampler(nn.Module):
    def __init__(
        self,
        source_dim: int,
        condition_dim: int,
        condition_tokens: int,
        heads: int,
        depth: int,
    ):
        super().__init__()
        self.source_proj = nn.Linear(source_dim, condition_dim)
        self.queries = nn.Parameter(
            torch.randn(condition_tokens, condition_dim) / math.sqrt(condition_dim)
        )
        self.blocks = nn.ModuleList(
            [CrossAttentionBlock(condition_dim, heads) for _ in range(max(int(depth), 1))]
        )
        self.output_norm = nn.LayerNorm(condition_dim)

    def forward(
        self,
        source: torch.Tensor,
        source_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        source = self.source_proj(source.float())
        query = self.queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        for block in self.blocks:
            query = block(query, source, source_valid)
        return self.output_norm(query)


class GTObjectResampler(nn.Module):
    """Create K appearance tokens per GT object plus K background tokens."""

    def __init__(self, feature_dim: int, hidden_dim: int, tokens_per_region: int, patch_tokens: int):
        super().__init__()
        self.tokens_per_region = int(tokens_per_region)
        self.patch_tokens = int(patch_tokens)
        self.patch_proj = nn.Linear(feature_dim, hidden_dim)
        self.patch_position = nn.Parameter(
            torch.randn(patch_tokens, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.region_queries = nn.Parameter(
            torch.randn(self.tokens_per_region, hidden_dim) / math.sqrt(hidden_dim)
        )
        self.key_norm = nn.LayerNorm(hidden_dim)
        self.value_norm = nn.LayerNorm(hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        patches: torch.Tensor,
        object_masks: torch.Tensor,
        object_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, patch_tokens, _ = patches.shape
        if patch_tokens != self.patch_tokens:
            raise ValueError(f"Expected {self.patch_tokens} patch tokens, got {patch_tokens}.")

        union = object_masks.amax(dim=1).clamp(0.0, 1.0)
        background_mask = (1.0 - union).clamp(0.0, 1.0)
        region_masks = torch.cat([object_masks, background_mask.unsqueeze(1)], dim=1)
        background_valid = background_mask.sum(dim=-1) > 1e-3
        region_valid = torch.cat([object_valid, background_valid.unsqueeze(1)], dim=1)

        patch_hidden = self.patch_proj(patches.float()) + self.patch_position.unsqueeze(0)
        keys = self.key_norm(patch_hidden)
        values = self.value_norm(patch_hidden)
        queries = self.region_queries
        logits = torch.einsum("kd,bpd->bkp", queries, keys) / math.sqrt(keys.shape[-1])
        logits = logits.unsqueeze(1) + torch.log(region_masks.clamp_min(1e-6)).unsqueeze(2)
        logits = logits.masked_fill(region_masks.unsqueeze(2) <= 1e-6, -1e4)
        weights = F.softmax(logits, dim=-1)
        tokens = torch.einsum("brkp,bpd->brkd", weights, values)
        tokens = self.output_norm(tokens + queries.view(1, 1, self.tokens_per_region, -1))
        tokens = tokens * region_valid.unsqueeze(-1).unsqueeze(-1).to(tokens.dtype)
        token_valid = region_valid.unsqueeze(-1).expand(-1, -1, self.tokens_per_region)
        return tokens.flatten(1, 2), token_valid.flatten(1, 2)


class OracleConditioner(nn.Module):
    def __init__(
        self,
        *,
        oracle: str,
        source_dim: int,
        condition_dim: int,
        condition_tokens: int,
        patch_tokens: int,
        object_tokens: int,
        heads: int,
        depth: int,
    ):
        super().__init__()
        self.oracle = oracle
        self.object_resampler = None
        resampler_source_dim = source_dim
        if oracle == "c_gtobj":
            self.object_resampler = GTObjectResampler(
                feature_dim=source_dim,
                hidden_dim=condition_dim,
                tokens_per_region=object_tokens,
                patch_tokens=patch_tokens,
            )
            resampler_source_dim = condition_dim
        self.condition_resampler = ConditionResampler(
            source_dim=resampler_source_dim,
            condition_dim=condition_dim,
            condition_tokens=condition_tokens,
            heads=heads,
            depth=depth,
        )

    def forward(
        self,
        source: torch.Tensor,
        source_valid: Optional[torch.Tensor] = None,
        object_masks: Optional[torch.Tensor] = None,
        object_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.object_resampler is not None:
            if object_masks is None or object_valid is None:
                raise ValueError("c_gtobj requires object masks and validity.")
            source, source_valid = self.object_resampler(source, object_masks, object_valid)
        return self.condition_resampler(source, source_valid)


@torch.no_grad()
def extract_current_pack(model, batch: Dict[str, torch.Tensor], device: torch.device):
    """Extract the frozen PGOT condition and all oracle source features once."""
    if not bool(getattr(model.config, "pgot_v14_enable", False)):
        raise ValueError("Object bottleneck oracles require a V14-style checkpoint.")
    router_dtype = next(model.model.parameters()).dtype
    if next(model.pgot_v14_router.parameters()).dtype != router_dtype:
        model.pgot_v14_router.to(device=device, dtype=router_dtype)
    features = model._pgot_v14_forward_features(
        images=batch["images"].to(device),
        target_images=batch["target_images"].to(device),
        caption_input_ids=batch["caption_input_ids"].to(device),
        caption_attention_mask=batch["caption_attention_mask"].to(device),
        ovt_positions_in_caption=batch["ovt_positions_in_caption"].to(device),
        ovt_valid_mask=batch["ovt_valid_mask"].to(device),
        output_hidden_states=False,
    )
    teacher = model._captionslot_prepare_diffusion_condition(
        features["condition_hidden"]
    ).float()
    return {
        "teacher_condition": teacher.detach(),
        "target": features["gt_siglip"].float().detach(),
        "ovt_states": features["ovt_states"].float().detach(),
        "ovt_valid": features["ovt_valid"].bool().detach(),
    }


@torch.no_grad()
def extract_frozen_source(args, model, batch, device, pack=None):
    pack = extract_current_pack(model, batch, device) if pack is None else pack
    target = pack["target"]
    if args.oracle == "c_ovt":
        return (
            pack["ovt_states"],
            pack["ovt_valid"],
            target,
            None,
            None,
            pack["teacher_condition"],
        )

    if args.oracle == "c_full":
        valid = torch.ones(target.shape[:2], device=device, dtype=torch.bool)
        return target, valid, target, None, None, pack["teacher_condition"]

    if args.oracle == "c_current":
        return None, None, target, None, None, pack["teacher_condition"]

    masks, valid_objects = resize_object_masks(
        batch["gt_masks_per_ovt"].to(device),
        batch["ovt_valid_mask"].to(device),
        n_ovt_per_object=args.n_ovt_per_object,
        target_tokens=target.shape[1],
    )
    return target, None, target, masks, valid_objects, pack["teacher_condition"]


def configure_trainable_dit(model, last_n_blocks: int) -> Dict[str, nn.Parameter]:
    model.requires_grad_(False)
    trainable = {}
    if int(last_n_blocks) <= 0:
        log.info("Trainable DiT: none (frozen pretrained DiT)")
        return trainable

    for name, param in model.diff_head.named_parameters():
        if "adaLN_modulation" in name:
            param.requires_grad_(True)
            trainable[name] = param

    blocks = model.diff_head.model.dit_blocks
    start = max(0, len(blocks) - max(int(last_n_blocks), 0))
    for index in range(start, len(blocks)):
        for suffix, param in blocks[index].named_parameters():
            param.requires_grad_(True)
            trainable[f"model.dit_blocks.{index}.{suffix}"] = param
    log.info(
        "Trainable DiT: all AdaLN + blocks %d..%d (%d tensors, %.2fM parameters)",
        start,
        len(blocks) - 1,
        len(trainable),
        sum(p.numel() for p in trainable.values()) / 1e6,
    )
    return trainable


def build_conditioner(args, model, first_batch, device) -> OracleConditioner:
    source, _, target, _, _, teacher = extract_frozen_source(
        args, model, first_batch, device
    )
    condition_dim = int(teacher.shape[-1])
    condition_tokens = int(teacher.shape[1])
    conditioner = OracleConditioner(
        oracle=args.oracle,
        source_dim=int(source.shape[-1]),
        condition_dim=condition_dim,
        condition_tokens=condition_tokens,
        patch_tokens=int(target.shape[1]),
        object_tokens=args.object_tokens,
        heads=args.adapter_heads,
        depth=args.adapter_depth,
    ).to(device=device, dtype=torch.float32)
    log.info(
        "Conditioner: oracle=%s source=%s -> condition=(%d,%d), %.2fM parameters",
        args.oracle,
        tuple(source.shape),
        condition_tokens,
        condition_dim,
        sum(p.numel() for p in conditioner.parameters()) / 1e6,
    )
    return conditioner


def save_checkpoint(
    args,
    conditioner,
    model,
    trainable_dit,
    losses,
    distill_losses,
    diffusion_losses,
) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "oracle_checkpoint.pt"
    torch.save(
        {
            "oracle": args.oracle,
            "object_tokens": int(args.object_tokens),
            "train_steps": int(args.train_steps),
            "conditioner": {k: v.detach().cpu() for k, v in conditioner.state_dict().items()},
            "dit": {k: p.detach().cpu() for k, p in trainable_dit.items()},
            "last_loss": losses[-1] if losses else None,
            "mean_last_20": sum(losses[-20:]) / max(len(losses[-20:]), 1),
            "last_distill_loss": distill_losses[-1] if distill_losses else None,
            "last_diffusion_loss": diffusion_losses[-1] if diffusion_losses else None,
        },
        path,
    )
    return path


def train_oracle(args, model, conditioner, trainable_dit, train_loader, device):
    conditioner.train()
    model.diff_head.train(bool(trainable_dit))
    adapter_params = list(conditioner.parameters())
    parameter_groups = [{"params": adapter_params, "lr": args.adapter_lr}]
    base_lrs = [args.adapter_lr]
    if trainable_dit:
        parameter_groups.append({"params": list(trainable_dit.values()), "lr": args.dit_lr})
        base_lrs.append(args.dit_lr)
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    warmup = max(int(args.warmup_steps), 0)

    def lr_scale(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(float(step + 1) / float(warmup), 1.0)

    loader = cycle(train_loader)
    losses = []
    distill_losses = []
    diffusion_losses = []
    accumulation = max(int(args.gradient_accumulation_steps), 1)
    progress = tqdm(range(args.train_steps), desc=f"Train {args.oracle}")
    for optimizer_step in progress:
        optimizer.zero_grad(set_to_none=True)
        step_total = 0.0
        step_distill = 0.0
        step_diffusion = 0.0
        use_diffusion = (
            optimizer_step >= int(args.distill_steps)
            and float(args.diffusion_loss_weight) > 0.0
        )
        for _ in range(accumulation):
            batch = next(loader)
            source, source_valid, target, masks, object_valid, teacher = extract_frozen_source(
                args, model, batch, device
            )
            condition = conditioner(source, source_valid, masks, object_valid).float()
            distill_loss = F.mse_loss(condition, teacher.float())
            diffusion_loss = condition.new_zeros(())
            if use_diffusion:
                condition_for_diff = condition
                if args.cfg_drop_rate > 0.0:
                    drop_mask = (
                        torch.rand(condition.shape[0], device=device) < args.cfg_drop_rate
                    ).view(-1, 1, 1)
                    condition_for_diff = condition * (~drop_mask).to(condition.dtype)
                diffusion_loss = model.diff_head.training_loss(
                    z=condition_for_diff,
                    x=target.float(),
                ).mean()
            loss = (
                float(args.distill_loss_weight) * distill_loss
                + float(args.diffusion_loss_weight) * diffusion_loss
            )
            (loss / accumulation).backward()
            step_total += float(loss.detach().cpu()) / accumulation
            step_distill += float(distill_loss.detach().cpu()) / accumulation
            step_diffusion += float(diffusion_loss.detach().cpu()) / accumulation

        torch.nn.utils.clip_grad_norm_(
            adapter_params + list(trainable_dit.values()), args.max_grad_norm
        )
        scale = lr_scale(optimizer_step)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group["lr"] = base_lr * scale
        optimizer.step()

        losses.append(step_total)
        distill_losses.append(step_distill)
        diffusion_losses.append(step_diffusion)
        if (
            optimizer_step % max(int(args.log_every), 1) == 0
            or optimizer_step + 1 == args.train_steps
        ):
            phase = "distill" if not use_diffusion else "joint"
            progress.set_postfix(
                phase=phase,
                loss=f"{step_total:.4f}",
                distill=f"{step_distill:.4f}",
                diffusion=f"{step_diffusion:.4f}",
            )
    return losses, distill_losses, diffusion_losses


@torch.no_grad()
def evaluate_oracle(args, model, conditioner, val_loader, device):
    if conditioner is not None:
        conditioner.eval()
    model.diff_head.eval()
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    fid = None if args.skip_fid else FIDAccumulator(device=device, feature=2048)
    metrics = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    samples = 0

    towers = model.get_vision_tower_aux_list()
    target_processor = towers[1].image_processor if len(towers) > 1 else towers[0].image_processor
    target_mean = torch.tensor(target_processor.image_mean)
    target_std = torch.tensor(target_processor.image_std)

    for batch in tqdm(val_loader, desc=f"Eval {args.oracle}"):
        source, source_valid, _, masks, object_valid, teacher = extract_frozen_source(
            args, model, batch, device
        )
        condition = (
            teacher
            if args.oracle == "c_current"
            else conditioner(source, source_valid, masks, object_valid)
        )
        generated = infer_from_z(model, condition, args.guidance_scale)
        fake = decode_to_image(decoder, generated, device)
        real = denormalize_images(batch["target_images"].to(device).float(), target_mean, target_std)
        if real.shape[-2:] != fake.shape[-2:]:
            real = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
        if fid is not None:
            fid.add(real, fake)
        batch_metrics = compute_recon_metrics(real, fake)
        for key in metrics:
            metrics[key].extend(float(x) for x in batch_metrics[key].detach().cpu())
        samples += int(real.shape[0])

    summary = {
        "oracle": args.oracle,
        "object_tokens": int(args.object_tokens) if args.oracle == "c_gtobj" else None,
        "num_samples": samples,
        "rFID": None if fid is None else float(fid.compute()),
        "guidance_scale": float(args.guidance_scale),
        "diffusion_inference_steps": int(args.diffusion_inference_steps),
        "train_steps": int(args.train_steps),
        "distill_steps": int(args.distill_steps),
        "distill_loss_weight": float(args.distill_loss_weight),
        "diffusion_loss_weight": float(args.diffusion_loss_weight),
        "cfg_drop_rate": float(args.cfg_drop_rate),
        "dit_last_n_blocks": int(args.dit_last_n_blocks),
    }
    for key, values in metrics.items():
        summary[f"recon_{key}"] = sum(values) / max(len(values), 1)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", default="/home/jovyan/PGOT/data/pgot_train.jsonl")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--oracle", choices=sorted(ORACLES), required=True)
    parser.add_argument("--object_tokens", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--adapter_lr", type=float, default=1e-4)
    parser.add_argument("--dit_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dit_last_n_blocks", type=int, default=0)
    parser.add_argument("--distill_steps", type=int, default=2000)
    parser.add_argument("--distill_loss_weight", type=float, default=1.0)
    parser.add_argument("--diffusion_loss_weight", type=float, default=0.1)
    parser.add_argument("--cfg_drop_rate", type=float, default=0.1)
    parser.add_argument("--adapter_heads", type=int, default=16)
    parser.add_argument("--adapter_depth", type=int, default=2)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--no_save_checkpoint", action="store_true")
    args = parser.parse_args()
    if args.object_tokens <= 0:
        parser.error("--object_tokens must be positive")
    if args.oracle == "c_current" and args.train_steps != 0:
        parser.error("c_current is a frozen control and requires --train_steps 0")
    if args.oracle != "c_current" and not 0 <= args.distill_steps <= args.train_steps:
        parser.error("--distill_steps must be between 0 and --train_steps")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model, tokenizer, device, _ = load_model_and_tokenizer(args)
    train_loader = None
    if args.oracle != "c_current":
        train_loader = build_loader(
            args,
            tokenizer,
            model,
            args.train_jsonl,
            shuffle=True,
            max_samples=args.max_train_samples,
        )
    val_loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )

    conditioner = None
    trainable_dit = {}
    losses = []
    distill_losses = []
    diffusion_losses = []
    if args.oracle != "c_current":
        first_batch = next(iter(train_loader))
        conditioner = build_conditioner(args, model, first_batch, device)
        trainable_dit = configure_trainable_dit(model, args.dit_last_n_blocks)
        losses, distill_losses, diffusion_losses = train_oracle(
            args, model, conditioner, trainable_dit, train_loader, device
        )

    checkpoint_path = None
    if conditioner is not None and not args.no_save_checkpoint:
        checkpoint_path = save_checkpoint(
            args,
            conditioner,
            model,
            trainable_dit,
            losses,
            distill_losses,
            diffusion_losses,
        )
        log.info("Saved %s", checkpoint_path)

    summary = evaluate_oracle(args, model, conditioner, val_loader, device)
    summary.update(
        {
            "model_path": args.model_path,
            "adapter_lr": float(args.adapter_lr),
            "dit_lr": float(args.dit_lr),
            "last_train_loss": losses[-1] if losses else None,
            "mean_last_20_train_loss": sum(losses[-20:]) / max(len(losses[-20:]), 1),
            "last_distill_loss": distill_losses[-1] if distill_losses else None,
            "last_diffusion_loss": diffusion_losses[-1] if diffusion_losses else None,
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        }
    )
    with open(Path(args.output_dir) / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    log.info("RESULT %s", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
