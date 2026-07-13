"""Diagnostics for the V14 OVT bottleneck router.

This script checks two questions:
  1. Does the V14 route/void assignment generalize on validation samples?
  2. Does reconstruction actually depend on object OVT / void content once the
     diffusion condition is forced through the V14 bottleneck?

It intentionally evaluates the bottleneck path itself instead of the legacy
RAE-query hidden, because V14 reconstruction uses router-produced
``condition_hidden``.
"""

import argparse
import gc
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.pgot_inference import generate_siglip_latent
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.eval.visualize_ovt_overlays import (
    _concat_grid,
    _heat_overlay,
    _large_mask_overlay,
    _load_model,
    _segment_labels,
    _source_from_target_tensor,
)
from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset

log = logging.getLogger("pgot.v14_bottleneck_diag")


def parse_model(spec: str) -> tuple[str, str]:
    label, path = spec.split("|", 1)
    return label, path


def fixed_recon_loss(model, condition_hidden, target, seed: int):
    devices = [condition_hidden.device.index] if condition_hidden.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return model._captionslot_compute_diffusion_loss(
            hidden=condition_hidden,
            target_features=target,
        )


def _object_cover_from_batch(batch, device, n_ovt_per_object: int):
    gt_masks = batch["gt_masks_per_ovt"].to(device).float()
    ovt_valid = batch["ovt_valid_mask"].to(device, dtype=torch.bool)
    B, M, P = gt_masks.shape
    n = max(int(n_ovt_per_object), 1)
    K = M // n
    if K <= 0:
        return gt_masks.new_zeros(B, 0, P), ovt_valid.new_zeros(B, 0)
    cover = gt_masks[:, : K * n].reshape(B, K, n, P).amax(dim=2).clamp(0.0, 1.0)
    obj_valid = ovt_valid[:, : K * n].reshape(B, K, n).any(dim=2)
    cover = cover.masked_fill(~obj_valid.unsqueeze(-1), 0.0)
    return cover, obj_valid


def _resize_cover_to_p(cover: torch.Tensor, target_p: int) -> torch.Tensor:
    if cover.shape[-1] == target_p:
        return cover
    src_side = int(round(float(cover.shape[-1]) ** 0.5))
    dst_side = int(round(float(target_p) ** 0.5))
    if src_side * src_side != cover.shape[-1] or dst_side * dst_side != target_p:
        raise ValueError(f"expected square masks, got cover={cover.shape[-1]} target={target_p}")
    B, K, _ = cover.shape
    return F.interpolate(
        cover.reshape(B * K, 1, src_side, src_side),
        size=(dst_side, dst_side),
        mode="area",
    ).reshape(B, K, target_p).clamp(0.0, 1.0)


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / den.clamp_min(1e-6)


def _up_2d(x: torch.Tensor, size: int, grid: int):
    return F.interpolate(
        x.reshape(1, -1, grid, grid).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0]


def _infer_square_grid(n_patches: int, *, name: str) -> int:
    grid = int(round(float(n_patches) ** 0.5))
    if grid * grid != int(n_patches):
        raise ValueError(f"{name} patch count must be square, got {n_patches}")
    return grid


def v14_forward_variant(
    model,
    batch,
    *,
    zero_input_object: bool = False,
    zero_input_void: bool = False,
    zero_final_object: bool = False,
    zero_final_void: bool = False,
    swap_object_source: torch.Tensor | None = None,
    swap_target_object: int = 0,
    swap_source_object: int = 0,
    require_ovt_grad: bool = False,
):
    """Run V14 with optional intervention at input or bottleneck states."""
    device = model._pgot_model_device()
    dtype = model._aurora_model_dtype()
    seq = model._pgot_build_sequence_inputs(
        images=batch["images"].to(device),
        target_images=batch["target_images"].to(device),
        caption_input_ids=batch["caption_input_ids"].to(device),
        caption_attention_mask=batch["caption_attention_mask"].to(device, dtype=torch.bool),
        ovt_positions_in_caption=batch["ovt_positions_in_caption"].to(device),
        ovt_valid_mask=batch["ovt_valid_mask"].to(device, dtype=torch.bool),
    )
    inputs = seq["inputs_embeds"]
    if zero_input_object or zero_input_void:
        inputs = inputs.clone()
        if zero_input_object:
            for b_idx in range(seq["ovt_abs_positions"].shape[0]):
                valid_pos = seq["ovt_abs_positions"][b_idx][seq["ovt_valid_mask"][b_idx]]
                if valid_pos.numel() > 0:
                    inputs[b_idx, valid_pos] = 0
        if zero_input_void:
            s = int(seq["positions"].get("null_bg_s", 0))
            e = int(seq["positions"].get("null_bg_e", s))
            if e > s:
                inputs[:, s:e] = 0

    out = model.model(
        inputs_embeds=inputs.to(dtype=dtype),
        attention_bias=seq["attn_bias"],
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden = out.last_hidden_state
    ovt_pack = model._pgot_v12_build_ovt_states(
        hidden_states=hidden,
        positions=seq["positions"],
        ovt_abs_positions=seq["ovt_abs_positions"],
        ovt_valid_mask=seq["ovt_valid_mask"],
    )
    object_ovts = ovt_pack["object_ovts"]
    void_ovts = ovt_pack["void_ovts"]
    if require_ovt_grad:
        object_ovts = object_ovts.detach().clone().requires_grad_(True)
        void_ovts = void_ovts.detach().clone().requires_grad_(True)
    else:
        object_ovts = object_ovts.clone()
        void_ovts = void_ovts.clone()

    if zero_final_object:
        object_ovts = torch.zeros_like(object_ovts)
    if zero_final_void:
        void_ovts = torch.zeros_like(void_ovts)
    if swap_object_source is not None and object_ovts.shape[1] > 0:
        tgt = max(0, min(int(swap_target_object), object_ovts.shape[1] - 1))
        src = max(0, min(int(swap_source_object), swap_object_source.shape[1] - 1))
        object_ovts = object_ovts.clone()
        object_ovts[:, tgt] = swap_object_source[:, src].to(object_ovts.device, dtype=object_ovts.dtype)

    ovt_states = torch.cat([object_ovts, void_ovts], dim=1)
    ovt_valid = torch.cat([ovt_pack["object_valid"], ovt_pack["void_valid"]], dim=1)
    query_base = model.get_model().latent_queries.unsqueeze(0).expand(
        hidden.shape[0], -1, -1
    ).to(device=hidden.device, dtype=hidden.dtype)
    route = model.pgot_v14_router(
        query_base=query_base,
        ovt_states=ovt_states,
        ovt_valid_mask=ovt_valid,
        temperature=float(getattr(model.config, "pgot_v14_route_temperature", 1.0)),
        position_weight=float(getattr(model.config, "pgot_v14_position_weight", 1.0)),
    )
    K = object_ovts.shape[1]
    return {
        **seq,
        "hidden": hidden,
        "hidden_states": out.hidden_states,
        "img_hidden": hidden[:, seq["positions"]["img_s"]:seq["positions"]["img_e"], :],
        "rae_hidden": hidden[:, seq["positions"]["rae_s"]:seq["positions"]["rae_e"], :],
        "condition_hidden": route["condition_hidden"],
        "owner_logits": route["owner_logits"],
        "owner_probs": route["owner_probs"],
        "object_probs": route["owner_probs"][:, :K],
        "void_probs": route["owner_probs"][:, K:],
        "object_valid": ovt_pack["object_valid"],
        "ovt_valid": ovt_valid,
        "object_ovts": object_ovts,
        "void_ovts": void_ovts,
    }


def route_stats(model, out, batch, n_ovt_per_object: int):
    device = out["condition_hidden"].device
    loss = model._pgot_v14_compute_route_loss(
        out["owner_logits"],
        out["owner_probs"],
        batch["gt_masks_per_ovt"].to(device).float(),
        batch["ovt_valid_mask"].to(device, dtype=torch.bool),
    )
    cover, obj_valid = _object_cover_from_batch(batch, device, n_ovt_per_object)
    obj_probs = out["object_probs"].float()
    K = min(cover.shape[1], obj_probs.shape[1])
    if K <= 0:
        per_object = []
        fg_union = obj_probs.new_zeros(obj_probs.shape[0], obj_probs.shape[-1])
    else:
        cover = _resize_cover_to_p(cover[:, :K], obj_probs.shape[-1])
        obj_probs = obj_probs[:, :K]
        obj_valid = obj_valid[:, :K]
        fg_union = cover.amax(dim=1)
        per_object = []
        for k in range(K):
            mask = cover[:, k]
            prob = obj_probs[:, k]
            area = mask.sum(dim=-1)
            outside = (1.0 - mask).clamp(0.0, 1.0)
            prob_mass = prob.sum(dim=-1)
            inside_mass = (prob * mask).sum(dim=-1)
            outside_mass = (prob * outside).sum(dim=-1)
            per_object.append({
                "index": int(k),
                "valid": bool(obj_valid[0, k].detach().cpu()),
                "gt_area_frac": float((area / mask.shape[-1]).mean().detach()),
                "prob_sum": float(prob_mass.mean().detach()),
                "inside_prob_mean": float(_safe_ratio(inside_mass, area).mean().detach()),
                "outside_prob_mean": float(_safe_ratio(outside_mass, outside.sum(dim=-1)).mean().detach()),
                "inside_fraction_of_object_prob": float(_safe_ratio(inside_mass, prob_mass).mean().detach()),
            })
    void_prob = out["void_probs"].float().sum(dim=1) if out["void_probs"].shape[1] > 0 else torch.zeros_like(fg_union)
    bg = (1.0 - fg_union).clamp(0.0, 1.0)
    extra = {
        "void_prob_on_gt_bg_mean": float(_safe_ratio((void_prob * bg).sum(dim=-1), bg.sum(dim=-1)).mean().detach()),
        "void_prob_on_gt_fg_mean": float(_safe_ratio((void_prob * fg_union).sum(dim=-1), fg_union.sum(dim=-1)).mean().detach()),
        "object_prob_on_gt_bg_mean": float(_safe_ratio(((1.0 - void_prob) * bg).sum(dim=-1), bg.sum(dim=-1)).mean().detach()),
        "foreground_area_frac": float(fg_union.mean().detach()),
        "per_object": per_object,
    }
    scalar_loss = {
        key: float(val.detach()) if torch.is_tensor(val) else val
        for key, val in loss.items()
    }
    scalar_loss.update(extra)
    return scalar_loss


def gradient_comparison(model, batch, seed: int, n_ovt_per_object: int):
    params = list(model.parameters())
    old_flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        out_recon = v14_forward_variant(model, batch, require_ovt_grad=True)
        recon = fixed_recon_loss(model, out_recon["condition_hidden"], out_recon["gt_siglip"], seed)
        grad_obj_recon, grad_void_recon = torch.autograd.grad(
            recon,
            [out_recon["object_ovts"], out_recon["void_ovts"]],
            retain_graph=False,
            allow_unused=True,
        )

        out_route = v14_forward_variant(model, batch, require_ovt_grad=True)
        route_loss = model._pgot_v14_compute_route_loss(
            out_route["owner_logits"],
            out_route["owner_probs"],
            batch["gt_masks_per_ovt"].to(out_route["condition_hidden"].device).float(),
            batch["ovt_valid_mask"].to(out_route["condition_hidden"].device, dtype=torch.bool),
        )["loss"]
        grad_obj_route, grad_void_route = torch.autograd.grad(
            route_loss,
            [out_route["object_ovts"], out_route["void_ovts"]],
            retain_graph=False,
            allow_unused=True,
        )
    finally:
        for p, flag in zip(params, old_flags):
            p.requires_grad_(flag)

    def stats(g):
        if g is None or g.numel() == 0:
            return {"l2": 0.0, "token_norm_mean": 0.0, "token_norm_max": 0.0}
        token = g.float().norm(dim=-1)
        return {
            "l2": float(g.float().flatten(1).norm(dim=-1).mean().detach()),
            "token_norm_mean": float(token.mean().detach()),
            "token_norm_max": float(token.max().detach()),
        }

    def cosine(a, b):
        if a is None or b is None or a.numel() == 0 or b.numel() == 0:
            return None
        af = a.float().flatten()
        bf = b.float().flatten()
        denom = af.norm() * bf.norm()
        if float(denom.detach()) == 0.0:
            return None
        return float((af @ bf / denom).detach())

    return {
        "loss_recon": float(recon.detach()),
        "loss_route": float(route_loss.detach()),
        "grad_norms": {
            "recon_object_ovt": stats(grad_obj_recon),
            "recon_void_ovt": stats(grad_void_recon),
            "route_object_ovt": stats(grad_obj_route),
            "route_void_ovt": stats(grad_void_route),
        },
        "cosine_recon_vs_route": {
            "object_ovt": cosine(grad_obj_recon, grad_obj_route),
            "void_ovt": cosine(grad_void_recon, grad_void_route),
        },
    }


def _source_tile(source: Image.Image, title: str):
    canvas = Image.new("RGB", (source.width, source.height + 30), "white")
    canvas.paste(source, (0, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    draw.text((8, 7), title[:90], fill=(0, 0, 0), font=font)
    return canvas


def _pred_from_owner(out, batch, eval_size: int, grid_size: int, n_ovt_per_object: int, map_stuff_to_bg: bool):
    owner_grid = _infer_square_grid(out["object_probs"].shape[-1], name="owner_probs")
    return build_pred_mask_ovt_owner_eval(
        ovt_object_probs=out["object_probs"],
        ovt_void_probs=out["void_probs"],
        ovt_valid_mask=batch["ovt_valid_mask"].to(out["condition_hidden"].device, dtype=torch.bool),
        ovt_is_thing=batch["ovt_is_thing"].to(out["condition_hidden"].device, dtype=torch.bool),
        target_size=eval_size,
        n_ovt_per_object=n_ovt_per_object,
        patch_grid=owner_grid,
        map_stuff_to_bg=map_stuff_to_bg,
    )


def save_route_visuals(
    out,
    batch,
    raw,
    target_proc,
    output_path: Path,
    *,
    eval_size: int,
    grid_size: int,
    n_ovt_per_object: int,
    max_object_maps: int,
):
    source = _source_from_target_tensor(batch["target_images"][0], target_proc)
    source = source.resize((eval_size, eval_size), Image.BILINEAR)
    owner_grid = _infer_square_grid(out["object_probs"].shape[-1], name="owner_probs")
    pred_thing = _pred_from_owner(out, batch, eval_size, grid_size, n_ovt_per_object, True)[0].detach().cpu()
    pred_all = _pred_from_owner(out, batch, eval_size, grid_size, n_ovt_per_object, False)[0].detach().cpu()

    cover, obj_valid = _object_cover_from_batch(batch, out["condition_hidden"].device, n_ovt_per_object)
    K = min(cover.shape[1], out["object_probs"].shape[1])
    gt_label = torch.zeros(eval_size, eval_size, dtype=torch.long)
    if K > 0:
        cover = _resize_cover_to_p(cover[:, :K], out["object_probs"].shape[-1])
        cover_up = _up_2d(cover, eval_size, owner_grid).detach().cpu()
        assign = cover_up.argmax(dim=0)
        fg = cover_up.amax(dim=0) > 0.0
        for k in range(K):
            if bool(obj_valid[0, k].detach().cpu()):
                gt_label[(assign == k) & fg] = k + 1

    labels = _segment_labels(raw.get("segments", []))
    tiles = [
        _source_tile(source, f"source idx={raw.get('sample_index', '?')} image_id={raw.get('image_id', '?')}"),
        _large_mask_overlay(source, gt_label, "GT object/stuff from training masks", labels, size=eval_size),
        _large_mask_overlay(source, pred_all, "V14 route pred: object+stuff", labels, size=eval_size),
        _large_mask_overlay(source, pred_thing, "V14 route pred: thing eval", labels, size=eval_size),
    ]
    void = out["void_probs"].float().sum(dim=1)
    if void.shape[1] == owner_grid * owner_grid:
        void_up = _up_2d(void, eval_size, owner_grid)[0].detach().cpu()
        tiles.append(_heat_overlay(source, void_up, "void probability"))
    obj_up = _up_2d(out["object_probs"].float()[:, :K], eval_size, owner_grid).detach().cpu()
    for k in range(min(K, max_object_maps)):
        cat = raw.get("segments", [{}])[k].get("category", f"obj{k}") if k < len(raw.get("segments", [])) else f"obj{k}"
        tiles.append(_heat_overlay(source, obj_up[k], f"object route {k + 1}: {cat}"))
    _concat_grid(tiles, cols=3).save(output_path)


def decode_variant_grid(model, decoder, variants, target_image, target_proc, output_path, seed, guidance):
    mean = torch.tensor(target_proc.image_mean)
    std = torch.tensor(target_proc.image_std)
    source = denormalize_images(target_image.float(), mean, std)[0]
    source_pil = Image.fromarray((source.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8))
    tiles = [_source_tile(source_pil, "source")]
    for name, condition_hidden in variants.items():
        torch.manual_seed(seed)
        latent = generate_siglip_latent(model, condition_hidden, guidance_level=guidance)
        img = decode_to_image(decoder, latent, condition_hidden.device)[0].cpu().clamp(0, 1)
        pil = Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        tiles.append(_source_tile(pil, name))
    _concat_grid(tiles, cols=min(3, len(tiles))).save(output_path)


def summarize_records(records):
    out = {}
    if not records:
        return out
    scalar_keys = [
        "loss", "object_loss", "void_loss", "fg_acc", "bg_acc",
        "void_prob_on_fg", "object_prob_on_bg", "entropy",
        "void_prob_on_gt_bg_mean", "void_prob_on_gt_fg_mean",
        "object_prob_on_gt_bg_mean", "foreground_area_frac",
    ]
    for key in scalar_keys:
        vals = [r["route_stats"][key] for r in records if key in r.get("route_stats", {})]
        if vals:
            out[key] = float(np.mean(vals))
    variants = records[0]["recon_loss"].keys()
    out["recon_loss_mean"] = {
        name: float(np.mean([r["recon_loss"][name] for r in records if name in r["recon_loss"]]))
        for name in variants
    }
    out["delta_recon_loss_mean"] = {
        name: float(np.mean([r["recon_shift"][name]["delta_recon_loss"] for r in records if name in r["recon_shift"]]))
        for name in variants
        if name != "baseline"
    }
    out["condition_relative_l2_mean"] = {
        name: float(np.mean([r["recon_shift"][name]["relative_l2_to_baseline"] for r in records if name in r["recon_shift"]]))
        for name in variants
        if name != "baseline"
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label|checkpoint")
    parser.add_argument("--sample_indices", default="0,1020,1407")
    parser.add_argument("--swap_sample_index", type=int, default=1020)
    parser.add_argument("--swap_target_object", type=int, default=0)
    parser.add_argument("--swap_source_object", type=int, default=0)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/v14_bottleneck_diagnostics")
    parser.add_argument("--grid_size", type=int, default=16)
    parser.add_argument("--eval_size", type=int, default=224)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--max_object_maps", type=int, default=8)
    parser.add_argument("--compute_gradients", action="store_true")
    parser.add_argument("--decode_recon", action="store_true")
    parser.add_argument("--diffusion_inference_steps", type=int, default=15)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_indices = [int(x) for x in args.sample_indices.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]

    all_results = {}
    for label, model_path in map(parse_model, args.model):
        log.info("Loading %s: %s", label, model_path)
        model, tokenizer = _load_model(model_path, dtype, device)
        if not bool(getattr(model.config, "pgot_v14_enable", False)):
            raise ValueError(f"{label} is not a V14 checkpoint.")
        vt_list = model.get_vision_tower_aux_list()
        image_proc = vt_list[0].image_processor
        target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
        dataset = Pix2CapPGOTDataset(
            jsonl_path=args.val_jsonl,
            tokenizer=tokenizer,
            image_processor=image_proc,
            target_image_processor=target_proc,
            grid_size=args.grid_size,
            max_caption_tokens=2048,
            n_ovt_per_object=args.n_ovt_per_object,
            max_objects=args.max_objects,
            panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
            image_preprocess_mode=args.image_preprocess_mode,
            coda_crop_size=args.coda_crop_size,
        )
        collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
        decoder = load_rae_decoder(model, device, dtype) if args.decode_recon else None
        if args.decode_recon and args.diffusion_inference_steps != 50:
            from scale_rae.model.diffusion_loss.diffusion import create_diffusion

            inf = model.diff_head.inference_flow
            model.diff_head.inference_flow = create_diffusion(
                str(args.diffusion_inference_steps),
                noise_schedule="linear",
                use_kl=False,
                sigma_small=False,
                predict_xstart=False,
                learn_sigma=False,
                rescale_learned_sigmas=False,
                diffusion_steps=int(getattr(inf, "diffusion_steps", 1000)),
                input_base_dimension_ratio=float(getattr(inf, "size_ratio", 1.0)),
                diffusion_type="rf",
                use_loss_weighting=False,
            )

        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)
        model_results = []

        swap_batch = collator([dataset[args.swap_sample_index]]) if args.swap_sample_index is not None else None
        with torch.no_grad():
            swap_out = v14_forward_variant(model, swap_batch) if swap_batch is not None else None

        for sample_idx in sample_indices:
            raw = dict(raw_samples[sample_idx])
            raw["sample_index"] = sample_idx
            batch = collator([dataset[sample_idx]])
            with torch.no_grad():
                base = v14_forward_variant(model, batch)
                variants = {
                    "baseline": base,
                    "zero_final_object_ovt": v14_forward_variant(model, batch, zero_final_object=True),
                    "zero_final_void": v14_forward_variant(model, batch, zero_final_void=True),
                    "zero_final_object_and_void": v14_forward_variant(model, batch, zero_final_object=True, zero_final_void=True),
                    "zero_input_object_ovt": v14_forward_variant(model, batch, zero_input_object=True),
                    "zero_input_void": v14_forward_variant(model, batch, zero_input_void=True),
                }
                if swap_out is not None:
                    variants["swap_final_object_ovt"] = v14_forward_variant(
                        model,
                        batch,
                        swap_object_source=swap_out["object_ovts"].detach(),
                        swap_target_object=args.swap_target_object,
                        swap_source_object=args.swap_source_object,
                    )

                recon_loss = {}
                shift = {}
                for name, out in variants.items():
                    recon_loss[name] = float(fixed_recon_loss(model, out["condition_hidden"], out["gt_siglip"], args.seed).detach())
                    shift[name] = {
                        "cosine_to_baseline": float(F.cosine_similarity(
                            base["condition_hidden"].float().flatten(1),
                            out["condition_hidden"].float().flatten(1),
                        ).mean().detach()),
                        "relative_l2_to_baseline": float(
                            (out["condition_hidden"].float() - base["condition_hidden"].float()).norm()
                            / base["condition_hidden"].float().norm().clamp_min(1e-8)
                        ),
                        "delta_recon_loss": recon_loss[name] - recon_loss["baseline"],
                    }

            rec = {
                "sample_index": int(sample_idx),
                "image_id": int(raw_samples[sample_idx]["image_id"]),
                "n_ovt_tokens": int(batch["ovt_valid_mask"][0].sum().item()),
                "n_objects": int(batch["ovt_valid_mask"][0].sum().item() // args.n_ovt_per_object),
                "route_stats": route_stats(model, base, batch, args.n_ovt_per_object),
                "recon_loss": recon_loss,
                "recon_shift": shift,
                "variant_order": list(variants.keys()),
            }
            if args.compute_gradients:
                rec["gradient_comparison"] = gradient_comparison(
                    model, batch, args.seed, args.n_ovt_per_object
                )

            route_path = model_dir / f"sample{sample_idx}_route_visuals.png"
            save_route_visuals(
                base,
                batch,
                raw,
                target_proc,
                route_path,
                eval_size=args.eval_size,
                grid_size=args.grid_size,
                n_ovt_per_object=args.n_ovt_per_object,
                max_object_maps=args.max_object_maps,
            )
            rec["route_visuals"] = str(route_path)

            if args.decode_recon:
                recon_path = model_dir / f"sample{sample_idx}_bottleneck_recon_grid.png"
                decode_variant_grid(
                    model,
                    decoder,
                    {k: v["condition_hidden"] for k, v in variants.items()},
                    batch["target_images"].to(device),
                    target_proc,
                    recon_path,
                    args.seed,
                    args.guidance_scale,
                )
                rec["recon_grid"] = str(recon_path)

            model_results.append(rec)
            with (model_dir / f"sample{sample_idx}.json").open("w") as f:
                json.dump(rec, f, indent=2)
            log.info(
                "%s sample=%s route_loss=%.4f recon=%s",
                label,
                sample_idx,
                rec["route_stats"].get("loss", float("nan")),
                recon_loss,
            )

        summary = {
            "model": label,
            "model_path": model_path,
            "sample_indices": sample_indices,
            "image_preprocess_mode": args.image_preprocess_mode,
            "summary": summarize_records(model_results),
            "records": model_results,
        }
        all_results[label] = summary
        with (model_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        del decoder, dataset, tokenizer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with (output_dir / "summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    notes = []
    for label, summary in all_results.items():
        s = summary["summary"]
        notes.append(f"# {label}")
        notes.append(f"- route loss mean: {s.get('loss')}")
        notes.append(f"- fg_acc / bg_acc: {s.get('fg_acc')} / {s.get('bg_acc')}")
        notes.append(f"- object_prob_on_bg mean: {s.get('object_prob_on_gt_bg_mean')}")
        notes.append(f"- void_prob_on_fg mean: {s.get('void_prob_on_gt_fg_mean')}")
        notes.append(f"- recon loss mean: {s.get('recon_loss_mean')}")
        notes.append(f"- delta recon loss mean: {s.get('delta_recon_loss_mean')}")
        notes.append(f"- condition relative L2 mean: {s.get('condition_relative_l2_mean')}")
    (output_dir / "analysis_notes.md").write_text("\n".join(notes) + "\n")
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
