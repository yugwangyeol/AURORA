"""V12-specific visual diagnostics for OVT ownership and RAE usage.

This script answers three questions for a trained V12 checkpoint:
  - how OVT-owner maps evolve across the inserted OVT-update blocks;
  - whether RAE queries attend to thing OVTs, stuff OVTs, void, or self;
  - whether blocking an object's OVT pair changes reconstruction loss.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import (
    fari_metric,
    mbo_metric,
    miou_metric,
    preproc_masks_overlap,
)
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.eval.visualize_ovt_overlays import (
    _add_title,
    _concat_grid,
    _large_mask_overlay,
    _load_font,
    _load_model,
    _source_from_target_tensor,
)
from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _parse_indices(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _resolve_layers(model, spec: str) -> list[int]:
    if spec == "v12_blocks":
        return [int(x) for x in getattr(model, "pgot_v12_layers", [])]
    if hasattr(model, "_resolve_llm_qk_outside_layers"):
        return model._resolve_llm_qk_outside_layers(spec)
    n_layers = len(model.model.layers)
    if spec.startswith("last"):
        n = int(spec[4:] or "1")
        return list(range(max(0, n_layers - n), n_layers))
    return [int(x) for x in spec.split(",") if x.strip()]


def _label_segments(segments: list[dict], thing_categories: set[str]) -> list[str]:
    counts = {}
    labels = []
    for seg in segments:
        cat = seg.get("category", "object")
        counts[cat] = counts.get(cat, 0) + 1
        kind = "thing" if cat in thing_categories else "stuff"
        labels.append(f"{cat} {counts[cat]} ({kind})")
    return labels


def _thing_object_mask(batch: dict, n_per_obj: int) -> torch.Tensor:
    ovt_is_thing = batch["ovt_is_thing"]
    B, M = ovt_is_thing.shape
    K = M // max(int(n_per_obj), 1)
    return ovt_is_thing[:, : K * n_per_obj].reshape(B, K, n_per_obj).any(dim=2)


def _valid_object_mask(batch: dict, n_per_obj: int) -> torch.Tensor:
    valid = batch["ovt_valid_mask"]
    B, M = valid.shape
    K = M // max(int(n_per_obj), 1)
    return valid[:, : K * n_per_obj].reshape(B, K, n_per_obj).any(dim=2)


def _select_objects(batch: dict, max_objects: int, n_per_obj: int) -> list[int]:
    valid = _valid_object_mask(batch, n_per_obj)[0]
    thing = _thing_object_mask(batch, n_per_obj)[0]
    masks = batch["gt_masks_per_ovt"][0]
    K = int(valid.numel())
    areas = []
    for k in range(K):
        s = k * n_per_obj
        areas.append(float(masks[s].float().sum().item()))
    thing_ids = [k for k in range(K) if bool(valid[k]) and bool(thing[k])]
    stuff_ids = [k for k in range(K) if bool(valid[k]) and not bool(thing[k])]
    thing_ids.sort(key=lambda k: areas[k])
    stuff_ids.sort(key=lambda k: areas[k], reverse=True)
    return (thing_ids + stuff_ids)[:max_objects]


def _upsample_map(values: torch.Tensor, target_size: int, patch_grid: int) -> torch.Tensor:
    return F.interpolate(
        values.float().reshape(1, 1, patch_grid, patch_grid),
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def _heat_overlay(source: Image.Image, values: torch.Tensor, title: str, size: int = 256) -> Image.Image:
    base = np.asarray(source.resize((size, size), Image.BILINEAR)).astype(np.float32)
    heat = values.detach().cpu().float()
    if tuple(heat.shape) != (size, size):
        side = int(round(float(heat.numel()) ** 0.5))
        heat = _upsample_map(heat.reshape(side * side), size, side)
    heat = heat - heat.min()
    heat = heat / heat.max().clamp_min(1e-6)
    h = heat.numpy()
    color = np.zeros_like(base)
    color[..., 0] = 255.0
    color[..., 1] = 45.0
    color[..., 2] = 35.0
    out = base * (1.0 - 0.62 * h[..., None]) + color * (0.62 * h[..., None])
    return _add_title(Image.fromarray(out.clip(0, 255).astype(np.uint8)), title, height=34)


def _bar_chart(records: list[dict], output_path: Path) -> None:
    groups = ["thing_ovt", "stuff_ovt", "void", "rae_self", "other"]
    row_h = 28
    left = 122
    w = 760
    h = 52 + row_h * len(records) * len(groups)
    img = Image.new("RGB", (w, max(h, 180)), "white")
    draw = ImageDraw.Draw(img)
    font = _load_font(12)
    title_font = _load_font(16)
    draw.text((12, 12), "RAE query attention mass by source", fill=(0, 0, 0), font=title_font)
    colors = {
        "thing_ovt": (235, 72, 72),
        "stuff_ovt": (242, 162, 58),
        "void": (66, 124, 245),
        "rae_self": (84, 184, 104),
        "other": (155, 155, 155),
    }
    y = 44
    for rec in records:
        layer = rec["layer"]
        for group in groups:
            val = float(rec.get(group, 0.0))
            draw.text((12, y + 5), f"L{layer} {group}", fill=(30, 30, 30), font=font)
            draw.rectangle((left, y + 5, w - 24, y + 20), outline=(225, 225, 225))
            bar_w = int((w - left - 26) * max(0.0, min(1.0, val)))
            draw.rectangle((left, y + 5, left + bar_w, y + 20), fill=colors[group])
            draw.text((left + bar_w + 5, y + 3), f"{val:.3f}", fill=(30, 30, 30), font=font)
            y += row_h
    img.save(output_path)


def _pred_from_owner(
    object_probs: torch.Tensor,
    void_probs: torch.Tensor,
    batch: dict,
    args,
) -> torch.Tensor:
    return build_pred_mask_ovt_owner_eval(
        ovt_object_probs=object_probs,
        ovt_void_probs=void_probs,
        ovt_valid_mask=batch["ovt_valid_mask"].to(object_probs.device),
        ovt_is_thing=batch["ovt_is_thing"].to(object_probs.device),
        target_size=args.eval_size,
        n_ovt_per_object=args.n_ovt_per_object,
        patch_grid=args.grid_size,
        map_stuff_to_bg=(args.gt_source == "coco_instance_coda"),
    )[0].detach().cpu()


def _fixed_recon_loss(model, rae_hidden: torch.Tensor, target: torch.Tensor, seed: int) -> float:
    devices = [rae_hidden.device.index] if rae_hidden.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        loss = model._captionslot_compute_diffusion_loss(
            hidden=rae_hidden,
            target_features=target,
        )
    return float(loss.detach().cpu())


def _exact_rae_source_mass(
    model,
    hidden_states,
    attention_bias: torch.Tensor,
    positions: dict,
    ovt_abs_positions: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    ovt_is_thing: torch.Tensor,
    layer_ids: list[int],
) -> list[dict]:
    records = []
    if hidden_states is None:
        return records
    B = int(attention_bias.shape[0])
    if B != 1:
        raise ValueError("diagnose_v12_attention_paths currently expects batch size 1.")
    device = attention_bias.device
    L = int(attention_bias.shape[-1])
    rae_s, rae_e = int(positions["rae_s"]), int(positions["rae_e"])
    null_s, null_e = int(positions.get("null_bg_s", 0)), int(positions.get("null_bg_e", 0))

    thing_mask = torch.zeros(L, dtype=torch.bool, device=device)
    stuff_mask = torch.zeros(L, dtype=torch.bool, device=device)
    valid = ovt_valid_mask[0].to(device=device, dtype=torch.bool)
    is_thing = ovt_is_thing[0].to(device=device, dtype=torch.bool)
    ovt_pos = ovt_abs_positions[0].to(device=device, dtype=torch.long)
    if bool(valid.any()):
        thing_pos = ovt_pos[valid & is_thing]
        stuff_pos = ovt_pos[valid & ~is_thing]
        thing_mask[thing_pos] = True
        stuff_mask[stuff_pos] = True
    void_mask = torch.zeros(L, dtype=torch.bool, device=device)
    if null_e > null_s:
        void_mask[null_s:null_e] = True
    self_mask = torch.zeros(L, dtype=torch.bool, device=device)
    self_mask[rae_s:rae_e] = True
    known = thing_mask | stuff_mask | void_mask | self_mask

    def mass(probs: torch.Tensor, mask: torch.Tensor) -> float:
        if not bool(mask.any()):
            return 0.0
        return float(probs[..., mask].sum(dim=-1).mean().detach().cpu())

    for layer_idx in layer_ids:
        if layer_idx >= len(hidden_states) - 1:
            continue
        block = model.model.layers[layer_idx]
        attn = block.self_attn
        x = block.input_layernorm(hidden_states[layer_idx])
        _, seq_len, _ = x.shape
        q_input = x[:, rae_s:rae_e, :]
        q_proj = attn.q_proj(q_input)
        k_proj = attn.k_proj(x)
        num_heads = int(getattr(attn, "num_heads", model.config.num_attention_heads))
        num_kv_heads = int(getattr(attn, "num_key_value_heads", model.config.num_key_value_heads))
        head_dim = int(getattr(attn, "head_dim", q_proj.shape[-1] // num_heads))
        q = q_proj.view(1, rae_e - rae_s, num_heads, head_dim).transpose(1, 2)
        k = k_proj.view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(k, seq_len=seq_len)
        cos = cos.to(device=q.device, dtype=q.dtype)
        sin = sin.to(device=q.device, dtype=q.dtype)
        key_positions = torch.arange(seq_len, device=device, dtype=torch.long)
        query_positions = torch.arange(rae_s, rae_e, device=device, dtype=torch.long)
        k_cos = cos[key_positions].view(1, 1, seq_len, head_dim)
        k_sin = sin[key_positions].view(1, 1, seq_len, head_dim)
        q_cos = cos[query_positions].view(1, 1, rae_e - rae_s, head_dim)
        q_sin = sin[query_positions].view(1, 1, rae_e - rae_s, head_dim)
        q = (q * q_cos) + (model._pgot_rotate_half(q) * q_sin)
        k = (k * k_cos) + (model._pgot_rotate_half(k) * k_sin)

        if num_kv_heads != num_heads:
            groups = int(getattr(attn, "num_key_value_groups", num_heads // num_kv_heads))
            k = k.repeat_interleave(groups, dim=1)

        scores = torch.matmul(q.float(), k.float().transpose(2, 3)) / math.sqrt(float(head_dim))
        scores = scores + attention_bias[:, :, rae_s:rae_e, :].float()
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        rec = {
            "layer": int(layer_idx),
            "thing_ovt": mass(probs, thing_mask),
            "stuff_ovt": mass(probs, stuff_mask),
            "void": mass(probs, void_mask),
            "rae_self": mass(probs, self_mask),
            "other": mass(probs, ~known),
        }
        records.append(rec)
    return records


def _average_records(records: list[dict]) -> dict:
    if not records:
        return {}
    keys = [k for k in records[0] if k != "layer"]
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


def _load_gt(raw: dict, args):
    if args.gt_source == "pix2cap_panoptic":
        seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
        return load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size), None
    cache = CocoInstanceMaskCache(args.coco_mask_cache)
    gt = cache.get(int(raw["image_id"]))
    overlap = cache.get_overlap(int(raw["image_id"]))
    if gt is None:
        seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
        gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
        overlap = None
    else:
        args.eval_size = cache.size
    return gt, overlap


def _metric_scores(gt: torch.Tensor, pred: torch.Tensor, overlap: torch.Tensor | None) -> dict:
    ov = overlap.unsqueeze(0) if overlap is not None else None
    return {
        "fARI": float(fari_metric(gt.unsqueeze(0), pred.unsqueeze(0), ov)),
        "mBO": float(mbo_metric(gt.unsqueeze(0), pred.unsqueeze(0), ov)),
        "mIoU": float(miou_metric(gt.unsqueeze(0), pred.unsqueeze(0), ov)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--sample_indices", default="0,1020,1407")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/v12_attention_diagnostics")
    parser.add_argument(
        "--gt_source",
        choices=["coco_instance_coda", "pix2cap_panoptic"],
        default="coco_instance_coda",
    )
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--max_objects_to_show", type=int, default=8)
    parser.add_argument("--max_object_drops", type=int, default=6)
    parser.add_argument("--rae_layers", default="last4")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    sample_indices = _parse_indices(args.sample_indices)
    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]

    model, tokenizer = _load_model(args.model_path, dtype=dtype, device=device)
    if not bool(getattr(model.config, "pgot_v12_enable", False)):
        raise ValueError("This diagnostic expects a V12 checkpoint with pgot_v12_enable=True.")
    vt_list = model.get_vision_tower_aux_list()
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=vt_list[0].image_processor,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
    )
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
    layer_ids = _resolve_layers(model, args.rae_layers)

    all_summary = {
        "model_path": args.model_path,
        "sample_indices": sample_indices,
        "gt_source": args.gt_source,
        "rae_layers": layer_ids,
        "samples": [],
    }

    for sample_idx in sample_indices:
        raw = raw_samples[sample_idx]
        sample_dir = output_dir / f"sample{sample_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        batch = collator([dataset[sample_idx]])
        source = _source_from_target_tensor(batch["target_images"][0], target_proc)
        source = source.resize((args.eval_size, args.eval_size), Image.BILINEAR)
        kwargs = dict(
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        with torch.no_grad():
            out = pgot_forward_eval(
                model,
                **kwargs,
                return_hidden_states=True,
                return_v12_block_maps=True,
            )

        gt, overlap = _load_gt(raw, args)
        gt_eval, _ = preproc_masks_overlap(gt, gt.clone(), overlap) if overlap is not None else (gt, gt)
        labels = _label_segments(raw["segments"], thing_categories)
        selected = _select_objects(batch, args.max_objects_to_show, args.n_ovt_per_object)
        block_records = out.get("v12_block_owner_records") or []

        seg_tiles = [
            _add_title(source, f"source idx={sample_idx}", height=40),
            _large_mask_overlay(source, gt_eval, f"GT {args.gt_source}", None, size=args.eval_size),
        ]
        block_metrics = []
        for rec in block_records:
            pred = _pred_from_owner(rec["object_probs"], rec["void_probs"], batch, args)
            scores = _metric_scores(gt, pred, overlap)
            scores["layer"] = int(rec["layer"])
            block_metrics.append(scores)
            seg_tiles.append(
                _large_mask_overlay(
                    source,
                    pred,
                    f"L{rec['layer']} owner pred",
                    None,
                    size=args.eval_size,
                )
            )
        final_pred = _pred_from_owner(out["ovt_object_probs"], out["ovt_void_probs"], batch, args)
        final_scores = _metric_scores(gt, final_pred, overlap)
        seg_tiles.append(_large_mask_overlay(source, final_pred, "final owner pred", None, size=args.eval_size))
        _concat_grid(seg_tiles, cols=min(3, len(seg_tiles))).save(sample_dir / "block_segmentation.png")

        map_tiles = [_add_title(source, f"source idx={sample_idx}", height=34)]
        for obj_idx in selected:
            label = labels[obj_idx] if obj_idx < len(labels) else f"object {obj_idx + 1}"
            for rec in block_records:
                obj_map = rec["object_probs"][0, obj_idx].detach().cpu()
                map_tiles.append(_heat_overlay(source, obj_map, f"L{rec['layer']} {obj_idx + 1}: {label}", size=224))
            final_map = out["ovt_object_probs"][0, obj_idx].detach().cpu()
            map_tiles.append(_heat_overlay(source, final_map, f"final {obj_idx + 1}: {label}", size=224))
        _concat_grid(map_tiles, cols=5).save(sample_dir / "object_owner_maps.png")

        rae_records = _exact_rae_source_mass(
            model=model,
            hidden_states=out["hidden_states"],
            attention_bias=out["attn_bias"],
            positions=out["positions"],
            ovt_abs_positions=out["ovt_abs_positions"],
            ovt_valid_mask=out["ovt_valid_mask"],
            ovt_is_thing=batch["ovt_is_thing"].to(out["attn_bias"].device),
            layer_ids=layer_ids,
        )
        _bar_chart(rae_records, sample_dir / "rae_source_mass.png")

        drop_records = []
        base_recon_loss = _fixed_recon_loss(model, out["rae_hidden"], out["gt_siglip"], args.seed)
        for obj_idx in selected[: args.max_object_drops]:
            block_ovts = tuple(
                obj_idx * args.n_ovt_per_object + j
                for j in range(args.n_ovt_per_object)
            )
            with torch.no_grad():
                dropped = pgot_forward_eval(
                    model,
                    **kwargs,
                    rae_block_ovt_indices=block_ovts,
                )
            drop_loss = _fixed_recon_loss(model, dropped["rae_hidden"], dropped["gt_siglip"], args.seed)
            drop_records.append(
                {
                    "object_index": int(obj_idx),
                    "label": labels[obj_idx] if obj_idx < len(labels) else f"object {obj_idx + 1}",
                    "blocked_ovt_indices": list(block_ovts),
                    "recon_loss": drop_loss,
                    "delta_from_baseline": drop_loss - base_recon_loss,
                }
            )

        sample_summary = {
            "sample_index": int(sample_idx),
            "image_id": int(raw["image_id"]),
            "gt_source": args.gt_source,
            "source_uses_eval_target_image": True,
            "paths": {
                "block_segmentation": str(sample_dir / "block_segmentation.png"),
                "object_owner_maps": str(sample_dir / "object_owner_maps.png"),
                "rae_source_mass": str(sample_dir / "rae_source_mass.png"),
            },
            "selected_objects": [
                {
                    "object_index": int(k),
                    "label": labels[k] if k < len(labels) else f"object {k + 1}",
                    "is_thing": bool(_thing_object_mask(batch, args.n_ovt_per_object)[0, k]),
                }
                for k in selected
            ],
            "block_metrics": block_metrics,
            "final_metrics": final_scores,
            "rae_source_mass": rae_records,
            "rae_source_mass_mean": _average_records(rae_records),
            "baseline_recon_loss": base_recon_loss,
            "object_drop": drop_records,
        }
        with open(sample_dir / "summary.json", "w") as f:
            json.dump(sample_summary, f, indent=2)
        all_summary["samples"].append(sample_summary)
        print(json.dumps({
            "sample_index": sample_idx,
            "final_metrics": final_scores,
            "rae_source_mass_mean": sample_summary["rae_source_mass_mean"],
            "baseline_recon_loss": base_recon_loss,
        }, indent=2))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)
    print(output_dir / "summary.json")


if __name__ == "__main__":
    main()
