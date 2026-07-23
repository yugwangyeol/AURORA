"""Diagnose why outside-only PGOT supervision does not yield good slots.

The script intentionally does not train or sweep checkpoints. It runs the
current checkpoints once and records:
  - CODA-style segmentation symptoms (FG predicted as bg / split / merge)
  - thing vs stuff OVT-map statistics
  - whether an object's own GT patches are actually won by its own OVT map
  - cross-OVT map overlap / ambiguous ownership
  - presentation-friendly overlays for selected samples
"""

import argparse
import gc
import json
import math
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import (
    build_pred_mask_spatial_readout,
    fari_metric,
    mbo_metric,
    miou_metric,
    ovt_logits_to_pred_mask,
    preproc_masks_overlap,
)
from pgot.eval.run_eval import (
    CocoInstanceMaskCache,
    load_gt_panoptic_mask,
    load_thing_categories,
)
from pgot.eval.visualize_fari_diagnostic import (
    _contingency,
    _error_overlay,
    _matrix_image,
)
from pgot.eval.visualize_ovt_overlays import (
    _add_title,
    _color_mask_overlay,
    _concat_grid,
    _heat_overlay,
    _large_mask_overlay,
    _load_model,
    _palette,
    _safe_name,
    _segment_labels,
)
from pgot.model.pgot_utils import (
    build_pred_mask_competition_eval,
    build_pred_mask_llm_attention_eval,
    build_pred_mask_null_bg_eval,
)
from pgot.train.pgot_dataset import (
    PGOTDataCollator,
    Pix2CapPGOTDataset,
    _coda_center_crop_image,
)


def _parse_model_spec(spec: str):
    parts = spec.split("|")
    if len(parts) not in (2, 4):
        raise ValueError("--model must be 'label|path' or 'label|path|readout|merge'")
    return {
        "label": parts[0],
        "path": parts[1],
        "readout": parts[2] if len(parts) == 4 else "threshold",
        "merge": parts[3] if len(parts) == 4 else "mean",
    }


def _mean(xs):
    vals = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(vals)) if vals else float("nan")


def _weighted_mean(vals, weights):
    pairs = [
        (float(v), float(w))
        for v, w in zip(vals, weights)
        if np.isfinite(float(v)) and float(w) > 0
    ]
    if not pairs:
        return float("nan")
    return float(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))


def _bucket(n_obj: int):
    if n_obj <= 3:
        return "1-3"
    if n_obj <= 6:
        return "4-6"
    if n_obj <= 10:
        return "7-10"
    return "11+"


def _object_size_bucket(area_frac: float):
    if area_frac < 0.01:
        return "tiny"
    if area_frac < 0.03:
        return "small"
    if area_frac < 0.10:
        return "medium"
    return "large"


def _remap_thing_flags(batch, samples_iter, batch_idx, batch_size, thing_categories, n_ovt_per_object):
    valid = batch["ovt_valid_mask"]
    ovt_is_thing = batch.get("ovt_is_thing")
    if ovt_is_thing is None:
        ovt_is_thing = torch.zeros_like(valid, dtype=torch.bool)
    else:
        ovt_is_thing = ovt_is_thing.clone().to(dtype=torch.bool)

    if thing_categories is None:
        return ovt_is_thing

    for b in range(valid.shape[0]):
        global_idx = batch_idx * batch_size + b
        segs = samples_iter[global_idx]["segments"]
        for k, seg in enumerate(segs):
            s = k * n_ovt_per_object
            e = s + n_ovt_per_object
            if e <= ovt_is_thing.shape[1] and seg.get("category") in thing_categories:
                ovt_is_thing[b, s:e] = True
    return ovt_is_thing


def _build_pred(args, spec, out, batch, samples_iter, batch_idx, thing_categories):
    valid = out["ovt_valid_mask"].clone()
    ovt_is_thing = _remap_thing_flags(
        batch,
        samples_iter,
        batch_idx,
        args.batch_size,
        thing_categories,
        args.n_ovt_per_object,
    ).to(valid.device)

    readout = spec["readout"]
    if readout == "threshold":
        valid_for_threshold = valid & ovt_is_thing if args.gt_source == "coco_instance" else valid
        return ovt_logits_to_pred_mask(
            out["ovt_logits"],
            valid_for_threshold,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            bg_threshold=args.bg_threshold,
        )
    if readout == "spatial_trainmatch":
        spatial_temp = float(
            getattr(
                args,
                "checkpoint_spatial_temperature",
                getattr(args, "spatial_temperature", 1.0),
            )
        )
        return build_pred_mask_spatial_readout(
            out["ovt_logits"],
            valid,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge="mean",
            temp=spatial_temp,
            ovt_is_thing=ovt_is_thing,
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    if readout == "spatial":
        return build_pred_mask_spatial_readout(
            out["ovt_logits"],
            valid,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
            temp=args.spatial_temperature,
            ovt_is_thing=ovt_is_thing,
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    if readout == "nullbg":
        return build_pred_mask_null_bg_eval(
            out["ovt_logits"],
            out["null_bg_logits"],
            valid,
            ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
        )
    if readout == "llm_attention":
        if out.get("llm_attention_maps") is None:
            raise ValueError(f"{spec['label']} requested llm_attention but no maps were returned")
        bg_maps = out.get("llm_attention_void_maps")
        if bg_maps is None or bg_maps.numel() == 0:
            bg_maps = out.get("llm_attention_register_maps")
        return build_pred_mask_llm_attention_eval(
            out["llm_attention_maps"],
            bg_maps,
            valid,
            ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    return build_pred_mask_competition_eval(
        out["ovt_logits"],
        out["reg_logits"],
        valid,
        ovt_is_thing,
        target_size=args.eval_size,
        n_ovt_per_object=args.n_ovt_per_object,
        patch_grid=args.grid_size,
        merge=spec["merge"],
    )


def _object_maps(args, spec, out):
    """Return object maps at patch resolution and an optional void/bg map.

    Values intentionally match the requested readout family:
      - threshold/competition/nullbg: sigmoid(dot-product) maps
      - spatial/spatial_trainmatch: patch-axis softmax maps
      - llm_attention: exact internal attention maps
    """
    n = max(int(args.n_ovt_per_object), 1)
    if spec["readout"] == "llm_attention":
        maps = out["llm_attention_maps"].float()
        B, M, P = maps.shape
        K = M // n
        maps = maps[:, : K * n].reshape(B, K, n, P)
        obj = maps.amax(dim=2) if spec["merge"] == "max" else maps.mean(dim=2)
        bg = out.get("llm_attention_void_maps")
        if bg is None or bg.numel() == 0:
            bg = out.get("llm_attention_register_maps")
        if bg is not None and bg.numel() > 0:
            bg = bg.float().mean(dim=1)
        return obj, bg, "llm_attention"

    logits = out["ovt_logits"].float()
    B, M, P = logits.shape
    K = M // n
    logits = logits[:, : K * n].reshape(B, K, n, P)
    if spec["readout"] in {"spatial", "spatial_trainmatch"}:
        temp = float(args.spatial_temperature)
        if spec["readout"] == "spatial_trainmatch":
            temp = float(getattr(args, "checkpoint_spatial_temperature", temp))
        maps = torch.softmax(logits / max(temp, 1e-6), dim=-1)
        kind = "patch_softmax"
    else:
        maps = torch.sigmoid(logits)
        kind = "sigmoid_dot"
    obj = maps.amax(dim=2) if spec["merge"] == "max" else maps.mean(dim=2)
    bg = None
    if spec["readout"] == "nullbg" and out.get("null_bg_logits") is not None:
        bg = torch.sigmoid(out["null_bg_logits"].float()).mean(dim=1)
    return obj, bg, kind


def _gt_object_masks(batch, n_ovt_per_object):
    masks = batch["gt_masks_per_ovt"].float()
    B, M, P = masks.shape
    n = max(int(n_ovt_per_object), 1)
    K = M // n
    return masks[:, : K * n].reshape(B, K, n, P).amax(dim=2).clamp(0.0, 1.0)


def _object_valid_and_thing(batch, samples_iter, batch_idx, args, thing_categories, device):
    valid = batch["ovt_valid_mask"]
    ovt_is_thing = _remap_thing_flags(
        batch,
        samples_iter,
        batch_idx,
        args.batch_size,
        thing_categories,
        args.n_ovt_per_object,
    )
    n = max(int(args.n_ovt_per_object), 1)
    B, M = valid.shape
    K = M // n
    obj_valid = valid[:, : K * n].reshape(B, K, n).any(dim=2).to(device)
    obj_thing = ovt_is_thing[:, : K * n].reshape(B, K, n).any(dim=2).to(device)
    return obj_valid, obj_thing


def _normalized_pair_overlap(obj_maps, obj_valid):
    vals = []
    B, K, P = obj_maps.shape
    maps = obj_maps.float().clamp_min(0.0)
    maps = maps / maps.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    for b in range(B):
        idx = obj_valid[b].nonzero(as_tuple=False).flatten()
        if idx.numel() < 2:
            vals.append(float("nan"))
            continue
        overlaps = []
        for ii in range(int(idx.numel())):
            for jj in range(ii + 1, int(idx.numel())):
                overlaps.append(torch.minimum(maps[b, idx[ii]], maps[b, idx[jj]]).sum())
        vals.append(float(torch.stack(overlaps).mean().item()) if overlaps else float("nan"))
    return vals


def _map_diagnostics(
    args,
    spec,
    out,
    batch,
    samples_iter,
    batch_idx,
    thing_categories,
    global_indices=None,
):
    obj_maps, bg_map, map_kind = _object_maps(args, spec, out)
    device = obj_maps.device
    gt_obj = _gt_object_masks(batch, args.n_ovt_per_object).to(device)
    obj_valid, obj_thing = _object_valid_and_thing(
        batch,
        samples_iter,
        batch_idx,
        args,
        thing_categories,
        device,
    )
    obj_maps = obj_maps * obj_valid.unsqueeze(-1).float()
    gt_obj = gt_obj * obj_valid.unsqueeze(-1).float()

    B, K, P = obj_maps.shape
    union = gt_obj.amax(dim=1)
    valid_float = obj_valid.float()
    neg = torch.finfo(obj_maps.dtype).min
    masked_maps = obj_maps.masked_fill(~obj_valid.unsqueeze(-1), neg)
    top_vals, top_idx = masked_maps.topk(k=min(2, K), dim=1)
    winner = top_idx[:, 0]
    top1 = top_vals[:, 0]
    if K >= 2:
        top2 = top_vals[:, 1]
    else:
        top2 = torch.zeros_like(top1)

    annotated = union > 0.0
    ratio = top2.clamp_min(0.0) / top1.clamp_min(1e-8)
    ambiguity = ((ratio > 0.75) & annotated).float().sum(dim=-1) / annotated.float().sum(dim=-1).clamp_min(1.0)
    top_margin = ((top1 - top2) * annotated.float()).sum(dim=-1) / annotated.float().sum(dim=-1).clamp_min(1.0)
    pair_overlap = _normalized_pair_overlap(obj_maps, obj_valid)

    sample_records = []
    object_records = []
    for b in range(B):
        global_idx = (
            int(global_indices[b])
            if global_indices is not None
            else batch_idx * args.batch_size + b
        )
        raw = samples_iter[b] if global_indices is not None else samples_iter[global_idx]
        n_valid_obj = int(obj_valid[b].sum().item())
        sample_records.append({
            "sample_index": int(global_idx),
            "image_id": int(raw["image_id"]),
            "map_kind": map_kind,
            "map_pair_overlap_mean": float(pair_overlap[b]),
            "map_ambiguous_patch_frac": float(ambiguity[b].item()),
            "map_top1_top2_margin_mean": float(top_margin[b].item()),
            "valid_object_count": n_valid_obj,
        })

        for k, seg in enumerate(raw["segments"][:K]):
            if not bool(obj_valid[b, k]):
                continue
            mask = gt_obj[b, k].float()
            if float(mask.sum().item()) <= 0:
                continue
            own = obj_maps[b, k].float().clamp_min(0.0)
            other = gt_obj[b, torch.arange(K, device=device) != k].amax(dim=0) if K > 1 else torch.zeros_like(mask)
            neutral = (1.0 - union[b]).clamp(0.0, 1.0)
            outside = (1.0 - mask).clamp(0.0, 1.0)
            own_sum = own.sum().clamp_min(1e-8)
            area = mask.sum().clamp_min(1e-8)
            outside_area = outside.sum().clamp_min(1e-8)
            other_area = other.sum().clamp_min(1e-8)
            neutral_area = neutral.sum().clamp_min(1e-8)
            win_frac = ((winner[b] == k).float() * mask).sum() / area
            record = {
                "sample_index": int(global_idx),
                "image_id": int(raw["image_id"]),
                "object_index": int(k),
                "category": seg.get("category", "object"),
                "is_thing": bool(obj_thing[b, k].item()),
                "area_frac": float((mask.sum() / float(P)).item()),
                "size_bucket": _object_size_bucket(float((mask.sum() / float(P)).item())),
                "map_self_mass_frac": float(((own * mask).sum() / own_sum).item()),
                "map_outside_mass_frac": float(((own * outside).sum() / own_sum).item()),
                "map_other_region_mass_frac": float(((own * other).sum() / own_sum).item()),
                "map_neutral_mass_frac": float(((own * neutral).sum() / own_sum).item()),
                "map_inside_mean": float(((own * mask).sum() / area).item()),
                "map_outside_mean": float(((own * outside).sum() / outside_area).item()),
                "map_other_region_mean": float(((own * other).sum() / other_area).item()),
                "map_neutral_mean": float(((own * neutral).sum() / neutral_area).item()),
                "own_gt_patch_win_frac": float(win_frac.item()),
            }
            if bg_map is not None:
                bg = bg_map[b].float().clamp_min(0.0)
                record["bg_map_on_self_mean"] = float(((bg * mask).sum() / area).item())
            object_records.append(record)
    return sample_records, object_records


def _sample_segmentation_diagnosis(gt: torch.Tensor, pred: torch.Tensor, overlap: torch.Tensor | None):
    gt_eval, pred_eval = gt, pred
    if overlap is not None:
        gt_eval, pred_eval = preproc_masks_overlap(gt, pred, overlap)
    gt_np = gt_eval.detach().cpu().numpy().astype(np.int64)
    pred_np = pred_eval.detach().cpu().numpy().astype(np.int64)
    mat, gt_ids, pred_ids, _fg = _contingency(gt_np, pred_np)
    fg_pixels = int(mat.sum())
    if fg_pixels <= 0 or mat.size == 0:
        return {
            "foreground_pixels": fg_pixels,
            "n_gt": 0,
            "n_pred_on_fg": 0,
            "bg_miss_frac": float("nan"),
            "split_frac": float("nan"),
            "merge_frac": float("nan"),
            "object_dominant_frac_mean": float("nan"),
        }

    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    bg_col = pred_to_col.get(0, None)
    bg_pixels = int(mat[:, bg_col].sum()) if bg_col is not None else 0
    row_sum = mat.sum(axis=1).astype(np.float64)
    row_max = mat.max(axis=1).astype(np.float64)
    split_frac = float((row_sum - row_max).sum() / max(float(fg_pixels), 1.0))

    pred_fg_cols = [j for j, pid in enumerate(pred_ids.tolist()) if int(pid) != 0]
    merge_pixels = 0.0
    pred_fg_pixels = 0.0
    for j in pred_fg_cols:
        col = mat[:, j].astype(np.float64)
        total = float(col.sum())
        pred_fg_pixels += total
        merge_pixels += total - float(col.max())
    return {
        "foreground_pixels": fg_pixels,
        "n_gt": int(len(gt_ids)),
        "n_pred_on_fg": int(len(pred_fg_cols)),
        "bg_miss_frac": float(bg_pixels / max(fg_pixels, 1)),
        "split_frac": split_frac,
        "merge_frac": float(merge_pixels / max(pred_fg_pixels, 1.0)),
        "object_dominant_frac_mean": float(np.mean(row_max / np.maximum(row_sum, 1.0))),
    }


def _summarize_sample_records(records):
    weights = [r["foreground_pixels"] for r in records]
    return {
        "n": len(records),
        "fARI": _mean([r["fARI"] for r in records]),
        "mBO": _mean([r["mBO"] for r in records]),
        "mIoU": _mean([r["mIoU"] for r in records]),
        "bg_miss_frac": _weighted_mean([r["bg_miss_frac"] for r in records], weights),
        "split_frac": _weighted_mean([r["split_frac"] for r in records], weights),
        "merge_frac": _weighted_mean([r["merge_frac"] for r in records], weights),
        "n_gt_mean": _mean([r["n_gt"] for r in records]),
        "n_pred_on_fg_mean": _mean([r["n_pred_on_fg"] for r in records]),
        "map_pair_overlap_mean": _mean([r["map_pair_overlap_mean"] for r in records]),
        "map_ambiguous_patch_frac": _mean([r["map_ambiguous_patch_frac"] for r in records]),
        "map_top1_top2_margin_mean": _mean([r["map_top1_top2_margin_mean"] for r in records]),
    }


def _summarize_object_records(records):
    out = {}
    groups = {
        "thing": [r for r in records if r["is_thing"]],
        "stuff": [r for r in records if not r["is_thing"]],
        "all": list(records),
    }
    for name, recs in groups.items():
        out[name] = {
            "n_objects": len(recs),
            "area_frac_mean": _mean([r["area_frac"] for r in recs]),
            "self_mass_frac": _mean([r["map_self_mass_frac"] for r in recs]),
            "outside_mass_frac": _mean([r["map_outside_mass_frac"] for r in recs]),
            "other_region_mass_frac": _mean([r["map_other_region_mass_frac"] for r in recs]),
            "neutral_mass_frac": _mean([r["map_neutral_mass_frac"] for r in recs]),
            "inside_mean": _mean([r["map_inside_mean"] for r in recs]),
            "outside_mean": _mean([r["map_outside_mean"] for r in recs]),
            "other_region_mean": _mean([r["map_other_region_mean"] for r in recs]),
            "own_gt_patch_win_frac": _mean([r["own_gt_patch_win_frac"] for r in recs]),
        }
    by_size = defaultdict(list)
    for r in records:
        by_size[r["size_bucket"]].append(r)
    out["by_size"] = {
        k: {
            "n_objects": len(v),
            "self_mass_frac": _mean([r["map_self_mass_frac"] for r in v]),
            "other_region_mass_frac": _mean([r["map_other_region_mass_frac"] for r in v]),
            "own_gt_patch_win_frac": _mean([r["own_gt_patch_win_frac"] for r in v]),
            "inside_mean": _mean([r["map_inside_mean"] for r in v]),
        }
        for k, v in sorted(by_size.items())
    }
    return out


def _load_gt_for_sample(args, raw, coco_cache):
    if args.gt_source == "coco_instance":
        gt = coco_cache.get(int(raw["image_id"]))
        if gt is not None:
            return gt.to(torch.int64)
    seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
    return load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size).to(torch.int64)


def _winner_region_mask(args, obj_maps, obj_valid, source_size):
    B, K, P = obj_maps.shape
    maps_2d = obj_maps.reshape(B, K, args.grid_size, args.grid_size)
    up = F.interpolate(maps_2d, size=(source_size, source_size), mode="bilinear", align_corners=False)
    up = up.masked_fill(~obj_valid.view(B, K, 1, 1), torch.finfo(up.dtype).min)
    return (up.argmax(dim=1) + 1).to(torch.int64)


def _probability_heat_overlay(source, probability, title):
    """Overlay a true [0,1] owner probability without per-map min/max scaling."""
    src = np.asarray(source.convert("RGB")).astype(np.float32)
    prob = probability.detach().cpu().float().clamp(0.0, 1.0)
    prob_np = prob.numpy()
    color = np.zeros_like(src)
    color[..., 0] = 255.0
    color[..., 1] = 40.0
    color[..., 2] = 40.0
    strength = 0.68 * np.sqrt(prob_np)[..., None]
    out = src * (1.0 - strength) + color * strength
    stats = (
        f"{title} | p min={float(prob.min()):.3f} "
        f"mean={float(prob.mean()):.3f} max={float(prob.max()):.3f}"
    )
    return _add_title(
        Image.fromarray(out.clip(0, 255).astype(np.uint8)),
        stats,
    )


def _competition_visuals(args, out, obj_valid, source, labels):
    object_probs = out.get("e3_competition_object_probs")
    background_probs = out.get("e3_competition_background_probs")
    if (
        object_probs is None
        or background_probs is None
        or object_probs.numel() == 0
        or background_probs.numel() == 0
    ):
        return None, None

    object_probs = object_probs.float()
    background_probs = background_probs.float()
    B, K, P = object_probs.shape
    side = int(round(float(P) ** 0.5))
    all_probs = torch.cat([object_probs, background_probs], dim=1)
    valid_full = torch.cat(
        [
            obj_valid[:, :K],
            torch.ones(B, 1, device=obj_valid.device, dtype=torch.bool),
        ],
        dim=1,
    )
    all_probs = all_probs.masked_fill(
        ~valid_full.unsqueeze(-1),
        0.0,
    )
    up = F.interpolate(
        all_probs.reshape(B, K + 1, side, side),
        size=(args.eval_size, args.eval_size),
        mode="bilinear",
        align_corners=False,
    )
    up = up / up.sum(dim=1, keepdim=True).clamp_min(1e-8)

    tiles = [_add_title(source, "E3 Competition CE owner probabilities")]
    for k in obj_valid[0, :K].nonzero(as_tuple=False).flatten().tolist()[:8]:
        label = labels[k] if k < len(labels) else f"object {k + 1}"
        tiles.append(
            _probability_heat_overlay(
                source,
                up[0, k],
                f"owner: {label}",
            )
        )
    tiles.append(
        _probability_heat_overlay(
            source,
            up[0, K],
            "owner: register background",
        )
    )

    entropy = -(
        up[0].clamp_min(1e-8) * up[0].clamp_min(1e-8).log()
    ).sum(dim=0)
    entropy = entropy / max(math.log(max(K + 1, 2)), 1e-8)
    tiles.append(
        _probability_heat_overlay(
            source,
            entropy,
            "normalized owner entropy (red = ambiguous)",
        )
    )

    assign = up.argmax(dim=1)
    winner = torch.zeros(
        (B, args.eval_size, args.eval_size),
        dtype=torch.int64,
        device=up.device,
    )
    for b in range(B):
        rank = 0
        for k in range(K):
            if not bool(obj_valid[b, k]):
                continue
            rank += 1
            winner[b][assign[b] == k] = rank
    winner_overlay = _large_mask_overlay(
        source,
        winner[0].detach().cpu(),
        "E3 Competition CE owner winner (register = background)",
        labels,
        size=args.large_overlay_size,
        alpha=args.large_overlay_alpha,
    )
    return _concat_grid(tiles, cols=min(4, len(tiles))), winner_overlay


def _caption_object_labels(batch, fallback_segments):
    texts = batch.get("caption_texts") or []
    text = texts[0] if texts else ""
    labels = []
    for chunk in str(text).split("<thing>")[1:]:
        label = chunk.split("<ovt>", 1)[0].strip().rstrip(".")
        if label:
            labels.append(label)
    return labels or _segment_labels(fallback_segments)


def _make_sample_images(
    args,
    spec,
    raw,
    batch,
    out,
    pred,
    gt,
    overlap,
    samples_dir,
    sample_index,
    thing_categories,
):
    os.makedirs(samples_dir, exist_ok=True)
    source = _load_source_image(args, raw)
    safe = _safe_name(spec["label"])

    labels_all = _caption_object_labels(batch, raw["segments"])
    pred_overlay = _large_mask_overlay(
        source,
        pred.cpu(),
        f"{spec['label']} pred {spec['readout']}/{spec['merge']}",
        None,
        size=args.large_overlay_size,
        alpha=args.large_overlay_alpha,
    )
    pred_path = os.path.join(samples_dir, f"{safe}_pred_overlay.png")
    pred_overlay.save(pred_path)

    obj_maps, bg_map, map_kind = _object_maps(args, spec, out)
    obj_valid, _obj_thing = _object_valid_and_thing(
        batch,
        [raw],
        0,
        SimpleNamespace(batch_size=1, n_ovt_per_object=args.n_ovt_per_object),
        thing_categories,
        obj_maps.device,
    )
    winner = _winner_region_mask(args, obj_maps, obj_valid, args.eval_size)[0].detach().cpu()
    winner_overlay = _large_mask_overlay(
        source,
        winner,
        f"{spec['label']} all-region winner before thing-only mapping",
        labels_all,
        size=args.large_overlay_size,
        alpha=args.large_overlay_alpha,
    )
    winner_path = os.path.join(samples_dir, f"{safe}_all_region_winner.png")
    winner_overlay.save(winner_path)

    heat_tiles = [_add_title(source, f"{spec['label']} source ({map_kind})")]
    labels = labels_all
    for k in obj_valid[0].nonzero(as_tuple=False).flatten().tolist()[:8]:
        patch_side = int(round(float(obj_maps.shape[-1]) ** 0.5))
        heat = F.interpolate(
            obj_maps[0, k].reshape(1, 1, patch_side, patch_side),
            size=(args.eval_size, args.eval_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        label = labels[k] if k < len(labels) else f"object {k + 1}"
        heat_tiles.append(_heat_overlay(source, heat, label))
    if bg_map is not None and bg_map.numel() > 0:
        patch_side = int(round(float(bg_map.shape[-1]) ** 0.5))
        heat = F.interpolate(
            bg_map[0].reshape(1, 1, patch_side, patch_side),
            size=(args.eval_size, args.eval_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        heat_tiles.append(_heat_overlay(source, heat, "register mean (background)"))
    attention_path = os.path.join(samples_dir, f"{safe}_attention_heatmaps.png")
    _concat_grid(heat_tiles, cols=min(5, len(heat_tiles))).save(attention_path)

    competition_heatmaps, competition_winner = _competition_visuals(
        args,
        out,
        obj_valid,
        source,
        labels_all,
    )
    competition_heatmaps_path = None
    competition_winner_path = None
    if competition_heatmaps is not None and competition_winner is not None:
        competition_heatmaps_path = os.path.join(
            samples_dir,
            f"{safe}_competition_probability_heatmaps.png",
        )
        competition_winner_path = os.path.join(
            samples_dir,
            f"{safe}_competition_owner_winner.png",
        )
        competition_heatmaps.save(competition_heatmaps_path)
        competition_winner.save(competition_winner_path)

    gt_eval, pred_eval = gt.cpu(), pred.cpu()
    if overlap is not None:
        gt_eval, pred_eval = preproc_masks_overlap(gt_eval, pred_eval, overlap.cpu())
    mat, gt_ids, pred_ids, _fg = _contingency(gt_eval.numpy(), pred_eval.numpy())
    error = _error_overlay(source, gt_eval.numpy(), pred_eval.numpy(), mat, gt_ids, pred_ids)
    matrix = _matrix_image(mat, gt_ids, pred_ids, f"{spec['label']} FG-ARI contingency")
    error_path = os.path.join(samples_dir, f"{safe}_fari_error_overlay.png")
    matrix_path = os.path.join(samples_dir, f"{safe}_contingency_matrix.png")
    error.save(error_path)
    matrix.save(matrix_path)
    return {
        "pred_overlay": pred_path,
        "all_region_winner": winner_path,
        "attention_heatmaps": attention_path,
        "competition_probability_heatmaps": competition_heatmaps_path,
        "competition_owner_winner": competition_winner_path,
        "fari_error_overlay": error_path,
        "contingency_matrix": matrix_path,
    }


def _load_source_image(args, raw):
    source = Image.open(raw["image_path"]).convert("RGB")
    if args.image_preprocess_mode == "coda_center_crop":
        source = _coda_center_crop_image(source, args.coda_crop_size)
    return source.resize((args.eval_size, args.eval_size), Image.BILINEAR)


def _write_csv(path, records):
    if not records:
        return
    keys = sorted({k for r in records for k in r.keys()})
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for rec in records:
            vals = []
            for k in keys:
                v = rec.get(k, "")
                if isinstance(v, str):
                    vals.append('"' + v.replace('"', '""') + '"')
                else:
                    vals.append(str(v))
            f.write(",".join(vals) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/outside_failure_diagnostics")
    parser.add_argument("--gt_source", choices=["coco_instance", "pix2cap_panoptic"], default="coco_instance")
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--sample_indices", default="0,1020,1407")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--bg_threshold", type=float, default=0.05)
    parser.add_argument("--spatial_temperature", type=float, default=1.0)
    parser.add_argument("--large_overlay_size", type=int, default=640)
    parser.add_argument("--large_overlay_alpha", type=float, default=0.52)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    specs = [_parse_model_spec(s) for s in args.model]
    requested_samples = [
        int(x.strip()) for x in args.sample_indices.split(",") if x.strip()
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]

    coco_cache = None
    thing_categories = None
    if args.gt_source == "coco_instance":
        coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
        args.eval_size = coco_cache.size
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")

    run_indices = list(range(len(raw_samples)))
    if args.max_samples == 0:
        # Visualization-only mode: evaluate just --sample_indices.
        run_indices = []
    elif args.max_samples is not None and args.max_samples > 0:
        run_indices = run_indices[: min(args.max_samples, len(run_indices))]
    sample_indices = sorted({i for i in requested_samples if 0 <= i < len(raw_samples)})
    dataset_indices = sorted(set(run_indices) | set(sample_indices))

    all_model_summaries = []
    sample_image_registry = defaultdict(lambda: {"models": {}})

    for spec in specs:
        print(f"[diagnose] loading {spec['label']} from {spec['path']}")
        model, tokenizer = _load_model(spec["path"], dtype=dtype, device=device)
        checkpoint_spatial_temperature = float(
            getattr(
                model.config,
                "pgot_mask_spatial_outside_log_temperature",
                getattr(model.config, "pgot_mask_spatial_temperature", args.spatial_temperature),
            )
        )
        args.checkpoint_spatial_temperature = checkpoint_spatial_temperature
        vt_list = model.get_vision_tower_aux_list()
        dataset = Pix2CapPGOTDataset(
            jsonl_path=args.val_jsonl,
            tokenizer=tokenizer,
            image_processor=vt_list[0].image_processor,
            target_image_processor=vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor,
            grid_size=args.grid_size,
            max_caption_tokens=args.max_caption_tokens,
            n_ovt_per_object=args.n_ovt_per_object,
            max_objects=args.max_objects,
            panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
            image_preprocess_mode=args.image_preprocess_mode,
            coda_crop_size=args.coda_crop_size,
        )
        subset = torch.utils.data.Subset(dataset, dataset_indices)
        collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        model_dir = os.path.join(args.output_dir, _safe_name(spec["label"]))
        os.makedirs(model_dir, exist_ok=True)
        sample_records = []
        object_records = []

        for batch_idx, batch in enumerate(tqdm(loader, desc=spec["label"])):
            global_indices = dataset_indices[
                batch_idx * args.batch_size : batch_idx * args.batch_size + len(batch["image_ids"])
            ]
            samples_iter_for_batch = [raw_samples[i] for i in global_indices]
            with torch.no_grad():
                out = pgot_forward_eval(
                    model,
                    images=batch["images"],
                    target_images=batch["target_images"],
                    caption_input_ids=batch["caption_input_ids"],
                    caption_attention_mask=batch["caption_attention_mask"],
                    ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                    ovt_valid_mask=batch["ovt_valid_mask"],
                    return_llm_attention_maps=(spec["readout"] == "llm_attention"),
                )
                pred_mask = _build_pred(
                    args,
                    spec,
                    out,
                    batch,
                    samples_iter_for_batch,
                    0,
                    thing_categories,
                ).detach().cpu().to(torch.int64)
                map_sample_recs, map_object_recs = _map_diagnostics(
                    args,
                    spec,
                    out,
                    batch,
                    samples_iter_for_batch,
                    0,
                    thing_categories,
                    global_indices=global_indices,
                )

            map_by_index = {r["sample_index"]: r for r in map_sample_recs}
            object_records.extend(map_object_recs)

            for local_b, global_idx in enumerate(global_indices):
                raw = raw_samples[global_idx]
                gt = _load_gt_for_sample(args, raw, coco_cache)
                overlap = coco_cache.get_overlap(int(raw["image_id"])) if coco_cache is not None else None
                if overlap is not None:
                    overlap = overlap.to(torch.uint8)
                pred = pred_mask[local_b]
                scores = {
                    "fARI": float(fari_metric(gt.unsqueeze(0), pred.unsqueeze(0), overlap.unsqueeze(0) if overlap is not None else None)),
                    "mBO": float(mbo_metric(gt.unsqueeze(0), pred.unsqueeze(0), overlap.unsqueeze(0) if overlap is not None else None)),
                    "mIoU": float(miou_metric(gt.unsqueeze(0), pred.unsqueeze(0), overlap.unsqueeze(0) if overlap is not None else None)),
                }
                seg_diag = _sample_segmentation_diagnosis(gt, pred, overlap)
                rec = {
                    "sample_index": int(global_idx),
                    "image_id": int(raw["image_id"]),
                    "n_objects_caption": int(batch["n_objects_list"][local_b]),
                    **scores,
                    **seg_diag,
                    **map_by_index.get(global_idx, {}),
                }
                sample_records.append(rec)

                if global_idx in sample_indices:
                    sample_dir = os.path.join(args.output_dir, f"sample_{global_idx:04d}")
                    if "source" not in sample_image_registry[global_idx]:
                        source = _load_source_image(args, raw)
                        source_path = os.path.join(sample_dir, "source.png")
                        gt_path = os.path.join(sample_dir, "gt_overlay.png")
                        os.makedirs(sample_dir, exist_ok=True)
                        _add_title(source, f"source idx={global_idx} image_id={raw['image_id']}", height=46).save(source_path)
                        _large_mask_overlay(
                            source,
                            gt,
                            f"GT {args.gt_source}",
                            None,
                            size=args.large_overlay_size,
                            alpha=args.large_overlay_alpha,
                        ).save(gt_path)
                        sample_image_registry[global_idx]["source"] = source_path
                        sample_image_registry[global_idx]["gt_overlay"] = gt_path
                    sample_batch = {
                        k: (
                            v[local_b:local_b + 1]
                            if torch.is_tensor(v) and v.shape[0] == len(global_indices)
                            else [v[local_b]]
                            if isinstance(v, list) and len(v) == len(global_indices)
                            else v
                        )
                        for k, v in batch.items()
                    }
                    image_paths = _make_sample_images(
                        args,
                        spec,
                        raw,
                        sample_batch,
                        {k: (v[local_b:local_b + 1] if torch.is_tensor(v) and v.shape[0] == len(global_indices) else v) for k, v in out.items()},
                        pred,
                        gt,
                        overlap,
                        sample_dir,
                        global_idx,
                        thing_categories,
                    )
                    sample_image_registry[global_idx]["models"][spec["label"]] = image_paths

            del out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        by_bucket = defaultdict(list)
        for r in sample_records:
            by_bucket[_bucket(int(r["n_gt"]))].append(r)

        summary = {
            "label": spec["label"],
            "path": spec["path"],
            "readout": spec["readout"],
            "merge": spec["merge"],
            "num_eval_samples": len([r for r in sample_records if r["sample_index"] in run_indices]),
            "num_total_records_with_requested_samples": len(sample_records),
            "checkpoint_spatial_temperature": checkpoint_spatial_temperature,
            "overall": _summarize_sample_records([r for r in sample_records if r["sample_index"] in run_indices]),
            "by_gt_object_count": {
                k: _summarize_sample_records(v) for k, v in sorted(by_bucket.items())
            },
            "object_map_stats": _summarize_object_records(object_records),
            "ranked": {
                "lowest_fari": sorted(sample_records, key=lambda r: r["fARI"])[:20],
                "highest_bg_miss": sorted(sample_records, key=lambda r: r["bg_miss_frac"], reverse=True)[:20],
                "highest_split": sorted(sample_records, key=lambda r: r["split_frac"], reverse=True)[:20],
                "highest_merge": sorted(sample_records, key=lambda r: r["merge_frac"], reverse=True)[:20],
                "lowest_ownership": sorted(sample_records, key=lambda r: r.get("map_top1_top2_margin_mean", 0.0))[:20],
            },
        }
        with open(os.path.join(model_dir, "diagnostic_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(model_dir, "sample_records.json"), "w") as f:
            json.dump(sample_records, f, indent=2)
        with open(os.path.join(model_dir, "object_records.json"), "w") as f:
            json.dump(object_records, f, indent=2)
        _write_csv(os.path.join(model_dir, "sample_records.csv"), sample_records)
        _write_csv(os.path.join(model_dir, "object_records.csv"), object_records)
        all_model_summaries.append(summary)

        del model, tokenizer, dataset, subset, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for sample_idx, reg in sample_image_registry.items():
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_idx:04d}")
        pred_tiles = [Image.open(reg["source"]).convert("RGB"), Image.open(reg["gt_overlay"]).convert("RGB")]
        winner_tiles = [Image.open(reg["source"]).convert("RGB"), Image.open(reg["gt_overlay"]).convert("RGB")]
        error_tiles = [Image.open(reg["source"]).convert("RGB"), Image.open(reg["gt_overlay"]).convert("RGB")]
        for spec in specs:
            paths = reg["models"].get(spec["label"])
            if not paths:
                continue
            pred_tiles.append(Image.open(paths["pred_overlay"]).convert("RGB"))
            winner_tiles.append(Image.open(paths["all_region_winner"]).convert("RGB"))
            error_tiles.append(Image.open(paths["fari_error_overlay"]).convert("RGB"))
        _concat_grid(pred_tiles, cols=min(3, len(pred_tiles))).save(
            os.path.join(sample_dir, "comparison_pred_overlays.png")
        )
        _concat_grid(winner_tiles, cols=min(3, len(winner_tiles))).save(
            os.path.join(sample_dir, "comparison_all_region_winners.png")
        )
        _concat_grid(error_tiles, cols=min(3, len(error_tiles))).save(
            os.path.join(sample_dir, "comparison_fari_errors.png")
        )

    comparison = {
        "gt_source": args.gt_source,
        "coda_overlap_excluded": bool(coco_cache is not None and coco_cache.overlap_masks is not None),
        "max_samples": args.max_samples,
        "run_indices_count": len(run_indices),
        "sample_indices": sample_indices,
        "models": all_model_summaries,
        "sample_visualizations": sample_image_registry,
    }
    with open(os.path.join(args.output_dir, "diagnostic_comparison_summary.json"), "w") as f:
        json.dump(comparison, f, indent=2)
    print(json.dumps({
        m["label"]: {
            "fARI": m["overall"]["fARI"],
            "mBO": m["overall"]["mBO"],
            "mIoU": m["overall"]["mIoU"],
            "thing_ownership": m["object_map_stats"]["thing"]["own_gt_patch_win_frac"],
            "thing_self_mass": m["object_map_stats"]["thing"]["self_mass_frac"],
            "thing_other_mass": m["object_map_stats"]["thing"]["other_region_mass_frac"],
            "pair_overlap": m["overall"]["map_pair_overlap_mean"],
        }
        for m in all_model_summaries
    }, indent=2))
    print(os.path.join(args.output_dir, "diagnostic_comparison_summary.json"))


if __name__ == "__main__":
    main()
