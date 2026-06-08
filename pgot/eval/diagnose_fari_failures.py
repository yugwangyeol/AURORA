"""Dataset-level FG-ARI failure diagnosis for PGOT predictions.

This script keeps the segmentation readout identical to run_eval.py, then
summarizes why FG-ARI is low:
  - GT foreground predicted as background
  - one GT object split across multiple predicted clusters
  - several GT objects merged into one predicted cluster
"""

import argparse
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import (
    build_pred_mask_spatial_readout,
    fari_metric,
    mbo_metric,
    miou_metric,
    ovt_logits_to_pred_mask,
)
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.eval.visualize_ovt_overlays import _load_model
from pgot.model.pgot_utils import build_pred_mask_competition_eval, build_pred_mask_null_bg_eval
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _contingency(gt: np.ndarray, pred: np.ndarray):
    fg = gt > 0
    gt_fg = gt[fg].astype(np.int64)
    pred_fg = pred[fg].astype(np.int64)
    gt_ids = np.array(sorted(np.unique(gt_fg).tolist()), dtype=np.int64)
    pred_ids = np.array(sorted(np.unique(pred_fg).tolist()), dtype=np.int64)
    mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    gt_to_row = {int(v): i for i, v in enumerate(gt_ids)}
    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    for g, p in zip(gt_fg, pred_fg):
        mat[gt_to_row[int(g)], pred_to_col[int(p)]] += 1
    return mat, gt_ids, pred_ids


def _sample_diagnosis(gt: torch.Tensor, pred: torch.Tensor):
    gt_np = gt.detach().cpu().numpy().astype(np.int64)
    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    mat, gt_ids, pred_ids = _contingency(gt_np, pred_np)
    fg_pixels = int(mat.sum())
    if fg_pixels <= 0 or mat.size == 0:
        return {
            "foreground_pixels": fg_pixels,
            "n_gt": 0,
            "n_pred_on_fg": 0,
            "bg_miss_frac": float("nan"),
            "split_frac": float("nan"),
            "merge_frac": float("nan"),
            "object_bg_frac_mean": float("nan"),
            "object_dominant_frac_mean": float("nan"),
            "small_object_bg_frac_mean": float("nan"),
            "small_object_count": 0,
        }

    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    bg_col = pred_to_col.get(0, None)
    bg_pixels = int(mat[:, bg_col].sum()) if bg_col is not None else 0
    bg_miss_frac = bg_pixels / max(fg_pixels, 1)

    row_sum = mat.sum(axis=1).astype(np.float64)
    row_max = mat.max(axis=1).astype(np.float64)
    row_bg = mat[:, bg_col].astype(np.float64) if bg_col is not None else np.zeros_like(row_sum)
    object_bg_frac = row_bg / np.maximum(row_sum, 1.0)
    object_dominant_frac = row_max / np.maximum(row_sum, 1.0)

    # Pixel-weighted fraction of GT foreground not assigned to each GT object's
    # dominant predicted cluster. This includes background if background is not
    # the dominant cluster; if background dominates, the object is effectively missed.
    split_pixels = float((row_sum - row_max).sum())
    split_frac = split_pixels / max(float(fg_pixels), 1.0)

    pred_fg_cols = [j for j, pid in enumerate(pred_ids.tolist()) if int(pid) != 0]
    merge_pixels = 0.0
    pred_fg_pixels = 0.0
    for j in pred_fg_cols:
        col = mat[:, j].astype(np.float64)
        total = float(col.sum())
        if total <= 0:
            continue
        pred_fg_pixels += total
        merge_pixels += total - float(col.max())
    merge_frac = merge_pixels / max(pred_fg_pixels, 1.0)

    small = row_sum <= np.percentile(row_sum, 25) if len(row_sum) > 0 else np.zeros_like(row_sum, dtype=bool)
    small_bg = object_bg_frac[small] if small.any() else np.array([], dtype=np.float64)

    return {
        "foreground_pixels": fg_pixels,
        "n_gt": int(len(gt_ids)),
        "n_pred_on_fg": int(len(pred_fg_cols)),
        "bg_miss_frac": float(bg_miss_frac),
        "split_frac": float(split_frac),
        "merge_frac": float(merge_frac),
        "object_bg_frac_mean": float(np.mean(object_bg_frac)),
        "object_dominant_frac_mean": float(np.mean(object_dominant_frac)),
        "small_object_bg_frac_mean": float(np.mean(small_bg)) if small_bg.size else float("nan"),
        "small_object_count": int(small.sum()),
    }


def _mean(xs):
    xs = [float(x) for x in xs if np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _weighted_mean(vals, weights):
    pairs = [(float(v), float(w)) for v, w in zip(vals, weights) if np.isfinite(float(v)) and float(w) > 0]
    if not pairs:
        return float("nan")
    num = sum(v * w for v, w in pairs)
    den = sum(w for _, w in pairs)
    return float(num / den)


def _bucket(n_obj: int):
    if n_obj <= 3:
        return "1-3"
    if n_obj <= 6:
        return "4-6"
    if n_obj <= 10:
        return "7-10"
    return "11+"


def _build_pred_batch(args, spec, out, batch, samples_iter, batch_idx, thing_categories):
    valid_for_pred = out["ovt_valid_mask"].clone()
    ovt_is_thing = batch.get("ovt_is_thing")
    if ovt_is_thing is not None:
        ovt_is_thing = ovt_is_thing.to(valid_for_pred.device, dtype=torch.bool)
    else:
        ovt_is_thing = torch.zeros_like(valid_for_pred, dtype=torch.bool)

    if thing_categories is not None:
        for b in range(valid_for_pred.shape[0]):
            global_idx = batch_idx * args.batch_size + b
            segs = samples_iter[global_idx]["segments"]
            for k, seg in enumerate(segs):
                s = k * args.n_ovt_per_object
                e = s + args.n_ovt_per_object
                if e <= valid_for_pred.shape[1] and seg["category"] in thing_categories:
                    ovt_is_thing[b, s:e] = True
    elif batch.get("ovt_is_thing") is None:
        ovt_is_thing = valid_for_pred.clone()

    readout = spec["readout"]
    if readout == "threshold":
        valid_thing_only = valid_for_pred & ovt_is_thing if args.gt_source == "coco_instance" else valid_for_pred
        return ovt_logits_to_pred_mask(
            ovt_logits=out["ovt_logits"],
            ovt_valid_mask=valid_thing_only,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            bg_threshold=args.bg_threshold,
        )
    if readout == "nullbg":
        return build_pred_mask_null_bg_eval(
            ovt_logits=out["ovt_logits"],
            null_bg_logits=out["null_bg_logits"],
            ovt_valid_mask=valid_for_pred,
            ovt_is_thing=ovt_is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spec["merge"],
        )
    if readout in {"spatial", "spatial_trainmatch"}:
        if readout == "spatial_trainmatch":
            spatial_temp = float(
                getattr(
                    args,
                    "checkpoint_spatial_temperature",
                    getattr(args, "spatial_temperature", 1.0),
                )
            )
            spatial_merge = "mean"
        else:
            spatial_temp = args.spatial_temperature
            spatial_merge = spec["merge"]
        return build_pred_mask_spatial_readout(
            ovt_logits=out["ovt_logits"],
            ovt_valid_mask=valid_for_pred,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=spatial_merge,
            temp=spatial_temp,
            ovt_is_thing=ovt_is_thing,
            map_stuff_to_bg=(args.gt_source == "coco_instance"),
        )
    return build_pred_mask_competition_eval(
        ovt_logits=out["ovt_logits"],
        reg_logits=out["reg_logits"],
        ovt_valid_mask=valid_for_pred,
        ovt_is_thing=ovt_is_thing,
        target_size=args.eval_size,
        n_ovt_per_object=args.n_ovt_per_object,
        patch_grid=args.grid_size,
        merge=spec["merge"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_main_v8_3_bce1_outlog1")
    parser.add_argument("--label", default="V8.3")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/fari_failure_diagnosis_v83")
    parser.add_argument("--gt_source", choices=["coco_instance", "pix2cap_panoptic"], default="coco_instance")
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    parser.add_argument("--readout", choices=["threshold", "spatial", "competition", "nullbg", "spatial_trainmatch"], default="threshold")
    parser.add_argument("--merge", choices=["mean", "max"], default="mean")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--bg_threshold", type=float, default=0.05)
    parser.add_argument("--spatial_temperature", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    with open(args.val_jsonl) as f:
        raw_samples = [json.loads(line) for line in f]

    model, tokenizer = _load_model(args.model_path, dtype=dtype, device=device)
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
    )
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    samples_iter = dataset.dataset.samples if isinstance(dataset, torch.utils.data.Subset) else dataset.samples

    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    coco_cache = None
    thing_categories = None
    if args.gt_source == "coco_instance":
        coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
        args.eval_size = coco_cache.size
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")

    records = []
    buckets = defaultdict(list)
    spec = {"label": args.label, "path": args.model_path, "readout": args.readout, "merge": args.merge}

    for batch_idx, batch in enumerate(tqdm(loader, desc="Diagnose")):
        with torch.no_grad():
            out = pgot_forward_eval(
                model,
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                ovt_valid_mask=batch["ovt_valid_mask"],
            )
            pred_mask = _build_pred_batch(
                args,
                spec,
                out,
                batch,
                samples_iter,
                batch_idx,
                thing_categories,
            ).detach().cpu().to(torch.int64)

        for b in range(pred_mask.shape[0]):
            global_idx = batch_idx * args.batch_size + b
            raw = samples_iter[global_idx]
            if args.gt_source == "coco_instance":
                gt = coco_cache.get(int(raw["image_id"]))
                if gt is None:
                    seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
                    gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
            else:
                seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
                gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
            gt = gt.detach().cpu().to(torch.int64)
            pred = pred_mask[b]
            scores = {
                "fARI": float(fari_metric(gt.unsqueeze(0), pred.unsqueeze(0))),
                "mBO": float(mbo_metric(gt.unsqueeze(0), pred.unsqueeze(0))),
                "mIoU": float(miou_metric(gt.unsqueeze(0), pred.unsqueeze(0))),
            }
            diag = _sample_diagnosis(gt, pred)
            rec = {
                "sample_index": int(global_idx),
                "image_id": int(raw["image_id"]),
                "n_objects_caption": int(batch["n_objects_list"][b]),
                **scores,
                **diag,
            }
            records.append(rec)
            buckets[_bucket(rec["n_gt"])].append(rec)

    def summarize(recs):
        weights = [r["foreground_pixels"] for r in recs]
        return {
            "n": len(recs),
            "fARI": _mean([r["fARI"] for r in recs]),
            "mBO": _mean([r["mBO"] for r in recs]),
            "mIoU": _mean([r["mIoU"] for r in recs]),
            "bg_miss_frac_mean": _mean([r["bg_miss_frac"] for r in recs]),
            "bg_miss_frac_pixel_weighted": _weighted_mean([r["bg_miss_frac"] for r in recs], weights),
            "split_frac_mean": _mean([r["split_frac"] for r in recs]),
            "split_frac_pixel_weighted": _weighted_mean([r["split_frac"] for r in recs], weights),
            "merge_frac_mean": _mean([r["merge_frac"] for r in recs]),
            "merge_frac_pixel_weighted": _weighted_mean([r["merge_frac"] for r in recs], weights),
            "object_bg_frac_mean": _mean([r["object_bg_frac_mean"] for r in recs]),
            "small_object_bg_frac_mean": _mean([r["small_object_bg_frac_mean"] for r in recs]),
            "n_gt_mean": _mean([r["n_gt"] for r in recs]),
            "n_pred_on_fg_mean": _mean([r["n_pred_on_fg"] for r in recs]),
        }

    summary = {
        "model_path": args.model_path,
        "label": args.label,
        "readout": args.readout,
        "merge": args.merge,
        "gt_source": args.gt_source,
        "overall": summarize(records),
        "by_gt_object_count": {k: summarize(v) for k, v in sorted(buckets.items())},
        "ranked": {
            "lowest_fari": sorted(records, key=lambda r: r["fARI"])[:20],
            "highest_bg_miss": sorted(records, key=lambda r: r["bg_miss_frac"], reverse=True)[:20],
            "highest_split": sorted(records, key=lambda r: r["split_frac"], reverse=True)[:20],
            "highest_merge": sorted(records, key=lambda r: r["merge_frac"], reverse=True)[:20],
            "low_fari_high_miou": sorted(
                [r for r in records if r["mIoU"] >= 0.35],
                key=lambda r: (r["fARI"], -r["mIoU"]),
            )[:20],
        },
    }
    with open(os.path.join(args.output_dir, "diagnosis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "sample_records.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(json.dumps(summary["overall"], indent=2))
    print(os.path.join(args.output_dir, "diagnosis_summary.json"))


if __name__ == "__main__":
    main()
