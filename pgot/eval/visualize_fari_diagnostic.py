"""Visualize why FG-ARI changes for one PGOT prediction.

FG-ARI evaluates clustering over foreground GT pixels. This diagnostic shows:
  - source / GT / prediction overlays
  - a foreground-only error overlay
  - the GT-object x predicted-cluster contingency matrix used by ARI
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.eval.visualize_ovt_overlays import (
    _add_title,
    _build_pred,
    _color_mask_overlay,
    _concat_grid,
    _large_mask_overlay,
    _load_font,
    _load_model,
    _palette,
)
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset


def _contingency(gt: np.ndarray, pred: np.ndarray):
    fg = gt > 0
    gt_fg = gt[fg].astype(np.int64)
    pred_fg = pred[fg].astype(np.int64)
    gt_ids = np.array(sorted(np.unique(gt_fg).tolist()), dtype=np.int64)
    pred_ids = np.array(sorted(np.unique(pred_fg).tolist()), dtype=np.int64)
    gt_to_row = {int(v): i for i, v in enumerate(gt_ids)}
    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.int64)
    for g, p in zip(gt_fg, pred_fg):
        mat[gt_to_row[int(g)], pred_to_col[int(p)]] += 1
    return mat, gt_ids, pred_ids, fg


def _ari_parts(mat: np.ndarray):
    n = mat.sum()
    rows = mat.sum(axis=1)
    cols = mat.sum(axis=0)
    rindex = float((mat * (mat - 1)).sum())
    aindex = float((rows * (rows - 1)).sum())
    bindex = float((cols * (cols - 1)).sum())
    expected = aindex * bindex / max(float(n * (n - 1)), 1.0)
    max_r = (aindex + bindex) / 2.0
    denom = max_r - expected
    ari = 1.0 if denom == 0 else (rindex - expected) / denom
    return {
        "foreground_pixels": int(n),
        "same_gt_same_pred_pairs": rindex,
        "same_gt_pairs": aindex,
        "same_pred_pairs": bindex,
        "expected_random_pairs": expected,
        "max_possible_pairs": max_r,
        "fari": ari,
    }


def _matrix_image(mat: np.ndarray, gt_ids: np.ndarray, pred_ids: np.ndarray, title: str):
    row_frac = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)
    cell = 68
    left = 110
    top = 72
    w = left + cell * len(pred_ids) + 20
    h = top + cell * len(gt_ids) + 36
    img = Image.new("RGB", (max(w, 420), max(h, 180)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = _load_font(18)
    font = _load_font(12)
    small = _load_font(10)
    draw.text((12, 14), title, fill=(0, 0, 0), font=title_font)
    draw.text((left, 46), "Pred cluster id (0 = predicted background)", fill=(30, 30, 30), font=font)
    for j, pid in enumerate(pred_ids):
        x = left + j * cell
        draw.text((x + 10, top - 20), str(int(pid)), fill=(0, 0, 0), font=font)
    for i, gid in enumerate(gt_ids):
        y = top + i * cell
        draw.text((12, y + 20), f"GT {int(gid)}", fill=(0, 0, 0), font=font)
        for j, _pid in enumerate(pred_ids):
            x = left + j * cell
            frac = float(row_frac[i, j])
            color = (
                int(255 - 210 * frac),
                int(255 - 120 * frac),
                int(255 - 35 * frac),
            )
            draw.rectangle((x, y, x + cell - 3, y + cell - 3), fill=color, outline=(210, 210, 210))
            count = int(mat[i, j])
            if count > 0:
                draw.text((x + 7, y + 16), str(count), fill=(0, 0, 0), font=font)
                draw.text((x + 7, y + 34), f"{frac*100:.0f}%", fill=(60, 60, 60), font=small)
    return img


def _error_overlay(source: Image.Image, gt: np.ndarray, pred: np.ndarray, mat: np.ndarray,
                   gt_ids: np.ndarray, pred_ids: np.ndarray, alpha: float = 0.62):
    src = np.asarray(source.convert("RGB")).astype(np.float32)
    out = src.copy()
    gt_to_row = {int(v): i for i, v in enumerate(gt_ids)}
    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    row_best = mat.argmax(axis=1) if mat.size else np.zeros((0,), dtype=np.int64)
    col_best = mat.argmax(axis=0) if mat.size else np.zeros((0,), dtype=np.int64)

    # Colors explain per-pixel symptoms of the pairwise ARI score.
    colors = {
        "good": np.array([50, 210, 90], dtype=np.float32),
        "split": np.array([255, 70, 45], dtype=np.float32),
        "merge": np.array([175, 70, 255], dtype=np.float32),
        "bg": np.array([45, 130, 255], dtype=np.float32),
    }
    fg = gt > 0
    for y, x in zip(*np.where(fg)):
        g = int(gt[y, x])
        p = int(pred[y, x])
        r = gt_to_row.get(g)
        c = pred_to_col.get(p)
        if r is None or c is None:
            continue
        if p == 0:
            key = "bg"
        elif row_best[r] != c:
            key = "split"
        elif col_best[c] != r:
            key = "merge"
        else:
            key = "good"
        out[y, x] = out[y, x] * (1.0 - alpha) + colors[key] * alpha

    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    canvas = _add_title(img, "FG-ARI error view: green ok, red split, purple merge, blue predicted bg", height=42)
    return canvas


def _pred_labels(pred_ids):
    return [("pred bg" if int(i) == 0 else f"pred {int(i)}") for i in pred_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_main_v8_3_bce1_outlog1")
    parser.add_argument("--label", default="V8.3")
    parser.add_argument("--readout", choices=["threshold", "spatial", "competition", "nullbg"], default="threshold")
    parser.add_argument("--merge", choices=["mean", "max"], default="mean")
    parser.add_argument("--sample_index", type=int, default=1020)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/fari_diagnostic_v83_sample1020")
    parser.add_argument("--gt_source", choices=["coco_instance", "pix2cap_panoptic"], default="coco_instance")
    parser.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
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
    raw = raw_samples[args.sample_index]
    source = Image.open(raw["image_path"]).convert("RGB").resize((args.eval_size, args.eval_size), Image.BILINEAR)

    if args.gt_source == "coco_instance":
        cache = CocoInstanceMaskCache(args.coco_mask_cache)
        gt = cache.get(int(raw["image_id"]))
        if gt is None:
            seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
            gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
        else:
            args.eval_size = cache.size
            source = source.resize((args.eval_size, args.eval_size), Image.BILINEAR)
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
    else:
        seg_ids = [int(s["segment_id"]) for s in raw["segments"]]
        gt = load_gt_panoptic_mask(raw["panoptic_mask_path"], seg_ids, args.eval_size)
        thing_categories = None

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
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    batch = collator([dataset[args.sample_index]])
    with torch.no_grad():
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
            return_llm_qk_maps=False,
        )
        spec = {"label": args.label, "path": args.model_path, "readout": args.readout, "merge": args.merge}
        pred = _build_pred(args, spec, out, batch, raw_samples, thing_categories)[0].detach().cpu()

    gt_t = gt.detach().cpu().to(torch.int64)
    pred_t = pred.to(torch.int64)
    gt_np = gt_t.numpy()
    pred_np = pred_t.numpy()
    mat, gt_ids, pred_ids, fg = _contingency(gt_np, pred_np)
    parts = _ari_parts(mat)
    scores = {
        "fARI": float(fari_metric(gt_t.unsqueeze(0), pred_t.unsqueeze(0))),
        "mBO": float(mbo_metric(gt_t.unsqueeze(0), pred_t.unsqueeze(0))),
        "mIoU": float(miou_metric(gt_t.unsqueeze(0), pred_t.unsqueeze(0))),
    }

    gt_overlay = _large_mask_overlay(source, gt_t, f"GT foreground ({args.gt_source})", None, size=512)
    pred_overlay = _large_mask_overlay(source, pred_t, f"{args.label} pred ({args.readout}/{args.merge})", None, size=512)
    pred_fg = pred_np.copy()
    pred_fg[~fg] = 0
    pred_fg_overlay = _large_mask_overlay(
        source,
        torch.from_numpy(pred_fg),
        "Prediction viewed only on GT foreground pixels",
        _pred_labels(pred_ids),
        size=512,
    )
    error = _error_overlay(source, gt_np, pred_np, mat, gt_ids, pred_ids)
    matrix = _matrix_image(mat, gt_ids, pred_ids, f"Contingency matrix for FG-ARI: fARI={scores['fARI']:.3f}")

    gt_overlay.save(os.path.join(args.output_dir, "01_gt_foreground_overlay.png"))
    pred_overlay.save(os.path.join(args.output_dir, "02_pred_overlay.png"))
    pred_fg_overlay.save(os.path.join(args.output_dir, "03_pred_on_gt_foreground.png"))
    error.save(os.path.join(args.output_dir, "04_fari_error_overlay.png"))
    matrix.save(os.path.join(args.output_dir, "05_gt_pred_contingency_matrix.png"))

    # Compact dashboard.
    dashboard = _concat_grid(
        [
            _add_title(source, f"source idx={args.sample_index}", height=42),
            gt_overlay.resize((512, gt_overlay.height * 512 // gt_overlay.width)),
            pred_overlay.resize((512, pred_overlay.height * 512 // pred_overlay.width)),
            error.resize((512, error.height * 512 // error.width)),
            matrix,
        ],
        cols=2,
    )
    dashboard.save(os.path.join(args.output_dir, "fari_diagnostic_dashboard.png"))

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "sample_index": args.sample_index,
                "image_id": raw["image_id"],
                "model_path": args.model_path,
                "readout": args.readout,
                "merge": args.merge,
                "scores": scores,
                "ari_parts": parts,
                "gt_ids": gt_ids.tolist(),
                "pred_ids_on_gt_foreground": pred_ids.tolist(),
                "contingency": mat.tolist(),
            },
            f,
            indent=2,
        )
    print(os.path.join(args.output_dir, "fari_diagnostic_dashboard.png"))
    print(json.dumps({"scores": scores, "ari_parts": parts}, indent=2))


if __name__ == "__main__":
    main()
