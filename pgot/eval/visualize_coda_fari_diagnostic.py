"""CODA foreground-ARI diagnostic on PGOT validation sample indices.

CODA produces an unsupervised slot partition, so predicted label 0 is a real
slot, not background. For visualization we shift slots to 1..N and reserve 0
only for display background outside GT foreground.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/jovyan/PGOT")
sys.path.insert(0, "/home/jovyan/coda")

from pgot.eval.visualize_fari_diagnostic import _ari_parts, _contingency  # noqa: E402
from pgot.eval.visualize_ovt_overlays import (  # noqa: E402
    _add_title,
    _concat_grid,
    _large_mask_overlay,
    _load_font,
)


def _patch_coda_imports():
    import diffusers
    import torch.hub

    try:
        torch.hub._validate_not_a_forked_repo = lambda *args, **kwargs: None
    except Exception:
        pass

    if not hasattr(diffusers, "StableDiffusion3Pipeline"):
        class _DummySD3:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise RuntimeError("StableDiffusion3Pipeline dummy should not be used.")

        diffusers.StableDiffusion3Pipeline = _DummySD3


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
    draw.text((left, 46), "Pred slot id (CODA has no explicit predicted background)", fill=(30, 30, 30), font=font)
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


def _error_overlay_no_bg(source: Image.Image, gt: np.ndarray, pred: np.ndarray, mat: np.ndarray,
                         gt_ids: np.ndarray, pred_ids: np.ndarray, alpha: float = 0.62):
    src = np.asarray(source.convert("RGB")).astype(np.float32)
    out = src.copy()
    gt_to_row = {int(v): i for i, v in enumerate(gt_ids)}
    pred_to_col = {int(v): i for i, v in enumerate(pred_ids)}
    row_best = mat.argmax(axis=1) if mat.size else np.zeros((0,), dtype=np.int64)
    col_best = mat.argmax(axis=0) if mat.size else np.zeros((0,), dtype=np.int64)

    colors = {
        "good": np.array([50, 210, 90], dtype=np.float32),
        "split": np.array([255, 70, 45], dtype=np.float32),
        "merge": np.array([175, 70, 255], dtype=np.float32),
    }
    fg = gt > 0
    for y, x in zip(*np.where(fg)):
        g = int(gt[y, x])
        p = int(pred[y, x])
        r = gt_to_row.get(g)
        c = pred_to_col.get(p)
        if r is None or c is None:
            continue
        if row_best[r] != c:
            key = "split"
        elif col_best[c] != r:
            key = "merge"
        else:
            key = "good"
        out[y, x] = out[y, x] * (1.0 - alpha) + colors[key] * alpha

    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    return _add_title(img, "FG-ARI error view: green ok, red split, purple merge", height=42)


def _to_source_image(image: torch.Tensor) -> Image.Image:
    arr = ((image.detach().cpu() * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _shift_pred_for_display(pred: torch.Tensor) -> torch.Tensor:
    return pred.to(torch.int64) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_index", type=int, required=True)
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--coda_root", default="/home/jovyan/coda")
    parser.add_argument("--coda_model_path", default="/home/jovyan/coda/pretrained/coda/coco")
    parser.add_argument("--coco_root", default="/home/jovyan/data/coco")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/coda_fari_diagnostic")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sample_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    _patch_coda_imports()

    from experiment.dataset.coco import COCO2017Dataset, COCOCollater
    from src.metric.segmentation import fARI_metric, mbo_metric, miou_metric, preproc_masks_overlap
    from src.model.encoder import RegisterSlotDiffusion

    with open(args.val_jsonl) as f:
        pgot_samples = [json.loads(line) for line in f]
    raw = pgot_samples[args.sample_index]
    image_id = int(raw["image_id"])

    dataset = COCO2017Dataset(
        args.coco_root,
        split="val",
        img_size=args.img_size,
        load_annotation=True,
        load_sem_mask=True,
    )
    fallback_image_dir = os.path.join(args.coco_root, "val2017")
    if not os.path.isdir(dataset.image_dir) and os.path.isdir(fallback_image_dir):
        dataset.image_dir = fallback_image_dir

    image_id_to_idx = {int(v): i for i, v in enumerate(dataset.image_ids)}
    if image_id not in image_id_to_idx:
        raise ValueError(f"image_id={image_id} from PGOT val sample {args.sample_index} not found in CODA COCO val.")
    coda_idx = image_id_to_idx[image_id]

    batch = COCOCollater()([dataset[coda_idx]])
    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    encoder = RegisterSlotDiffusion.from_pretrained(
        os.path.join(args.coda_model_path, "encoder"),
        low_cpu_mem_usage=False,
    )
    encoder.eval().to(device)

    image = batch["image"].to(device)
    with torch.no_grad():
        out = encoder(image)
        attn = rearrange(out["attn"], "b 1 (h w) n -> b n h w", h=args.sample_size, w=args.sample_size)
        up = F.interpolate(attn, size=batch["mask"].shape[-2:], mode="bilinear", align_corners=False)
        pred_raw = up.argmax(dim=1).detach().cpu().to(torch.int64)[0]

    gt_raw = batch["mask"][0].to(torch.int64)
    overlap = batch.get("inst_overlap_mask")
    overlap_raw = overlap[0].to(torch.int64) if overlap is not None else None
    gt_eval, pred_eval_raw = preproc_masks_overlap(gt_raw, pred_raw, overlap_raw)
    pred_display = _shift_pred_for_display(pred_eval_raw)

    scores = {
        "fARI": float(fARI_metric(gt_raw.unsqueeze(0), pred_raw.unsqueeze(0), overlap)),
        "mBO": float(mbo_metric(gt_raw.unsqueeze(0), pred_raw.unsqueeze(0), overlap)),
        "mIoU": float(miou_metric(gt_raw.unsqueeze(0), pred_raw.unsqueeze(0), overlap)),
    }

    source = _to_source_image(batch["image"][0])
    gt_np = gt_eval.detach().cpu().numpy().astype(np.int64)
    pred_np = pred_display.detach().cpu().numpy().astype(np.int64)
    mat, gt_ids, pred_ids, fg = _contingency(gt_np, pred_np)
    parts = _ari_parts(mat)

    gt_overlay = _large_mask_overlay(source, gt_eval, "GT foreground (CODA COCO instance, overlap removed)", None, size=512)
    pred_overlay = _large_mask_overlay(source, pred_display, "CODA pred slots (full image partition)", None, size=512)
    pred_fg = pred_np.copy()
    pred_fg[~fg] = 0
    pred_fg_overlay = _large_mask_overlay(
        source,
        torch.from_numpy(pred_fg),
        "CODA slots viewed only on GT foreground pixels",
        [f"slot {i}" for i in pred_ids.tolist()],
        size=512,
    )
    error = _error_overlay_no_bg(source, gt_np, pred_np, mat, gt_ids, pred_ids)
    matrix = _matrix_image(mat, gt_ids, pred_ids, f"CODA contingency matrix for FG-ARI: fARI={scores['fARI']:.3f}")

    gt_overlay.save(os.path.join(args.output_dir, "01_gt_foreground_overlay.png"))
    pred_overlay.save(os.path.join(args.output_dir, "02_pred_overlay.png"))
    pred_fg_overlay.save(os.path.join(args.output_dir, "03_pred_on_gt_foreground.png"))
    error.save(os.path.join(args.output_dir, "04_fari_error_overlay.png"))
    matrix.save(os.path.join(args.output_dir, "05_gt_pred_contingency_matrix.png"))

    dashboard = _concat_grid(
        [
            _add_title(source, f"source PGOT idx={args.sample_index}, COCO image_id={image_id}", height=42),
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
                "image_id": image_id,
                "coda_idx": int(coda_idx),
                "model_path": args.coda_model_path,
                "scores": scores,
                "ari_parts_from_shifted_display": parts,
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
