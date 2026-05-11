#!/usr/bin/env python3
"""Build a contact sheet for low-mBO per-image attention failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def load_per_image_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        image_id = row.get("image_id") or image_id_from_path(row.get("image", ""))
        if image_id:
            out[str(image_id)] = row
    return out


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        raise KeyError(f"Metric '{metric}' not found in per-image row")
    return float(value)


def classify_failure(row: dict[str, Any]) -> str:
    fg = float(row.get("pred_fg_ratio", 0.0))
    strict_mbo = float(row.get("strict_MBO", row.get("MBO", 0.0)))
    strict_miou = float(row.get("strict_mIoU", row.get("mIoU", 0.0)))
    bg005_mbo = float(row.get("bg_thr0.05_MBO", strict_mbo))
    bg010_miou = float(row.get("bg_thr0.10_mIoU", strict_miou))
    loc = float(row.get("loc_acc_top16", 0.0))

    hints = []
    if fg < 0.01:
        hints.append("tiny predicted FG")
    elif fg < 0.02:
        hints.append("small predicted FG")
    elif fg > 0.08:
        hints.append("large/bleeding FG")

    if bg005_mbo - strict_mbo > 0.08:
        hints.append("background threshold helps")
    if strict_miou - strict_mbo > 0.08:
        hints.append("coarse overlap, poor best-object cover")
    if bg010_miou - strict_miou > 0.08:
        hints.append("background suppresses noise")
    if loc < 0.35:
        hints.append("weak localization")
    if not hints:
        hints.append("inspect merge/boundary")
    return "; ".join(hints)


def open_or_blank(path: Path | None, size: tuple[int, int], label: str) -> Image.Image:
    if path is not None and path.exists():
        img = Image.open(path).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - img.width) // 2
        y = (size[1] - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas
    canvas = Image.new("RGB", size, (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), label, fill=(80, 80, 80))
    return canvas


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    width_chars: int,
    fill: tuple[int, int, int] = (20, 20, 20),
) -> None:
    y = xy[1]
    for line in textwrap.wrap(text, width=width_chars)[:8]:
        draw.text((xy[0], y), line, font=font, fill=fill)
        y += 15


def write_contact_sheet(
    rows: list[dict[str, Any]],
    per_image: dict[str, dict[str, Any]],
    out_path: Path,
    image_dir: Path,
    metric: str,
    columns: int,
    thumb_size: int,
) -> None:
    font = ImageFont.load_default()
    tile_w = thumb_size * 2 + 24
    tile_h = thumb_size + 176
    rows_n = math.ceil(len(rows) / columns)
    sheet = Image.new("RGB", (tile_w * columns, tile_h * rows_n), "white")
    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(rows):
        image_id = str(row["image_id"])
        meta = per_image.get(image_id, {})
        image_path = Path(meta.get("image") or image_dir / f"{image_id}.jpg")
        overlay_path = Path(row["attn_overlay"]) if row.get("attn_overlay") else None

        x0 = (idx % columns) * tile_w
        y0 = (idx // columns) * tile_h
        orig = open_or_blank(image_path, (thumb_size, thumb_size), "missing original")
        overlay = open_or_blank(overlay_path, (thumb_size, thumb_size), "missing overlay")
        sheet.paste(orig, (x0 + 8, y0 + 8))
        sheet.paste(overlay, (x0 + thumb_size + 16, y0 + 8))

        text_y = y0 + thumb_size + 16
        lines = [
            f"{idx + 1:02d} {image_id}  {metric}={metric_value(row, metric):.4f}",
            "strict FG/MBO/mIoU="
            f"{row.get('strict_fARI', 0):.3f}/"
            f"{row.get('strict_MBO', 0):.3f}/"
            f"{row.get('strict_mIoU', 0):.3f}",
            "bg05 MBO="
            f"{row.get('bg_thr0.05_MBO', 0):.3f}  "
            "bg10 mIoU="
            f"{row.get('bg_thr0.10_mIoU', 0):.3f}",
            f"fg_ratio={row.get('pred_fg_ratio', 0):.3f}  loc={row.get('loc_acc_top16', 0):.3f}",
            classify_failure(row),
        ]
        for line in lines:
            draw.text((x0 + 8, text_y), line, font=font, fill=(20, 20, 20))
            text_y += 15

        caption = str(row.get("caption") or meta.get("caption") or "")
        draw_wrapped(draw, (x0 + 8, text_y + 2), caption, font, width_chars=58, fill=(55, 55, 55))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--metric", default="strict_MBO")
    parser.add_argument("--top-k", default=32, type=int)
    parser.add_argument("--image-dir", default=Path("/home/jovyan/data/coco/val2017"), type=Path)
    parser.add_argument("--columns", default=4, type=int)
    parser.add_argument("--thumb-size", default=224, type=int)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()

    per_attn = args.run_dir / "per_image_attn.jsonl"
    if not per_attn.exists():
        raise FileNotFoundError(f"Missing {per_attn}")

    rows = read_jsonl(per_attn)
    rows = [r for r in rows if r.get(args.metric) is not None]
    rows.sort(key=lambda r: metric_value(r, args.metric))
    selected = rows[: args.top_k]
    per_image = load_per_image_map(args.run_dir / "per_image.jsonl")

    out_dir = args.out_dir or args.run_dir / "failure_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"lowest_{args.metric}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "image_id",
                args.metric,
                "strict_fARI",
                "strict_MBO",
                "strict_mIoU",
                "bg_thr0.05_MBO",
                "bg_thr0.10_mIoU",
                "pred_fg_ratio",
                "loc_acc_top16",
                "hint",
                "attn_overlay",
                "caption",
            ]
        )
        for rank, row in enumerate(selected, 1):
            writer.writerow(
                [
                    rank,
                    row.get("image_id"),
                    row.get(args.metric),
                    row.get("strict_fARI"),
                    row.get("strict_MBO"),
                    row.get("strict_mIoU"),
                    row.get("bg_thr0.05_MBO"),
                    row.get("bg_thr0.10_mIoU"),
                    row.get("pred_fg_ratio"),
                    row.get("loc_acc_top16"),
                    classify_failure(row),
                    row.get("attn_overlay"),
                    row.get("caption"),
                ]
            )

    sheet_path = out_dir / f"lowest_{args.metric}.jpg"
    write_contact_sheet(
        selected,
        per_image,
        sheet_path,
        args.image_dir,
        args.metric,
        args.columns,
        args.thumb_size,
    )
    print(f"Wrote {sheet_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
