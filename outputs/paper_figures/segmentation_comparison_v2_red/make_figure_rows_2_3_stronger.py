"""Render rows 2 and 3 with stronger segmentation overlays."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = ROOT / "segmentation_comparison_rows_2_3_stronger"
ALPHA = 0.84
COLUMNS = [
    ("image", "Image"),
    ("gt", "GT"),
    ("adaslot", "AdaSlot"),
    ("metaslot", "MetaSlot"),
    ("coda", "CODA"),
    ("ours", "DeOR (Ours)"),
]
ROWS = [
    (
        "COCO",
        "coco_under",
        Path("/home/jovyan/PGOT/outputs/paper_figures/undersegmentation_v1/coco_under"),
    ),
    (
        "VOC",
        "voc_over",
        Path("/home/jovyan/PGOT/outputs/paper_figures/oversegmentation_v1/voc_over"),
    ),
]


def get_font(size, bold=False):
    suffix = "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}", size
    )


def center_text(draw, text, box, text_font):
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (left + (width - (x1 - x0)) / 2 - x0,
         top + (height - (y1 - y0)) / 2 - y0),
        text,
        font=text_font,
        fill="black",
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    Image.init()
    panel = 512
    panel_gap = 8
    row_gap = 28
    outer = 16
    dataset_width = 208
    header = 88
    grid_left = outer + dataset_width
    width = grid_left + len(COLUMNS) * panel + (len(COLUMNS) - 1) * panel_gap + outer
    height = outer + header + 2 * panel + row_gap + outer

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(46, bold=True)
    dataset_font = get_font(48, bold=True)

    for column_index, (_, title) in enumerate(COLUMNS):
        x = grid_left + column_index * (panel + panel_gap)
        center_text(draw, title, (x, outer, panel, header), title_font)

    manifest_rows = []
    for row_index, (dataset, group, source_dir) in enumerate(ROWS):
        y = outer + header + row_index * (panel + row_gap)
        center_text(draw, dataset, (outer, y, dataset_width, panel), dataset_font)

        image_path = source_dir / "image.png"
        with Image.open(image_path) as source_image:
            source_image = source_image.convert("RGB")
            if source_image.size != (panel, panel):
                raise ValueError(f"Unexpected source size {source_image.size}: {image_path}")

        record = {
            "dataset": dataset,
            "group": group,
            "overlay_alpha": ALPHA,
            "panels": [],
        }
        for column_index, (key, title) in enumerate(COLUMNS):
            x = grid_left + column_index * (panel + panel_gap)
            if key == "image":
                rendered = source_image
                map_path = None
            else:
                map_path = source_dir / f"{key}_map.png"
                with Image.open(map_path) as color_map:
                    color_map = color_map.convert("RGB")
                    if color_map.size != (panel, panel):
                        raise ValueError(f"Unexpected map size {color_map.size}: {map_path}")
                    rendered = Image.blend(source_image, color_map, ALPHA)
            canvas.paste(rendered, (x, y))
            record["panels"].append(
                {
                    "title": title,
                    "image_source": str(image_path),
                    "image_sha256": sha256(image_path),
                    "map_source": str(map_path) if map_path else None,
                    "map_sha256": sha256(map_path) if map_path else None,
                    "box": [x, y, x + panel, y + panel],
                }
            )
        manifest_rows.append(record)

    divider_y = outer + header + panel + row_gap // 2
    draw.line((outer, divider_y, width - outer, divider_y), fill=(188, 192, 198), width=2)

    png = OUTPUT_STEM.with_suffix(".png")
    pdf = OUTPUT_STEM.with_suffix(".pdf")
    preview = ROOT / "segmentation_comparison_rows_2_3_stronger_preview.png"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, resolution=300)
    preview_width = 1800
    canvas.resize(
        (preview_width, round(height * preview_width / width)), Image.Resampling.LANCZOS
    ).save(preview)

    manifest = {
        "layout": "Original rows 2 and 3 only; 2 rows x 6 columns",
        "revision": "stronger segmentation color overlays and DeOR (Ours) header",
        "overlay_alpha": ALPHA,
        "previous_overlay_alpha": 0.65,
        "prediction_labels_modified": False,
        "display_maps_modified": False,
        "rows": manifest_rows,
    }
    (ROOT / "segmentation_comparison_rows_2_3_stronger_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
