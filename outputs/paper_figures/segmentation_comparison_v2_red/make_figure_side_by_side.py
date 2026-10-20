"""Arrange the saved COCO and VOC segmentation blocks side by side.

The source panels are reused without changing masks, colors, or opacity.
"""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "manifest.json"
OUTPUT_PNG = ROOT / "segmentation_comparison_side_by_side.png"
OUTPUT_PDF = ROOT / "segmentation_comparison_side_by_side.pdf"
OUTPUT_PREVIEW = ROOT / "segmentation_comparison_side_by_side_preview.png"
OUTPUT_MANIFEST = ROOT / "segmentation_comparison_side_by_side_manifest.json"

COLUMNS = ["Image", "GT", "AdaSlot", "MetaSlot", "CODA", "DeOR"]
DATASETS = [("COCO", ["coco_over", "coco_under"]),
            ("VOC", ["voc_over", "voc_under"])]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}", size
    )


def centered(draw: ImageDraw.ImageDraw, text: str, box, text_font) -> None:
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=text_font)
    draw.text(
        (left + (width - (x1 - x0)) / 2 - x0,
         top + (height - (y1 - y0)) / 2 - y0),
        text,
        font=text_font,
        fill="black",
    )


def main() -> None:
    source = json.loads(SOURCE_MANIFEST.read_text())
    rows = {row["group"]: row for row in source["rows"]}

    panel = 512
    panel_gap = 8
    row_gap = 12
    block_gap = 72
    outer = 16
    dataset_header = 72
    column_header = 76
    block_width = len(COLUMNS) * panel + (len(COLUMNS) - 1) * panel_gap
    content_height = 2 * panel + row_gap
    width = 2 * outer + 2 * block_width + block_gap
    height = 2 * outer + dataset_header + column_header + content_height

    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    dataset_font = font(50, bold=True)
    column_font = font(39, bold=True)

    output_rows = []
    for dataset_index, (dataset_name, groups) in enumerate(DATASETS):
        block_x = outer + dataset_index * (block_width + block_gap)
        centered(
            draw,
            dataset_name,
            (block_x, outer, block_width, dataset_header),
            dataset_font,
        )

        for column_index, title in enumerate(COLUMNS):
            x = block_x + column_index * (panel + panel_gap)
            centered(
                draw,
                title,
                (x, outer + dataset_header, panel, column_header),
                column_font,
            )

        for row_index, group in enumerate(groups):
            source_row = rows[group]
            y = outer + dataset_header + column_header + row_index * (panel + row_gap)
            output_row = {
                "dataset": dataset_name,
                "group": group,
                "image_id": source_row["image_id"],
                "panels": [],
            }
            source_panels = {entry["title"]: entry for entry in source_row["panels"]}
            for column_index, title in enumerate(COLUMNS):
                source_path = Path(source_panels[title]["source"])
                x = block_x + column_index * (panel + panel_gap)
                with Image.open(source_path) as image:
                    image = image.convert("RGB")
                    if image.size != (panel, panel):
                        raise ValueError(f"Unexpected panel size {image.size}: {source_path}")
                    figure.paste(image, (x, y))
                output_row["panels"].append(
                    {
                        "title": title,
                        "source": str(source_path),
                        "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "box": [x, y, x + panel, y + panel],
                    }
                )
            output_rows.append(output_row)

    divider_x = outer + block_width + block_gap // 2
    draw.line(
        (divider_x, outer, divider_x, height - outer),
        fill=(188, 192, 198),
        width=2,
    )

    figure.save(OUTPUT_PNG, dpi=(300, 300))
    figure.save(OUTPUT_PDF, resolution=300)
    preview_width = 2400
    preview_height = round(height * preview_width / width)
    figure.resize(
        (preview_width, preview_height), Image.Resampling.LANCZOS
    ).save(OUTPUT_PREVIEW)

    output_manifest = {
        "layout": "COCO and VOC side by side; each dataset is 2 rows x 6 columns",
        "column_titles": COLUMNS,
        "native_panel_size": panel,
        "canvas_size": [width, height],
        "prediction_panels_modified": False,
        "source_manifest": str(SOURCE_MANIFEST),
        "overlay_alpha": source.get("overlay_alpha"),
        "background_color_rgb": source.get("background_color_rgb"),
        "rows": output_rows,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(output_manifest, indent=2))
    print(OUTPUT_PNG)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
