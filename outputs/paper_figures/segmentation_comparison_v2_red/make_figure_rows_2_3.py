"""Compose only rows 2 and 3 of the original segmentation figure."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "manifest.json"
OUTPUT_STEM = ROOT / "segmentation_comparison_rows_2_3"
COLUMNS = ["Image", "GT", "AdaSlot", "MetaSlot", "CODA", "DeOR"]
SELECTED_ROWS = [("COCO", "coco_under"), ("VOC", "voc_over")]


def get_font(size: int, bold: bool = False):
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


def main():
    source = json.loads(SOURCE_MANIFEST.read_text())
    rows_by_group = {row["group"]: row for row in source["rows"]}

    panel = 512
    panel_gap = 8
    row_gap = 28
    outer = 16
    dataset_width = 208
    header = 88
    grid_left = outer + dataset_width
    width = grid_left + len(COLUMNS) * panel + (len(COLUMNS) - 1) * panel_gap + outer
    height = outer + header + 2 * panel + row_gap + outer

    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    title_font = get_font(48, bold=True)
    dataset_font = get_font(48, bold=True)

    for column_index, title in enumerate(COLUMNS):
        x = grid_left + column_index * (panel + panel_gap)
        center_text(draw, title, (x, outer, panel, header), title_font)

    output_rows = []
    for row_index, (dataset, group) in enumerate(SELECTED_ROWS):
        y = outer + header + row_index * (panel + row_gap)
        center_text(draw, dataset, (outer, y, dataset_width, panel), dataset_font)
        source_row = rows_by_group[group]
        source_panels = {panel_data["title"]: panel_data for panel_data in source_row["panels"]}
        output_row = {
            "dataset": dataset,
            "group": group,
            "image_id": source_row["image_id"],
            "panels": [],
        }
        for column_index, title in enumerate(COLUMNS):
            source_path = Path(source_panels[title]["source"])
            x = grid_left + column_index * (panel + panel_gap)
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

    divider_y = outer + header + panel + row_gap // 2
    draw.line((outer, divider_y, width - outer, divider_y), fill=(188, 192, 198), width=2)

    figure.save(OUTPUT_STEM.with_suffix(".png"), dpi=(300, 300))
    figure.save(OUTPUT_STEM.with_suffix(".pdf"), resolution=300)
    preview_width = 1800
    preview_height = round(height * preview_width / width)
    figure.resize((preview_width, preview_height), Image.Resampling.LANCZOS).save(
        ROOT / "segmentation_comparison_rows_2_3_preview.png"
    )

    manifest = {
        "layout": "Original rows 2 and 3 only; 2 rows x 6 columns",
        "selected_groups": [group for _, group in SELECTED_ROWS],
        "column_titles": COLUMNS,
        "canvas_size": [width, height],
        "prediction_panels_modified": False,
        "source_manifest": str(SOURCE_MANIFEST),
        "overlay_alpha": source.get("overlay_alpha"),
        "background_color_rgb": source.get("background_color_rgb"),
        "rows": output_rows,
    }
    (ROOT / "segmentation_comparison_rows_2_3_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(OUTPUT_STEM.with_suffix(".png"))
    print(OUTPUT_STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
