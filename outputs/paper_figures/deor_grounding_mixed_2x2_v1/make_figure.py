"""Create a mixed 2x2 description-region alignment figure.

Top row: first two examples from description_region_alignment_v3.
Bottom row: bottom two examples from deor_grounding_2x2_v3.
Stored panels and verbatim descriptions are reused without modification.
"""

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT_DIR = Path(__file__).resolve().parent
TOP_MANIFEST = Path(
    "/home/jovyan/PGOT/outputs/paper_figures/appendix_qualitative_v1/"
    "e2_description_region_alignment/manifest_v3.json"
)
BOTTOM_MANIFEST = Path(
    "/home/jovyan/PGOT/outputs/paper_figures/deor_grounding_2x2_v3/manifest.json"
)


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


def first_sentence(text):
    return re.split(r"(?<=[.!?])\s+", text)[0]


def wrap_pixels(text, draw, text_font, max_width):
    lines = []
    for word in text.split():
        if lines and draw.textlength(lines[-1] + " " + word, font=text_font) <= max_width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def load_entries():
    top = json.loads(TOP_MANIFEST.read_text())
    bottom = json.loads(BOTTOM_MANIFEST.read_text())

    entries = []
    for example in top["examples"][:2]:
        entries.append(
            {
                "origin": "description_region_alignment_v3",
                "dataset": example["dataset"],
                "id": example["id"],
                "descriptions": [
                    first_sentence(item["text"]) for item in example["descriptions"]
                ],
                "paths": [Path(item["path"]) for item in example["sources"]],
            }
        )

    for example in bottom["sets"][2:4]:
        entries.append(
            {
                "origin": "deor_grounding_2x2_v3_bottom_row",
                "dataset": example["dataset"],
                "id": example["id"],
                "descriptions": [first_sentence(text) for text in example["descriptions"]],
                "paths": [Path(item["source"]) for item in example["panels"]],
            }
        )
    return entries


def main():
    Image.init()
    entries = load_entries()
    if len(entries) != 4:
        raise ValueError("The 2x2 layout requires exactly four examples")

    panel = 512
    panel_gap = 8
    horizontal_gap = 48
    vertical_gap = 52
    outer = 16
    title_font = get_font(38, bold=True)
    description_font = get_font(34)
    line_height = 42

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = [
        [wrap_pixels(text, dummy, description_font, panel - 24) for text in entry["descriptions"]]
        for entry in entries
    ]
    header = max(len(lines) for entry_lines in wrapped for lines in entry_lines) * line_height + 32
    tile_width = panel * 3 + panel_gap * 2
    tile_height = header + panel
    width = outer * 2 + tile_width * 2 + horizontal_gap
    height = outer * 2 + tile_height * 2 + vertical_gap

    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    output_sets = []

    for index, entry in enumerate(entries):
        tile_x = outer + (index % 2) * (tile_width + horizontal_gap)
        tile_y = outer + (index // 2) * (tile_height + vertical_gap)
        record = {
            "origin": entry["origin"],
            "dataset": entry["dataset"],
            "id": entry["id"],
            "grid_position": [index // 2, index % 2],
            "descriptions": entry["descriptions"],
            "panels": [],
        }
        for column, path in enumerate(entry["paths"]):
            x = tile_x + column * (panel + panel_gap)
            if column == 0:
                center_text(draw, "Image", (x, tile_y + header - 52, panel, 52), title_font)
            else:
                lines = wrapped[index][column - 1]
                last_bottom = draw.textbbox((0, 0), lines[-1], font=description_font)[3]
                text_top = tile_y + header - 12 - (len(lines) - 1) * line_height - last_bottom
                for line_index, line in enumerate(lines):
                    draw.text(
                        (x + 12, text_top + line_index * line_height),
                        line,
                        fill="black",
                        font=description_font,
                    )
            with Image.open(path) as image:
                if image.size != (panel, panel):
                    raise ValueError(f"Unexpected panel size {image.size}: {path}")
                figure.paste(image.convert("RGB"), (x, tile_y + header))
            record["panels"].append(
                {
                    "source": str(path.resolve()),
                    "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "box": [x, tile_y + header, x + panel, tile_y + header + panel],
                }
            )
        output_sets.append(record)

    png = OUTPUT_DIR / "description_region_alignment_mixed_2x2.png"
    pdf = OUTPUT_DIR / "description_region_alignment_mixed_2x2.pdf"
    preview = OUTPUT_DIR / "preview.png"
    figure.save(png, dpi=(300, 300))
    figure.save(pdf, resolution=300)
    figure.resize((width // 2, height // 2), Image.Resampling.LANCZOS).save(preview)

    manifest = {
        "layout": "2x2 sets; each set contains Image and two description-region panels",
        "top_row_source": str(TOP_MANIFEST),
        "bottom_row_source": str(BOTTOM_MANIFEST),
        "input": "image-only autoregressive",
        "caption_display": "verbatim generated first sentences",
        "prediction_panels_modified": False,
        "canvas_size": [width, height],
        "sets": output_sets,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
