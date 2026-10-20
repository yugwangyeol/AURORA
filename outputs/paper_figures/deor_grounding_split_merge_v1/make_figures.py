"""Split the main 2x2 grounding figure and extend the appendix v3 figure."""

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).resolve().parent
MAIN_MANIFEST = Path(
    "/home/jovyan/PGOT/outputs/paper_figures/deor_grounding_2x2_v3/manifest.json"
)
V3_MANIFEST = Path(
    "/home/jovyan/PGOT/outputs/paper_figures/appendix_qualitative_v1/"
    "e2_description_region_alignment/manifest_v3.json"
)


def font(size, bold=False):
    suffix = "-Bold.ttf" if bold else ".ttf"
    return ImageFont.truetype(
        f"/usr/share/fonts/truetype/dejavu/DejaVuSerif{suffix}", size
    )


def centered(draw, text, box, text_font):
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


def wrap_pixels(draw, text, text_font, max_width):
    lines = []
    for word in text.split():
        if lines and draw.textlength(lines[-1] + " " + word, font=text_font) <= max_width:
            lines[-1] += " " + word
        else:
            lines.append(word)
    return lines


def source_entries():
    main = json.loads(MAIN_MANIFEST.read_text())
    v3 = json.loads(V3_MANIFEST.read_text())

    def from_main(example):
        return {
            "origin": "deor_grounding_2x2_v3",
            "dataset": example["dataset"],
            "id": example["id"],
            "descriptions": [first_sentence(text) for text in example["descriptions"]],
            "paths": [Path(panel["source"]) for panel in example["panels"]],
        }

    def from_v3(example):
        return {
            "origin": "description_region_alignment_v3",
            "dataset": example["dataset"],
            "id": example["id"],
            "descriptions": [first_sentence(item["text"]) for item in example["descriptions"]],
            "paths": [Path(source["path"]) for source in example["sources"]],
        }

    top = [from_main(example) for example in main["sets"][:2]]
    bottom = [from_main(example) for example in main["sets"][2:4]]
    appendix = [from_v3(example) for example in v3["examples"]]
    return top, bottom, appendix


def panel_record(path, box):
    return {
        "source": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "box": list(box),
    }


def render_main_style(entries, rows, columns, stem):
    if len(entries) != rows * columns:
        raise ValueError(f"Expected {rows * columns} entries, got {len(entries)}")
    panel, panel_gap, tile_gap, outer = 512, 8, 48, 16
    vertical_gap = 52
    heading_font, text_font = font(38, True), font(34)
    line_height = 42
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    wrapped = [
        [wrap_pixels(dummy, text, text_font, panel - 24) for text in entry["descriptions"]]
        for entry in entries
    ]
    header = max(len(lines) for entry in wrapped for lines in entry) * line_height + 32
    tile_w, tile_h = panel * 3 + panel_gap * 2, header + panel
    width = outer * 2 + columns * tile_w + (columns - 1) * tile_gap
    height = outer * 2 + rows * tile_h + (rows - 1) * vertical_gap
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    records = []

    for index, entry in enumerate(entries):
        x0 = outer + (index % columns) * (tile_w + tile_gap)
        y0 = outer + (index // columns) * (tile_h + vertical_gap)
        record = {
            "origin": entry["origin"],
            "dataset": entry["dataset"],
            "id": entry["id"],
            "grid_position": [index // columns, index % columns],
            "descriptions": entry["descriptions"],
            "panels": [],
        }
        for column, path in enumerate(entry["paths"]):
            x = x0 + column * (panel + panel_gap)
            if column == 0:
                centered(draw, "Image", (x, y0 + header - 52, panel, 52), heading_font)
            else:
                lines = wrapped[index][column - 1]
                last_bottom = draw.textbbox((0, 0), lines[-1], font=text_font)[3]
                text_top = y0 + header - 12 - (len(lines) - 1) * line_height - last_bottom
                for line_index, line in enumerate(lines):
                    draw.text(
                        (x + 12, text_top + line_index * line_height),
                        line,
                        fill="black",
                        font=text_font,
                    )
            with Image.open(path) as image:
                if image.size != (panel, panel):
                    raise ValueError(f"Unexpected panel size {image.size}: {path}")
                canvas.paste(image.convert("RGB"), (x, y0 + header))
            record["panels"].append(
                panel_record(path, (x, y0 + header, x + panel, y0 + header + panel))
            )
        records.append(record)

    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, resolution=300)
    return canvas, records


def render_v3_extended(entries, stem):
    panel, panel_gap, tile_gap, outer, header = 300, 5, 30, 12, 150
    columns = 2
    rows = (len(entries) + columns - 1) // columns
    tile_w, tile_h = panel * 3 + panel_gap * 2, header + panel
    width = outer * 2 + tile_w * columns + tile_gap
    height = outer * 2 + tile_h * rows + tile_gap * (rows - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    heading_font, text_font = font(23, True), font(20)
    records = []

    for index, entry in enumerate(entries):
        x0 = outer + (index % columns) * (tile_w + tile_gap)
        y0 = outer + (index // columns) * (tile_h + tile_gap)
        record = {
            "origin": entry["origin"],
            "dataset": entry["dataset"],
            "id": entry["id"],
            "grid_position": [index // columns, index % columns],
            "descriptions": entry["descriptions"],
            "panels": [],
        }
        for column, path in enumerate(entry["paths"]):
            x = x0 + column * (panel + panel_gap)
            if column == 0:
                centered(draw, "Image", (x, y0 + header - 36, panel, 30), heading_font)
            else:
                lines = wrap_pixels(draw, entry["descriptions"][column - 1], text_font, panel - 18)
                line_height = 25
                text_top = y0 + max(0, header - len(lines) * line_height - 6)
                for line_index, line in enumerate(lines):
                    draw.text(
                        (x + 9, text_top + line_index * line_height),
                        line,
                        font=text_font,
                        fill="black",
                    )
            with Image.open(path) as image:
                resized = image.convert("RGB").resize((panel, panel), Image.Resampling.LANCZOS)
                canvas.paste(resized, (x, y0 + header))
            record["panels"].append(
                panel_record(path, (x, y0 + header, x + panel, y0 + header + panel))
            )
        records.append(record)

    png, pdf = OUT / f"{stem}.png", OUT / f"{stem}.pdf"
    canvas.save(png, dpi=(300, 300))
    canvas.save(pdf, resolution=300)
    return canvas, records


def main():
    Image.init()
    top, bottom, appendix = source_entries()

    bottom_row, bottom_row_records = render_main_style(
        bottom, 1, 2, "deor_grounding_bottom_row_1x2"
    )
    bottom_column, bottom_column_records = render_main_style(
        bottom, 2, 1, "deor_grounding_bottom_column_2x1"
    )
    # Keep the original v3 order and append the old main figure's top row.
    extended, extended_records = render_v3_extended(
        appendix + top, "description_region_alignment_v3_plus_top_row"
    )

    bottom_row.resize(
        (1592, round(bottom_row.height * 1592 / bottom_row.width)), Image.Resampling.LANCZOS
    ).save(OUT / "deor_grounding_bottom_row_1x2_preview.png")
    bottom_column.resize(
        (1000, round(bottom_column.height * 1000 / bottom_column.width)), Image.Resampling.LANCZOS
    ).save(OUT / "deor_grounding_bottom_column_2x1_preview.png")
    extended.resize(
        (1200, round(extended.height * 1200 / extended.width)), Image.Resampling.LANCZOS
    ).save(OUT / "description_region_alignment_v3_plus_top_row_preview.png")

    manifest = {
        "source_main": str(MAIN_MANIFEST),
        "source_appendix_v3": str(V3_MANIFEST),
        "prediction_panels_modified": False,
        "bottom_row_1x2": bottom_row_records,
        "bottom_column_2x1": bottom_column_records,
        "appendix_v3_plus_top_row": extended_records,
        "appendix_extension_order": "original v3 examples followed by the old main figure top row",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
