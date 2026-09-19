"""Compose curated image-only AR description-region sets in a 2x2 figure.

Panel pixels and verbatim generated first sentences are reused unchanged.
Ordering is row-major: upper-left, upper-right, lower-left, lower-right.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

from pgot.eval.figure_ar_grounding_examples import first_sentence
from pgot.eval.render_segmentation_paper_figure import center_text, get_font


def wrap_pixels(text, draw, font, max_width):
    lines = []
    for word in text.split():
        if lines and draw.textlength(lines[-1]+" "+word, font=font) <= max_width:
            lines[-1] += " "+word
        else:
            lines.append(word)
    return lines


def compose(source_dir, output_dir, ranks=(2, 6, 10, 4)):
    if len(ranks) != 4 or len(set(ranks)) != 4:
        raise ValueError("2x2 layout requires four distinct selections")
    Image.init()
    source, out = Path(source_dir), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source/"curated_selection.json").read_text())
    by_rank = {row["rank"]: row for row in selection["selected"]}
    rows = [by_rank[rank] for rank in ranks]
    entries = [json.loads((Path(row["directory"])/"metadata.json").read_text()) for row in rows]
    if any(len(entry["slots"]) != 2 for entry in entries):
        raise ValueError("This matching-width layout expects two objects per scene")
    panel, panel_gap, horizontal_gap, vertical_gap, outer = 512, 8, 48, 52, 16
    title_font, description_font = get_font(38, True), get_font(34)
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    texts = [[first_sentence(slot["description"]) for slot in entry["slots"]] for entry in entries]
    wrapped = [[wrap_pixels(text, dummy, description_font, panel-24) for text in row] for row in texts]
    line_height = 42
    header = max(len(lines) for row in wrapped for lines in row)*line_height+32
    tile_width, tile_height = panel*3+panel_gap*2, header+panel
    width, height = outer*2+tile_width*2+horizontal_gap, outer*2+tile_height*2+vertical_gap
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    manifest = {"layout": "2x2 sets; each set Image / description-region 1 / description-region 2",
        "model_display_name": "DeOR", "input": "image-only AR; NOT external text queries",
        "caption_display": "verbatim generated first sentences; no object-name headings",
        "description_alignment": "bottom-aligned immediately above each overlay, uniform 12px gap",
        "image_heading_position": "immediately above original photo", "prediction_panels_modified": False,
        "rank_order": list(ranks), "canvas_size": [width, height], "sets": []}
    for index, (row, entry) in enumerate(zip(rows, entries)):
        tile_x = outer+(index%2)*(tile_width+horizontal_gap)
        tile_y = outer+(index//2)*(tile_height+vertical_gap)
        folder = Path(row["directory"])
        paths = [folder/"image.png", folder/"object_01_overlay.png", folder/"object_02_overlay.png"]
        record = {"rank": row["rank"], "dataset": row["dataset"], "id": row["id"],
                  "grid_position": [index//2, index%2], "descriptions": texts[index], "panels": []}
        for column, path in enumerate(paths):
            x = tile_x+column*(panel+panel_gap)
            if column == 0:
                center_text(draw, "Image", (x, tile_y+header-52, panel, 52), title_font)
            if column:
                lines = wrapped[index][column-1]
                # Anchor the visible bottom of the last line to the photo,
                # not the paragraph top; short descriptions no longer float.
                last_bottom = draw.textbbox((0, 0), lines[-1], font=description_font)[3]
                text_top = tile_y+header-12-(len(lines)-1)*line_height-last_bottom
                for line_index, line in enumerate(lines):
                    draw.text((x+12, text_top+line_index*line_height), line, fill="black", font=description_font)
            with Image.open(path) as image:
                if image.size != (panel, panel):
                    raise ValueError(f"Unexpected native panel dimensions: {path}")
                figure.paste(image.convert("RGB"), (x, tile_y+header))
            record["panels"].append({"source": str(path.resolve()),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "box": [x, tile_y+header, x+panel, tile_y+header+panel]})
        manifest["sets"].append(record)
    figure.save(out/"deor_grounding_2x2.png", dpi=(300, 300))
    figure.save(out/"deor_grounding_2x2.pdf", resolution=300)
    figure.resize((width//2, height//2), Image.Resampling.LANCZOS).save(out/"preview.png")
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out/"deor_grounding_2x2.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dir", default="/home/jovyan/PGOT/outputs/figure_deor_ar_grounding")
    p.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/paper_figures/deor_grounding_2x2_v3")
    p.add_argument("--ranks", type=int, nargs=4, default=[2, 6, 10, 4])
    args = p.parse_args()
    compose(args.source_dir, args.output_dir, args.ranks)


if __name__ == "__main__":
    main()
