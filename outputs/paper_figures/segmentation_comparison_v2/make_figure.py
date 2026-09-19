"""Compose four selected segmentation examples into one paper figure.

Use the saved v2 panels unchanged. Column names appear once; COCO occupies
the first two rows and VOC the final two, with spanning dataset labels only.
"""

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PALETTE = [(245, 55, 104), (65, 235, 130), (255, 211, 45),
           (90, 110, 255), (204, 65, 238), (35, 223, 223),
           (255, 131, 30), (135, 55, 225), (35, 165, 240),
           (212, 240, 55), (242, 80, 194), (25, 190, 120)]

def get_font(size, bold=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif" + ("-Bold.ttf" if bold else ".ttf")
    return ImageFont.truetype(path, size)


def center_text(draw, text, box, font):
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((left+(width-(x1-x0))/2-x0, top+(height-(y1-y0))/2-y0),
              text, font=font, fill="black")



COLUMNS = [("image", "Image"), ("gt_overlay", "GT"),
           ("adaslot_overlay", "AdaSlot"), ("metaslot_overlay", "MetaSlot"),
           ("coda_overlay", "CODA"), ("ours_overlay", "DeOR")]


def compose(source_root, output_dir, source_version="v2"):
    Image.init()
    source_root, out = Path(source_root), Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, gap, row_gap, group_gap = 512, 8, 12, 48
    outer, dataset_width, header = 16, 208, 88
    grid_left = outer + dataset_width
    width = grid_left + 6*panel + 5*gap + outer
    group_height = 2*panel + row_gap
    height = outer + header + 2*group_height + group_gap + outer
    figure = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(figure)
    title_font, dataset_font = get_font(48, True), get_font(52, True)
    for index, (_, title) in enumerate(COLUMNS):
        center_text(draw, title, (grid_left+index*(panel+gap), outer, panel, header), title_font)
    manifest = {"layout": "4 rows x 6 panels, COCO first two rows, VOC last two; spanning dataset labels, no error-type labels",
                "column_titles": [title for _, title in COLUMNS],
                "native_panel_size": panel, "canvas_size": [width, height],
                "prediction_panels_modified": False, "rows": []}
    previous_protocol = None
    for group_index, dataset in enumerate(["coco", "voc"]):
        group_y = outer+header+group_index*(group_height+group_gap)
        center_text(draw, dataset.upper(), (outer, group_y, dataset_width, group_height), dataset_font)
        for row_index, mode in enumerate(["over", "under"]):
            source = source_root / f"{mode}segmentation_{source_version}"
            protocol = json.loads((source / "manifest.json").read_text())
            current = (protocol["overlay_alpha"], tuple(protocol.get("background_color_rgb", PALETTE[0])), protocol["model_order"])
            if previous_protocol is not None and previous_protocol != current:
                raise ValueError("Both source figures must have identical colors, opacity and model order")
            previous_protocol = current
            y = group_y+row_index*(panel+row_gap)
            group = f"{dataset}_{mode}"
            row = {"group": group, "image_id": protocol[group]["id"],
                   "strict_eligible": protocol[group].get("strict_eligible"), "panels": []}
            for index, (filename, title) in enumerate(COLUMNS):
                path = source / group / f"{filename}.png"
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    if image.size != (panel, panel):
                        raise ValueError(f"Unexpected panel size: {path}")
                    x = grid_left+index*(panel+gap)
                    figure.paste(image, (x, y))
                row["panels"].append({"title": title, "source": str(path.resolve()),
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "box": [x, y, x+panel, y+panel]})
            manifest["rows"].append(row)
    # Thin divider separates datasets without repeating column headers.
    divider_y = outer+header+group_height+group_gap//2
    draw.line((outer, divider_y, width-outer, divider_y), fill=(188, 192, 198), width=2)
    manifest["overlay_alpha"], manifest["background_color_rgb"], _ = previous_protocol
    figure.save(out / "segmentation_comparison.png", dpi=(300, 300))
    figure.save(out / "segmentation_comparison.pdf", resolution=300)
    preview_width = 1722
    figure.resize((preview_width, round(height*preview_width/width)), Image.Resampling.LANCZOS).save(out / "preview.png")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(out / "segmentation_comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_root", default="/home/jovyan/PGOT/outputs/paper_figures")
    parser.add_argument("--source_version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--output_dir", default=str(Path(__file__).resolve().parent / "reproduced"))
    args = parser.parse_args()
    compose(args.source_root, args.output_dir, args.source_version)


if __name__ == "__main__":
    main()
