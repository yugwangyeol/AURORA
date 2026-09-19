"""Render user-selected cached segmentation masks without running inference.

GT matching assigns display colors only; it never merges or changes labels.
Every predicted segment, including background fragments, keeps its own color.
"""

import argparse
import colorsys
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

PALETTE = [(245, 55, 104), (65, 235, 130), (255, 211, 45),
           (90, 110, 255), (204, 65, 238), (35, 223, 223),
           (255, 131, 30), (135, 55, 225), (35, 165, 240),
           (212, 240, 55), (242, 80, 194), (25, 190, 120)]
MODEL_COLUMNS = [("adaslot", "AdaSlot"), ("metaslot", "MetaSlot"),
                 ("coda", "CODA"), ("ours", "Ours")]
USER_SELECTIONS = {"coco_over": 1, "coco_under": 3,
                   "voc_over": 5, "voc_under": 7}


def palette_color(index):
    if index < len(PALETTE):
        return PALETTE[index]
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(
        (index * .61803398875) % 1, .86, .97))


def label_colors(labels, gt, ignore, background_color=None):
    """One-to-one IoU color alignment, excluding unknown GT pixels."""
    labels, gt, ignore = np.asarray(labels), np.asarray(gt), np.asarray(ignore, bool)
    pred_ids, gt_ids = np.unique(labels), np.unique(gt[~ignore])
    table = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
    valid = ~ignore
    for r, g in enumerate(gt_ids):
        ground = (gt == g) & valid
        for c, p in enumerate(pred_ids):
            predicted = (labels == p) & valid
            union = np.count_nonzero(ground | predicted)
            table[r, c] = np.count_nonzero(ground & predicted) / max(union, 1)
    rows, cols = linear_sum_assignment(table, maximize=True)
    matched = {int(pred_ids[c]): int(gt_ids[r]) for r, c in zip(rows, cols)
               if table[r, c] > 0}
    unmatched = sorted((int(p) for p in pred_ids if int(p) not in matched),
                       key=lambda p: (-np.count_nonzero((labels == p) & valid), p))
    color_indices = dict(matched)
    color_indices.update({p: int(gt.max()) + 1 + i for i, p in enumerate(unmatched)})
    colors = {p: tuple(background_color) if index == 0 and background_color is not None
              else palette_color(index) for p, index in color_indices.items()}
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for p, color in colors.items():
        rgb[labels == p] = color
    rgb[ignore] = (160, 160, 160)
    return rgb, colors


def blend(image, colors, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Overlay alpha must lie in [0, 1]")
    return np.rint(np.asarray(image, float) * (1-alpha) +
                   np.asarray(colors, float) * alpha).clip(0, 255).astype(np.uint8)


def get_font(size, bold=False):
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSerif" + ("-Bold.ttf" if bold else ".ttf")
    return ImageFont.truetype(path, size)


def center_text(draw, text, box, font):
    left, top, width, height = box
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    draw.text((left+(width-(x1-x0))/2-x0, top+(height-(y1-y0))/2-y0),
              text, font=font, fill="black")


def render(args):
    # Pillow's PDF writer needs the JPEG save handler to be registered even
    # when every source input was PNG/NPZ.
    Image.init()
    root, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((root / "selection.json").read_text())
    kind = getattr(args, "kind", "over")
    figure_name = "oversegmentation_figure" if kind == "over" else "undersegmentation_figure"
    background_color = getattr(args, "background_color", PALETTE[0])
    manifest = {"overlay_alpha": args.alpha, "background_colored": True,
                "background_color_rgb": list(background_color),
                "prediction_labels_modified": False, "gt_matching": "display colors only",
                "ignore_display": "neutral gray; excluded from color assignment",
                "model_order": [name for _, name in MODEL_COLUMNS],
                "user_selections": {}, "rendered_groups": [f"coco_{kind}", f"voc_{kind}"]}
    for group, number in USER_SELECTIONS.items():
        entry = selection[group]["examples"][number-1]
        manifest["user_selections"][group] = {"number": number, "id": entry["id"]}
    panel, gap, header, label_height = 512, 6, 64, 38
    width = panel*6 + gap*5
    row_height = label_height + panel
    combined = Image.new("RGB", (width, header+row_height*2+20), "white")
    combined_draw = ImageDraw.Draw(combined)
    names = ["Image", "GT"] + [name for _, name in MODEL_COLUMNS]
    font, label_font = get_font(36, True), get_font(25)
    for i, name in enumerate(names):
        center_text(combined_draw, name, (i*(panel+gap), 0, panel, header), font)
    for row_index, group in enumerate(manifest["rendered_groups"]):
        entry = selection[group]["examples"][USER_SELECTIONS[group]-1]
        cache = root / "cache" / entry["dataset"] / entry["id"]
        data = np.load(cache / "gt.npz")
        gt, ignore = data["gt"], data["ignore"].astype(bool)
        original = np.asarray(Image.open(cache / "input.png").convert("RGB"))
        assert original.shape[:2] == gt.shape == ignore.shape == (panel, panel)
        dest = out / group
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(original).save(dest / "image.png")
        columns = [original]
        details = {}
        for model in ["gt"] + [m for m, _ in MODEL_COLUMNS]:
            source = cache / ("gt.npz" if model == "gt" else f"{model}.npz")
            labels = gt if model == "gt" else np.load(source)["prediction"]
            before = labels.copy()
            colors, mapping = label_colors(labels, gt, ignore, background_color)
            rgb = blend(original, colors, args.alpha)
            assert np.array_equal(labels, before)
            Image.fromarray(rgb).save(dest / f"{model}_overlay.png")
            Image.fromarray(colors).save(dest / f"{model}_map.png")
            columns.append(rgb)
            details[model] = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "label_colors": mapping, "unique_labels": np.unique(labels).tolist()}
        single = Image.new("RGB", (width, header+panel), "white")
        single_draw = ImageDraw.Draw(single)
        row_top = header+row_index*(row_height+20)
        dataset_label = "COCO" if entry["dataset"] == "coco" else "PASCAL VOC"
        combined_draw.text((2, row_top+3), dataset_label, fill="black", font=label_font)
        for i, (name, rgb) in enumerate(zip(names, columns)):
            image = Image.fromarray(rgb)
            x = i*(panel+gap)
            center_text(single_draw, name, (x, 0, panel, header), font)
            single.paste(image, (x, header))
            combined.paste(image, (x, row_top+label_height))
        single.save(out / f"{group}.png", dpi=(300, 300))
        single.save(out / f"{group}.pdf", resolution=300)
        manifest[group] = {"id": entry["id"], "strict_eligible": entry["strict_eligible"],
                           "scores": entry["scores"], "panels": details}
    combined.save(out / f"{figure_name}.png", dpi=(300, 300))
    combined.save(out / f"{figure_name}.pdf", resolution=300)
    combined.resize((1551, combined.height//2), Image.Resampling.LANCZOS).save(out / "preview.png")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(out / f"{figure_name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="/home/jovyan/PGOT/outputs/figure_segmentation_over_under")
    parser.add_argument("--kind", choices=["over", "under"], default="over")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--alpha", type=float, default=.65)
    parser.add_argument("--background_color", type=int, nargs=3, default=PALETTE[0],
                        metavar=("R", "G", "B"), help="Display color for the background-matched label")
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("alpha must lie in [0, 1]")
    if any(not 0 <= c <= 255 for c in args.background_color):
        parser.error("background RGB values must lie in [0, 255]")
    if args.output_dir is None:
        name = "oversegmentation" if args.kind == "over" else "undersegmentation"
        args.output_dir = f"/home/jovyan/PGOT/outputs/paper_figures/{name}_v1"
    render(args)


if __name__ == "__main__":
    main()
