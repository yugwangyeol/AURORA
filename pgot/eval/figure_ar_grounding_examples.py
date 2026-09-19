"""Curate image-only AR description/region pairs from unchanged PGOT caches.

Qualitative source mining is GT-assisted, not a new benchmark. No descriptions
are rewritten and no GT masks are used to route or change predicted regions.
The cache compacts THING owners; STUFF owners must not shift caption labels.
"""

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.ar_binding_router import parse_ar_slots

COLORS = [(235, 65, 70), (65, 220, 110), (80, 110, 245),
          (240, 210, 50), (205, 75, 225)]


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def thing_slots(entry):
    slots = [dict(s) for s in parse_ar_slots(entry["ar_caption"]["text"],
              entry["ar_caption"]["ovt_count"]) if s["is_thing"]]
    for index, slot in enumerate(slots):
        slot["prediction_label"] = index+1
    return slots


def overlay(image, mask, color, alpha=.60):
    rgb = np.asarray(image).copy()
    rgb[mask] = np.rint((1-alpha)*rgb[mask]+alpha*np.array(color)).astype(np.uint8)
    return Image.fromarray(rgb)


def first_sentence(description):
    return re.split(r"(?<=[.!?])\s+", description)[0]


def candidates(root):
    source = json.loads((root/"candidates.json").read_text())
    rows = []
    for entry in source["entries"]:
        cache = root/"cache"/entry["dataset"]/entry["id"]
        if not (cache/"ours.json").exists():
            continue
        slots = thing_slots(entry)
        metrics = json.loads((cache/"ours.json").read_text())
        if not 2 <= len(slots) <= 5 or not all(s["parsed"] for s in slots) or metrics["mIoU"] < .67:
            continue
        if hashlib.sha256(entry["ar_caption"]["text"].encode()).hexdigest() != entry["caption_sha256"]:
            raise ValueError("AR caption provenance mismatch")
        pred = np.load(cache/"ours.npz")["prediction"]
        if any(np.count_nonzero(pred == s["prediction_label"]) < 1311 for s in slots):
            continue
        relation = sum(bool(re.search(r"behind|beside|next to|holding|wearing|left|right|foreground|under|near", s["description"], re.I)) for s in slots)
        rank = metrics["mIoU"]+.035*len(set(s["normalized_category"] for s in slots))+.02*relation
        rank -= .0003*sum(max(0, len(s["description"])-280) for s in slots)
        rows.append(dict(entry, slots=slots, cached_metrics=metrics, rank=rank, source_cache=str(cache.resolve())))
    return sorted(rows, key=lambda row: row["rank"], reverse=True), source["checkpoint"]


def shortlist(args):
    root, out = Path(args.source_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, checkpoint = candidates(root)
    chosen = []
    for dataset in ["coco", "voc"]:
        chosen += [r for r in rows if r["dataset"] == dataset][:args.candidates_per_dataset]
    (out/"shortlist.json").write_text(json.dumps(dict(checkpoint=checkpoint,
        qualitative_gt_assisted_selection=True, candidates=chosen), indent=2))
    for start in range(0, len(chosen), 20):
        page = chosen[start:start+20]
        sheet = Image.new("RGB", (5*288, 4*336), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            x, y = (index%5)*288, (index//5)*336
            image = Image.open(Path(row["source_cache"])/"input.png").convert("RGB")
            sheet.paste(image.resize((280, 280)), (x, y+24))
            draw.text((x+2, y+2), f'{row["dataset"]} {row["id"]} | {row["cached_metrics"]["mIoU"]:.2f}', fill="black", font=font(15, True))
            names = ", ".join(s["category"] for s in row["slots"])
            draw.text((x+2, y+305), textwrap.shorten(names, width=40), fill="black", font=font(13))
        sheet.save(out/f"shortlist_page_{start//20+1:02}.png")
    print('Eligible', len(rows), 'shortlisted', len(chosen))


def render(args):
    out = Path(args.output_dir)
    shortlist = json.loads((out/"shortlist.json").read_text())
    rows = {f'{r["dataset"]}:{r["id"]}': r for r in shortlist["candidates"]}
    selected = [rows[key] for key in args.selection_ids.split(",")]
    selections, previews = [], []
    reviews = json.loads((out/"visual_review.json").read_text()) if (out/"visual_review.json").exists() else {}
    for rank, row in enumerate(selected, 1):
        cache = Path(row["source_cache"])
        dest = out/args.selection_subdir/f'{rank:02}_{row["dataset"]}_{row["id"]}'
        dest.mkdir(parents=True, exist_ok=True)
        image = Image.open(cache/"input.png").convert("RGB")
        pred = np.load(cache/"ours.npz")["prediction"]
        before = pred.copy()
        wrapped = [textwrap.wrap(first_sentence(s["description"]), width=38) for s in row["slots"]]
        header = max(95, max(len(lines) for lines in wrapped)*29+45)
        width = (1+len(row["slots"]))*512
        paper = Image.new("RGB", (width, header+512), "white")
        draw = ImageDraw.Draw(paper)
        draw.text((14, 10), "Image", font=font(24, True), fill="black")
        paper.paste(image, (0, header))
        image.save(dest/"image.png")
        masks = {"prediction": pred}
        md = [f'# {row["dataset"]} {row["id"]}: DeOR image-only AR',
              'Figure headers are verbatim first sentences, not input prompts. Full descriptions follow.']
        for index, (slot, lines) in enumerate(zip(row["slots"], wrapped)):
            mask = pred == slot["prediction_label"]
            panel = overlay(image, mask, COLORS[index])
            x = (index+1)*512
            draw.text((x+12, 6), f'{slot["category"]} · AR object {index+1}', fill="black", font=font(21, True))
            draw.multiline_text((x+12, 36), "\n".join(lines), font=font(23), fill="black", spacing=5)
            paper.paste(panel, (x, header))
            panel.save(dest/f'object_{index+1:02}_overlay.png')
            masks[f'object_{index+1:02}'] = mask
            md += [f'\n## Object {index+1}: {slot["category"]}', slot["description"]]
        assert np.array_equal(pred, before)
        paper.save(dest/"paper_panel.png")
        np.savez_compressed(dest/"masks.npz", **masks)
        (dest/"captions.md").write_text("\n\n".join(md))
        review = reviews.get(f'{row["dataset"]}:{row["id"]}', {"status": "unreviewed"})
        meta = dict(row, checkpoint=shortlist["checkpoint"], input="image-only AR; generated captions replayed for region readout",
            external_text_query=False, gt_mask_input=False, gt_caption_input=False,
            caption_display="verbatim first sentence; complete originals in captions.md and metadata",
            masks_modified=False, visual_review=review, prediction_sha256=hashlib.sha256((cache/"ours.npz").read_bytes()).hexdigest())
        (dest/"metadata.json").write_text(json.dumps(meta, indent=2))
        selections.append(dict(rank=rank, dataset=row["dataset"], id=row["id"],
            object_count=len(row["slots"]), directory=str(dest.resolve()), visual_review=review))
        previews.append(paper.resize((1280, round(paper.height*1280/paper.width)), Image.Resampling.LANCZOS))
    selected_dir = out/args.selection_subdir
    for start in range(0, len(previews), 5):
        page = previews[start:start+5]
        sheet = Image.new("RGB", (1280, sum(p.height for p in page)+32*len(page)), "white")
        draw = ImageDraw.Draw(sheet)
        y = 0
        for index, panel in enumerate(page, start):
            row = selected[index]
            draw.text((8, y+5), f'{index+1:02} | {row["dataset"].upper()} {row["id"]} | DeOR generated description-region pairs', fill="black", font=font(18, True))
            sheet.paste(panel, (0, y+32))
            y += panel.height+32
        sheet.save(selected_dir/f'contact_page_{start//5+1:02}.png')
    (out/f"{args.selection_subdir}_selection.json").write_text(json.dumps(dict(qualitative_gt_assisted_selection=True,
        not_a_benchmark=True, checkpoint=shortlist["checkpoint"], selected=selections), indent=2))
    print('Exported', len(selected), 'images with', sum(s['object_count'] for s in selections), 'original AR object-region pairs')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["shortlist", "render"], required=True)
    p.add_argument("--source_dir", default="/home/jovyan/PGOT/outputs/figure_segmentation_over_under")
    p.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/figure_deor_ar_grounding")
    p.add_argument("--candidates_per_dataset", type=int, default=25)
    p.add_argument("--selection_ids", default="")
    p.add_argument("--selection_subdir", default="curated")
    args = p.parse_args()
    if args.stage == "render" and not args.selection_ids:
        p.error('render requires explicit dataset:id selection_ids')
    {"shortlist": shortlist, "render": render}[args.stage](args)


if __name__ == "__main__":
    main()
