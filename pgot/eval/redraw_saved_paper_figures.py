"""Rebuild saved paper figures with matplotlib, retaining exact image tiles.

Frozen reference PNG crops are authoritative: source panel paths can have been
overwritten since composition. No inference, AI image editing, recoloring,
segmentation changes or description paraphrasing is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

from pgot.eval.language_object_editing import file_digest, write_json


DEFAULT_SOURCES = [
    "/home/jovyan/PGOT/outputs/paper_figures/deor_grounding_2x2_v3",
    "/home/jovyan/PGOT/outputs/paper_figures/segmentation_comparison_v2_red",
    "/home/jovyan/PGOT/outputs/paper_figures/segmentation_comparison_v2",
]


def wrap_text(text, figure, max_pixels, fontsize):
    renderer = figure.canvas.get_renderer()
    probe = figure.text(0,0,"",fontsize=fontsize)
    lines = []
    for word in text.split():
        candidate = (lines[-1]+" "+word) if lines else word
        probe.set_text(candidate)
        if lines and probe.get_window_extent(renderer).width > max_pixels:
            lines.append(word)
        elif lines:
            lines[-1] = candidate
        else:
            lines.append(word)
    probe.remove()
    return "\n".join(lines)


def redraw(source_dir, output_dir, dpi=200):
    source, out = Path(source_dir), Path(output_dir)
    if source.resolve() == out.resolve():
        raise ValueError("Preserve original figures: choose a separate output directory")
    manifest_path = source/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    grounding = "sets" in manifest
    filename = "deor_grounding_2x2" if grounding else "segmentation_comparison"
    original_path = source/(filename+".png")
    reference_hash = file_digest(original_path)
    with Image.open(original_path) as image:
        reference = image.convert("RGB")
    width,height = manifest["canvas_size"]
    if reference.size != (width,height):
        raise ValueError("Reference PNG dimensions differ from saved provenance")
    groups = manifest["sets"] if grounding else manifest["rows"]
    plt.rcParams.update({"font.family":"DejaVu Sans","pdf.fonttype":42,"svg.fonttype":"none"})
    figure = plt.figure(figsize=(width/dpi,height/dpi),dpi=dpi,facecolor="white")
    records,artists = [],[]
    for group in groups:
        panel_records = []
        for column,entry in enumerate(group["panels"]):
            x0,y0,x1,y1 = entry["box"]
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                raise ValueError("Invalid crop coordinates")
            # Exact embedded original tile, not a re-inferred/recolored mask.
            tile = reference.crop((x0,y0,x1,y1))
            ax = figure.add_axes([x0/width,(height-y1)/height,(x1-x0)/width,(y1-y0)/height])
            ax.imshow(tile,interpolation="none");ax.set_axis_off()
            panel_records.append(dict(entry,embedded_tile_rgb_sha256=hashlib.sha256(tile.tobytes()).hexdigest()))
            if grounding:
                if column == 0:
                    artist = figure.text((x0+x1)/2/width,(height-y0+12)/height,"Image",
                        fontsize=16,fontweight="bold",ha="center",va="bottom")
                else:
                    text = wrap_text(group["descriptions"][column-1],figure,x1-x0-24,12.5)
                    artist = figure.text((x0+12)/width,(height-y0+12)/height,text,
                        fontsize=12.5,ha="left",va="bottom",linespacing=1.17)
                artists.append(artist)
        records.append(dict(group,panels=panel_records))
    if not grounding:
        first = groups[0]["panels"]
        for title,entry in zip(manifest["column_titles"],first):
            x0,y0,x1,y1 = entry["box"]
            artists.append(figure.text((x0+x1)/2/width,(height-y0+25)/height,title,
                fontsize=18,fontweight="bold",ha="center",va="bottom"))
        left_edge = first[0]["box"][0]
        for start,label in ((0,"COCO"),(2,"VOC")):
            upper = groups[start]["panels"][0]["box"][1]
            lower = groups[start+1]["panels"][0]["box"][3]
            artists.append(figure.text(left_edge/2/width,(height-(upper+lower)/2)/height,label,
                fontsize=20,fontweight="bold",ha="center",va="center"))
        divider = (groups[1]["panels"][0]["box"][3]+groups[2]["panels"][0]["box"][1])/2
        figure.add_artist(Line2D([16/width,1-16/width],[(height-divider)/height]*2,
            transform=figure.transFigure,color="#bcc0c6",linewidth=0.7))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for artist in artists:
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width+0.1 or box.y1 > height+0.1:
            raise ValueError("Text clipped; enlarge header or canvas, do not silently shrink")
    if grounding:
        # Captions may not overlap image tiles in the preceding row either.
        for index,group in enumerate(groups):
            previous_bottom = groups[index-2]["panels"][0]["box"][3] if index >= 2 else 0
            for artist in artists[index*3:index*3+3]:
                if artist.get_window_extent(renderer).y1 > height-previous_bottom+0.1:
                    raise ValueError("Caption overlaps previous row; enlarge vertical spacing")
    out.mkdir(parents=True,exist_ok=True)
    for suffix in ("png","pdf","svg"):
        figure.savefig(out/(filename+"."+suffix),dpi=dpi,facecolor="white")
    figure.savefig(out/"preview.png",dpi=dpi/2,facecolor="white")
    plt.close(figure)
    if file_digest(original_path) != reference_hash:
        raise ValueError("Original figure changed during redraw")
    result = dict(original_manifest=manifest,renderer="matplotlib",ai_image_editing=False,
        original_figure=str(original_path.resolve()),original_sha256=reference_hash,
        original_manifest_sha256=file_digest(manifest_path),figure_pixels=[width,height],dpi=dpi,
        font="DejaVu Sans",image_tiles_reused_without_changes=True,
        masks_recolored_or_modified=False,descriptions_paraphrased=False,panels=records,
        code=str(Path(__file__).resolve()),code_sha256=file_digest(__file__))
    write_json(out/"manifest.json",result)
    print(out/(filename+".png"),flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source_dirs",nargs="+",default=DEFAULT_SOURCES)
    p.add_argument("--output_root",default="/home/jovyan/PGOT/outputs/paper_figures/matplotlib_redraws")
    args = p.parse_args()
    for source in args.source_dirs:
        redraw(source,Path(args.output_root)/Path(source).name)


if __name__ == "__main__":
    main()
