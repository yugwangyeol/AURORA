"""Figures from actual recorded external-LLM decisions and native PGOT edits."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion

from pgot.eval.compose_object_transfer_figure import wrap_prompt
from pgot.eval.language_object_editing import checked_selection, file_digest, write_json


OUTLINE_COLOR = (0, 185, 50)
REMOVE_MASK_ALPHA = 0.38


def outlined(image_path, mask_path, fill_alpha=0.0):
    """Display-only predicted-mask annotation; never write to source assets."""
    if not 0 <= fill_alpha <= 1:
        raise ValueError("Mask opacity must be in [0,1]")
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("RGB")).copy()
    with Image.open(mask_path) as mask:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != pixels.shape[:2]:
        raise ValueError("Predicted mask and source image sizes differ")
    if fill_alpha:
        pixels[mask] = np.rint((1-fill_alpha)*pixels[mask] +
                              fill_alpha*np.asarray(OUTLINE_COLOR)).astype(np.uint8)
    # Six-pixel contour at the saved source resolution, visible after figure resizing.
    edge = binary_dilation(mask, iterations=2) & ~binary_erosion(mask, iterations=4)
    pixels[edge] = OUTLINE_COLOR
    return Image.fromarray(pixels)


def panel(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def colored_prompt(fig, text, center, top, available, width, height, size=23):
    wrapped = wrap_prompt(text, fig, available, size)
    renderer = fig.canvas.get_renderer()
    artists = []
    for index, line in enumerate(wrapped.split("\n")):
        verb, space, remainder = line.partition(" ")
        parts = ([(verb, "#c72b2b", "bold"), (space + remainder, "black", "normal")]
                 if index == 0 and verb in ("Remove", "Transfer", "Replace") else [(line, "black", "normal")])
        sizes = []
        for text, color, weight in parts:
            probe = fig.text(0, 0, text, fontsize=size, fontweight=weight)
            sizes.append(probe.get_window_extent(renderer).width / fig.dpi)
            probe.remove()
        x = center - sum(sizes)/2
        for (text, color, weight), text_width in zip(parts, sizes):
            artists.append(fig.text(x/width, (top-index*size/72*1.25)/height, text,
                fontsize=size, color=color, fontweight=weight, ha="left", va="top"))
            x += text_width
    return artists


def compose(root, figure_output_dir=None, display_overrides=None):
    root = Path(root)
    overrides = display_overrides or {}
    m = json.loads((root / "manifest.json").read_text())
    if set(overrides) - {q["id"] for q in m["requests"]}:
        raise ValueError("Figure override refers to an unknown request")
    validation = json.loads((root / "validation.json").read_text())
    if not validation["exact_llm_selection_execution_validated"] or validation["checked_requests"] != len(m["requests"]):
        raise ValueError("Require full exact-selection audit before figure delivery")
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"})
    side, gap, center_gap = 2.77, 0.08, 0.28
    widths = [3*side+2*gap, 4*side+3*gap]
    lefts = [0, widths[0]+center_gap]
    width, height = sum(widths)+center_gap, 12.1
    fig = plt.figure(figsize=(width, height), dpi=200, facecolor="white")
    line_x = (widths[0]+center_gap/2)/width
    fig.add_artist(Line2D([line_x,line_x],[0.025,0.99],transform=fig.transFigure,
                         color="#999999",linewidth=0.8))
    records, artists, all_reviews = [], [], []
    lines = ["# Actual language-selected editing", "",
        "External controller: text-only Qwen3-VL-4B-Instruct. No image/GT/reference IDs or manual correction.",
        "All actual decisions below, including abstentions; not a benchmark accuracy estimate.", "",
        "| Request | Command | Actual operation | Selected A / B | Selected categories A / B | Generation |", "| --- | --- | --- | --- | --- | --- |"]
    for index, query in enumerate(m["requests"]):
        selection, resolved = checked_selection(root, m, query)
        folder = root/"edits"/query["id"]
        metadata = json.loads((folder/"metadata.json").read_text())
        if metadata["selection_sha256"] != file_digest(root/"plans"/(query["id"]+".json")):
            raise ValueError("Native edit no longer matches recorded LLM selection")
        generated = metadata["status"] == "generated"
        override = overrides.get(query["id"], {})
        if set(override) - {"execution_mode", "display_prompt"}:
            raise ValueError("Unknown figure override field")
        display_prompt = override.get("display_prompt", query["request"])
        execution_mode = metadata.get("execution_mode", resolved["operation"] if resolved else None)
        edited_name = "edited_native.png"
        if "execution_mode" in override:
            execution_mode = override["execution_mode"]
            if (not generated or not metadata.get("replacement_comparison") or
                    resolved["operation"] != "replace" or
                    execution_mode not in ("visual_only", "full_bundle")):
                raise ValueError("Override requires a recorded native replacement comparison")
            edited_name = execution_mode + "_native.png"
            comparison = json.loads((root / "figure" / ("replacement_comparison_" + query["id"] + ".json")).read_text())
            expected = next(p for p in comparison["native_panels"] if Path(p["path"]).name == edited_name)
            if comparison["actual_selection"] != selection["plan"] or file_digest(folder / edited_name) != expected["sha256"]:
                raise ValueError("Comparison artifact differs from recorded selection or image")
        images, titles, paths = [], [], []
        if generated:
            if metadata["executed"] != resolved:
                raise ValueError("Figure would display an edit with different selected IDs")
            images.append(panel(folder/"source_A.png"))
            titles.append("Source (A)")
            paths.append(folder/"source_A.png")
            if resolved["operation"] in ("transfer", "replace"):
                images[0] = outlined(folder/"source_A.png",folder/"target_prediction_mask.png")
                images.append(outlined(folder/"donor_B.png",folder/"donor_prediction_mask.png"))
                titles.append("Donor (B)")
                paths.append(folder/"donor_B.png")
            else:
                images[0] = outlined(folder/"source_A.png",folder/"target_prediction_mask.png",
                                     fill_alpha=REMOVE_MASK_ALPHA)
            for name, title in (("reconstruction_native.png","Reconstruction"),(edited_name,"Edited result")):
                images.append(panel(folder/name));titles.append(title);paths.append(folder/name)
        # All requests, including failures, get an explicitly status-labelled review.
        review = Image.new("RGB",(4*320,400),"white")
        draw = ImageDraw.Draw(review)
        draw.text((8,5),f"{query['id']} | {selection['status']} | {metadata['status']} | actual external-LLM plan",fill="black")
        draw.text((8,24),display_prompt,fill="black")
        draw.text((8,43),str(selection["plan"]),fill="black")
        if images:
            for column, image in enumerate(images):
                draw.text((column*320+8,67),titles[column],fill="black")
                review.paste(image.resize((320,320),Image.Resampling.LANCZOS),(column*320,80))
        else:
            draw.text((8,95),"NO EDIT GENERATED; no manually chosen or reused fallback image.",fill="red")
        all_reviews.append(review)
        operation = resolved["operation"] if resolved else "none/invalid"
        target = resolved["target"]["id"] if resolved else "—"
        donor = resolved["donor"]["id"] if resolved and resolved["donor"] else "—"
        target_category = resolved["target"]["category"] if resolved else "—"
        donor_category = resolved["donor"]["category"] if resolved and resolved["donor"] else "—"
        lines.append(f"| {query['id']} | {query['request']} | {operation} | {target} / {donor} | {target_category} / {donor_category} | {metadata['status']} |")
        if index >= 6:
            continue
        row,column = divmod(index,2)
        top = height-0.55-row*3.85
        left, available = lefts[column], widths[column]
        start = left+(available-len(images)*side-max(0,len(images)-1)*gap)/2
        for j,(image,title) in enumerate(zip(images,titles)):
            x = start+j*(side+gap)
            ax = fig.add_axes([x/width,(top-side)/height,side/width,side/height])
            ax.imshow(image,interpolation="none");ax.set_axis_off()
            if row == 0:
                fig.text((x+side/2)/width,(top+0.13)/height,title,fontsize=21,ha="center",va="bottom")
        if not images:
            fig.text((left+available/2)/width,(top-side/2)/height,
                "No edit generated\n"+metadata["status"],fontsize=23,color="#c72b2b",ha="center")
        artists.extend(colored_prompt(fig,display_prompt,left+available/2,top-side-0.19,
                                      available-0.15,width,height))
        records.append(dict(query=query,selection=selection["plan"],resolved=resolved,status=metadata["status"],
            display_prompt=display_prompt, display_prompt_is_actual_selector_input=display_prompt == query["request"],
            execution_mode=execution_mode,
            source_selection=str((root/"plans"/(query["id"]+".json")).resolve()),
            selection_sha256=metadata["selection_sha256"],
            native_panels=[dict(path=str(p.resolve()),sha256=file_digest(p)) for p in paths]))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        box = artist.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width*fig.dpi or box.y1 > height*fig.dpi:
            raise ValueError("Figure prompt clipped")
    out = Path(figure_output_dir) if figure_output_dir else root/"figure"
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png","pdf","svg"):
        fig.savefig(out/f"object_editing_language_selected.{suffix}",dpi=200,facecolor="white")
    fig.savefig(out/"preview.jpg",dpi=100,facecolor="white",pil_kwargs={"quality":95})
    plt.close(fig)
    sheet = Image.new("RGB",(1280,400*len(all_reviews)),"white")
    for i,review in enumerate(all_reviews):
        sheet.paste(review,(0,i*400))
    review_path = out/"review_all_requests.jpg" if figure_output_dir else root/"review_all_requests.jpg"
    sheet.save(review_path,quality=95)
    write_json(out/"manifest.json",dict(source_experiment=m["fingerprint"],
        actual_recorded_llm_decisions=True, native_edits_regenerated_from_selected_ids=True,
        pixels_retouched=False, generated_image_pixels_retouched=False,
        display_overlays=dict(mask_source="actual selected owner's predicted mask (not GT)",
            remove_source_fill_alpha=REMOVE_MASK_ALPHA, replace_fill_alpha=0,
            contour_color=list(OUTLINE_COLOR), contour_inside_pixels=4, contour_outside_pixels=2),
        manual_corrections=False, prompts_are_actual_selector_inputs=all(r["display_prompt_is_actual_selector_input"] for r in records),
        figure_policy_overrides=overrides,
        figure_requests=records, all_requests_review=str(review_path.resolve())))
    report_path = out/"selection_report.md" if figure_output_dir else root/"selection_report.md"
    report_path.write_text("\n".join(lines)+"\n")
    compose_replacement_comparisons(root, m, out)
    print(out/"object_editing_language_selected.png",flush=True)


def compose_replacement_comparisons(root, manifest, out):
    """Display native alternatives for the same actual selection and diffusion noise."""
    for query in manifest["requests"]:
        folder = root / "edits" / query["id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        if metadata["status"] != "generated" or not metadata.get("replacement_comparison"):
            continue
        selection, resolved = checked_selection(root, manifest, query)
        if resolved["operation"] != "replace" or metadata["executed"] != resolved:
            raise ValueError("Replacement comparison differs from actual selection")
        paths = [folder / name for name in ("source_A.png", "donor_B.png",
            "reconstruction_native.png", "visual_only_native.png", "full_bundle_native.png")]
        images = [outlined(paths[0], folder / "target_prediction_mask.png"),
                  outlined(paths[1], folder / "donor_prediction_mask.png")]
        images.extend(panel(path) for path in paths[2:])
        titles = ["Source (A)", "Donor (B)", "Reconstruction",
                  "Visual-only\n$(s_A, M_B)$", "Full bundle\n$(s_B, M_B)$"]
        fig, axes = plt.subplots(1, 5, figsize=(15, 4.25), dpi=200)
        for ax, image, title in zip(axes, images, titles):
            ax.imshow(image, interpolation="none")
            ax.set_axis_off()
            ax.set_title(title, fontsize=17, pad=12)
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.18, top=0.79, wspace=0.04)
        fig.text(0.5, 0.075, query["request"], ha="center", fontsize=16)
        stem = "replacement_comparison_" + query["id"]
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / (stem + "." + suffix), dpi=200, facecolor="white")
        plt.close(fig)
        write_json(out / (stem + ".json"), dict(query=query, actual_selection=selection["plan"],
            same_initial_noise=True, unchanged_other_owners=True,
            visual_only="source semantic + donor visual memory",
            full_bundle="donor semantic + donor visual memory",
            native_panels=[dict(path=str(path.resolve()), sha256=file_digest(path)) for path in paths]))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output_dir",default="/home/jovyan/PGOT/outputs/language_object_editing_gpu1")
    p.add_argument("--figure_output_dir")
    p.add_argument("--display_overrides", help="JSON file choosing audited native comparison panels and display captions")
    args = p.parse_args()
    overrides = json.loads(Path(args.display_overrides).read_text()) if args.display_overrides else None
    compose(args.output_dir, args.figure_output_dir, overrides)
