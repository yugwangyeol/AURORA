"""Deliver explicitly reviewed native examples without altering generated pixels."""
import argparse
import json
from pathlib import Path
import re
import shutil

from PIL import Image, ImageDraw

from pgot.eval.run_natural_object_selection import write_json

ASSETS=("figure.png","ablation.png","source_A.png","donor_B.png",
        "reconstruction_native.png","text_only_native.png","semantic_only_native.png",
        "visual_only_native.png","bundle_swap_native.png",
        "target_prediction_mask.png","donor_prediction_mask.png","metadata.json")
COMPARABLE=("checkpoint","checkpoint_signature","dtype","seed","steps","guidance_scale",
            "eval_size","max_objects","max_caption_tokens","conditions","gt_inference_input",
            "external_selector","pixel_paste","reader_prior","writer_rewrite","spatial_warp")


def stack(panels,path):
    if not panels:raise ValueError("Empty gallery")
    if len({p.width for p in panels})!=1:raise ValueError("Figure widths differ")
    sheet=Image.new("RGB",(panels[0].width,sum(p.height for p in panels)),"white")
    offset=0
    for panel in panels:
        sheet.paste(panel,(0,offset));offset+=panel.height
    sheet.save(path,quality=94)


def assemble(selection,output_dir):
    selection=Path(selection)
    chosen=json.loads(selection.read_text())
    entries=chosen["examples"]
    if not entries or len({r["pair_id"] for r in entries})!=len(entries):
        raise ValueError("Empty/duplicate selected examples")
    out=Path(output_dir)
    if (out/"selection.json").exists():
        previous=json.loads((out/"selection.json").read_text())
        if previous["requested_selection"]!=chosen:
            raise ValueError("Existing gallery selection differs; choose a fresh output directory")
    loaded=[];reference=None
    for row in entries:
        if not re.fullmatch(r"[A-Za-z0-9_-]+",row["pair_id"]):raise ValueError("Unsafe pair ID")
        source=Path(row["source_root"]).resolve()
        if source==out.resolve():raise ValueError("Gallery cannot overwrite its source root")
        manifest=json.loads((source/"manifest.json").read_text())
        settings={k:manifest["config"][k] for k in COMPARABLE}
        if settings["writer_rewrite"] or settings["reader_prior"] or settings["pixel_paste"]:
            raise ValueError("Only unchanged native-Reader transfer cases belong in this gallery")
        if reference is not None and settings!=reference:raise ValueError("Selected examples use different inference protocols")
        reference=settings
        folder=source/"pairs"/row["pair_id"]
        metadata=json.loads((folder/"metadata.json").read_text())
        if metadata["fingerprint"]!=manifest["fingerprint"] or metadata["pair_id"]!=row["pair_id"]:
            raise ValueError("Pair provenance mismatch")
        if not all((folder/name).is_file() for name in ASSETS):raise ValueError("Incomplete generated example")
        loaded.append((row,folder,metadata,manifest["fingerprint"]))
    out.mkdir(parents=True,exist_ok=True)
    panels=[];review_panels=[];records=[]
    for rank,(row,folder,metadata,fingerprint) in enumerate(loaded,1):
        dest=out/f"{rank:02}_{row['pair_id']}";dest.mkdir(parents=True,exist_ok=True)
        for name in ASSETS:shutil.copy2(folder/name,dest/name)
        picture=Image.open(dest/"figure.png").convert("RGB");panels.append(picture)
        labelled=Image.new("RGB",(picture.width,picture.height+40),"white");labelled.paste(picture,(0,40))
        draw=ImageDraw.Draw(labelled)
        draw.text((8,5),f"{rank}. {row['label']} | selected qualitative example; not benchmark accuracy",fill="black")
        draw.text((8,23),row["note"],fill="black")
        labelled.save(dest/"review.png");review_panels.append(labelled)
        records.append(dict(row,source_fingerprint=fingerprint,source_metadata=metadata,directory=str(dest.resolve())))
    stack(panels,out/"overview.jpg");stack(review_panels,out/"review_overview.jpg")
    write_json(out/"selection.json",dict(requested_selection=chosen,comparable_protocol=reference,
        examples=records,selection_is_qualitative=True,blind_review=False,
        generated_outputs_copied_without_pixel_changes=True))
    lines=["# Selected native object-transfer examples","",chosen["scope"],"",
           "Qualitative candidate selection, not an unbiased benchmark or a success-rate estimate.","",
           "| Row | Pair | Note | Figure |","| --- | --- | --- | --- |"]
    for rank,row in enumerate(records,1):
        lines.append(f"| {rank} | {row['pair_id']} | {row['note']} | [figure]({rank:02}_{row['pair_id']}/figure.png) |")
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print(f"Delivered {len(records)} native examples: {out/'overview.jpg'}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection",required=True)
    parser.add_argument("--output_dir",required=True)
    args=parser.parse_args();assemble(args.selection,args.output_dir)


if __name__=="__main__":main()
