"""GT-assisted qualitative mining of native CTRL-O versus cached Ours AR.

Category-only protocol uses the authors' released LLM2Vec embeddings, NOT a
replacement encoder. Ours is its original AR caption/region pair; CTRL-O is
conditioned on that pair's category. This is not arbitrary text-query Ours.
"""
import argparse
import hashlib
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.figure_segmentation_examples import write_json


def queries(entry, pred, embeddings):
    slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
    # The cached figure readout compacts THING labels, unlike the AR router.
    things = [s for s in slots if s["is_thing"]]
    counts = Counter(s["normalized_category"] for s in things if s["parsed"])
    return [dict(s, prediction_label=i + 1, query=s["normalized_category"])
            for i, s in enumerate(things)
            if s["parsed"] and counts[s["normalized_category"]] == 1
            and s["normalized_category"] in embeddings and (pred == i + 1).any()][:7]


def iou(a, b, valid):
    a, b = np.asarray(a, bool) & valid, np.asarray(b, bool) & valid
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.


def load_ctrlo(args):
    import torch
    import hydra_zen
    from omegaconf import OmegaConf
    sys.path.insert(0, args.ctrlo_repo)
    import ocl.cli._config  # Registers official config resolvers.
    from ocl.utils.routing import Combined
    cfg = OmegaConf.load(Path(args.checkpoint_dir) / "config.yaml")
    # Frozen backbone weights are present in the released checkpoint.
    cfg.models.feature_extractor.pretrained = False
    model = Combined(**hydra_zen.instantiate(cfg.models, _convert_="all"))
    path = Path(args.checkpoint_dir) / "pretrained_model.ckpt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = {k[len("models."):]: v for k, v in checkpoint["state_dict"].items()
               if k.startswith("models.")}
    model.load_state_dict(weights, strict=True)
    info = {"checkpoint": str(path), "global_step": checkpoint.get("global_step"),
            "loaded_model_tensors": len(weights), "strict": True,
            "conditioning": str(cfg.models.conditioning._target_),
            "query_protocol": "category-only; official precomputed LLM2Vec embeddings",
            "image_preprocessing": "shared 512 center-crop; official PIL bilinear 224 square resize + ImageNet normalize",
            "slots": 7, "seed": args.seed, "dtype": "float32",
            "point_input": "-1 sentinel as official visualization; LangConditioning ignores points",
            "selection": "GT-assisted qualitative success/failure mining, NOT benchmark results"}
    return model.cuda().eval(), info


def infer(args):
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("A free CUDA device is required")
    source, out = Path(args.source_dir), Path(args.output_dir)
    manifest = json.loads((source / "candidates.json").read_text())
    with open(Path(args.ctrlo_repo) / "scripts/datasets/data/category_name_to_llama3_emb.pkl", "rb") as stream:
        embeddings = pickle.load(stream)  # Official author-distributed asset.
    model, info = load_ctrlo(args)
    info["ours_checkpoint"] = manifest["checkpoint"]
    info["ours_protocol"] = "cached image-only AR; its own category/caption-associated mask, not GT-routed"
    write_json(out / "protocol.json", info)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    entries = [e for e in manifest["entries"] if e["dataset"] == "coco"]
    if args.limit:
        entries = entries[:args.limit]
    for entry in tqdm(entries, desc="native CTRL-O"):
        dest = out / "cache" / entry["id"]
        if (dest / "metadata.json").exists():
            continue
        cache = source / "cache/coco" / entry["id"]
        ours = np.load(cache / "ours.npz")["prediction"]
        selected = queries(entry, ours, embeddings)
        if not selected:
            continue
        prompts = [s["query"] for s in selected] + ["other"] * (7 - len(selected))
        image_seed = int.from_bytes(hashlib.sha256(f'{args.seed}/coco/{entry["id"]}'.encode()).digest()[:4], "little")
        torch.manual_seed(image_seed)
        torch.cuda.manual_seed_all(image_seed)
        image = Image.open(cache / "input.png").convert("RGB")
        inputs = {
            "image": transform(image)[None].cuda(), "batch_size": 1,
            "name_embedding": torch.from_numpy(np.stack([embeddings[p] for p in prompts]))[None].float().cuda(),
            "contrastive_loss_mask": torch.tensor([[1] * len(selected) + [0] * (7 - len(selected))], device="cuda"),
            "bbox_centroids": torch.full((1, 7, 2), -1., device="cuda"),
            "instance_bbox": torch.full((1, 7, 4), -1., device="cuda")}
        with torch.inference_mode():
            outputs = model(inputs={"input": inputs})
            masks = outputs["object_decoder"].masks_as_image
            if masks.shape != (1, 7, 224, 224):
                raise ValueError(f"Unexpected official mask shape: {masks.shape}")
            soft = F.interpolate(masks, (512, 512), mode="bilinear", align_corners=False)[0].cpu().numpy()
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "prediction.npz", prediction=soft.argmax(0).astype(np.uint8), soft=soft.astype(np.float16))
        write_json(dest / "metadata.json", {"id": entry["id"], "source_cache": str(cache),
                   "prompts": prompts, "queries": selected, "seed": image_seed,
                   "ar_caption_sha256": entry["caption_sha256"], "ar_text": entry["ar_caption"]["text"]})


def target_masks(coco, image_id):
    from pgot.eval.build_coco_category_cache import preprocess_coda
    result = {}
    for ann in coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=False)):
        if ann.get("ignore") or ann.get("area", 0) <= 0:
            continue
        mask = preprocess_coda(coco.annToMask(ann), 512).astype(bool)
        if mask.any():
            name = coco.cats[ann["category_id"]]["name"]
            result.setdefault(name, []).append((ann["id"], mask))
    return result


def overlay(image, mask):
    rgb = np.asarray(image).astype(float)
    # Same color/alpha for BOTH models. No GT alignment or mask modification.
    rgb[mask] = .55 * rgb[mask] + .45 * np.array([235, 55, 65])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def render(row, dest):
    meta = row["meta"]
    cache = Path(meta["source_cache"])
    rgb = Image.open(cache / "input.png").convert("RGB")
    ours = np.load(cache / "ours.npz")["prediction"] == row["prediction_label"]
    ctrl = np.load(Path(row["ctrl_cache"]) / "prediction.npz")["prediction"] == row["query_index"]
    panels = [rgb, overlay(rgb, row["target"]), overlay(rgb, ours), overlay(rgb, ctrl)]
    titles = ["Image", "GT (reference only)", f'Ours AR | IoU {row["ours_iou"]:.2f}',
              f'CTRL-O | IoU {row["ctrlo_iou"]:.2f}']
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    canvas = Image.new("RGB", (2048, 586), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), f'COCO {row["id"]} | Query: {row["query"]} | category-only | selected qualitative example', fill="black", font=font)
    dest.mkdir(parents=True, exist_ok=True)
    for i, (panel, title) in enumerate(zip(panels, titles)):
        canvas.paste(panel, (512*i, 74))
        draw.text((512*i+12, 42), title, fill="black", font=font)
        panel.save(dest / f'{i}_{("input", "gt", "ours", "ctrlo")[i]}.png')
    canvas.save(dest / "comparison.png")
    paper = Image.new("RGB", (1536, 550), "white")
    paper_draw = ImageDraw.Draw(paper)
    for i, (panel, title) in enumerate(zip([panels[0], panels[2], panels[3]],
            ["Image", f'Ours (AR): {row["query"]}', f'CTRL-O: {row["query"]}'])):
        paper_draw.text((512*i+12, 7), title, fill="black", font=font)
        paper.paste(panel, (512*i, 38))
    paper.save(dest / "paper_panel.png")
    np.savez_compressed(dest / "masks.npz", gt=row["target"], ours=ours, ctrlo=ctrl)
    record = {k: v for k, v in row.items() if k not in {"target", "meta"}}
    record["original_ar_description"] = meta["queries"][row["query_index"]]["description"]
    record["ctrlo_input_prompts"] = meta["prompts"]
    record["original_ar_description_is_not_ctrlo_input"] = True
    record["source_cache"] = meta["source_cache"]
    write_json(dest / "metadata.json", record)
    return canvas


def select(args):
    from pycocotools.coco import COCO
    out = Path(args.output_dir)
    coco = COCO(args.annotations)
    rows = []
    for meta_path in sorted((out / "cache").glob("*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        cache = Path(meta["source_cache"])
        gt = np.load(cache / "gt.npz")
        valid = ~gt["ignore"].astype(bool)
        ours = np.load(cache / "ours.npz")["prediction"]
        ctrl = np.load(meta_path.parent / "prediction.npz")["prediction"]
        targets = target_masks(coco, int(meta["id"]))
        for q, slot in enumerate(meta["queries"]):
            # Without a point/description, repeated same-class targets are ambiguous.
            objects = targets.get(slot["query"], [])
            if len(objects) != 1:
                continue
            annotation_id, target = objects[0]
            if np.count_nonzero(target & valid) < 2621:
                continue
            oi, ci = iou(ours == slot["prediction_label"], target, valid), iou(ctrl == q, target, valid)
            rows.append({"id": meta["id"], "query": slot["query"], "query_index": q,
                         "prediction_label": slot["prediction_label"], "annotation_id": annotation_id,
                         "ours_iou": oi, "ctrlo_iou": ci, "gap": oi-ci,
                         "strict": oi >= .7 and ci <= .4 and oi-ci >= .35,
                         "target": target, "meta": meta, "ctrl_cache": str(meta_path.parent)})
    rows.sort(key=lambda r: (r["strict"], r["gap"], r["ours_iou"]), reverse=True)
    write_json(out / "scores.json", [{k: v for k,v in r.items() if k not in {"target", "meta"}} for r in rows])
    chosen, seen = [], set()
    ordered = rows
    if args.selection_ids:
        ordered = []
        for image_id in args.selection_ids.split(","):
            matches = [r for r in rows if r["id"] == image_id and r["strict"]]
            if not matches:
                raise ValueError(f"Manually requested image {image_id} has no strict candidate")
            ordered.append(matches[0])
    for row in ordered:
        if not row["strict"] or row["id"] in seen:
            continue
        seen.add(row["id"])
        chosen.append(row)
        if len(chosen) == args.examples:
            break
    sheets = []
    selection = []
    for rank, row in enumerate(chosen, 1):
        dest = out / args.selection_subdir / f'{rank:02}_{row["id"]}_{row["query"].replace(" ", "_")}'
        panel = render(row, dest)
        sheets.append(panel.resize((1024, 293)))
        selection.append({"rank": rank, "id": row["id"], "query": row["query"],
                          "ours_iou": row["ours_iou"], "ctrlo_iou": row["ctrlo_iou"], "file": str(dest / "comparison.png")})
    if sheets:
        sheet = Image.new("RGB", (1024, 293*len(sheets)), "white")
        for i, panel in enumerate(sheets):
            sheet.paste(panel, (0, 293*i))
        sheet.save(out / args.selection_subdir / "contact_sheet.jpg", quality=94)
    write_json(out / f"{args.selection_subdir}_selection.json", {"qualitative_not_representative": True,
               "manual_visual_curation": bool(args.selection_ids),
               "selection_ids": args.selection_ids,
               "eligible_queries": len(rows), "strict_queries": sum(r["strict"] for r in rows),
               "selected": selection, "requested": args.examples})
    print(json.dumps(selection, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["infer", "select"], required=True)
    parser.add_argument("--source_dir", default="/home/jovyan/PGOT/outputs/figure_segmentation_over_under")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/figure_ours_ctrlo_category")
    parser.add_argument("--ctrlo_repo", default="/home/jovyan/baselines/CTRL-O")
    parser.add_argument("--checkpoint_dir", default="/home/jovyan/baselines/ckpt/ctrlo")
    parser.add_argument("--annotations", default="/home/jovyan/data/coco/annotations/instances_val2017.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection_ids", default="", help="Optional visually curated image IDs in order; still require strict scores")
    parser.add_argument("--selection_subdir", default="selected")
    args = parser.parse_args()
    {"infer": infer, "select": select}[args.stage](args)


if __name__ == "__main__":
    main()
