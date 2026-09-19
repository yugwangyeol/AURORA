"""Replay cached image-only AR captions and mine qualitative split/merge examples.

All models run in separate processes with their established validation adapters.
Selection is explicitly GT-assisted qualitative mining, NOT a new benchmark.
Raw hard predictions are retained; GT matching changes DISPLAY COLORS only.
"""

import argparse
import colorsys
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[2]
MODELS = ("adaslot", "metaslot", "coda", "ours")
NAMES = dict(zip(MODELS, ("AdaSlot", "MetaSlot", "CODA", "Ours (AR)")))


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def prepare(args):
    out = Path(args.output_dir)
    if (out / "candidates.json").exists():
        raise FileExistsError("Candidate manifest exists; reuse it with inference/select stages")
    voc_ids = Path(args.voc_root, "sets/val.txt").read_text().split()
    entries = []
    for dataset in ("coco", "voc"):
        source = PROJECT / "outputs/segeval_pgot_memory8_full_coco_voc_ar" / f"{dataset}_ar"
        summary = json.loads((source / "summary.json").read_text())
        if Path(summary["ckpt"]).resolve() != Path(args.model_path).resolve():
            raise ValueError("Cached AR checkpoint differs from requested model")
        records = read_jsonl(source / "per_image.jsonl")
        captions = {r["image_id"]: r for r in read_jsonl(source / "generated_captions.jsonl")}
        selected = set()
        for kind, predicate in (("over", lambda n: 1 <= n <= 6), ("under", lambda n: n >= 7)):
            candidates = [r for r in records if predicate(r["gt_objects"]) and r["mIoU"] >= .35]
            # Equal search budget per type, sorted by own quality, NOT by chosen seeds.
            candidates.sort(key=lambda r: (r["mIoU"] + .12 * r["fARI"], r["mIoU"]), reverse=True)
            for row in candidates[:args.candidates_per_type]:
                iid = row["image_id"]
                if iid in selected:
                    continue
                selected.add(iid)
                sample_id = str(iid) if dataset == "coco" else voc_ids[int(iid)]
                caption = captions[iid]
                entries.append({"dataset": dataset, "id": sample_id, "source_image_id": iid,
                                "screening_type": kind, "prior_metrics": row,
                                "ar_caption": caption, "caption_source": str(source / "generated_captions.jsonl"),
                                "caption_sha256": hashlib.sha256(caption["text"].encode()).hexdigest()})
    write_json(out / "candidates.json", {"checkpoint": args.model_path, "seed": args.seed,
               "selection": "GT-assisted qualitative shortlist; not representative benchmark",
               "candidate_budget_per_dataset_type": args.candidates_per_type,
               "entries": entries})
    print("Prepared", len(entries), "candidate images", flush=True)


def make_dataset(args, name):
    from pgot.eval.eval_baselines_voc_zeroshot import VOCZeroShot
    from pgot.eval.coco_baseline_dataset import COCOBaselineDataset
    if name == "voc":
        return VOCZeroShot(args.voc_root, 512)
    return COCOBaselineDataset(args.coco_root, str(PROJECT / "data/coco_inst_mask_cache_coda512"),
                               str(PROJECT / "data/coco_cat_mask_cache_coda512"), 512)


def contingency(gt, pred, ignore=None):
    gt, pred = np.asarray(gt, dtype=np.int64).copy(), np.asarray(pred, dtype=np.int64).copy()
    if ignore is not None:
        # Exact PGOT metric policy; overlaps become a separate predicted label.
        invalid = np.asarray(ignore, dtype=bool)
        gt[invalid] = 0
        pred[invalid] = int(pred.max()) + 1
    shape = (int(gt.max()) + 1, int(pred.max()) + 1)
    inter = np.bincount((gt * shape[1] + pred).ravel(), minlength=shape[0]*shape[1]).reshape(shape).astype(float)
    union = inter.sum(1)[:, None] + inter.sum(0)[None, :] - inter
    iou = np.divide(inter, union + 1e-8)
    return inter, iou


def diagnose(gt, pred, ignore, min_area=786, fraction=.20):
    """Illustrative split/merge counts, NOT standard published metrics.

    Split: a sizable GT object has >=20% of its valid pixels in >=2 labels.
    Merge: >=2 sizable GT objects each put >=20% of their pixels in one label.
    For all models every hard label is considered; no inferred BG is removed.
    Ignore pixels are excluded; GT background fragmentation cannot count as split.
    """
    valid = ~np.asarray(ignore, dtype=bool)
    g, p = np.asarray(gt)[valid], np.asarray(pred)[valid]
    shape = (int(np.asarray(gt).max()) + 1, int(np.asarray(pred).max()) + 1)
    inter = np.bincount(g * shape[1] + p, minlength=shape[0]*shape[1]).reshape(shape).astype(float)
    areas = inter.sum(1)
    significant = np.flatnonzero((areas >= min_area) & (np.arange(len(areas)) > 0))
    coverage = inter[significant] / np.maximum(areas[significant, None], 1)
    edges = coverage >= fraction
    split_labels = significant[(edges.sum(1) >= 2)].tolist()
    merged_labels = np.flatnonzero(edges.sum(0) >= 2).tolist()
    return {"significant_gt_objects": len(significant), "split_count": len(split_labels),
            "merge_count": len(merged_labels), "split_gt_labels": split_labels,
            "merged_pred_labels": merged_labels,
            "merge_gt_labels": {str(label): significant[edges[:, label]].tolist() for label in merged_labels},
            "min_gt_pixels": min_area, "minimum_object_fraction": fraction}


def score(gt, pred, ignore):
    from sklearn.metrics import adjusted_rand_score
    inter, iou = contingency(gt, pred, ignore)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    fg = (np.asarray(gt) > 0) & ~np.asarray(ignore, dtype=bool)
    ari = float(adjusted_rand_score(np.asarray(gt)[fg], np.asarray(pred)[fg])) if fg.any() else 0.
    return {"fARI": ari, "mIoU": float(iou[rows, cols].sum() / len(iou)),
            "mBO": float(iou[1:].max(1).mean()) if len(iou) > 1 else 0.,
            "pred_segments": len(np.unique(pred)), **diagnose(gt, pred, ignore)}


def inference(args):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    torch.set_num_threads(3)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for this bounded qualitative inference run")
    out = Path(args.output_dir)
    manifest = json.loads((out / "candidates.json").read_text())
    if manifest["checkpoint"] != args.model_path or manifest["seed"] != args.seed:
        raise ValueError("Manifest checkpoint/seed mismatch")
    entries = manifest["entries"][:args.limit] if args.limit else manifest["entries"]
    pending = [e for e in entries if not (out / "cache" / e["dataset"] / e["id"] / f"{args.model}.json").exists()]
    if not pending:
        print(args.model, "already cached", flush=True)
        return
    if args.model == "ours":
        from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
        from pgot.eval.pgot_inference import pgot_forward_eval
        from pgot.model.pgot_utils import build_pred_mask_ovt_owner_eval
        model, tokenizer, device, dtype = load_model_and_tokenizer(args)
        towers = model.get_vision_tower_aux_list()
        info = {"checkpoint": args.model_path, "caption_mode": "cached_image_only_AR",
                "dtype": args.dtype, "readout": "ovt_owner", "register_eval_route": "unrestricted",
                "gt_caption_used": False, "gt_mask_passed_to_model": False,
                "batch_size": 1, "max_objects": 50}
    else:
        from pgot.eval.eval_baselines_voc_zeroshot import BASELINES
        baseline = BASELINES[args.model](args, "cuda")
        info = dict(baseline.info, seed=args.seed, batch_size=1,
                    randomness="fixed per-image seed, no retry or best-seed selection")
    write_json(out / f"{args.model}_model_info.json", info)
    datasets = {name: make_dataset(args, name) for name in {e["dataset"] for e in pending}}
    from pgot.train.pgot_dataset import _coda_center_crop_image
    indices = {name: {str(iid): i for i, iid in enumerate(ds.ids)} for name, ds in datasets.items()}
    for entry in tqdm(pending, desc=args.model):
        ds = datasets[entry["dataset"]]
        item = ds[indices[entry["dataset"]][entry["id"]]]
        path = out / "cache" / entry["dataset"] / entry["id"]
        path.mkdir(parents=True, exist_ok=True)
        gt, ignore = item["instance"].numpy(), item["overlap"].numpy().astype(bool)
        cropped = _coda_center_crop_image(Image.fromarray(item["image"]), 512)
        if not (path / "input.png").exists():
            cropped.save(path / "input.png")
            np.savez_compressed(path / "gt.npz", gt=gt.astype(np.int16), ignore=ignore)
        if args.model == "ours":
            text = entry["ar_caption"]["text"]
            ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)[None]
            positions = (ids[0] == tokenizer.convert_tokens_to_ids("<ovt>")).nonzero().flatten()
            if len(positions) != entry["ar_caption"]["ovt_count"] or len(positions) > 50:
                raise ValueError("Cached AR OVT count mismatch/capacity overflow")
            pos, valid = torch.zeros((1, 50), dtype=torch.long), torch.zeros((1, 50), dtype=torch.bool)
            pos[0, :len(positions)], valid[0, :len(positions)] = positions, True
            thing = valid.clone()
            chunks = text.split("<ovt>")[:-1]
            for i, chunk in enumerate(chunks):
                thing[0, i] = chunk.rfind("<thing>") > chunk.rfind("<stuff>")
            with torch.inference_mode():
                if len(positions):
                    image = towers[0].image_processor.preprocess(cropped, return_tensors="pt")["pixel_values"]
                    target = (towers[1] if len(towers)>1 else towers[0]).image_processor.preprocess(cropped, return_tensors="pt")["pixel_values"]
                    result = pgot_forward_eval(model, images=image, target_images=target,
                             caption_input_ids=ids, caption_attention_mask=torch.ones_like(ids, dtype=torch.bool),
                             ovt_positions_in_caption=pos, ovt_valid_mask=valid)
                    pred = build_pred_mask_ovt_owner_eval(result["ovt_object_probs"], result["ovt_void_probs"],
                             valid.to(device), thing.to(device), 512, 1)[0].cpu().numpy()
                    del result
                else:
                    pred = np.zeros_like(gt)
            diagnostics = {"generated_objects": entry["ar_caption"]["object_count"],
                           "caption_sha256": entry["caption_sha256"]}
        else:
            # Stable seeds per image allow resume and do not optimize stochastic outputs.
            image_seed = int.from_bytes(hashlib.sha256(f"{args.seed}/{entry['dataset']}/{entry['id']}".encode()).digest()[:4], "little")
            torch.manual_seed(image_seed)
            torch.cuda.manual_seed_all(image_seed)
            np.random.seed(image_seed)
            random.seed(image_seed)
            soft = baseline([item["image"]])
            pred = F.interpolate(soft.float(), size=(512, 512), mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy()
            diagnostics = {"image_seed": image_seed}
            if getattr(baseline, "last_diagnostics", None):
                diagnostics.update(baseline.last_diagnostics[0])
        np.savez_compressed(path / f"{args.model}.npz", prediction=pred.astype(np.int16))
        write_json(path / f"{args.model}.json", dict(score(gt, pred, ignore), **diagnostics))
    print("Completed", args.model, "images", len(pending), flush=True)


def color_map(labels, gt, ignore):
    _, iou = contingency(gt, labels, None)
    rr, cc = linear_sum_assignment(iou, maximize=True)
    assignment = {int(c): int(r) for r, c in zip(rr, cc)}
    palette = {}
    for label in np.unique(labels):
        target = assignment.get(int(label), int(gt.max()) + 1 + int(label))
        if target == 0:
            color = (45, 45, 45)
        else:
            color = tuple(int(v*255) for v in colorsys.hsv_to_rgb((target*.61803398875)%1, .72, .94))
        palette[int(label)] = color
    rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label, color in palette.items():
        rgb[labels == label] = color
    rgb[ignore] = (180, 180, 180)
    return rgb


def render(path, scores, destination, title):
    destination.mkdir(parents=True, exist_ok=True)
    data = np.load(path / "gt.npz")
    gt, ignore = data["gt"], data["ignore"]
    original = np.asarray(Image.open(path / "input.png").convert("RGB"))
    fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font, small = ImageFont.truetype(fontpath, 19), ImageFont.truetype(fontpath, 15)
    except OSError:
        font = small = ImageFont.load_default()
    size, header = 320, 70
    grid = Image.new("RGB", (size*6, header+size*2+35), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((8, 8), title, font=font, fill="black")
    images = [("Input", original, ""), ("GT", color_map(gt, gt, ignore), "")]
    for model in MODELS:
        pred = np.load(path / f"{model}.npz")["prediction"]
        images.append((NAMES[model], color_map(pred, gt, ignore),
                      f"mIoU {100*scores[model]['mIoU']:.1f} | split {scores[model]['split_count']} merge {scores[model]['merge_count']}"))
    for i, (name, rgb, note) in enumerate(images):
        draw.text((i*size+8, 36), name, font=font, fill="black")
        Image.fromarray(rgb).save(destination / f"{i}_{name.split()[0].lower()}_map.png")
        overlay = original if i==0 else (original*.48+rgb*.52).astype(np.uint8)
        Image.fromarray(overlay).save(destination / f"{i}_{name.split()[0].lower()}_overlay.png")
        grid.paste(Image.fromarray(rgb).resize((size,size)), (i*size,header))
        grid.paste(Image.fromarray(overlay).resize((size,size)), (i*size,header+size))
        draw.text((i*size+3, header+size*2+7), note, font=small, fill="black")
    grid.save(destination / "comparison.png")


def select(args):
    out = Path(args.output_dir)
    entries = json.loads((out / "candidates.json").read_text())["entries"]
    groups = {f"{ds}_{kind}": [] for ds in ("coco", "voc") for kind in ("over", "under")}
    complete = []
    for entry in entries:
        path = out / "cache" / entry["dataset"] / entry["id"]
        if not all((path / f"{m}.json").exists() for m in MODELS):
            continue
        scores = {m: json.loads((path / f"{m}.json").read_text()) for m in MODELS}
        complete.append(dict(entry, scores=scores))
        own = scores["ours"]
        gap = own["mIoU"]-max(scores[m]["mIoU"] for m in MODELS[:-1])
        for kind, field, opposite in (("over","split_count","merge_count"),("under","merge_count","split_count")):
            failures = sum(scores[m][field] > own[field] for m in MODELS[:-1])
            strict = (failures == 3 and own["split_count"] == 0 and own["merge_count"] == 0
                      and own["mIoU"] >= .60 and gap >= .10)
            # Never relabel relaxed examples as strict ones. Keep separate review pools.
            rank = (int(strict), failures, own["mIoU"] >= .6, gap,
                    -own[field], min(scores[m][field] for m in MODELS[:-1]), own["mIoU"])
            record = dict(entry, scores=scores, requested_type=kind, strict_eligible=bool(strict),
                          baseline_failure_count=failures, own_mIoU_gap=gap, ranking=list(rank))
            if failures >= 2 and gap > .03 and own["mIoU"] >= .4:
                groups[f"{entry['dataset']}_{kind}"].append(record)
    write_json(out / "all_scores.json", complete)
    selected = {}
    for group, candidates in groups.items():
        candidates.sort(key=lambda r: tuple(r["ranking"]), reverse=True)
        strict_candidates = [r for r in candidates if r["strict_eligible"]]
        selected[group] = {"strict_available":len(strict_candidates), "review_available":len(candidates),
                           "examples":candidates[:args.examples_per_type]}
        for index, entry in enumerate(candidates[:args.examples_per_type]):
            src = out / "cache" / entry["dataset"] / entry["id"]
            dst = out / "selected" / group / f"{index+1:02d}_{entry['id']}"
            title = f"{group} | {entry['id']} | {'STRICT candidate' if entry['strict_eligible'] else 'RELAXED candidate - review required'}"
            render(src, entry["scores"], dst, title)
            write_json(dst / "metadata.json", entry)
        # Small contact sheets are for manual audit, not the paper figure.
        previews = []
        for index, entry in enumerate(candidates[:args.examples_per_type]):
            img = Image.open(out / "selected" / group / f"{index+1:02d}_{entry['id']}" / "comparison.png")
            previews.append(img.resize((960, int(img.height/2))))
        if previews:
            sheet = Image.new("RGB", (960, sum(im.height for im in previews)), "white")
            y=0
            for im in previews:
                sheet.paste(im,(0,y));y+=im.height
            sheet.save(out / "selected" / f"{group}_contact_sheet.jpg", quality=90)
        print(group,"strict",len(strict_candidates),"review pool",len(candidates),"exported",len(previews),flush=True)
    write_json(out / "selection.json", selected)
    with (out / "scores.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["dataset","id","model","mIoU","fARI","split","merge"])
        for entry in complete:
            for m,s in entry["scores"].items(): writer.writerow([entry["dataset"],entry["id"],m,s["mIoU"],s["fARI"],s["split_count"],s["merge_count"]])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["prepare","infer","select"],required=True)
    p.add_argument("--output_dir",required=True)
    p.add_argument("--model",choices=MODELS,default="ours")
    p.add_argument("--model_path",default=str(PROJECT/"checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000"))
    p.add_argument("--dtype",choices=["fp32","bf16"],default="fp32")
    p.add_argument("--diffusion_inference_steps",type=int,default=50)
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--candidates_per_type",type=int,default=80)
    p.add_argument("--examples_per_type",type=int,default=10)
    p.add_argument("--limit",type=int)
    p.add_argument("--voc_root",default="/home/jovyan/data/voc")
    p.add_argument("--coco_root",default="/home/jovyan/data/coco")
    p.add_argument("--adaslot_root",default="/home/jovyan/baselines/AdaSlot")
    p.add_argument("--adaslot_ckpt",default="/home/jovyan/baselines/ckpt/adaslot-coco.ckpt")
    p.add_argument("--metaslot_root",default="/home/jovyan/baselines/MetaSlot")
    p.add_argument("--metaslot_cfg",default="/home/jovyan/baselines/MetaSlot/Config/config-metaslot/dinosaur_r-coco.py")
    p.add_argument("--metaslot_ckpt",default="/home/jovyan/baselines/ckpt/metaslot-dinosaur-coco.pth")
    p.add_argument("--coda_root",default="/home/jovyan/coda")
    p.add_argument("--coda_ckpt",default="/home/jovyan/coda/pretrained/coda/coco")
    args=p.parse_args()
    if min(args.candidates_per_type,args.examples_per_type)<1 or (args.limit is not None and args.limit<1): p.error("Counts must be positive")
    {"prepare":prepare,"infer":inference,"select":select}[args.stage](args)


if __name__ == "__main__":
    main()
