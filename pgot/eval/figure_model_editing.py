"""Real PGOT owner-bundle editing figures, without GT captions/masks at inference.

Remove/add/replace/background-swap are executed on the learned Reader inputs.
This is NOT instruction-following fine-tuning or pixel paste/inpainting.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pgot.eval.ar_binding_router import parse_ar_slots
from pgot.eval.figure_segmentation_examples import write_json

MODES = ("remove", "add", "replace", "background")


def binary_iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.


def prepare(args):
    source, out = Path(args.source_dir), Path(args.output_dir)
    manifest = json.loads((source / "candidates.json").read_text())
    if Path(manifest["checkpoint"]).resolve() != Path(args.model_path).resolve():
        raise ValueError("Cached AR checkpoint does not match")
    pool = []
    for entry in manifest["entries"]:
        if entry["dataset"] != "coco":
            continue
        slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
        if not 1 <= len(slots) <= 5 or any(not s["is_thing"] or not s["parsed"] for s in slots):
            continue
        path = source / "cache/coco" / entry["id"]
        pred = np.load(path / "ours.npz")["prediction"]
        quality = json.loads((path / "ours.json").read_text())["mIoU"]
        if quality < .65:
            continue
        counts = Counter(s["normalized_category"] for s in slots)
        for s in slots:
            mask = pred == s["prediction_label"]
            area = float(mask.mean())
            if counts[s["normalized_category"]] == 1 and .06 <= area <= .60:
                pool.append(dict(s, id=entry["id"], entry=entry, mask=mask, area=area,
                                 pred=pred, quality=quality, cache=str(path)))
    # High-quality original decomposition, category diversity, no seed retries.
    pool.sort(key=lambda r: (r["quality"], r["area"]), reverse=True)
    targets, used_images, used_categories = [], set(), set()
    for require_diversity in (True, False):
        for row in pool:
            if row["id"] in used_images or (require_diversity and row["normalized_category"] in used_categories):
                continue
            targets.append(row)
            used_images.add(row["id"])
            used_categories.add(row["normalized_category"])
            if len(targets) >= args.candidates:
                break
        if len(targets) >= args.candidates:
            break
    tasks = []
    entries = {}
    for rank, target in enumerate(targets):
        entries[target["id"]] = target["entry"]
        categories = {s["normalized_category"] for s in parse_ar_slots(target["entry"]["ar_caption"]["text"], target["entry"]["ar_caption"]["ovt_count"])}
        donors = [d for d in pool if d["id"] != target["id"]]
        same = [d for d in donors if d["normalized_category"] == target["normalized_category"]]
        replacement = max(same or donors, key=lambda d: binary_iou(d["mask"], target["mask"]) - .2*abs(d["area"]-target["area"]))
        # Donor retains its learned native location. Avoid covering A foreground.
        available = [d for d in donors if d["normalized_category"] not in categories]
        if not available:
            continue
        def insertion_score(d):
            overlap = np.count_nonzero(d["mask"] & (target["pred"] > 0)) / max(np.count_nonzero(d["mask"]), 1)
            return d["quality"] - 1.8*overlap - .5*abs(d["area"]-.2)
        added = max(available, key=insertion_score)
        # Different scene semantics for visible background exchange.
        background = max(available, key=lambda d: d["quality"] + .2*float((d["pred"] == 0).mean()))
        for mode, donor in (("remove", None), ("add", added), ("replace", replacement), ("background", background)):
            if donor:
                entries[donor["id"]] = donor["entry"]
            tasks.append({"task_id": f"{mode}_{rank:02}_{target['id']}", "mode": mode,
                          "target_id": target["id"], "target_slot": target["slot_index"],
                          "target_category": target["normalized_category"],
                          "target_prediction_label": target["prediction_label"],
                          "donor_id": donor["id"] if donor else None,
                          "donor_slot": donor["slot_index"] if donor else None,
                          "donor_category": donor["normalized_category"] if donor else None,
                          "donor_prediction_label": donor["prediction_label"] if donor else None,
                          "placement": "donor native location; no text-controlled repositioning"})
    write_json(out / "tasks.json", {"checkpoint": args.model_path, "seed": args.seed,
               "steps": args.diffusion_inference_steps, "entries": entries, "tasks": tasks,
               "selection": "GT-assisted source quality shortlist, then predicted-region geometry; not benchmark"})
    print(f"Prepared {len(tasks)} tasks from {len(entries)} source images", flush=True)


def tensor_image(tensor):
    return Image.fromarray((tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()*255).astype(np.uint8))


def augment(args):
    """Additional fixed-noise search with larger add objects / nonhuman BG donors."""
    source, out = Path(args.source_dir), Path(args.output_dir)
    manifest = json.loads((out / "tasks.json").read_text())
    candidates = json.loads((source / "candidates.json").read_text())["entries"]
    pool = []
    for entry in candidates:
        if entry["dataset"] != "coco":
            continue
        slots = parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])
        if not 1 <= len(slots) <= 5 or any(not s["is_thing"] or not s["parsed"] for s in slots):
            continue
        pred = np.load(source / "cache/coco" / entry["id"] / "ours.npz")["prediction"]
        counts = Counter(s["normalized_category"] for s in slots)
        for slot in slots:
            mask = pred == slot["prediction_label"]
            area = float(mask.mean())
            if counts[slot["normalized_category"]] == 1 and .15 <= area <= .60:
                pool.append(dict(slot, entry=entry, id=entry["id"], pred=pred, mask=mask, area=area,
                                 categories=set(counts)))
    targets = [t for t in manifest["tasks"] if t["mode"] == "remove"]
    old_ids = {t["task_id"] for t in manifest["tasks"]}
    added_count = 0
    for target in targets:
        pred = np.load(source / "cache/coco" / target["target_id"] / "ours.npz")["prediction"]
        entry = manifest["entries"][target["target_id"]]
        categories = {s["normalized_category"] for s in parse_ar_slots(entry["ar_caption"]["text"], entry["ar_caption"]["ovt_count"])}
        for mode in ("add", "background"):
            task_id = f'{mode}_extra_{target["target_id"]}'
            if task_id in old_ids:
                continue
            original = next(t for t in manifest["tasks"] if t["mode"] == mode and t["target_id"] == target["target_id"])
            donors = [d for d in pool if d["id"] not in {target["target_id"], original["donor_id"]}]
            if mode == "add":
                donors = [d for d in donors if d["normalized_category"] not in categories]
                def score(d):
                    overlap = np.count_nonzero(d["mask"] & (pred > 0))/max(np.count_nonzero(d["mask"]), 1)
                    return .9*d["area"] - .8*overlap
            else:
                donors = [d for d in donors if "person" not in d["categories"] and float((d["pred"] == 0).mean()) >= .4]
                def score(d):
                    return float((d["pred"] == 0).mean()) + .15*binary_iou(d["pred"] > 0, pred > 0)
            if not donors:
                continue
            donor = max(donors, key=score)
            task = dict(original, task_id=task_id, donor_id=donor["id"], donor_slot=donor["slot_index"],
                        donor_category=donor["normalized_category"], donor_prediction_label=donor["prediction_label"],
                        additional_search="larger donor object" if mode == "add" else "nonhuman background donor")
            manifest["entries"][donor["id"]] = donor["entry"]
            manifest["tasks"].append(task)
            added_count += 1
    write_json(out / "tasks.json", manifest)
    print(f"Appended {added_count} tasks without altering original edits or noise seeds", flush=True)


def overlay(image, mask):
    rgb = np.asarray(image).astype(float)
    rgb[mask] = .55*rgb[mask] + .45*np.array([50, 200, 90])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def panel_grid(images, titles, path, size=320):
    canvas = Image.new("RGB", (size*len(images), size+38), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    for i, (image, title) in enumerate(zip(images, titles)):
        canvas.paste(image.resize((size, size)), (size*i, 38))
        draw.text((size*i+5, 9), title, fill="black", font=font)
    canvas.save(path)
    return canvas


def edit_stats(edit, recon, region):
    change = np.abs(edit-recon).mean(2)
    energy = ((edit-recon)**2).mean(2)
    inside = float(change[region].mean()) if region.any() else 0.
    outside = float(change[~region].mean()) if (~region).any() else 0.
    return {"inside_change": inside, "outside_change": outside,
            "change_energy_inside_fraction": float(energy[region].sum()/max(energy.sum(), 1e-12)),
            "region_fraction": float(region.mean()),
            "screening_score": inside - .75*outside}


def render(args):
    import torch
    import torch.nn.functional as F
    from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
    from pgot.eval.pgot_inference import pgot_forward_eval
    from pgot.eval.probe_object_editing import owner_units, condition_from_units
    from pgot.eval.probe_removal_and_instruction import rewrite_memories, units_with_memory
    from pgot.eval.run_compositional_gen import generate_images, seeded_noise
    from pgot.eval.run_eval import load_rae_decoder
    from pgot.model.visual_memory import PGOTOneShotMemoryReader
    from tqdm import tqdm
    torch.set_num_threads(3)
    if not torch.cuda.is_available():
        raise RuntimeError("Choose a free GPU")
    source, out = Path(args.source_dir), Path(args.output_dir)
    manifest = json.loads((out / "tasks.json").read_text())
    if manifest["checkpoint"] != args.model_path or manifest["seed"] != args.seed or manifest["steps"] != args.diffusion_inference_steps:
        raise ValueError("Manifest checkpoint/seed/steps mismatch")
    tasks = manifest["tasks"][:args.limit] if args.limit else manifest["tasks"]
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(model.pgot_e8_reader, PGOTOneShotMemoryReader):
        raise ValueError("Only one-shot memory Reader checkpoints supported")
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    towers = model.get_vision_tower_aux_list()
    obj_mem = int(model.pgot_e8_writer.object_memories_per_owner)
    reg_mem = int(model.pgot_e8_writer.register_memories_per_owner)
    write_json(out / "protocol.json", {"checkpoint": args.model_path, "dtype": args.dtype,
               "caption_source": "cached image-only AR", "gt_caption_used": False, "gt_masks_used": False,
               "editing": "learned semantic + visual-memory owner bundles -> original Reader -> DiT -> RAE decoder",
               "removal": "drop owner plus official existing predicted-mask dilated Writer patch exclusion",
               "background": "donor registers rewritten with ALL predicted donor foreground patches excluded",
               "guidance_scale": args.guidance_scale, "steps": args.diffusion_inference_steps,
               "same_noise_for_reconstruction_and_edit": True, "seed": args.seed,
               "no_external_inpainting_or_pixel_paste": True,
               "object_memories_per_owner": obj_mem, "register_memories_per_owner": reg_mem,
               "not_freeform_instruction_following": True})
    if any(t.get("reader_prior_strength", 0.) > 0 for t in tasks):
        protocol = json.loads((out / "protocol.json").read_text())
        protocol["variants"] = {"native": "unchanged learned Reader; bundle swap/append",
            "predicted_guided": "inference-only predicted ownership log-prior in Reader and foreground-only object Writer pooling; NO GT prior"}
        protocol["variant_interventions_recorded_per_task"] = True
        write_json(out / "protocol.json", protocol)
    forward_dir = out / "owner_cache"
    forward_dir.mkdir(parents=True, exist_ok=True)
    shared_dir = Path(args.owner_cache_dir) if getattr(args, "owner_cache_dir", "") else None
    shared_manifest = json.loads((shared_dir.parent / "tasks.json").read_text()) if shared_dir else None
    if shared_manifest and shared_manifest["checkpoint"] != args.model_path:
        raise ValueError("Shared owner-cache checkpoint mismatch")
    keys = ["semantic_slots", "visual_memory", "ovt_object_valid", "raw_rae_hidden",
            "ovt_object_probs", "ovt_void_probs", "img_hidden", "raw_img_features", "rae_hidden"]

    def get_forward(image_id):
        path = forward_dir / f"{image_id}.pt"
        if not path.exists() and shared_dir and (shared_dir / f"{image_id}.pt").exists():
            if shared_manifest["entries"][image_id]["caption_sha256"] != manifest["entries"][image_id]["caption_sha256"]:
                raise ValueError("Shared owner-cache caption mismatch")
            path = shared_dir / f"{image_id}.pt"
        if path.exists():
            return {k: v.to(device) for k, v in torch.load(path, map_location="cpu", weights_only=True).items()}
        entry = manifest["entries"][image_id]
        image = Image.open(source / "cache/coco" / image_id / "input.png").convert("RGB")
        ids = torch.tensor(tokenizer.encode(entry["ar_caption"]["text"], add_special_tokens=False))[None]
        positions = (ids[0] == tokenizer.convert_tokens_to_ids("<ovt>")).nonzero().flatten()
        if len(positions) != entry["ar_caption"]["ovt_count"] or len(positions) > 50:
            raise ValueError("AR object position mismatch")
        packed, valid = torch.zeros(1, 50, dtype=torch.long), torch.zeros(1, 50, dtype=torch.bool)
        packed[0, :len(positions)], valid[0, :len(positions)] = positions, True
        result = pgot_forward_eval(model,
            images=towers[0].image_processor.preprocess(image, return_tensors="pt")["pixel_values"],
            target_images=towers[-1].image_processor.preprocess(image, return_tensors="pt")["pixel_values"],
            caption_input_ids=ids, caption_attention_mask=torch.ones_like(ids, dtype=torch.bool),
            ovt_positions_in_caption=packed, ovt_valid_mask=valid)
        result = {k: result[k] for k in keys}
        torch.save({k: v.detach().cpu() for k, v in result.items()}, path)
        return result

    def exclusion(result, slots):
        probs = torch.cat([result["ovt_object_probs"], result["ovt_void_probs"]], 1)
        hit = torch.isin(probs.argmax(1), torch.tensor(slots, device=device))
        side = int(round(hit.shape[-1]**.5))
        return F.max_pool2d(hit.reshape(1, 1, side, side).float(), 3, 1, 1).reshape(1, -1) > 0

    with torch.inference_mode():
        for task in tqdm(tasks, desc="PGOT real editing"):
            dest = out / "candidates" / task["mode"] / task["task_id"]
            if (dest / "metadata.json").exists():
                continue
            a = get_forward(task["target_id"])
            objects, registers = owner_units(a)
            condition = condition_from_units(model, a["raw_rae_hidden"], objects, registers, obj_mem, reg_mem)
            relative = float((condition-a["rae_hidden"]).abs().max()/a["rae_hidden"].abs().mean().clamp_min(1e-12))
            if relative > 1e-3:
                raise ValueError(f"Owner rebuild identity check failed: {relative}")
            noise = seeded_noise(model, [int(task["target_id"])], args.seed, device)
            recon_path = out / "recon_cache" / f'{task["target_id"]}.png'
            recon_path.parent.mkdir(parents=True, exist_ok=True)
            if not recon_path.exists():
                tensor_image(generate_images(model, decoder, condition, noise, args.guidance_scale, device)[0]).save(recon_path)
            mode, k = task["mode"], task["target_slot"]
            if mode in {"remove", "replace"}:
                cleaned = rewrite_memories(model, a, exclude=exclusion(a, [k]))
                objects, registers = units_with_memory(a, cleaned)
                objects = [u for i, u in enumerate(objects) if i != k]
            if task["donor_id"]:
                b = get_forward(task["donor_id"])
                donor_objects, donor_registers = owner_units(b)
                if mode in {"add", "replace"}:
                    objects.append(donor_objects[task["donor_slot"]])
                elif mode == "background":
                    blocked = exclusion(b, list(range(len(donor_objects))))
                    if blocked.all():
                        raise ValueError("No donor background patches remain")
                    _, registers = units_with_memory(b, rewrite_memories(model, b, exclude=blocked))
            edited_condition = condition_from_units(model, a["raw_rae_hidden"], objects, registers, obj_mem, reg_mem)
            intervention = {"variant": task.get("variant", "native"), "reader_prior_strength": 0.}
            if task.get("reader_prior_strength", 0.) > 0:
                from pgot.eval.retry_model_editing import guided_condition
                edited_condition, intervention = guided_condition(model, a, b, objects, registers, task, obj_mem, reg_mem)
            edited = tensor_image(generate_images(model, decoder, edited_condition, noise, args.guidance_scale, device)[0])
            dest.mkdir(parents=True, exist_ok=True)
            edited.save(dest / "edited_native.png")
            recon = Image.open(recon_path).convert("RGB")
            image_a = Image.open(source / "cache/coco" / task["target_id"] / "input.png").convert("RGB")
            pred_a = np.load(source / "cache/coco" / task["target_id"] / "ours.npz")["prediction"]
            target_mask = pred_a == task["target_prediction_label"]
            if mode == "remove":
                panels, titles = [image_a, overlay(image_a, target_mask), edited], ["Original image", f'Remove: {task["target_category"]}', "Edited image"]
                region = target_mask
            else:
                image_b = Image.open(source / "cache/coco" / task["donor_id"] / "input.png").convert("RGB")
                pred_b = np.load(source / "cache/coco" / task["donor_id"] / "ours.npz")["prediction"]
                donor_mask = pred_b == task["donor_prediction_label"]
                if mode == "add":
                    panels, titles = [image_a, overlay(image_b, donor_mask), edited], ["Original image", f'Add: {task["donor_category"]}', "Edited image"]
                    region = donor_mask
                elif mode == "replace":
                    panels, titles = [overlay(image_a, target_mask), overlay(image_b, donor_mask), edited], [f'Target: {task["target_category"]}', f'Replacement: {task["donor_category"]}', "Edited image"]
                    region = target_mask | donor_mask
                else:
                    panels, titles = [overlay(image_a, pred_a == 0), overlay(image_b, pred_b == 0), edited], ["Target background", "Replacement background", "Edited image"]
                    region = pred_a == 0
                image_b.save(dest / "donor.png")
            image_a.save(dest / "original.png")
            recon.save(dest / "reconstruction_native.png")
            for i, panel in enumerate(panels):
                panel.save(dest / f"panel_{i}.png")
            panel_grid(panels, titles, dest / "figure.png")
            panel_grid([image_a, recon, edited], ["Original image", "Unedited reconstruction", "Actual model edit"], dest / "sanity.png")
            native_region = np.asarray(Image.fromarray(region).resize(edited.size, Image.Resampling.NEAREST), bool)
            stats = edit_stats(np.asarray(edited).astype(float)/255, np.asarray(recon).astype(float)/255, native_region)
            write_json(dest / "metadata.json", dict(task, stats=stats, identity_relative_error=relative,
                       native_output_size=list(edited.size), visual_success_verified=False, inference_intervention=intervention))
            del a, condition, edited_condition
            if task["donor_id"]:
                del b
            print(f"{task['task_id']} -> {dest / 'figure.png'}", flush=True)


def select(args):
    import shutil
    out = Path(args.output_dir)
    review_path = out / "visual_review.json"
    reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
    result = {}
    for mode in MODES:
        candidates = []
        for path in (out / "candidates" / mode).glob("*/metadata.json"):
            meta = json.loads(path.read_text())
            if getattr(args, "task_filter", "") and args.task_filter not in meta["task_id"]:
                continue
            review = reviews.get(meta["task_id"], {"status": "unreviewed"})
            meta["visual_review"] = review
            meta["visual_success_verified"] = review["status"] == "success"
            priority = {"success": 3, "partial": 2, "failure": 0, "unreviewed": 1}[review["status"]]
            candidates.append(((priority, meta["stats"]["screening_score"]), path.parent, meta))
        candidates.sort(key=lambda r: r[0], reverse=True)
        chosen, seen = [], set()
        for candidate in candidates:
            if candidate[2]["target_id"] in seen:
                continue
            seen.add(candidate[2]["target_id"])
            chosen.append(candidate)
            if len(chosen) == args.examples:
                break
        panels, rows = [], []
        for rank, (_, src, meta) in enumerate(chosen, 1):
            dest = out / args.selection_subdir / mode / f'{rank:02}_{meta["target_id"]}'
            dest.mkdir(parents=True, exist_ok=True)
            for name in ("figure.png", "sanity.png", "panel_0.png", "panel_1.png", "panel_2.png", "original.png", "donor.png", "edited_native.png", "reconstruction_native.png", "metadata.json"):
                if (src / name).exists():
                    shutil.copy2(src / name, dest / name)
            write_json(dest / "metadata.json", meta)
            if review_path.exists():
                # Keep clean figure available, but make review limitations visible.
                original_figure = Image.open(dest / "figure.png").convert("RGB")
                review_canvas = Image.new("RGB", (original_figure.width, original_figure.height+28), "white")
                review_canvas.paste(original_figure, (0, 28))
                text = f'{meta["task_id"]} | {meta["visual_review"]["status"].upper()} | see metadata for limitations'
                ImageDraw.Draw(review_canvas).text((5, 6), text, fill="black")
                review_canvas.save(dest / "review.png")
                panels.append(review_canvas)
            else:
                panels.append(Image.open(dest / "figure.png").convert("RGB"))
            rows.append(dict(meta, directory=str(dest)))
        if panels:
            sheet = Image.new("RGB", (panels[0].width, sum(p.height for p in panels)), "white")
            y = 0
            for panel in panels:
                sheet.paste(panel, (0, y))
                y += panel.height
            sheet.save(out / args.selection_subdir / f"{mode}_contact_sheet.jpg", quality=94)
            for page_start in range(0, len(panels), 8):
                page = panels[page_start:page_start+8]
                page_sheet = Image.new("RGB", (page[0].width, sum(p.height for p in page)), "white")
                y = 0
                for panel in page:
                    page_sheet.paste(panel, (0, y))
                    y += panel.height
                page_sheet.save(out / args.selection_subdir / f"{mode}_page_{page_start//8+1:02}.jpg", quality=94)
        result[mode] = rows
    if all(result[mode] for mode in MODES):
        # Reference-like quadrants, retaining review labels rather than implying
        # that all requested examples are clean semantic successes.
        sheets = {mode: Image.open(out / args.selection_subdir / f"{mode}_contact_sheet.jpg").convert("RGB") for mode in MODES}
        width = max(sheet.width for sheet in sheets.values())
        heights = [max(sheets[a].height, sheets[b].height) + 32 for a, b in [("remove", "replace"), ("add", "background")]]
        overview = Image.new("RGB", (2*width, sum(heights)), "white")
        draw = ImageDraw.Draw(overview)
        y = 0
        for height, pair in zip(heights, [("remove", "replace"), ("add", "background")]):
            for column, mode in enumerate(pair):
                draw.text((column*width+8, y+8), f"{mode.upper()} | {len(result[mode])} reviewed examples; labels are qualitative", fill="black")
                overview.paste(sheets[mode], (column*width, y+32))
            y += height
        overview.save(out / args.selection_subdir / "overview_all_32.jpg", quality=94)
    write_json(out / f"{args.selection_subdir}_selection.json", {"selection_method": "visual-review priority if provided, then change localization screening; NOT quantitative semantic edit success",
               "requested_per_type": args.examples, "groups": result})
    print({k: len(v) for k,v in result.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "augment", "render", "select"], required=True)
    parser.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000")
    parser.add_argument("--source_dir", default="/home/jovyan/PGOT/outputs/figure_segmentation_over_under")
    parser.add_argument("--output_dir", default="/home/jovyan/PGOT/outputs/figure_model_editing_memory8_register32_noprior")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dtype", default="fp32")
    parser.add_argument("--diffusion_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--selection_subdir", default="selected")
    parser.add_argument("--task_filter", default="", help="Optional task-ID substring for review-only export")
    args = parser.parse_args()
    {"prepare": prepare, "augment": augment, "render": render, "select": select}[args.stage](args)


if __name__ == "__main__":
    main()
