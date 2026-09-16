"""Object-level image editing probe for PGOT one-shot memory checkpoints.

Can the decomposed owner set {(s_k, M_k)} be edited to edit the image?  Every
condition decodes with the SAME per-image diffusion noise, so differences from
the reconstruction come only from the edit:

  recon          model's own owner set (identity-checked against rae_hidden)
  recon_seed2    same owner set, different noise  -> sampling-variability floor
  remove         drop object k's (s_k, M_k)
  replace        object k <- largest object of another probe image (same category first)
  text_color     object k's caption: a colour word swapped (or "bright red" prepended)
  text_category  object k's caption: category renamed (e.g. dog -> cat)
  insert_text    new caption chunk for an object absent from the image

Set edits act only on the Reader input; caption edits re-run the LLM and the
Writer on the same image.  Object k is the largest caption object whose 16x16
GT area lies in [min_area, max_area].  Per condition we report pixel change
inside/outside object k's GT mask and the inside mean-colour shift.
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.run_compositional_gen import build_condition, generate_images, seeded_noise
from pgot.eval.run_eval import denormalize_images, load_rae_decoder
from pgot.model.visual_memory import PGOTOneShotMemoryReader
from pgot.train.pgot_dataset import Pix2CapPGOTDataset

COLOR_SWAP = {
    "red": "blue", "blue": "red", "green": "red", "yellow": "blue", "white": "black",
    "black": "white", "brown": "blue", "gray": "red", "grey": "red", "orange": "blue",
    "pink": "green", "purple": "yellow", "silver": "red", "tan": "blue",
}
CATEGORY_SWAP = {
    "person": "dog", "man": "dog", "woman": "dog", "dog": "cat", "cat": "dog", "horse": "cow",
    "cow": "horse", "sheep": "dog", "elephant": "horse", "bear": "dog", "zebra": "horse",
    "giraffe": "horse", "car": "bus", "bus": "car", "truck": "car", "train": "bus",
    "motorcycle": "bicycle", "bicycle": "motorcycle", "airplane": "bird", "bird": "airplane",
    "boat": "car", "chair": "sofa", "couch": "chair", "bed": "sofa", "dining table": "bed",
    "pizza": "cake", "cake": "pizza", "laptop": "book", "tv": "laptop",
}
CONDITIONS = ["recon", "recon_seed2", "remove", "replace", "text_color", "text_category", "insert_text"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path",
                   default="/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000")
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--min_objects", type=int, default=2)
    p.add_argument("--min_area", type=float, default=0.08)
    p.add_argument("--max_area", type=float, default=0.6)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--diffusion_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_caption_tokens", type=int, default=1024)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--insert_category", default="dog")
    p.add_argument("--insert_description", default="a large brown dog standing in the center of the scene")
    return p.parse_args()


def caption_tensors(text, tokenizer, max_caption_tokens, max_objects):
    ids = tokenizer.encode(text, add_special_tokens=False)[:max_caption_tokens]
    token_ids = torch.tensor(ids, dtype=torch.long)[None]
    attention = torch.ones_like(token_ids, dtype=torch.bool)
    ovt_id = int(tokenizer.convert_tokens_to_ids("<ovt>"))
    positions = torch.nonzero(token_ids[0] == ovt_id, as_tuple=False).flatten()[:max_objects]
    ovt_positions = torch.zeros((1, max_objects), dtype=torch.long)
    ovt_valid = torch.zeros((1, max_objects), dtype=torch.bool)
    ovt_positions[0, : positions.numel()] = positions
    ovt_valid[0, : positions.numel()] = True
    return token_ids, attention, ovt_positions, ovt_valid


@torch.no_grad()
def forward_caption(model, sample, text, tokenizer, args):
    ids, attention, positions, valid = caption_tensors(text, tokenizer, args.max_caption_tokens, args.max_objects)
    return pgot_forward_eval(
        model,
        images=sample["image"][None],
        target_images=sample["target_image"][None],
        caption_input_ids=ids,
        caption_attention_mask=attention,
        ovt_positions_in_caption=positions,
        ovt_valid_mask=valid,
    )


def owner_units(out):
    """Caption-ordered object units and register units of a single image."""
    K = out["ovt_object_valid"].shape[1]
    semantic, memory = out["semantic_slots"][0], out["visual_memory"][0]
    valid = out["ovt_object_valid"][0].bool().tolist()
    objects = [(semantic[k], memory[k]) for k in range(K) if valid[k]]
    registers = [(semantic[K + r], memory[K + r]) for r in range(semantic.shape[0] - K)]
    return objects, registers


def condition_from_units(model, raw_rae, objects, registers, obj_mem, reg_mem):
    units = list(objects) + list(registers)
    semantic = torch.stack([s for s, _ in units])[None]
    memory = torch.stack([m for _, m in units])[None]
    object_valid = torch.ones(1, len(objects), dtype=torch.bool, device=semantic.device)
    return build_condition(model, raw_rae, semantic, memory, object_valid, obj_mem, reg_mem)


def color_edit(description, category):
    for word in re.findall(r"[A-Za-z]+", description):
        new = COLOR_SWAP.get(word.lower())
        if new:
            return re.sub(rf"\b{re.escape(word)}\b", new, description, count=1), f"{word.lower()} -> {new}"
    return f"a bright red {category}, {description}", "prepend 'a bright red'"


def category_edit(segment):
    old = str(segment["category"])
    new = CATEGORY_SWAP.get(old.lower(), "cat" if old.lower() == "dog" else "dog")
    edited = dict(segment)
    edited["category"] = new
    description = re.sub(rf"\b{re.escape(old)}\b", new, str(segment["description"]), flags=re.IGNORECASE)
    if new not in description.lower():
        description = f"a {new}, {description}"
    edited["description"] = description
    return edited, f"{old} -> {new}"


def change_stats(edit, ref, mask):
    diff = (edit - ref).abs().mean(0)
    energy = (edit - ref).pow(2).mean(0)
    inside = mask.float()
    outside = 1.0 - inside
    stats = {
        "inside_mean_abs": float((diff * inside).sum() / inside.sum().clamp_min(1)),
        "outside_mean_abs": float((diff * outside).sum() / outside.sum().clamp_min(1)),
        "global_mean_abs": float(diff.mean()),
        "energy_inside_fraction": float((energy * inside).sum() / energy.sum().clamp_min(1e-12)),
        "mask_area_fraction": float(inside.mean()),
    }
    if mask.any():
        stats["inside_rgb_ref"] = [round(float(v), 4) for v in ref[:, mask].mean(1)]
        stats["inside_rgb_edit"] = [round(float(v), 4) for v in edit[:, mask].mean(1)]
    return stats


def labelled(img_tensor, text, size=224):
    arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(arr).resize((size, size))
    canvas = Image.new("RGB", (size, size + 18), (255, 255, 255))
    canvas.paste(img, (0, 18))
    ImageDraw.Draw(canvas).text((3, 3), text[:34], fill=(0, 0, 0))
    return canvas


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(3)  # container CPU quota is 3 cores
    out_dir = Path(args.output_dir)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    start = time.time()

    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(getattr(model, "pgot_e8_reader", None), PGOTOneShotMemoryReader):
        raise ValueError("probe_object_editing needs a one-shot memory checkpoint")
    obj_mem = int(model.pgot_e8_writer.object_memories_per_owner)
    reg_mem = int(model.pgot_e8_writer.register_memories_per_owner)
    rae_decoder = load_rae_decoder(model, device=device, dtype=torch.float32)

    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    t_mean = torch.tensor(target_proc.image_mean).view(1, -1, 1, 1)
    t_std = torch.tensor(target_proc.image_std).view(1, -1, 1, 1)
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl, tokenizer=tokenizer, image_processor=image_proc,
        target_image_processor=target_proc, grid_size=32, max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=1, max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode="coda_center_crop", coda_crop_size=512,
    )

    picks = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        n = int(sample["n_objects"])
        if n < args.min_objects:
            continue
        areas = sample["gt_rae_masks_per_ovt"][:n].sum(-1) / 256.0
        k = int(areas.argmax())
        if not args.min_area <= float(areas[k]) <= args.max_area:
            continue
        segments = copy.deepcopy(dataset.samples[idx]["segments"])[:n]
        if dataset._build_caption_with_ovt(segments)[0] != sample["caption_text"]:
            continue  # caption not rebuildable 1:1 from segments; skip rather than edit a mismatch
        picks.append({"idx": idx, "k": k, "sample": sample, "segments": segments, "area": float(areas[k])})
        if len(picks) == args.num_images:
            break
    print(f"picked {len(picks)} images: {[(p['idx'], p['segments'][p['k']]['category']) for p in picks]}")

    for pick in picks:  # base forwards, reused as donors
        pick["out"] = forward_caption(model, pick["sample"], pick["sample"]["caption_text"], tokenizer, args)

    records, per_condition = [], {c: [] for c in CONDITIONS if c != "recon"}
    identity_max_rel = 0.0
    for i, pick in enumerate(picks):
        sample, k, segments, out = pick["sample"], pick["k"], pick["segments"], pick["out"]
        raw_rae = out["raw_rae_hidden"]
        objects, registers = owner_units(out)
        x_end = seeded_noise(model, [pick["idx"]], args.seed, device)
        x_end2 = seeded_noise(model, [pick["idx"]], args.seed + 1, device)

        recon_cond = condition_from_units(model, raw_rae, objects, registers, obj_mem, reg_mem)
        rel = float((recon_cond.float() - out["rae_hidden"].float()).abs().max()
                    / out["rae_hidden"].float().abs().mean().clamp_min(1e-12))
        identity_max_rel = max(identity_max_rel, rel)

        target_cat = str(segments[k]["category"]).lower()
        donors = [p for j, p in enumerate(picks) if j != i]
        same = [p for p in donors if str(p["segments"][p["k"]]["category"]).lower() == target_cat]
        donor = (same or donors)[0]
        donor_objects, _ = owner_units(donor["out"])

        color_segments = copy.deepcopy(segments)
        color_segments[k]["description"], color_note = color_edit(
            str(segments[k]["description"]), str(segments[k]["category"]))
        category_segments = copy.deepcopy(segments)
        category_segments[k], category_note = category_edit(segments[k])
        insert_segments = copy.deepcopy(segments) + [{
            "category": args.insert_category, "description": args.insert_description,
            "is_thing": True, "category_id": -1,
        }]
        texts = {name: dataset._build_caption_with_ovt(segs)
                 for name, segs in (("text_color", color_segments), ("text_category", category_segments),
                                    ("insert_text", insert_segments))}

        conditions = {
            "recon": (recon_cond, x_end),
            "recon_seed2": (recon_cond, x_end2),
            "remove": (condition_from_units(model, raw_rae, objects[:k] + objects[k + 1:], registers,
                                            obj_mem, reg_mem), x_end),
            "replace": (condition_from_units(model, raw_rae, objects[:k] + [donor_objects[donor["k"]]]
                                             + objects[k + 1:], registers, obj_mem, reg_mem), x_end),
        }
        for name, (text, used) in texts.items():
            edited_out = forward_caption(model, sample, text, tokenizer, args)
            conditions[name] = (edited_out["rae_hidden"], x_end)

        images = {name: generate_images(model, rae_decoder, cond, noise, args.guidance_scale, device)[0]
                  for name, (cond, noise) in conditions.items()}
        real = denormalize_images(sample["target_image"][None].to(device).float(), t_mean, t_std)
        real = F.interpolate(real, size=images["recon"].shape[-2:], mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(sample["gt_rae_masks_per_ovt"][k].view(1, 1, 16, 16).float(),
                             size=images["recon"].shape[-2:], mode="bilinear", align_corners=False)[0, 0] > 0.5
        mask = mask.to(images["recon"].device)

        stats = {name: change_stats(images[name], images["recon"], mask) for name in conditions if name != "recon"}
        for name, value in stats.items():
            per_condition[name].append(value)
        records.append({
            "sample_index": pick["idx"], "image_id": int(sample["image_id"]), "object_index": k,
            "object_category": segments[k]["category"], "object_area_fraction_16x16": round(pick["area"], 4),
            "num_objects": len(objects),
            "donor": {"image_id": int(donor["sample"]["image_id"]),
                      "category": donor["segments"][donor["k"]]["category"],
                      "same_category": bool(same)},
            "text_color_edit": color_note, "text_category_edit": category_note,
            "insert_kept_in_caption": texts["insert_text"][1] == len(insert_segments),
            "stats": stats,
        })

        panels = [labelled(real, "input"), labelled(images["recon"], "recon"),
                  labelled(images["recon_seed2"], "recon (other noise)"),
                  labelled(images["remove"], f"remove {segments[k]['category']}"),
                  labelled(images["replace"], f"replace <- {donor['segments'][donor['k']]['category']}"),
                  labelled(images["text_color"], f"text {color_note}"),
                  labelled(images["text_category"], f"text {category_note}"),
                  labelled(images["insert_text"], f"text + {args.insert_category}"),
                  labelled(mask[None].float().expand(3, -1, -1), f"mask: {segments[k]['category']}")]
        grid = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), (255, 255, 255))
        x = 0
        for panel in panels:
            grid.paste(panel, (x, 0))
            x += panel.width
        grid.save(out_dir / "grids" / f"{i:02d}_img{int(sample['image_id'])}_{segments[k]['category']}.png")
        print(f"[{i}] image {int(sample['image_id'])} obj={segments[k]['category']} "
              + " ".join(f"{n}:in={s['inside_mean_abs']:.3f}/out={s['outside_mean_abs']:.3f}"
                         for n, s in stats.items()))

    def agg(values, key):
        return round(float(np.mean([v[key] for v in values])), 4) if values else None

    summary = {
        "model_path": args.model_path, "num_images": len(records),
        "object_memories_per_owner": obj_mem, "register_memories_per_owner": reg_mem,
        "identity_check_max_relative_error": identity_max_rel,
        "diffusion_inference_steps": args.diffusion_inference_steps, "seed": args.seed,
        "aggregate": {name: {key: agg(values, key) for key in
                             ("inside_mean_abs", "outside_mean_abs", "global_mean_abs",
                              "energy_inside_fraction", "mask_area_fraction")}
                      for name, values in per_condition.items()},
        "records": records, "elapsed_sec": round(time.time() - start, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["aggregate"], indent=1))
    print(f"identity max rel err={identity_max_rel:.2e} -> {out_dir}")


if __name__ == "__main__":
    main()
