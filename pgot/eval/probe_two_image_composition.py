"""Two-image, prompt-selected composition probe for PGOT one-shot memory checkpoints.

"Put the <category> from image 2 into image 1": the category name selects an
object token from image 2's caption, and the Reader decodes a hand-built owner
set.  For each pair (A, B) every condition uses A's noise:

  recon_A / recon_B     each image's own owner set
  add                   A objects + B's selected object + A registers
  background_swap       A objects + B registers
  object_on_A_bg        B's selected object + A registers (A objects removed)
  B_objects_on_A_bg     all B objects + A registers

Placement: both images are CODA 512 center crops decoded on the same grid, so
the pixels changed by ``add`` (vs recon_A) can be compared with B's object mask
at its ORIGINAL location.  ``donor_location_iou`` is the IoU between B's object
mask and the top-changed pixels (same area as the mask); a high value means the
object is pasted where it was in B, i.e. placement is not prompt-controllable.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.probe_object_editing import condition_from_units, forward_caption, labelled, owner_units
from pgot.eval.run_compositional_gen import generate_images, seeded_noise
from pgot.eval.run_eval import denormalize_images, load_rae_decoder
from pgot.model.visual_memory import PGOTOneShotMemoryReader
from pgot.train.pgot_dataset import Pix2CapPGOTDataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path",
                   default="/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000")
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_pairs", type=int, default=8)
    p.add_argument("--scan_images", type=int, default=120)
    p.add_argument("--min_area", type=float, default=0.06)
    p.add_argument("--max_area", type=float, default=0.35)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--diffusion_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_caption_tokens", type=int, default=1024)
    p.add_argument("--max_objects", type=int, default=50)
    return p.parse_args()


def mask_224(rae_mask, size):
    return F.interpolate(rae_mask.view(1, 1, 16, 16).float(), size=size,
                         mode="bilinear", align_corners=False)[0, 0] > 0.5


def placement_stats(edit, ref, donor_mask, target_masks):
    energy = (edit - ref).pow(2).mean(0)
    area = int(donor_mask.sum())
    stats = {"donor_mask_area_fraction": float(donor_mask.float().mean())}
    if area == 0:
        return stats
    thresh = torch.topk(energy.flatten(), area).values[-1]
    changed = energy >= thresh
    inter = float((changed & donor_mask).sum())
    stats["donor_location_iou"] = inter / float((changed | donor_mask).sum())
    stats["energy_in_donor_mask"] = float((energy * donor_mask).sum() / energy.sum().clamp_min(1e-12))
    if target_masks is not None and target_masks.any():
        stats["energy_in_A_objects"] = float((energy * target_masks).sum() / energy.sum().clamp_min(1e-12))
        stats["A_objects_area_fraction"] = float(target_masks.float().mean())
    ys, xs = torch.nonzero(changed, as_tuple=True)
    dy, dx = torch.nonzero(donor_mask, as_tuple=True)
    h, w = energy.shape
    stats["changed_centroid_xy"] = [round(float(xs.float().mean()) / w, 3), round(float(ys.float().mean()) / h, 3)]
    stats["donor_centroid_xy"] = [round(float(dx.float().mean()) / w, 3), round(float(dy.float().mean()) / h, 3)]
    return stats


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(3)  # container CPU quota is 3 cores
    out_dir = Path(args.output_dir)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)
    start = time.time()

    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(getattr(model, "pgot_e8_reader", None), PGOTOneShotMemoryReader):
        raise ValueError("probe_two_image_composition needs a one-shot memory checkpoint")
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

    # Candidate images: caption objects with categories; the "donor" object of an
    # image is its largest mid-sized object.
    pool = []
    for idx in range(min(args.scan_images, len(dataset))):
        sample = dataset[idx]
        n = int(sample["n_objects"])
        if n < 1:
            continue
        categories = [str(s["category"]).lower() for s in dataset.samples[idx]["segments"][:n]]
        areas = sample["gt_rae_masks_per_ovt"][:n].sum(-1) / 256.0
        ok = [k for k in range(n) if args.min_area <= float(areas[k]) <= args.max_area]
        if not ok:
            continue
        k = max(ok, key=lambda j: float(areas[j]))
        pool.append({"idx": idx, "sample": sample, "categories": categories, "k": k})
    rng = random.Random(args.seed)
    pairs = []
    for a in pool:
        for b in rng.sample(pool, len(pool)):
            if b["idx"] == a["idx"] or b["categories"][b["k"]] in a["categories"]:
                continue  # the added category must be absent from A
            pairs.append((a, b))
            break
        if len(pairs) == args.num_pairs:
            break
    print(f"pool={len(pool)} pairs={[(a['idx'], b['idx'], b['categories'][b['k']]) for a, b in pairs]}")

    outs = {}
    records, agg = [], {}
    for i, (a, b) in enumerate(pairs):
        for item in (a, b):
            if item["idx"] not in outs:
                outs[item["idx"]] = forward_caption(model, item["sample"], item["sample"]["caption_text"], tokenizer, args)
        out_a, out_b = outs[a["idx"]], outs[b["idx"]]
        objs_a, regs_a = owner_units(out_a)
        objs_b, regs_b = owner_units(out_b)
        category = b["categories"][b["k"]]
        prompt = f"Put the {category} from image 2 into image 1."
        # Prompt -> token: pick image 2's caption object whose category name is in the prompt.
        chosen = [k for k, c in enumerate(b["categories"]) if c in prompt.lower()]
        k_b = b["k"] if b["k"] in chosen else chosen[0]
        raw_rae = out_a["raw_rae_hidden"]
        x_end = seeded_noise(model, [a["idx"]], args.seed, device)

        conditions = {
            "recon_A": condition_from_units(model, raw_rae, objs_a, regs_a, obj_mem, reg_mem),
            "recon_B": condition_from_units(model, raw_rae, objs_b, regs_b, obj_mem, reg_mem),
            "add": condition_from_units(model, raw_rae, objs_a + [objs_b[k_b]], regs_a, obj_mem, reg_mem),
            "background_swap": condition_from_units(model, raw_rae, objs_a, regs_b, obj_mem, reg_mem),
            "object_on_A_bg": condition_from_units(model, raw_rae, [objs_b[k_b]], regs_a, obj_mem, reg_mem),
            "B_objects_on_A_bg": condition_from_units(model, raw_rae, objs_b, regs_a, obj_mem, reg_mem),
        }
        images = {n: generate_images(model, rae_decoder, c, x_end, args.guidance_scale, device)[0]
                  for n, c in conditions.items()}
        size = images["recon_A"].shape[-2:]
        real = {}
        for tag, item in (("A", a), ("B", b)):
            r = denormalize_images(item["sample"]["target_image"][None].to(device).float(), t_mean, t_std)
            real[tag] = F.interpolate(r, size=size, mode="bilinear", align_corners=False)[0]
        donor_mask = mask_224(b["sample"]["gt_rae_masks_per_ovt"][k_b], size).to(device)
        n_a = int(a["sample"]["n_objects"])
        a_masks = (F.interpolate(a["sample"]["gt_rae_masks_per_ovt"][:n_a].view(n_a, 1, 16, 16).float(),
                                 size=size, mode="bilinear", align_corners=False)[:, 0] > 0.5).any(0).to(device)

        stats = {
            "add": placement_stats(images["add"], images["recon_A"], donor_mask, a_masks),
            "object_on_A_bg": placement_stats(images["object_on_A_bg"], images["recon_A"], donor_mask, a_masks),
        }
        for name, s in stats.items():
            for key, value in s.items():
                if isinstance(value, float):
                    agg.setdefault(name, {}).setdefault(key, []).append(value)
        records.append({
            "prompt": prompt, "image1_id": int(a["sample"]["image_id"]), "image2_id": int(b["sample"]["image_id"]),
            "image1_categories": a["categories"], "selected_object": {"index": k_b, "category": category},
            "stats": stats,
        })

        panels = [labelled(real["A"], "image 1"), labelled(real["B"], "image 2"),
                  labelled(images["recon_A"], "recon 1"),
                  labelled(images["add"], f"1 + {category} (from 2)"),
                  labelled(images["object_on_A_bg"], f"{category} on bg of 1"),
                  labelled(images["background_swap"], "objects 1 on bg of 2"),
                  labelled(images["B_objects_on_A_bg"], "objects 2 on bg of 1"),
                  labelled(donor_mask[None].float().expand(3, -1, -1), f"{category} mask in 2")]
        grid = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), (255, 255, 255))
        x = 0
        for panel in panels:
            grid.paste(panel, (x, 0))
            x += panel.width
        grid.save(out_dir / "grids" / f"{i:02d}_img{int(a['sample']['image_id'])}_plus_{category.replace(' ', '_')}.png")
        s = stats["add"]
        print(f"[{i}] '{prompt}' img1={int(a['sample']['image_id'])} img2={int(b['sample']['image_id'])} "
              f"add: donor_iou={s.get('donor_location_iou', float('nan')):.3f} "
              f"E_donor={s.get('energy_in_donor_mask', float('nan')):.3f} (area {s['donor_mask_area_fraction']:.3f}) "
              f"centroid changed={s.get('changed_centroid_xy')} donor={s.get('donor_centroid_xy')}")

    summary = {
        "model_path": args.model_path, "num_pairs": len(records),
        "aggregate": {name: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for name, d in agg.items()},
        "records": records, "elapsed_sec": round(time.time() - start, 1),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["aggregate"], indent=1))


if __name__ == "__main__":
    main()
