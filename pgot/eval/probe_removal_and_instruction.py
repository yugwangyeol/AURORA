"""Object removal fixes and instruction-prefix probe for PGOT one-shot memory checkpoints.

Part A -- removal.  Dropping object k's (s_k, M_k) leaves the object visible
because, without the ownership prior, register and other memories also copy its
patches.  Every variant decodes with the reconstruction's noise:

  recon                  model's own owner set
  remove_set             drop object k's unit (previous probe)
  remove_caption         delete object k's caption chunk, re-run LLM + Writer
  remove_masked_pred     re-write all memories with object k's predicted patches
                         (owner argmax == k) blocked, then drop k
  remove_masked_pred_dil same, predicted mask dilated by one patch
  remove_masked_gt       same with the GT 32x32 mask (oracle)
  remove_prior           re-write with the ownership log-prior switched on, drop k
  recon_prior            ownership prior on, nothing dropped (cost of the prior)

Removal is scored by re-encoding each decoded image with the model's target
SigLIP encoder (cosine to the reconstruction inside / outside object k's mask)
and by PGOT re-captioning the decoded image autoregressively (does k's category
still appear among the generated object names?).

Part B -- instruction prefix, no training.  The user-turn instruction is
extended with "leave out the <category>" in two phrasings; PGOT generates its
caption autoregressively under that prompt and the image is rendered from it.
Scored by whether the category disappears from the generated object list and
from the render.
"""
import argparse
import copy
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.probe_object_editing import condition_from_units, forward_caption, labelled, owner_units
from pgot.eval.run_compositional_gen import generate_images, seeded_noise
from pgot.eval.run_eval import denormalize_images, generate_pgot_caption_batch, load_rae_decoder
from pgot.model.visual_memory import PGOTOneShotMemoryReader, _build_memory_valid_mask
from pgot.train.pgot_dataset import Pix2CapPGOTDataset

NAME_RE = re.compile(r"<(?:thing|stuff)>\s*([^:<]+?)\s+\d+\s*:", re.IGNORECASE)
REMOVAL = ["remove_set", "remove_caption", "remove_masked_pred", "remove_masked_pred_dil",
           "remove_masked_gt", "remove_prior", "recon_prior"]


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
    p.add_argument("--ar_max_new_tokens", type=int, default=512)
    p.add_argument("--ar_batch_size", type=int, default=4)
    p.add_argument("--skip_recaption", action="store_true")
    p.add_argument("--skip_instruction", action="store_true")
    return p.parse_args()


def object_names(text):
    return [re.sub(r"\s+", " ", m).strip().lower() for m in NAME_RE.findall(text)]


def mentions(names, category):
    category = category.lower().strip()
    return any(n == category or n == category + "s" or category in n.split() or n in category for n in names)


@torch.no_grad()
def rewrite_memories(model, out, exclude=None, use_owner_prior=False):
    """Re-run PGOTOneShotMemoryWriter on the model's own inputs, optionally with
    patches blocked for every owner and/or the ownership log-prior switched on."""
    w = model.pgot_e8_writer
    if w.softmax_axis != "patch":
        raise ValueError("rewrite_memories replicates only the patch-softmax Writer")
    semantic = out["semantic_slots"]
    B, S, D = semantic.shape
    K = out["ovt_object_valid"].shape[1]
    slot_valid = torch.cat([out["ovt_object_valid"].bool(),
                            torch.ones(B, S - K, dtype=torch.bool, device=semantic.device)], dim=1)
    valid = _build_memory_valid_mask(
        slot_valid=slot_valid, object_count=K, object_memories_per_owner=w.object_memories_per_owner,
        register_memories_per_owner=w.register_memories_per_owner, max_memories_per_owner=w.memories_per_owner)
    dtype = w.query.weight.dtype
    query = w.query(w.semantic_norm(semantic.to(dtype)))[:, :, None] + w.memory_id_embeddings[None, None]
    key = w.key(w.image_norm(out["img_hidden"].to(dtype)))
    logits = torch.einsum("bsjd,bpd->bsjp", query.float(), key.float()) / math.sqrt(D) / max(w.temperature, 1e-6)
    if use_owner_prior:
        routing = torch.cat([out["ovt_object_probs"], out["ovt_void_probs"]], dim=1).float()
        routing = routing * slot_valid[..., None].float()
        routing = routing / routing.sum(dim=1, keepdim=True).clamp_min(1e-8)
        logits = logits + routing.clamp_min(1e-8).log()[:, :, None]
    if exclude is not None:
        logits = logits.masked_fill(exclude.view(B, 1, 1, -1).to(logits.device), -1e4)
    weights = F.softmax(logits, dim=-1) * valid[..., None].float()
    raw = out["raw_img_features"].to(w.raw_value.weight.dtype)
    values = w.raw_value(w.raw_value_norm(raw))
    return torch.einsum("bsjp,bpd->bsjd", weights.to(values.dtype), values).to(semantic.dtype)


def units_with_memory(out, memory):
    K = out["ovt_object_valid"].shape[1]
    semantic = out["semantic_slots"][0]
    valid = out["ovt_object_valid"][0].bool().tolist()
    objects = [(semantic[k], memory[0, k]) for k in range(K) if valid[k]]
    registers = [(semantic[K + r], memory[0, K + r]) for r in range(semantic.shape[0] - K)]
    return objects, registers


def to_pixel_values(proc, img):
    arr = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return proc.preprocess(Image.fromarray(arr), return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def siglip_latent(model, sample, pixel_values):
    ids = sample["caption_input_ids"][None]
    out = pgot_forward_eval(
        model, images=sample["image"][None], target_images=pixel_values[None],
        caption_input_ids=ids, caption_attention_mask=torch.ones_like(ids, dtype=torch.bool),
        ovt_positions_in_caption=sample["ovt_positions_in_caption"][None],
        ovt_valid_mask=sample["ovt_valid_mask"][None])
    return out["gt_siglip"][0].float()  # [256, C]


def similarity(a, b, mask16):
    cos = F.cosine_similarity(a, b, dim=-1)
    return float(cos[mask16].mean()) if mask16.any() else float("nan"), \
        float(cos[~mask16].mean()) if (~mask16).any() else float("nan")


@torch.no_grad()
def recaption(model, tokenizer, pixel_values_list, args):
    texts = []
    for start in range(0, len(pixel_values_list), args.ar_batch_size):
        batch = torch.stack(pixel_values_list[start:start + args.ar_batch_size])
        gen = generate_pgot_caption_batch(model, tokenizer, batch, max_new_tokens=args.ar_max_new_tokens,
                                          max_objects=args.max_objects, n_ovt_per_object=1)
        texts.extend(r["text"] for r in gen["records"])
    return texts


def pick_images(dataset, args):
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
            continue
        picks.append({"idx": idx, "k": k, "sample": sample, "segments": segments, "area": float(areas[k])})
        if len(picks) == args.num_images:
            break
    return picks


def save_grid(panels, path):
    grid = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), (255, 255, 255))
    x = 0
    for panel in panels:
        grid.paste(panel, (x, 0))
        x += panel.width
    grid.save(path)


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(3)  # container CPU quota is 3 cores
    out_dir = Path(args.output_dir)
    for sub in ("removal_grids", "instruction_grids"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    start = time.time()

    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(getattr(model, "pgot_e8_reader", None), PGOTOneShotMemoryReader):
        raise ValueError("needs a one-shot memory checkpoint")
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
        image_preprocess_mode="coda_center_crop", coda_crop_size=512)
    picks = pick_images(dataset, args)
    print(f"picked {len(picks)}: {[(p['idx'], p['segments'][p['k']]['category']) for p in picks]}")
    base_suffix = tokenizer.decode(model.pgot_user_suffix_ids)
    print(f"base user suffix: {base_suffix!r}")

    removal_records, instruction_records = [], []
    replication_max_rel = 0.0
    for i, pick in enumerate(picks):
        sample, k, segments = pick["sample"], pick["k"], pick["segments"]
        category = str(segments[k]["category"])
        out = forward_caption(model, sample, sample["caption_text"], tokenizer, args)
        raw_rae = out["raw_rae_hidden"]
        x_end = seeded_noise(model, [pick["idx"]], args.seed, device)

        # ---------------- Part A: removal ----------------
        base_memory = rewrite_memories(model, out)
        rel = float((base_memory.float() - out["visual_memory"].float()).abs().max()
                    / out["visual_memory"].float().abs().mean().clamp_min(1e-12))
        replication_max_rel = max(replication_max_rel, rel)

        owner_probs = torch.cat([out["ovt_object_probs"], out["ovt_void_probs"]], dim=1)
        pred = (owner_probs.argmax(dim=1) == k)  # [1, 1024]
        pred_dil = F.max_pool2d(pred.view(1, 1, 32, 32).float(), 3, stride=1, padding=1).view(1, -1) > 0
        gt = (sample["gt_masks_per_ovt"][k] > 0.5).view(1, -1).to(pred.device)

        objects, registers = owner_units(out)
        conds = {"recon": condition_from_units(model, raw_rae, objects, registers, obj_mem, reg_mem),
                 "remove_set": condition_from_units(model, raw_rae, objects[:k] + objects[k + 1:], registers,
                                                    obj_mem, reg_mem)}
        caption_wo = dataset._build_caption_with_ovt(segments[:k] + segments[k + 1:])[0]
        conds["remove_caption"] = forward_caption(model, sample, caption_wo, tokenizer, args)["rae_hidden"]
        for name, mem_kwargs in (("remove_masked_pred", {"exclude": pred}),
                                 ("remove_masked_pred_dil", {"exclude": pred_dil}),
                                 ("remove_masked_gt", {"exclude": gt}),
                                 ("remove_prior", {"use_owner_prior": True}),
                                 ("recon_prior", {"use_owner_prior": True})):
            o, r = units_with_memory(out, rewrite_memories(model, out, **mem_kwargs))
            if name != "recon_prior":
                o = o[:k] + o[k + 1:]
            conds[name] = condition_from_units(model, raw_rae, o, r, obj_mem, reg_mem)
        images = {n: generate_images(model, rae_decoder, c, x_end, args.guidance_scale, device)[0]
                  for n, c in conds.items()}

        mask16 = (sample["gt_rae_masks_per_ovt"][k] > 0.5).to(device)
        pix = {n: to_pixel_values(target_proc, img) for n, img in images.items()}
        lat = {n: siglip_latent(model, sample, pv) for n, pv in pix.items()}
        rec = {"image_id": int(sample["image_id"]), "object": category, "k": k,
               "area_16x16": round(pick["area"], 3),
               "masks_patches": {"pred": int(pred.sum()), "pred_dil": int(pred_dil.sum()), "gt": int(gt.sum())},
               "conditions": {}}
        for n in REMOVAL:
            inside, outside = similarity(lat[n], lat["recon"], mask16)
            rec["conditions"][n] = {"siglip_cos_to_recon_inside": inside, "siglip_cos_to_recon_outside": outside}
        if not args.skip_recaption:
            order = ["recon"] + REMOVAL
            texts = recaption(model, tokenizer, [to_pixel_values(image_proc, images[n]) for n in order], args)
            for n, text in zip(order, texts):
                names = object_names(text)
                entry = rec["conditions"].setdefault(n, {})
                entry["recaption_names"] = names
                entry["recaption_mentions_object"] = mentions(names, category)
        removal_records.append(rec)
        real = denormalize_images(sample["target_image"][None].to(device).float(), t_mean, t_std)
        real = F.interpolate(real, size=images["recon"].shape[-2:], mode="bilinear", align_corners=False)[0]
        pred_img = F.interpolate(pred.view(1, 1, 32, 32).float(), size=images["recon"].shape[-2:],
                                 mode="nearest")[0].expand(3, -1, -1)
        save_grid([labelled(real, "input"), labelled(images["recon"], "recon")]
                  + [labelled(images[n], n) for n in REMOVAL]
                  + [labelled(pred_img, f"pred mask: {category}")],
                  out_dir / "removal_grids" / f"{i:02d}_img{int(sample['image_id'])}_{category.replace(' ', '_')}.png")
        print(f"[A{i}] {category}: " + " ".join(
            f"{n}:in={rec['conditions'][n]['siglip_cos_to_recon_inside']:.2f}"
            f"/out={rec['conditions'][n]['siglip_cos_to_recon_outside']:.2f}"
            f"{'/cap=' + ('Y' if rec['conditions'][n].get('recaption_mentions_object') else 'N') if not args.skip_recaption else ''}"
            for n in REMOVAL))

        # ---------------- Part B: instruction prefix ----------------
        if args.skip_instruction:
            continue
        variants = {"default": base_suffix}
        body, _, tail = base_suffix.rpartition("<|im_end|>")
        variants["instr_do_not_include"] = f"{body.rstrip()} Do not include the {category} in the description.<|im_end|>{tail}"
        if "Describe all objects and regions in this scene" in body:
            variants["instr_except"] = base_suffix.replace(
                "Describe all objects and regions in this scene",
                f"Describe all objects and regions in this scene except the {category}")
        else:
            variants["instr_except"] = f"{body.rstrip()} Exclude the {category}.<|im_end|>{tail}"
        original_suffix = list(model.pgot_user_suffix_ids)
        renders, irec = {}, {"image_id": int(sample["image_id"]), "object": category, "variants": {}}
        try:
            for name, suffix in variants.items():
                model.pgot_user_suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                gen = generate_pgot_caption_batch(model, tokenizer, sample["image"][None],
                                                  max_new_tokens=args.ar_max_new_tokens,
                                                  max_objects=args.max_objects, n_ovt_per_object=1)
                record = gen["records"][0]
                names = object_names(record["text"])
                cond = pgot_forward_eval(
                    model, images=sample["image"][None], target_images=sample["target_image"][None],
                    caption_input_ids=gen["caption_input_ids"], caption_attention_mask=gen["caption_attention_mask"],
                    ovt_positions_in_caption=gen["ovt_positions_in_caption"],
                    ovt_valid_mask=gen["ovt_valid_mask"])["rae_hidden"]
                renders[name] = generate_images(model, rae_decoder, cond, x_end, args.guidance_scale, device)[0]
                irec["variants"][name] = {
                    "prompt_suffix": suffix, "caption": record["text"], "names": names,
                    "mentions_object": mentions(names, category), "object_count": record["object_count"],
                    "format_valid": record["format_valid"]}
        finally:
            model.pgot_user_suffix_ids = original_suffix
        ref = siglip_latent(model, sample, to_pixel_values(target_proc, renders["default"]))
        for name in renders:
            if name == "default":
                continue
            inside, outside = similarity(siglip_latent(model, sample, to_pixel_values(target_proc, renders[name])),
                                         ref, mask16)
            irec["variants"][name]["siglip_cos_to_default_render_inside"] = inside
            irec["variants"][name]["siglip_cos_to_default_render_outside"] = outside
        instruction_records.append(irec)
        save_grid([labelled(real, "input")] + [labelled(img, n) for n, img in renders.items()],
                  out_dir / "instruction_grids" / f"{i:02d}_img{int(sample['image_id'])}_{category.replace(' ', '_')}.png")
        print(f"[B{i}] {category}: " + " ".join(
            f"{n}:mention={v['mentions_object']}/objs={v['object_count']}" for n, v in irec["variants"].items()))

    def mean(values):
        values = [v for v in values if isinstance(v, (int, float, bool)) and not (isinstance(v, float) and math.isnan(v))]
        return round(float(np.mean([float(v) for v in values])), 4) if values else None

    removal_agg = {}
    for n in REMOVAL + ["recon"]:
        rows = [r["conditions"].get(n, {}) for r in removal_records]
        removal_agg[n] = {key: mean([row.get(key) for row in rows]) for key in
                          ("siglip_cos_to_recon_inside", "siglip_cos_to_recon_outside", "recaption_mentions_object")}
    instruction_agg = {}
    for n in ("default", "instr_do_not_include", "instr_except"):
        rows = [r["variants"].get(n, {}) for r in instruction_records]
        instruction_agg[n] = {key: mean([row.get(key) for row in rows]) for key in
                              ("mentions_object", "object_count", "format_valid",
                               "siglip_cos_to_default_render_inside", "siglip_cos_to_default_render_outside")}
    summary = {
        "model_path": args.model_path, "num_images": len(picks),
        "writer_replication_max_relative_error": replication_max_rel,
        "removal_aggregate": removal_agg, "instruction_aggregate": instruction_agg,
        "removal_records": removal_records, "instruction_records": instruction_records,
        "elapsed_sec": round(time.time() - start, 1)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"writer_replication_max_relative_error": replication_max_rel,
                      "removal": removal_agg, "instruction": instruction_agg}, indent=1))


if __name__ == "__main__":
    main()
