"""External-controller image editing with PGOT object tokens (PGOT side).

Stage ``captions`` (scale_rae env): PGOT captions every image autoregressively
-- no annotation captions -- and the generated object list of each image pair
is saved with three instructions whose expected token plan is known:
  add      "Put the <B object> from the second picture into the first picture."
  remove   "Remove the <A object> from the first picture."
  replace  "Replace the <A object> in the first picture with the <B object> from the second one."

Stage ``plan`` runs in another env (``pgot/eval/controller_plan_llm.py``): an
instruction-tuned LLM reads the two object lists + instruction and writes
{"background", "remove_image1", "add_image2"}.

Stage ``render`` (scale_rae env): executes each plan on the PGOT owner set --
image-1 objects not removed + image-2 objects added + the chosen background's
registers -- and decodes with image 1's noise.  Removal can use plain token
dropping (``set``) or re-write memories with the removed objects' predicted
patches blocked (``masked_pred_dil``) or with the ownership prior (``prior``).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.probe_object_editing import condition_from_units, labelled, owner_units
from pgot.eval.probe_removal_and_instruction import rewrite_memories, units_with_memory
from pgot.eval.run_compositional_gen import generate_images, seeded_noise
from pgot.eval.run_eval import denormalize_images, generate_pgot_caption_batch, load_rae_decoder
from pgot.model.visual_memory import PGOTOneShotMemoryReader
from pgot.train.pgot_dataset import Pix2CapPGOTDataset

OBJECT_RE = re.compile(r"<thing>\s*([^:<]+?)\s+(\d+)\s*:\s*(.*?)<ovt>", re.IGNORECASE | re.DOTALL)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["captions", "render"], required=True)
    p.add_argument("--model_path",
                   default="/home/jovyan/PGOT/checkpoints/pgot_oneshot_memory8_register32_noprior/checkpoint-5000")
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_pix2cap_generated_val5k.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_pairs", type=int, default=6)
    p.add_argument("--scan_images", type=int, default=60)
    p.add_argument("--plans", default=None, help="plans.json from controller_plan_llm.py (render stage)")
    p.add_argument("--removal", choices=["set", "masked_pred_dil", "prior"], default="masked_pred_dil")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--diffusion_inference_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max_caption_tokens", type=int, default=1024)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--ar_max_new_tokens", type=int, default=512)
    return p.parse_args()


def parse_objects(text):
    objects = []
    for match in OBJECT_RE.finditer(text):
        name = re.sub(r"\s+", " ", match.group(1)).strip()
        description = re.sub(r"\s+", " ", match.group(3)).strip().rstrip(".")
        objects.append({"id": len(objects), "name": name.lower(), "label": f"{name} {match.group(2)}",
                        "description": " ".join(description.split()[:14])})
    return objects


def unique_name(objects, exclude=()):
    counts = {}
    for obj in objects:
        counts[obj["name"]] = counts.get(obj["name"], 0) + 1
    for obj in objects:  # PGOT lists objects in descending visible area
        if counts[obj["name"]] == 1 and obj["name"] not in exclude:
            return obj
    return None


def tensors_from_ids(ids, tokenizer, max_objects):
    token_ids = torch.tensor(ids, dtype=torch.long)[None]
    ovt_id = int(tokenizer.convert_tokens_to_ids("<ovt>"))
    positions = torch.nonzero(token_ids[0] == ovt_id, as_tuple=False).flatten()[:max_objects]
    ovt_positions = torch.zeros((1, max_objects), dtype=torch.long)
    ovt_valid = torch.zeros((1, max_objects), dtype=torch.bool)
    ovt_positions[0, : positions.numel()] = positions
    ovt_valid[0, : positions.numel()] = True
    return token_ids, torch.ones_like(token_ids, dtype=torch.bool), ovt_positions, ovt_valid


def build(args):
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(getattr(model, "pgot_e8_reader", None), PGOTOneShotMemoryReader):
        raise ValueError("needs a one-shot memory checkpoint")
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl, tokenizer=tokenizer, image_processor=image_proc,
        target_image_processor=target_proc, grid_size=32, max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=1, max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode="coda_center_crop", coda_crop_size=512)
    return model, tokenizer, device, dataset, target_proc


@torch.no_grad()
def stage_captions(args):
    model, tokenizer, device, dataset, _ = build(args)
    images = []
    for idx in range(min(args.scan_images, len(dataset))):
        sample = dataset[idx]
        gen = generate_pgot_caption_batch(model, tokenizer, sample["image"][None],
                                          max_new_tokens=args.ar_max_new_tokens,
                                          max_objects=args.max_objects, n_ovt_per_object=1)
        record = gen["records"][0]
        objects = parse_objects(record["text"])
        if not record["format_valid"] or len(objects) != record["object_count"] or len(objects) < 1:
            continue
        ids = gen["caption_input_ids"][0][gen["caption_attention_mask"][0].bool()].tolist()
        images.append({"sample_index": idx, "image_id": int(sample["image_id"]), "caption": record["text"],
                       "objects": objects, "caption_ids": ids})
        print(f"captioned {idx}: {[o['label'] for o in objects]}")
        if len(images) >= 2 * args.num_pairs + 6:
            break
    pairs, used = [], set()
    for a in images:
        if len(pairs) == args.num_pairs or a["sample_index"] in used:
            continue
        obj_a = unique_name(a["objects"])
        for b in images:
            if b["sample_index"] in used or b["sample_index"] == a["sample_index"] or obj_a is None:
                continue
            obj_b = unique_name(b["objects"], exclude={o["name"] for o in a["objects"]})
            if obj_b is None:
                continue
            used.update({a["sample_index"], b["sample_index"]})
            ia, jb = obj_a["id"], obj_b["id"]
            pairs.append({"pair_id": len(pairs), "image1": a, "image2": b, "instructions": [
                {"type": "add", "text": f"Put the {obj_b['name']} from the second picture into the first picture.",
                 "expected": {"background": "image1", "remove_image1": [], "add_image2": [jb]}},
                {"type": "remove", "text": f"Remove the {obj_a['name']} from the first picture.",
                 "expected": {"background": "image1", "remove_image1": [ia], "add_image2": []}},
                {"type": "replace",
                 "text": f"Replace the {obj_a['name']} in the first picture with the {obj_b['name']} from the second one.",
                 "expected": {"background": "image1", "remove_image1": [ia], "add_image2": [jb]}},
            ]})
            break
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pairs.json", "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"saved {len(pairs)} pairs -> {out / 'pairs.json'}")


@torch.no_grad()
def stage_render(args):
    model, tokenizer, device, dataset, target_proc = build(args)
    obj_mem = int(model.pgot_e8_writer.object_memories_per_owner)
    reg_mem = int(model.pgot_e8_writer.register_memories_per_owner)
    rae_decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    t_mean = torch.tensor(target_proc.image_mean).view(1, -1, 1, 1)
    t_std = torch.tensor(target_proc.image_std).view(1, -1, 1, 1)
    plans = json.load(open(args.plans))
    out_dir = Path(args.output_dir) / f"render_{args.removal}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pair in plans:
        forwards = {}
        for tag in ("image1", "image2"):
            info = pair[tag]
            sample = dataset[info["sample_index"]]
            ids, attention, positions, valid = tensors_from_ids(info["caption_ids"], tokenizer, args.max_objects)
            forwards[tag] = (sample, pgot_forward_eval(
                model, images=sample["image"][None], target_images=sample["target_image"][None],
                caption_input_ids=ids, caption_attention_mask=attention,
                ovt_positions_in_caption=positions, ovt_valid_mask=valid))
        sample1, out1 = forwards["image1"]
        sample2, out2 = forwards["image2"]
        objs1, regs1 = owner_units(out1)
        objs2, regs2 = owner_units(out2)
        x_end = seeded_noise(model, [pair["image1"]["sample_index"]], args.seed, device)
        raw_rae = out1["raw_rae_hidden"]
        panels = []
        for tag, sample in (("image 1", sample1), ("image 2", sample2)):
            real = denormalize_images(sample["target_image"][None].to(device).float(), t_mean, t_std)
            panels.append(labelled(F.interpolate(real, size=(224, 224), mode="bilinear", align_corners=False)[0], tag))
        recon = generate_images(model, rae_decoder, condition_from_units(model, raw_rae, objs1, regs1, obj_mem, reg_mem),
                                x_end, args.guidance_scale, device)[0]
        panels.append(labelled(recon, "recon 1"))
        for inst in pair["instructions"]:
            plan = inst.get("plan")
            if not isinstance(plan, dict):
                panels.append(labelled(torch.zeros_like(recon), f"{inst['type']}: no plan"))
                continue
            remove = [int(i) for i in plan.get("remove_image1", []) if 0 <= int(i) < len(objs1)]
            add = [int(j) for j in plan.get("add_image2", []) if 0 <= int(j) < len(objs2)]
            kept1, reg1 = objs1, regs1
            if remove and args.removal != "set":
                owner_probs = torch.cat([out1["ovt_object_probs"], out1["ovt_void_probs"]], dim=1)
                if args.removal == "masked_pred_dil":
                    hit = torch.isin(owner_probs.argmax(dim=1), torch.tensor(remove, device=owner_probs.device))
                    hit = F.max_pool2d(hit.view(1, 1, 32, 32).float(), 3, stride=1, padding=1).view(1, -1) > 0
                    memory = rewrite_memories(model, out1, exclude=hit)
                else:
                    memory = rewrite_memories(model, out1, use_owner_prior=True)
                kept1, reg1 = units_with_memory(out1, memory)
            objects = [u for i, u in enumerate(kept1) if i not in remove] + [objs2[j] for j in add]
            registers = regs2 if plan.get("background") == "image2" else reg1
            cond = condition_from_units(model, raw_rae, objects, registers, obj_mem, reg_mem)
            image = generate_images(model, rae_decoder, cond, x_end, args.guidance_scale, device)[0]
            panels.append(labelled(image, f"{inst['type']}: -{remove} +{add}"))
            results.append({"pair_id": pair["pair_id"], "type": inst["type"], "instruction": inst["text"],
                            "plan": plan, "correct": inst.get("correct"), "removal": args.removal})
        grid = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), (255, 255, 255))
        x = 0
        for panel in panels:
            grid.paste(panel, (x, 0))
            x += panel.width
        grid.save(out_dir / f"pair{pair['pair_id']:02d}_img{pair['image1']['image_id']}_img{pair['image2']['image_id']}.png")
        print(f"pair {pair['pair_id']} rendered")
    with open(out_dir / "render_log.json", "w") as f:
        json.dump(results, f, indent=2)


def main():
    args = parse_args()
    torch.set_num_threads(3)  # container CPU quota is 3 cores
    start = time.time()
    if args.stage == "captions":
        stage_captions(args)
    else:
        if not args.plans:
            raise ValueError("--plans is required for --stage render")
        stage_render(args)
    print(f"done in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
