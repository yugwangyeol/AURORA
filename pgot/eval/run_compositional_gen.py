"""CODA-style compositional generation for PGOT one-shot memory checkpoints.

CODA (arXiv:2601.01224, Table 4) evaluates compositional generation on COCO:
"Following Wu et al. (2023), these configurations are created by randomly
mixing slots within a batch", scored with FID and KID x1e3.  The CODA release
ships no mixing script, so every choice the paper leaves open (register slots,
object count, sampling with/without replacement) is an explicit flag and is
recorded in summary.json.

Why composition is exact for one-shot memory models:
  * RAE query rows are self-only (``_pgot_e8_block_standard_rae_values``), so
    ``raw_rae_hidden`` is the same 256 queries for every image.
  * ``PGOTOneShotMemoryReader`` consumes only owner units (semantic slot s_k,
    visual memories M_k) plus validity; it never sees patches or ownership.
A mixed owner set therefore decodes through the same Reader -> DiT -> RAE
decoder path as a reconstruction.  Rebuilding the unmixed set must reproduce
the model's own condition; that is checked on every batch.

Usage:
    bash scripts/eval_pgot_compositional_gen.sh            # full val set
    bash scripts/smoke_pgot_compositional_gen.sh           # 8-image smoke
"""
import argparse
import json
import logging
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.eval.eval_recon_oracles import load_model_and_tokenizer
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import FIDAccumulator, KIDAccumulator
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.model.visual_memory import PGOTOneShotMemoryReader, _build_memory_valid_mask
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.compositional_gen")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Images per batch; slots are mixed only within a batch (CODA protocol).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--max_caption_tokens", type=int, default=1024)
    p.add_argument("--n_ovt_per_object", type=int, default=1)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--image_preprocess_mode", default="coda_center_crop")
    p.add_argument("--coda_crop_size", type=int, default=512)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--diffusion_inference_steps", type=int, default=10)
    p.add_argument("--mix_mode", choices=["random_slots", "identity"], default="random_slots",
                   help="random_slots: each object slot comes from another image in the batch. "
                        "identity: no mixing (reconstruction through the same code path).")
    p.add_argument("--register_source", choices=["random", "target", "per_register"], default="random",
                   help="Background registers: all from one random other image (random), kept from "
                        "the target (target), or each register from an independent random image.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--kid_subsets", type=int, default=100)
    p.add_argument("--kid_subset_size", type=int, default=1000)
    p.add_argument("--save_examples", type=int, default=8,
                   help="Number of batches saved as [real | identity recon | composed] grids.")
    p.add_argument("--identity_tolerance", type=float, default=1e-3,
                   help="Max relative error between the rebuilt identity condition and the model's.")
    return p.parse_args()


def build_condition(model, raw_rae, semantic, memory, object_valid, object_memories, register_memories):
    """Reader condition for an arbitrary [objects | registers] owner set."""
    B, K = object_valid.shape
    R = semantic.shape[1] - K
    slot_valid = torch.cat(
        [object_valid.bool(), torch.ones(B, R, device=object_valid.device, dtype=torch.bool)], dim=1
    )
    memory_valid = _build_memory_valid_mask(
        slot_valid=slot_valid,
        object_count=K,
        object_memories_per_owner=object_memories,
        register_memories_per_owner=register_memories,
        max_memories_per_owner=memory.shape[2],
    )
    return model.pgot_e8_reader(
        rae_queries=raw_rae,
        semantic_slots=semantic,
        visual_memory=memory,
        slot_valid=slot_valid,
        memory_valid=memory_valid,
    )["condition_hidden"]


def plan_mixing(object_valid, n_registers, rng, register_source):
    """Choose donor (image, slot) pairs for every target image in the batch."""
    B, K = object_valid.shape
    valid = object_valid.cpu().tolist()
    pool = [(j, k) for j in range(B) for k in range(K) if valid[j][k]]
    plans = []
    for i in range(B):
        own = [(i, k) for k in range(K) if valid[i][k]]
        others = [j for j in range(B) if j != i]
        plan = {"with_replacement": False, "unmixed": False}
        donors = [p for p in pool if p[0] != i]
        if not others:
            plan["objects"], plan["unmixed"] = own, True
        elif len(donors) >= len(own):
            plan["objects"] = rng.sample(donors, len(own))
        elif donors:
            plan["objects"] = [rng.choice(donors) for _ in own]
            plan["with_replacement"] = True
        else:
            plan["objects"], plan["unmixed"] = own, True
        if not others or register_source == "target":
            plan["registers"] = [(i, r) for r in range(n_registers)]
        elif register_source == "random":
            j = rng.choice(others)
            plan["registers"] = [(j, r) for r in range(n_registers)]
        else:
            plan["registers"] = [(rng.choice(others), r) for r in range(n_registers)]
        plans.append(plan)
    return plans


def gather_mixed(semantic, memory, n_objects_src, plans):
    """Materialize mixed [objects | registers] tensors from donor plans."""
    B, S, D = semantic.shape
    J = memory.shape[2]
    R = S - n_objects_src
    K_mix = max((len(p["objects"]) for p in plans), default=0)
    sem = semantic.new_zeros(B, K_mix + R, D)
    mem = memory.new_zeros(B, K_mix + R, J, D)
    obj_valid = torch.zeros(B, K_mix, device=semantic.device, dtype=torch.bool)
    for i, plan in enumerate(plans):
        for t, (j, k) in enumerate(plan["objects"]):
            sem[i, t] = semantic[j, k]
            mem[i, t] = memory[j, k]
            obj_valid[i, t] = True
        for r, (j, rr) in enumerate(plan["registers"]):
            sem[i, K_mix + r] = semantic[j, n_objects_src + rr]
            mem[i, K_mix + r] = memory[j, n_objects_src + rr]
    return sem, mem, obj_valid


def seeded_noise(model, indices, seed, device):
    """Per-sample fixed x_end so reruns and identity/composed pairs share noise."""
    diff_head = model.diff_head
    C = int(diff_head.diffusion_channels)
    side = int(round(int(diff_head.diffusion_tokens) ** 0.5))
    noise = []
    for idx in indices:
        g = torch.Generator(device="cpu").manual_seed(int(seed) * 1_000_003 + int(idx))
        noise.append(torch.randn((C, side, side), generator=g, dtype=torch.float32))
    return torch.stack(noise).to(device)


@torch.no_grad()
def generate_images(model, rae_decoder, condition, x_end, guidance, device):
    cond = model._captionslot_prepare_diffusion_condition(condition).float()
    model.set_diff_fp32()
    latent = model.diff_head.infer(z=cond, x_end=x_end, guidance_level=guidance)
    return decode_to_image(rae_decoder, latent, device)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.time()

    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    if not isinstance(getattr(model, "pgot_e8_reader", None), PGOTOneShotMemoryReader):
        raise ValueError(
            "Compositional generation needs a one-shot memory checkpoint "
            "(pgot_one_shot_readout_mode memory_content/memory_id); got "
            f"{type(getattr(model, 'pgot_e8_reader', None)).__name__}"
        )
    writer = model.pgot_e8_writer
    object_memories = int(writer.object_memories_per_owner)
    register_memories = int(writer.register_memories_per_owner)
    rae_decoder = load_rae_decoder(model, device=device, dtype=torch.float32)

    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    t_mean = torch.tensor(target_proc.image_mean).view(1, -1, 1, 1)
    t_std = torch.tensor(target_proc.image_std).view(1, -1, 1, 1)
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_proc,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
        panoptic_categories_json="/home/jovyan/data/coco/annotations/panoptic_val2017.json",
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    n_total = len(dataset)
    log.info("Eval set: %d images | mix_mode=%s register_source=%s batch=%d",
             n_total, args.mix_mode, args.register_source, args.batch_size)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=PGOTDataCollator(pad_token_id=tokenizer.pad_token_id),
        num_workers=args.num_workers, pin_memory=True,
    )

    fid_acc = FIDAccumulator(device=device, feature=2048)
    kid_subset_size = min(int(args.kid_subset_size), n_total)
    kid_acc = KIDAccumulator(device=device, feature=2048, subsets=int(args.kid_subsets),
                             subset_size=kid_subset_size) if kid_subset_size >= 2 else None

    rng = random.Random(args.seed)
    comp_path = os.path.join(args.output_dir, "compositions.jsonl")
    comp_file = open(comp_path, "w")
    example_files = []
    identity_max_rel = 0.0
    rae_rel_std = 0.0
    mixed_delta_sum, mixed_delta_batches = 0.0, 0
    n_objects_total = n_foreign = n_replacement = n_unmixed = 0
    n_generated = n_real = 0
    fake_hw = real_hw = None
    global_index = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="compgen")):
        B = batch["images"].shape[0]
        indices = list(range(global_index, global_index + B))
        global_index += B
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        semantic = out["semantic_slots"]
        memory = out["visual_memory"]
        object_valid = out["ovt_object_valid"].bool()
        raw_rae = out["raw_rae_hidden"]
        K = object_valid.shape[1]
        R = semantic.shape[1] - K

        identity_cond = build_condition(model, raw_rae, semantic, memory, object_valid,
                                        object_memories, register_memories)
        own = out["rae_hidden"].float()
        rel = float((identity_cond.float() - own).abs().max() / own.abs().mean().clamp_min(1e-12))
        identity_max_rel = max(identity_max_rel, rel)
        if B > 1:
            rq = raw_rae.float()
            rae_rel_std = max(rae_rel_std, float(rq.std(dim=0).max() / rq.abs().mean().clamp_min(1e-12)))

        image_ids = list(batch["image_ids"])
        if args.mix_mode == "identity":
            plans = [{"objects": [(i, k) for k in range(K) if object_valid[i, k]],
                      "registers": [(i, r) for r in range(R)],
                      "with_replacement": False, "unmixed": True} for i in range(B)]
            condition = identity_cond
        else:
            plans = plan_mixing(object_valid, R, rng, args.register_source)
            sem_mix, mem_mix, obj_valid_mix = gather_mixed(semantic, memory, K, plans)
            condition = build_condition(model, raw_rae, sem_mix, mem_mix, obj_valid_mix,
                                        object_memories, register_memories)
            mixed_delta_sum += float((condition.float() - identity_cond.float()).abs().mean())
            mixed_delta_batches += 1

        x_end = seeded_noise(model, indices, args.seed, device)
        fake = generate_images(model, rae_decoder, condition, x_end, args.guidance_scale, device)
        real = denormalize_images(batch["target_images"].to(device).float(), t_mean, t_std)
        real_hw = tuple(real.shape[-2:])
        if real.shape[-2:] != fake.shape[-2:]:
            real = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
        fake_hw = tuple(fake.shape[-2:])
        fid_acc.add(real, fake)
        if kid_acc is not None:
            kid_acc.add(real, fake)
        n_generated += fake.shape[0]
        n_real += real.shape[0]

        for i, plan in enumerate(plans):
            n_objects_total += len(plan["objects"])
            n_foreign += sum(1 for j, _ in plan["objects"] if j != i)
            n_replacement += int(plan["with_replacement"])
            n_unmixed += int(plan["unmixed"])
            comp_file.write(json.dumps({
                "index": indices[i],
                "target_image_id": image_ids[i],
                "objects": [{"image_id": image_ids[j], "object_index": k} for j, k in plan["objects"]],
                "registers": [{"image_id": image_ids[j], "register_index": r} for j, r in plan["registers"]],
                "with_replacement": plan["with_replacement"],
            }) + "\n")

        if batch_idx < args.save_examples:
            from torchvision.utils import save_image

            recon = fake if args.mix_mode == "identity" else generate_images(
                model, rae_decoder, identity_cond, x_end, args.guidance_scale, device)
            grid_path = os.path.join(args.output_dir, "examples", f"batch{batch_idx:03d}_real_recon_composed.png")
            os.makedirs(os.path.dirname(grid_path), exist_ok=True)
            save_image(torch.cat([real, recon, fake]).cpu(), grid_path, nrow=B, padding=2)
            example_files.append(grid_path)
    comp_file.close()

    fid = fid_acc.compute()
    kid_mean = kid_std = float("nan")
    if kid_acc is not None:
        kid_mean, kid_std = kid_acc.compute()
    summary = {
        "task": "compositional_generation",
        "protocol": ("CODA Table 4 (arXiv:2601.01224): 'configurations are created by randomly "
                     "mixing slots within a batch'; FID and KID x1e3 of composed images vs real images. "
                     "CODA publishes no mixing script: register handling, object count and sampling "
                     "rule below are PGOT choices."),
        "mixing_rule": ("each target keeps its own object count; every object slot (semantic slot + "
                        "its visual memories) is drawn without replacement from the other images' "
                        "valid objects in the same batch; registers follow register_source"),
        "model_path": args.model_path,
        "one_shot_readout_mode": str(getattr(model.config, "pgot_one_shot_readout_mode", "")),
        "object_memories_per_owner": object_memories,
        "register_memories_per_owner": register_memories,
        "mix_mode": args.mix_mode,
        "register_source": args.register_source,
        "batch_size_mixing_pool": args.batch_size,
        "seed": args.seed,
        "caption_mode": "teacher_forced",
        "diffusion_inference_steps": args.diffusion_inference_steps,
        "guidance_scale": args.guidance_scale,
        "real_reference": "the same val images (CODA-512 center crop), one real per generated image",
        "real_resolution_before_resize": list(real_hw) if real_hw else None,
        "metric_resolution": list(fake_hw) if fake_hw else None,
        "n_real": n_real,
        "n_generated": n_generated,
        "FID": fid,
        "KID_x1e3_mean": kid_mean * 1e3,
        "KID_x1e3_std": kid_std * 1e3,
        "kid_subsets": int(args.kid_subsets),
        "kid_subset_size": kid_subset_size,
        "identity_check": {
            "passed": identity_max_rel <= args.identity_tolerance,
            "max_relative_error": identity_max_rel,
            "tolerance": args.identity_tolerance,
        },
        "raw_rae_query_batch_std": rae_rel_std,
        "mix_stats": {
            "mean_objects_per_image": n_objects_total / max(n_generated, 1),
            "foreign_object_slot_fraction": n_foreign / max(n_objects_total, 1),
            "images_sampled_with_replacement": n_replacement,
            "images_left_unmixed": n_unmixed,
            "mixed_condition_mean_abs_delta": (mixed_delta_sum / mixed_delta_batches
                                               if mixed_delta_batches else 0.0),
        },
        "peak_gpu_memory_GiB": (torch.cuda.max_memory_allocated() / 2 ** 30
                                if torch.cuda.is_available() else None),
        "elapsed_sec": time.time() - start,
        "compositions_file": comp_path,
        "example_files": example_files,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("FID=%.3f KID(x1e3)=%.3f±%.3f | identity max rel err=%.2e | n=%d -> %s",
             fid, kid_mean * 1e3, kid_std * 1e3, identity_max_rel, n_generated, args.output_dir)
    if not summary["identity_check"]["passed"]:
        raise RuntimeError(f"Identity check failed: {summary['identity_check']}")


if __name__ == "__main__":
    main()
