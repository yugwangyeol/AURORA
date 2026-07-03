"""Post-hoc readout sweep (NO retraining).

Caches OVT logits + GT masks ONCE (single model forward over the eval set),
then evaluates many readout configs to find the one that maximizes fARI:

    merge        : mean | max
    competition  : sigmoid | softmax   (per-pixel object competition)
    temp         : eval-time sharpening
    bg_threshold : background channel score
    use_bg       : bg as competing channel vs post-hoc threshold

Reports a table sorted by fARI, and computes mBO/mIoU for the top configs.

Usage:
    PYTHONPATH=/home/jovyan/PGOT python -m pgot.eval.sweep_readout \
        --model_path /home/jovyan/PGOT/checkpoints/pgot_main_v3/checkpoint-14000 \
        --gt_source coco_instance --max_samples 500 \
        --output_dir /home/jovyan/PGOT/outputs/sweep_readout_v3
"""
import argparse
import json
import logging
import os
import sys
import itertools

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")
from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN, NEW_SPECIAL_TOKENS
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.train.pgot_dataset import Pix2CapPGOTDataset, PGOTDataCollator
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import (
    build_pred_mask_readout, fari_metric, mbo_metric, miou_metric,
)
from pgot.eval.run_eval import load_gt_panoptic_mask, CocoInstanceMaskCache, load_thing_categories

from transformers import AutoTokenizer, AutoConfig

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s :: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pgot.sweep_readout")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--max_caption_tokens", type=int, default=2048)
    p.add_argument("--n_ovt_per_object", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--gt_source", choices=["pix2cap_panoptic", "coco_instance"], default="coco_instance")
    p.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    p.add_argument(
        "--image_preprocess_mode",
        choices=["default", "coda_center_crop"],
        default="default",
        help="Match run_eval.py image preprocessing.",
    )
    p.add_argument("--coda_crop_size", type=int, default=512)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    # Sweep grids (comma-separated)
    p.add_argument("--bg_thresholds", default="0.0,0.01,0.03,0.05,0.10,0.15,0.20")
    p.add_argument("--temps", default="0.3,0.5,0.7,1.0")
    p.add_argument("--merges", default="mean,max")
    p.add_argument("--competitions", default="sigmoid,softmax")
    p.add_argument("--use_bgs", default="1,0")
    p.add_argument("--topk_full_metric", type=int, default=5, help="compute mBO/mIoU for top-K fARI configs")
    p.add_argument("--pareto", action="store_true", help="compute mBO/mIoU across the whole fARI range (Pareto front)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    # ---- Load model
    log.info(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    raw_config_path = os.path.join(args.model_path, "config.json")
    raw_name_or_path = args.model_path
    if os.path.exists(raw_config_path):
        with open(raw_config_path, "r") as f:
            raw_cfg = json.load(f)
        raw_name_or_path = str(raw_cfg.get("_name_or_path", args.model_path))
    import glob, safetensors.torch as safe_torch
    has_lora = False
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_json = json.load(f)
        has_lora = any(
            "lora_" in k or ".base_layer." in k
            for k in index_json.get("weight_map", {}).keys()
        )
    else:
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                if any("lora_" in k or ".base_layer." in k for k in f.keys()):
                    has_lora = True
                    break
    model_init_path = raw_name_or_path if has_lora else args.model_path
    if has_lora:
        log.info("[LoRA] adapter checkpoint detected; bootstrap model from base path: %s", model_init_path)
    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_init_path, config=config, torch_dtype=dtype, ignore_mismatched_sizes=True)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643

    if has_lora:
        from peft import LoraConfig, inject_adapter_in_model
        inject_adapter_in_model(LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                task_type="CAUSAL_LM"), model, adapter_name="default")
        sd = {}
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("LoRA re-loaded | missing=%d unexpected=%d", len(missing), len(unexpected))

    parsed_towers = getattr(config, "mm_vision_tower_aux_list", None) or json.loads(
        getattr(config, "vision_tower_aux_list", '["google/siglip2-so400m-patch14-224"]'))
    parsed_token_lens = getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]
    from types import SimpleNamespace
    vt_args = SimpleNamespace(
        vision_tower_aux_list=parsed_towers, vision_tower_aux_token_len_list=parsed_token_lens,
        mm_vision_select_layer=-1, mm_vision_select_feature="patch", mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=True, mm_use_im_patch_token=False, unfreeze_mm_vision_tower=False,
        vision_hidden_size=1024, connector_only=True, pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None))
    model.get_model().initialize_vision_modules(model_args=vt_args, fsdp=None)
    model.load_vision_head(model_args=vt_args)
    for vt in model.get_vision_tower_aux_list():
        vt.to(dtype=dtype, device=device)
    model.pgot_ovt_token_id = tokenizer.convert_tokens_to_ids(OVT_TOKEN)
    model.pgot_scene_end_token_id = tokenizer.convert_tokens_to_ids(SCENE_END_TOKEN)
    blocks = {
        "pgot_system_prefix_ids": "<|im_start|>system\nYou are a vision assistant that describes scenes with grounded objects.",
        "pgot_system_suffix_ids": "<|im_end|>\n",
        "pgot_user_prefix_ids": "<|im_start|>user\n",
        "pgot_user_suffix_ids": "\nDescribe all objects and regions in this scene with grounded tokens.<|im_end|>\n",
        "pgot_assistant_prefix_ids": "<|im_start|>assistant\n",
        "pgot_assistant_suffix_ids": "<|im_end|>",
    }
    for attr, txt in blocks.items():
        setattr(model, attr, tokenizer.encode(txt, add_special_tokens=False))
    model.to(device=device, dtype=dtype)
    model.eval()
    log.info("Model loaded.")

    # ---- GT cache
    coco_cache = None
    thing_categories = None
    if args.gt_source == "coco_instance":
        coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
        args.eval_size = coco_cache.size
        thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")

    # ---- Dataset
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl, tokenizer=tokenizer,
        image_processor=image_proc, target_image_processor=target_proc,
        grid_size=args.grid_size, max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object, max_objects=args.max_objects,
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(args.max_samples)))
    samples_ref = dataset.dataset.samples if isinstance(dataset, torch.utils.data.Subset) else dataset.samples
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator, num_workers=args.num_workers, pin_memory=True)
    log.info(f"Eval set: {len(dataset)} samples")

    # =========================================================================
    # PASS 1: cache OVT logits + valid + GT mask (single forward over the set)
    # =========================================================================
    cached_logits = []   # list of (M,P) on CPU
    cached_valid = []     # list of (M,) bool
    cached_gt = []        # list of (H,W) int
    cached_nobj = []
    log.info("PASS 1: caching OVT logits + GT masks ...")
    gidx = 0
    for batch in tqdm(loader, desc="forward"):
        out = pgot_forward_eval(
            model,
            images=batch["images"], target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"])
        ovt_logits = out["ovt_logits"].detach().cpu()   # (B,M,P)
        ovt_valid = out["ovt_valid_mask"].detach().cpu()
        B = ovt_logits.shape[0]
        for b in range(B):
            samp = samples_ref[gidx]
            gidx += 1
            valid_b = ovt_valid[b].clone()
            # thing-only filter for coco_instance
            if thing_categories is not None:
                segs = samp["segments"]
                for k, seg in enumerate(segs):
                    if seg["category"] not in thing_categories:
                        s = k * args.n_ovt_per_object
                        e = s + args.n_ovt_per_object
                        if e <= valid_b.shape[0]:
                            valid_b[s:e] = False
            # GT mask
            if args.gt_source == "coco_instance":
                gt = coco_cache.get(int(samp["image_id"]))
                if gt is None:
                    seg_ids = [int(s["segment_id"]) for s in samp["segments"]]
                    gt = load_gt_panoptic_mask(samp["panoptic_mask_path"], seg_ids, args.eval_size)
            else:
                seg_ids = [int(s["segment_id"]) for s in samp["segments"]]
                gt = load_gt_panoptic_mask(samp["panoptic_mask_path"], seg_ids, args.eval_size)
            cached_logits.append(ovt_logits[b])
            cached_valid.append(valid_b)
            cached_gt.append(gt)
            cached_nobj.append(int(samp.get("n_objects", 0)))
    n = len(cached_gt)
    log.info(f"Cached {n} samples.")

    # =========================================================================
    # PASS 2: readout sweep
    # =========================================================================
    bg_list = [float(x) for x in args.bg_thresholds.split(",")]
    temp_list = [float(x) for x in args.temps.split(",")]
    merge_list = [m.strip() for m in args.merges.split(",")]
    comp_list = [c.strip() for c in args.competitions.split(",")]
    usebg_list = [bool(int(x)) for x in args.use_bgs.split(",")]

    configs = list(itertools.product(merge_list, comp_list, temp_list, bg_list, usebg_list))
    log.info(f"PASS 2: sweeping {len(configs)} readout configs ...")

    # Stack cached tensors to GPU in chunks for speed
    gt_stack = torch.stack(cached_gt).to(device)              # (N,H,W)
    logits_stack = torch.stack(cached_logits).to(device)     # (N,M,P)
    valid_stack = torch.stack(cached_valid).to(device)       # (N,M)

    results = []
    for (merge, comp, temp, bg, use_bg) in tqdm(configs, desc="readout"):
        # Build pred masks for all samples in batches (avoid OOM)
        fari_vals = []
        BS = 64
        for s in range(0, n, BS):
            e = min(s + BS, n)
            pred = build_pred_mask_readout(
                logits_stack[s:e], valid_stack[s:e],
                target_size=args.eval_size, n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size, merge=merge, competition=comp,
                temp=temp, bg_threshold=bg, use_bg_channel=use_bg)
            # batched fARI per sample
            from pgot.eval.pgot_metrics import _adjusted_rand_index
            ari = _adjusted_rand_index(gt_stack[s:e], pred, ignore_background=True)  # (b,)
            fari_vals.extend(ari.detach().cpu().tolist())
        fari = float(np.nanmean([v for v in fari_vals if not np.isnan(v)]))
        results.append({"merge": merge, "competition": comp, "temp": temp,
                        "bg": bg, "use_bg": int(use_bg), "fARI": fari})

    results.sort(key=lambda r: (-r["fARI"]))
    log.info("=" * 78)
    log.info("TOP READOUT CONFIGS (by fARI)")
    log.info("=" * 78)
    log.info(f"  {'merge':>5} {'comp':>8} {'temp':>5} {'bg':>5} {'use_bg':>6} {'fARI':>8}")
    for r in results[:15]:
        log.info(f"  {r['merge']:>5} {r['competition']:>8} {r['temp']:>5.2f} {r['bg']:>5.2f} {r['use_bg']:>6} {r['fARI']:>8.4f}")

    # ---- Pareto: compute mBO/mIoU across the WHOLE fARI range, not just top.
    # Select configs spanning unique fARI values so we see the trade-off curve.
    def _compute_full(r):
        mbo_vals, miou_vals = [], []
        BS = 64
        for s in range(0, n, BS):
            e = min(s + BS, n)
            pred = build_pred_mask_readout(
                logits_stack[s:e], valid_stack[s:e],
                target_size=args.eval_size, n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size, merge=r["merge"], competition=r["competition"],
                temp=r["temp"], bg_threshold=r["bg"], use_bg_channel=bool(r["use_bg"]))
            for b in range(pred.shape[0]):
                mbo_vals.append(mbo_metric(gt_stack[s+b:s+b+1], pred[b:b+1]))
                miou_vals.append(miou_metric(gt_stack[s+b:s+b+1], pred[b:b+1]))
        r["mBO"] = float(np.nanmean([v for v in mbo_vals if not np.isnan(v)]))
        r["mIoU"] = float(np.nanmean([v for v in miou_vals if not np.isnan(v)]))
        return r

    if bool(getattr(args, "pareto", False)):
        # One representative config per unique (rounded) fARI value → spans the curve.
        seen = set()
        pareto_cfgs = []
        for r in sorted(results, key=lambda x: x["fARI"]):
            key = round(r["fARI"], 3)
            if key not in seen:
                seen.add(key)
                pareto_cfgs.append(r)
        log.info(f"\nPareto: computing mBO/mIoU for {len(pareto_cfgs)} configs spanning the fARI range ...")
        for r in tqdm(pareto_cfgs, desc="pareto"):
            _compute_full(r)
        log.info("\n" + "=" * 78)
        log.info("PARETO FRONT (fARI vs mBO vs mIoU)")
        log.info("=" * 78)
        log.info(f"  {'fARI':>7} {'mBO':>7} {'mIoU':>7}  | {'merge':>5} {'comp':>8} {'temp':>5} {'bg':>5} {'ubg':>4}")
        log.info("  " + "-" * 70)
        for r in sorted(pareto_cfgs, key=lambda x: x["fARI"]):
            log.info(f"  {r['fARI']:>7.4f} {r['mBO']:>7.4f} {r['mIoU']:>7.4f}  | "
                     f"{r['merge']:>5} {r['competition']:>8} {r['temp']:>5.2f} {r['bg']:>5.2f} {r['use_bg']:>4}")
        log.info("\n  CODA: fARI=0.475 mBO=0.363 mIoU=0.364")
        # Find configs that beat CODA on ALL three (if any)
        all3 = [r for r in pareto_cfgs if r["fARI"] > 0.475 and r["mBO"] > 0.363 and r["mIoU"] > 0.364]
        if all3:
            log.info(f"\n  ★ Configs beating CODA on ALL 3 metrics: {len(all3)}")
            for r in all3:
                log.info(f"    fARI={r['fARI']:.4f} mBO={r['mBO']:.4f} mIoU={r['mIoU']:.4f}")
        else:
            log.info("\n  ⚠️ NO single readout beats CODA on all 3 (fARI, mBO, mIoU). Trade-off confirmed.")
            # Best balanced: maximize min over normalized metrics vs CODA
            for r in pareto_cfgs:
                r["_bal"] = min(r["fARI"]/0.475, r["mBO"]/0.363, r["mIoU"]/0.364)
            bestbal = max(pareto_cfgs, key=lambda x: x["_bal"])
            log.info(f"  Best 'balanced' config (max of min-ratio-vs-CODA = {bestbal['_bal']:.3f}):")
            log.info(f"    fARI={bestbal['fARI']:.4f} mBO={bestbal['mBO']:.4f} mIoU={bestbal['mIoU']:.4f}  "
                     f"| {bestbal['merge']}/{bestbal['competition']}/T{bestbal['temp']}/bg{bestbal['bg']}/ubg{bestbal['use_bg']}")
    else:
        log.info("\nComputing mBO/mIoU for top configs ...")
        for r in results[: args.topk_full_metric]:
            _compute_full(r)
            log.info(f"  [{r['merge']}/{r['competition']}/T{r['temp']}/bg{r['bg']}/usebg{r['use_bg']}] "
                     f"fARI={r['fARI']:.4f} mBO={r['mBO']:.4f} mIoU={r['mIoU']:.4f}")

    # ---- save
    out_json = os.path.join(args.output_dir, "sweep_readout.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "n_samples": n,
                "gt_source": args.gt_source,
                "image_preprocess_mode": args.image_preprocess_mode,
                "coda_crop_size": int(args.coda_crop_size),
                "results": results,
            },
            f,
            indent=2,
        )
    log.info(f"\nSaved: {out_json}")
    best = results[0]
    log.info(f"\n★ BEST fARI = {best['fARI']:.4f}  with "
             f"merge={best['merge']} comp={best['competition']} temp={best['temp']} "
             f"bg={best['bg']} use_bg={best['use_bg']}")
    log.info(f"   (vs CODA fARI 0.475; current default readout fARI ~0.364)")


if __name__ == "__main__":
    main()
