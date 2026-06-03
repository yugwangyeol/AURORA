"""Step-0 readout diagnostics for PGOT.

This script runs one PGOT forward pass over the eval set, caches OVT/register
logits, then evaluates diagnostic readouts that isolate:

  A) thing-only object competition, no explicit background
  B) thing-only object competition with constant background threshold
  C) thing+stuff competition where stuff winners become background
  D) thing+stuff+register competition where register is background
  E) oracle foreground mask applied to A/B predictions

The goal is not to find a final readout, but to identify whether low scores
come from foreground object clustering, stuff/background competition, or
register background stealing foreground patches.
"""

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.constants import OVT_TOKEN, SCENE_END_TOKEN
from pgot.eval.pgot_inference import pgot_forward_eval
from pgot.eval.pgot_metrics import fari_metric, mbo_metric, miou_metric
from pgot.eval.run_eval import CocoInstanceMaskCache, load_gt_panoptic_mask, load_thing_categories
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.model.pgot_utils import build_pred_mask_competition_eval
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.diagnose_readout")


def _csv_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _load_model(args, device: str, dtype: torch.dtype):
    log.info(f"Loading model: {args.model_path}")
    config = AutoConfig.from_pretrained(args.model_path)
    model = PGOTQwen2ForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643

    import glob
    import safetensors.torch as safe_torch

    has_lora = False
    for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
        with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
            if any("lora_" in k for k in f.keys()):
                has_lora = True
                break
    if has_lora:
        from peft import LoraConfig, inject_adapter_in_model

        inject_adapter_in_model(
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
            model,
            adapter_name="default",
        )
        sd = {}
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info(f"LoRA re-loaded | missing={len(missing)} unexpected={len(unexpected)}")

    parsed_towers = getattr(config, "mm_vision_tower_aux_list", None) or json.loads(
        getattr(config, "vision_tower_aux_list", '["google/siglip2-so400m-patch14-224"]')
    )
    parsed_token_lens = getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]

    from types import SimpleNamespace

    vt_args = SimpleNamespace(
        vision_tower_aux_list=parsed_towers,
        vision_tower_aux_token_len_list=parsed_token_lens,
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        mm_projector_type="mlp2x_gelu",
        mm_use_im_start_end=True,
        mm_use_im_patch_token=False,
        unfreeze_mm_vision_tower=False,
        vision_hidden_size=1024,
        connector_only=True,
        pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None),
    )
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
    return model, tokenizer


def _object_logits(
    ovt_logits: torch.Tensor,
    n_ovt_per_object: int,
    patch_grid: int,
    target_size: int,
    merge: str,
    temp: float,
) -> torch.Tensor:
    B, M, P = ovt_logits.shape
    if P != patch_grid * patch_grid:
        raise ValueError(f"Expected {patch_grid * patch_grid} patches, got {P}")
    K = M // n_ovt_per_object
    logits = ovt_logits[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object, P).float()
    obj = logits.amax(dim=2) if merge == "max" else logits.mean(dim=2)
    obj = obj.reshape(B, K, patch_grid, patch_grid)
    obj = F.interpolate(obj, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return obj / max(float(temp), 1e-6)


def _object_valid(valid: torch.Tensor, n_ovt_per_object: int) -> torch.Tensor:
    B, M = valid.shape
    K = M // n_ovt_per_object
    return valid[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object).any(dim=2)


def _pred_object_competition(
    ovt_logits: torch.Tensor,
    valid: torch.Tensor,
    is_thing: torch.Tensor,
    *,
    target_size: int,
    n_ovt_per_object: int,
    patch_grid: int,
    merge: str,
    temp: float,
    include_stuff: bool,
    bg_threshold: float = None,
) -> torch.Tensor:
    """Argmax over object logits. Thing winners become 1..N, stuff winners become 0.

    If bg_threshold is not None, prepend a constant background channel before
    argmax. This isolates thresholded background without using register logits.
    """
    B, M, _ = ovt_logits.shape
    K = M // n_ovt_per_object
    logits = _object_logits(ovt_logits, n_ovt_per_object, patch_grid, target_size, merge, temp)
    obj_valid = _object_valid(valid, n_ovt_per_object)
    obj_thing = _object_valid(is_thing, n_ovt_per_object)
    if not include_stuff:
        obj_valid = obj_valid & obj_thing

    neg = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~obj_valid.view(B, K, 1, 1), neg)

    has_any = obj_valid.any(dim=1)
    if bg_threshold is not None:
        bg = torch.full((B, 1, target_size, target_size), float(bg_threshold), device=masked.device, dtype=masked.dtype)
        assign = torch.cat([bg, masked], dim=1).argmax(dim=1) - 1
    else:
        assign = masked.argmax(dim=1)
    assign[~has_any] = -1

    pred = torch.zeros((B, target_size, target_size), dtype=torch.int64, device=ovt_logits.device)
    for b in range(B):
        rank = 0
        for k in range(K):
            if not bool(obj_valid[b, k]):
                continue
            if bool(obj_thing[b, k]):
                rank += 1
                pred[b][assign[b] == k] = rank
            # stuff or explicit bg stays 0
    return pred


def _apply_oracle_fg(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    out = pred.clone()
    out[gt == 0] = 0
    return out


def _metrics(gt: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    fari_vals, mbo_vals, miou_vals = [], [], []
    for b in range(pred.shape[0]):
        gt_b = gt[b : b + 1]
        pr_b = pred[b : b + 1]
        fa = fari_metric(gt_b, pr_b)
        mb = mbo_metric(gt_b, pr_b)
        mi = miou_metric(gt_b, pr_b)
        if not np.isnan(fa):
            fari_vals.append(fa)
        if not np.isnan(mb):
            mbo_vals.append(mb)
        if not np.isnan(mi):
            miou_vals.append(mi)
    return {
        "fARI": float(np.nanmean(fari_vals)) if fari_vals else float("nan"),
        "mBO": float(np.nanmean(mbo_vals)) if mbo_vals else float("nan"),
        "mIoU": float(np.nanmean(miou_vals)) if miou_vals else float("nan"),
    }


def _accumulate(acc: Dict[str, List[float]], scores: Dict[str, float]) -> None:
    for k in ("fARI", "mBO", "mIoU"):
        if not np.isnan(scores[k]):
            acc.setdefault(k, []).append(scores[k])


def _finalize(acc: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.nanmean(v)) if v else float("nan") for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=500)
    p.add_argument("--grid_size", type=int, default=32)
    p.add_argument("--eval_size", type=int, default=256)
    p.add_argument("--max_caption_tokens", type=int, default=2048)
    p.add_argument("--n_ovt_per_object", type=int, default=2)
    p.add_argument("--max_objects", type=int, default=50)
    p.add_argument("--coco_mask_cache", default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--merge", choices=["mean", "max"], default="max")
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--bg_thresholds", default="0.05,0.10,0.15,0.20,0.30")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    bg_thresholds = _csv_floats(args.bg_thresholds)

    model, tokenizer = _load_model(args, device=device, dtype=dtype)

    coco_cache = CocoInstanceMaskCache(args.coco_mask_cache)
    args.eval_size = coco_cache.size
    thing_categories = load_thing_categories("/home/jovyan/data/coco/annotations/panoptic_val2017.json")
    log.info(f"Loaded {len(thing_categories)} COCO thing categories.")

    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=args.val_jsonl,
        tokenizer=tokenizer,
        image_processor=image_proc,
        target_image_processor=target_proc,
        grid_size=args.grid_size,
        max_caption_tokens=args.max_caption_tokens,
        n_ovt_per_object=args.n_ovt_per_object,
        max_objects=args.max_objects,
    )
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = torch.utils.data.Subset(dataset, list(range(args.max_samples)))
    samples_ref = dataset.dataset.samples if isinstance(dataset, torch.utils.data.Subset) else dataset.samples

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=PGOTDataCollator(pad_token_id=tokenizer.pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    log.info(f"Eval set: {len(dataset)} samples")

    acc: Dict[str, Dict[str, List[float]]] = {}
    gidx = 0
    for batch in tqdm(loader, desc="diagnose"):
        out = pgot_forward_eval(
            model,
            images=batch["images"],
            target_images=batch["target_images"],
            caption_input_ids=batch["caption_input_ids"],
            caption_attention_mask=batch["caption_attention_mask"],
            ovt_positions_in_caption=batch["ovt_positions_in_caption"],
            ovt_valid_mask=batch["ovt_valid_mask"],
        )
        ovt_logits = out["ovt_logits"]
        reg_logits = out["reg_logits"]
        valid = out["ovt_valid_mask"]
        B = ovt_logits.shape[0]

        gt_list = []
        is_thing = torch.zeros_like(valid, dtype=torch.bool)
        for b in range(B):
            samp = samples_ref[gidx]
            gidx += 1
            gt = coco_cache.get(int(samp["image_id"]))
            if gt is None:
                seg_ids = [int(s["segment_id"]) for s in samp["segments"]]
                gt = load_gt_panoptic_mask(samp["panoptic_mask_path"], seg_ids, args.eval_size)
            gt_list.append(gt)
            for k, seg in enumerate(samp["segments"]):
                s = k * args.n_ovt_per_object
                e = s + args.n_ovt_per_object
                if e <= is_thing.shape[1] and seg["category"] in thing_categories:
                    is_thing[b, s:e] = True

        gt = torch.stack(gt_list).to(device=ovt_logits.device)

        # A: thing-only, no explicit background.
        pred_a = _pred_object_competition(
            ovt_logits,
            valid,
            is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=args.merge,
            temp=args.temp,
            include_stuff=False,
            bg_threshold=None,
        )
        _accumulate(acc.setdefault("A_thing_only_no_bg", {}), _metrics(gt, pred_a))
        _accumulate(acc.setdefault("E_oracle_fg_on_A", {}), _metrics(gt, _apply_oracle_fg(pred_a, gt)))

        # B: thing-only + constant background channel.
        for bg in bg_thresholds:
            pred_b = _pred_object_competition(
                ovt_logits,
                valid,
                is_thing,
                target_size=args.eval_size,
                n_ovt_per_object=args.n_ovt_per_object,
                patch_grid=args.grid_size,
                merge=args.merge,
                temp=args.temp,
                include_stuff=False,
                bg_threshold=bg,
            )
            name = f"B_thing_only_const_bg_{bg:g}"
            _accumulate(acc.setdefault(name, {}), _metrics(gt, pred_b))
            _accumulate(acc.setdefault(f"E_oracle_fg_on_B_{bg:g}", {}), _metrics(gt, _apply_oracle_fg(pred_b, gt)))

        # C: thing+stuff object competition. Stuff winners are output as background.
        pred_c = _pred_object_competition(
            ovt_logits,
            valid,
            is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=args.merge,
            temp=args.temp,
            include_stuff=True,
            bg_threshold=None,
        )
        _accumulate(acc.setdefault("C_thing_plus_stuff_no_register", {}), _metrics(gt, pred_c))
        _accumulate(acc.setdefault("E_oracle_fg_on_C", {}), _metrics(gt, _apply_oracle_fg(pred_c, gt)))

        # D: current v5/v6-style register background competition.
        pred_d = build_pred_mask_competition_eval(
            ovt_logits=ovt_logits,
            reg_logits=reg_logits,
            ovt_valid_mask=valid,
            ovt_is_thing=is_thing,
            target_size=args.eval_size,
            n_ovt_per_object=args.n_ovt_per_object,
            patch_grid=args.grid_size,
            merge=args.merge,
        )
        _accumulate(acc.setdefault("D_thing_stuff_register_bg", {}), _metrics(gt, pred_d))
        _accumulate(acc.setdefault("E_oracle_fg_on_D", {}), _metrics(gt, _apply_oracle_fg(pred_d, gt)))

    results = []
    for name, vals in acc.items():
        row = {"variant": name, **_finalize(vals)}
        results.append(row)
    results.sort(key=lambda r: (-(r.get("fARI", float("nan")) if not np.isnan(r.get("fARI", float("nan"))) else -1)))

    log.info("=" * 86)
    log.info("STEP-0 READOUT DIAGNOSTICS")
    log.info("=" * 86)
    log.info(f"{'variant':<38} {'fARI':>8} {'mBO':>8} {'mIoU':>8}")
    for r in results:
        log.info(f"{r['variant']:<38} {r['fARI']:>8.4f} {r['mBO']:>8.4f} {r['mIoU']:>8.4f}")

    out_path = os.path.join(args.output_dir, "diagnose_readout.json")
    payload = {
        "n_samples": len(dataset),
        "gt_source": "coco_instance",
        "merge": args.merge,
        "temp": args.temp,
        "bg_thresholds": bg_thresholds,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
