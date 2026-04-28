#!/usr/bin/env python
"""CaptionSlot checkpoint eval without slot-attention / loss reporting.

Computes rFID / SSIM / PSNR / MSE only. Single-GPU shard — run two in parallel
via the accompanying launcher. Prints a running rFID every batch so you can
watch convergence mid-run.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import spacy
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
INFERENCE_ROOT = REPO_ROOT / "inference"
SCALE_RAE_ROOT = Path("/home/jovyan/Scale-RAE")
SCALE_RAE_INFERENCE_ROOT = SCALE_RAE_ROOT / "inference"
for _p in (
    str(REPO_ROOT),
    str(INFERENCE_ROOT),
    str(SCALE_RAE_ROOT),
    str(SCALE_RAE_INFERENCE_ROOT),
):
    if _p not in sys.path:
        sys.path.append(_p)

if "IPython" not in sys.modules:
    _stub = types.ModuleType("IPython")
    _stub.get_ipython = lambda: None
    _stub.version_info = (0, 0, 0, "")
    sys.modules["IPython"] = _stub

from build_stagea_captionslot_train_cache import extract_noun_info_from_token_ids_with_doc  # type: ignore
from cli import ensure_output_dir  # type: ignore
from eval_caption_to_image_rfid import (  # type: ignore
    InceptionFeatureExtractor,
    build_abs_diff_image,
    compute_basic_metrics,
    compute_fid,
    save_json,
    save_triptych,
    tensor_to_pil,
)
from scale_rae.train.captionslot_trainer import (  # type: ignore
    _register_captionslot_template_token_ids,
    _register_im_start_end_token_ids,
)
from scale_rae.utils import disable_torch_init  # type: ignore
from utils.load_model import load_scale_rae_model  # type: ignore


DEFAULT_MODEL_PATH = "/home/jovyan/AURORA/checkpoints/captionslot_firstslot_headprior_s1.0_stage1"
DEFAULT_IMAGE_DIR = "/home/jovyan/data/coco/val2017"
DEFAULT_CAPTIONS_JSONL = "/home/jovyan/Scale-RAE/outputs/caption_to_image_recon_concise/captions.jsonl"
DEFAULT_OUTPUT_DIR = "/home/jovyan/AURORA/outputs/captionslot_headprior_s1.0_stage1_noslot_eval"


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--action", choices=["eval", "merge"], default="eval")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    p.add_argument("--captions-jsonl", default=DEFAULT_CAPTIONS_JSONL)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--per-device-eval-batch-size", type=int, default=32)
    p.add_argument("--dataloader-num-workers", type=int, default=4)
    p.add_argument("--max-caption-tokens", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--guidance-level", type=float, default=1.0)
    p.add_argument("--diffusion-steps", type=int, default=20,
                   help="RF sampler steps (model default 50; 10-20 usually fine).")
    p.add_argument("--save-images", type=int, default=0,
                   help="If >0, save that many input|generated|diff triptychs this shard.")
    p.add_argument("--rfid-min-samples", type=int, default=32,
                   help="Skip running rFID print until at least this many samples accumulated.")
    p.add_argument("--spacy-model", default="en_core_web_sm")
    # merge-only:
    p.add_argument("--shard-dir", dest="shard_dirs", action="append", default=[])
    return p.parse_args()


_DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --------------------------------------------------------------------------- #
# data                                                                        #
# --------------------------------------------------------------------------- #

def normalize_caption(text: str) -> str:
    return " ".join(str(text).strip().split())


def load_caption_records(
    jsonl_path: str, image_dir: str, max_samples: Optional[int]
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            caption = normalize_caption(item.get("caption", ""))
            if not caption:
                continue
            img = item.get("image") or item.get("file_name")
            if img is None:
                raise KeyError(f"Missing image/file_name: {item}")
            img_path = img if os.path.isabs(img) else os.path.join(image_dir, img)
            records.append({
                "image_id": Path(img_path).stem,
                "image": img_path,
                "file_name": Path(img_path).name,
                "caption": caption,
            })
            if max_samples is not None and len(records) >= max_samples:
                break
    records.sort(key=lambda x: x["file_name"])
    return records


class CaptionSlotDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        tokenizer,
        image_processor,
        max_slots: int,
        max_caption_tokens: int,
        spacy_model: str,
    ):
        self.image_processor = image_processor
        self.max_slots = max_slots
        nlp = spacy.load(spacy_model)
        docs = list(nlp.pipe([s["caption"] for s in samples], batch_size=128))
        self.entries: List[Dict[str, Any]] = []
        for s, doc in zip(samples, docs):
            token_ids = tokenizer.encode(s["caption"], add_special_tokens=False)[:max_caption_tokens]
            info = extract_noun_info_from_token_ids_with_doc(
                token_ids=token_ids, tokenizer=tokenizer, doc=doc,
                max_caption_tokens=max_caption_tokens,
            )
            noun_chunks = (info.get("noun_chunks") or [])[:max_slots]
            if not token_ids or not noun_chunks:
                continue
            self.entries.append({**s, "token_ids": token_ids, "noun_chunks": noun_chunks})
        if not self.entries:
            raise ValueError("No valid eval entries after noun-chunk extraction.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        image = Image.open(e["image"]).convert("RGB")
        image_t = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        token_ids = torch.tensor(e["token_ids"], dtype=torch.long)
        spans = torch.full((self.max_slots, 2), -1, dtype=torch.long)
        n = min(len(e["noun_chunks"]), self.max_slots)
        for k, c in enumerate(e["noun_chunks"][: self.max_slots]):
            spans[k, 0] = int(c["token_start"])
            spans[k, 1] = int(c["token_end"])
        return {
            "image": image_t,
            "caption_input_ids": token_ids,
            "noun_chunk_spans": spans,
            "n_slots": torch.tensor(n, dtype=torch.long),
            "head_prior_maps": torch.zeros((self.max_slots, 256), dtype=torch.float32),
            "head_prior_valid_mask": torch.zeros(self.max_slots, dtype=torch.bool),
            "image_id": e["image_id"],
            "image_path": e["image"],
            "caption": e["caption"],
            "file_name": e["file_name"],
        }


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        B = len(batch)
        max_len = max(b["caption_input_ids"].shape[0] for b in batch)
        ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        mask = torch.zeros((B, max_len), dtype=torch.bool)
        for i, b in enumerate(batch):
            L = b["caption_input_ids"].shape[0]
            ids[i, :L] = b["caption_input_ids"]
            mask[i, :L] = True
        return {
            "images": torch.stack([b["image"] for b in batch]),
            "caption_input_ids": ids,
            "caption_attention_mask": mask,
            "noun_chunk_spans": torch.stack([b["noun_chunk_spans"] for b in batch]),
            "n_slots": torch.stack([b["n_slots"] for b in batch]),
            "head_prior_maps": torch.stack([b["head_prior_maps"] for b in batch]),
            "head_prior_valid_mask": torch.stack([b["head_prior_valid_mask"] for b in batch]),
            "image_ids": [b["image_id"] for b in batch],
            "image_path": [b["image_path"] for b in batch],
            "caption": [b["caption"] for b in batch],
            "file_name": [b["file_name"] for b in batch],
        }


# --------------------------------------------------------------------------- #
# model helpers                                                               #
# --------------------------------------------------------------------------- #

def load_eval_decoder(model):
    from huggingface_hub import hf_hub_download
    from scale_rae.model.multimodal_decoder import MultimodalDecoder  # type: ignore
    encoder_path = getattr(model.config, "mm_vision_tower_aux_list",
                           ["google/siglip2-so400m-patch14-224"])[0]
    encoder_path = encoder_path.split("-interp")[0]
    num_patches = int(getattr(model, "num_image_tokens", 256))
    config_path = hf_hub_download(repo_id="nyu-visionx/siglip2_decoder", filename="config.json")
    ckpt_path = hf_hub_download(repo_id="nyu-visionx/siglip2_decoder", filename="model.pt")
    return MultimodalDecoder(
        pretrained_encoder_path=encoder_path,
        general_decoder_config=config_path,
        num_patches=num_patches,
        drop_cls_token=True,
        decoder_path=ckpt_path,
    )


def patch_diffusion_steps(model, n_steps: int) -> None:
    try:
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion  # type: ignore
    except ImportError:
        print("[WARN] diffusion helpers not importable; keeping default step count.")
        return
    diff_head = getattr(model, "diff_head", None)
    if diff_head is None:
        return
    inf_flow = getattr(diff_head, "inference_flow", None)
    if inf_flow is None:
        return
    cur = len(getattr(inf_flow, "used_timesteps", [None] * 50))
    if cur == n_steps:
        return
    size_ratio = float(getattr(inf_flow, "size_ratio", 1.0))
    diffusion_steps = int(getattr(inf_flow, "diffusion_steps", 1000))
    diff_head.inference_flow = create_diffusion(
        str(n_steps), noise_schedule="linear", use_kl=False, sigma_small=False,
        predict_xstart=False, learn_sigma=False, rescale_learned_sigmas=False,
        diffusion_steps=diffusion_steps, input_base_dimension_ratio=size_ratio,
        diffusion_type="rf", use_loss_weighting=False, use_schedule_shift=True,
    )
    print(f"[INFO] diffusion inference_flow: {cur} -> {n_steps} steps")


def force_cast_model(model, dtype: torch.dtype) -> None:
    """`respect_torch_dtype=True` in load_pretrained_model silently fails on this
    checkpoint (all sub-modules come back as fp32 regardless of the request).
    Force the cast ourselves, visiting main params, buffers, and vision towers."""
    if dtype == torch.float32:
        return
    cast_params = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point() and p.dtype != dtype:
                p.data = p.data.to(dtype=dtype)
                cast_params += 1
        for b_name, b in model.named_buffers():
            if b.is_floating_point() and b.dtype != dtype:
                if "normalize_data" in b_name or "bn_stats" in b_name:
                    continue  # keep diffusion norm stats in fp32
                b.data = b.data.to(dtype=dtype)
        vt_cast = 0
        for vt in model.get_vision_tower_aux_list():
            for p in vt.parameters():
                if p.is_floating_point() and p.dtype != dtype:
                    p.data = p.data.to(dtype=dtype)
                    vt_cast += 1
            for _, b in vt.named_buffers():
                if b.is_floating_point() and b.dtype != dtype:
                    b.data = b.data.to(dtype=dtype)
    print(f"[DTYPE-cast] {cast_params} main + {vt_cast} vision-tower params -> {dtype}",
          flush=True)


def dtype_snapshot(model, tag: str) -> None:
    def _fp(mod):
        try:
            return next(mod.parameters()).dtype
        except Exception:
            return None
    print(
        f"[DTYPE-{tag}] llm={_fp(model.model)} "
        f"lm_head={_fp(getattr(model, 'lm_head', None))} "
        f"diff_head={_fp(getattr(model, 'diff_head', None))} "
        f"vision_tower={_fp(model.get_vision_tower_aux_list()[0])}",
        flush=True,
    )


def get_image_stats(image_processor):
    mean = torch.tensor(getattr(image_processor, "image_mean", [0.5, 0.5, 0.5]),
                        dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(getattr(image_processor, "image_std", [0.5, 0.5, 0.5]),
                       dtype=torch.float32).view(3, 1, 1)
    return mean, std


def decode_to_pixels(decoder, generated: torch.Tensor, device: torch.device) -> torch.Tensor:
    dec_dtype = next(decoder.parameters()).dtype
    generated = generated.to(device=device, dtype=dec_dtype)
    if hasattr(decoder, "image_mean") and hasattr(decoder, "image_std"):
        decoder.image_mean = decoder.image_mean.to(device=device, dtype=dec_dtype)
        decoder.image_std = decoder.image_std.to(device=device, dtype=dec_dtype)
    cls = torch.zeros((generated.shape[0], 1, generated.shape[-1]),
                      device=device, dtype=dec_dtype)
    feats = torch.cat([cls, generated], dim=1)
    recon = decoder(feats)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return recon.detach().float().cpu()


# --------------------------------------------------------------------------- #
# eval shard                                                                  #
# --------------------------------------------------------------------------- #

def run_eval(args: argparse.Namespace) -> None:
    ensure_output_dir(args.output_dir)
    disable_torch_init()

    all_samples = load_caption_records(args.captions_jsonl, args.image_dir, args.max_samples)
    if not all_samples:
        raise SystemExit("No evaluation samples found.")
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Invalid --num-shards / --shard-index.")
    shard_samples = all_samples[args.shard_index :: args.num_shards]
    if not shard_samples:
        raise SystemExit("This shard received zero samples.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = _DTYPE_MAP[args.dtype]

    tokenizer, model, image_processors, _ = load_scale_rae_model(
        model_path=args.model_path, device=str(device), dtype=dtype,
    )
    image_processor = image_processors[0]
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    _register_im_start_end_token_ids(model, tokenizer)
    if not hasattr(model, "captionslot_system_prefix_ids"):
        _register_captionslot_template_token_ids(model, tokenizer)
    model = model.to(device)

    dtype_snapshot(model, "pre-cast")
    force_cast_model(model, dtype)
    dtype_snapshot(model, "post-cast")

    model.eval()
    patch_diffusion_steps(model, args.diffusion_steps)

    max_slots = int(getattr(model.config, "captionslot_max_slots", 10))
    dataset = CaptionSlotDataset(
        samples=shard_samples,
        tokenizer=tokenizer,
        image_processor=image_processor,
        max_slots=max_slots,
        max_caption_tokens=args.max_caption_tokens,
        spacy_model=args.spacy_model,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        collate_fn=Collator(tokenizer.pad_token_id),
        drop_last=False,
    )
    decoder = load_eval_decoder(model).to(device)
    inception = InceptionFeatureExtractor().to(device)
    img_mean, img_std = get_image_stats(image_processor)

    captions_path = os.path.join(args.output_dir, "captions.jsonl")
    per_image_path = os.path.join(args.output_dir, "per_image.jsonl")
    samples_dir = os.path.join(args.output_dir, "samples")
    if args.save_images > 0:
        os.makedirs(samples_dir, exist_ok=True)

    real_feats: List[np.ndarray] = []
    fake_feats: List[np.ndarray] = []
    metrics_list: List[Dict[str, float]] = []
    generated_count = failed_count = saved_count = 0
    total_batches = len(loader)

    with (
        open(captions_path, "w", encoding="utf-8") as cap_f,
        open(per_image_path, "w", encoding="utf-8") as img_f,
        torch.inference_mode(),
    ):
        pbar = tqdm(total=len(dataset), desc=f"shard {args.shard_index}/{args.num_shards}",
                    unit="img", dynamic_ncols=True)
        for batch_idx, batch in enumerate(loader, start=1):
            imgs = batch["images"].to(device, non_blocking=True)
            out = model.generate_captionslot(
                images=imgs,
                caption_input_ids=batch["caption_input_ids"].to(device, non_blocking=True),
                caption_attention_mask=batch["caption_attention_mask"].to(device, non_blocking=True),
                noun_chunk_spans=batch["noun_chunk_spans"].to(device, non_blocking=True),
                n_slots=batch["n_slots"].to(device, non_blocking=True),
                head_prior_maps=batch["head_prior_maps"].to(device, non_blocking=True),
                head_prior_valid_mask=batch["head_prior_valid_mask"].to(device, non_blocking=True),
                guidance_level=args.guidance_level,
                return_generated=True,
            )
            recon = decode_to_pixels(decoder, out["generated"], device)
            source = (imgs.detach().float().cpu() * img_std + img_mean).clamp(0.0, 1.0)

            try:
                real_f = inception(source.to(device)).detach().cpu().numpy()
                fake_f = inception(recon.to(device)).detach().cpu().numpy()
            except Exception:
                real_f = fake_f = None

            B = imgs.shape[0]
            for i in range(B):
                image_path = batch["image_path"][i]
                caption = batch["caption"][i]
                record: Dict[str, Any] = {
                    "image": os.path.abspath(image_path),
                    "file_name": batch["file_name"][i],
                    "image_id": batch["image_ids"][i],
                    "caption": caption,
                    "generated_image": False,
                }
                try:
                    m = compute_basic_metrics(source[i : i + 1], recon[i : i + 1])
                    metrics_list.append(m)
                    record["generated_image"] = True
                    record["metrics"] = m
                    if real_f is not None:
                        real_feats.append(real_f[i : i + 1])
                        fake_feats.append(fake_f[i : i + 1])
                    generated_count += 1
                    if saved_count < args.save_images:
                        stem = Path(image_path).stem
                        sdir = os.path.join(samples_dir, stem)
                        os.makedirs(sdir, exist_ok=True)
                        inp_img = tensor_to_pil(source[i : i + 1])
                        rec_img = tensor_to_pil(recon[i : i + 1])
                        diff_img = build_abs_diff_image(source[i : i + 1], recon[i : i + 1])
                        inp_img.save(os.path.join(sdir, "input_processed.png"))
                        rec_img.save(os.path.join(sdir, "generated.png"))
                        diff_img.save(os.path.join(sdir, "abs_diff.png"))
                        save_triptych(inp_img, rec_img, diff_img,
                                      os.path.join(sdir, "comparison_triptych.png"))
                        saved_count += 1
                except Exception as exc:
                    failed_count += 1
                    record["error"] = repr(exc)

                cap_f.write(json.dumps(
                    {"image": os.path.abspath(image_path),
                     "file_name": batch["file_name"][i],
                     "caption": caption},
                    ensure_ascii=False) + "\n")
                img_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            cap_f.flush()
            img_f.flush()

            # --- running aggregate + per-batch rFID print ---------------------
            running_psnr = float(np.mean([m["psnr"] for m in metrics_list])) if metrics_list else float("nan")
            running_ssim = float(np.mean([m["ssim"] for m in metrics_list])) if metrics_list else float("nan")
            running_mse = float(np.mean([m["mse"] for m in metrics_list])) if metrics_list else float("nan")
            rfid_str = "n/a"
            if len(real_feats) >= args.rfid_min_samples:
                r_arr = np.concatenate(real_feats)
                f_arr = np.concatenate(fake_feats)
                try:
                    rfid_val = compute_fid(r_arr, f_arr)
                    rfid_str = f"{rfid_val:.2f}"
                except Exception as exc:
                    rfid_str = f"err({type(exc).__name__})"

            tqdm.write(
                f"[shard {args.shard_index} | batch {batch_idx}/{total_batches} "
                f"| n={generated_count}] "
                f"PSNR={running_psnr:.3f}  SSIM={running_ssim:.4f}  "
                f"MSE={running_mse:.5f}  rFID={rfid_str}",
                file=sys.stdout,
            )
            pbar.update(B)
            pbar.set_postfix(gen=generated_count, fail=failed_count, rfid=rfid_str)
        pbar.close()

    real_arr = np.concatenate(real_feats) if real_feats else np.empty((0, 2048), np.float32)
    fake_arr = np.concatenate(fake_feats) if fake_feats else np.empty((0, 2048), np.float32)
    np.savez_compressed(os.path.join(args.output_dir, "fid_features.npz"),
                        real=real_arr, fake=fake_arr)

    summary: Dict[str, Any] = {
        "model_path": os.path.abspath(args.model_path),
        "image_dir": os.path.abspath(args.image_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "captions_jsonl": os.path.abspath(args.captions_jsonl),
        "dtype": args.dtype,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "diffusion_steps": args.diffusion_steps,
        "guidance_level": float(args.guidance_level),
        "max_slots": max_slots,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "sample_count": len(dataset),
        "global_sample_count": len(all_samples),
        "generated_count": generated_count,
        "failed_count": failed_count,
        "saved_images": saved_count,
    }
    recon: Dict[str, Any] = {"requested_samples": len(dataset)}
    if metrics_list:
        recon["PSNR"] = float(np.mean([m["psnr"] for m in metrics_list]))
        recon["SSIM"] = float(np.mean([m["ssim"] for m in metrics_list]))
        recon["MSE"] = float(np.mean([m["mse"] for m in metrics_list]))
        recon["MAE"] = float(np.mean([m["mae"] for m in metrics_list]))
    if args.num_shards == 1 and real_arr.shape[0] >= 2:
        recon["rFID"] = float(compute_fid(real_arr, fake_arr))
    summary["reconstruction_metrics"] = recon

    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# merge                                                                       #
# --------------------------------------------------------------------------- #

def run_merge(args: argparse.Namespace) -> None:
    if not args.shard_dirs:
        raise SystemExit("--action merge requires one or more --shard-dir arguments.")
    ensure_output_dir(args.output_dir)

    real_all: List[np.ndarray] = []
    fake_all: List[np.ndarray] = []
    metrics_all: List[Dict[str, float]] = []
    per_image_merged: List[Dict[str, Any]] = []
    captions_merged: List[Dict[str, Any]] = []
    gen_total = fail_total = sample_total = saved_total = 0
    first_summary: Dict[str, Any] = {}
    merged_samples_dir = os.path.join(args.output_dir, "samples")

    for shard_dir in args.shard_dirs:
        sd = os.path.abspath(shard_dir)
        summary_path = os.path.join(sd, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        if not first_summary:
            first_summary = s
        gen_total += int(s.get("generated_count", 0))
        fail_total += int(s.get("failed_count", 0))
        sample_total += int(s.get("sample_count", 0))
        saved_total += int(s.get("saved_images", 0))

        shard_samples = os.path.join(sd, "samples")
        if os.path.isdir(shard_samples):
            os.makedirs(merged_samples_dir, exist_ok=True)
            for stem in sorted(os.listdir(shard_samples)):
                src = os.path.join(shard_samples, stem)
                dst = os.path.join(merged_samples_dir, stem)
                if os.path.isdir(src) and not os.path.exists(dst):
                    shutil.copytree(src, dst)

        per_image_path = os.path.join(sd, "per_image.jsonl")
        if os.path.isfile(per_image_path):
            with open(per_image_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    per_image_merged.append(rec)
                    m = rec.get("metrics")
                    if isinstance(m, dict):
                        metrics_all.append(m)

        captions_path = os.path.join(sd, "captions.jsonl")
        if os.path.isfile(captions_path):
            with open(captions_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        captions_merged.append(json.loads(line))

        feats_path = os.path.join(sd, "fid_features.npz")
        if os.path.isfile(feats_path):
            with np.load(feats_path) as payload:
                if payload["real"].size:
                    real_all.append(payload["real"])
                    fake_all.append(payload["fake"])

    per_image_merged.sort(key=lambda r: r.get("image", ""))
    captions_merged.sort(key=lambda r: r.get("image", ""))
    with open(os.path.join(args.output_dir, "per_image.jsonl"), "w", encoding="utf-8") as f:
        for r in per_image_merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.output_dir, "captions.jsonl"), "w", encoding="utf-8") as f:
        for r in captions_merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    real_arr = np.concatenate(real_all) if real_all else np.empty((0, 2048), np.float32)
    fake_arr = np.concatenate(fake_all) if fake_all else np.empty((0, 2048), np.float32)
    np.savez_compressed(os.path.join(args.output_dir, "fid_features.npz"),
                        real=real_arr, fake=fake_arr)

    recon: Dict[str, Any] = {
        "requested_samples": sample_total,
        "generated_count": gen_total,
        "failed_count": fail_total,
        "success_rate": (gen_total / sample_total) if sample_total else 0.0,
    }
    if metrics_all:
        recon["PSNR"] = float(np.mean([m["psnr"] for m in metrics_all]))
        recon["SSIM"] = float(np.mean([m["ssim"] for m in metrics_all]))
        recon["MSE"] = float(np.mean([m["mse"] for m in metrics_all]))
        recon["MAE"] = float(np.mean([m["mae"] for m in metrics_all]))
    recon["rFID"] = float(compute_fid(real_arr, fake_arr)) if real_arr.shape[0] >= 2 else None

    merged: Dict[str, Any] = {
        "model_path": first_summary.get("model_path"),
        "image_dir": first_summary.get("image_dir"),
        "output_dir": os.path.abspath(args.output_dir),
        "captions_jsonl": first_summary.get("captions_jsonl"),
        "dtype": first_summary.get("dtype"),
        "per_device_eval_batch_size": first_summary.get("per_device_eval_batch_size"),
        "diffusion_steps": first_summary.get("diffusion_steps"),
        "guidance_level": first_summary.get("guidance_level"),
        "max_slots": first_summary.get("max_slots"),
        "num_shards": len(args.shard_dirs),
        "shard_dirs": [os.path.abspath(d) for d in args.shard_dirs],
        "sample_count": sample_total,
        "generated_count": gen_total,
        "failed_count": fail_total,
        "saved_images": saved_total,
        "reconstruction_metrics": recon,
    }
    save_json(os.path.join(args.output_dir, "summary.json"), merged)
    print(json.dumps(merged, indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------- #
# entry                                                                       #
# --------------------------------------------------------------------------- #

def main() -> None:
    args = parse_args()
    if args.action == "merge":
        run_merge(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
