"""Reconstruction oracle ladder for PGOT/Scale-RAE.

This script measures where reconstruction fidelity is lost:

  o0_decoder_gt
      GT SigLIP target features -> RAE pixel decoder. No DiT.

  o1_gt_projected
      GT SigLIP target features -> learned 1x1 projector -> frozen DiT -> decoder.

  o2_gt_mask_routed_projected
      GT object-mask-routed averaged SigLIP features -> learned 1x1 projector
      -> frozen DiT -> decoder.

  o3_pgot_condition
      Current PGOT condition_hidden/rae_hidden -> frozen DiT -> decoder.
      This is a reference baseline, not a strict oracle.
"""

import argparse
import csv
import json
import logging
import os
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.constants import NEW_SPECIAL_TOKENS, OVT_TOKEN, SCENE_END_TOKEN
from pgot.eval.pgot_inference import generate_siglip_latent, pgot_forward_eval
from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.model.pgot_qwen2 import PGOTQwen2ForCausalLM
from pgot.train.pgot_dataset import PGOTDataCollator, Pix2CapPGOTDataset
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.recon_oracles")


ORACLE_CHOICES = {
    "o0_decoder_gt",
    "o1_gt_projected",
    "o2_gt_mask_routed_projected",
    "o3_pgot_condition",
}


def parse_oracles(spec: str) -> List[str]:
    out = [x.strip() for x in str(spec).split(",") if x.strip()]
    bad = [x for x in out if x not in ORACLE_CHOICES]
    if bad:
        raise ValueError(f"Unknown oracle(s): {bad}. Choices: {sorted(ORACLE_CHOICES)}")
    return out


def patch_diffusion_steps(model, steps: int) -> None:
    if not steps or int(steps) == 50:
        return
    try:
        from scale_rae.model.diffusion_loss.diffusion import create_diffusion

        inf = model.diff_head.inference_flow
        size_ratio = float(getattr(inf, "size_ratio", 1.0))
        d_steps = int(getattr(inf, "diffusion_steps", 1000))
        model.diff_head.inference_flow = create_diffusion(
            str(int(steps)),
            noise_schedule="linear",
            use_kl=False,
            sigma_small=False,
            predict_xstart=False,
            learn_sigma=False,
            rescale_learned_sigmas=False,
            diffusion_steps=d_steps,
            input_base_dimension_ratio=size_ratio,
            diffusion_type="rf",
            use_loss_weighting=False,
        )
        log.info("Patched RF inference_flow to %d steps.", int(steps))
    except Exception as exc:
        log.warning("Could not patch diffusion steps: %s", exc)


def load_model_and_tokenizer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    log.info("Loading model from: %s", args.model_path)
    raw_config_path = os.path.join(args.model_path, "config.json")
    raw_name_or_path = args.model_path
    if os.path.exists(raw_config_path):
        with open(raw_config_path, "r") as f:
            raw_cfg = json.load(f)
        raw_name_or_path = str(raw_cfg.get("_name_or_path", args.model_path))

    config = AutoConfig.from_pretrained(args.model_path)
    index_path = os.path.join(args.model_path, "model.safetensors.index.json")
    has_lora_in_ckpt = False
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_json = json.load(f)
        has_lora_in_ckpt = any(
            "lora_" in k or ".base_layer." in k
            for k in index_json.get("weight_map", {}).keys()
        )

    model_init_path = raw_name_or_path if has_lora_in_ckpt else args.model_path
    if has_lora_in_ckpt:
        log.info("[LoRA] adapter checkpoint detected; bootstrap from %s", model_init_path)

    model = PGOTQwen2ForCausalLM.from_pretrained(
        model_init_path,
        config=config,
        torch_dtype=dtype,
        ignore_mismatched_sizes=True,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643
    if OVT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
        model.resize_token_embeddings(len(tokenizer))

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

    if has_lora_in_ckpt:
        import glob
        import safetensors.torch as safe_torch
        from peft import LoraConfig, inject_adapter_in_model

        lora_cfg = LoraConfig(
            r=int(getattr(config, "captionslot_lora_r", 16)),
            lora_alpha=int(getattr(config, "captionslot_lora_alpha", 32)),
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        inject_adapter_in_model(lora_cfg, model, adapter_name="default")
        sd = {}
        for shard in sorted(glob.glob(os.path.join(args.model_path, "*.safetensors"))):
            with safe_torch.safe_open(shard, framework="pt", device="cpu") as f:
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log.info("[LoRA] loaded checkpoint | missing=%d unexpected=%d", len(missing), len(unexpected))

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
    model.diff_head = model.diff_head.to(device)
    model.set_diff_fp32()
    patch_diffusion_steps(model, args.diffusion_inference_steps)
    log.info("Model loaded on %s.", device)
    return model, tokenizer, device, dtype


def build_loader(args, tokenizer, model, jsonl_path: str, *, shuffle: bool, max_samples: int | None):
    vt_list = model.get_vision_tower_aux_list()
    image_proc = vt_list[0].image_processor
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else image_proc
    dataset = Pix2CapPGOTDataset(
        jsonl_path=jsonl_path,
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
    if max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, list(range(min(int(max_samples), len(dataset)))))
    collator = PGOTDataCollator(pad_token_id=tokenizer.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def encode_gt_siglip(model, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    images = batch["images"].to(device)
    target_images = batch["target_images"].to(device)
    _, _, gt_siglip = model._encode_images_aurora(images, target_images=target_images)
    return gt_siglip.float()


def gt_mask_routed_features(
    gt_siglip: torch.Tensor,
    gt_masks_per_ovt: torch.Tensor,
    ovt_valid_mask: torch.Tensor,
    n_ovt_per_object: int,
) -> torch.Tensor:
    B, P, C = gt_siglip.shape
    side = int(round(P ** 0.5))
    masks = gt_masks_per_ovt.to(device=gt_siglip.device, dtype=torch.float32).clamp(0.0, 1.0)
    valid = ovt_valid_mask.to(device=gt_siglip.device, dtype=torch.bool)
    M = masks.shape[1]
    K = M // int(n_ovt_per_object)
    if K == 0:
        return gt_siglip

    masks = masks[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object, -1).amax(dim=2)
    valid_obj = valid[:, : K * n_ovt_per_object].reshape(B, K, n_ovt_per_object).any(dim=2)
    if masks.shape[-1] != P:
        src_side = int(round(masks.shape[-1] ** 0.5))
        masks = F.interpolate(
            masks.reshape(B * K, 1, src_side, src_side),
            size=(side, side),
            mode="bilinear",
            align_corners=False,
        ).reshape(B, K, P)
    masks = masks * valid_obj.unsqueeze(-1).float()

    denom = masks.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    obj_feat = torch.einsum("bkp,bpc->bkc", masks, gt_siglip) / denom

    union = masks.amax(dim=1).clamp(0.0, 1.0)
    bg_mask = (1.0 - union).clamp(0.0, 1.0)
    bg_denom = bg_mask.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    bg_feat = torch.einsum("bp,bpc->bc", bg_mask, gt_siglip) / bg_denom
    global_feat = gt_siglip.mean(dim=1)
    bg_feat = torch.where((bg_denom > 1e-5), bg_feat, global_feat)

    all_feat = torch.cat([obj_feat, bg_feat.unsqueeze(1)], dim=1)
    all_weight = torch.cat([masks, bg_mask.unsqueeze(1)], dim=1)
    weight_sum = all_weight.sum(dim=1, keepdim=True).clamp_min(1e-6)
    routed = torch.einsum("bkp,bkc->bpc", all_weight / weight_sum, all_feat)
    return routed.float()


def make_source_features(oracle: str, model, batch, device, args) -> torch.Tensor:
    gt_siglip = encode_gt_siglip(model, batch, device)
    if oracle == "o1_gt_projected":
        return gt_siglip
    if oracle == "o2_gt_mask_routed_projected":
        return gt_mask_routed_features(
            gt_siglip,
            batch["gt_masks_per_ovt"],
            batch["ovt_valid_mask"],
            args.n_ovt_per_object,
        )
    raise ValueError(f"No projected source for oracle={oracle}")


def projector_path(args, oracle: str) -> Path:
    return Path(args.output_dir) / oracle / "condition_projector.pt"


def load_or_fit_projector(args, oracle: str, model, train_loader, device) -> nn.Linear:
    z_dim = int(model.diff_head.z_channels)
    x_dim = int(model.diff_head.diffusion_channels)
    proj = nn.Linear(x_dim, z_dim).to(device=device, dtype=torch.float32)
    path = projector_path(args, oracle)
    if path.exists() and not args.refit_projector:
        state = torch.load(path, map_location=device)
        proj.load_state_dict(state["state_dict"])
        log.info("[%s] Loaded projector: %s", oracle, path)
        return proj.eval()

    path.parent.mkdir(parents=True, exist_ok=True)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    model.diff_head.eval()
    proj.train()
    opt = torch.optim.AdamW(proj.parameters(), lr=args.projector_lr, weight_decay=args.projector_weight_decay)
    losses: List[float] = []
    loader_iter = cycle(train_loader)
    pbar = tqdm(range(int(args.projector_steps)), desc=f"Fit {oracle}")
    for step in pbar:
        batch = next(loader_iter)
        src = make_source_features(oracle, model, batch, device, args)
        gt = encode_gt_siglip(model, batch, device)
        z = F.layer_norm(proj(src.float()), (z_dim,))
        loss = model.diff_head.training_loss(z=z.float(), x=gt.float()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % 10 == 0:
            pbar.set_postfix(loss=f"{losses[-1]:.4f}")
    torch.save(
        {
            "state_dict": proj.state_dict(),
            "oracle": oracle,
            "steps": int(args.projector_steps),
            "lr": float(args.projector_lr),
            "last_loss": losses[-1] if losses else None,
            "mean_last_50": sum(losses[-50:]) / max(len(losses[-50:]), 1),
        },
        path,
    )
    log.info("[%s] Saved projector: %s", oracle, path)
    return proj.eval()


@torch.no_grad()
def infer_from_z(model, z: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    model.diff_head = model.diff_head.to(z.device)
    model.set_diff_fp32()
    diff_head = model.diff_head
    P = int(diff_head.diffusion_tokens)
    C = int(diff_head.diffusion_channels)
    side = int(round(P ** 0.5))
    x_end = torch.randn((z.shape[0], C, side, side), device=z.device, dtype=torch.float32)
    return diff_head.infer(z=z.float(), x_end=x_end, guidance_level=guidance_scale)


@torch.no_grad()
def run_oracle_eval(args, oracle: str, model, val_loader, decoder, device, projector=None) -> Dict[str, float]:
    fid_acc = FIDAccumulator(device=device, feature=2048)
    metric_lists = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    n_samples = 0

    vt_list = model.get_vision_tower_aux_list()
    target_proc = vt_list[1].image_processor if len(vt_list) > 1 else vt_list[0].image_processor
    t_mean = torch.tensor(target_proc.image_mean)
    t_std = torch.tensor(target_proc.image_std)

    for batch in tqdm(val_loader, desc=f"Eval {oracle}"):
        gt_siglip = encode_gt_siglip(model, batch, device)
        if oracle == "o0_decoder_gt":
            generated = gt_siglip
        elif oracle in {"o1_gt_projected", "o2_gt_mask_routed_projected"}:
            src = make_source_features(oracle, model, batch, device, args)
            z_dim = int(model.diff_head.z_channels)
            z = F.layer_norm(projector(src.float()), (z_dim,))
            generated = infer_from_z(model, z, args.guidance_scale)
        elif oracle == "o3_pgot_condition":
            out = pgot_forward_eval(
                model,
                images=batch["images"],
                target_images=batch["target_images"],
                caption_input_ids=batch["caption_input_ids"],
                caption_attention_mask=batch["caption_attention_mask"],
                ovt_positions_in_caption=batch["ovt_positions_in_caption"],
                ovt_valid_mask=batch["ovt_valid_mask"],
            )
            generated = generate_siglip_latent(model, out["rae_hidden"], guidance_level=args.guidance_scale)
        else:
            raise ValueError(f"Unknown oracle: {oracle}")

        fake = decode_to_image(decoder, generated, device)
        real = denormalize_images(batch["target_images"].to(device).float(), t_mean, t_std)
        if real.shape[-2:] != fake.shape[-2:]:
            real = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
        fid_acc.add(real, fake)
        rec = compute_recon_metrics(real, fake)
        for key in metric_lists:
            metric_lists[key].extend([float(x) for x in rec[key].detach().cpu()])
        n_samples += int(real.shape[0])

    try:
        rfid = fid_acc.compute()
    except RuntimeError as exc:
        if n_samples < 2:
            log.warning("[%s] rFID unavailable with %d sample(s): %s", oracle, n_samples, exc)
            rfid = float("nan")
        else:
            raise

    summary = {
        "oracle": oracle,
        "num_samples": n_samples,
        "rFID": rfid,
        "guidance_scale": float(args.guidance_scale),
        "diffusion_inference_steps": int(args.diffusion_inference_steps),
    }
    for key, vals in metric_lists.items():
        summary[f"recon_{key}"] = sum(vals) / max(len(vals), 1)
    out_dir = Path(args.output_dir) / oracle
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def write_combined_summary(output_dir: str, summaries: Iterable[Dict[str, float]]) -> None:
    rows = list(summaries)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "summary.json", "w") as f:
        json.dump({"oracles": rows}, f, indent=2)
    fields = [
        "oracle",
        "num_samples",
        "rFID",
        "recon_psnr",
        "recon_ssim",
        "recon_mse",
        "recon_mae",
        "guidance_scale",
        "diffusion_inference_steps",
    ]
    with open(Path(output_dir) / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_jsonl", default="/home/jovyan/PGOT/data/pgot_train.jsonl")
    parser.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--oracles", default="o0_decoder_gt,o1_gt_projected,o2_gt_mask_routed_projected,o3_pgot_condition")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--max_caption_tokens", type=int, default=2048)
    parser.add_argument("--n_ovt_per_object", type=int, default=2)
    parser.add_argument("--max_objects", type=int, default=50)
    parser.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    parser.add_argument("--coda_crop_size", type=int, default=512)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--diffusion_inference_steps", type=int, default=25)
    parser.add_argument("--projector_steps", type=int, default=1000)
    parser.add_argument("--projector_lr", type=float, default=1e-4)
    parser.add_argument("--projector_weight_decay", type=float, default=0.0)
    parser.add_argument("--refit_projector", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    oracles = parse_oracles(args.oracles)
    model, tokenizer, device, dtype = load_model_and_tokenizer(args)
    train_loader = None
    if any(o in {"o1_gt_projected", "o2_gt_mask_routed_projected"} for o in oracles):
        train_loader = build_loader(
            args,
            tokenizer,
            model,
            args.train_jsonl,
            shuffle=True,
            max_samples=args.max_train_samples,
        )
        log.info("Projection train set ready.")
    val_loader = build_loader(
        args,
        tokenizer,
        model,
        args.val_jsonl,
        shuffle=False,
        max_samples=args.max_samples,
    )
    log.info("Eval set ready.")
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)

    summaries = []
    for oracle in oracles:
        projector = None
        if oracle in {"o1_gt_projected", "o2_gt_mask_routed_projected"}:
            projector = load_or_fit_projector(args, oracle, model, train_loader, device)
        summary = run_oracle_eval(args, oracle, model, val_loader, decoder, device, projector=projector)
        log.info("[%s] %s", oracle, json.dumps(summary, sort_keys=True))
        summaries.append(summary)
    write_combined_summary(args.output_dir, summaries)
    log.info("Wrote %s", Path(args.output_dir) / "summary.csv")


if __name__ == "__main__":
    main()
