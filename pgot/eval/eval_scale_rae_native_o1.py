"""Native Scale-RAE O1 reconstruction oracle.

This measures the "true O1" upper bound:

    image -> original Scale-RAE query path -> native rae condition -> DiT -> decoder

Unlike the earlier projected O1/O2 probes, this script does not train or use a
condition projector. It builds the same query-mode interface used by the base
Scale-RAE model: context image features are inserted into the prompt, then the
native latent query block is appended after the image-start token, and the final
query hidden states are sampled by the frozen diffusion head.
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, "/home/jovyan/PGOT")

from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics
from pgot.eval.run_eval import decode_to_image, denormalize_images, load_rae_decoder
from pgot.train.pgot_dataset import _coda_center_crop_image
from scale_rae import ScaleRAEQwenForCausalLM
from scale_rae.constants import DEFAULT_IM_START_TOKEN
from transformers import AutoConfig, AutoTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s :: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pgot.scale_rae_o1")


class JsonlImageDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_processor,
        *,
        max_samples: int | None = None,
        image_preprocess_mode: str = "coda_center_crop",
        coda_crop_size: int = 512,
    ):
        self.image_processor = image_processor
        self.image_preprocess_mode = str(image_preprocess_mode)
        self.coda_crop_size = int(coda_crop_size)
        if self.image_preprocess_mode not in {"default", "coda_center_crop"}:
            raise ValueError(f"Unknown image_preprocess_mode={self.image_preprocess_mode}")
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f]
        if max_samples is not None:
            rows = rows[: int(max_samples)]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_preprocess_mode == "coda_center_crop":
            image = _coda_center_crop_image(image, self.coda_crop_size)
        tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return {
            "images": tensor,
            "target_images": tensor,
            "image_id": int(row.get("image_id", idx)),
        }


@dataclass
class ImageCollator:
    def __call__(self, instances: Sequence[Dict]) -> Dict:
        return {
            "images": torch.stack([x["images"] for x in instances]),
            "target_images": torch.stack([x["target_images"] for x in instances]),
            "image_ids": [x["image_id"] for x in instances],
        }


def patch_diffusion_steps(model, steps: int) -> None:
    if not steps or int(steps) == 50:
        return
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


def _model_args_from_config(config, args):
    from types import SimpleNamespace

    towers = getattr(config, "mm_vision_tower_aux_list", None) or ["google/siglip2-so400m-patch14-224"]
    token_lens = getattr(config, "mm_vision_tower_aux_token_len_list", None) or [256]
    return SimpleNamespace(
        vision_tower_aux_list=list(towers),
        vision_tower_aux_token_len_list=list(token_lens),
        mm_vision_select_layer=getattr(config, "mm_vision_select_layer", -1),
        mm_vision_select_feature=getattr(config, "mm_vision_select_feature", "patch"),
        mm_projector_type=getattr(config, "mm_projector_type", "mlp2x_gelu"),
        mm_use_im_start_end=True,
        mm_use_im_patch_token=False,
        tune_mm_mlp_adapter=False,
        tune_adapter_and_vision_head=False,
        unfreeze_mm_vision_tower=False,
        vision_hidden_size=getattr(config, "vision_hidden_size", 1024),
        connector_only=getattr(config, "connector_only", True),
        pretrain_mm_mlp_adapter=None,
        pretrain_adapter_and_vision_head=None,
        diffusion_norm_stats_path=getattr(config, "diffusion_norm_stats_path", None),
    )


def load_scale_rae(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    log.info("Loading native Scale-RAE from: %s", args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = ScaleRAEQwenForCausalLM.from_pretrained(
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

    model_args = _model_args_from_config(config, args)
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
    model.load_vision_head(model_args=model_args)
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    im_start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    if im_start_id is None or im_start_id == tokenizer.unk_token_id:
        raise RuntimeError(f"{DEFAULT_IM_START_TOKEN} is missing from tokenizer after initialization.")
    model.im_start_id = im_start_id
    model.config.im_start_id = im_start_id

    for vt in model.get_vision_tower_aux_list():
        vt.to(dtype=dtype, device=device)
    model.to(device=device, dtype=dtype)
    model.diff_head = model.diff_head.to(device)
    model.set_diff_fp32()
    patch_diffusion_steps(model, args.diffusion_inference_steps)
    model.eval()
    log.info(
        "Loaded native model | dtype=%s | vision_loss=%s | vision_loss_mode=%s | image_tokens=%s | diff_tokens=%s",
        dtype,
        getattr(model, "vision_loss", None),
        getattr(model, "vision_loss_mode", None),
        getattr(model, "num_image_tokens", None),
        getattr(model, "diffusion_target_token_len", None),
    )
    return model, tokenizer, device, dtype


def build_loader(args, model) -> DataLoader:
    image_processor = model.get_vision_tower_aux_list()[0].image_processor
    dataset = JsonlImageDataset(
        args.val_jsonl,
        image_processor,
        max_samples=args.max_samples,
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ImageCollator(),
        num_workers=args.num_workers,
        pin_memory=True,
    )


def _embed_ids(model, ids: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
    embeds = model.get_model().embed_tokens(ids.to(device))
    embeds = embeds.unsqueeze(0).expand(batch_size, -1, -1).to(device=device, dtype=dtype)
    return embeds


@torch.no_grad()
def native_o1_generate(model, tokenizer, images: torch.Tensor, guidance_scale: float) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    images = images.to(device=device, dtype=model_dtype)
    B = int(images.shape[0])

    vt = model.get_vision_tower_aux_list()[0]
    vt_dtype = next(vt.parameters()).dtype
    img_features = vt(images.to(dtype=vt_dtype))
    proj_dtype = next(model.get_model().mm_projector.parameters()).dtype
    img_features = model.get_model().mm_projector(img_features.to(dtype=proj_dtype)).to(dtype=model_dtype)

    prefix = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    suffix = tokenizer.encode("\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    start_id = tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
    prefix_embeds = _embed_ids(model, torch.tensor(prefix, dtype=torch.long), B, device, model_dtype)
    suffix_embeds = _embed_ids(model, torch.tensor(suffix, dtype=torch.long), B, device, model_dtype)
    start_embed = _embed_ids(model, torch.tensor([start_id], dtype=torch.long), B, device, model_dtype)
    latent_queries = model.get_model().latent_queries.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=model_dtype)

    inputs_embeds = torch.cat(
        [prefix_embeds, img_features, suffix_embeds, start_embed, latent_queries],
        dim=1,
    )
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=device, dtype=torch.bool)
    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    rae_hidden = outputs.last_hidden_state[:, -latent_queries.shape[1] :, :]
    z = rae_hidden
    if getattr(model, "use_diff_head_projector", False):
        proj_dtype = next(model.diff_head_projector.parameters()).dtype
        z = model.diff_head_projector(z.to(dtype=proj_dtype))

    model.diff_head = model.diff_head.to(device)
    model.set_diff_fp32()
    generated = model.diff_head.infer(z.float(), guidance_level=guidance_scale)
    return {
        "generated": generated,
        "rae_hidden": rae_hidden.detach(),
        "condition": z.detach(),
    }


@torch.no_grad()
def run_eval(args, model, tokenizer, loader, decoder, device) -> Dict[str, float]:
    fid_acc = FIDAccumulator(device=device, feature=2048)
    metric_lists = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    n_samples = 0
    processor = model.get_vision_tower_aux_list()[0].image_processor
    mean = torch.tensor(processor.image_mean)
    std = torch.tensor(processor.image_std)

    cond_mean, cond_std = [], []
    hidden_mean, hidden_std = [], []
    saved_rows = []
    max_save = int(getattr(args, "save_images", 0) or 0)

    for batch in tqdm(loader, desc="Eval native_o1"):
        out = native_o1_generate(model, tokenizer, batch["images"], args.guidance_scale)
        generated = out["generated"]
        fake = decode_to_image(decoder, generated, device)
        real = denormalize_images(batch["target_images"].to(device).float(), mean, std)
        if real.shape[-2:] != fake.shape[-2:]:
            real = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
        if max_save > 0 and len(saved_rows) < max_save:
            room = max_save - len(saved_rows)
            take = min(room, int(real.shape[0]))
            for i in range(take):
                saved_rows.append(torch.stack([real[i].cpu(), fake[i].cpu()], dim=0))
        fid_acc.add(real, fake)
        rec = compute_recon_metrics(real, fake)
        for key in metric_lists:
            metric_lists[key].extend([float(x) for x in rec[key].detach().cpu()])
        n_samples += int(real.shape[0])

        cond = out["condition"].float()
        hid = out["rae_hidden"].float()
        cond_mean.append(float(cond.mean().detach().cpu()))
        cond_std.append(float(cond.std().detach().cpu()))
        hidden_mean.append(float(hid.mean().detach().cpu()))
        hidden_std.append(float(hid.std().detach().cpu()))

    try:
        rfid = fid_acc.compute()
    except RuntimeError as exc:
        if n_samples < 2:
            log.warning("rFID unavailable with %d sample(s): %s", n_samples, exc)
            rfid = float("nan")
        else:
            raise

    summary = {
        "oracle": "o1_native_scale_rae",
        "model_path": args.model_path,
        "num_samples": n_samples,
        "rFID": rfid,
        "guidance_scale": float(args.guidance_scale),
        "diffusion_inference_steps": int(args.diffusion_inference_steps),
        "dtype": args.dtype,
        "condition_mean": sum(cond_mean) / max(len(cond_mean), 1),
        "condition_std": sum(cond_std) / max(len(cond_std), 1),
        "rae_hidden_mean": sum(hidden_mean) / max(len(hidden_mean), 1),
        "rae_hidden_std": sum(hidden_std) / max(len(hidden_std), 1),
    }
    for key, vals in metric_lists.items():
        summary[f"recon_{key}"] = sum(vals) / max(len(vals), 1)
    if saved_rows:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        rows = torch.cat(saved_rows, dim=0).clamp(0.0, 1.0)
        grid = make_grid(rows, nrow=2, padding=4)
        # Each row is GT then O1 reconstruction.
        save_image(grid, out / "recon_grid_gt_o1.png")
    return summary


def write_summary(output_dir: str, summary: Dict[str, float]) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    fields = list(summary.keys())
    with open(out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/jovyan/data/Scale-RAE-Qwen1.5B_DiT2.4B")
    p.add_argument("--val_jsonl", default="/home/jovyan/PGOT/data/pgot_val.jsonl")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    p.add_argument("--coda_crop_size", type=int, default=512)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--guidance_scale", type=float, default=2.5)
    p.add_argument("--diffusion_inference_steps", type=int, default=25)
    p.add_argument("--save_images", type=int, default=0, help="Save N GT/O1 image pairs as recon_grid_gt_o1.png")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer, device, _ = load_scale_rae(args)
    loader = build_loader(args, model)
    decoder = load_rae_decoder(model, device=device, dtype=torch.float32)
    summary = run_eval(args, model, tokenizer, loader, decoder, device)
    write_summary(args.output_dir, summary)
    log.info("%s", json.dumps(summary, sort_keys=True))
    log.info("Wrote %s", Path(args.output_dir) / "summary.csv")


if __name__ == "__main__":
    main()
