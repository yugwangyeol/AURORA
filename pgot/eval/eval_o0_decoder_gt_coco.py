"""Evaluate the O0 decoder floor on COCO val images.

O0 means:

    GT SigLIP target latent -> SigLIP pixel decoder

No DiT, no PGOT bottleneck, no sampling. This measures the decoder floor using
the same torchmetrics FID backend as PGOT eval.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pgot.eval.pgot_metrics import FIDAccumulator, compute_recon_metrics


def _coda_center_crop_image(image: Image.Image, crop_size: int = 512) -> Image.Image:
    """Resize min side to crop_size, then center crop. Mirrors CODA COCO eval."""
    width, height = image.size
    scale = float(crop_size) / float(min(width, height))
    resized = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
    left = max((resized.width - crop_size) // 2, 0)
    top = max((resized.height - crop_size) // 2, 0)
    return resized.crop((left, top, left + crop_size, top + crop_size))


def _denormalize_images(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    mean = mean.to(device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    std = std.to(device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


class CocoValImageDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        captions_json: str,
        image_processor,
        target_image_processor,
        *,
        max_samples: int | None = None,
        image_preprocess_mode: str = "coda_center_crop",
        coda_crop_size: int = 512,
    ) -> None:
        self.image_processor = image_processor
        self.target_image_processor = target_image_processor
        self.image_preprocess_mode = str(image_preprocess_mode)
        self.coda_crop_size = int(coda_crop_size)
        if self.image_preprocess_mode not in {"default", "coda_center_crop"}:
            raise ValueError(f"Unknown image_preprocess_mode={self.image_preprocess_mode}")

        with open(captions_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        images = sorted(payload["images"], key=lambda x: str(x["file_name"]))
        if max_samples is not None:
            images = images[: int(max_samples)]
        self.rows: List[Dict] = []
        for item in images:
            path = os.path.join(image_dir, str(item["file_name"]))
            if os.path.exists(path):
                self.rows.append(
                    {
                        "image_id": int(item["id"]),
                        "file_name": str(item["file_name"]),
                        "image_path": path,
                    }
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_preprocess_mode == "coda_center_crop":
            image = _coda_center_crop_image(image, self.coda_crop_size)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        target_tensor = self.target_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return {
            "image": image_tensor,
            "target_image": target_tensor,
            "image_id": row["image_id"],
            "file_name": row["file_name"],
        }


def collate(batch: List[Dict]) -> Dict:
    return {
        "images": torch.stack([x["image"] for x in batch]),
        "target_images": torch.stack([x["target_image"] for x in batch]),
        "image_ids": [int(x["image_id"]) for x in batch],
        "file_names": [str(x["file_name"]) for x in batch],
    }


def load_target_siglip_and_decoder(args, device: torch.device, dtype: torch.dtype):
    from types import SimpleNamespace

    from huggingface_hub import hf_hub_download
    from scale_rae.model.multimodal_decoder import MultimodalDecoder
    from scale_rae.model.multimodal_encoder.siglip_encoder import SiglipVisionTower

    tower_name = f"{args.target_vision_tower}-interp{int(args.target_tokens)}"
    tower_args = SimpleNamespace(
        mm_vision_select_layer=-1,
        mm_vision_select_feature="patch",
        unfreeze_mm_vision_tower=False,
        normalize_vision=True,
    )
    target_tower = SiglipVisionTower(tower_name, args=tower_args)
    target_tower.eval()
    target_tower.to(device=device, dtype=dtype)

    config_path = hf_hub_download(repo_id=args.decoder_repo, filename="config.json")
    ckpt_path = hf_hub_download(repo_id=args.decoder_repo, filename="model.pt")
    decoder = MultimodalDecoder(
        pretrained_encoder_path=args.target_vision_tower,
        general_decoder_config=config_path,
        num_patches=int(args.target_tokens),
        drop_cls_token=True,
        decoder_path=ckpt_path,
    )
    decoder.eval()
    decoder.to(device=device, dtype=torch.float32)
    if hasattr(decoder, "image_mean"):
        decoder.image_mean = decoder.image_mean.to(device=device, dtype=torch.float32)
        decoder.image_std = decoder.image_std.to(device=device, dtype=torch.float32)
    return target_tower, decoder


@torch.no_grad()
def encode_gt_siglip(target_tower, target_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    tower_dtype = next(target_tower.parameters()).dtype
    gt_siglip = target_tower(target_images.to(device=device, dtype=tower_dtype))
    return gt_siglip.detach().float()


@torch.no_grad()
def decode_to_image(decoder, generated: torch.Tensor, device: torch.device) -> torch.Tensor:
    decoder_dtype = next(decoder.parameters()).dtype
    generated = generated.to(device=device, dtype=decoder_dtype)
    empty_cls = torch.zeros((generated.shape[0], 1, generated.shape[-1]), device=device, dtype=decoder_dtype)
    image_features = torch.cat([empty_cls, generated], dim=1)
    recon = decoder(image_features)
    recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
    return recon.clamp(0.0, 1.0).detach().float()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="/home/jovyan/PGOT/checkpoints/pgot_main_v16_bce_bottleneck_void4")
    p.add_argument("--target_vision_tower", default="google/siglip2-so400m-patch14-224")
    p.add_argument("--target_tokens", type=int, default=256)
    p.add_argument("--decoder_repo", default="nyu-visionx/siglip2_decoder")
    p.add_argument("--image_dir", default="/home/jovyan/data/coco/val2017")
    p.add_argument("--captions_json", default="/home/jovyan/data/coco/annotations/captions_val2017.json")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--diffusion_inference_steps", type=int, default=25, help="Unused by O0; kept for loader compatibility.")
    p.add_argument("--image_preprocess_mode", choices=["default", "coda_center_crop"], default="coda_center_crop")
    p.add_argument("--coda_crop_size", type=int, default=512)
    p.add_argument("--save_images", type=int, default=0, help="Save N GT/O0 pairs as recon_grid_gt_o0.png")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    target_tower, decoder = load_target_siglip_and_decoder(args, device=device, dtype=dtype)
    image_proc = target_tower.image_processor
    target_proc = target_tower.image_processor

    dataset = CocoValImageDataset(
        args.image_dir,
        args.captions_json,
        image_proc,
        target_proc,
        max_samples=args.max_samples,
        image_preprocess_mode=args.image_preprocess_mode,
        coda_crop_size=args.coda_crop_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=collate,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    fid_acc = FIDAccumulator(device=device, feature=2048)
    metric_lists = {"psnr": [], "ssim": [], "mse": [], "mae": []}
    saved = []
    n_samples = 0

    mean = torch.tensor(target_proc.image_mean)
    std = torch.tensor(target_proc.image_std)

    for batch in tqdm(loader, desc="O0 decoder_gt COCO"):
        gt_siglip = encode_gt_siglip(target_tower, batch["target_images"], device)
        fake = decode_to_image(decoder, gt_siglip, device)
        real = _denormalize_images(batch["target_images"].to(device).float(), mean, std)
        if real.shape[-2:] != fake.shape[-2:]:
            real = F.interpolate(real, size=fake.shape[-2:], mode="bilinear", align_corners=False)
        fid_acc.add(real, fake)
        rec = compute_recon_metrics(real, fake)
        for key in metric_lists:
            metric_lists[key].extend([float(x) for x in rec[key].detach().cpu()])
        n_samples += int(real.shape[0])

        if len(saved) < int(args.save_images):
            room = int(args.save_images) - len(saved)
            take = min(room, int(real.shape[0]))
            for i in range(take):
                saved.append(torch.stack([real[i].cpu(), fake[i].cpu()], dim=0))

    summary = {
        "oracle": "o0_decoder_gt_coco",
        "model_path": args.model_path,
        "image_dir": os.path.abspath(args.image_dir),
        "captions_json": os.path.abspath(args.captions_json),
        "num_samples": n_samples,
        "image_preprocess_mode": args.image_preprocess_mode,
        "coda_crop_size": int(args.coda_crop_size),
        "fid_backend": "pgot_torchmetrics_fid",
        "rFID": fid_acc.compute(),
    }
    for key, vals in metric_lists.items():
        summary[f"recon_{key}"] = sum(vals) / max(len(vals), 1)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        keys = list(summary.keys())
        f.write(",".join(keys) + "\n")
        f.write(",".join(str(summary[k]) for k in keys) + "\n")

    if saved:
        from torchvision.utils import make_grid, save_image

        rows = torch.cat(saved, dim=0).clamp(0.0, 1.0)
        grid = make_grid(rows, nrow=2, padding=4)
        save_image(grid, out_dir / "recon_grid_gt_o0.png")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
