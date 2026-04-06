#!/usr/bin/env python3
"""
Compute per-channel mean/var statistics of SigLIP2 features over a dataset.

Produces a .pt file compatible with FullSequenceRectifiedFlowProjector's
batchnorm_path (keys: "running_mean", "running_var").

Usage:
    python scripts/compute_siglip_stats.py \
        --image_folder /home/jovyan/data/coco/train2017 \
        --output_path /home/jovyan/data/siglip2_bn_stats.pt \
        --num_images 10000
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, SiglipImageProcessor

VISION_TOWER_NAME = "google/siglip2-so400m-patch14-224"
HIDDEN_SIZE = 1152
IMAGE_SIZE = 224
INTERP_SIZE = 256  # matches vision_tower_aux_token_len_list


def load_vision_tower(device: str = "cuda:0"):
    """Load SigLIP2 vision tower (same as training pipeline)."""
    full_model = AutoModel.from_pretrained(VISION_TOWER_NAME)
    vision_tower = full_model.vision_model.to(device).eval()
    processor = SiglipImageProcessor.from_pretrained(VISION_TOWER_NAME)
    processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    return vision_tower, processor


@torch.no_grad()
def extract_features(vision_tower, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract features matching the training pipeline.

    Training uses transformers 4.37 where hidden_states[-1] returns the
    last encoder layer output *before* the model's post_layernorm.
    Newer transformers (>=5.x) may not return hidden_states at all.

    Either way, the SiglipVisionTower.interpolate() applies an
    unparameterised F.layer_norm which re-normalises the features,
    making pre-vs-post-LN negligible for channel-wise statistics.
    """
    outputs = vision_tower(pixel_values, output_hidden_states=True, return_dict=True)
    if outputs.hidden_states is not None:
        # Prefer pre-post-layernorm features (matches training path)
        features = outputs.hidden_states[-1]
    else:
        features = outputs.last_hidden_state
    # Unparameterised LayerNorm (same as siglip_encoder.py interpolate)
    features = F.layer_norm(features, (HIDDEN_SIZE,), weight=None, bias=None, eps=1e-6)
    return features  # (B, 256, 1152)


def collect_image_paths(image_folder: str, max_images: int = 0) -> list:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(
        p for p in Path(image_folder).iterdir()
        if p.suffix.lower() in exts
    )
    if max_images > 0:
        paths = paths[:max_images]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Compute SigLIP2 feature statistics")
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Max images to use (0 = all)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print(f"Loading SigLIP2 vision tower: {VISION_TOWER_NAME}")
    vision_tower, processor = load_vision_tower(args.device)
    vt_dtype = next(vision_tower.parameters()).dtype

    image_paths = collect_image_paths(args.image_folder, args.num_images)
    print(f"Found {len(image_paths)} images in {args.image_folder}")
    if len(image_paths) == 0:
        print("ERROR: No images found.")
        sys.exit(1)

    # Welford online algorithm for numerically stable mean/var
    n = 0
    mean = torch.zeros(HIDDEN_SIZE, dtype=torch.float64)
    M2 = torch.zeros(HIDDEN_SIZE, dtype=torch.float64)

    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="Computing stats"):
        batch_paths = image_paths[start : start + args.batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"  skip {p}: {e}")
                continue
        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(args.device, dtype=vt_dtype)

        features = extract_features(vision_tower, pixel_values)  # (B, 256, 1152)
        # Flatten to (B*256, 1152) — treat each patch independently
        flat = features.reshape(-1, HIDDEN_SIZE).to(dtype=torch.float64, device="cpu")

        for i in range(flat.shape[0]):
            n += 1
            delta = flat[i] - mean
            mean += delta / n
            delta2 = flat[i] - mean
            M2 += delta * delta2

    if n < 2:
        print("ERROR: Not enough samples.")
        sys.exit(1)

    var = M2 / (n - 1)
    mean_f32 = mean.float()
    var_f32 = var.float()

    std = torch.sqrt(var_f32 + 1e-5)
    print(f"\n=== Statistics over {n} patches ===")
    print(f"  mean  — min: {mean_f32.min():.4f}, max: {mean_f32.max():.4f}, "
          f"abs_mean: {mean_f32.abs().mean():.4f}")
    print(f"  std   — min: {std.min():.4f}, max: {std.max():.4f}, "
          f"mean: {std.mean():.4f}")
    print(f"  var   — min: {var_f32.min():.4f}, max: {var_f32.max():.4f}, "
          f"mean: {var_f32.mean():.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save({"running_mean": mean_f32, "running_var": var_f32}, args.output_path)
    print(f"\nSaved to {args.output_path}")


if __name__ == "__main__":
    main()
