"""
Convert 114K individual DINO .pt files into a single memory-mapped numpy array.

Output:
  {root}/dino_memmap.npy      — float16 array of shape (N, 256, 768)
  {root}/dino_memmap_index.pt — dict mapping image_id (str) -> row index (int)

Usage:
  python scripts/build_dino_memmap.py \
      --root /home/jovyan/processed_coco/training_data_v4_patch_dino \
      --feature_name dino_256x768_fp16.pt \
      --num_workers 32
"""
import argparse
import pathlib
import torch
import numpy as np
from multiprocessing.pool import ThreadPool
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/jovyan/processed_coco/training_data_v4_patch_dino")
    p.add_argument("--feature_name", default="dino_256x768_fp16.pt")
    p.add_argument("--num_workers", type=int, default=32)
    return p.parse_args()


def load_one(args):
    idx, img_id, feat_path = args
    obj = torch.load(feat_path, map_location="cpu")
    feat = obj["features"] if isinstance(obj, dict) else obj
    return idx, img_id, feat.numpy().astype(np.float16)


def main():
    args = parse_args()
    root = pathlib.Path(args.root)

    # Collect all image dirs
    img_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    valid = [(i, d.name, d / args.feature_name) for i, d in enumerate(img_dirs) if (d / args.feature_name).exists()]
    N = len(valid)
    print(f"Found {N} feature files under {root}")

    # Peek at shape
    sample_obj = torch.load(valid[0][2], map_location="cpu")
    sample_feat = sample_obj["features"] if isinstance(sample_obj, dict) else sample_obj
    feat_shape = tuple(sample_feat.shape)  # e.g. (256, 768)
    print(f"Feature shape per sample: {feat_shape}, dtype: float16")
    total_gb = N * np.prod(feat_shape) * 2 / 1e9
    print(f"Total output size: {total_gb:.1f} GB")

    # Create memmap
    out_path = root / "dino_memmap.npy"
    idx_path = root / "dino_memmap_index.pt"
    arr = np.lib.format.open_memmap(
        str(out_path), mode="w+", dtype=np.float16, shape=(N, *feat_shape)
    )

    id_to_idx = {}
    print(f"Loading with {args.num_workers} threads...")
    with ThreadPool(args.num_workers) as pool:
        for row_idx, img_id, feat in tqdm(pool.imap_unordered(load_one, valid), total=N):
            arr[row_idx] = feat
            id_to_idx[img_id] = row_idx

    arr.flush()
    torch.save(id_to_idx, str(idx_path))
    print(f"Saved memmap:  {out_path}  shape={arr.shape}")
    print(f"Saved index:   {idx_path}  ({len(id_to_idx)} entries)")


if __name__ == "__main__":
    main()
