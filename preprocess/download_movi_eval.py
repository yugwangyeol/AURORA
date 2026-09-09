"""Download one MOVi split in the directory layout used by CODA evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, bytes):
            return obj.decode()
        return super().default(obj)


def as_dense_numpy(value):
    """Convert TF tensors, including ragged tensors, to dense NumPy arrays."""
    if hasattr(value, "to_tensor"):
        value = value.to_tensor()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=Path, required=True)
    parser.add_argument("--level", choices=["c", "e"], required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--split", default="validation")
    return parser.parse_args()


def main():
    args = parse_args()

    import tensorflow_datasets as tfds

    dataset_name = f"movi_{args.level}/{args.image_size}x{args.image_size}:1.0.0"
    dataset = tfds.load(
        dataset_name,
        split=args.split,
        data_dir="gs://kubric-public/tfds",
    )
    split_root = args.out_path / args.split
    split_root.mkdir(parents=True, exist_ok=True)

    for video_index, record in enumerate(tqdm(tfds.as_numpy(dataset), desc=dataset_name)):
        video = record["video"]
        masks = record["segmentations"]
        if masks.shape[0] != video.shape[0]:
            raise ValueError(
                f"Frame/mask count mismatch: video={video.shape}, masks={masks.shape}"
            )

        video_root = split_root / f"{video_index:08d}"
        video_root.mkdir(parents=True, exist_ok=True)
        num_frames = int(video.shape[0])

        for frame_index in range(num_frames):
            stem = f"{frame_index:08d}"
            image = np.asarray(video[frame_index], dtype=np.uint8)
            mask = np.asarray(masks[frame_index, ..., 0], dtype=np.uint8)
            Image.fromarray(image, mode="RGB").save(video_root / f"{stem}.jpg")
            Image.fromarray(mask, mode="L").save(video_root / f"{stem}_mask.png")

            frame_instances = {}
            for key, value in record["instances"].items():
                value = as_dense_numpy(value)
                if value.ndim > 1 and value.shape[1] == num_frames:
                    value = value[:, frame_index]
                frame_instances[key] = value
            with open(video_root / f"{stem}_instances.json", "w", encoding="utf-8") as writer:
                json.dump(frame_instances, writer, cls=NumpyJSONEncoder)


if __name__ == "__main__":
    main()
