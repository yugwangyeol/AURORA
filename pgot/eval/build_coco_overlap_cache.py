"""Build CODA-style COCO instance overlap masks aligned to an existing cache."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm


def preprocess_coda(mask: np.ndarray, size: int) -> np.ndarray:
    h, w = mask.shape[:2]
    factor = max(size / h, size / w)
    rh, rw = int(round(h * factor)), int(round(w * factor))
    mask = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    top = max((rh - size) // 2, 0)
    left = max((rw - size) // 2, 0)
    return mask[top:top + size, left:left + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir",
        default="/home/jovyan/PGOT/data/coco_inst_mask_cache_coda256",
    )
    parser.add_argument(
        "--instances_json",
        default="/home/jovyan/data/coco/annotations/instances_val2017.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = cache_dir / "overlap_masks.npy"
    if output_path.exists() and not args.overwrite:
        print(f"Already exists: {output_path}")
        return

    image_ids = np.load(cache_dir / "image_ids.npy")
    with (cache_dir / "meta.json").open() as f:
        meta = json.load(f)
    size = int(meta["size"])
    coco = COCO(args.instances_json)
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(image_ids), size, size),
    )

    for idx, image_id in enumerate(tqdm(image_ids, desc="COCO overlap masks")):
        info = coco.loadImgs([int(image_id)])[0]
        count = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=[int(image_id)], iscrowd=False)
        for ann in coco.loadAnns(ann_ids):
            if ann.get("ignore", False) or ann.get("area", 0) <= 0:
                continue
            count += coco.annToMask(ann).astype(np.uint8)
        output[idx] = preprocess_coda((count > 1).astype(np.uint8), size)

    output.flush()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
