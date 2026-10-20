"""Create compact direct RefCOCO PGOT manifests.

Each official referring expression becomes one training example with exactly
one OVT.  The JSONL stores only the COCO annotation id; the dataset resolves
the polygon/RLE from the official instances.json at runtime.  This avoids
duplicating the same large segmentation polygon for every sentence.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from PIL import Image


RESERVED = (
    "<thing>", "<stuff>", "<ovt>", "<scene_end>",
    "<|im_start|>", "<|im_end|>",
)


def referring_caption(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("RefCOCO contains an empty referring expression")
    if any(token in text for token in RESERVED):
        raise ValueError("A referring expression contains a reserved model token")
    return f"<thing> Object 1: {text} <ovt>. <scene_end>"


def resolve_image(root: Path, image: dict) -> Path:
    image_id = int(image["id"])
    filename = Path(image["file_name"]).name
    candidates = [
        root / "train2017" / f"{image_id:012d}.jpg",
        root / "val2017" / f"{image_id:012d}.jpg",
        root / "train2014" / filename,
        root / "val2014" / filename,
        root / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"Missing COCO image {image_id} below {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/home/jovyan/data/coco")
    parser.add_argument("--dataset", default="refcoco", choices=("refcoco",))
    parser.add_argument("--split-by", default="unc", choices=("unc",))
    parser.add_argument("--output-dir", default="/home/jovyan/PGOT/data/refcoco_direct")
    parser.add_argument("--splits", nargs="+", default=("train", "val"))
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Deterministic per-split cap for smoke data; 0 keeps all")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    annotation_root = data_root / args.dataset
    refs_path = annotation_root / f"refs({args.split_by}).p"
    instances_path = annotation_root / "instances.json"
    if not refs_path.is_file() or not instances_path.is_file():
        raise FileNotFoundError(f"Missing official RefCOCO files below {annotation_root}")
    with refs_path.open("rb") as stream:
        refs = pickle.load(stream, encoding="latin1")
    instances = json.loads(instances_path.read_text())
    images = {int(image["id"]): image for image in instances["images"]}
    annotations = {int(ann["id"]): ann for ann in instances["annotations"]}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": args.dataset,
        "split_by": args.split_by,
        "refs_path": str(refs_path.resolve()),
        "instances_json": str(instances_path.resolve()),
        "protocol": "one original expression -> one Object 1 OVT and its referred GT mask",
        "splits": {},
    }
    for split in args.splits:
        selected = sorted(
            (ref for ref in refs if ref["split"] == split),
            key=lambda ref: int(ref["ref_id"]),
        )
        if not selected:
            raise ValueError(f"No references in official split {split}")
        records = []
        for ref in selected:
            annotation_id = int(ref["ann_id"])
            image_id = int(ref["image_id"])
            annotation = annotations[annotation_id]
            if int(annotation["image_id"]) != image_id:
                raise ValueError("Reference and annotation image ids disagree")
            image_info = images[image_id]
            image_path = resolve_image(data_root, image_info)
            with Image.open(image_path) as image:
                if image.size != (int(image_info["width"]), int(image_info["height"])):
                    raise ValueError(f"Image dimensions disagree: {image_path}")
            for sentence in sorted(ref["sentences"], key=lambda row: int(row["sent_id"])):
                expression = sentence.get("raw", sentence.get("sent"))
                records.append({
                    "dataset_type": "coco_instance",
                    "task": "refcoco_direct",
                    "split": split,
                    "ref_id": int(ref["ref_id"]),
                    "sent_id": int(sentence["sent_id"]),
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "width": int(image_info["width"]),
                    "height": int(image_info["height"]),
                    "caption": referring_caption(expression),
                    "segments": [{
                        "ann_id": annotation_id,
                        "category_id": -1,
                        "category": "Object",
                        "description": expression,
                        "is_thing": True,
                    }],
                })
                if args.max_samples and len(records) >= args.max_samples:
                    break
            if args.max_samples and len(records) >= args.max_samples:
                break
        path = output_dir / f"{split}.jsonl"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(path)
        summary["splits"][split] = {
            "expressions": len(records),
            "images": len({record["image_id"] for record in records}),
            "references": len({record["ref_id"] for record in records}),
            "path": str(path.resolve()),
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
