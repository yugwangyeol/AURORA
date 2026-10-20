"""Shared utilities for manifest-locked RefCOCO MLLM baseline evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_samples(manifest_path, subset="nonoverlap", train_path=None, max_samples=0, splits=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    samples = list(manifest["samples"])
    if splits:
        allowed = set(splits)
        samples = [sample for sample in samples if sample["split"] in allowed]
    if subset == "nonoverlap":
        if train_path is None:
            raise ValueError("train_path is required for the nonoverlap subset")
        train_ids = {
            int(json.loads(line)["image_id"])
            for line in Path(train_path).open()
            if line.strip()
        }
        samples = [sample for sample in samples if int(sample["image_id"]) not in train_ids]
    elif subset != "full":
        raise ValueError(f"Unknown subset: {subset}")
    if max_samples:
        samples = samples[:max_samples]
    if not samples:
        raise ValueError("The selected RefCOCO manifest subset is empty")
    return manifest, samples


def group_samples_by_image(samples):
    groups = OrderedDict()
    for sample in samples:
        groups.setdefault(int(sample["image_id"]), []).append(sample)
    return list(groups.values())


def prediction_folder(output_root, sample):
    return Path(output_root) / sample["split"] / str(sample["ref_id"]) / str(sample["sent_id"])


def prediction_complete(output_root, sample):
    folder = prediction_folder(output_root, sample)
    mask_path, metadata_path = folder / "prediction.npz", folder / "metadata.json"
    if not mask_path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text())
    for key in ("query_sha256", "image_sha256"):
        if metadata.get(key) != sample[key]:
            raise ValueError(f"Cached prediction provenance mismatch: {folder}")
    return True


def save_prediction(output_root, sample, mask, model, **metadata):
    mask = np.asarray(mask, dtype=np.bool_)
    expected_shape = (int(sample["height"]), int(sample["width"]))
    if mask.shape != expected_shape:
        raise ValueError(f"Expected mask {expected_shape}, got {mask.shape}")
    folder = prediction_folder(output_root, sample)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / "prediction.tmp.npz"
    np.savez_compressed(temporary, mask=mask)
    temporary.replace(folder / "prediction.npz")
    record = {
        "model": model,
        "query_sha256": sample["query_sha256"],
        "image_sha256": sample["image_sha256"],
        "empty_mask": not bool(mask.any()),
        **metadata,
    }
    (folder / "metadata.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)
