#!/usr/bin/env python
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded CaptionSlot eval outputs.")
    parser.add_argument("--root-output-dir", required=True)
    parser.add_argument("--shard-dir", dest="shard_dirs", action="append", required=True)
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def matrix_sqrt_psd(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_eigvals = np.sqrt(eigvals)
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    eps = 1e-6
    sigma_real = sigma_real + np.eye(sigma_real.shape[0], dtype=sigma_real.dtype) * eps
    sigma_fake = sigma_fake + np.eye(sigma_fake.shape[0], dtype=sigma_fake.dtype) * eps

    sqrt_sigma_real = matrix_sqrt_psd(sigma_real)
    cov_prod = sqrt_sigma_real @ sigma_fake @ sqrt_sigma_real
    cov_prod = 0.5 * (cov_prod + cov_prod.T)
    covmean = matrix_sqrt_psd(cov_prod)

    diff = mu_real - mu_fake
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return float(fid)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_samples(shard_dirs: List[str], root_output_dir: str) -> None:
    merge_named_dirs(shard_dirs, root_output_dir, "samples")


def merge_named_dirs(shard_dirs: List[str], root_output_dir: str, subdir_name: str) -> None:
    root_samples_dir = os.path.join(root_output_dir, subdir_name)
    ensure_output_dir(root_samples_dir)
    for shard_dir in shard_dirs:
        shard_samples_dir = os.path.join(shard_dir, subdir_name)
        if not os.path.isdir(shard_samples_dir):
            continue
        for sample_name in sorted(os.listdir(shard_samples_dir)):
            src_dir = os.path.join(shard_samples_dir, sample_name)
            dst_dir = os.path.join(root_samples_dir, sample_name)
            if os.path.isfile(src_dir):
                shutil.copy2(src_dir, dst_dir)
                continue
            if not os.path.isdir(src_dir):
                continue
            ensure_output_dir(dst_dir)
            for file_name in sorted(os.listdir(src_dir)):
                src_file = os.path.join(src_dir, file_name)
                dst_file = os.path.join(dst_dir, file_name)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)


def weighted_average(items: List[Dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    numer = 0.0
    denom = 0.0
    for item in items:
        value = item.get(value_key)
        weight = item.get(weight_key, 0)
        if value is None or weight is None:
            continue
        numer += float(value) * float(weight)
        denom += float(weight)
    if denom == 0:
        return None
    return numer / denom


def merge_metric_summary(
    shard_summaries: List[Dict[str, Any]],
    summary_key: str,
    metric_keys: List[str],
    weight_key: str = "evaluated_images",
) -> Dict[str, Any] | None:
    entries = [summary.get(summary_key, {}) for summary in shard_summaries]
    if not any(entry for entry in entries):
        return None
    merged: Dict[str, Any] = {}
    first = next((entry for entry in entries if entry), {})
    for key, value in first.items():
        if key not in metric_keys and key != weight_key:
            merged[key] = value
    merged[weight_key] = int(sum(int(entry.get(weight_key, 0)) for entry in entries))
    for metric_key in metric_keys:
        merged[metric_key] = weighted_average(entries, metric_key, weight_key)
    return merged


def main() -> None:
    args = parse_args()
    root_output_dir = os.path.abspath(args.root_output_dir)
    shard_dirs = [os.path.abspath(path) for path in args.shard_dirs]
    ensure_output_dir(root_output_dir)

    shard_summaries: List[Dict[str, Any]] = []
    captions_records: List[Dict[str, Any]] = []
    per_image_records: List[Dict[str, Any]] = []
    per_image_attn_records: List[Dict[str, Any]] = []
    real_features: List[np.ndarray] = []
    fake_features: List[np.ndarray] = []

    for shard_dir in shard_dirs:
        summary_path = os.path.join(shard_dir, "summary.json")
        if not os.path.isfile(summary_path):
            raise FileNotFoundError(f"Missing shard summary: {summary_path}")
        summary = read_json(summary_path)
        shard_summaries.append(summary)

        captions_path = os.path.join(shard_dir, "captions.jsonl")
        if os.path.isfile(captions_path):
            captions_records.extend(iter_jsonl(captions_path))

        per_image_path = os.path.join(shard_dir, "per_image.jsonl")
        if os.path.isfile(per_image_path):
            per_image_records.extend(iter_jsonl(per_image_path))

        per_image_attn_path = os.path.join(shard_dir, "per_image_attn.jsonl")
        if os.path.isfile(per_image_attn_path):
            per_image_attn_records.extend(iter_jsonl(per_image_attn_path))

        features_path = os.path.join(shard_dir, "fid_features.npz")
        if os.path.isfile(features_path):
            with np.load(features_path) as payload:
                real_arr = payload["real"]
                fake_arr = payload["fake"]
            if real_arr.size > 0:
                real_features.append(real_arr)
                fake_features.append(fake_arr)

    captions_records.sort(key=lambda item: item.get("image", ""))
    per_image_records.sort(key=lambda item: item.get("image", ""))
    per_image_attn_records.sort(key=lambda item: (item.get("image_id", ""), item.get("caption", "")))
    merged_captions = os.path.join(root_output_dir, "captions.jsonl")
    merged_per_image = os.path.join(root_output_dir, "per_image.jsonl")
    merged_per_image_attn = os.path.join(root_output_dir, "per_image_attn.jsonl")
    write_jsonl(merged_captions, captions_records)
    write_jsonl(merged_per_image, per_image_records)
    write_jsonl(merged_per_image_attn, per_image_attn_records)
    merge_samples(shard_dirs, root_output_dir)
    merge_named_dirs(shard_dirs, root_output_dir, "attn_maps")

    first = shard_summaries[0]
    loss_summaries = [summary.get("loss_metrics", {}) for summary in shard_summaries]
    loss_metrics = {
        "eval/loss": weighted_average(loss_summaries, "eval/loss", "loss_eval_num_batches"),
        "eval/loss_recon": weighted_average(loss_summaries, "eval/loss_recon", "loss_eval_num_batches"),
        "loss_eval_num_batches": int(sum(int(item.get("loss_eval_num_batches", 0)) for item in loss_summaries)),
        "loss_eval_num_samples": int(sum(int(item.get("loss_eval_num_samples", 0)) for item in loss_summaries)),
    }

    recon_records = [record["metrics"] for record in per_image_records if isinstance(record.get("metrics"), dict)]
    if recon_records:
        reconstruction_metrics: Dict[str, Any] = {
            "PSNR": float(np.mean([item["psnr"] for item in recon_records])),
            "SSIM": float(np.mean([item["ssim"] for item in recon_records])),
            "MSE": float(np.mean([item["mse"] for item in recon_records])),
            "MAE": float(np.mean([item["mae"] for item in recon_records])),
        }
    else:
        reconstruction_metrics = {"PSNR": None, "SSIM": None, "MSE": None, "MAE": None}

    generated_count = int(sum(int(summary.get("reconstruction_metrics", {}).get("generated_count", 0)) for summary in shard_summaries))
    failed_count = int(sum(int(summary.get("reconstruction_metrics", {}).get("failed_count", 0)) for summary in shard_summaries))
    requested_samples = int(sum(int(summary.get("reconstruction_metrics", {}).get("requested_samples", 0)) for summary in shard_summaries))

    if real_features:
        real_arr = np.concatenate(real_features, axis=0)
        fake_arr = np.concatenate(fake_features, axis=0)
        rfid_value = compute_fid(real_arr, fake_arr) if real_arr.shape[0] >= 2 else None
    else:
        real_arr = np.empty((0, 2048), dtype=np.float32)
        fake_arr = np.empty((0, 2048), dtype=np.float32)
        rfid_value = None

    merged_features = os.path.join(root_output_dir, "fid_features.npz")
    np.savez_compressed(merged_features, real=real_arr, fake=fake_arr)

    reconstruction_metrics.update(
        {
            "requested_samples": requested_samples,
            "generated_count": generated_count,
            "failed_count": failed_count,
            "success_rate": (generated_count / requested_samples) if requested_samples else 0.0,
            "rFID": rfid_value,
            "captions_jsonl": os.path.abspath(merged_captions),
            "per_image_jsonl": os.path.abspath(merged_per_image),
            "fid_features_path": os.path.abspath(merged_features),
        }
    )

    summary = {
        "model_path": first.get("model_path"),
        "image_dir": first.get("image_dir"),
        "output_dir": os.path.abspath(root_output_dir),
        "dtype": first.get("dtype"),
        "guidance_level": first.get("guidance_level"),
        "max_slots": first.get("max_slots"),
        "num_shards": len(shard_summaries),
        "shard_dirs": shard_dirs,
        "sample_count": int(sum(int(summary.get("sample_count", 0)) for summary in shard_summaries)),
        "global_sample_count": first.get("global_sample_count"),
        "captions_jsonl": first.get("captions_jsonl"),
        "per_image_attn_jsonl": os.path.abspath(merged_per_image_attn),
        "loss_metrics": loss_metrics,
        "reconstruction_metrics": reconstruction_metrics,
        "notes": list(first.get("notes", [])) + ["This summary was merged from sharded runs and sorted by image path."],
    }
    slot_attention_metrics = merge_metric_summary(
        shard_summaries,
        "slot_attention_metrics",
        ["MBO", "mIoU", "loc_acc_top16"],
    )
    if slot_attention_metrics is not None:
        slot_attention_metrics["per_image_attn_jsonl"] = os.path.abspath(merged_per_image_attn)
        summary["slot_attention_metrics"] = slot_attention_metrics

    segmentation_metrics_strict = merge_metric_summary(
        shard_summaries,
        "segmentation_metrics_strict",
        ["fARI", "MBO", "mIoU"],
    )
    if segmentation_metrics_strict is not None:
        summary["segmentation_metrics_strict"] = segmentation_metrics_strict

    segmentation_metrics_bg = merge_metric_summary(
        shard_summaries,
        "segmentation_metrics_bg",
        ["fARI", "MBO", "mIoU"],
    )
    if segmentation_metrics_bg is not None:
        summary["segmentation_metrics_bg"] = segmentation_metrics_bg

    # Per-threshold sections: discover keys across shards and aggregate each.
    thr_section_keys = set()
    for s in shard_summaries:
        for k in s.keys():
            if k.startswith("segmentation_metrics_bg_thr") or k.startswith("slot_attention_metrics_thr"):
                thr_section_keys.add(k)
    for key in sorted(thr_section_keys):
        if key.startswith("segmentation_metrics_bg_thr"):
            metric_keys = ["fARI", "MBO", "mIoU"]
        else:  # slot_attention_metrics_thr
            metric_keys = ["MBO", "mIoU", "loc_acc_top16"]
        merged = merge_metric_summary(shard_summaries, key, metric_keys)
        if merged is not None:
            summary[key] = merged

    write_json(os.path.join(root_output_dir, "summary.json"), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
