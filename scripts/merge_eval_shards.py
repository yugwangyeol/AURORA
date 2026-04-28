#!/usr/bin/env python
"""Merge sharded eval outputs into a single summary.

Combines per-image and per-image-attn JSONL files from shard_0/, shard_1/, ...
under BASE_DIR and writes BASE_DIR/summary_merged.json with aggregated metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_shard_summary(shard_dir: Path) -> Dict[str, Any]:
    p = shard_dir / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def compute_fid(real: np.ndarray, fake: np.ndarray) -> float:
    from scipy import linalg
    mu_r, mu_f = real.mean(0), fake.mean(0)
    sig_r = np.cov(real, rowvar=False)
    sig_f = np.cov(fake, rowvar=False)
    diff = mu_r - mu_f
    covmean, _ = linalg.sqrtm(sig_r @ sig_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig_r + sig_f - 2 * covmean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=2)
    args = ap.parse_args()

    base = Path(args.base_dir)
    per_image_all: List[Dict[str, Any]] = []
    per_image_attn_all: List[Dict[str, Any]] = []
    real_feats: List[np.ndarray] = []
    fake_feats: List[np.ndarray] = []
    shard_summaries: List[Dict[str, Any]] = []

    for i in range(args.num_shards):
        sd = base / f"shard_{i}"
        per_image_all.extend(load_jsonl(sd / "per_image.jsonl"))
        per_image_attn_all.extend(load_jsonl(sd / "per_image_attn.jsonl"))
        shard_summaries.append(load_shard_summary(sd))
        feats_path = sd / "fid_features.npz"
        if feats_path.exists():
            with np.load(feats_path) as z:
                if z["real"].size:
                    real_feats.append(z["real"])
                if z["fake"].size:
                    fake_feats.append(z["fake"])

    # Aggregate reconstruction metrics
    recon_metrics = [r.get("metrics") for r in per_image_all if r.get("generated_image") and r.get("metrics")]
    psnr = float(np.mean([m["psnr"] for m in recon_metrics])) if recon_metrics else None
    ssim = float(np.mean([m["ssim"] for m in recon_metrics])) if recon_metrics else None
    mse  = float(np.mean([m["mse"]  for m in recon_metrics])) if recon_metrics else None
    mae  = float(np.mean([m["mae"]  for m in recon_metrics])) if recon_metrics else None

    rfid = None
    if real_feats and fake_feats:
        real_arr = np.concatenate(real_feats)
        fake_arr = np.concatenate(fake_feats)
        if len(real_arr) >= 2 and len(fake_arr) >= 2:
            rfid = compute_fid(real_arr, fake_arr)

    # Aggregate slot-attention metrics
    mbo = [r["MBO"]          for r in per_image_attn_all if "MBO" in r]
    miou = [r["mIoU"]         for r in per_image_attn_all if "mIoU" in r]
    loc = [r["loc_acc_top16"] for r in per_image_attn_all if "loc_acc_top16" in r]

    # Aggregate loss
    losses = [s.get("loss_metrics", {}).get("eval/loss")        for s in shard_summaries]
    losses_recon = [s.get("loss_metrics", {}).get("eval/loss_recon") for s in shard_summaries]
    losses = [x for x in losses if x is not None]
    losses_recon = [x for x in losses_recon if x is not None]

    merged: Dict[str, Any] = {
        "base_dir": str(base.resolve()),
        "num_shards": args.num_shards,
        "total_samples": len(per_image_all),
        "total_generated": sum(1 for r in per_image_all if r.get("generated_image")),
        "reconstruction_metrics": {
            "PSNR": psnr, "SSIM": ssim, "MSE": mse, "MAE": mae, "rFID": rfid,
            "samples": len(recon_metrics),
        },
        "slot_attention_metrics": {
            "MBO":           float(np.mean(mbo))  if mbo  else None,
            "mIoU":          float(np.mean(miou)) if miou else None,
            "loc_acc_top16": float(np.mean(loc))  if loc  else None,
            "samples":       len(mbo),
        },
        "loss_metrics": {
            "eval/loss":       float(np.mean(losses))       if losses       else None,
            "eval/loss_recon": float(np.mean(losses_recon)) if losses_recon else None,
        },
        "shard_summaries": shard_summaries,
    }

    out_path = base / "summary_merged.json"
    out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(merged, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
