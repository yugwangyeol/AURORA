#!/usr/bin/env python3
"""Paired bootstrap comparison for matched object-memory editing runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


PAIR_KEYS = (
    "pair_index", "category_id", "target_sample_index", "target_object_index",
    "donor_sample_index", "donor_object_index", "diffusion_seed",
)
METRICS = (
    "donor_gain", "donor_closer", "outside_target_lpips", "target_change",
    "inside_target_latent_mse", "outside_target_latent_mse",
    "outside_target_pixel_l1",
)


def load(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def interval(samples: np.ndarray) -> list[float]:
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    full = load(args.full)
    ablation = load(args.ablation)
    if len(full) != len(ablation) or not full:
        raise ValueError("Editing runs must contain the same nonzero number of pairs")
    for index, (left, right) in enumerate(zip(full, ablation)):
        if tuple(left[key] for key in PAIR_KEYS) != tuple(right[key] for key in PAIR_KEYS):
            raise ValueError(f"Editing pair mismatch at row {index}")

    count = len(full)
    rng = np.random.default_rng(args.seed)
    sampled = rng.integers(0, count, size=(args.iterations, count))
    result = {
        "full": str(args.full),
        "ablation": str(args.ablation),
        "num_matched_pairs": count,
        "bootstrap_iterations": args.iterations,
        "definition": (
            "Pair-level bootstrap with identical sampled pair indices in both conditions; "
            "delta is Full minus w/o Reader supervision."
        ),
        "metrics": {},
    }
    for metric in METRICS:
        full_values = np.asarray([row[metric] for row in full], dtype=np.float64)
        ablation_values = np.asarray([row[metric] for row in ablation], dtype=np.float64)
        difference = full_values - ablation_values
        full_bootstrap = full_values[sampled].mean(axis=1)
        ablation_bootstrap = ablation_values[sampled].mean(axis=1)
        difference_bootstrap = difference[sampled].mean(axis=1)
        result["metrics"][metric] = {
            "full_mean": float(full_values.mean()),
            "full_ci95_paired_resamples": interval(full_bootstrap),
            "ablation_mean": float(ablation_values.mean()),
            "ablation_ci95_paired_resamples": interval(ablation_bootstrap),
            "full_minus_ablation": float(difference.mean()),
            "delta_ci95": interval(difference_bootstrap),
            "delta_excludes_zero": bool(
                np.quantile(difference_bootstrap, 0.025) > 0
                or np.quantile(difference_bootstrap, 0.975) < 0
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    markdown = args.output.with_suffix(".md")
    lines = [
        "| Metric | Full | w/o RS | Full − w/o RS | Paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, values in result["metrics"].items():
        low, high = values["delta_ci95"]
        lines.append(
            f"| {metric} | {values['full_mean']:.6f} | {values['ablation_mean']:.6f} | "
            f"{values['full_minus_ablation']:+.6f} | [{low:+.6f}, {high:+.6f}] |"
        )
    markdown.write_text("\n".join(lines) + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
