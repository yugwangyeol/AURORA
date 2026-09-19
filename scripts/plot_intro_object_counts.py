#!/usr/bin/env python3
"""Plot COCO per-image count distributions for the paper introduction."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/paper_figures/intro_object_count_v11_wrap"
DEFAULT_SOURCES = {
    "AdaSlot": PROJECT_ROOT / "outputs/baselines_coco_object_counts/adaslot/per_image.jsonl",
    "MetaSlot": PROJECT_ROOT / "outputs/baselines_coco_object_counts/metaslot/per_image.jsonl",
    "CODA": PROJECT_ROOT / "outputs/baselines_coco_object_counts/coda/per_image.jsonl",
    "DeOR": PROJECT_ROOT / "outputs/segeval_pgot_memory8_full_coco_voc_ar/coco_ar/per_image.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tail-bin", type=int, default=20)
    for name, path in DEFAULT_SOURCES.items():
        parser.add_argument(f"--{name.lower()}-records", type=Path, default=path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing count results: {path}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def extract_counts(name: str, rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    prediction_key = "generated_objects" if name == "DeOR" else "active_objects"
    try:
        ids = [int(row["id"]) for row in rows]
        gt = np.asarray([int(row["gt_objects"]) for row in rows])
        pred = np.asarray([int(row[prediction_key]) for row in rows])
    except KeyError as exc:
        raise KeyError(f"Missing {exc.args[0]!r} in {name} count records") from exc
    if len(set(ids)) != len(ids) or np.any(gt < 0) or np.any(pred < 0):
        raise ValueError(f"Invalid IDs/counts for {name}")
    return gt, pred, ids


def discrete_distribution(values: np.ndarray, tail_bin: int) -> np.ndarray:
    """Return the per-count probability mass, grouping the tail in the last bin."""
    clipped = np.minimum(values, tail_bin)
    counts = Counter(int(value) for value in clipped)
    return np.asarray([counts.get(index, 0) / len(values) for index in range(tail_bin + 1)])


def main() -> None:
    args = parse_args()
    if args.tail_bin < 8:
        raise ValueError("--tail-bin must be at least 8")

    paths = {
        name: getattr(args, f"{name.lower()}_records")
        for name in DEFAULT_SOURCES
    }
    data = {}
    reference_ids = reference_gt = None
    for name, path in paths.items():
        gt, pred, ids = extract_counts(name, read_jsonl(path))
        if reference_ids is None:
            reference_ids, reference_gt = ids, gt
        elif ids != reference_ids or not np.array_equal(gt, reference_gt):
            raise ValueError(f"{name} does not use the identical COCO image/GT sequence")
        data[name] = {
            "gt": gt,
            "pred": pred,
            "mean": float(pred.mean()),
            "mae": float(np.abs(pred - gt).mean()),
        }

    assert reference_gt is not None
    gt_mean = float(reference_gt.mean())
    x = np.arange(args.tail_bin + 1)
    gt_distribution = discrete_distribution(reference_gt, args.tail_bin)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    colors = {
        "GT": "#32363D",
        "AdaSlot": "#E28A15",
        "MetaSlot": "#D9538F",
        "CODA": "#7756B3",
        "DeOR": "#008A9A",
        "grid": "#D9DEE6",
        "axis": "#8993A1",
        "text": "#20242A",
        "muted": "#5E6877",
    }

    # Native size targets a roughly 0.45\textwidth wrapfigure. Keeping the PDF
    # close to its final physical width prevents LaTeX from shrinking all text.
    fig, axis = plt.subplots(figsize=(3.65, 2.65))
    axis.bar(
        x,
        gt_distribution,
        width=0.86,
        color="#AEB7C4",
        alpha=0.42,
        edgecolor="none",
        label="Visible GT objects",
        zorder=1,
    )
    axis.plot(
        x,
        gt_distribution,
        color=colors["GT"],
        linewidth=2.0,
        linestyle=(0, (4, 2.2)),
        zorder=5,
    )

    styles = {
        "AdaSlot": {"linestyle": "-", "marker": "o", "linewidth": 1.8},
        "MetaSlot": {"linestyle": "-", "marker": "^", "linewidth": 1.8},
        "CODA": {"linestyle": "-", "marker": None, "linewidth": 2.2},
        "DeOR": {"linestyle": "-", "marker": "o", "linewidth": 3.0},
    }
    model_handles = []
    for name in DEFAULT_SOURCES:
        style = styles[name]
        if name == "CODA":
            # CODA always activates exactly seven slots. Its 100% point is
            # labelled explicitly so the other distributions retain a useful
            # shared vertical scale.
            axis.vlines(7, 0, 0.335, color=colors[name], linewidth=style["linewidth"], zorder=4)
            axis.annotate(
                "100% at 7",
                xy=(7, 0.332),
                xytext=(7, 0.342),
                color=colors[name],
                fontsize=7.6,
                fontweight="bold",
                ha="center",
                va="bottom",
            )
        else:
            distribution = discrete_distribution(data[name]["pred"], args.tail_bin)
            axis.plot(
                x,
                distribution,
                color=colors[name],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=4.0 if name == "DeOR" else 3.4,
                markevery=1,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=6 if name == "DeOR" else 3,
            )
        model_handles.append(
            Line2D(
                [0], [0], color=colors[name], linewidth=style["linewidth"],
                marker=style["marker"], markersize=4.2,
                label=name,
            )
        )

    tick_positions = [0, 2, 4, 6, 8, 12, 16, args.tail_bin]
    tick_positions = list(dict.fromkeys(value for value in tick_positions if value <= args.tail_bin))
    tick_labels = [str(value) for value in tick_positions]
    tick_labels[-1] = f"{tick_positions[-1]}+"
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_xlim(-0.45, args.tail_bin + 0.45)
    axis.set_ylim(0, 0.35)
    axis.set_xlabel("Objects per image")
    axis.set_ylabel("Percentage of images")
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.grid(axis="y", color=colors["grid"], linewidth=0.75, alpha=0.9, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(colors["axis"])
    axis.spines["bottom"].set_color(colors["axis"])
    axis.tick_params(length=3, color=colors["axis"])

    gt_handle = Line2D(
        [0], [0], color=colors["GT"], linewidth=2.0, linestyle=(0, (4, 2.2)),
        label="COCO val2017 GT",
    )
    axis.legend(
        handles=[gt_handle, *model_handles],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        frameon=False,
        borderaxespad=0.0,
        labelspacing=0.25,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.17, right=0.985, top=0.98, bottom=0.20)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "COCO per-image object-count distributions across slot-based models",
        "Description": "Overlaid per-image active-slot and generated-object distributions on COCO val2017.",
    }
    save_options = {"bbox_inches": "tight", "pad_inches": 0.04}
    fig.savefig(
        args.output.with_suffix(".png"),
        dpi=300,
        metadata=metadata,
        **save_options,
    )
    fig.savefig(
        args.output.with_suffix(".pdf"),
        metadata={"Title": metadata["Title"], "Subject": metadata["Description"]},
        **save_options,
    )
    fig.savefig(args.output.with_suffix(".svg"), metadata=metadata, **save_options)
    plt.close(fig)

    print(f"Saved {args.output.with_suffix('.png')}")
    for name in DEFAULT_SOURCES:
        print(f"{name}: mean={data[name]['mean']:.4f}, MAE={data[name]['mae']:.4f}")
    print(f"GT: mean={gt_mean:.4f}, n={len(reference_gt)}")


if __name__ == "__main__":
    main()
