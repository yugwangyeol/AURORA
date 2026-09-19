#!/usr/bin/env python3
"""Paper plot for reconstruction/compositional FID versus memory capacity.

Only the directly comparable Memory-8 and Memory-16, Register-32 checkpoints
are populated by default. The complete 1/2/4/8/16 x-axis is retained so later
capacity results and compositional FID can be added without redesigning the
figure.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MEMORY_SIZES = np.array([1, 2, 4, 8, 16], dtype=float)
DEFAULT_RFID = {8: 13.428787231445312, 16: 13.499321937561035}


def values_at_sizes(values):
    return np.array([values.get(int(size), np.nan) for size in MEMORY_SIZES])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_figures/memory_capacity_fid"),
        help="Output stem; PNG, PDF, and SVG are written.",
    )
    parser.add_argument("--rfid-8", type=float, default=DEFAULT_RFID[8])
    parser.add_argument("--rfid-16", type=float, default=DEFAULT_RFID[16])
    parser.add_argument("--compositional-fid-8", type=float)
    parser.add_argument("--compositional-fid-16", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    rfid = {8: args.rfid_8, 16: args.rfid_16}
    compositional = {
        size: value
        for size, value in (
            (8, args.compositional_fid_8),
            (16, args.compositional_fid_16),
        )
        if value is not None
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    ax.set_xscale("log", base=2)
    ax.set_xticks(MEMORY_SIZES)
    ax.set_xticklabels([str(int(value)) for value in MEMORY_SIZES])

    rfid_values = values_at_sizes(rfid)
    ax.plot(
        MEMORY_SIZES,
        rfid_values,
        color="#276FBF",
        marker="o",
        markersize=7,
        linewidth=2.0,
        label="Reconstruction FID",
        zorder=3,
    )
    for size, value in rfid.items():
        ax.annotate(
            f"{value:.2f}",
            (size, value),
            xytext=(0, 9 if size == 8 else -16),
            textcoords="offset points",
            ha="center",
            va="bottom" if size == 8 else "top",
            fontsize=10,
        )

    if compositional:
        comp_values = values_at_sizes(compositional)
        ax.plot(
            MEMORY_SIZES,
            comp_values,
            color="#D97706",
            marker="s",
            markersize=6.5,
            linewidth=2.0,
            linestyle="--",
            label="Compositional FID",
            zorder=3,
        )
        for size, value in compositional.items():
            ax.annotate(
                f"{value:.2f}",
                (size, value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    all_values = list(rfid.values()) + list(compositional.values())
    low, high = min(all_values), max(all_values)
    span = max(high - low, 0.20)
    padding = max(0.18 * span, 0.08)
    ax.set_ylim(low - padding, high + padding)
    ax.set_xlim(0.8, 20)
    ax.set_xlabel("Object memories per owner")
    ax.set_ylabel("FID (lower is better)")
    ax.set_title("Memory Capacity vs. Generation Quality", pad=10)
    ax.text(
        0.01,
        0.98,
        "Registers fixed at 32",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    title = "Memory Capacity vs. Generation Quality"
    description = (
        "Memory-8 and Memory-16 use Register-32. Native rFID values come "
        "from the 4,720-image autoregressive evaluation."
    )
    fig.savefig(
        args.output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    fig.savefig(
        args.output.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Title": title, "Subject": description},
    )
    fig.savefig(
        args.output.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    plt.close(fig)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
