#!/usr/bin/env python3
"""Plot all currently available object-memory rFID and compositional FID results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MEMORY_SIZES = np.array([1, 4, 8, 16], dtype=float)
POSITIONS = np.arange(len(MEMORY_SIZES), dtype=float)
RFID = np.array([16.74650764465332, 16.602493286132812, 13.428787231445312, 13.499321937561035])
COMP_FID = np.array([22.709266662597656, 22.808855056762695, 21.039810180664062, 20.616315841674805])
OUTPUT = Path("outputs/paper_figures/object_memory_ablation_fid_mixed")


def annotate(ax, xs, ys, *, offsets):
    for x, y, offset in zip(xs, ys, offsets):
        ax.annotate(
            f"{y:.2f}",
            (x, y),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            va="bottom" if offset[1] >= 0 else "top",
            fontsize=18,
        )


def main():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 20,
            "axes.labelsize": 24,
            "axes.titlesize": 20,
            "legend.fontsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.linewidth": 1.3,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 3.5), constrained_layout=True)
    ax.set_xticks(POSITIONS)
    ax.set_xticklabels(["1", "4", "8", "16"])

    ax.plot(
        POSITIONS,
        RFID,
        color="#276FBF",
        marker="o",
        markersize=10,
        linewidth=3.0,
        label="rFID",
        zorder=3,
    )
    ax.plot(
        POSITIONS,
        COMP_FID,
        color="#D97706",
        marker="s",
        markersize=9.5,
        linewidth=3.0,
        linestyle="--",
        label="Compositional FID",
        zorder=3,
    )

    annotate(ax, POSITIONS, RFID, offsets=[(0, 9), (0, 9), (0, 9), (0, 9)])
    annotate(ax, POSITIONS, COMP_FID, offsets=[(0, -11), (0, 9), (0, 9), (0, -11)])

    ax.set_xlim(-0.22, 3.22)
    ax.set_ylim(12.6, 23.75)
    ax.set_xlabel("Visual memories per object")
    ax.set_ylabel("FID $\downarrow$")
    ax.grid(axis="y", color="#D0D0D0", linewidth=1.0, alpha=0.75)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        columnspacing=2.0,
        handlelength=2.2,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    title = "Object Memory Capacity vs. Generation Quality"
    description = "Available O1, O4, O8, and O16 results with registers fixed at 32."
    fig.savefig(
        OUTPUT.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    fig.savefig(
        OUTPUT.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Title": title, "Subject": description},
    )
    fig.savefig(
        OUTPUT.with_suffix(".svg"),
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    plt.close(fig)

    for suffix in (".png", ".pdf", ".svg"):
        print(OUTPUT.with_suffix(suffix))


if __name__ == "__main__":
    main()
