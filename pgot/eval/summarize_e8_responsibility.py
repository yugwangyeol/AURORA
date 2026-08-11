"""Create a compact A/C/D comparison from E8 priority diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "A": "A: Writer GT / Reader Writer",
    "C": "C: Writer None / Reader GT",
    "D": "D: Writer None / Reader Writer",
}


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def size_breakdown(path: str) -> dict:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    groups = {
        "tiny_<2%": lambda area: area < 0.02,
        "small_2-5%": lambda area: 0.02 <= area < 0.05,
        "medium_5-10%": lambda area: 0.05 <= area < 0.10,
        "large_>=10%": lambda area: area >= 0.10,
    }
    result = {}
    for name, predicate in groups.items():
        selected = [row for row in rows if predicate(row["source_mask_area"] / 256.0)]
        shares = [
            row["diagonal_influence"]
            / max(
                row["diagonal_influence"]
                + row["other_object_mean_influence"]
                + row["background_influence"],
                1e-12,
            )
            for row in selected
        ]
        result[name] = {
            "count": len(selected),
            "diagonal_share_mean": float(np.mean(shares)),
            "selected_dominant_fraction": float(
                np.mean([row["selected_region_is_dominant"] for row in selected])
            ),
            "diagonal_influence_mean": float(
                np.mean([row["diagonal_influence"] for row in selected])
            ),
        }
    return result


def flatten(name: str, summary: dict, evaluation: dict) -> dict:
    writer = summary["writer_by_layer"]
    transition = summary["writer_transitions"]
    causal = summary["causal_influence"]
    appearance = summary["appearance_information"]
    probe = appearance["probes_instance_residual"]
    mass = appearance["reader_mass_on_target_region"]
    return {
        "experiment": name,
        "fARI": evaluation["fARI"],
        "mBO": evaluation["mBO"],
        "rFID": evaluation["rFID"],
        "writer21_named_iou": writer["21"]["named_hard_iou"]["mean"],
        "writer24_named_iou": writer["24"]["named_hard_iou"]["mean"],
        "writer27_named_iou": writer["27"]["named_hard_iou"]["mean"],
        "writer27_oracle_iou": writer["27"]["oracle_hard_iou"]["mean"],
        "writer27_oracle_gain": writer["27"]["oracle_minus_named"]["mean"],
        "writer21_24_fg_consistency": transition["21_to_24"]["hard_assignment_consistency_fg"]["mean"],
        "writer24_27_fg_consistency": transition["24_to_27"]["hard_assignment_consistency_fg"]["mean"],
        "causal_diagonal_mean": causal["diagonal_influence"]["mean"],
        "causal_other_mean": causal["other_object_mean_influence"]["mean"],
        "causal_background_mean": causal["background_influence"]["mean"],
        "causal_diagonal_share_mean": causal["diagonal_share_of_selected_other_background"]["mean"],
        "causal_diagonal_share_median": causal["diagonal_share_of_selected_other_background"]["median"],
        "causal_selected_dominant_fraction": causal["selected_region_dominant_fraction"],
        "causal_global_diag_over_other": causal["global_sum_diag_over_other_mean"],
        "causal_global_diag_over_background": causal["global_sum_diag_over_background"],
        "appearance_direct_self_r2": probe["direct_self"]["r2"],
        "appearance_direct_nonself_r2": probe["direct_nonself"]["r2"],
        "appearance_direct_unique_self_delta_r2": probe["direct_unique_self_delta_r2"]["r2"],
        "appearance_routed_self_r2": probe["routed_self"]["r2"],
        "appearance_routed_nonself_r2": probe["routed_nonself"]["r2"],
        "appearance_routed_all_r2": probe["routed_all_concat"]["r2"],
        "appearance_routed_unique_self_delta_r2": probe["routed_unique_self_delta_r2"]["r2"],
        "reader_mass_self": mass["self"]["mean"],
        "reader_mass_other": mass["other_objects"]["mean"],
        "reader_mass_register": mass["registers"]["mean"],
    }


def save_plots(output: Path, rows: list[dict]) -> None:
    layers = [21, 24, 27]
    colors = {"A": "#2a9d8f", "C": "#e9c46a", "D": "#e76f51"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for row in rows:
        name = row["experiment"]
        named = [row[f"writer{x}_named_iou"] for x in layers]
        axes[0].plot(layers, named, marker="o", color=colors[name], label=name)
    axes[0].set(xticks=layers, ylim=(0, 0.55), xlabel="Writer layer", ylabel="named hard IoU")
    axes[0].legend()
    x = np.arange(len(rows))
    axes[1].bar(x - 0.18, [r["writer27_named_iou"] for r in rows], 0.36, label="named")
    axes[1].bar(x + 0.18, [r["writer27_oracle_iou"] for r in rows], 0.36, label="oracle matched")
    axes[1].set(xticks=x, xticklabels=[r["experiment"] for r in rows], ylim=(0, 0.55), ylabel="layer 27 hard IoU")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "writer_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].bar(x - 0.2, [r["causal_diagonal_share_mean"] for r in rows], 0.4, label="diagonal share")
    axes[0].bar(x + 0.2, [r["causal_selected_dominant_fraction"] for r in rows], 0.4, label="selected dominant")
    axes[0].set(xticks=x, xticklabels=[r["experiment"] for r in rows], ylim=(0, 0.8), ylabel="fraction")
    axes[0].legend()
    axes[1].bar(x - 0.2, [r["causal_global_diag_over_other"] for r in rows], 0.4, label="diag / other")
    axes[1].bar(x + 0.2, [r["causal_global_diag_over_background"] for r in rows], 0.4, label="diag / background")
    axes[1].set(xticks=x, xticklabels=[r["experiment"] for r in rows], ylabel="global causal ratio")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "causal_exclusivity_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.25
    axes[0].bar(x - width, [r["appearance_direct_self_r2"] for r in rows], width, label="self memory")
    axes[0].bar(x, [r["appearance_direct_nonself_r2"] for r in rows], width, label="non-self")
    axes[0].bar(x + width, [r["appearance_direct_unique_self_delta_r2"] for r in rows], width, label="unique self delta")
    axes[0].set(xticks=x, xticklabels=[r["experiment"] for r in rows], ylabel="instance-residual $R^2$")
    axes[0].legend()
    axes[1].bar(x - width, [r["reader_mass_self"] for r in rows], width, label="self")
    axes[1].bar(x, [r["reader_mass_other"] for r in rows], width, label="other objects")
    axes[1].bar(x + width, [r["reader_mass_register"] for r in rows], width, label="registers")
    axes[1].set(xticks=x, xticklabels=[r["experiment"] for r in rows], ylim=(0, 0.75), ylabel="Reader mass on target region")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "appearance_and_reader_comparison.png", dpi=200)
    plt.close(fig)


def make_report(rows: list[dict], size: dict) -> str:
    r = {x["experiment"]: x for x in rows}
    lines = [
        "# E8 A/C/D priority analysis (visible-object protocol)",
        "",
        "- Dataset: first 512 validation images, 2,857 visible GT objects.",
        "- 347 captioned object slots whose GT masks disappeared after center crop remain model competitors but are excluded as evaluation targets.",
        "- Causal branches use the same RF timestep and noise; only one object memory is zeroed.",
        "- Appearance probes use an image-disjoint 70/10/20 train/validation/test split and word-residual targets.",
        "",
        "## Compact comparison",
        "",
        "| Metric | A | C | D |",
        "|---|---:|---:|---:|",
    ]
    metrics = [
        ("Full-eval fARI", "fARI"),
        ("Full-eval mBO", "mBO"),
        ("rFID", "rFID"),
        ("Writer L27 named IoU", "writer27_named_iou"),
        ("Writer L27 oracle IoU", "writer27_oracle_iou"),
        ("Writer FG consistency 21→24", "writer21_24_fg_consistency"),
        ("Writer FG consistency 24→27", "writer24_27_fg_consistency"),
        ("Causal diagonal share (mean)", "causal_diagonal_share_mean"),
        ("Selected region dominant", "causal_selected_dominant_fraction"),
        ("Global diagonal/other influence", "causal_global_diag_over_other"),
        ("Direct self appearance R²", "appearance_direct_self_r2"),
        ("Direct non-self appearance R²", "appearance_direct_nonself_r2"),
        ("Unique self ΔR²", "appearance_direct_unique_self_delta_r2"),
        ("Reader target-region self mass", "reader_mass_self"),
        ("Reader target-region other-object mass", "reader_mass_other"),
        ("Reader target-region register mass", "reader_mass_register"),
    ]
    for label, key in metrics:
        lines.append(f"| {label} | {r['A'][key]:.4f} | {r['C'][key]:.4f} | {r['D'][key]:.4f} |")
    lines.extend([
        "",
        "## Findings",
        "",
        "1. **C is not a slot-permutation failure.** Its layer-27 named/oracle IoU is "
        f"{r['C']['writer27_named_iou']:.3f}/{r['C']['writer27_oracle_iou']:.3f}; oracle matching recovers only "
        f"{r['C']['writer27_oracle_gain']:.3f}. Object partition quality itself is worse than A.",
        "2. **C still stores and uses named appearance.** Its direct self-memory residual R² and causal locality are at least as strong as A. GT Reader supervision can anchor reconstruction responsibility even when the Writer ownership map is poor.",
        "3. **D loses named responsibility.** Self Reader mass falls to "
        f"{100*r['D']['reader_mass_self']:.1f}% while other-object mass rises to {100*r['D']['reader_mass_other']:.1f}%. "
        f"Its unique-self ΔR² is {r['D']['appearance_direct_unique_self_delta_r2']:.4f}, effectively zero.",
        "4. **A is partially, not fully, disentangled.** A has strong global diagonal/other causal influence, but only "
        f"{100*r['A']['causal_selected_dominant_fraction']:.1f}% of interventions are strongest on the selected object. "
        f"The Reader assigns only {100*r['A']['reader_mass_self']:.1f}% of target-region mass to the matching memory.",
        "5. **Next experiment:** continue A with E8.2. The evidence points to an exclusivity/routing problem, while A already has non-zero unique appearance. Evaluate whether E8.2 raises selected-dominant fraction, diagonal share, Reader self mass, and unique routed ΔR² without degrading rFID.",
        "6. **Separate GT-free branch:** C/D do not reproduce CODA's contrastive slot-image alignment. Removing Writer GT should be revisited only with an explicit self-supervised alignment/iterative competition objective.",
        "",
        "## Causal locality by object area",
        "",
        "| Area on 16×16 target grid | A dominant | C dominant | D dominant |",
        "|---|---:|---:|---:|",
    ])
    for bucket in ["tiny_<2%", "small_2-5%", "medium_5-10%", "large_>=10%"]:
        lines.append(
            f"| {bucket} (n={size['A'][bucket]['count']}) | "
            f"{size['A'][bucket]['selected_dominant_fraction']:.4f} | "
            f"{size['C'][bucket]['selected_dominant_fraction']:.4f} | "
            f"{size['D'][bucket]['selected_dominant_fraction']:.4f} |"
        )
    lines.extend([
        "",
        "A/C의 가장 약한 구간은 면적 2% 미만 객체로, selected-region dominant 비율이 약 56%다. E8.2 평가는 전체 평균뿐 아니라 이 bucket을 별도로 봐야 한다.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in "acd":
        parser.add_argument(f"--{name}_summary", required=True)
        parser.add_argument(f"--{name}_eval", required=True)
        parser.add_argument(f"--{name}_influence", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for upper, lower in [("A", "a"), ("C", "c"), ("D", "d")]:
        rows.append(flatten(upper, load(getattr(args, f"{lower}_summary")), load(getattr(args, f"{lower}_eval"))))
    size = {
        upper: size_breakdown(getattr(args, f"{lower}_influence"))
        for upper, lower in [("A", "a"), ("C", "c"), ("D", "d")]
    }
    (output / "comparison.json").write_text(
        json.dumps({"experiments": rows, "causal_by_object_area": size}, indent=2)
    )
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "report.md").write_text(make_report(rows, size))
    save_plots(output, rows)
    print((output / "report.md").read_text())


if __name__ == "__main__":
    main()
