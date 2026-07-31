"""Create compact figures for the adaptive-rollout confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONTROL = "cta-exact-fixed-control"
PAPER = "paper-neural-optimized"


def quality_cost_figure(report: dict, output: Path) -> None:
    dimensions = ("2", "4", "8", "16")
    x = np.arange(len(dimensions))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = ("#2563eb", "#ea580c")
    labels = ("ED2", "held-out SW1")
    for axis, comparator, title in (
        (axes[0], CONTROL, "Adaptive / prior control"),
        (axes[1], PAPER, "Adaptive / paper comparator"),
    ):
        for offset, metric, color, label in zip(
            (-width / 2, width / 2),
            ("ed2", "heldout_sw1"),
            colors,
            labels,
            strict=True,
        ):
            values = [
                report["quality"][comparator][metric]["by_dimension"][dimension][
                    "geometric_mean_ratio"
                ]
                for dimension in dimensions
            ]
            bars = axis.bar(x + offset, values, width, color=color, label=label)
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(x, [f"{dimension}D" for dimension in dimensions])
        axis.set_ylabel("paired geometric ratio (lower is better)")
        axis.set_title(title)
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=0.2)

    cost = report["cost"][CONTROL]
    names = ("Wall time", "Projection work", "Sort work", "Kernel pairs")
    values = [
        cost["wall_seconds"]["geometric_mean_ratio"],
        cost["projection_scalar_products"]["geometric_mean_ratio"],
        cost["sort_work"]["geometric_mean_ratio"],
        cost["paper_kernel_pairs"]["geometric_mean_ratio"],
    ]
    bars = axes[2].bar(
        np.arange(len(names)), values, color="#64748b", width=0.66
    )
    axes[2].bar_label(bars, fmt="%.2fx", fontsize=8, padding=2)
    axes[2].axhline(1.0, color="black", linewidth=1, linestyle="--")
    axes[2].set_xticks(
        np.arange(len(names)), names, rotation=25, ha="right"
    )
    axes[2].set_ylabel("adaptive / prior control")
    axes[2].set_title("Measured training cost")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Dimension-adaptive persistent transport: unseen 32-target confirmation",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rare_core_figure(report: dict, output: Path) -> None:
    rare = report["rare_diagnostics_vs_fixed"]
    dimensions = ("8", "16")
    x = np.arange(len(dimensions))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    core_candidate = [
        rare["rare_core_mass"]["by_dimension"][dimension]["candidate_mean"]
        for dimension in dimensions
    ]
    core_control = [
        rare["rare_core_mass"]["by_dimension"][dimension]["comparator_mean"]
        for dimension in dimensions
    ]
    for offset, values, label, color in (
        (-width / 2, core_control, "prior control", "#94a3b8"),
        (width / 2, core_candidate, "adaptive rollout", "#2563eb"),
    ):
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    axes[0].set_xticks(x, [f"{dimension}D" for dimension in dimensions])
    axes[0].set_ylabel("output mass inside target-calibrated rare core")
    axes[0].set_title("Genuine rare-core occupancy")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    error_candidate = [
        rare["rare_mass_error"]["by_dimension"][dimension]["candidate_mean"]
        for dimension in dimensions
    ]
    error_control = [
        rare["rare_mass_error"]["by_dimension"][dimension]["comparator_mean"]
        for dimension in dimensions
    ]
    for offset, values, label, color in (
        (-width / 2, error_control, "prior control", "#94a3b8"),
        (width / 2, error_candidate, "adaptive rollout", "#ea580c"),
    ):
        bars = axes[1].bar(x + offset, values, width, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.4f", fontsize=8, padding=2)
    axes[1].set_xticks(x, [f"{dimension}D" for dimension in dimensions])
    axes[1].set_ylabel("absolute rare nearest-component mass error")
    axes[1].set_title("Rare-mass calibration (lower is better)")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("Rare-GMM mechanism diagnostics on unseen targets", fontsize=13)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path(__file__).with_name(
            "adaptive_rollout_confirmation_analysis.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name(
            "adaptive_rollout_confirmation_figures"
        ),
    )
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    quality_cost_figure(report, args.output_dir / "quality_and_cost.png")
    rare_core_figure(report, args.output_dir / "rare_core_mechanism.png")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()

