from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PLOT_DATA = [
    {"scheme": 0, "flops": 3.46, "acc": 96.29, "remaining": 25},
    {"scheme": 1, "flops": 3.16, "acc": 96.31, "remaining": 23},
    {"scheme": 2, "flops": 3.16, "acc": 95.96, "remaining": 23},
    {"scheme": 3, "flops": 3.16, "acc": 96.28, "remaining": 23},
    {"scheme": 4, "flops": 2.86, "acc": 95.43, "remaining": 21},
    {"scheme": 5, "flops": 2.86, "acc": 95.91, "remaining": 21},
    {"scheme": 6, "flops": 2.57, "acc": 94.86, "remaining": 19},
    {"scheme": 7, "flops": 2.57, "acc": 95.12, "remaining": 19},
    {"scheme": 8, "flops": 2.57, "acc": 94.78, "remaining": 19},
    {"scheme": 9, "flops": 1.86, "acc": 93.97, "remaining": 14},
    {"scheme": 10, "flops": 1.86, "acc": 94.23, "remaining": 14},
    {"scheme": 11, "flops": 1.86, "acc": 93.86, "remaining": 14},
    {"scheme": 12, "flops": 2.00, "acc": 93.81, "remaining": 15},
    {"scheme": 13, "flops": 1.58, "acc": 94.04, "remaining": 12},
    {"scheme": 14, "flops": 1.58, "acc": 93.81, "remaining": 12},
    {"scheme": 15, "flops": 1.30, "acc": 93.73, "remaining": 10},
    {"scheme": 16, "flops": 1.30, "acc": 94.12, "remaining": 10},
    {"scheme": 17, "flops": 1.30, "acc": 93.67, "remaining": 10},
    {"scheme": 18, "flops": 1.30, "acc": 93.92, "remaining": 10},
    {"scheme": 19, "flops": 1.17, "acc": 93.70, "remaining": 9},
]

POINT_COLOR = "#4C78A8"
FRONTIER_COLOR = "#D64F45"


def is_pareto_optimal(candidate: dict, all_points: list[dict]) -> bool:
    for other in all_points:
        if other["scheme"] == candidate["scheme"]:
            continue
        no_worse = other["flops"] <= candidate["flops"] and other["acc"] >= candidate["acc"]
        strictly_better = other["flops"] < candidate["flops"] or other["acc"] > candidate["acc"]
        if no_worse and strictly_better:
            return False
    return True


def build_frontier(points: list[dict]) -> list[dict]:
    frontier = [p for p in points if is_pareto_optimal(p, points)]
    return sorted(frontier, key=lambda item: item["flops"])


def annotate_frontier_ids(ax: plt.Axes, frontier: list[dict]) -> None:
    for point in frontier:
        x = point["flops"]
        if point["scheme"] == 19:
            x -= 0.03
        ax.text(
            x,
            point["acc"] + 0.07,
            str(point["scheme"]),
            fontsize=11,
            fontweight="semibold",
            color="#1F2D3A",
            ha="center",
            va="bottom",
            zorder=6,
        )


def make_plot(output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.edgecolor": "#4A5560",
            "axes.linewidth": 0.9,
            "xtick.color": "#2F3A45",
            "ytick.color": "#2F3A45",
            "grid.color": "#D7DEE5",
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frontier = build_frontier(PLOT_DATA)

    xvals = [p["flops"] for p in PLOT_DATA]
    yvals = [p["acc"] for p in PLOT_DATA]
    xmin, xmax = min(xvals) - 0.18, max(xvals) + 0.20
    ymin, ymax = min(yvals) - 0.25, max(yvals) + 0.35

    fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.scatter(
        [p["flops"] for p in PLOT_DATA],
        [p["acc"] for p in PLOT_DATA],
        s=46,
        marker="o",
        c=POINT_COLOR,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.7,
        zorder=3,
    )

    ax.plot(
        [p["flops"] for p in frontier],
        [p["acc"] for p in frontier],
        color=FRONTIER_COLOR,
        lw=2.2,
        linestyle="-",
        zorder=4,
    )

    ax.scatter(
        [p["flops"] for p in frontier],
        [p["acc"] for p in frontier],
        s=58,
        marker="o",
        c=FRONTIER_COLOR,
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    annotate_frontier_ids(ax, frontier)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("GFLOPS per sample")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.grid(True, linestyle="--", linewidth=0.65, alpha=0.65)
    ax.set_axisbelow(True)

    legend_items = [
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor=POINT_COLOR,
            markeredgecolor="white", markersize=7.5, label="Pruning configuration"
        ),
        Line2D([0], [0], color=FRONTIER_COLOR, lw=2.2, label="Pareto frontier"),
    ]
    legend = ax.legend(
        handles=legend_items,
        loc="lower right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.6,
    )
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_edgecolor("#C7D0D9")

    fig.tight_layout()

    png_path = output_dir / "legacy_joint_pruning_pareto.png"
    pdf_path = output_dir / "legacy_joint_pruning_pareto.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=400)
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to {png_path}")
    print(f"Saved figure to {pdf_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    make_plot(repo_root / "paper" / "figures")
