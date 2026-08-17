from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def _set_paper_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="paper",
        palette="colorblind",
        font_scale=1.0,
    )
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 11.5,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "lines.linewidth": 1.0,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.6,
        }
    )


BASE = Path("scenarios/scenario_grocery_retailer/outputs/multirun/paper_results")
OUTPUT = Path("scenarios/scenario_grocery_retailer/outputs/combined_operational_decision_makespan.pdf")

VARIANTS = [
    ("sshape", "SShape"),
    ("return", "RET"),
    ("largest_gap", "LG"),
    ("midpoint", "MP"),
    ("nearest_neighbour", "NN"),
    ("adaptive_routing", "Portfolio"),
]


def _load_tracker(variant_key: str) -> dict:
    path = BASE / variant_key / "makespan" / "kpis" / "tracker.json"
    with open(path) as f:
        return json.load(f)


def _plot_operational_behavior(axes, palette) -> None:
    for i, (variant_key, _) in enumerate(VARIANTS):
        tracker = _load_tracker(variant_key)
        color = palette[i]

        dock = tracker["dock_utilization"]
        ax_dock, ax_mk, ax_ut = axes
        ax_dock.step(
            [d[0] for d in dock], [d[1] for d in dock], where="post", color=color
        )

        mk = tracker["avg_makespan"]
        ax_mk.step(
            [m[0] for m in mk], [m[1] for m in mk], where="post", color=color
        )

        ut = tracker["avg_utilization"]
        ax_ut.step(
            [u[0] for u in ut], [u[1] for u in ut], where="post", color=color
        )

    ax_dock, ax_mk, ax_ut = axes
    ax_dock.axhline(y=98, color="red", linestyle="--", linewidth=1.0)

    ax_dock.set_ylabel("Dock\npallets")
    ax_mk.set_ylabel("Avg. tour\nmakespan [s]")
    ax_ut.set_ylabel("Avg. picker\nutilization")
    ax_ut.set_xlabel("Time [s]")

    ax_dock.set_title("(a) Operational behavior")

    for ax in axes:
        ax.margins(x=0.01)
        ax.grid(True, axis="y", alpha=0.25)
        ax.grid(False, axis="x")

    ax_dock.yaxis.set_major_locator(MultipleLocator(50))
    ax_mk.yaxis.set_major_locator(MultipleLocator(500))
    ax_ut.yaxis.set_major_locator(MultipleLocator(0.25))

    ymin_ut = ax_ut.get_ylim()[0]
    if ymin_ut > 0:
        ax_ut.set_ylim(bottom=0)

    plt.setp(ax_dock.get_xticklabels(), visible=False)
    plt.setp(ax_mk.get_xticklabels(), visible=False)


def _plot_portfolio_decisions(ax, palette) -> None:
    path = BASE / "adaptive_routing" / "makespan" / "decisions.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)

    orsp = [x for x in data["decisions"] if x[0] == "ORSP"]

    x = list(range(len(orsp)))
    y = [d[6] * 1000 for d in orsp]

    colors = [
        palette[0] if d[2].split("_")[2] == "SShape" else palette[1] for d in orsp
    ]

    ax.scatter(x, y, c=colors, s=4 * math.pi, zorder=2, edgecolors="none")

    ax.set_xlabel("Decision index")
    ax.set_ylabel("Decision time [ms]")
    ax.set_title("(b) Portfolio ORSP decisions")

    ax.text(
        0.02,
        0.04,
        f"{len(orsp)} decisions",
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=11.5,
    )

    handles = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=palette[0],
            markersize=5, label="SShape_EDD",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=palette[1],
            markersize=5, label="NN_EDD",
        ),
    ]
    ax.legend(handles=handles, frameon=True, loc="upper right")

    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")


def _add_bottom_legend(fig, palette) -> None:
    handles = []
    for i, (_, label) in enumerate(VARIANTS):
        handles.append(
            Line2D([0], [0], color=palette[i], linewidth=1.0, label=label)
        )
    handles.append(
        Line2D(
            [0], [0], color="red", linestyle="--", linewidth=1.0,
            label="Dock Capacity (98)",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        frameon=True,
        bbox_to_anchor=(0.36, 0.01),
        handlelength=1.33,
        handletextpad=0.71,
        columnspacing=1.28,
    )


def main() -> None:
    _set_paper_style()
    palette = sns.color_palette("colorblind", 6)

    fig = plt.figure(figsize=(10.0, 4.53))
    gs = GridSpec(
        3, 2, figure=fig, width_ratios=[2, 1], height_ratios=[1, 1, 1],
        hspace=0.28, wspace=0.28,
    )

    ax_dock = fig.add_subplot(gs[0, 0])
    ax_mk = fig.add_subplot(gs[1, 0], sharex=ax_dock)
    ax_ut = fig.add_subplot(gs[2, 0], sharex=ax_dock)
    ax_dec = fig.add_subplot(gs[:, 1])

    _plot_operational_behavior([ax_dock, ax_mk, ax_ut], palette)
    _plot_portfolio_decisions(ax_dec, palette)
    _add_bottom_legend(fig, palette)

    fig.subplots_adjust(
        left=0.12, right=0.99, top=0.94, bottom=0.23,
        hspace=0.30, wspace=0.28,
    )
    fig.savefig(OUTPUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
