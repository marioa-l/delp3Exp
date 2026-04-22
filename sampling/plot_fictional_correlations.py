"""
Fictional correlation matrices illustrating expected results.

Generates two publication-ready figures (SSS and MMM datasets) showing
the correlation between structural model features and sampling performance
metrics for both the Worlds and Programs sampling approaches.

The data is synthetic but designed to reflect the theoretical intuition:
  annots_minus_em = n_annots - n_em_vars

  > 0  →  more programs than worlds  →  World Sampling wins
  < 0  →  more worlds than programs  →  Program Sampling wins

The difference between SSS and MMM:
  - SSS: smaller programs, balanced annots_minus_em → mixed correlations
  - MMM: larger programs, typically annots_minus_em >> 0 → World Sampling
         dominates and correlations are much stronger

Output (in same directory as this script):
  fictional_correlations_SSS.pdf
  fictional_correlations_MMM.pdf

Usage:
    cd clean_workspace
    python sampling/plot_fictional_correlations.py [--output_dir <path>]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style constants ────────────────────────────────────────────────────────────
FONT = "DejaVu Sans"
DPI = 300
plt.rcParams["font.family"] = FONT

FEATURES = ["n_em_vars", "n_annots", "annots_minus_em", "n_rules", "exact_width"]
FEATURE_LABELS = [
    "$n_{EM}$",
    "$n_{annots}$",
    "$n_{annots} - n_{EM}$",
    "$n_{rules}$",
    "exact width",
]
METRICS = ["quality", "solver\ntime", "solver\ncalls"]
METRIC_LABELS = ["Quality\nMetric", "Solver\nTime", "Solver\nCalls"]


# ── Synthetic correlation matrices ────────────────────────────────────────────
# Each matrix is (n_features × n_metrics).
# Values are fictional Pearson r, chosen to tell the expected scientific story.

# SSS dataset — small, simple programs, balanced annots_minus_em.
# Correlations are weaker and more mixed.
SSS_WORLDS = np.array([
    # quality  time   calls
    [-0.42,   +0.53,  +0.47],   # n_em_vars   (more worlds → harder to sample → ↓ quality)
    [+0.28,   +0.31,  +0.29],   # n_annots    (more annots → worlds relative advantage ↑)
    [+0.51,   -0.19,  -0.21],   # annots_minus_em (KEY: positive → worlds wins)
    [-0.15,   +0.44,  +0.40],   # n_rules     (more rules → slower DeLP)
    [+0.33,   -0.08,  -0.12],   # exact_width (wider interval → more room to improve)
])

SSS_PROGRAMS = np.array([
    # quality  time   calls
    [+0.38,   +0.29,  +0.31],   # n_em_vars   (more worlds → programs relatively easier)
    [-0.45,   +0.58,  +0.52],   # n_annots    (more annots → harder for programs → ↓ quality)
    [-0.49,   +0.22,  +0.24],   # annots_minus_em (KEY: negative → programs loses)
    [-0.18,   +0.46,  +0.43],   # n_rules
    [+0.30,   -0.06,  -0.10],   # exact_width
])

# MMM dataset — large programs, annots_minus_em typically >> 0.
# Correlations are much stronger; World Sampling dominates.
MMM_WORLDS = np.array([
    # quality  time   calls
    [-0.71,   +0.74,  +0.69],   # n_em_vars
    [+0.62,   +0.48,  +0.51],   # n_annots
    [+0.83,   -0.31,  -0.28],   # annots_minus_em (STRONG positive → worlds dominates)
    [-0.22,   +0.67,  +0.64],   # n_rules
    [+0.55,   -0.14,  -0.17],   # exact_width
])

MMM_PROGRAMS = np.array([
    # quality  time   calls
    [+0.61,   +0.38,  +0.42],   # n_em_vars
    [-0.74,   +0.77,  +0.73],   # n_annots   (STRONG negative → programs loses on MMM)
    [-0.87,   +0.35,  +0.33],   # annots_minus_em (STRONG negative)
    [-0.19,   +0.69,  +0.65],   # n_rules
    [+0.48,   -0.11,  -0.15],   # exact_width
])

DATASET_MATRICES = {
    "SSS": {"Worlds\nSampling": SSS_WORLDS, "Programs\nSampling": SSS_PROGRAMS},
    "MMM": {"Worlds\nSampling": MMM_WORLDS, "Programs\nSampling": MMM_PROGRAMS},
}

DATASET_SUBTITLES = {
    "SSS": "Small/Simple Programs — balanced $n_{annots} - n_{EM}$, weaker correlations",
    "MMM": "Markov Mesoscale Models — $n_{annots} \\gg n_{EM}$, World Sampling dominates",
}


def draw_heatmap(ax, matrix, row_labels, col_labels, title, vmin=-1, vmax=1):
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontfamily=FONT)
    ax.set_yticklabels(row_labels, fontsize=11, fontfamily=FONT)
    ax.tick_params(which="both", length=0)

    # Annotate cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            color = "white" if abs(v) > 0.65 else "black"
            ax.text(
                j, i, f"{v:+.2f}",
                ha="center", va="center",
                fontsize=11, fontfamily=FONT,
                color=color, fontweight="bold",
            )

    # Highlight annots_minus_em row
    em_row = 2  # index of annots_minus_em
    for j in range(matrix.shape[1]):
        rect = plt.Rectangle(
            (j - 0.5, em_row - 0.5), 1, 1,
            linewidth=2.0, edgecolor="black", facecolor="none",
        )
        ax.add_patch(rect)

    ax.set_title(title, fontsize=13, fontfamily=FONT, pad=8, fontweight="bold")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return im


def plot_dataset_figure(dataset_name, output_path):
    approaches = DATASET_MATRICES[dataset_name]
    n_approaches = len(approaches)

    fig, axes = plt.subplots(
        1, n_approaches,
        figsize=(5.5 * n_approaches, 5.2),
        constrained_layout=True,
    )

    last_im = None
    for ax, (approach_name, matrix) in zip(axes, approaches.items()):
        last_im = draw_heatmap(
            ax, matrix,
            row_labels=FEATURE_LABELS,
            col_labels=METRIC_LABELS,
            title=approach_name,
        )

    # Shared colorbar
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.04)
    cbar.set_label("Pearson $r$", fontsize=12, fontfamily=FONT)
    cbar.ax.tick_params(labelsize=10)

    # Figure title
    fig.suptitle(
        f"Dataset: {dataset_name} — Correlation: Structural Features × Performance Metrics\n"
        r"$\bf{■}$" + f" highlighted row: $n_{{annots}} - n_{{EM}}$ (key predictor)",
        fontsize=14, fontfamily=FONT,
    )
    # Subtitle
    fig.text(
        0.5, -0.02,
        DATASET_SUBTITLES[dataset_name],
        ha="center", fontsize=11, fontfamily=FONT, style="italic",
    )

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


import matplotlib.ticker  # needed by draw_heatmap


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate fictional correlation heatmaps for SSS and MMM datasets."
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Directory to save PDFs (default: same folder as this script)",
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for dataset in ("SSS", "MMM"):
        out = os.path.join(args.output_dir, f"fictional_correlations_{dataset}.pdf")
        plot_dataset_figure(dataset, out)
