"""
Figures for the leaf-guided heuristic experiment.

Three modes of the method the tree recommends are run on each query under an
identical cap, so every comparison below is paired:

    plain     uniform sampling
    generic   bn_centrality (worlds) / bn_marginal (programs)
    guided    hub_strata (worlds) / usage_strata (programs), which reuse the
              attributes the tree already consulted

Quality and exactness have to be read together: a biased sampler can report a
tighter interval while reaching the exact interval far less often, and a panel
showing only quality would present that as an improvement.

Reads:
    results_tte/leaf_heuristics_v2/leaf_heuristic_results.csv

Writes PDF and PNG to --output_dir (default results_tte/figures/):
    leaf_heuristic_overview.{pdf,png}

Usage:
    python sampling/generate_leaf_heuristic_figures.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None

FONT = "DejaVu Sans"
LABEL, TITLE = 7.5, 8.5
C_PLAIN   = "#8C8C8C"
C_GENERIC = "#D55E00"
C_GUIDED  = "#0072B2"
C_OK      = "#27AE60"
C_BAD     = "#C0392B"
# generic is evaluated and reported in the text, but the figure contrasts the
# two families the paper's question is about: uniform draws against draws
# steered by the attributes the tree already consulted.
FAMS = [("random", "random", C_PLAIN),
        ("guided", "guided", C_GUIDED)]


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL, fontname=FONT)
    ax.set_title(title, fontsize=TITLE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=7)
    ax.minorticks_on()


def sem(s):
    s = s.dropna()
    return s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0


def panel_quality(ax, d):
    """Mean interval quality per family, overall and per method."""
    groups = [("all", d), ("worlds", d[d.method == "worlds"]),
              ("programs", d[d.method == "programs"])]
    x = np.arange(len(groups))
    w = 0.36
    for i, (key, lab, col) in enumerate(FAMS):
        vals = [g[f"{key}_quality"].mean() for _, g in groups]
        errs = [sem(g[f"{key}_quality"]) for _, g in groups]
        off = (i - 0.5) * w
        ax.bar(x + off, vals, w, yerr=errs, label=lab, color=col,
               edgecolor="white", error_kw=dict(lw=1.0, ecolor="#333333"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n(n={len(g)})" for lab, g in groups],
                       fontsize=7, fontname=FONT)
    ax.set_ylim(0, 1.28)
    style(ax, ylabel="Mean interval quality (1.0 = exact)",
          title="A")
    ax.legend(loc="upper center", fontsize=6, framealpha=0.9, ncol=3)


def panel_exact(ax, d):
    """How often each family actually reached the exact interval."""
    groups = [("all", d), ("worlds", d[d.method == "worlds"]),
              ("programs", d[d.method == "programs"])]
    x = np.arange(len(groups))
    w = 0.36
    for i, (key, lab, col) in enumerate(FAMS):
        fr = [g[f"{key}_time_to_exact"].notna().mean() for _, g in groups]
        cnt = [int(g[f"{key}_time_to_exact"].notna().sum()) for _, g in groups]
        off = (i - 0.5) * w
        ax.bar(x + off, fr, w, label=lab, color=col, edgecolor="white")
        for xi, v, c, (_, g) in zip(x + off, fr, cnt, groups):
            ax.text(xi, v + 0.02, str(c), ha="center", fontsize=5.5,
                    fontname=FONT, va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in groups], fontsize=7, fontname=FONT)
    ax.set_ylim(0, 1.24)
    style(ax, ylabel="Fraction reaching the exact interval",
          title="B")
    ax.legend(loc="upper right", fontsize=6, framealpha=0.9)


def panel_delta(ax, d):
    """Paired per-query delta against plain, by leaf."""
    d = d.copy()
    d["short"] = d.leaf.map({k: f"L{i+1}" for i, k in
                             enumerate(sorted(d.leaf.unique()))})
    order = sorted(d.short.unique())
    pos = np.arange(len(order))
    w = 0.36
    for i, (key, lab, col) in enumerate([("generic","generic",C_GENERIC), FAMS[1]]):
        meds, errs = [], []
        for s in order:
            g = d[d.short == s]
            delta = (g[f"{key}_quality"] - g["random_quality"]).dropna()
            meds.append(delta.mean())
            errs.append(sem(delta))
        off = (i - 0.5) * w
        ax.barh(pos + off, meds, w, xerr=errs, label=f"{lab} $-$ random",
                color=col, edgecolor="white",
                error_kw=dict(lw=0.9, ecolor="#333333"))
    ax.axvline(0, color="black", lw=1.0, ls=":")
    ax.set_yticks(pos)
    ax.set_yticklabels([f"{s}  (n={len(d[d.short==s])})" for s in order],
                       fontsize=7, fontname=FONT)
    style(ax, xlabel=r"mean quality delta vs random",
          title="C")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)


def panel_calls(ax, d):
    """Solver calls: a heuristic that buys quality with more calls is no win."""
    groups = [("worlds", d[d.method == "worlds"]),
              ("programs", d[d.method == "programs"])]
    x = np.arange(len(groups))
    w = 0.26
    for i, (key, lab, col) in enumerate(FAMS):
        vals = [g[f"{key}_delp_calls"].median() for _, g in groups]
        off = (i - 1) * w
        ax.bar(x + off, vals, w, label=lab, color=col, edgecolor="white")
        for xi, v in zip(x + off, vals):
            ax.text(xi, v * 1.02, f"{v:.0f}", ha="center", fontsize=9,
                    fontname=FONT)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in groups], fontsize=7, fontname=FONT)
    style(ax, ylabel="Median solver calls",
          title="D")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv",
                   default="results_tte/leaf_heuristics_v3/"
                           "leaf_heuristic_results.csv")
    p.add_argument("--output_dir", default="results_tte/figures")
    p.add_argument("--full", action="store_true",
                   help="Four-panel version for analysis; the default is the "
                        "column-width two-panel version used in the paper.")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    d = pd.read_csv(args.input_csv, sep=";")
    d = d[d["ok"] == True]  # noqa: E712
    print(f"Queries: {len(d)}  |  leaves: {d.leaf.nunique()}")

    if args.full:
        fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
        panel_quality(axes[0, 0], d)
        panel_exact(axes[0, 1], d)
        panel_delta(axes[1, 0], d)
        panel_calls(axes[1, 1], d)
    else:
        # Column width (345 pt = 4.79 in) so the figure is included at 1:1 and
        # the type keeps the size it is set at. Four panels do not fit that
        # width legibly, so the paper figure carries the two that make the
        # argument; the per-leaf deltas and the solver counts are in the text.
        fig, axes = plt.subplots(1, 2, figsize=(4.79, 2.6))
        panel_quality(axes[0], d)
        panel_exact(axes[1], d)

    # No suptitle: the caption carries the headline numbers.
    gd = (d["guided_quality"] - d["random_quality"]).dropna()
    gn = (d["generic_quality"] - d["random_quality"]).dropna()
    print(f"  caption numbers -> guided-random: median {gd.median():+.3f}, "
          f"mean {gd.mean():+.4f} +/- {sem(gd):.4f}")
    print(f"                     generic-random: median {gn.median():+.3f}, "
          f"mean {gn.mean():+.4f} +/- {sem(gn):.4f}")
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = os.path.join(args.output_dir, f"leaf_heuristic_overview.{ext}")
        fig.savefig(out, format=ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
