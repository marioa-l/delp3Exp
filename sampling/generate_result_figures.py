"""
Publication-quality figures for the tree-validation and heuristic experiments.

Reads:
    results_5min_val/analysis/validation_predictions.csv
    results_5min_val/analysis/validation_report.txt   (only for headline numbers)
    results_5min/heuristics/heuristic_results.csv

Writes PDFs to a target directory (default: results_5min/figures/):
    validation_per_config.pdf
    validation_confusion.pdf
    heuristic_per_region.pdf
    heuristic_delta_box.pdf
    heuristic_scatter.pdf

Usage:
    python sampling/generate_result_figures.py \
        --val_csv results_5min_val/analysis/validation_predictions.csv \
        --heur_csv results_5min/heuristics/heuristic_results.csv \
        --output_dir results_5min/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 15
COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"
COLOR_ABSTRACT = "#009E73"
COLOR_COMPARE  = "#CC79A7"


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.minorticks_on()


# ── Validation figures ──────────────────────────────────────────────────────
def fig_per_config(val_df: pd.DataFrame, out_path: str):
    """Grouped bar chart of comparison and abstract tree accuracy per config."""
    grp = val_df.groupby("config").agg(
        n=("winner", "size"),
        cmp=("match_cmp", "mean"),
        abs_=("match_abs", "mean"),
    ).reset_index()
    grp = grp.sort_values("n", ascending=False)
    configs = grp["config"].values
    x = np.arange(len(configs))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, grp["cmp"].values, width, label="Comparison tree",
           color=COLOR_COMPARE, edgecolor="white")
    ax.bar(x + width / 2, grp["abs_"].values, width, label="Abstract tree",
           color=COLOR_ABSTRACT, edgecolor="white")

    for i, (c, a, n) in enumerate(zip(grp["cmp"], grp["abs_"], grp["n"])):
        ax.text(i - width / 2, c + 0.02, f"{c:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(i + width / 2, a + 0.02, f"{a:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(i, -0.06, f"n={n}", ha="center", va="top",
                fontsize=8, color="dimgray", fontname=FONT)

    ax.axhline(0.5, color="gray", lw=0.6, ls=":", label="Chance level (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, fontsize=11, fontname=FONT)
    ax.set_ylim(-0.12, 1.15)
    style_ax(ax, xlabel="Configuration",
             ylabel="Accuracy on unseen models",
             title="Tree validation on 30 unseen models per configuration")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_confusion(val_df: pd.DataFrame, out_path: str):
    """2x2 confusion matrix for both trees side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, title in (
            (axes[0], "pred_cmp_method", "Comparison tree"),
            (axes[1], "pred_abs_method", "Abstract tree")):
        if col not in val_df.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ct = pd.crosstab(val_df["winner"], val_df[col],
                         rownames=["actual"], colnames=["predicted"])
        for c in ("programs", "worlds"):
            if c not in ct.columns:
                ct[c] = 0
        for c in ("programs", "worlds"):
            if c not in ct.index:
                ct.loc[c] = 0
        ct = ct.loc[["programs", "worlds"], ["programs", "worlds"]]
        mat = ct.values
        acc = np.trace(mat) / mat.sum()

        im = ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_yticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_xlabel("predicted", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_ylabel("actual", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_title(f"{title}  (acc={acc:.3f})",
                     fontsize=TITLE_SIZE, fontname=FONT, pad=8)
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=14, color=color, fontname=FONT)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_abstract_per_leaf(val_df: pd.DataFrame, out_path: str,
                          problem_threshold: float = 0.6):
    """
    Per-leaf accuracy of the abstract tree, sorted by accuracy. Bars are
    coloured red when the leaf is below `problem_threshold` (the "failure
    regions" the paper needs to flag) and green otherwise. Every bar is
    annotated with its sample count so the reader can tell apart a
    catastrophic leaf that covers 100+ samples from a noisy leaf that
    covers only 4.
    """
    if "pred_abs_region" not in val_df.columns:
        print(f"  [warn] no pred_abs_region column — skipping {out_path}")
        return

    grp = val_df.groupby("pred_abs_region").agg(
        n=("match_abs", "size"),
        acc=("match_abs", "mean"),
        majority_pred=("pred_abs_method",
                       lambda s: s.mode().iloc[0] if not s.empty else "?"),
    ).reset_index().sort_values("acc")

    n_leaves = len(grp)
    fig_h = max(4, 0.55 * n_leaves + 1.5)

    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(n_leaves)

    colors = ["#C0392B" if a < problem_threshold else "#27AE60"
              for a in grp["acc"].values]
    ax.barh(y, grp["acc"].values, color=colors, edgecolor="white", height=0.65)

    for i, (acc, n, pred) in enumerate(zip(grp["acc"], grp["n"],
                                           grp["majority_pred"])):
        ax.text(min(acc + 0.02, 1.02), i,
                f"{acc:.2f}   (n={n}, pred={pred})",
                va="center", fontsize=10, fontname=FONT)

    ax.axvline(problem_threshold, color="black", lw=0.7, ls="--",
               label=f"Threshold ({problem_threshold:.1f})")
    ax.axvline(0.5, color="gray", lw=0.6, ls=":", label="Chance level (0.5)")

    ax.set_yticks(y)
    ax.set_yticklabels(grp["pred_abs_region"], fontname=FONT, fontsize=11)
    ax.set_xlim(0, 1.30)
    style_ax(ax, xlabel="Accuracy on validation set",
             ylabel="Abstract-tree leaf",
             title="Abstract tree: failure concentrates in a few leaves")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Heuristic figures ──────────────────────────────────────────────────────
def fig_per_region(heur_df: pd.DataFrame, out_path: str):
    """Baseline vs heuristic average quality per region."""
    grp = heur_df.groupby(["region", "method"]).agg(
        base=("base_quality", "mean"),
        heur=("heur_quality", "mean"),
        wins=("quality_delta", lambda s: int((s > 0).sum())),
        n=("quality_delta", "size"),
    ).reset_index()
    grp = grp.sort_values("region")
    x = np.arange(len(grp))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, grp["base"], width, label="Baseline (uniform)",
           color="#666666", edgecolor="white")
    ax.bar(x + width / 2, grp["heur"], width, label="Heuristic",
           color=COLOR_WORLDS, edgecolor="white")

    for i, r in grp.iterrows():
        ax.text(x[i] - width / 2, r["base"] + 0.015, f"{r['base']:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(x[i] + width / 2, r["heur"] + 0.015, f"{r['heur']:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(x[i], -0.06,
                f"{r['region']}\n{r['method']}\nwins {r['wins']}/{r['n']}",
                ha="center", va="top", fontsize=9, fontname=FONT,
                color="dimgray")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(-0.15, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(x))
    style_ax(ax, ylabel="Mean approximation quality",
             title="Heuristic vs baseline, per canonical region")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_delta_box(heur_df: pd.DataFrame, out_path: str):
    """Distribution of quality_delta per region (positive = heuristic better)."""
    regions = sorted(heur_df["region"].unique())
    data = [heur_df.loc[heur_df["region"] == r, "quality_delta"].values
            for r in regions]
    methods = [heur_df.loc[heur_df["region"] == r, "method"].iloc[0]
               for r in regions]
    colors = [COLOR_WORLDS if m == "worlds" else COLOR_PROGRAMS for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bp = ax.boxplot(data, labels=regions, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", linewidth=1.4),
                    showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    # scatter of individual points
    for i, arr in enumerate(data, start=1):
        jitter = np.random.uniform(-0.08, 0.08, size=len(arr))
        ax.scatter(np.full(len(arr), i) + jitter, arr,
                   color="black", alpha=0.55, s=18, zorder=3)

    ax.axhline(0, color="black", lw=0.7, ls=":")
    ax.set_ylim(-1.05, 1.05)
    style_ax(ax, xlabel="Region",
             ylabel="quality_delta  =  heur_quality − base_quality",
             title="Per-query quality gain of the heuristic (positive = better)")

    # annotate median and wins
    for i, (arr, r) in enumerate(zip(data, regions), start=1):
        med = float(np.median(arr))
        wins = int((arr > 0).sum())
        ax.text(i, 1.02, f"Δ̃={med:+.2f}\nwins {wins}/{len(arr)}",
                ha="center", va="bottom", fontsize=9, fontname=FONT)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_scatter(heur_df: pd.DataFrame, out_path: str):
    """Scatter of base_quality vs heur_quality, colored by region."""
    regions = sorted(heur_df["region"].unique())
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    palette = {
        "A": COLOR_WORLDS,
        "C": "#56B4E9",
        "B": COLOR_PROGRAMS,
        "D": "#E69F00",
    }
    for r in regions:
        sub = heur_df[heur_df["region"] == r]
        ax.scatter(sub["base_quality"], sub["heur_quality"],
                   s=60, alpha=0.85, edgecolor="white",
                   color=palette.get(r, "gray"),
                   label=f"Region {r}  ({sub['method'].iloc[0]})")

    ax.plot([0, 1], [0, 1], color="black", lw=0.7, ls="--",
            label="parity (base = heur)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    style_ax(ax, xlabel="Baseline quality",
             ylabel="Heuristic quality",
             title="Per-query comparison  (above the line: heuristic wins)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",
                        default="results_5min_val/analysis/validation_predictions.csv")
    parser.add_argument("--heur_csv",
                        default="results_5min/heuristics/heuristic_results.csv")
    parser.add_argument("--output_dir", default="results_5min/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    if os.path.exists(args.val_csv):
        print(f"Loading validation predictions: {args.val_csv}")
        val_df = pd.read_csv(args.val_csv, sep=";")
        val_df = val_df.dropna(subset=["winner"])
        val_df = val_df[val_df["winner"].isin(["worlds", "programs"])]
        print(f"  {len(val_df)} rows, {val_df['config'].nunique()} configs")
        fig_per_config(val_df, os.path.join(args.output_dir,
                                            "validation_per_config.pdf"))
        fig_confusion(val_df, os.path.join(args.output_dir,
                                           "validation_confusion.pdf"))
        fig_abstract_per_leaf(
            val_df,
            os.path.join(args.output_dir, "abstract_tree_per_leaf.pdf"),
            problem_threshold=0.6)
    else:
        print(f"[warn] validation CSV not found at {args.val_csv} — skipping.")

    if os.path.exists(args.heur_csv):
        print(f"\nLoading heuristic results: {args.heur_csv}")
        heur_df = pd.read_csv(args.heur_csv, sep=";")
        heur_df = heur_df[heur_df["ok"] == True]
        heur_df = heur_df.dropna(subset=["quality_delta"])
        print(f"  {len(heur_df)} queries evaluated")
        fig_per_region(heur_df, os.path.join(args.output_dir,
                                             "heuristic_per_region.pdf"))
        fig_delta_box(heur_df, os.path.join(args.output_dir,
                                            "heuristic_delta_box.pdf"))
        fig_scatter(heur_df, os.path.join(args.output_dir,
                                          "heuristic_scatter.pdf"))
    else:
        print(f"[warn] heuristic CSV not found at {args.heur_csv} — skipping.")


if __name__ == "__main__":
    main()
