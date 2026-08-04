"""
Publication-quality figures for the time-to-exact (ground-truth) validation.

Reads:
    results_tte/analysis/validation_tte_predictions.csv
    results_tte/all_results.csv                          (for gt distribution)

Writes PDFs to --output_dir (default: results_tte/figures/):
    tte_headline_accuracy.pdf        overall bar + baseline
    tte_per_config.pdf               grouped bars per configuration
    tte_confusion.pdf                2x2 confusion, both trees
    tte_abstract_per_leaf.pdf        per-leaf accuracy of abstract tree
    tte_comparison_per_region.pdf    per-region accuracy of comparison tree
    tte_gt_distribution.pdf          winner mix per configuration
    tte_time_scatter.pdf             t_worlds vs t_programs, coloured by winner
    tte_time_distribution.pdf        per-method time histograms

Usage:
    python sampling/generate_tte_figures.py
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
COLOR_UNRES    = "#999999"


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=11)
    ax.minorticks_on()


# ── Headline ────────────────────────────────────────────────────────────────
def fig_headline(val_df: pd.DataFrame, out_path: str):
    """Overall accuracy of both trees vs the majority baseline."""
    n = len(val_df)
    acc_cmp = val_df["match_cmp"].mean()
    acc_abs = val_df["match_abs"].mean()
    base = (val_df["gt_winner"] == val_df["gt_winner"].mode().iloc[0]).mean()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = ["Majority\nbaseline", "Abstract\ntree", "Comparison\ntree"]
    values = [base, acc_abs, acc_cmp]
    colors = [COLOR_UNRES, COLOR_ABSTRACT, COLOR_COMPARE]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.55)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=12, fontname=FONT)

    ax.axhline(0.5, color="gray", lw=0.6, ls=":", label="Chance level (0.5)")
    ax.set_ylim(0, 1.05)
    style_ax(ax, ylabel="Accuracy (time-to-exact ground truth)",
             title=f"Overall accuracy on {n} resolved queries")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Per-config ──────────────────────────────────────────────────────────────
def fig_per_config(val_df: pd.DataFrame, out_path: str):
    grp = val_df.groupby("config").agg(
        n=("gt_winner", "size"),
        cmp=("match_cmp", "mean"),
        abs_=("match_abs", "mean"),
    ).reset_index()
    grp = grp.sort_values("n", ascending=False)
    configs = grp["config"].values
    x = np.arange(len(configs))
    width = 0.4

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - width / 2, grp["cmp"].values, width, label="Comparison tree",
           color=COLOR_COMPARE, edgecolor="white")
    ax.bar(x + width / 2, grp["abs_"].values, width, label="Abstract tree",
           color=COLOR_ABSTRACT, edgecolor="white")

    for i, (c, a, n) in enumerate(zip(grp["cmp"], grp["abs_"], grp["n"])):
        ax.text(i - width / 2, c + 0.02, f"{c:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(i + width / 2, a + 0.02, f"{a:.2f}",
                ha="center", fontsize=9, fontname=FONT)
        ax.text(i, -0.08, f"n={n}", ha="center", va="top",
                fontsize=9, color="dimgray", fontname=FONT)

    ax.axhline(0.5, color="gray", lw=0.6, ls=":", label="Chance level (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=0, fontsize=11, fontname=FONT)
    ax.set_ylim(-0.15, 1.15)
    style_ax(ax, xlabel="Configuration",
             ylabel="Accuracy (time-to-exact ground truth)",
             title="Per-configuration accuracy on the time-to-exact benchmark")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Confusion ───────────────────────────────────────────────────────────────
def fig_confusion(val_df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, col, title in (
            (axes[0], "pred_cmp_method", "Comparison tree"),
            (axes[1], "pred_abs_method", "Abstract tree")):
        if col not in val_df.columns:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        ct = pd.crosstab(val_df["gt_winner"], val_df[col],
                         rownames=["actual"], colnames=["predicted"])
        for c in ("programs", "worlds"):
            if c not in ct.columns:
                ct[c] = 0
            if c not in ct.index:
                ct.loc[c] = 0
        ct = ct.loc[["programs", "worlds"], ["programs", "worlds"]]
        mat = ct.values
        acc = np.trace(mat) / mat.sum()

        ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_yticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_xlabel("predicted", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_ylabel("actual", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_title(f"{title}  (acc={acc:.3f})",
                     fontsize=TITLE_SIZE, fontname=FONT, pad=8)
        # Recall per class as sub-annotation
        for i in range(2):
            row_total = mat[i].sum()
            recall = mat[i, i] / row_total if row_total else 0
            ax.text(2.05, i, f"recall={recall:.2f}",
                    va="center", fontsize=10, color="dimgray", fontname=FONT)
        for i in range(2):
            for j in range(2):
                color = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=14, color=color, fontname=FONT)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Per-leaf / per-region ───────────────────────────────────────────────────
def _per_region_barh(val_df, region_col, match_col, pred_col,
                     out_path, title, problem_threshold=0.6):
    if region_col not in val_df.columns:
        print(f"  [warn] no {region_col} — skipping {out_path}")
        return
    grp = val_df.groupby(region_col).agg(
        n=(match_col, "size"),
        acc=(match_col, "mean"),
        pred=(pred_col, lambda s: s.mode().iloc[0] if not s.empty else "?"),
    ).reset_index().sort_values("acc")

    n_leaves = len(grp)
    fig_h = max(4, 0.55 * n_leaves + 1.5)

    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(n_leaves)
    colors = ["#C0392B" if a < problem_threshold else "#27AE60"
              for a in grp["acc"].values]
    ax.barh(y, grp["acc"].values, color=colors, edgecolor="white", height=0.65)

    for i, (acc, n, pred) in enumerate(zip(grp["acc"], grp["n"], grp["pred"])):
        ax.text(min(acc + 0.02, 1.02), i,
                f"{acc:.2f}   (n={n}, pred={pred})",
                va="center", fontsize=10, fontname=FONT)

    ax.axvline(problem_threshold, color="black", lw=0.7, ls="--",
               label=f"Threshold ({problem_threshold:.1f})")
    ax.axvline(0.5, color="gray", lw=0.6, ls=":", label="Chance level (0.5)")

    ax.set_yticks(y)
    ax.set_yticklabels(grp[region_col], fontname=FONT, fontsize=11)
    ax.set_xlim(0, 1.30)
    style_ax(ax, xlabel="Accuracy (time-to-exact)",
             ylabel="Region",
             title=title)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Ground-truth distribution ───────────────────────────────────────────────
def fig_gt_distribution(all_df: pd.DataFrame, out_path: str):
    """Stacked bars per config: worlds / programs / unresolved counts."""
    grp = all_df.groupby(["config", "gt_winner"]).size().unstack(fill_value=0)
    for c in ("worlds", "programs", "unresolved"):
        if c not in grp.columns:
            grp[c] = 0
    grp = grp[["worlds", "programs", "unresolved"]]
    grp["total"] = grp.sum(axis=1)
    grp = grp.sort_values("total", ascending=False)
    grp = grp.drop(columns="total")

    configs = grp.index.values
    x = np.arange(len(configs))
    w = grp["worlds"].values
    p = grp["programs"].values
    u = grp["unresolved"].values

    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x, w, color=COLOR_WORLDS, label="worlds (fastest)",
           edgecolor="white")
    ax.bar(x, p, bottom=w, color=COLOR_PROGRAMS, label="programs (fastest)",
           edgecolor="white")
    ax.bar(x, u, bottom=w + p, color=COLOR_UNRES,
           label="unresolved (neither reached exact)", edgecolor="white")

    for i, (wi, pi, ui) in enumerate(zip(w, p, u)):
        total = wi + pi + ui
        if wi > 5:
            ax.text(i, wi / 2, f"{wi}", ha="center", va="center",
                    color="white", fontsize=9, fontname=FONT)
        if pi > 5:
            ax.text(i, wi + pi / 2, f"{pi}", ha="center", va="center",
                    color="white", fontsize=9, fontname=FONT)
        if ui > 5:
            ax.text(i, wi + pi + ui / 2, f"{ui}", ha="center",
                    va="center", color="white", fontsize=9, fontname=FONT)
        ax.text(i, total + 4, f"n={total}", ha="center", va="bottom",
                fontsize=9, color="dimgray", fontname=FONT)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontname=FONT, fontsize=11)
    style_ax(ax, xlabel="Configuration",
             ylabel="Number of INTEREST queries",
             title="Ground-truth winner distribution per configuration "
                   "(15-min safety cap)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Time comparison ─────────────────────────────────────────────────────────
def fig_time_scatter(val_df: pd.DataFrame, out_path: str):
    """t_worlds vs t_programs on log-log axes, coloured by winner."""
    both = val_df.dropna(subset=["time_worlds", "time_programs"]).copy()
    if both.empty:
        print(f"  [warn] no rows where both methods reached exact")
        return

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for winner, color in (("worlds", COLOR_WORLDS),
                          ("programs", COLOR_PROGRAMS)):
        sub = both[both["gt_winner"] == winner]
        ax.scatter(sub["time_worlds"], sub["time_programs"],
                   s=22, alpha=0.6, color=color,
                   label=f"gt_winner = {winner}  (n={len(sub)})",
                   edgecolors="none")

    lo = max(1e-2, min(both["time_worlds"].min(), both["time_programs"].min()))
    hi = max(both["time_worlds"].max(), both["time_programs"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="diagonal (equal time)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    style_ax(ax, xlabel="World-sampling time to exact (s)",
             ylabel="Program-sampling time to exact (s)",
             title="Time to reach the exact interval "
                   "(rows where both methods succeeded)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_time_distribution(val_df: pd.DataFrame, out_path: str):
    """Log-scale histogram of time-to-exact per method."""
    tw = val_df["time_worlds"].dropna().values
    tp = val_df["time_programs"].dropna().values

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.logspace(-2, np.log10(900), 40)
    ax.hist(tw, bins=bins, alpha=0.55, color=COLOR_WORLDS,
            label=f"worlds  (n={len(tw)}, median={np.median(tw):.1f}s)",
            edgecolor="white")
    ax.hist(tp, bins=bins, alpha=0.55, color=COLOR_PROGRAMS,
            label=f"programs (n={len(tp)}, median={np.median(tp):.1f}s)",
            edgecolor="white")

    ax.axvline(np.median(tw), color=COLOR_WORLDS, lw=1.4, ls="--")
    ax.axvline(np.median(tp), color=COLOR_PROGRAMS, lw=1.4, ls="--")
    ax.axvline(900, color="black", lw=0.9, ls=":",
               label="15-min safety cap")

    ax.set_xscale("log")
    style_ax(ax, xlabel="Time to reach the exact interval (s, log scale)",
             ylabel="Number of queries",
             title="Distribution of time-to-exact per sampling method")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",
                        default="results_tte/analysis/"
                                "validation_tte_predictions.csv")
    parser.add_argument("--all_csv",
                        default="results_tte/all_results.csv",
                        help="Full CSV including 'unresolved' rows (for the "
                             "winner-distribution figure).")
    parser.add_argument("--output_dir", default="results_tte/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    if not os.path.exists(args.val_csv):
        raise SystemExit(f"Validation CSV not found: {args.val_csv}")
    print(f"Loading validation predictions: {args.val_csv}")
    val_df = pd.read_csv(args.val_csv, sep=";")
    val_df = val_df.dropna(subset=["gt_winner"])
    val_df = val_df[val_df["gt_winner"].isin(["worlds", "programs"])]
    print(f"  {len(val_df)} evaluable rows, "
          f"{val_df['config'].nunique()} configs")

    fig_headline(val_df, os.path.join(args.output_dir,
                                      "tte_headline_accuracy.pdf"))
    fig_per_config(val_df, os.path.join(args.output_dir,
                                        "tte_per_config.pdf"))
    fig_confusion(val_df, os.path.join(args.output_dir,
                                       "tte_confusion.pdf"))
    _per_region_barh(val_df, "pred_abs_region", "match_abs",
                     "pred_abs_method",
                     os.path.join(args.output_dir,
                                  "tte_abstract_per_leaf.pdf"),
                     "Abstract tree: per-leaf accuracy on time-to-exact")
    _per_region_barh(val_df, "pred_cmp_region", "match_cmp",
                     "pred_cmp_method",
                     os.path.join(args.output_dir,
                                  "tte_comparison_per_region.pdf"),
                     "Comparison tree: per-region accuracy on time-to-exact")
    fig_time_scatter(val_df, os.path.join(args.output_dir,
                                          "tte_time_scatter.pdf"))
    fig_time_distribution(val_df, os.path.join(args.output_dir,
                                               "tte_time_distribution.pdf"))

    if os.path.exists(args.all_csv):
        print(f"\nLoading full CSV (with unresolved): {args.all_csv}")
        all_df = pd.read_csv(args.all_csv, sep=";")
        all_df = all_df.dropna(subset=["gt_winner"])
        fig_gt_distribution(all_df, os.path.join(args.output_dir,
                                                 "tte_gt_distribution.pdf"))
    else:
        print(f"[warn] {args.all_csv} not found — skipping gt distribution.")


if __name__ == "__main__":
    main()
