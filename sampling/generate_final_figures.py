"""
Headline figures for the time-to-exact study.

Compares three recommenders against the ground truth (which sampler reaches
the exact interval first):

    baseline      always answer "worlds" (the majority class)
    abstract      depth-3 categorical tree trained on the 5-minute budget label
    comparison    hand-written pairwise-comparison rules
    gt-tree       depth-1 tree trained on the time-to-exact label itself,
                  splitting on space_log_ratio = n_annots - em_var

Reads:
    results_tte/analysis/validation_tte_predictions.csv   (abstract + comparison)
    results_tte/analysis/gt_tree_predictions.csv          (out-of-fold gt-tree)
    results_tte/all_results.csv                           (for unresolved counts)

Writes PDFs to --output_dir (default results_tte/figures/):
    final_model_comparison.pdf   overall accuracy + per-class recall
    final_per_config.pdf         per-configuration accuracy, all three models
    final_rule.pdf               the learned rule and its decision boundary
    final_confusion.pdf          confusion matrices side by side

Usage:
    python sampling/generate_final_figures.py
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
COLOR_GT       = "#3B3B8F"
COLOR_BASE     = "#999999"


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=11)
    ax.minorticks_on()


def load(args):
    val = pd.read_csv(args.val_csv, sep=";")
    val = val[val["gt_winner"].isin(["worlds", "programs"])].copy()
    gt = pd.read_csv(args.gt_csv, sep=";")
    gt = gt[gt["gt_winner"].isin(["worlds", "programs"])].copy()

    key = ["config", "model", "literal"]
    merged = val.merge(gt[key + ["pred_gt", "match_gt"]], on=key, how="inner")
    if os.path.exists(args.final_csv):
        fin = pd.read_csv(args.final_csv, sep=";")
        merged = merged.merge(fin[key + ["pred_final", "match_final"]],
                              on=key, how="inner")
    merged["pred_base"] = "worlds"
    merged["match_base"] = merged["gt_winner"] == "worlds"
    return merged


COLOR_FINAL = "#7B1FA2"

MODELS = [
    ("match_base", "pred_base", "Majority\nbaseline", COLOR_BASE),
    ("match_abs",  "pred_abs_method", "Abstract\ntree\n(5-min)", COLOR_ABSTRACT),
    ("match_cmp",  "pred_cmp_method", "Comparison\ntree\n(hand-written)", COLOR_COMPARE),
    ("match_gt",   "pred_gt", "Binned tree\n(time-to-\nexact)", COLOR_GT),
    ("match_final", "pred_final", "Comparison\ntree\n(time-to-exact)", COLOR_FINAL),
]


def fig_model_comparison(df, out_path):
    """Overall accuracy plus per-class recall — accuracy alone hides the
    class imbalance, so recall on the minority class is shown beside it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 5.6))

    labels = [m[2] for m in MODELS]
    colors = [m[3] for m in MODELS]
    accs = [df[m[0]].mean() for m in MODELS]

    bars = ax1.bar(labels, accs, color=colors, edgecolor="white", width=0.6)
    for b, v in zip(bars, accs):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", fontsize=12, fontname=FONT)
    base = accs[0]
    ax1.axhline(base, color=COLOR_BASE, lw=1.0, ls="--",
                label=f"Baseline ({base:.3f})")
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(axis="x", labelsize=9)
    style_ax(ax1, ylabel="Accuracy",
             title=f"Overall accuracy  (n={len(df)})")
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # Per-class recall
    x = np.arange(len(MODELS))
    w = 0.38
    rec_p, rec_w = [], []
    for match_col, pred_col, _, _ in MODELS:
        for cls, store in (("programs", rec_p), ("worlds", rec_w)):
            sub = df[df["gt_winner"] == cls]
            store.append((sub[pred_col] == cls).mean())

    ax2.bar(x - w / 2, rec_p, w, label="recall: programs (minority)",
            color=COLOR_PROGRAMS, edgecolor="white")
    ax2.bar(x + w / 2, rec_w, w, label="recall: worlds (majority)",
            color=COLOR_WORLDS, edgecolor="white")
    for i, (p, wv) in enumerate(zip(rec_p, rec_w)):
        ax2.text(i - w / 2, p + 0.012, f"{p:.2f}", ha="center",
                 fontsize=10, fontname=FONT)
        ax2.text(i + w / 2, wv + 0.012, f"{wv:.2f}", ha="center",
                 fontsize=10, fontname=FONT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontname=FONT)
    ax2.set_ylim(0, 1.12)
    n_p = int((df["gt_winner"] == "programs").sum())
    n_w = int((df["gt_winner"] == "worlds").sum())
    style_ax(ax2, ylabel="Recall",
             title=f"Per-class recall  (programs n={n_p}, worlds n={n_w})")
    ax2.legend(loc="lower left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_per_config(df, out_path):
    grp = df.groupby("config").agg(
        n=("gt_winner", "size"),
        base=("match_base", "mean"),
        abs_=("match_abs", "mean"),
        cmp_=("match_cmp", "mean"),
        gt=("match_gt", "mean"),
        fin=("match_final", "mean"),
    ).reset_index().sort_values("n", ascending=False)

    x = np.arange(len(grp))
    w = 0.165
    fig, ax = plt.subplots(figsize=(15, 5.4))
    series = [("base", "Majority baseline", COLOR_BASE),
              ("abs_", "Abstract tree (5-min label)", COLOR_ABSTRACT),
              ("cmp_", "Comparison tree (hand-written)", COLOR_COMPARE),
              ("gt", "Binned tree (time-to-exact)", COLOR_GT),
              ("fin", "Comparison tree (time-to-exact)", COLOR_FINAL)]
    for i, (col, lab, c) in enumerate(series):
        off = (i - 2.0) * w
        ax.bar(x + off, grp[col].values, w, label=lab, color=c,
               edgecolor="white")

    for i, n in enumerate(grp["n"]):
        ax.text(i, -0.07, f"n={n}", ha="center", va="top",
                fontsize=8.5, color="dimgray", fontname=FONT)

    ax.set_xticks(x)
    ax.set_xticklabels(grp["config"], fontsize=11, fontname=FONT)
    ax.set_ylim(-0.14, 1.14)
    style_ax(ax, xlabel="Configuration", ylabel="Accuracy",
             title="Per-configuration accuracy on the time-to-exact ground truth")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_rule(df, out_path, cut=-8.0):
    """The learned decision boundary on space_log_ratio."""
    d = df.copy()
    d["ratio"] = d["n_annots"] - d["em_var"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1.0]})

    bins = np.arange(d["ratio"].min() - 1, d["ratio"].max() + 2, 2)
    for cls, color in (("worlds", COLOR_WORLDS), ("programs", COLOR_PROGRAMS)):
        ax1.hist(d.loc[d.gt_winner == cls, "ratio"], bins=bins, alpha=0.65,
                 color=color, edgecolor="white",
                 label=f"{cls} reaches exact first "
                       f"(n={(d.gt_winner==cls).sum()})")
    ax1.axvline(cut, color="black", lw=1.6, ls="--")
    ax1.annotate(f"cut at {cut:.0f}", xy=(cut, ax1.get_ylim()[1] * 0.72),
                 xytext=(cut + 18, ax1.get_ylim()[1] * 0.72),
                 fontsize=11, fontname=FONT, va="center",
                 arrowprops=dict(arrowstyle="->", lw=1.0, color="black"))
    style_ax(ax1, xlabel=r"space_log_ratio  =  n_annots $-$ em_var",
             ylabel="Number of queries",
             title="The rule: sample programs when the program space\n"
                   "is at least $2^{8}$ times smaller than the world space")
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)

    # Purity of each side of the cut
    left = d[d["ratio"] <= cut]
    right = d[d["ratio"] > cut]
    sides = [("ratio <= -8\n-> programs", left, COLOR_PROGRAMS, "programs"),
             ("ratio > -8\n-> worlds", right, COLOR_WORLDS, "worlds")]
    xs = np.arange(2)
    purity = [(s[1]["gt_winner"] == s[3]).mean() for s in sides]
    ns = [len(s[1]) for s in sides]
    bars = ax2.bar(xs, purity, color=[s[2] for s in sides],
                   edgecolor="white", width=0.55)
    for b, p, n in zip(bars, purity, ns):
        ax2.text(b.get_x() + b.get_width() / 2, p + 0.015,
                 f"{p:.3f}\n(n={n})", ha="center", fontsize=11, fontname=FONT)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s[0] for s in sides], fontsize=11, fontname=FONT)
    ax2.set_ylim(0, 1.12)
    style_ax(ax2, ylabel="Purity of the branch",
             title="The worlds branch is near-pure;\n"
                   "the programs branch carries the residual error")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_confusion(df, out_path):
    show = [m for m in MODELS if m[0] != "match_base"]
    fig, axes = plt.subplots(1, len(show), figsize=(5 * len(show), 5))
    for ax, (match_col, pred_col, title, _) in zip(axes, show):
        ct = pd.crosstab(df["gt_winner"], df[pred_col])
        for c in ("programs", "worlds"):
            if c not in ct.columns:
                ct[c] = 0
            if c not in ct.index:
                ct.loc[c] = 0
        ct = ct.loc[["programs", "worlds"], ["programs", "worlds"]]
        mat = ct.values
        acc = np.trace(mat) / mat.sum()

        ax.imshow(mat, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_yticklabels(["programs", "worlds"], fontsize=11, fontname=FONT)
        ax.set_xlabel("predicted", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_ylabel("actual", fontsize=LABEL_SIZE, fontname=FONT)
        ax.set_title(f"{title.replace(chr(10),' ')}\nacc={acc:.3f}",
                     fontsize=12.5, fontname=FONT, pad=8)
        for i in range(2):
            for j in range(2):
                col = "white" if mat[i, j] > mat.max() * 0.5 else "black"
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        fontsize=14, color=col, fontname=FONT)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_csv",
                        default="results_tte/analysis/validation_tte_predictions.csv")
    parser.add_argument("--gt_csv",
                        default="results_tte/analysis/gt_tree_predictions.csv")
    parser.add_argument("--final_csv",
                        default="results_tte/analysis/"
                                "final_tree_predictions.csv")
    parser.add_argument("--output_dir", default="results_tte/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    df = load(args)
    print(f"Merged rows: {len(df)}  ({df.config.nunique()} configs)")

    fig_model_comparison(df, os.path.join(args.output_dir,
                                          "final_model_comparison.pdf"))
    fig_per_config(df, os.path.join(args.output_dir, "final_per_config.pdf"))
    fig_rule(df, os.path.join(args.output_dir, "final_rule.pdf"))
    fig_confusion(df, os.path.join(args.output_dir, "final_confusion.pdf"))


if __name__ == "__main__":
    main()
