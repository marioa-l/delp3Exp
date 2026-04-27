"""
Compare the structural features of "mismatch" literals (where the decision
tree rule fails) against "correct" literals (where the rule works).

A mismatch is defined as: predicted winner != actual winner AND the margin
is non-trivial (|quality_diff| > 0.01 — excludes technical ties where both
methods score ~1.0 on trivial literals).

Reads:  results/respaldo/all_results.csv
Writes: results/analysis/
    - mismatch_features.pdf    : feature comparison (mismatch vs correct)
    - mismatch_detail_ssm.pdf  : per-literal view of ssm mismatch cases
    - mismatch_summary.txt     : numerical report
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

INPUT_CSV  = "results/respaldo/all_results.csv"
OUTPUT_DIR = "results/analysis"

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 15
COLOR_CORRECT  = "#0072B2"   # blue
COLOR_MISMATCH = "#D55E00"   # orange

LITERAL_FEATS = [
    "lit_is_negated", "lit_head_def", "lit_head_strict",
    "lit_body_count", "lit_is_fact", "lit_is_ann_fact",
    "lit_complement_body", "lit_complement_head",
]
AF_FEATS = [
    "af_pct_annotated", "af_avg_annot_vars", "af_avg_connectors",
    "af_avg_body_size", "af_max_body_size",
]
STRUCT_FEATS = [
    "annots_minus_em", "em_var", "n_annots", "exact_width",
]


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
    ax.minorticks_on()


def load_and_flag():
    df = pd.read_csv(INPUT_CSV, sep=";")
    df["predicted"] = df["annots_minus_em"].apply(
        lambda x: "worlds" if x > -6 else "programs"
    )
    df["quality_diff"] = df["worlds_quality"] - df["progs_quality"]
    df["margin"] = df["quality_diff"].abs()
    df["mismatch"] = df["predicted"] != df["winner"]
    # Real mismatches exclude ties/trivial cases (margin ~0)
    df["real_mismatch"] = df["mismatch"] & (df["margin"] > 0.01)
    return df


def plot_feature_comparison(df, out):
    """For each feature, show boxplot of correct vs real-mismatch groups."""
    correct = df[~df["real_mismatch"]]
    mismatch = df[df["real_mismatch"]]

    all_feats = STRUCT_FEATS + LITERAL_FEATS + AF_FEATS
    n = len(all_feats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, all_feats):
        data = [correct[feat].values, mismatch[feat].values]
        bp = ax.boxplot(data, labels=["correct", "mismatch"],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.2),
                        showfliers=True)
        for patch, c in zip(bp["boxes"], [COLOR_CORRECT, COLOR_MISMATCH]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        # Mann–Whitney test (non-parametric, OK with small sample)
        try:
            _, p = stats.mannwhitneyu(correct[feat], mismatch[feat],
                                      alternative="two-sided")
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            title = f"{feat}\n(p={p:.3f} {star})"
        except Exception:
            title = feat

        style_ax(ax, title=title)
        ax.tick_params(axis="x", labelsize=10)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Feature distributions: correct (n={len(correct)}) vs real mismatches (n={len(mismatch)})",
        fontsize=TITLE_SIZE + 1, fontname=FONT, y=1.00,
    )
    fig.tight_layout()
    path = os.path.join(out, "mismatch_features.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ssm_detail(df, out):
    """Per-literal view of ssm cases. Highlights the mismatches."""
    ssm = df[df["config"] == "ssm"].copy()
    ssm = ssm.sort_values(["real_mismatch", "annots_minus_em", "model"],
                          ascending=[True, True, True]).reset_index(drop=True)

    labels = [f"{r['model']}/{r['literal']}" for _, r in ssm.iterrows()]
    x = np.arange(len(ssm))
    wq = ssm["worlds_quality"].values
    pq = ssm["progs_quality"].values
    is_mm = ssm["real_mismatch"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.38
    bars_w = ax.bar(x - width / 2, wq, width, label="worlds quality",
                    color=COLOR_CORRECT, alpha=0.85)
    bars_p = ax.bar(x + width / 2, pq, width, label="progs quality",
                    color=COLOR_MISMATCH, alpha=0.85)

    # Highlight real mismatches with a red underline on the x label
    for i, mm in enumerate(is_mm):
        if mm:
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.08, zorder=0)

    # Annotate annots_minus_em below each literal
    for i, v in enumerate(ssm["annots_minus_em"].values):
        ax.text(i, -0.07, f"{v}", ha="center", va="top",
                fontsize=9, color="dimgray", fontname=FONT)
    ax.text(-0.8, -0.07, "a−em:", ha="right", va="top",
            fontsize=9, color="dimgray", fontname=FONT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(-0.12, 1.1)

    style_ax(ax, ylabel="approximation quality",
             title=f"ssm literals — real mismatches highlighted ({is_mm.sum()} / {len(ssm)})")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out, "mismatch_detail_ssm.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def write_summary(df, out, lines):
    path = os.path.join(out, "mismatch_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    df = load_and_flag()
    real = df[df["real_mismatch"]]
    correct = df[~df["real_mismatch"]]

    lines = []
    lines.append("=== Dataset ===")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Any-mismatch (pred != winner): {df['mismatch'].sum()}")
    lines.append(f"Real mismatches (margin > 0.01): {df['real_mismatch'].sum()}")
    lines.append(f"Correct (incl. ties w/ margin <= 0.01): {(~df['real_mismatch']).sum()}")

    lines.append("\n=== Real mismatches (detailed) ===")
    cols = ["config", "model", "literal", "annots_minus_em",
            "exact_width", "worlds_quality", "progs_quality",
            "predicted", "winner", "margin"]
    lines.append(real[cols].to_string(index=False))

    lines.append("\n=== Mann–Whitney U test: correct vs real mismatches ===")
    for feat in STRUCT_FEATS + LITERAL_FEATS + AF_FEATS:
        try:
            u, p = stats.mannwhitneyu(correct[feat], real[feat],
                                      alternative="two-sided")
            med_c = correct[feat].median()
            med_m = real[feat].median()
            flag = "*" if p < 0.05 else ""
            lines.append(f"  {feat:25s}  median corr={med_c:7.3f}  mism={med_m:7.3f}  p={p:.3f} {flag}")
        except Exception as e:
            lines.append(f"  {feat:25s}  ERROR: {e}")

    print("\n1. Feature comparison plot...")
    plot_feature_comparison(df, OUTPUT_DIR)

    print("\n2. ssm detail plot...")
    plot_ssm_detail(df, OUTPUT_DIR)

    print("\n3. Summary text...")
    write_summary(df, OUTPUT_DIR, lines)

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
