"""
Correlation and decision analysis for World vs Program sampling experiment.

Reads:  results/all_results.csv
Writes: results/analysis/
    - correlation_heatmap.pdf     : Spearman correlations of all features vs target variables
    - feature_scatter.pdf         : scatter plots of top predictors colored by winner
    - decision_tree.pdf           : sklearn DecisionTree with feature importances
    - analysis_summary.txt        : numerical summary

Usage:
    cd clean_workspace
    python sampling/analyze.py
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
INPUT_CSV  = "results/all_results.csv"
OUTPUT_DIR = "results/analysis"
FONT       = "DejaVu Sans"
LABEL_SIZE = 13
TITLE_SIZE = 16
COLOR_WORLDS   = "#0072B2"   # blue  (color-blind friendly)
COLOR_PROGRAMS = "#D55E00"   # orange

# Features to analyse as potential predictors
PREDICTORS = [
    # Structural sizes
    "annots_minus_em", "em_var", "n_annots", "n_rules",
    "exact_width",
    # Literal features
    "lit_head_def", "lit_head_strict", "lit_body_count",
    "lit_is_negated", "lit_is_fact", "lit_is_ann_fact",
    "lit_complement_body", "lit_complement_head",
    # AF basic features
    "af_pct_annotated", "af_avg_annot_vars", "af_avg_connectors",
    "af_avg_body_size", "af_max_body_size",
    # AM complexity (dialectical graph)
    "am_n_arguments", "am_n_defeaters", "am_n_trees",
    "am_avg_def_rules", "am_avg_arg_lines", "am_avg_height_lines",
    # EM complexity (Bayesian Network)
    "em_n_arcs", "em_treewidth", "em_avg_in_degree", "em_max_in_degree",
    "em_entropy",
    # AF extended complexity
    "af_n_em_vars_used", "af_avg_complexity", "af_max_complexity",
]

TARGETS = ["winner_binary", "quality_diff"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=10)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=11)
    ax.minorticks_on()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── Load & prepare data ───────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(INPUT_CSV, sep=";")

    # Binary target: 1 = worlds wins, 0 = programs wins
    df["winner_binary"] = (df["winner"] == "worlds").astype(int)

    # Continuous target: positive = worlds better, negative = programs better
    df["quality_diff"] = df["worlds_quality"] - df["progs_quality"]

    # n_programs can exceed int64 (2^64) and arrive as object — coerce to float
    df["n_worlds"]   = df["n_worlds"].astype(float)
    df["n_programs"] = df["n_programs"].astype(float)

    # Log-space features (useful for tree, less so for correlation)
    df["log_n_worlds"]   = np.log2(df["n_worlds"])
    df["log_n_programs"] = np.log2(df["n_programs"])
    df["log_space_ratio"] = df["log_n_worlds"] - df["log_n_programs"]  # = annots_minus_em

    return df


# ── 1. Correlation heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(df, out):
    """Spearman correlations of all predictors vs each target."""
    corr_rows = []
    for feat in PREDICTORS:
        if feat not in df.columns:
            continue
        row = {"feature": feat}
        for tgt in TARGETS:
            r, p = stats.spearmanr(df[feat], df[tgt], nan_policy="omit")
            row[f"r_{tgt}"] = r
            row[f"p_{tgt}"] = p
        corr_rows.append(row)

    corr_df = pd.DataFrame(corr_rows).set_index("feature")

    # Build matrix for heatmap
    mat = corr_df[[f"r_{t}" for t in TARGETS]].values
    col_labels = ["winner\n(worlds=1)", "quality diff\n(W−P)"]

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels(col_labels, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_yticks(range(len(PREDICTORS)))
    ax.set_yticklabels(PREDICTORS, fontsize=11, fontname=FONT)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Annotate cells
    for i, feat in enumerate(PREDICTORS):
        for j, tgt in enumerate(TARGETS):
            r = corr_df.loc[feat, f"r_{tgt}"]
            p = corr_df.loc[feat, f"p_{tgt}"]
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            color = "white" if abs(r) > 0.6 else "black"
            ax.text(j, i, f"{r:+.2f}{star}", ha="center", va="center",
                    fontsize=9, color=color, fontname=FONT)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman r", fontsize=LABEL_SIZE, fontname=FONT)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title("Feature correlations with sampling outcome\n(* p<0.05, ** p<0.01, *** p<0.001)",
                 fontsize=TITLE_SIZE, fontname=FONT, pad=28)

    save(fig, "correlation_heatmap.pdf")
    return corr_df


# ── 2. Scatter plots of top predictors ───────────────────────────────────────
def plot_feature_scatters(df, corr_df, out):
    """Scatter plots: top predictors vs quality_diff, colored by winner."""
    # Pick top 6 by abs(r) with quality_diff
    top_feats = (corr_df["r_quality_diff"]
                 .abs()
                 .sort_values(ascending=False)
                 .head(6)
                 .index.tolist())

    n = len(top_feats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    colors = {1: COLOR_WORLDS, 0: COLOR_PROGRAMS}
    labels = {1: "Worlds wins", 0: "Programs wins"}

    for ax, feat in zip(axes, top_feats):
        for wb, grp in df.groupby("winner_binary"):
            ax.scatter(grp[feat], grp["quality_diff"],
                       color=colors[wb], label=labels[wb],
                       alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

        # Trend line
        r, p = stats.spearmanr(df[feat], df["quality_diff"])
        m, b, *_ = stats.linregress(df[feat], df["quality_diff"])
        xs = np.linspace(df[feat].min(), df[feat].max(), 100)
        ax.plot(xs, m * xs + b, color="gray", lw=1.2, ls="--")

        ax.axhline(0, color="black", lw=0.8, ls=":")
        style_ax(ax, xlabel=feat, ylabel="quality diff (W−P)",
                 title=f"r = {r:+.2f}{'*' if p < 0.05 else ''}")

    # Legend on last used ax
    for ax in axes[:n]:
        break
    handles = [mpatches.Patch(color=COLOR_WORLDS, label="Worlds wins"),
               mpatches.Patch(color=COLOR_PROGRAMS, label="Programs wins")]
    axes[0].legend(handles=handles, fontsize=10, framealpha=0.8)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Top predictor features vs quality difference (Worlds − Programs)",
                 fontsize=TITLE_SIZE, fontname=FONT, y=1.01)
    fig.tight_layout()
    save(fig, "feature_scatter.pdf")


# ── 3. Decision tree ──────────────────────────────────────────────────────────
def fit_decision_tree(df, out, summary_lines):
    predictors = [p for p in PREDICTORS if p in df.columns]
    df_imp = df.copy()
    for col in predictors:
        if df_imp[col].isna().any():
            df_imp[col] = df_imp[col].fillna(df_imp[col].median())
    X = df_imp[predictors].values
    y = df_imp["winner_binary"].values

    # Cross-validated accuracy at different depths
    summary_lines.append("\n=== Decision Tree CV Accuracy ===")
    best_depth, best_acc = 1, 0
    for depth in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        scores = cross_val_score(dt, X, y, cv=5, scoring="accuracy")
        summary_lines.append(f"  depth={depth}: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_acc:
            best_acc, best_depth = scores.mean(), depth

    summary_lines.append(f"  -> Best depth: {best_depth} (acc={best_acc:.3f})")

    # Fit final tree at best depth
    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
    dt.fit(X, y)

    summary_lines.append("\n=== Decision Tree Rules ===")
    summary_lines.append(export_text(dt, feature_names=predictors))

    # Feature importances
    importances = pd.Series(dt.feature_importances_, index=predictors)
    importances = importances[importances > 0].sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(importances) * 0.4 + 2)))

    # Left: feature importances
    ax = axes[0]
    bars = ax.barh(importances.index, importances.values,
                   color=COLOR_WORLDS, edgecolor="white", height=0.6)
    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=10, fontname=FONT)
    style_ax(ax, xlabel="Gini importance", title=f"Feature importances (depth={best_depth})")

    # Right: tree structure
    ax2 = axes[1]
    plot_tree(dt, feature_names=predictors,
              class_names=["programs", "worlds"],
              filled=True, rounded=True, ax=ax2,
              fontsize=9, impurity=False,
              node_ids=False)
    ax2.set_title(f"Decision tree (depth={best_depth}, acc={best_acc:.2f})",
                  fontsize=TITLE_SIZE, fontname=FONT, pad=10)

    fig.tight_layout()
    save(fig, "decision_tree.pdf")

    return dt, importances


# ── 4. Summary stats ──────────────────────────────────────────────────────────
def print_summary(df, corr_df, summary_lines):
    summary_lines.append("=== Dataset overview ===")
    summary_lines.append(f"Total rows: {len(df)}")
    for cfg, grp in df.groupby("config"):
        w = (grp["winner"] == "worlds").sum()
        p = (grp["winner"] == "programs").sum()
        summary_lines.append(
            f"  {cfg}: {len(grp)} rows | worlds={w} programs={p} | "
            f"mean quality_diff={grp['quality_diff'].mean():+.3f} ± {grp['quality_diff'].std():.3f}"
        )

    summary_lines.append("\n=== Spearman correlations with winner_binary (|r| sorted) ===")
    top = (corr_df["r_winner_binary"]
           .abs()
           .sort_values(ascending=False))
    for feat, r_abs in top.items():
        r = corr_df.loc[feat, "r_winner_binary"]
        p = corr_df.loc[feat, "p_winner_binary"]
        summary_lines.append(f"  {feat:30s}  r={r:+.3f}  p={p:.4f}")

    summary_lines.append("\n=== Spearman correlations with quality_diff (|r| sorted) ===")
    top2 = (corr_df["r_quality_diff"]
            .abs()
            .sort_values(ascending=False))
    for feat, r_abs in top2.items():
        r = corr_df.loc[feat, "r_quality_diff"]
        p = corr_df.loc[feat, "p_quality_diff"]
        summary_lines.append(f"  {feat:30s}  r={r:+.3f}  p={p:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} rows, {df['config'].nunique()} configs, "
          f"{(df['winner']=='worlds').sum()} worlds / {(df['winner']=='programs').sum()} programs")

    print("\n1. Correlation heatmap...")
    corr_df = plot_correlation_heatmap(df, OUTPUT_DIR)

    print("\n2. Scatter plots...")
    plot_feature_scatters(df, corr_df, OUTPUT_DIR)

    print("\n3. Decision tree...")
    summary_lines = []
    print_summary(df, corr_df, summary_lines)
    dt, importances = fit_decision_tree(df, OUTPUT_DIR, summary_lines)

    # Write summary
    txt_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(summary_lines))
    print(f"  Saved: {txt_path}")

    # Print to stdout too
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
