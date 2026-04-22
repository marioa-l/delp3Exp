"""
Correlation analysis and decision tree for sampling approach recommendation.

Reads the CSV produced by main_mmm_csv.py (which includes structural model features
alongside experiment results), then:
  1. Derives winner label and quality_diff for each literal.
  2. Plots a correlation heatmap (structural features vs quality metrics).
  3. Plots scatter plots of key features vs quality difference.
  4. Fits a Decision Tree classifier (winner ~ features) and reports CV accuracy.
  5. Saves the tree diagram, feature importances, and text rules to output_dir.

Usage:
    python sampling/decision_analysis.py \
        --csv data/results/mmm_results.csv \
        --output_dir data/results/analysis/
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONT = "DejaVu Sans"
DPI = 300
STRUCTURAL_FEATURES = [
    "n_em_vars",
    "n_annots",
    "annots_minus_em",
    "n_rules",
    "n_literals_total",
    "exact_width",
]
COLORS = {"worlds": "#1f77b4", "programs": "#ff7f0e", "tie": "#2ca02c"}


def load_and_prepare(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["worlds_metric", "progs_metric"])
    df["annots_minus_em"] = df["n_annots"] - df["n_em_vars"]
    df["exact_width"] = df["exact_u"] - df["exact_l"]
    df["quality_diff"] = df["worlds_metric"] - df["progs_metric"]
    df["winner"] = df.apply(
        lambda r: "worlds"
        if r["worlds_metric"] > r["progs_metric"]
        else ("programs" if r["progs_metric"] > r["worlds_metric"] else "tie"),
        axis=1,
    )
    return df


def plot_correlation_heatmap(df, output_path):
    metric_cols = ["worlds_metric", "progs_metric", "quality_diff"]
    cols = STRUCTURAL_FEATURES + metric_cols
    available = [c for c in cols if c in df.columns]
    corr = df[available].corr()

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(len(available) * 1.1 + 1, len(available) * 0.9 + 1))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=12, fontfamily=FONT)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(
        corr.columns, rotation=45, ha="right", fontsize=10, fontfamily=FONT
    )
    ax.set_yticklabels(corr.index, fontsize=10, fontfamily=FONT)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            color = "white" if abs(v) > 0.6 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_title(
        "Feature–Metric Correlation Matrix", fontsize=16, fontfamily=FONT, pad=12
    )
    ax.tick_params(which="both", direction="in", length=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_quality(df, output_path):
    features_to_plot = [
        ("annots_minus_em", "n_annots − n_em_vars\n(positive → more programs than worlds)"),
        ("n_em_vars", "n_em_vars  (log₂ worlds)"),
        ("exact_width", "Exact Interval Width"),
    ]

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(6 * len(features_to_plot), 5))

    for ax, (feat, xlabel) in zip(axes, features_to_plot):
        if feat not in df.columns:
            ax.set_visible(False)
            continue
        for winner, grp in df.groupby("winner"):
            ax.scatter(
                grp[feat],
                grp["quality_diff"],
                c=COLORS.get(winner, "gray"),
                label=winner,
                alpha=0.55,
                s=35,
                edgecolors="none",
            )
        ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=12, fontfamily=FONT)
        ax.set_ylabel("Quality Diff (Worlds − Programs)", fontsize=12, fontfamily=FONT)
        ax.tick_params(which="both", direction="in")
        ax.minorticks_on()
        ax.legend(fontsize=10, framealpha=0.8)

    fig.suptitle(
        "Structural Features vs Approximation Quality Difference",
        fontsize=16,
        fontfamily=FONT,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def fit_and_plot_decision_tree(df, output_dir):
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
    except ImportError:
        print("scikit-learn not found. Skipping decision tree. Install with: pip install scikit-learn")
        return None, None, None

    df_model = df[df["winner"] != "tie"].copy()
    available = [f for f in STRUCTURAL_FEATURES if f in df_model.columns]
    X = df_model[available].fillna(0).values
    y = df_model["winner"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=3, random_state=42)
    clf.fit(X, y_enc)

    scores = cross_val_score(clf, X, y_enc, cv=min(5, len(df_model)), scoring="accuracy")
    print(f"\nDecision Tree (depth={clf.get_depth()}, n={len(df_model)} samples):")
    print(f"  Classes: {list(le.classes_)}")
    print(f"  CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    rules_text = export_text(clf, feature_names=available)
    print(f"\nTree Rules:\n{rules_text}")

    rules_path = os.path.join(output_dir, "decision_tree_rules.txt")
    with open(rules_path, "w") as f:
        f.write(f"Decision Tree — Best Sampling Approach Predictor\n")
        f.write(f"Classes: {list(le.classes_)}\n")
        f.write(f"CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}\n\n")
        f.write(rules_text)
    print(f"Saved: {rules_path}")

    # Decision tree plot
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(max(16, clf.get_n_leaves() * 2), 10))
    plot_tree(
        clf,
        feature_names=available,
        class_names=list(le.classes_),
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
        impurity=False,
    )
    ax.set_title(
        f"Decision Tree: Best Sampling Approach\n"
        f"CV Accuracy: {scores.mean():.2f} ± {scores.std():.2f}  (n={len(df_model)})",
        fontsize=16,
        fontfamily=FONT,
    )
    tree_path = os.path.join(output_dir, "decision_tree.pdf")
    plt.savefig(tree_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {tree_path}")

    # Feature importances
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        range(len(importances)),
        importances[sorted_idx],
        color="#1f77b4",
        edgecolor="white",
    )
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(
        [available[i] for i in sorted_idx],
        rotation=35,
        ha="right",
        fontsize=11,
        fontfamily=FONT,
    )
    ax.set_ylabel("Gini Importance", fontsize=13, fontfamily=FONT)
    ax.set_xlabel("Feature", fontsize=13, fontfamily=FONT)
    ax.set_title("Decision Tree Feature Importances", fontsize=16, fontfamily=FONT)
    for bar, imp in zip(bars, importances[sorted_idx]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{imp:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontfamily=FONT,
        )
    ax.tick_params(which="both", direction="in")
    ax.minorticks_on()
    ax.set_ylim(0, max(importances) * 1.15)
    plt.tight_layout()
    imp_path = os.path.join(output_dir, "feature_importances.pdf")
    plt.savefig(imp_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {imp_path}")

    return clf, le, scores


def print_summary(df):
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY  ({len(df)} literal–model pairs)")
    print(f"{'='*60}")
    print(f"\nWinner distribution:\n{df['winner'].value_counts().to_string()}")
    print(f"\nQuality diff (worlds − programs):")
    print(f"  mean : {df['quality_diff'].mean():.4f}")
    print(f"  std  : {df['quality_diff'].std():.4f}")
    print(f"  min  : {df['quality_diff'].min():.4f}")
    print(f"  max  : {df['quality_diff'].max():.4f}")
    print(f"\nModels processed: {df['model'].nunique()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Correlation and decision tree analysis for sampling approach."
    )
    parser.add_argument(
        "--csv", required=True, help="Path to experiment CSV (from main_mmm_csv.py)"
    )
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save output plots and reports"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_prepare(args.csv)
    print_summary(df)

    plot_correlation_heatmap(
        df, os.path.join(args.output_dir, "correlation_heatmap.pdf")
    )
    plot_scatter_quality(df, os.path.join(args.output_dir, "scatter_quality.pdf"))
    fit_and_plot_decision_tree(df, args.output_dir)
