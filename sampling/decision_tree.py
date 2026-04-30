"""
Build a decision tree to recommend the best sampling approach (Worlds vs
Programs) for a DeLP3E literal, using only features that can be computed
WITHOUT running the experiment itself.

Excluded predictors (would leak the answer or require exact ground truth):
    - exact_l, exact_u, exact_width
    - worlds_*, progs_*
    - winner

Reads:  results/all_results.csv
Writes: results/analysis/
    - decision_tree_full.pdf      : tree visualization + feature importance
    - decision_tree_summary.txt   : CV accuracy, rules, importances

Usage:
    python sampling/decision_tree.py
    python sampling/decision_tree.py --max_depth 5
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold

INPUT_CSV  = "results/all_results.csv"
OUTPUT_DIR = "results/analysis"

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 16
COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"

# Features available BEFORE running any sampling — i.e. structural / static.
PREDICTORS = [
    # Model structural features
    "em_var", "n_annots", "n_rules", "annots_minus_em",
    # n_worlds and n_programs are 2^em_var and 2^n_annots — redundant but useful
    # in log space for tree splits
    # Literal features
    "lit_is_negated", "lit_head_def", "lit_head_strict", "lit_body_count",
    "lit_is_fact", "lit_is_ann_fact",
    "lit_complement_body", "lit_complement_head",
    # AF global features
    "af_pct_annotated", "af_avg_annot_vars", "af_avg_connectors",
    "af_avg_body_size", "af_max_body_size",
]


def load_data():
    df = pd.read_csv(INPUT_CSV, sep=";")
    # Coerce big-int columns to float (some > int64)
    for c in ("n_worlds", "n_programs"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    # Drop rows without a valid winner (ties or missing)
    df = df.dropna(subset=["winner"])
    df = df[df["winner"].isin(["worlds", "programs"])].copy()
    df["winner_binary"] = (df["winner"] == "worlds").astype(int)
    return df


def cross_validated_accuracy(X, y, max_depth_range=range(1, 11), seed=42):
    """Return CV mean accuracy and std for each depth."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    for d in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=d, random_state=seed,
                                    class_weight="balanced")
        scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")
        results.append((d, scores.mean(), scores.std()))
    return results


def plot_tree_and_importance(dt, predictors, depth, acc, out_path):
    fig = plt.figure(figsize=(20, 10))

    # Right: feature importances
    ax_imp = plt.subplot2grid((1, 3), (0, 0))
    importances = pd.Series(dt.feature_importances_, index=predictors)
    importances = importances[importances > 0].sort_values(ascending=True)

    bars = ax_imp.barh(importances.index, importances.values,
                       color=COLOR_WORLDS, edgecolor="white", height=0.6)
    for bar, val in zip(bars, importances.values):
        ax_imp.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=10, fontname=FONT)
    ax_imp.set_xlabel("Gini importance", fontsize=LABEL_SIZE, fontname=FONT)
    ax_imp.set_title("Feature importances", fontsize=TITLE_SIZE,
                     fontname=FONT, pad=10)
    ax_imp.tick_params(which="both", direction="in", labelsize=11)

    # Tree
    ax_tree = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    plot_tree(dt, feature_names=predictors,
              class_names=["programs", "worlds"],
              filled=True, rounded=True, ax=ax_tree,
              fontsize=9, impurity=False, proportion=True)
    ax_tree.set_title(f"Decision tree (depth={depth}, CV acc={acc:.3f})",
                      fontsize=TITLE_SIZE, fontname=FONT, pad=10)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_depth", type=int, default=None,
                        help="Force a specific depth (default: pick best by CV)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    df = load_data()
    predictors = [p for p in PREDICTORS if p in df.columns]
    X = df[predictors].values
    y = df["winner_binary"].values

    n_total = len(df)
    n_w = (y == 1).sum()
    n_p = (y == 0).sum()

    lines = []
    lines.append(f"=== Dataset ===")
    lines.append(f"Total rows: {n_total}  (worlds={n_w}, programs={n_p})")
    lines.append(f"Configs: {df['config'].nunique()} ({sorted(df['config'].unique())})")
    lines.append(f"Predictors used: {len(predictors)}")
    for p in predictors:
        lines.append(f"  - {p}")
    lines.append("")

    # Pick depth
    if args.max_depth is None:
        lines.append("=== Cross-validated accuracy by depth ===")
        results = cross_validated_accuracy(X, y, range(1, 11), args.seed)
        best_d, best_acc, best_std = max(results, key=lambda r: r[1])
        for d, m, s in results:
            mark = "  <-- best" if d == best_d else ""
            lines.append(f"  depth={d:2d}: {m:.3f} ± {s:.3f}{mark}")
        lines.append("")
        depth = best_d
        acc = best_acc
    else:
        depth = args.max_depth
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        dt = DecisionTreeClassifier(max_depth=depth, random_state=args.seed,
                                    class_weight="balanced")
        scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")
        acc = scores.mean()
        lines.append(f"=== Forced depth={depth} ===")
        lines.append(f"CV accuracy: {acc:.3f} ± {scores.std():.3f}")
        lines.append("")

    # Final tree fitted on all data
    dt = DecisionTreeClassifier(max_depth=depth, random_state=args.seed,
                                class_weight="balanced")
    dt.fit(X, y)

    # Importances
    importances = pd.Series(dt.feature_importances_, index=predictors)
    importances = importances[importances > 0].sort_values(ascending=False)
    lines.append("=== Feature importances (non-zero) ===")
    for feat, imp in importances.items():
        lines.append(f"  {feat:25s}  {imp:.4f}")
    lines.append("")

    # Rules
    lines.append("=== Decision rules ===")
    lines.append(export_text(dt, feature_names=predictors))

    # Save artifacts
    pdf_path = os.path.join(OUTPUT_DIR, "decision_tree_full.pdf")
    plot_tree_and_importance(dt, predictors, depth, acc, pdf_path)

    txt_path = os.path.join(OUTPUT_DIR, "decision_tree_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {txt_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
