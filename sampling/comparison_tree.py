"""
Decision tree whose every split is a PAIRWISE COMPARISON between two
measurable features of the DeLP3E model (e.g. ``em_treewidth < am_avg_arg_lines``).

The whole point: no split depends on an absolute numeric threshold that came
from the training distribution. Each rule expresses a relative ordering
between two structural / informational properties of the model, so it
remains meaningful for any new program no matter how its features are
scaled in absolute terms.

Procedure:
    1. Pick every numeric column of the enriched CSV that is NOT
       an output / ground-truth / identifier / trivially redundant.
       (~60 features.)
    2. For each unordered pair (A, B), construct the binary feature
       ``A_LT_B = (A < B)``.
    3. Drop comparisons whose variance is < ``--variance_threshold``
       (default 0.05). Near-constant comparisons don't discriminate.
    4. Train a decision tree at several depths and run 5-fold +
       leave-one-config-out cross-validation.
    5. Compare with the baseline tree trained on raw numeric features
       (same as in decision_tree.py) so we can tell whether the
       comparison-only restriction costs accuracy.
    6. Save a custom-rendered PDF (leaves coloured) and natural-language
       rules where each internal node reads "A < B".

Reads:   results_5min/all_results.csv (after enrich_dataset + enrich_network).
Writes:  results_5min/analysis/
    - comparison_tree_d{D}.pdf
    - comparison_tree_rules_d{D}.txt
    - comparison_tree_summary.txt
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score)
from sklearn.tree import DecisionTreeClassifier, export_text

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 15
COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"

# Anything that is NOT a measurable a-priori property of the model is excluded.
EXCLUDED = {
    # Identifiers
    "config", "model", "literal",
    # Outputs / ground truth — unknown for new programs
    "exact_l", "exact_u", "exact_width",
    "worlds_l", "worlds_u", "worlds_quality",
    "worlds_solver_time", "worlds_delp_calls",
    "progs_l", "progs_u", "progs_quality",
    "progs_solver_time", "progs_delp_calls",
    "winner", "winner_binary", "quality_diff",
    # Trivial log redundancies — the tree gets the same info from em_var / n_annots
    "n_worlds", "n_programs",
    "annots_minus_em",
}


# ── Feature engineering ─────────────────────────────────────────────────────
def select_measurable_features(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDED:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        cols.append(c)
    return sorted(cols)


def impute_median(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())
    return out


def make_pair_comparisons(df: pd.DataFrame, features: list[str],
                          variance_threshold: float):
    """
    For every unordered pair (A, B), produce one binary feature
    ``A_LT_B = (A < B)`` and keep it iff its variance >= threshold.

    Returns
    -------
    X       : DataFrame (n_rows x n_kept_pairs)
    pairs   : list[(A, B)] in the same order as X.columns
    """
    cols: dict[str, np.ndarray] = {}
    pairs: list[tuple[str, str]] = []
    for i, a in enumerate(features):
        col_a = df[a].astype(float).values
        for b in features[i + 1:]:
            col_b = df[b].astype(float).values
            vals = (col_a < col_b).astype(np.int8)
            v = float(vals.var())
            if v >= variance_threshold:
                name = f"{a}_LT_{b}"
                cols[name] = vals
                pairs.append((a, b))
    return pd.DataFrame(cols, index=df.index), pairs


# ── Leaf counts (robust against class_weight reweighting) ───────────────────
def compute_leaf_counts(dt, X, y):
    leaf_ids = dt.apply(X)
    counts: dict[int, tuple[int, int]] = {}
    for lid, label in zip(leaf_ids, y):
        n_p, n_w = counts.get(lid, (0, 0))
        if label == 1:
            n_w += 1
        else:
            n_p += 1
        counts[lid] = (n_p, n_w)
    return {lid: (n_p, n_w, n_p + n_w) for lid, (n_p, n_w) in counts.items()}


# ── Natural-language rules ──────────────────────────────────────────────────
def split_phrases(feat_name: str) -> tuple[str, str]:
    """'A_LT_B' -> ('A < B', 'A >= B')."""
    a, b = feat_name.split("_LT_")
    return f"{a} < {b}", f"{a} >= {b}"


def render_rules(dt, predictors, leaf_counts) -> str:
    from sklearn.tree import _tree
    tree = dt.tree_
    lines = []

    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feat = predictors[tree.feature[node]]
            yes_p, no_p = split_phrases(feat)
            # children_right corresponds to feature > threshold (i.e. == 1, comparison TRUE)
            lines.append(f"{indent}IF {yes_p}:")
            recurse(tree.children_right[node], depth + 1)
            lines.append(f"{indent}ELSE  ({no_p}):")
            recurse(tree.children_left[node], depth + 1)
        else:
            n_p, n_w, total = leaf_counts.get(node, (0, 0, 0))
            cls = "WORLDS" if n_w >= n_p else "PROGRAMS"
            purity = max(n_p, n_w) / total if total else 0
            lines.append(f"{indent}-> recommend {cls}  "
                         f"(n_progs={n_p}, n_worlds={n_w}, "
                         f"total={total}, purity={purity * 100:.0f}%)")

    recurse(0, 0)
    return "\n".join(lines)


# ── PDF renderer (custom, leaf-only coloring) ───────────────────────────────
def _layout_tree(dt):
    from sklearn.tree import _tree
    tree = dt.tree_
    is_leaf = (tree.children_left == _tree.TREE_LEAF)

    leaf_x = {}
    counter = [0]
    def visit(node):
        if is_leaf[node]:
            leaf_x[node] = counter[0]
            counter[0] += 1
        else:
            visit(tree.children_left[node])
            visit(tree.children_right[node])
    visit(0)

    positions = {}
    def assign_x(node):
        if is_leaf[node]:
            x = leaf_x[node]
        else:
            xl = assign_x(tree.children_left[node])
            xr = assign_x(tree.children_right[node])
            x = (xl + xr) / 2
        positions[node] = x
        return x
    assign_x(0)

    depths = {}
    def assign_depth(node, d):
        depths[node] = d
        if not is_leaf[node]:
            assign_depth(tree.children_left[node], d + 1)
            assign_depth(tree.children_right[node], d + 1)
    assign_depth(0, 0)

    return positions, depths, is_leaf, leaf_x


def plot_tree_pdf(dt, predictors, depth, acc, out_path, title, leaf_counts):
    from sklearn.tree import _tree
    tree = dt.tree_
    n_nodes = tree.node_count
    positions, depths, is_leaf, leaf_x = _layout_tree(dt)
    n_leaves = len(leaf_x)
    max_depth = max(depths.values()) if depths else 0

    fig_w = max(12, n_leaves * 1.5)
    fig_h = max(6, (max_depth + 1) * 2.0)

    fig = plt.figure(figsize=(fig_w + 5, fig_h))
    gs = fig.add_gridspec(1, 5)

    ax_imp = fig.add_subplot(gs[0, 0])
    importances = pd.Series(dt.feature_importances_, index=predictors)
    importances = importances[importances > 0].sort_values(ascending=True)
    # Translate "A_LT_B" -> "A < B" for the importance plot
    importances.index = [split_phrases(n)[0] for n in importances.index]
    bars = ax_imp.barh(importances.index, importances.values,
                       color="#666666", edgecolor="white", height=0.6)
    for bar, val in zip(bars, importances.values):
        ax_imp.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=8, fontname=FONT)
    ax_imp.set_xlabel("Gini importance", fontsize=LABEL_SIZE, fontname=FONT)
    ax_imp.set_title("Feature importances", fontsize=TITLE_SIZE,
                     fontname=FONT, pad=10)
    ax_imp.tick_params(direction="in", labelsize=9)

    ax = fig.add_subplot(gs[0, 1:])
    ax.axis("off")
    ax.set_xlim(-1, n_leaves + 1)
    ax.set_ylim(-max_depth - 0.6, 0.6)

    # Edges
    for node in range(n_nodes):
        if is_leaf[node]:
            continue
        x_p, y_p = positions[node], -depths[node]
        for child in (tree.children_left[node], tree.children_right[node]):
            x_c, y_c = positions[child], -depths[child]
            ax.plot([x_p, x_c], [y_p - 0.16, y_c + 0.16],
                    color="#999999", lw=0.9, zorder=1)

    # Nodes
    for node in range(n_nodes):
        x, y = positions[node], -depths[node]
        if is_leaf[node]:
            n_p, n_w, total = leaf_counts.get(node, (0, 0, 0))
            if n_w >= n_p:
                cls = "WORLDS"; color = COLOR_WORLDS
                purity = n_w / total if total else 0
            else:
                cls = "PROGRAMS"; color = COLOR_PROGRAMS
                purity = n_p / total if total else 0
            box_text = f"{cls}\n{total} samples\n{purity*100:.0f}% pure"
            ax.text(x, y, box_text, ha="center", va="center",
                    fontsize=9, fontname=FONT, color="white",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=color, edgecolor=color, lw=1.5),
                    zorder=2)
        else:
            feat = predictors[tree.feature[node]]
            yes_p, _ = split_phrases(feat)
            ax.text(x, y, yes_p, ha="center", va="center",
                    fontsize=9, fontname=FONT, color="black",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor="#333333", lw=0.9),
                    zorder=2)
            # Branch labels
            x_l = positions[tree.children_left[node]]
            x_r = positions[tree.children_right[node]]
            y_c = -depths[node] - 0.5
            # left = comparison FALSE; right = comparison TRUE
            ax.text((x + x_l) / 2, y_c, "no", ha="center", va="center",
                    fontsize=7, color="#444444", fontname=FONT, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))
            ax.text((x + x_r) / 2, y_c, "yes", ha="center", va="center",
                    fontsize=7, color="#444444", fontname=FONT, zorder=2,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", edgecolor="none", alpha=0.85))

    ax.set_title(f"{title}  (depth={depth}, CV acc={acc:.3f})",
                 fontsize=TITLE_SIZE, fontname=FONT, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Validation ──────────────────────────────────────────────────────────────
def cv_acc(clf, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv, scoring="accuracy")


def loco_acc(clf, X, y, groups):
    return cross_val_score(clf, X, y, cv=LeaveOneGroupOut(), groups=groups,
                           scoring="accuracy")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results/all_results.csv")
    parser.add_argument("--output_dir", default="results/analysis")
    parser.add_argument("--variance_threshold", type=float, default=0.05,
                        help=(
                            "Drop pairwise comparisons whose variance "
                            "(in [0, 0.25] for binary features) is below "
                            "this value. 0.05 keeps comparisons with at "
                            "least ~5/95 class split."))
    parser.add_argument("--depths", default="3,5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";")
    for c in ("n_worlds", "n_programs"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.dropna(subset=["winner"])
    df = df[df["winner"].isin(["worlds", "programs"])].copy()
    df["winner_binary"] = (df["winner"] == "worlds").astype(int)
    print(f"  {len(df)} rows | {df.config.nunique()} configs")

    features = select_measurable_features(df)
    print(f"\nMeasurable base features: {len(features)}")
    df = impute_median(df, features)

    X_pair, pairs = make_pair_comparisons(
        df, features, args.variance_threshold)
    print(f"Pairwise comparison features (after variance filter "
          f">= {args.variance_threshold}): {X_pair.shape[1]}  "
          f"(out of {len(features) * (len(features) - 1) // 2} possible)")

    y = df["winner_binary"].values
    groups = df["config"].values

    # Train + CV at multiple depths
    print("\nCross-validated accuracy by depth (comparison tree):")
    best_d, best_acc = 1, 0
    for d in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=d, random_state=args.seed,
                                    class_weight="balanced")
        s = cv_acc(dt, X_pair.values, y, args.seed)
        print(f"  depth={d:2d}: {s.mean():.3f} ± {s.std():.3f}")
        if s.mean() > best_acc:
            best_d, best_acc = d, s.mean()
    print(f"  -> best depth = {best_d} (acc={best_acc:.3f})")

    # Leave-one-config-out
    print("\nLeave-one-config-out:")
    loco = loco_acc(
        DecisionTreeClassifier(max_depth=best_d, random_state=args.seed,
                               class_weight="balanced"),
        X_pair.values, y, groups)
    print(f"  comparison tree: {loco.mean():.3f} ± {loco.std():.3f}")

    # Baseline: raw numeric features
    raw_predictors = [c for c in features
                      if c not in ("lit_is_negated", "lit_is_fact",
                                   "lit_is_ann_fact")]
    X_raw = df[raw_predictors].values
    best_raw_d, best_raw_acc = 1, 0
    for d in range(1, 11):
        s = cv_acc(
            DecisionTreeClassifier(max_depth=d, random_state=args.seed,
                                   class_weight="balanced"),
            X_raw, y, args.seed)
        if s.mean() > best_raw_acc:
            best_raw_d, best_raw_acc = d, s.mean()
    loco_raw = loco_acc(
        DecisionTreeClassifier(max_depth=best_raw_d, random_state=args.seed,
                               class_weight="balanced"),
        X_raw, y, groups)
    print(f"\nBaseline (raw numeric, best depth={best_raw_d}):")
    print(f"  5-fold CV: {best_raw_acc:.3f}")
    print(f"  LOCO     : {loco_raw.mean():.3f} ± {loco_raw.std():.3f}")

    # Save trees at the requested depths
    depths_to_save = sorted({int(s.strip()) for s in args.depths.split(",")}
                            | {best_d})

    saved = []
    for fixed_d in depths_to_save:
        dt = DecisionTreeClassifier(max_depth=fixed_d, random_state=args.seed,
                                    class_weight="balanced")
        dt.fit(X_pair.values, y)
        scores = cv_acc(dt, X_pair.values, y, args.seed)
        d_acc = float(scores.mean())
        leaf_counts = compute_leaf_counts(dt, X_pair.values, y)

        pdf_path = os.path.join(args.output_dir,
                                f"comparison_tree_d{fixed_d}.pdf")
        plot_tree_pdf(dt, list(X_pair.columns), fixed_d, d_acc,
                      pdf_path, "Comparison decision tree", leaf_counts)

        rules_path = os.path.join(args.output_dir,
                                  f"comparison_tree_rules_d{fixed_d}.txt")
        with open(rules_path, "w") as f:
            f.write(f"Comparison-only decision tree "
                    f"(depth={fixed_d}, CV acc={d_acc:.3f})\n")
            f.write("=" * 70 + "\n\n")
            f.write(render_rules(dt, list(X_pair.columns), leaf_counts))
            f.write("\n\n\nExport (raw sklearn rule text — feature names "
                    "use 'A_LT_B' for 'A < B'):\n")
            f.write(export_text(dt, feature_names=list(X_pair.columns)))
        print(f"  Saved: {rules_path}")
        saved.append((fixed_d, d_acc))

    # Summary file
    summary = []
    summary.append("=== Comparison decision tree ===")
    summary.append(f"Input CSV: {args.input_csv}")
    summary.append(f"Rows: {len(df)}  configs: {df.config.nunique()}")
    summary.append(f"Base features: {len(features)}")
    summary.append(f"Pairwise comparison features kept "
                   f"(variance >= {args.variance_threshold}): "
                   f"{X_pair.shape[1]}")
    summary.append("")
    summary.append("=== Trees saved ===")
    for d_, a_ in saved:
        summary.append(f"  depth={d_:2d}  CV acc={a_:.3f}")
    summary.append("")
    summary.append("=== Validation ===")
    summary.append(f"Best comparison tree 5-fold CV: depth={best_d} "
                   f"acc={best_acc:.3f}")
    summary.append(f"Comparison tree LOCO          : "
                   f"{loco.mean():.3f} ± {loco.std():.3f}")
    summary.append(f"Baseline raw 5-fold CV         : depth={best_raw_d} "
                   f"acc={best_raw_acc:.3f}")
    summary.append(f"Baseline raw LOCO              : "
                   f"{loco_raw.mean():.3f} ± {loco_raw.std():.3f}")
    summary.append("")

    # Top comparisons by importance (from the best-depth tree)
    dt_best = DecisionTreeClassifier(max_depth=best_d, random_state=args.seed,
                                     class_weight="balanced")
    dt_best.fit(X_pair.values, y)
    imp = pd.Series(dt_best.feature_importances_, index=X_pair.columns)
    summary.append("=== Top comparisons by importance (best depth tree) ===")
    for name, val in imp[imp > 0].sort_values(ascending=False).head(20).items():
        yes_p, _ = split_phrases(name)
        summary.append(f"  {yes_p:60s}  {val:.4f}")

    summary_path = os.path.join(args.output_dir,
                                "comparison_tree_summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(summary))
    print(f"  Saved: {summary_path}")
    print("\n" + "\n".join(summary))


if __name__ == "__main__":
    main()
