"""
Build an "abstract" decision tree / random forest using ONLY scale-invariant
features at categorical granularity.

Goal: replace concrete numeric thresholds (e.g. em_n_arcs <= 15.5) with
abstract rules (e.g. "em_density = low AND af_richness = sparse") and check
that we don't lose predictive power.

Workflow:
1. Load the enriched CSV.
2. DERIVE ratio / normalised features. Drop all raw scale-dependent ones.
3. BIN each numeric feature into N levels (default 3 = low/medium/high) using
   either theoretical thresholds (for features bounded in [0,1]) or quantile
   thresholds (computed once from the full dataset).
4. Encode categorical features and train BOTH a single decision tree and a
   random forest on the abstract features only.
5. Validate:
       - 5-fold stratified cross-validation
       - leave-one-config-out cross-validation
       - compare with the baseline tree built on raw features
6. Output PDF tree (natural-language splits), text rules and a summary.

Usage:
    python sampling/abstract_tree.py \
        --input_csv results_5min/all_results.csv \
        --output_dir results_5min/analysis \
        --n_bins 3
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedKFold,
                                     cross_val_score)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 15
COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"


# ── Feature definitions ───────────────────────────────────────────────────────
# Derived ratio / normalised features.
# Each entry: (output_name, function(row) -> float).
# Functions return NaN if any input column is missing/zero.

def _safe_div(num, den):
    try:
        if den is None or pd.isna(den) or den == 0:
            return np.nan
        return float(num) / float(den)
    except Exception:
        return np.nan


DERIVED_FEATURES = [
    # EM density / entropy ratios
    ("em_density",            lambda r: _safe_div(r.get("em_n_arcs"), r.get("em_var"))),
    ("em_entropy_per_var",    lambda r: _safe_div(r.get("em_entropy"), r.get("em_var"))),
    ("em_treewidth_ratio",    lambda r: _safe_div(r.get("em_treewidth"), r.get("em_var"))),
    ("em_max_indeg_ratio",    lambda r: _safe_div(r.get("em_max_in_degree"), r.get("em_var"))),
    # AM ratios
    ("am_arg_density",        lambda r: _safe_div(r.get("am_n_arguments"), r.get("n_rules"))),
    ("am_defeat_ratio",       lambda r: _safe_div(r.get("am_n_defeaters"), r.get("am_n_arguments"))),
    ("am_tree_density",       lambda r: _safe_div(r.get("am_n_trees"), r.get("am_n_arguments"))),
    # AF coverage
    ("af_em_coverage",        lambda r: _safe_div(r.get("af_n_em_vars_used"), r.get("em_var"))),
    # Literal-rate features
    ("lit_head_def_rate",     lambda r: _safe_div(r.get("lit_head_def"), r.get("n_rules"))),
    ("lit_head_strict_rate",  lambda r: _safe_div(r.get("lit_head_strict"), r.get("n_rules"))),
    ("lit_body_rate",         lambda r: _safe_div(r.get("lit_body_count"), r.get("n_rules"))),
    ("lit_complement_body_rate", lambda r: _safe_div(r.get("lit_complement_body"), r.get("n_rules"))),
    ("lit_complement_head_rate", lambda r: _safe_div(r.get("lit_complement_head"), r.get("n_rules"))),
]

# Features kept as-is (already scale-invariant or categorical).
PASS_THROUGH = [
    # Literal binary indicators
    "lit_is_negated", "lit_is_fact", "lit_is_ann_fact",
    # AF features already normalised
    "af_pct_annotated", "af_avg_annot_vars", "af_avg_connectors",
    "af_avg_complexity",
    # AM averages already normalised per argument/tree
    "am_avg_def_rules", "am_avg_arg_lines", "am_avg_height_lines",
    # Network metrics already scale-invariant
    "em_graph_density", "em_clustering_coef", "em_avg_shortest_path",
    "em_deg_centrality_avg", "em_deg_centrality_max", "em_deg_centrality_std",
    "em_closeness_avg", "em_closeness_max",
    "em_betweenness_avg", "em_betweenness_max",
    "em_eigenvector_avg", "em_eigenvector_max",
    "em_in_degree_gini",
    "am_attack_density", "am_attack_clustering", "am_attack_avg_path",
    "am_attack_reciprocity",
    "am_attack_deg_avg", "am_attack_deg_max", "am_attack_deg_std",
    "am_attack_closeness_avg", "am_attack_closeness_max",
    "am_attack_betweenness_avg", "am_attack_betweenness_max",
    "am_attack_eigenvector_avg", "am_attack_eigenvector_max",
    "am_attack_in_degree_gini",
    "af_bipartite_density", "af_var_usage_gini",
    "af_var_coverage", "af_rule_to_var_ratio",
]

# Categorical features built from sign / direction
def _space_orientation(r) -> str:
    v = r.get("annots_minus_em")
    if v is None or pd.isna(v):
        return "unknown"
    if v < -1:
        return "programs-favored"
    if v > 1:
        return "worlds-favored"
    return "balanced"


SIGN_FEATURES = [("space_orientation", _space_orientation)]


# Features with a known theoretical [0,1] (or similar) range — use fixed thresholds
THEORETICAL_BINS = {
    # Densities, fractions and gini coefficients all live in [0, 1]
    "em_density":              [0.0, 0.33, 0.66, 1.0],
    "em_entropy_per_var":      [0.0, 0.5, 0.9, 1.0001],
    "em_treewidth_ratio":      [0.0, 0.15, 0.3, 1.0001],
    "em_max_indeg_ratio":      [0.0, 0.2, 0.4, 1.0001],
    "em_graph_density":        [0.0, 0.1, 0.3, 1.0001],
    "em_clustering_coef":      [0.0, 0.1, 0.3, 1.0001],
    "em_in_degree_gini":       [0.0, 0.33, 0.66, 1.0001],
    "em_deg_centrality_max":   [0.0, 0.33, 0.66, 1.0001],
    "am_arg_density":          [0.0, 0.5, 1.5, np.inf],
    "am_defeat_ratio":         [0.0, 0.5, 1.5, np.inf],
    "am_tree_density":         [0.0, 0.33, 0.66, 1.0001],
    "am_attack_density":       [0.0, 0.1, 0.3, 1.0001],
    "am_attack_clustering":    [0.0, 0.1, 0.3, 1.0001],
    "am_attack_reciprocity":   [0.0, 0.1, 0.5, 1.0001],
    "am_attack_in_degree_gini":[0.0, 0.33, 0.66, 1.0001],
    "af_em_coverage":          [0.0, 0.33, 0.66, 1.0001],
    "af_pct_annotated":        [0.0, 0.3, 0.7, 1.0001],
    "af_bipartite_density":    [0.0, 0.1, 0.3, 1.0001],
    "af_var_usage_gini":       [0.0, 0.33, 0.66, 1.0001],
    "af_var_coverage":         [0.0, 0.33, 0.66, 1.0001],
    "lit_head_def_rate":       [0.0, 0.05, 0.15, 1.0001],
    "lit_head_strict_rate":    [0.0, 0.05, 0.15, 1.0001],
    "lit_body_rate":           [0.0, 0.05, 0.15, 1.0001],
    "lit_complement_body_rate":[0.0, 0.05, 0.15, 1.0001],
    "lit_complement_head_rate":[0.0, 0.05, 0.15, 1.0001],
}

LEVEL_LABELS = ["low", "medium", "high"]


def bin_with_theory_or_quantile(df: pd.DataFrame, col: str, n_bins: int,
                                report: dict) -> pd.Series:
    """Bin a numeric column. Use theoretical thresholds if defined; else quantile."""
    s = df[col].dropna()
    if len(s) == 0:
        report[col] = "all-NaN"
        return pd.Series([np.nan] * len(df), index=df.index)

    if col in THEORETICAL_BINS and n_bins == 3:
        bins = THEORETICAL_BINS[col]
        method = "theoretical"
    else:
        # Quantile-based bins
        try:
            quantiles = np.linspace(0, 1, n_bins + 1)
            bins = list(np.unique(np.quantile(s, quantiles)))
            if len(bins) < n_bins + 1:
                # Not enough variation; fallback to constant
                report[col] = "constant"
                return pd.Series(["medium"] * len(df), index=df.index)
            bins[0] = -np.inf
            bins[-1] = np.inf
            method = "quantile"
        except Exception:
            report[col] = "failed"
            return pd.Series([np.nan] * len(df), index=df.index)

    labels = LEVEL_LABELS if n_bins == 3 else [f"q{i+1}" for i in range(n_bins)]
    binned = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    report[col] = f"{method}: {[round(b, 3) for b in bins]}"
    return binned.astype("string")


def build_abstract_dataframe(df: pd.DataFrame, n_bins: int):
    out = pd.DataFrame(index=df.index)
    threshold_report = {}

    for name, fn in DERIVED_FEATURES:
        out[name] = df.apply(fn, axis=1)
    for col in PASS_THROUGH:
        if col in df.columns:
            out[col] = df[col]

    # Numeric -> categorical
    numeric_cols = [c for c in out.columns
                    if pd.api.types.is_numeric_dtype(out[c])
                    and c not in ("lit_is_negated", "lit_is_fact", "lit_is_ann_fact")]
    for col in numeric_cols:
        out[col] = bin_with_theory_or_quantile(out, col, n_bins, threshold_report)

    # Sign / direction features
    for name, fn in SIGN_FEATURES:
        out[name] = df.apply(fn, axis=1)
        threshold_report[name] = "categorical"

    return out, threshold_report


# ── Encoding ─────────────────────────────────────────────────────────────────
def encode_for_sklearn(X_cat: pd.DataFrame):
    """Stable integer encoding (low=0, medium=1, high=2) preferred where possible."""
    X = pd.DataFrame(index=X_cat.index)
    encoders = {}
    for col in X_cat.columns:
        vals = X_cat[col].fillna("missing").astype(str)
        unique = sorted(vals.unique())
        preferred = ["low", "medium", "high"]
        if all(v in preferred + ["missing"] for v in unique):
            order = [v for v in preferred if v in unique] + (["missing"] if "missing" in unique else [])
        else:
            order = unique
        mapping = {v: i for i, v in enumerate(order)}
        X[col] = vals.map(mapping)
        encoders[col] = order
    return X, encoders


# ── Validation ───────────────────────────────────────────────────────────────
def cv_accuracy(clf, X, y, seed=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return cross_val_score(clf, X, y, cv=cv, scoring="accuracy")


def loco_accuracy(clf, X, y, groups):
    logo = LeaveOneGroupOut()
    scores = cross_val_score(clf, X, y, cv=logo, groups=groups, scoring="accuracy")
    return scores


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_tree_pdf(dt, predictors, depth, acc, encoders, out_path, title):
    fig = plt.figure(figsize=(20, 10))
    ax_imp = plt.subplot2grid((1, 3), (0, 0))
    importances = pd.Series(dt.feature_importances_, index=predictors)
    importances = importances[importances > 0].sort_values(ascending=True)
    bars = ax_imp.barh(importances.index, importances.values,
                       color=COLOR_WORLDS, edgecolor="white", height=0.6)
    for bar, val in zip(bars, importances.values):
        ax_imp.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9, fontname=FONT)
    ax_imp.set_xlabel("Gini importance", fontsize=LABEL_SIZE, fontname=FONT)
    ax_imp.set_title("Feature importances", fontsize=TITLE_SIZE,
                     fontname=FONT, pad=10)
    ax_imp.tick_params(direction="in", labelsize=10)

    ax_tree = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    plot_tree(dt, feature_names=predictors,
              class_names=["programs", "worlds"], filled=True, rounded=True,
              ax=ax_tree, fontsize=8, impurity=False, proportion=True)
    ax_tree.set_title(f"{title}  (depth={depth}, CV acc={acc:.3f})",
                      fontsize=TITLE_SIZE, fontname=FONT, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def render_natural_rules(dt, predictors, encoders) -> str:
    """Translate sklearn integer thresholds back to category labels."""
    from sklearn.tree import _tree
    tree = dt.tree_
    lines = []

    def recurse(node, depth):
        indent = "  " * depth
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            feat = predictors[tree.feature[node]]
            thr  = tree.threshold[node]
            order = encoders.get(feat, [])
            # For "<= thr" predicate, the categories included are those whose
            # encoded id <= thr. Translate to a list of names.
            lo_idx = [i for i in range(len(order)) if i <= thr]
            hi_idx = [i for i in range(len(order)) if i > thr]
            lo_names = [order[i] for i in lo_idx] or ["?"]
            hi_names = [order[i] for i in hi_idx] or ["?"]
            lines.append(f"{indent}IF {feat} in {{{', '.join(lo_names)}}}:")
            recurse(tree.children_left[node], depth + 1)
            lines.append(f"{indent}ELSE  ({feat} in {{{', '.join(hi_names)}}}):")
            recurse(tree.children_right[node], depth + 1)
        else:
            counts = tree.value[node][0]
            n_progs, n_worlds = counts[0], counts[1]
            cls = "worlds" if n_worlds >= n_progs else "programs"
            lines.append(f"{indent}-> recommend {cls.upper()}  "
                         f"(n_progs={int(n_progs)}, n_worlds={int(n_worlds)})")
    recurse(0, 0)
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results/all_results.csv")
    parser.add_argument("--output_dir", default="results/analysis")
    parser.add_argument("--n_bins", type=int, default=3,
                        help="Number of categorical levels (3 = low/medium/high)")
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

    # Build abstract feature dataframe
    X_cat, thresholds = build_abstract_dataframe(df, args.n_bins)
    print(f"\nAbstract features: {X_cat.shape[1]}")

    X, encoders = encode_for_sklearn(X_cat)
    y = df["winner_binary"].values
    groups = df["config"].values

    # ── Train trees at multiple depths, pick best by 5-fold CV ───────────────
    print("\nCross-validated accuracy by depth (abstract):")
    best_d, best_acc = 1, 0
    for d in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=d, random_state=args.seed,
                                    class_weight="balanced")
        scores = cv_accuracy(dt, X, y, args.seed)
        print(f"  depth={d:2d}: {scores.mean():.3f} ± {scores.std():.3f}")
        if scores.mean() > best_acc:
            best_d, best_acc = d, scores.mean()
    print(f"  -> best depth = {best_d} (acc={best_acc:.3f})")

    dt = DecisionTreeClassifier(max_depth=best_d, random_state=args.seed,
                                class_weight="balanced")
    dt.fit(X, y)

    # Random forest for feature importances on the abstract space
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed,
                                class_weight="balanced", n_jobs=-1)
    rf_scores = cv_accuracy(rf, X, y, args.seed)
    rf.fit(X, y)
    print(f"\nRandom forest CV accuracy: {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")

    # ── Leave-one-config-out ──────────────────────────────────────────────────
    print("\nLeave-one-config-out (tests generalisation to unseen configs):")
    loco_dt = loco_accuracy(
        DecisionTreeClassifier(max_depth=best_d, random_state=args.seed,
                               class_weight="balanced"), X, y, groups)
    loco_rf = loco_accuracy(rf, X, y, groups)
    print(f"  tree mean acc: {loco_dt.mean():.3f} ± {loco_dt.std():.3f}")
    print(f"  RF   mean acc: {loco_rf.mean():.3f} ± {loco_rf.std():.3f}")

    # ── Baseline: same exercise with the RAW features ────────────────────────
    raw_predictors = [c for c in (
        "em_var", "n_annots", "n_rules", "annots_minus_em",
        "em_n_arcs", "em_treewidth", "em_entropy",
        "am_n_arguments", "am_n_defeaters", "am_n_trees",
        "af_pct_annotated", "af_n_em_vars_used") if c in df.columns]
    df_raw = df.copy()
    for c in raw_predictors:
        df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
        df_raw[c] = df_raw[c].fillna(df_raw[c].median())
    X_raw = df_raw[raw_predictors].values
    print("\nBaseline (RAW numeric features):")
    best_raw_d, best_raw_acc = 1, 0
    for d in range(1, 11):
        scores = cv_accuracy(
            DecisionTreeClassifier(max_depth=d, random_state=args.seed,
                                   class_weight="balanced"), X_raw, y, args.seed)
        if scores.mean() > best_raw_acc:
            best_raw_d, best_raw_acc = d, scores.mean()
    print(f"  best depth={best_raw_d}, CV acc={best_raw_acc:.3f}")
    loco_raw = loco_accuracy(
        DecisionTreeClassifier(max_depth=best_raw_d, random_state=args.seed,
                               class_weight="balanced"), X_raw, y, groups)
    print(f"  LOCO mean acc: {loco_raw.mean():.3f} ± {loco_raw.std():.3f}")

    # ── Outputs ──────────────────────────────────────────────────────────────
    pdf_path = os.path.join(args.output_dir, f"abstract_tree_n{args.n_bins}.pdf")
    plot_tree_pdf(dt, list(X.columns), best_d, best_acc, encoders,
                  pdf_path, "Abstract decision tree")

    rules_path = os.path.join(args.output_dir,
                              f"abstract_tree_rules_n{args.n_bins}.txt")
    with open(rules_path, "w") as f:
        f.write("Decision tree expressed with categorical labels\n")
        f.write("================================================\n\n")
        f.write(render_natural_rules(dt, list(X.columns), encoders))
        f.write("\n\n\nExport (raw sklearn rule text):\n")
        f.write(export_text(dt, feature_names=list(X.columns)))
    print(f"  Saved: {rules_path}")

    summary_path = os.path.join(args.output_dir,
                                f"abstract_tree_summary_n{args.n_bins}.txt")
    lines = []
    lines.append("=== Abstract decision tree summary ===")
    lines.append(f"Input CSV: {args.input_csv}")
    lines.append(f"Rows: {len(df)}  configs: {df.config.nunique()}")
    lines.append(f"Abstract features: {X.shape[1]}  bins: {args.n_bins}")
    lines.append("")
    lines.append("=== Validation ===")
    lines.append(f"5-fold CV (abstract tree, depth={best_d}): "
                 f"{best_acc:.3f}")
    lines.append(f"5-fold CV (random forest):                  "
                 f"{rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
    lines.append(f"LOCO (abstract tree):                        "
                 f"{loco_dt.mean():.3f} ± {loco_dt.std():.3f}")
    lines.append(f"LOCO (random forest):                        "
                 f"{loco_rf.mean():.3f} ± {loco_rf.std():.3f}")
    lines.append("")
    lines.append("=== Baseline (raw features) ===")
    lines.append(f"5-fold CV (raw tree, depth={best_raw_d}):  "
                 f"{best_raw_acc:.3f}")
    lines.append(f"LOCO (raw tree):                            "
                 f"{loco_raw.mean():.3f} ± {loco_raw.std():.3f}")
    lines.append("")
    lines.append("=== Feature importance (RF, abstract) ===")
    imp = pd.Series(rf.feature_importances_, index=X.columns)
    for feat, val in imp.sort_values(ascending=False).head(20).items():
        lines.append(f"  {feat:35s}  {val:.4f}")
    lines.append("")
    lines.append("=== Binning thresholds ===")
    for col, info in sorted(thresholds.items()):
        lines.append(f"  {col:30s}  {info}")

    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {summary_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
