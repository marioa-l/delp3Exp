"""
Identify and analyse mismatch cases — literals where the decision-tree
recommendation differs from the actual best sampler.

Workflow:
1. Load `results/all_results.csv`.
2. Train the same decision tree as `decision_tree.py` (structural predictors
   only, no `exact_*` / `worlds_*` / `progs_*`).
3. For each row, compare predicted vs actual winner.
4. A "real mismatch" requires a non-trivial margin (|quality_diff| > 0.01).
5. Compare features between correct and mismatch groups.
6. Plot per-config detail views for configs that have any real mismatches.

Reads:  results/all_results.csv
Writes: results/analysis/
    - mismatch_features.pdf       : feature distributions (correct vs mismatch)
    - mismatch_detail_<cfg>.pdf   : one plot per config with mismatches
    - mismatch_summary.txt        : full numerical report
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

DEFAULT_INPUT_CSV  = "results/all_results.csv"
DEFAULT_OUTPUT_DIR = "results/analysis"
INPUT_CSV  = DEFAULT_INPUT_CSV   # may be overridden by CLI
OUTPUT_DIR = DEFAULT_OUTPUT_DIR  # may be overridden by CLI

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 13, 15
COLOR_CORRECT  = "#0072B2"
COLOR_MISMATCH = "#D55E00"
MARGIN_THRESHOLD = 0.01

# Same predictors as decision_tree.py — structural features only.
PREDICTORS = [
    "em_var", "n_annots", "n_rules", "annots_minus_em",
    "lit_is_negated", "lit_head_def", "lit_head_strict", "lit_body_count",
    "lit_is_fact", "lit_is_ann_fact",
    "lit_complement_body", "lit_complement_head",
    "af_pct_annotated", "af_avg_annot_vars", "af_avg_connectors",
    "af_avg_body_size", "af_max_body_size",
    "am_n_arguments", "am_n_defeaters", "am_n_trees",
    "am_avg_def_rules", "am_avg_arg_lines", "am_avg_height_lines",
    "em_n_arcs", "em_treewidth", "em_avg_in_degree", "em_max_in_degree",
    "em_entropy",
    "af_n_em_vars_used", "af_avg_complexity", "af_max_complexity",
]


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=10)
    ax.minorticks_on()


def load_data():
    df = pd.read_csv(INPUT_CSV, sep=";")
    for c in ("n_worlds", "n_programs"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.dropna(subset=["winner"])
    df = df[df["winner"].isin(["worlds", "programs"])].copy()
    df["winner_binary"] = (df["winner"] == "worlds").astype(int)
    return df


def impute_predictors(df, predictors):
    """Fill NaN in predictor columns with the column median."""
    for col in predictors:
        if col in df.columns and df[col].isna().any():
            med = df[col].median()
            df[col] = df[col].fillna(med)
    return df


def fit_tree_and_predict(df, predictors, seed=42):
    """Train a tree at the best CV depth and return predictions for all rows."""
    X = df[predictors].values
    y = df["winner_binary"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_depth, best_acc = 1, 0
    for d in range(1, 11):
        dt = DecisionTreeClassifier(max_depth=d, random_state=seed,
                                    class_weight="balanced")
        scores = cross_val_score(dt, X, y, cv=cv, scoring="accuracy")
        if scores.mean() > best_acc:
            best_depth, best_acc = d, scores.mean()

    dt = DecisionTreeClassifier(max_depth=best_depth, random_state=seed,
                                class_weight="balanced")
    dt.fit(X, y)
    pred_binary = dt.predict(X)
    df["predicted"] = np.where(pred_binary == 1, "worlds", "programs")
    df["quality_diff"] = df["worlds_quality"] - df["progs_quality"]
    df["margin"] = df["quality_diff"].abs()
    df["mismatch"] = df["predicted"] != df["winner"]
    df["real_mismatch"] = df["mismatch"] & (df["margin"] > MARGIN_THRESHOLD)
    return df, best_depth, best_acc


def plot_feature_comparison(df, out):
    correct = df[~df["real_mismatch"]]
    mismatch = df[df["real_mismatch"]]
    if len(mismatch) == 0:
        print("  No real mismatches — skipping feature comparison plot.")
        return

    feats = PREDICTORS
    n = len(feats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    for ax, feat in zip(axes, feats):
        data = [correct[feat].values, mismatch[feat].values]
        bp = ax.boxplot(data, labels=["correct", "mismatch"],
                        patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=1.2),
                        showfliers=True)
        for patch, c in zip(bp["boxes"], [COLOR_CORRECT, COLOR_MISMATCH]):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)

        try:
            _, p = stats.mannwhitneyu(correct[feat], mismatch[feat],
                                      alternative="two-sided")
            star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            title = f"{feat}\n(p={p:.3f} {star})"
        except Exception:
            title = feat
        style_ax(ax, title=title)

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


def plot_config_detail(df, config_name, out):
    sub = df[df["config"] == config_name].copy()
    sub = sub.sort_values(["real_mismatch", "annots_minus_em", "model", "literal"],
                          ascending=[True, True, True, True]).reset_index(drop=True)

    labels = [f"{r['model']}/{r['literal']}" for _, r in sub.iterrows()]
    x = np.arange(len(sub))
    wq = sub["worlds_quality"].values
    pq = sub["progs_quality"].values
    is_mm = sub["real_mismatch"].values

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.18), 5))
    width = 0.4
    ax.bar(x - width / 2, wq, width, label="worlds quality",
           color=COLOR_CORRECT, alpha=0.85)
    ax.bar(x + width / 2, pq, width, label="progs quality",
           color=COLOR_MISMATCH, alpha=0.85)

    for i, mm in enumerate(is_mm):
        if mm:
            ax.axvspan(i - 0.5, i + 0.5, color="red", alpha=0.10, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(-0.05, 1.15)
    style_ax(ax, ylabel="approximation quality",
             title=f"{config_name}: real mismatches highlighted "
                   f"({is_mm.sum()}/{len(sub)})")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    fig.tight_layout()
    path = os.path.join(out, f"mismatch_detail_{config_name}.pdf")
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    global INPUT_CSV, OUTPUT_DIR
    INPUT_CSV = args.input_csv
    OUTPUT_DIR = args.output_dir

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    df = load_data()
    predictors = [p for p in PREDICTORS if p in df.columns]
    df = impute_predictors(df, predictors)

    print("Training decision tree...")
    df, depth, acc = fit_tree_and_predict(df, predictors)
    print(f"  Best depth={depth}, CV accuracy={acc:.3f}")

    correct = df[~df["real_mismatch"]]
    mismatch = df[df["real_mismatch"]]

    lines = []
    lines.append("=== Dataset ===")
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Tree: depth={depth}, CV accuracy={acc:.3f}")
    lines.append(f"Any-mismatch (pred != winner):     {df['mismatch'].sum()}")
    lines.append(f"Real mismatches (margin > {MARGIN_THRESHOLD}): {df['real_mismatch'].sum()}")
    lines.append(f"Correct (incl. trivial ties):       {len(correct)}")

    lines.append("\n=== Mismatches per config ===")
    per_cfg = df.groupby("config").agg(
        total=("real_mismatch", "size"),
        real_mm=("real_mismatch", "sum"),
    )
    per_cfg["pct"] = (100 * per_cfg["real_mm"] / per_cfg["total"]).round(1)
    lines.append(per_cfg.to_string())

    if len(mismatch) > 0:
        lines.append("\n=== Real mismatches (top 30 by margin) ===")
        cols = ["config", "model", "literal", "annots_minus_em",
                "exact_width", "worlds_quality", "progs_quality",
                "predicted", "winner", "margin"]
        top = mismatch.sort_values("margin", ascending=False).head(30)
        lines.append(top[cols].to_string(index=False))

        lines.append("\n=== Mann–Whitney U: correct vs real mismatches ===")
        lines.append(f"{'feature':25s}  {'median corr':>11s}  {'median mism':>11s}  {'p':>7s}")
        for feat in predictors:
            try:
                _, p = stats.mannwhitneyu(correct[feat], mismatch[feat],
                                          alternative="two-sided")
                med_c = correct[feat].median()
                med_m = mismatch[feat].median()
                flag = "*" if p < 0.05 else " "
                lines.append(f"  {feat:23s}  {med_c:11.3f}  {med_m:11.3f}  {p:7.3f} {flag}")
            except Exception as e:
                lines.append(f"  {feat:23s}  ERROR: {e}")

    print("\n1. Feature comparison plot...")
    plot_feature_comparison(df, OUTPUT_DIR)

    print("\n2. Per-config detail plots...")
    configs_with_mm = sorted(df[df["real_mismatch"]]["config"].unique())
    if not configs_with_mm:
        print("  No configs with real mismatches.")
    else:
        for cfg in configs_with_mm:
            plot_config_detail(df, cfg, OUTPUT_DIR)

    txt_path = os.path.join(OUTPUT_DIR, "mismatch_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n3. Summary written to: {txt_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
