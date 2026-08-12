"""
Retrain the abstract tree on the time-to-exact ground truth with a held-out
split grouped by model, so no literal of a model appears in both train and
test.

Motivation: the original tree was trained on ``winner`` (best interval within
a fixed 5-minute budget), which is a biased proxy for ``gt_winner`` (first to
reach the exact interval). On configs like scm/sms/sss the 5-minute label
calls programs the winner where, given unlimited time, worlds reaches exact
first — which is exactly where the deployed tree failed.

Writes to --output_dir:
    gt_tree_d<depth>.pkl            fitted model + binning state
    gt_tree_report.txt              held-out accuracy, per-config, per-leaf
    gt_tree_predictions.csv         test-split rows with predictions

Usage:
    python sampling/retrain_on_ground_truth.py
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.abstract_tree import (  # noqa: E402
    build_abstract_dataframe, encode_for_sklearn, export_portable,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_tte/all_results.csv")
    parser.add_argument("--output_dir", default="results_tte/analysis")
    parser.add_argument("--n_bins", type=int, default=3)
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv, sep=";")
    df = df.dropna(subset=["gt_winner"])
    df = df[df["gt_winner"].isin(["worlds", "programs"])].copy()
    df = df.reset_index(drop=True)
    y = (df["gt_winner"] == "worlds").astype(int).values
    groups = (df["config"] + "/" + df["model"]).values

    print(f"Loaded {len(df)} resolved rows | {df.config.nunique()} configs "
          f"| {len(set(groups))} distinct models")

    X_cat, thresholds, bins_numeric = build_abstract_dataframe(df, args.n_bins)
    # Reuse the project's encoder so low/medium/high keep their ordinal order
    # (a plain alphabetical sort would scramble it into high/low/medium).
    X, encoders = encode_for_sklearn(X_cat)

    depths = [int(d) for d in args.depths.split(",")]
    lines = ["=== Abstract tree retrained on the time-to-exact ground truth ===",
             f"Input CSV : {args.input_csv}",
             f"Rows      : {len(df)}  ({int((y==0).sum())} programs, "
             f"{int((y==1).sum())} worlds)",
             f"Grouping  : GroupKFold({args.n_splits}) by (config, model) — "
             f"no literal leakage",
             ""]

    gkf = GroupKFold(n_splits=args.n_splits)
    best = None
    for d in depths:
        accs, oof = [], np.zeros(len(df), dtype=int)
        for tr_i, te_i in gkf.split(X, y, groups):
            m = DecisionTreeClassifier(max_depth=d, random_state=args.seed,
                                       class_weight="balanced").fit(
                                           X.iloc[tr_i], y[tr_i])
            p = m.predict(X.iloc[te_i])
            oof[te_i] = p
            accs.append((p == y[te_i]).mean())
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        lines.append(f"depth={d}:  held-out accuracy = "
                     f"{mean_acc:.4f} +/- {std_acc:.4f}")
        if best is None or mean_acc > best[1]:
            best = (d, mean_acc, oof.copy())

    best_d, best_acc, best_oof = best
    lines.append("")
    lines.append(f"-> best depth = {best_d}  (held-out acc = {best_acc:.4f})")
    lines.append("")

    df["pred_gt"] = np.where(best_oof == 1, "worlds", "programs")
    df["match_gt"] = df["pred_gt"] == df["gt_winner"]

    lines.append("=== Held-out accuracy per configuration ===")
    for cfg, g in df.groupby("config"):
        lines.append(f"  {cfg:<8s}  n={len(g):>4d}  acc={g['match_gt'].mean():.3f}")
    lines.append("")

    lines.append("=== Confusion (out-of-fold predictions) ===")
    lines.append(pd.crosstab(df["gt_winner"], df["pred_gt"],
                             rownames=["actual"],
                             colnames=["predicted"]).to_string())
    lines.append("")

    # Refit on all data at the best depth and persist
    final = DecisionTreeClassifier(max_depth=best_d, random_state=args.seed,
                                   class_weight="balanced").fit(X, y)
    lines.append("=== Final tree (refit on all rows) ===")
    lines.append(export_text(final, feature_names=list(X.columns)))

    pkl_path = os.path.join(args.output_dir, f"gt_tree_d{best_d}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({
            "tree": final,
            "encoders": encoders,
            "bins_numeric": bins_numeric,
            "feature_columns": list(X.columns),
            "predictors_categorical": list(X.columns),
            "n_bins": args.n_bins,
            "cv_accuracy": best_acc,
            "depth": best_d,
            "target": "gt_winner",
        }, f)

    json_path = pkl_path[:-4] + ".json"
    export_portable(json_path, final, encoders, bins_numeric,
                    list(X.columns), args.n_bins, best_d, best_acc,
                    "gt_winner")

    report = os.path.join(args.output_dir, "gt_tree_report.txt")
    with open(report, "w") as f:
        f.write("\n".join(lines))
    df.to_csv(os.path.join(args.output_dir, "gt_tree_predictions.csv"),
              sep=";", index=False)

    print("\n".join(lines))
    print(f"\nSaved: {pkl_path}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
