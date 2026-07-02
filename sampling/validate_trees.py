"""
Evaluate the two recommenders (comparison tree and abstract tree) on a
validation CSV built from models the training experiment never saw.

Reads:
    --input_csv (default: results_5min_val/all_results.csv)
    --abstract_pkl (path to the pickle emitted by abstract_tree.py)

Writes to --output_dir:
    validation_predictions.csv — original CSV plus prediction columns
    validation_report.txt      — accuracy overall / per config / per region

Usage:
    python sampling/validate_trees.py \
        --input_csv results_5min_val/all_results.csv \
        --abstract_pkl results_5min/analysis/abstract_tree_n3_d3.pkl \
        --output_dir results_5min_val/analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.recommend import recommend_comparison, AbstractRecommender  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",
                        default="results_5min_val/all_results.csv")
    parser.add_argument("--abstract_pkl",
                        default="results_5min/analysis/abstract_tree_n3_d3.pkl")
    parser.add_argument("--output_dir",
                        default="results_5min_val/analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";")
    for c in ("n_worlds", "n_programs"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.dropna(subset=["winner"])
    df = df[df["winner"].isin(["worlds", "programs"])].copy()
    print(f"  {len(df)} rows, {df['config'].nunique()} configs")

    # Comparison tree — rule based, no pickle needed
    print("Applying comparison tree...")
    cmp_method, cmp_region = [], []
    for _, row in df.iterrows():
        m, r = recommend_comparison(row.to_dict())
        cmp_method.append(m)
        cmp_region.append(r)
    df["pred_cmp_method"] = cmp_method
    df["pred_cmp_region"] = cmp_region
    df["match_cmp"] = df["pred_cmp_method"] == df["winner"]

    # Abstract tree — needs pickle
    if os.path.exists(args.abstract_pkl):
        print(f"Applying abstract tree from {args.abstract_pkl}...")
        rec = AbstractRecommender(args.abstract_pkl)
        abs_method, abs_region = [], []
        for _, row in df.iterrows():
            m, r = rec.predict(row.to_dict())
            abs_method.append(m)
            abs_region.append(r)
        df["pred_abs_method"] = abs_method
        df["pred_abs_region"] = abs_region
        df["match_abs"] = df["pred_abs_method"] == df["winner"]
        has_abs = True
    else:
        print(f"[warn] abstract pickle not found at {args.abstract_pkl} "
              "— skipping abstract tree evaluation.")
        has_abs = False

    # Save predictions CSV
    out_csv = os.path.join(args.output_dir, "validation_predictions.csv")
    df.to_csv(out_csv, sep=";", index=False)
    print(f"Wrote {out_csv}")

    # Build report
    lines = []
    lines.append("=== Tree validation on unseen models ===")
    lines.append(f"Input CSV: {args.input_csv}")
    lines.append(f"Rows: {len(df)}  configs: {df['config'].nunique()}")
    lines.append("")

    acc_cmp = df["match_cmp"].mean()
    lines.append(f"Comparison tree accuracy: {acc_cmp:.3f}")
    if has_abs:
        acc_abs = df["match_abs"].mean()
        lines.append(f"Abstract tree accuracy:   {acc_abs:.3f}")
    lines.append("")

    lines.append("=== Accuracy per configuration ===")
    header = f"  {'config':<8s}  {'n':>4s}  {'cmp':>7s}"
    header += f"  {'abs':>7s}" if has_abs else ""
    lines.append(header)
    for cfg, grp in df.groupby("config"):
        row = f"  {cfg:<8s}  {len(grp):>4d}  {grp['match_cmp'].mean():>7.3f}"
        if has_abs:
            row += f"  {grp['match_abs'].mean():>7.3f}"
        lines.append(row)
    lines.append("")

    lines.append("=== Accuracy per region (comparison tree) ===")
    for reg, grp in df.groupby("pred_cmp_region"):
        lines.append(f"  {reg:<10s}  n={len(grp):<4d}  "
                     f"acc={grp['match_cmp'].mean():.3f}  "
                     f"pred={grp['pred_cmp_method'].iloc[0]}")
    lines.append("")

    if has_abs:
        lines.append("=== Accuracy per region (abstract tree) ===")
        for reg, grp in df.groupby("pred_abs_region"):
            majority_pred = grp["pred_abs_method"].mode().iloc[0]
            lines.append(f"  {reg:<20s}  n={len(grp):<4d}  "
                         f"acc={grp['match_abs'].mean():.3f}  "
                         f"pred={majority_pred}")
        lines.append("")

    lines.append("=== Confusion (comparison tree) ===")
    ct = pd.crosstab(df["winner"], df["pred_cmp_method"],
                     rownames=["actual"], colnames=["predicted"])
    lines.append(ct.to_string())
    lines.append("")
    if has_abs:
        lines.append("=== Confusion (abstract tree) ===")
        ct = pd.crosstab(df["winner"], df["pred_abs_method"],
                         rownames=["actual"], colnames=["predicted"])
        lines.append(ct.to_string())
        lines.append("")

    report_path = os.path.join(args.output_dir, "validation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {report_path}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
