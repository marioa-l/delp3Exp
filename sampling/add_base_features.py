"""
Add the base model + literal features (em_var, n_annots, af_avg_body_size,
lit_head_def, etc.) to a CSV that only carries sampling results.

Needed for CSVs produced outside run_experiment.py (e.g. the time-to-exact
experiment), so downstream recommenders can find every feature they expect.

Usage:
    python sampling/add_base_features.py --input_csv results_tte/all_results.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Reuse the exact functions that build these features in run_experiment.py
from sampling.run_experiment import (  # noqa: E402
    model_features, literal_features, load_json,
)

DEFAULT_ROOTS = [
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
]


def find_models_root(override: str | None) -> str:
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"Models root not found; tried {DEFAULT_ROOTS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--models_root", default=None)
    args = parser.parse_args()

    root = find_models_root(args.models_root)
    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv, sep=";")
    print(f"  {len(df)} rows")

    # Cache model_features once per (config, model)
    model_feat_cache: dict = {}
    def get_model_feats(cfg, model_name):
        key = (cfg, model_name)
        if key not in model_feat_cache:
            path = os.path.join(root, cfg, f"{model_name}.json")
            try:
                model_feat_cache[key] = (model_features(path), load_json(path))
            except Exception as e:
                print(f"  [warn] failed on {key}: {e}")
                model_feat_cache[key] = (None, None)
        return model_feat_cache[key]

    added = {}
    for i, row in df.iterrows():
        feats, data = get_model_feats(row["config"], row["model"])
        if feats is None:
            continue
        for k, v in feats.items():
            added.setdefault(k, [None] * len(df))
            added[k][i] = v
        # Literal features
        try:
            lit_feats = literal_features(row["literal"], data)
        except Exception:
            lit_feats = {}
        for k, v in lit_feats.items():
            added.setdefault(k, [None] * len(df))
            added[k][i] = v

    # Merge into df (overwrite if column exists — cache values are authoritative)
    added_n, overwritten = 0, 0
    for col, values in added.items():
        s = pd.Series(values, index=df.index)
        if col in df.columns:
            df[col] = s
            overwritten += 1
        else:
            df[col] = s
            added_n += 1

    # Backup + write
    backup = args.input_csv + ".pre_base"
    if not os.path.exists(backup):
        os.rename(args.input_csv, backup)
        print(f"  Backed up original to: {backup}")
    df.to_csv(args.input_csv, sep=";", index=False)
    print(f"Wrote {args.input_csv}")
    print(f"  Columns: {len(df.columns)}  (added {added_n}, overwritten {overwritten})")


if __name__ == "__main__":
    main()
