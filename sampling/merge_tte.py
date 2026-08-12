"""
Robust merger for the time-to-exact result CSVs.

Reads <root>/<config>/*_tte.csv for every root given with --roots (by default
results_tte and results_tte_long, the latter holding the longer-cap re-runs of
the queries that stayed unresolved).

Two robustness concerns are handled:

1. The per-config CSVs may mix rows with different schemas, because the
   experiment appends both "full data" rows (20 columns) and short
   informational rows into the same file. Only full-data rows are kept.
2. The same (config, model, literal) may appear once per cap. The row with
   the largest ``safety_cap`` wins, so a longer re-run always supersedes the
   shorter attempt it was launched to replace.

Usage:
    python sampling/merge_tte.py
    python sampling/merge_tte.py --roots results_tte --out results_tte/short_only.csv
"""

import argparse
import glob
import os

import pandas as pd

# Column order as emitted by _run_one_model in time_to_exact_experiment.py.
FULL_COLUMNS = [
    "config", "model", "literal", "exact_l", "exact_u", "safety_cap",
    "time_worlds", "worlds_wall_time", "worlds_l", "worlds_u",
    "worlds_delp_calls", "worlds_ok",
    "time_programs", "programs_wall_time", "programs_l", "programs_u",
    "programs_delp_calls", "programs_ok",
    "gt_winner", "ok",
]
N_FULL = len(FULL_COLUMNS)


def parse_file(path):
    """Return a DataFrame with only the rows that carry full-data structure."""
    keep = []
    seen_header = False
    with open(path) as fh:
        for line in fh:
            fields = line.rstrip("\n").split(";")
            if len(fields) != N_FULL:
                continue
            # Detect the actual header row (only when the CSV started with a
            # full-schema chunk). Skip it explicitly.
            if not seen_header and fields[0] == "config" and fields[1] == "model":
                seen_header = True
                continue
            keep.append(fields)
    if not keep:
        return pd.DataFrame(columns=FULL_COLUMNS)
    return pd.DataFrame(keep, columns=FULL_COLUMNS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", default="results_tte,results_tte_long",
                        help="Comma-separated result roots to merge. Later "
                             "roots do not take priority by position — the "
                             "largest safety_cap always wins.")
    parser.add_argument("--out", default="results_tte/all_results.csv")
    args = parser.parse_args()

    dfs = []
    for root in [r.strip() for r in args.roots.split(",") if r.strip()]:
        files = sorted(glob.glob(os.path.join(root, "*", "*_tte.csv")))
        if not files:
            print(f"  [skip] no CSVs under {root}")
            continue
        for f in files:
            df = parse_file(f)
            cfg = os.path.basename(os.path.dirname(f))
            print(f"  {root:>18s} / {cfg:<5s}: {len(df):>5d} rows")
            dfs.append(df)
    if not dfs:
        raise SystemExit(f"No *_tte.csv found under {args.roots}")

    merged = pd.concat(dfs, ignore_index=True)
    numeric_cols = [c for c in FULL_COLUMNS
                    if c not in ("config", "model", "literal",
                                 "gt_winner", "worlds_ok", "programs_ok",
                                 "ok")]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # Keep the longest-cap attempt per query. sort_values is stable, so ties
    # keep the first file read.
    before = len(merged)
    merged = (merged.sort_values("safety_cap", kind="stable")
                    .drop_duplicates(subset=["config", "model", "literal"],
                                     keep="last")
                    .reset_index(drop=True))
    dropped = before - len(merged)
    if dropped:
        print(f"\nSuperseded by a longer re-run: {dropped} rows dropped")

    merged.to_csv(args.out, sep=";", index=False)
    print(f"\nWrote {args.out}")
    print(f"Total rows       : {len(merged)}")
    print(f"Configs          : {merged['config'].nunique()}")
    print(f"Caps present     : {sorted(merged['safety_cap'].dropna().unique())}")
    print("gt_winner dist   :")
    print(merged["gt_winner"].value_counts().to_string())


if __name__ == "__main__":
    main()
