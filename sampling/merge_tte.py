"""
Robust merger for results_tte/<config>/*_tte.csv.

The per-config CSVs may mix rows with different schemas because the
experiment writes both "full data" rows (20 columns) and short informational
rows (e.g. "no INTEREST literals", 5 columns) into the same file. This
script recovers only the full-data rows.
"""

import csv
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
    files = sorted(glob.glob("results_tte/*/*_tte.csv"))
    if not files:
        raise SystemExit("No results_tte/*/*_tte.csv found.")

    dfs = []
    for f in files:
        df = parse_file(f)
        cfg = os.path.basename(os.path.dirname(f))
        print(f"  {cfg:>6s}: {len(df):>5d} full-data rows recovered from {f}")
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    # Coerce numeric columns
    numeric_cols = [c for c in FULL_COLUMNS
                    if c not in ("config", "model", "literal",
                                 "gt_winner", "worlds_ok", "programs_ok",
                                 "ok")]
    for c in numeric_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce")

    out = "results_tte/all_results.csv"
    merged.to_csv(out, sep=";", index=False)
    print(f"\nWrote {out}")
    print(f"Total rows       : {len(merged)}")
    print(f"Configs          : {merged['config'].nunique()}")
    print(f"gt_winner dist   :")
    print(merged["gt_winner"].value_counts().to_string())


if __name__ == "__main__":
    main()
