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

import numpy as np
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
                        help="Comma-separated result roots to merge. By "
                             "default the largest safety_cap wins on a "
                             "duplicate; see --supersede.")
    parser.add_argument("--out", default="results_tte/all_results.csv")
    parser.add_argument("--supersede", action="store_true",
                        help="Let later roots win on every duplicate, instead "
                             "of the largest safety_cap. Use it when a later "
                             "root recomputes queries the earlier one already "
                             "holds at a larger cap -- a recomputation is "
                             "newer evidence, however long the old run was "
                             "given.")
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

    before = len(merged)
    if args.supersede:
        # Root order decides, one method at a time: a later root overrides a
        # method's columns only if it ran that method (its *_ok is set), so a
        # programs-only re-run keeps the worlds half of the earlier row while
        # a "did not reach" from the re-run still replaces an earlier time.
        key = ["config", "model", "literal"]
        merged = merged.replace("", np.nan)
        out = merged[key + ["exact_l", "exact_u", "safety_cap"]
                     ].drop_duplicates(key, keep="last")
        for meth in ("worlds", "programs"):
            cols = [c for c in FULL_COLUMNS if meth in c]
            ran = merged[merged[f"{meth}_ok"].notna()]
            out = out.merge(ran[key + cols].drop_duplicates(key, keep="last"),
                            on=key, how="left")
        out["ok"] = True
        merged = out.reindex(columns=FULL_COLUMNS)
        how = "superseded by a later root"
    else:
        # Keep the longest-cap attempt per query. sort_values is stable, so
        # ties keep the first file read.
        merged = (merged.sort_values("safety_cap", kind="stable")
                        .drop_duplicates(subset=["config", "model", "literal"],
                                         keep="last")
                        .reset_index(drop=True))
        how = "superseded by a longer re-run"
    dropped = before - len(merged)
    if dropped:
        print(f"\n{dropped} rows {how}")

    # The winner is a function of the two times, so recompute it after the
    # merge rather than trusting a value written when only one method ran.
    tw, tp = merged["time_worlds"], merged["time_programs"]
    merged["gt_winner"] = np.select(
        [tw.notna() & tp.notna() & (tw < tp), tw.notna() & tp.notna(),
         tw.notna(), tp.notna()],
        ["worlds", "programs", "worlds", "programs"], default="unresolved")

    merged.to_csv(args.out, sep=";", index=False)
    print(f"\nWrote {args.out}")
    print(f"Total rows       : {len(merged)}")
    print(f"Configs          : {merged['config'].nunique()}")
    print(f"Caps present     : {sorted(merged['safety_cap'].dropna().unique())}")
    print("gt_winner dist   :")
    print(merged["gt_winner"].value_counts().to_string())


if __name__ == "__main__":
    main()
