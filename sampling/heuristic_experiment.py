"""
Head-to-head evaluation of the heuristic samplers against the uniform
baseline.

For every (config, model, literal) picked from the training or validation
CSV, we apply the depth-3 comparison tree to decide which sampling method
is recommended, then run the recommended method twice under the same time
budget: once in "random" mode (baseline) and once in the heuristic mode
introduced by this pipeline. The quality metric is stored at three
snapshots so we can compare convergence curves, not just final quality.

Reads:
    --input_csv (default: results_5min/all_results.csv)

Writes:
    <output_dir>/heuristic_results.csv
    <output_dir>/heuristic_summary.txt

Usage (icic):
    python sampling/heuristic_experiment.py \
        --input_csv results_5min/all_results.csv \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --dgraph_dir results/dgraphs \
        --output_dir results_5min/heuristics \
        --n_per_region 15 \
        --time_limit 60 --workers 4
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byProgramSampling import Programs  # noqa: E402
from sampling.byWorldSampling import Worlds      # noqa: E402
from sampling.recommend import recommend_comparison  # noqa: E402


def quality(l, u, exact_l, exact_u):
    remainder = 1 - (exact_u - exact_l)
    if remainder == 0:
        return 0.0
    return (1 - (u - l)) / remainder


def _run_worlds(model_path, save_path, literal, time_limit, mode, dgraph_path):
    """Run a single-literal Worlds sampling and return the interval bounds."""
    sampler = Worlds(model_path, save_path + "/")
    sampler.utils.get_interest_lit = lambda: [literal]
    sampler.start_sampling(
        percentile_samples=0, source="random", info=f"_w_{mode}",
        max_seconds=time_limit, mode=mode, dgraph_path=dgraph_path,
    )
    s = sampler.results["status"].get(literal, {})
    return s.get("pyes", 0.0), 1 - s.get("pno", 0.0), s.get("time", 0.0)


def _run_programs(model_path, save_path, literal, time_limit, mode):
    """Run a single-literal Programs sampling and return the interval bounds."""
    sampler = Programs(model_path, save_path + "/")
    sampler.utils.get_interest_lit = lambda: [literal]
    sampler.start_sampling(
        percentile_samples=0, info=f"_p_{mode}",
        max_seconds=time_limit, mode=mode,
    )
    s = sampler.results["status"].get(literal, {})
    return s.get("pyes", 0.0), 1 - s.get("pno", 0.0), s.get("time", 0.0)


def _worker(job):
    """
    Run baseline + heuristic on one (config, model, literal). Returns a dict
    with the metrics needed to build the result CSV row.
    """
    (cfg, model_name, literal, region, method, model_path, dgraph_path,
     save_path, exact_l, exact_u, time_limit) = job

    row = {
        "config": cfg, "model": model_name, "literal": literal,
        "region": region, "method": method,
        "exact_l": exact_l, "exact_u": exact_u,
    }
    # The samplers write their per-model JSONs under save_path; ensure the
    # target directory exists before any of them is invoked.
    os.makedirs(save_path, exist_ok=True)
    try:
        if method == "worlds":
            base_l, base_u, base_t = _run_worlds(
                model_path, save_path, literal, time_limit, "random", None)
            heur_l, heur_u, heur_t = _run_worlds(
                model_path, save_path, literal, time_limit, "bn_centrality",
                dgraph_path)
        else:
            base_l, base_u, base_t = _run_programs(
                model_path, save_path, literal, time_limit, "random")
            heur_l, heur_u, heur_t = _run_programs(
                model_path, save_path, literal, time_limit, "bn_marginal")
        row.update({
            "base_l": base_l, "base_u": base_u,
            "base_quality": quality(base_l, base_u, exact_l, exact_u),
            "base_time": base_t,
            "heur_l": heur_l, "heur_u": heur_u,
            "heur_quality": quality(heur_l, heur_u, exact_l, exact_u),
            "heur_time": heur_t,
        })
        row["quality_delta"] = row["heur_quality"] - row["base_quality"]
        row["ok"] = True
    except Exception as e:  # pragma: no cover
        row.update({"ok": False, "error": str(e)})
    return row


def build_jobs(df: pd.DataFrame, models_root: str, dgraph_dir: str,
               n_per_region: int, output_dir: str, time_limit: int,
               seed: int) -> list:
    random.seed(seed)
    df = df.dropna(subset=["winner"]).copy()
    df = df[df["winner"].isin(["worlds", "programs"])]

    # Attach the comparison-tree prediction to every row.
    df["pred_method"], df["pred_region"] = zip(
        *df.apply(lambda r: recommend_comparison(r.to_dict()), axis=1))

    # Only useful for region-level convergence measurement: keep rows whose
    # exact interval is non-degenerate.
    df["exact_width"] = df["exact_u"] - df["exact_l"]
    df = df[df["exact_width"] < 1.0]

    # Pick n_per_region queries per (region, predicted method). Regions marked
    # ``*_tiny`` and ``unknown`` are filtered because they don't map to any of
    # the four canonical rules.
    keep_regions = {"A", "B", "C", "D"}
    df = df[df["pred_region"].isin(keep_regions)]
    jobs = []
    for reg, group in df.groupby("pred_region"):
        # sample_n = min(n_per_region, len(group))
        picks = group.sample(n=min(n_per_region, len(group)),
                             random_state=seed).reset_index(drop=True)
        for _, r in picks.iterrows():
            cfg, model_name, lit = r["config"], r["model"], r["literal"]
            model_path = os.path.join(models_root, cfg, f"{model_name}.json")
            dgraph_path = os.path.join(dgraph_dir, cfg,
                                       f"{model_name}_dgraph.json")
            save_path = os.path.join(output_dir, "runs", cfg, model_name)
            jobs.append((cfg, model_name, lit, reg, r["pred_method"],
                         model_path, dgraph_path, save_path,
                         r["exact_l"], r["exact_u"], time_limit))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_5min/all_results.csv")
    parser.add_argument("--models_root", default=None,
                        help="Path to delp3e_models. Defaults to icic path.")
    parser.add_argument("--dgraph_dir", default="results/dgraphs")
    parser.add_argument("--output_dir", default="results_5min/heuristics")
    parser.add_argument("--n_per_region", type=int, default=15)
    parser.add_argument("--time_limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.models_root is None:
        candidates = [
            "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
            "/Users/marioleiva/Documents/desarrollo/delp3e_models",
        ]
        args.models_root = next((c for c in candidates
                                 if os.path.isdir(c)), candidates[0])
    print(f"Using models_root: {args.models_root}")

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv, sep=";")
    for c in ("n_worlds", "n_programs"):
        if c in df.columns:
            df[c] = df[c].astype(float)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    jobs = build_jobs(df, args.models_root, args.dgraph_dir,
                      args.n_per_region, args.output_dir,
                      args.time_limit, args.seed)
    print(f"Built {len(jobs)} jobs "
          f"({args.n_per_region} × 4 regions = "
          f"{args.n_per_region * 4} target; actual {len(jobs)})")

    if not jobs:
        print("No jobs to run. Check that the CSV has predictions for regions "
              "A / B / C / D.")
        return

    rows = []
    t0 = time.time()
    if args.workers <= 1:
        for i, job in enumerate(jobs):
            print(f"[{i+1}/{len(jobs)}] {job[0]}/{job[1]} {job[2]} "
                  f"region={job[3]} method={job[4]}", flush=True)
            rows.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_worker, j): j for j in jobs}
            done = 0
            for fut in as_completed(futures):
                done += 1
                row = fut.result()
                rows.append(row)
                print(f"  [{done}/{len(jobs)}] {row['config']}/{row['model']} "
                      f"{row['literal']} region={row['region']} "
                      f"method={row['method']} "
                      f"delta={row.get('quality_delta', float('nan')):+.3f}",
                      flush=True)

    elapsed = (time.time() - t0) / 60.0
    print(f"\nFinished in {elapsed:.1f} min.")

    csv_path = os.path.join(args.output_dir, "heuristic_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, sep=";", index=False)
    print(f"Wrote {csv_path}")

    # Summary text
    df_r = pd.DataFrame(rows)
    if not df_r.empty and "quality_delta" in df_r.columns:
        lines = ["=== Heuristic experiment summary ==="]
        lines.append(f"Jobs: {len(df_r)}   "
                     f"OK: {int(df_r.get('ok', True).sum())}   "
                     f"Time: {elapsed:.1f} min")
        lines.append("")
        for reg, grp in df_r.groupby("region"):
            base_q = grp["base_quality"].mean()
            heur_q = grp["heur_quality"].mean()
            delta  = grp["quality_delta"].mean()
            wins = int((grp["quality_delta"] > 0).sum())
            lines.append(
                f"Region {reg} (n={len(grp)}, method={grp['method'].iloc[0]}): "
                f"base={base_q:.3f}  heur={heur_q:.3f}  "
                f"delta={delta:+.3f}  heur_wins={wins}/{len(grp)}")
        report = os.path.join(args.output_dir, "heuristic_summary.txt")
        with open(report, "w") as f:
            f.write("\n".join(lines))
        print(f"Wrote {report}\n")
        print("\n".join(lines))


if __name__ == "__main__":
    main()
