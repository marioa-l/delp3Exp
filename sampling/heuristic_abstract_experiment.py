"""
Does adding a heuristic to the method the abstract tree recommends beat
running that same method plainly?

For every selected query the abstract tree is applied to obtain
(recommended method, leaf). The recommended method is then run twice under
an identical time cap:

    baseline   mode="random"          the classic uniform sampler
    heuristic  mode="bn_centrality"   for world sampling
               mode="bn_marginal"     for program sampling

Both runs also carry ``stop_at_exact``, so a single pass yields two paired
metrics per query:

    time_to_exact  seconds until the exact interval is reached (None if the
                   cap was hit first) -> speedup of the heuristic
    quality        (1 - width) / (1 - exact_width) at the end of the run
                   -> how much tighter the interval is when time runs out

The comparison is paired: both modes run on the same query, so per-query
deltas are meaningful and the analysis is grouped by the tree leaf that
produced the recommendation.

Designed for an overnight HPC run: resumable (finished queries are skipped
on restart), incremental (every query is appended as it finishes) and
parallel (one process per query).

Writes to <output_dir>:
    heuristic_abstract_results.csv    one row per query
    heuristic_abstract_summary.txt    per-leaf and per-method summary

Usage:
    python sampling/heuristic_abstract_experiment.py \\
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \\
        --n_per_leaf 40 --time_limit 120 --workers 16

Resume after an interruption: rerun the exact same command.
Preview the plan without running anything: add --dry_run.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byProgramSampling import Programs      # noqa: E402
from sampling.byWorldSampling import Worlds          # noqa: E402
from sampling.recommend import AbstractRecommender   # noqa: E402

# The heuristic that pairs with each sampling method.
HEURISTIC_MODE = {"worlds": "bn_centrality", "programs": "bn_marginal"}

DEFAULT_ROOTS = [
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
]

RESULT_COLUMNS = [
    "config", "model", "literal", "leaf", "method",
    "exact_l", "exact_u", "exact_width", "time_limit",
    "base_l", "base_u", "base_quality", "base_time_to_exact",
    "base_delp_calls", "base_wall_time",
    "heur_l", "heur_u", "heur_quality", "heur_time_to_exact",
    "heur_delp_calls", "heur_wall_time",
    "quality_delta", "speedup", "ok", "error",
]


def find_models_root(override):
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"Models root not found; tried {DEFAULT_ROOTS}")


def quality(l, u, exact_l, exact_u):
    """Normalised interval tightness: 1.0 means the exact interval was met."""
    remainder = 1 - (exact_u - exact_l)
    if remainder <= 0:
        return float("nan")
    return (1 - (u - l)) / remainder


def _run(method, mode, model_path, save_dir, literal, cap,
         dgraph_path, exact, epsilon):
    """Run one sampler in one mode; return (l, u, time_to_exact, delp_calls)."""
    os.makedirs(save_dir, exist_ok=True)
    stop = {literal: exact}
    if method == "worlds":
        s = Worlds(model_path, save_dir + "/")
        s.utils.get_interest_lit = lambda: [literal]
        s.start_sampling(percentile_samples=0, source="random",
                         info=f"_h_w_{mode}", max_seconds=cap, mode=mode,
                         dgraph_path=dgraph_path,
                         stop_at_exact=stop, stop_epsilon=epsilon)
    else:
        s = Programs(model_path, save_dir + "/")
        s.utils.get_interest_lit = lambda: [literal]
        s.start_sampling(percentile_samples=0, info=f"_h_p_{mode}",
                         max_seconds=cap, mode=mode,
                         stop_at_exact=stop, stop_epsilon=epsilon)
    st = s.results["status"].get(literal, {})
    return (st.get("pyes", 0.0), 1 - st.get("pno", 0.0),
            st.get("time_to_exact"), st.get("delp_calls", 0))


def _worker(job):
    (cfg, model_name, literal, leaf, method, model_path, dgraph_path,
     save_dir, exact_l, exact_u, cap, epsilon) = job

    row = {
        "config": cfg, "model": model_name, "literal": literal,
        "leaf": leaf, "method": method,
        "exact_l": exact_l, "exact_u": exact_u,
        "exact_width": exact_u - exact_l, "time_limit": cap,
        "error": "",
    }
    exact = (exact_l, exact_u)
    try:
        t0 = time.time()
        bl, bu, bt, bc = _run(method, "random", model_path, save_dir,
                              literal, cap, None, exact, epsilon)
        row["base_wall_time"] = time.time() - t0

        t1 = time.time()
        hl, hu, ht, hc = _run(method, HEURISTIC_MODE[method], model_path,
                              save_dir, literal, cap, dgraph_path,
                              exact, epsilon)
        row["heur_wall_time"] = time.time() - t1

        row.update({
            "base_l": bl, "base_u": bu, "base_time_to_exact": bt,
            "base_delp_calls": bc,
            "base_quality": quality(bl, bu, exact_l, exact_u),
            "heur_l": hl, "heur_u": hu, "heur_time_to_exact": ht,
            "heur_delp_calls": hc,
            "heur_quality": quality(hl, hu, exact_l, exact_u),
        })
        row["quality_delta"] = row["heur_quality"] - row["base_quality"]
        # Speedup is only defined when both runs reached the exact interval.
        row["speedup"] = (bt / ht) if (bt and ht and ht > 0) else None
        row["ok"] = True
    except Exception as e:  # pragma: no cover
        row["ok"] = False
        row["error"] = str(e)
    return row


def _worker_safe(job):
    try:
        return _worker(job)
    except Exception as e:  # pragma: no cover
        return {"config": job[0], "model": job[1], "literal": job[2],
                "leaf": job[3], "method": job[4], "ok": False,
                "error": str(e)}


def already_done(csv_path):
    """(config, model, literal) triples already written, for resuming."""
    if not os.path.exists(csv_path):
        return set()
    try:
        d = pd.read_csv(csv_path, sep=";", usecols=["config", "model",
                                                    "literal"])
    except Exception:
        return set()
    return set(map(tuple, d.itertuples(index=False, name=None)))


def build_jobs(args, models_root, csv_path):
    df = pd.read_csv(args.input_csv, sep=";")
    df = df.dropna(subset=["exact_l", "exact_u"]).copy()
    # A degenerate exact interval makes the quality metric undefined.
    df = df[(df["exact_u"] - df["exact_l"]) < 1.0]
    if not args.include_unresolved:
        df = df[df["gt_winner"].isin(["worlds", "programs"])]
    print(f"Candidate queries: {len(df)}")

    rec = AbstractRecommender(args.abstract_pkl)
    preds = [rec.predict(r) for r in df.to_dict("records")]
    df["method"] = [p[0] for p in preds]
    df["leaf"] = [p[1] for p in preds]

    # Stratify by leaf so every branch of the tree is represented.
    picks = []
    for leaf, g in df.groupby("leaf"):
        take = min(args.n_per_leaf, len(g))
        picks.append(g.sample(n=take, random_state=args.seed))
    sel = pd.concat(picks, ignore_index=True)

    done = already_done(csv_path)
    if done:
        before = len(sel)
        sel = sel[[(r.config, r.model, r.literal) not in done
                   for r in sel.itertuples()]]
        print(f"Resuming: {before - len(sel)} of {before} already on disk")

    jobs = []
    for r in sel.itertuples():
        model_path = os.path.join(models_root, r.config, f"{r.model}.json")
        if not os.path.exists(model_path):
            continue
        jobs.append((
            r.config, r.model, r.literal, r.leaf, r.method, model_path,
            os.path.join(args.dgraph_dir, r.config, f"{r.model}_dgraph.json"),
            os.path.join(args.output_dir, "runs", r.config, r.model),
            float(r.exact_l), float(r.exact_u), args.time_limit, args.epsilon,
        ))
    return jobs, sel


def write_summary(csv_path, out_path):
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["ok"] == True]  # noqa: E712
    df = df.dropna(subset=["quality_delta"])
    if df.empty:
        print("[warn] no successful rows; summary skipped")
        return

    L = ["=== Heuristic vs classic, within the method the abstract tree "
         "recommends ===", ""]
    L.append(f"Queries evaluated : {len(df)}")
    L.append(f"Time cap per run  : {int(df['time_limit'].iloc[0])} s")
    L.append("")
    L.append("Paired design: both modes run on the same query, so the delta "
             "is per-query.")
    L.append("")

    def block(title, g):
        n = len(g)
        qd = g["quality_delta"]
        wins = int((qd > 0).sum())
        ties = int((qd == 0).sum())
        L.append(f"{title}  (n={n})")
        L.append(f"    base quality  : {g['base_quality'].mean():.4f} "
                 f"+/- {g['base_quality'].std(ddof=1)/max(n,1)**0.5:.4f}")
        L.append(f"    heur quality  : {g['heur_quality'].mean():.4f} "
                 f"+/- {g['heur_quality'].std(ddof=1)/max(n,1)**0.5:.4f}")
        L.append(f"    delta         : {qd.mean():+.4f} "
                 f"+/- {qd.std(ddof=1)/max(n,1)**0.5:.4f} "
                 f"(median {qd.median():+.4f})")
        L.append(f"    heuristic wins: {wins}/{n} "
                 f"({wins/n:.1%}), ties {ties}")
        both = g.dropna(subset=["base_time_to_exact", "heur_time_to_exact"])
        if len(both):
            sp = both["base_time_to_exact"] / both["heur_time_to_exact"]
            L.append(f"    both reached exact: {len(both)} | "
                     f"median speedup {sp.median():.2f}x")
        nb = int(g["base_time_to_exact"].notna().sum())
        nh = int(g["heur_time_to_exact"].notna().sum())
        L.append(f"    reached exact : base {nb}/{n}, heuristic {nh}/{n}")
        L.append("")

    block("ALL QUERIES", df)
    for method, g in df.groupby("method"):
        block(f"METHOD = {method}  (heuristic: {HEURISTIC_MODE[method]})", g)
    for leaf, g in df.sort_values("leaf").groupby("leaf"):
        block(f"LEAF {leaf}  (method={g['method'].iloc[0]})", g)

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_tte/all_results.csv")
    parser.add_argument("--abstract_pkl",
                        default="results_5min/analysis/abstract_tree_n3_d3.pkl")
    parser.add_argument("--output_dir", default="results_tte/heuristics")
    parser.add_argument("--models_root", default=None)
    parser.add_argument("--dgraph_dir", default="results/dgraphs")
    parser.add_argument("--n_per_leaf", type=int, default=40,
                        help="Queries sampled per abstract-tree leaf")
    parser.add_argument("--time_limit", type=int, default=120,
                        help="Seconds per run (each query costs 2 runs)")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include_unresolved", action="store_true",
                        help="Also sample queries where neither sampler ever "
                             "reached the exact interval (quality is still "
                             "measurable there).")
    parser.add_argument("--summary_only", action="store_true",
                        help="Rebuild the summary from an existing CSV")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir,
                            "heuristic_abstract_results.csv")
    summary_path = os.path.join(args.output_dir,
                                "heuristic_abstract_summary.txt")

    if args.summary_only:
        write_summary(csv_path, summary_path)
        return

    models_root = find_models_root(args.models_root)
    print(f"Models root  : {models_root}")
    print(f"Abstract tree: {args.abstract_pkl}")
    print(f"Time cap     : {args.time_limit}s per run (2 runs per query)")
    print(f"Workers      : {args.workers}\n")

    jobs, sel = build_jobs(args, models_root, csv_path)
    if not jobs:
        print("Nothing left to run.")
        if os.path.exists(csv_path):
            write_summary(csv_path, summary_path)
        return

    print(f"\nJobs: {len(jobs)}")
    print(sel.groupby(["leaf", "method"]).size().to_string())
    worst_h = len(jobs) * 2 * args.time_limit / 3600 / max(args.workers, 1)
    print(f"\nWorst case (every run hits the cap): {worst_h:.1f} h "
          f"at {args.workers} workers")
    if args.dry_run:
        return

    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_worker_safe, j): j for j in jobs}
        for fut in as_completed(futures):
            n += 1
            row = fut.result()
            header = not os.path.exists(csv_path)
            pd.DataFrame([row]).reindex(
                columns=RESULT_COLUMNS).to_csv(
                    csv_path, mode="a", header=header, index=False, sep=";")
            d = row.get("quality_delta")
            d = f"{d:+.3f}" if isinstance(d, float) else "  n/a"
            print(f"  [{n}/{len(jobs)}] {row['config']}/{row['model']} "
                  f"{row['leaf']} {row['method']:<8s} delta={d} "
                  f"({(time.time()-t0)/3600:.2f} h)", flush=True)

    print(f"\nFinished in {(time.time()-t0)/3600:.2f} h")
    write_summary(csv_path, summary_path)


if __name__ == "__main__":
    main()
