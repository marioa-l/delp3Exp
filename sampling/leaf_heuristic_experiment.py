"""
Do the leaf-guided heuristics beat uniform sampling, and the earlier
generic heuristics, on the method the tree recommends?

For every selected query the comparison tree (final_tree.json) gives the
recommended method and the branch that produced it. That method is then run
three times under an identical time cap:

    random    mode="random"          uniform draws from the space: the trivial
                                     definition of each sampler, and the
                                     baseline the other two must beat
    generic   "bn_centrality" (W)    the earlier heuristics: they recompute
              "bn_marginal"   (P)    their own state and ignore the branch
    guided    "hub_strata"    (W)    the new leaf-guided heuristics, which
              "usage_strata"  (P)    reuse attributes the tree already paid for

The guided modes are H1/H2/H3 together: H1 (worlds) stratifies over the
highest-degree BN variables, H2 (programs) stratifies over the EM variables
the most annotations depend on, and H3 sets how many dimensions to stratify
over from the relative size of the two sampling spaces
(``hint_ratio = n_annots - em_var``), instead of a fixed constant.

Every run carries ``stop_at_exact``, so one pass yields both metrics that
matter: interval quality when time runs out, and time to reach the exact
interval when it is reached at all. Solver calls are recorded too, since a
heuristic that buys quality with extra solver work is not a win.

Resumable (finished queries are skipped on restart), incremental, parallel.

Writes to <output_dir>:
    leaf_heuristic_results.csv   one row per query, final state of each mode
    leaf_heuristic_traces.csv    the interval over time, per query and mode,
                                 resampled onto a grid that is dense over the
                                 first seconds and coarse afterwards
    leaf_heuristic_summary.txt

Usage:
    python sampling/leaf_heuristic_experiment.py --n_per_leaf 40 \\
        --time_limit 120 --workers 6
"""

from __future__ import annotations

import argparse
import json
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


def apply_plain(node, row):
    """
    Walk the portable comparison tree for one feature dict; returns
    (method, path).

    Deliberately kept here instead of imported from final_tree, which pulls
    in sklearn and matplotlib. Applying the tree needs neither, and the
    cluster does not always carry a matching sklearn.
    """
    path = []
    while node["kind"] != "leaf":
        a, b = row.get(node["a"]), row.get(node["b"])
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            went = False           # undecidable -> take the ELSE branch
        else:
            went = float(a) < float(b)
        path.append(f"{node['a']} {'<' if went else '>='} {node['b']}")
        node = node["yes"] if went else node["no"]
    return node["cls"], path

# mode name per (method, family)
#   random  — the trivial definition of each sampler: draw uniformly from the
#             space. Not the BN distribution, which would hand this family
#             structural information the comparison is meant to isolate.
#   generic — structural information about the models, independent of the tree.
#   guided  — the attributes the tree already consulted on the branch that
#             produced this recommendation.
MODES = {
    "worlds":   {"random": "random", "generic": "bn_centrality",
                 "guided": "hub_strata"},
    "programs": {"random": "random", "generic": "bn_marginal",
                 "guided": "program_strata"},
}
FAMILIES = ["random", "generic", "guided"]

# Convergence traces are resampled onto a fixed grid rather than kept per
# sample: the interesting question is how fast the interval closes in the
# first seconds, so the grid is dense early and coarse later.
def trace_grid(cap):
    g = [round(0.1 * i, 2) for i in range(1, 51)]          # 0.1 .. 5 s
    g += [round(5 + 0.5 * i, 2) for i in range(1, 51)]     # 5.5 .. 30 s
    g += [float(t) for t in range(32, int(cap) + 1, 2)]    # 32 .. cap
    return [t for t in g if t <= cap]

DEFAULT_ROOTS = [
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
]

TRACE_COLUMNS = ["config", "model", "literal", "method", "family",
                 "t", "n_samples", "delp_calls", "l", "u", "quality"]

COLUMNS = (["config", "model", "literal", "leaf", "method",
            "exact_l", "exact_u", "time_limit", "hint_ratio"]
           + [f"{f}_{m}" for f in FAMILIES
              for m in ("l", "u", "quality", "time_to_exact", "delp_calls",
                        "wall")]
           + ["ok", "error"])


def find_models_root(override):
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"Models root not found; tried {DEFAULT_ROOTS}")


def quality(l, u, exact_l, exact_u):
    rem = 1 - (exact_u - exact_l)
    return float("nan") if rem <= 0 else (1 - (u - l)) / rem


def _run(method, mode, model_path, save_dir, literal, cap, dgraph_path,
         exact, epsilon, hint_ratio):
    os.makedirs(save_dir, exist_ok=True)
    stop = {literal: exact}
    common = dict(percentile_samples=0, max_seconds=cap, mode=mode,
                  stop_at_exact=stop, stop_epsilon=epsilon,
                  hint_ratio=hint_ratio)
    if method == "worlds":
        s = Worlds(model_path, save_dir + "/")
        s.utils.get_interest_lit = lambda: [literal]
        hist = s.start_sampling(source="random", info=f"_lh_w_{mode}",
                                dgraph_path=dgraph_path, **common)
    else:
        s = Programs(model_path, save_dir + "/")
        s.utils.get_interest_lit = lambda: [literal]
        hist = s.start_sampling(info=f"_lh_p_{mode}", **common)
    st = s.results["status"].get(literal, {})
    return (st.get("pyes", 0.0), 1 - st.get("pno", 0.0),
            st.get("time_to_exact"), st.get("delp_calls", 0),
            (hist or {}).get(literal, []))


def resample(history, cap):
    """
    Step-sample the per-draw history onto the fixed grid: at each grid point,
    report the state as of the last draw at or before it. Points before the
    first draw are dropped, and the run's final state is carried to the end of
    the grid only while the run was still going, so a run that stopped early
    at the exact interval is not reported as still improving afterwards.
    """
    if not history:
        return []
    grid = trace_grid(cap)
    out, i, last = [], 0, None
    end = history[-1].get("time", 0.0)
    for t in grid:
        while i < len(history) and history[i].get("time", 0.0) <= t:
            last = history[i]
            i += 1
        if last is None or t > end:
            continue
        out.append((t, last.get("calls", 0), last.get("l", 0.0),
                    last.get("u", 1.0), i))
    # The run's final state, appended whatever the grid: a run that reaches the
    # exact interval between two grid points would otherwise be recorded as
    # still short of it, and that is precisely the moment of interest.
    f = history[-1]
    if not out or out[-1][0] < end:
        out.append((end, f.get("calls", 0), f.get("l", 0.0),
                    f.get("u", 1.0), len(history)))
    return out


def _worker(job):
    (cfg, model, literal, leaf, method, model_path, dgraph_path, save_dir,
     exact_l, exact_u, cap, epsilon, hint_ratio) = job
    row = {"config": cfg, "model": model, "literal": literal, "leaf": leaf,
           "method": method, "exact_l": exact_l, "exact_u": exact_u,
           "time_limit": cap, "hint_ratio": hint_ratio, "error": ""}
    traces = []
    try:
        for fam in FAMILIES:
            t0 = time.time()
            l, u, tte, calls, hist = _run(
                method, MODES[method][fam], model_path, save_dir, literal,
                cap, dgraph_path, (exact_l, exact_u), epsilon, hint_ratio)
            row[f"{fam}_l"] = l
            row[f"{fam}_u"] = u
            row[f"{fam}_quality"] = quality(l, u, exact_l, exact_u)
            row[f"{fam}_time_to_exact"] = tte
            row[f"{fam}_delp_calls"] = calls
            row[f"{fam}_wall"] = time.time() - t0
            for t, c, tl, tu, n in resample(hist, cap):
                traces.append((cfg, model, literal, method, fam, t, n, c,
                               tl, tu, quality(tl, tu, exact_l, exact_u)))
        row["ok"] = True
    except Exception as e:  # pragma: no cover
        row["ok"] = False
        row["error"] = str(e)
    return row, traces


def _worker_safe(job):
    try:
        return _worker(job)
    except Exception as e:  # pragma: no cover
        return ({"config": job[0], "model": job[1], "literal": job[2],
                 "leaf": job[3], "method": job[4], "ok": False,
                 "error": str(e)}, [])


def already_done(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        d = pd.read_csv(csv_path, sep=";",
                        usecols=["config", "model", "literal"])
    except Exception:
        return set()
    return set(map(tuple, d.itertuples(index=False, name=None)))


def build_jobs(args, models_root, csv_path):
    df = pd.read_csv(args.input_csv, sep=";")
    df = df.dropna(subset=["exact_l", "exact_u"])
    df = df[(df["exact_u"] - df["exact_l"]) < 1.0]
    if not args.include_unresolved:
        df = df[df["gt_winner"].isin(["worlds", "programs"])]

    tree = json.load(open(args.tree_json))["tree"]
    preds = [apply_plain(tree, r) for r in df.to_dict("records")]
    df = df.assign(method=[p[0] for p in preds],
                   leaf=[" & ".join(p[1]) for p in preds])
    print(f"Candidates: {len(df)} | "
          f"{df.method.value_counts().to_dict()}")

    picks = [g.sample(n=min(args.n_per_leaf, len(g)),
                      random_state=args.seed)
             for _, g in df.groupby("leaf")]
    sel = pd.concat(picks, ignore_index=True)

    done = already_done(csv_path)
    if done:
        before = len(sel)
        sel = sel[[(r.config, r.model, r.literal) not in done
                   for r in sel.itertuples()]]
        print(f"Resuming: {before - len(sel)} of {before} already on disk")

    jobs = []
    for r in sel.itertuples():
        mp = os.path.join(models_root, r.config, f"{r.model}.json")
        if not os.path.exists(mp):
            continue
        # H3 input: how much bigger this method's space is than the other one,
        # in log2 units. Positive means more coverage insurance is worthwhile.
        ratio = float(r.n_annots) - float(r.em_var)
        hint = ratio if r.method == "programs" else -ratio
        jobs.append((r.config, r.model, r.literal, r.leaf, r.method, mp,
                     os.path.join(args.dgraph_dir, r.config,
                                  f"{r.model}_dgraph.json"),
                     os.path.join(args.output_dir, "runs", r.config, r.model),
                     float(r.exact_l), float(r.exact_u), args.time_limit,
                     args.epsilon, hint))
    return jobs, sel


def write_summary(csv_path, out_path):
    df = pd.read_csv(csv_path, sep=";")
    df = df[df["ok"] == True]  # noqa: E712
    if df.empty:
        print("[warn] nothing to summarise")
        return
    L = ["=== Leaf-guided heuristics vs random and generic ===", "",
         f"Queries : {len(df)}",
         f"Time cap: {int(df['time_limit'].iloc[0])} s per run, "
         f"{len(FAMILIES)} runs per query", "",
         "Paired: all modes run on the same query.", ""]

    def block(title, g):
        n = len(g)
        L.append(f"{title}  (n={n})")
        for fam in FAMILIES:
            q = g[f"{fam}_quality"].dropna()
            ex = g[f"{fam}_time_to_exact"].notna().sum()
            calls = g[f"{fam}_delp_calls"].median()
            sem = q.std(ddof=1) / max(len(q), 1) ** 0.5 if len(q) > 1 else 0
            L.append(f"    {fam:<8s} quality {q.mean():.4f} +/- {sem:.4f}"
                     f"   reached exact {ex:>4d}/{n}"
                     f"   median solver calls {calls:>9.0f}")
        for fam in ("generic", "guided"):
            d = (g[f"{fam}_quality"] - g["random_quality"]).dropna()
            if len(d):
                wins = int((d > 0).sum())
                sem = d.std(ddof=1) / max(len(d), 1) ** 0.5
                L.append(f"    {fam} - random: mean {d.mean():+.4f} "
                         f"+/- {sem:.4f}, median {d.median():+.4f}, "
                         f"wins {wins}/{len(d)} ({wins/len(d):.1%})")
        L.append("")

    block("ALL QUERIES", df)
    for method, g in df.groupby("method"):
        block(f"METHOD = {method}", g)
    for leaf, g in df.groupby("leaf"):
        lab = leaf if len(leaf) < 70 else leaf[:67] + "..."
        block(f"LEAF  {lab}", g)

    with open(out_path, "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[:34]))
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="results_tte/all_results.csv")
    p.add_argument("--tree_json",
                   default="results_tte/analysis/final_tree.json")
    p.add_argument("--output_dir", default="results_tte/leaf_heuristics")
    p.add_argument("--models_root", default=None)
    p.add_argument("--dgraph_dir", default="results/dgraphs")
    p.add_argument("--n_per_leaf", type=int, default=40)
    p.add_argument("--time_limit", type=int, default=120)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_unresolved", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "leaf_heuristic_results.csv")
    sum_path = os.path.join(args.output_dir, "leaf_heuristic_summary.txt")
    trace_path = os.path.join(args.output_dir, "leaf_heuristic_traces.csv")

    if args.summary_only:
        write_summary(csv_path, sum_path)
        return

    models_root = find_models_root(args.models_root)
    print(f"Models root: {models_root}\nTree: {args.tree_json}\n"
          f"Cap: {args.time_limit}s x {len(FAMILIES)} runs/query | "
          f"workers: {args.workers}\n")

    jobs, sel = build_jobs(args, models_root, csv_path)
    if not jobs:
        print("Nothing left to run.")
        if os.path.exists(csv_path):
            write_summary(csv_path, sum_path)
        return

    print(f"\nJobs: {len(jobs)}")
    print(sel.groupby(["method", "leaf"]).size().to_string())
    worst = len(jobs) * len(FAMILIES) * args.time_limit / 3600 / max(args.workers, 1)
    print(f"\nWorst case (every run hits the cap): {worst:.1f} h "
          f"at {args.workers} workers")
    if args.dry_run:
        return

    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = {exe.submit(_worker_safe, j): j for j in jobs}
        for fut in as_completed(futs):
            n += 1
            row, traces = fut.result()
            header = not os.path.exists(csv_path)
            pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(
                csv_path, mode="a", header=header, index=False, sep=";")
            if traces:
                th = not os.path.exists(trace_path)
                pd.DataFrame(traces, columns=TRACE_COLUMNS).to_csv(
                    trace_path, mode="a", header=th, index=False, sep=";")
            g = row.get("guided_quality")
            b = row.get("random_quality")
            delta = (f"{g - b:+.3f}" if isinstance(g, float)
                     and isinstance(b, float) else "  n/a")
            print(f"  [{n}/{len(jobs)}] {row['config']}/{row['model']} "
                  f"{row['method']:<8s} guided-random={delta} "
                  f"({(time.time()-t0)/3600:.2f} h)", flush=True)

    print(f"\nFinished in {(time.time()-t0)/3600:.2f} h")
    write_summary(csv_path, sum_path)


if __name__ == "__main__":
    main()
