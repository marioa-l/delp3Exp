"""
Sample-by-sample convergence of both samplers, uniform and guided, on the
leaves of the final decision trees.

Every selected query runs four curves under the same time cap, whatever the
tree recommends, so a leaf shows both whether the guided order beats uniform
sampling within a method and whether the recommended method closes the
interval sooner than the other one:

    worlds    random   uniform over world ids, no repeats (as in the ground truth)
    worlds    guided   mode="ancestral": ancestral sampling from the EM without
                       replacement; heavy worlds first, P(w) from the CPTs
    programs  random   uniform over the enumerated programs, no repeats
    programs  guided   mode="marginal_order": the same programs in decreasing
                       product of the annotation marginals, no repeats

Queries: non-trivial only (the exact interval is not [1,1], [0,0] or [0,1]:
the answer changes with the world), resolved in the ground truth unless
--include_unresolved. Each tree in --trees routes every query to a leaf, and
up to --n_per_leaf queries are drawn per leaf of each tree (union over trees).

Per curve, one record per distinct sample k (a new world or a new program):
elapsed time, solver calls, l, u and the probability mass visited so far.
Every sample up to --dense_k is kept, then a geometric grid in k (ratio
1.05), the last sample before each point of a time grid, and always the last
one. Plotting against k compares the order of a
method's samples; plotting against time compares the methods, since a
program groups every world that induces it and costs more per sample.

Resumable (finished (query, run) pairs are skipped), incremental, parallel.

Writes to <output_dir>:
    results.csv            one row per query and run: leaf and recommendation
                           under each tree, final state of each curve
    traces/<config>.csv    the sample-by-sample records
    summary.txt            per tree and leaf: mass after k samples, gap to the
                           exact interval after t seconds, exact reached
    summary.csv            the same numbers in long format, for figures

Usage (icic):
    python sampling/leaf_heuristic_experiment.py --n_per_leaf 60 \\
        --time_limit 280 --workers 16 \\
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \\
        --output_dir results_tte/leaf_heuristics_v5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byProgramSampling import Programs      # noqa: E402
from sampling.byWorldSampling import Worlds          # noqa: E402

DEFAULT_ROOTS = [
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
]
DEFAULT_TREES = ("results_tte/analysis/d3/final_tree.json,"
                 "results_tte/analysis/d5/final_tree.json")

# (method, family, sampler mode)
CURVES = [("worlds", "random", "random"), ("worlds", "guided", "ancestral"),
          ("programs", "random", "random"), ("programs", "guided", "marginal_order")]
CURVE_FIELDS = ["k", "calls", "l", "u", "mass", "tte", "end", "wall", "error"]
TRACE_COLUMNS = ["config", "model", "literal", "run", "method", "family",
                 "k", "t", "calls", "l", "u", "mass"]
KEY = ["config", "model", "literal", "run"]
MASS_AT_K = (1, 2, 3, 5, 10, 100, 1000)
GAP_AT_T = (1, 10, 60)


def find_models_root(override):
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"Models root not found; tried {DEFAULT_ROOTS}")


def n_leaves(node):
    return 1 if node["kind"] == "leaf" else n_leaves(node["yes"]) + n_leaves(node["no"])


def apply_plain(node, row):
    """
    Walk the portable comparison tree for one feature dict; returns
    (method, leaf id, path). Leaves are numbered L1.. left to right, the
    "yes" branch first. Kept here instead of imported from final_tree, which
    pulls in sklearn and matplotlib that the cluster does not always match.
    """
    path, offset = [], 0
    while node["kind"] != "leaf":
        a, b = row.get(node["a"]), row.get(node["b"])
        went = not (a is None or b is None or pd.isna(a) or pd.isna(b)) \
            and float(a) < float(b)      # undecidable -> the ELSE branch
        path.append(f"{node['a']} {'<' if went else '>='} {node['b']}")
        if not went:
            offset += n_leaves(node["yes"])
        node = node["yes"] if went else node["no"]
    return node["cls"], f"L{offset + 1}", " & ".join(path)


def time_grid(cap):
    g = [round(0.1 * i, 2) for i in range(1, 51)]          # 0.1 .. 5 s
    g += [round(5 + 0.5 * i, 2) for i in range(1, 51)]     # 5.5 .. 30 s
    g += [float(t) for t in range(32, int(cap) + 1, 2)]    # 32 .. cap
    return g


def thin(history, dense_k, cap, ratio=1.05):
    """
    (k, record) pairs to keep: every sample up to dense_k, then a geometric
    grid in k, the last sample before each point of a time grid (so the state
    at any second can still be read off), and the very last sample.
    """
    grid, gi, out, nxt = time_grid(cap), 0, [], dense_k
    for k, h in enumerate(history, 1):
        while gi < len(grid) and grid[gi] < h["time"]:
            gi += 1
        nxt_t = history[k]["time"] if k < len(history) else float("inf")
        crosses = gi < len(grid) and grid[gi] < nxt_t
        if k <= dense_k or k >= nxt or crosses or k == len(history):
            out.append((k, h))
            if k > dense_k and k >= nxt:
                nxt = k * ratio
    return out


def _worker(job):
    (cfg, model, literal, run, model_path, save_dir, exact_l, exact_u,
     cap, epsilon, dense_k, seed, info) = job
    np.random.seed(zlib.crc32(f"{cfg}/{model}/{literal}/{run}/{seed}".encode()))
    os.makedirs(save_dir, exist_ok=True)
    row = {**info, "config": cfg, "model": model, "literal": literal,
           "run": run, "exact_l": exact_l, "exact_u": exact_u,
           "time_limit": cap}
    traces, samplers = [], {}
    for method, fam, mode in CURVES:
        c = f"{method}_{fam}"
        t0 = time.time()
        try:
            # One sampler per method: the program sampler enumerates its
            # program list once and both of its curves reuse it.
            if method not in samplers:
                cls = Worlds if method == "worlds" else Programs
                samplers[method] = cls(model_path, save_dir + "/")
                samplers[method].utils.get_interest_lit = lambda: [literal]
            s = samplers[method]
            kw = dict(percentile_samples=0, max_seconds=cap, mode=mode,
                      stop_at_exact={literal: (exact_l, exact_u)},
                      stop_epsilon=epsilon, info=f"_{literal}_{c}_r{run}")
            hist = (s.start_sampling(source="random", **kw) if method == "worlds"
                    else s.start_sampling(**kw)).get(literal, [])
            st = s.results["status"].get(literal, {})
            last = hist[-1] if hist else {}
            tte = st.get("time_to_exact")
            # The sampling loop only stops early when it reaches the exact
            # interval or runs out of samples; otherwise its clock passed the cap.
            end = ("exact" if tte is not None else
                   "cap" if s.results["data"]["time"] >= cap - 1e-3 else "exhausted")
            row.update({f"{c}_k": len(hist), f"{c}_calls": last.get("calls", 0),
                        f"{c}_l": last.get("l", 0.0), f"{c}_u": last.get("u", 1.0),
                        f"{c}_mass": last.get("mass", 0.0), f"{c}_tte": tte,
                        f"{c}_end": end})
            for k, h in thin(hist, dense_k, cap):
                traces.append((cfg, model, literal, run, method, fam, k,
                               h["time"], h["calls"], h["l"], h["u"], h["mass"]))
        except Exception as e:  # one curve failing does not lose the others
            row[f"{c}_error"] = str(e)[:200]
        row[f"{c}_wall"] = time.time() - t0
    row["ok"] = not any(f"{m}_{f}_error" in row for m, f, _ in CURVES)
    return row, traces


def _worker_safe(job):
    try:
        return _worker(job)
    except Exception as e:  # pragma: no cover
        return ({"config": job[0], "model": job[1], "literal": job[2],
                 "run": job[3], "ok": False, "error": str(e)[:200]}, [])


def load_trees(spec):
    return {Path(p).parent.name: json.load(open(p))["tree"]
            for p in spec.split(",") if p.strip()}


def select_queries(args, trees):
    df = pd.read_csv(args.input_csv, sep=";").dropna(subset=["exact_l", "exact_u"])
    e = 1e-6
    trivial = ((df.exact_l > 1 - e) | (df.exact_u < e)
               | ((df.exact_l < e) & (df.exact_u > 1 - e)))
    df = df[~trivial]
    if not args.include_unresolved:
        df = df[df.gt_winner.isin(["worlds", "programs"])]
    for tag, tree in trees.items():
        p = [apply_plain(tree, r) for r in df.to_dict("records")]
        df = df.assign(**{f"rec_{tag}": [x[0] for x in p],
                          f"leaf_{tag}": [x[1] for x in p],
                          f"path_{tag}": [x[2] for x in p]})
    keys = set()
    for tag in trees:
        for _, g in df.groupby(f"leaf_{tag}"):
            g = g.sample(n=min(args.n_per_leaf, len(g)), random_state=args.seed)
            keys |= set(zip(g.config, g.model, g.literal))
    sel = df[[k in keys for k in zip(df.config, df.model, df.literal)]]
    print(f"Non-trivial candidates: {len(df)}; selected: {len(sel)}")
    return sel


def build_jobs(args, sel, trees, models_root, csv_path):
    done = set()
    if os.path.exists(csv_path):
        d = pd.read_csv(csv_path, sep=";", usecols=KEY)
        done = set(map(tuple, d.astype({"run": int}).itertuples(index=False, name=None)))
    info_cols = (["gt_winner", "time_worlds", "time_programs"]
                 + [f"{x}_{t}" for t in trees for x in ("rec", "leaf", "path")])
    jobs = []
    for r in sel.to_dict("records"):
        mp = os.path.join(models_root, r["config"], f"{r['model']}.json")
        if not os.path.exists(mp):
            continue
        for run in range(args.runs):
            if (r["config"], r["model"], r["literal"], run) in done:
                continue
            jobs.append((r["config"], r["model"], r["literal"], run, mp,
                         os.path.join(args.output_dir, "runs", r["config"], r["model"]),
                         float(r["exact_l"]), float(r["exact_u"]), args.time_limit,
                         args.epsilon, args.dense_k, args.seed,
                         {c: r[c] for c in info_cols}))
    if done:
        print(f"Resuming: {len(done)} (query, run) pairs already on disk")
    return jobs


# ── Summary ──────────────────────────────────────────────────────────────────
def _state_at(tr, col, x, res):
    """Per (query, run, curve): the last record with tr[col] <= x."""
    s = tr[tr[col] <= x].groupby(KEY + ["method", "family"])[["mass", "l", "u"]].last()
    full = res.set_index(KEY)[["exact_l", "exact_u"]]
    idx = pd.MultiIndex.from_tuples(
        [k + (m, f) for k in full.index for m, f, _ in CURVES],
        names=KEY + ["method", "family"])
    s = s.reindex(idx)
    s["mass"] = s["mass"].fillna(0.0)       # no sample yet: nothing visited
    s["l"] = s["l"].fillna(0.0)
    s["u"] = s["u"].fillna(1.0)
    ex = full.reindex(s.index.droplevel(["method", "family"]))
    s["gap"] = ((ex.exact_l.values - s.l) + (s.u - ex.exact_u.values)).clip(lower=0)
    return s


def write_summary(output_dir, trees):
    res = pd.read_csv(os.path.join(output_dir, "results.csv"), sep=";")
    res = res[res.ok == True]  # noqa: E712
    tdir = os.path.join(output_dir, "traces")
    if res.empty or not os.path.isdir(tdir):
        print("[warn] nothing to summarise")
        return
    tr = pd.concat([pd.read_csv(os.path.join(tdir, f), sep=";")
                    for f in sorted(os.listdir(tdir)) if f.endswith(".csv")])
    tr = tr.merge(res[KEY], on=KEY)

    # One value per (query, run, curve) and metric, then averaged over runs.
    metrics = []
    for k in MASS_AT_K:
        s = _state_at(tr, "k", k, res)
        # A run stopped by the cap before reaching k samples is censored;
        # one that reached the exact interval or exhausted the space keeps
        # its final state.
        long = s.reset_index()
        ends = res.melt(id_vars=KEY, value_vars=[f"{m}_{f}_end" for m, f, _ in CURVES],
                        var_name="c", value_name="end")
        ks = res.melt(id_vars=KEY, value_vars=[f"{m}_{f}_k" for m, f, _ in CURVES],
                      var_name="c", value_name="kk")
        ends["c"], ks["c"] = ends.c.str[:-4], ks.c.str[:-2]
        long["c"] = long.method + "_" + long.family
        long = long.merge(ends, on=KEY + ["c"]).merge(ks, on=KEY + ["c"])
        long.loc[(long.end == "cap") & (long.kk < k), "mass"] = np.nan
        metrics.append(long.assign(metric=f"mass@k{k}", value=long.mass))
    for t in GAP_AT_T:
        long = _state_at(tr, "t", t, res).reset_index()
        metrics.append(long.assign(metric=f"gap@{t}s", value=long.gap))
    m = pd.concat(metrics)[KEY + ["method", "family", "metric", "value"]]
    m = m.groupby(["config", "model", "literal", "method", "family", "metric"],
                  as_index=False).value.mean()

    per_q = res.groupby(["config", "model", "literal"], as_index=False).first()
    for meth, fam, _ in CURVES:
        c = f"{meth}_{fam}"
        per_q[f"{c}_reached"] = res.groupby(["config", "model", "literal"])[
            f"{c}_tte"].apply(lambda s: s.notna().mean()).values

    L, rows = [f"=== Sample-by-sample convergence, {len(per_q)} queries, "
               f"{res.run.nunique()} run(s), cap {int(res.time_limit.iloc[0])} s ===", "",
               "mass@kN : probability mass visited after N distinct samples",
               "gap@Ts  : (exact_l - l) + (u - exact_u) after T seconds",
               "values  : mean +/- standard error over the queries of the leaf", ""], []
    cols = [f"mass@k{k}" for k in (1, 2, 3, 10, 100)] + [f"gap@{t}s" for t in GAP_AT_T]
    for tag in trees:
        L.append(f"##### TREE {tag}")
        L += _recommended_vs_other(per_q, tag)
        for leaf in sorted(per_q[f"leaf_{tag}"].unique(), key=lambda s: int(s[1:])):
            g = per_q[per_q[f"leaf_{tag}"] == leaf]
            rec = g[f"rec_{tag}"].iloc[0]
            gt = g.gt_winner.value_counts().to_dict()
            L.append(f"{tag} {leaf}  recommends {rec}  n={len(g)}  gt={gt}")
            L.append(f"    {g[f'path_{tag}'].iloc[0]}")
            L.append("    " + f"{'':18s}" + "".join(f"{c:>17s}" for c in cols)
                     + f"{'exact':>8s}{'med tte':>9s}")
            mg = m.merge(g[["config", "model", "literal"]])
            for meth, fam, _ in CURVES:
                c = f"{meth}_{fam}"
                line = f"    {meth + ' ' + fam:18s}"
                for col in cols:
                    v = mg[(mg.method == meth) & (mg.family == fam) & (mg.metric == col)].value.dropna()
                    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                    line += f"{v.mean():>10.4f}+/-{se:.4f}" if len(v) else f"{'n/a':>17s}"
                    rows.append(dict(tree=tag, leaf=leaf, recommends=rec, method=meth,
                                     family=fam, metric=col, mean=v.mean(), sem=se, n=len(v)))
                tte = res.merge(g[["config", "model", "literal"]])[f"{c}_tte"]
                L.append(line + f"{g[f'{c}_reached'].sum():>8.0f}{tte.median():>9.1f}")
                rows.append(dict(tree=tag, leaf=leaf, recommends=rec, method=meth, family=fam,
                                 metric="reached_exact", mean=g[f"{c}_reached"].mean(),
                                 sem=g[f"{c}_reached"].std(ddof=1) / np.sqrt(len(g))
                                 if len(g) > 1 else 0.0, n=len(g)))
            L.append("")
    open(os.path.join(output_dir, "summary.txt"), "w").write("\n".join(L))
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print("\n".join(L))


def _recommended_vs_other(per_q, tag):
    """How often the recommended method reaches the exact interval first,
    with uniform and with guided sampling (a method that never reaches it
    loses; queries where neither does are left out)."""
    out = []
    for fam in ("random", "guided"):
        tw = per_q[f"worlds_{fam}_tte"].fillna(np.inf)
        tp = per_q[f"programs_{fam}_tte"].fillna(np.inf)
        ok = (np.isfinite(tw) | np.isfinite(tp)).values
        win = np.where(tw < tp, "worlds", "programs")
        hit = pd.Series(win == per_q[f"rec_{tag}"].values)[ok].astype(float)
        se = hit.std(ddof=1) / np.sqrt(len(hit)) if len(hit) > 1 else 0.0
        out.append(f"recommended method first to the exact interval ({fam}): "
                   f"{hit.mean():.4f} +/- {se:.4f}  (n={ok.sum()})")
    return out + [""]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="results_tte/all_results.csv")
    p.add_argument("--trees", default=DEFAULT_TREES,
                   help="comma-separated final_tree.json paths; each is tagged "
                        "by its folder name (d3, d5)")
    p.add_argument("--output_dir", default="results_tte/leaf_heuristics_v5")
    p.add_argument("--models_root", default=None)
    p.add_argument("--n_per_leaf", type=int, default=60)
    p.add_argument("--runs", type=int, default=1,
                   help="repetitions of the four curves per query")
    p.add_argument("--time_limit", type=int, default=280)
    p.add_argument("--dense_k", type=int, default=1000)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include_unresolved", action="store_true")
    p.add_argument("--summary_only", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    trees = load_trees(args.trees)
    os.makedirs(os.path.join(args.output_dir, "traces"), exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")
    if args.summary_only:
        write_summary(args.output_dir, trees)
        return

    models_root = find_models_root(args.models_root)
    sel = select_queries(args, trees)
    jobs = build_jobs(args, sel, trees, models_root, csv_path)
    print(f"Models root: {models_root} | trees: {list(trees)} | "
          f"cap {args.time_limit}s x {len(CURVES)} curves x {args.runs} run(s)")
    for tag in trees:
        print(f"\n{tag}:\n" + sel.groupby([f"leaf_{tag}", f"rec_{tag}"]).size().to_string())
    worst = len(jobs) * len(CURVES) * args.time_limit / 3600 / max(args.workers, 1)
    print(f"\nJobs: {len(jobs)} | worst case (every curve hits the cap): "
          f"{worst:.1f} h at {args.workers} workers")
    if args.dry_run or not jobs:
        if not jobs and os.path.exists(csv_path):
            write_summary(args.output_dir, trees)
        return

    columns = (KEY + ["exact_l", "exact_u", "time_limit"] + list(jobs[0][-1])
               + [f"{m}_{f}_{x}" for m, f, _ in CURVES for x in CURVE_FIELDS]
               + ["ok", "error"])
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futs = [exe.submit(_worker_safe, j) for j in jobs]
        for fut in as_completed(futs):
            n += 1
            row, traces = fut.result()
            pd.DataFrame([row]).reindex(columns=columns).to_csv(
                csv_path, mode="a", header=not os.path.exists(csv_path),
                index=False, sep=";")
            if traces:
                tp = os.path.join(args.output_dir, "traces", f"{row['config']}.csv")
                pd.DataFrame(traces, columns=TRACE_COLUMNS).to_csv(
                    tp, mode="a", header=not os.path.exists(tp), index=False, sep=";",
                    float_format="%.12g")
            tte = " ".join(f"{m[0]}{f[0]}={row.get(f'{m}_{f}_tte') or float('nan'):.1f}"
                           for m, f, _ in CURVES)
            print(f"  [{n}/{len(jobs)}] {row['config']}/{row['model']} {row['literal']} "
                  f"tte {tte} ({(time.time() - t0) / 3600:.2f} h)", flush=True)

    print(f"\nFinished in {(time.time() - t0) / 3600:.2f} h")
    write_summary(args.output_dir, trees)


if __name__ == "__main__":
    main()
