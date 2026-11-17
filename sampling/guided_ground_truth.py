"""
Ground truth for the guided samplers, plus a cheap BN-entropy attribute.

The original ground truth (results_tte/all_results.csv) labels each query
with the sampler that reaches the exact interval first when both sample
uniformly; under uniform sampling the cost follows the size of the space, so
the tree reduces to "sample the smaller space". This script relabels every
non-trivial query with the sampler that gets there first when both use their
mass-guided order (worlds: ancestral, programs: marginal order), read from
the runs of leaf_heuristic_experiment.py, and adds

    em_entropy_mc   joint entropy of the EM in bits, the mean of -log2 P(w)
                    over worlds drawn by ancestral sampling. P(w) is the
                    product of the CPT entries drawn, so each draw is O(|E|)
                    with no inference (tier 1); em_entropy needs
                    LazyPropagation or enumeration (tier 2).

Queries that neither guided sampler finished within the cap are labelled
"unresolved", and final_tree.py leaves them out as it does with the original
ground truth. Rows keep every attribute of all_results.csv; the uniform label
stays as gt_winner_uniform (text, so never a predictor).

--evaluate DIR compares selection policies under the guided samplers on
every non-trivial query: the trees in DIR/d*/ (out-of-fold predictions from
final_tree_predictions.csv, and the shown tree), the uniform-trained trees in
results_tte/analysis/d*/, always one method, and the oracle. Cost is the time
to the exact interval capped at --cap, so a miss costs the cap. Writes
DIR/policy_report.txt and DIR/policy.csv.

Usage:
    python sampling/guided_ground_truth.py
    python sampling/final_tree.py --input_csv results_tte/all_results_guided.csv \\
        --depth 3 --ban_tiers 2,3 --min_leaf 25 \\
        --output_dir results_tte/analysis_guided/d3
    python sampling/guided_ground_truth.py --evaluate results_tte/analysis_guided
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyAgrum as gum

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byWorldSampling import UniqueAncestral                     # noqa: E402
from sampling.leaf_heuristic_experiment import (  # noqa: E402
    apply_plain, combine, find_models_root)

Q = ["config", "model", "literal"]
OLD_TREES = "results_tte/analysis"


def entropy_mc(bn, n, rng):
    """Mean and standard error of -log2 P(w) over n ancestral draws."""
    ua = UniqueAncestral(bn)                 # CPTs as arrays, topological order
    vals, logp = {}, np.zeros(n)
    for v, pa, cpt in zip(ua.order, ua.parents, ua.cpt):
        p1 = cpt[tuple(vals[q] for q in pa) + (1,)]       # P(v=1 | drawn parents)
        x = (rng.random(n) < p1).astype(np.int64)
        logp += np.log2(np.where(x == 1, p1, 1 - p1))
        vals[v] = x
    return -logp.mean(), logp.std(ddof=1) / np.sqrt(n)


def evaluate(args):
    g = pd.read_csv(args.output_csv, sep=";")
    lab = g.gt_winner.isin(["worlds", "programs"]).values
    tw, tp = (g[f"time_{m}"].fillna(np.inf).values for m in ("worlds", "programs"))
    rows = g.to_dict("records")
    route = lambda path: np.array([apply_plain(json.load(open(path))["tree"], r)[0] for r in rows])
    # Each rule is the algorithm it picks per query; the oracle (None) picks
    # the faster one under whichever order is being costed.
    n = len(g)
    choice = {"always worlds": np.full(n, "worlds"), "always programs": np.full(n, "programs")}
    for d in sorted(os.listdir(args.evaluate)):
        new = os.path.join(args.evaluate, d, "final_tree.json")
        if not os.path.exists(new):
            continue
        shown = route(new)
        oof = g[Q].merge(pd.read_csv(os.path.join(args.evaluate, d, "final_tree_predictions.csv"),
                                     sep=";")[Q + ["pred_final"]], on=Q, how="left").pred_final
        # unresolved rows have no out-of-fold prediction; they cost the cap either way
        choice[f"new {d}, held out"] = oof.fillna(pd.Series(shown)).values
        choice[f"new {d}, shown tree"] = shown
        if os.path.exists(os.path.join(OLD_TREES, d, "final_tree.json")):
            choice[f"old {d} (uniform labels)"] = route(os.path.join(OLD_TREES, d, "final_tree.json"))
    choice["oracle"] = None

    def cost(rec, w, p):
        t = np.minimum(w, p) if rec is None else np.where(rec == "worlds", w, p)
        return t, np.minimum(t, args.cap)

    best = cost(None, tw, tp)[1]
    L = [f"Policies under the guided samplers: {n} non-trivial queries "
         f"({lab.sum()} labelled), time to the exact interval capped at {args.cap:.0f} s",
         f"{'policy':34s}{'reached':>9s}{'mean time (s)':>18s}{'median':>8s}"
         f"{'extra vs oracle':>17s}{'accuracy':>16s}"]
    out = []
    for name, rec in choice.items():
        t, tc = cost(rec, tw, tp)
        se = tc.std(ddof=1) / np.sqrt(n)
        acc = ""
        row = dict(policy=name, reached=int(np.isfinite(t).sum()), n=n, mean_time=tc.mean(),
                   sem_time=se, median_time=np.median(tc), extra_vs_oracle=(tc - best).mean())
        if rec is not None:
            hit = (rec[lab] == g.gt_winner.values[lab]).astype(float)
            row.update(accuracy=hit.mean(), sem_accuracy=hit.std(ddof=1) / np.sqrt(len(hit)))
            acc = f"{row['accuracy']:.3f} +/- {row['sem_accuracy']:.3f}"
        L.append(f"{name:34s}{row['reached']:>9d}{tc.mean():>11.1f} +/- {se:4.1f}"
                 f"{np.median(tc):>8.1f}{row['extra_vs_oracle']:>17.1f}{acc:>16s}")
        out.append(row)
    L.append("accuracy: against the guided label, on the labelled queries; "
             "'held out' uses the GroupKFold-by-model predictions")

    # The same rules when both algorithms sample uniformly (convergence runs),
    # on the queries that have uniform runs.
    res = combine(pd.read_csv(os.path.join(args.results_dir, "results.csv"), sep=";"))
    u = g[Q].merge(res.groupby(Q, as_index=False)[["worlds_random_k", "worlds_random_tte",
                                                    "programs_random_tte"]].median(),
                   on=Q, how="left")
    have = u.worlds_random_k.notna().values
    if have.any():
        uw, up = (u[f"{m}_random_tte"].fillna(np.inf).values for m in ("worlds", "programs"))
        L += ["", f"Same rules, uniform against guided selection ({have.sum()} queries with "
                  f"uniform runs)",
              f"{'policy':34s}{'reached':>9s}{'uniform (s)':>18s}{'guided (s)':>18s}"]
        for name, rec in choice.items():
            r = None if rec is None else rec[have]
            t, tu = cost(r, uw[have], up[have])
            tg = cost(r, tw[have], tp[have])[1]
            m = len(tu)
            L.append(f"{name:34s}{int(np.isfinite(t).sum()):>9d}"
                     f"{tu.mean():>11.1f} +/- {tu.std(ddof=1) / np.sqrt(m):4.1f}"
                     f"{tg.mean():>11.1f} +/- {tg.std(ddof=1) / np.sqrt(m):4.1f}")
            out.append(dict(policy=name + ", uniform selection", reached=int(np.isfinite(t).sum()),
                            n=m, mean_time=tu.mean(), sem_time=tu.std(ddof=1) / np.sqrt(m),
                            median_time=np.median(tu)))
    open(os.path.join(args.evaluate, "policy_report.txt"), "w").write("\n".join(L) + "\n")
    pd.DataFrame(out).to_csv(os.path.join(args.evaluate, "policy.csv"), index=False)
    print("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="results_tte/all_results.csv")
    p.add_argument("--results_dir", default="results_tte/leaf_heuristics_v5")
    p.add_argument("--output_csv", default="results_tte/all_results_guided.csv")
    p.add_argument("--models_root", default=None)
    p.add_argument("--n_draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--evaluate", default=None,
                   help="folder with d*/final_tree.json trained on the guided labels")
    p.add_argument("--cap", type=float, default=280.0)
    args = p.parse_args()
    if args.evaluate:
        return evaluate(args)

    res = combine(pd.read_csv(os.path.join(args.results_dir, "results.csv"), sep=";"))
    res = res[res.worlds_guided_k.notna() & res.programs_guided_k.notna()]
    t = res[Q].assign(**{m: res[f"{m}_guided_tte"].fillna(np.inf)
                         for m in ("worlds", "programs")})
    t = t.groupby(Q, as_index=False)[["worlds", "programs"]].median()   # over runs
    t["gt_winner"] = np.select(
        [np.isinf(t.worlds) & np.isinf(t.programs), t.worlds < t.programs],
        ["unresolved", "worlds"], "programs")
    t = t.rename(columns={"worlds": "time_worlds", "programs": "time_programs"})
    t[["time_worlds", "time_programs"]] = t[["time_worlds", "time_programs"]].replace(np.inf, np.nan)

    gt = pd.read_csv(args.input_csv, sep=";")
    out = (gt.rename(columns={"gt_winner": "gt_winner_uniform"})
             .drop(columns=["time_worlds", "time_programs"]).merge(t, on=Q))

    root = find_models_root(args.models_root)
    ent = []
    for cfg, model in sorted(set(zip(out.config, out.model))):
        n = re.search(r"(\d+)model$", model).group(1)
        bn = gum.loadBN(os.path.join(root, cfg, f"BN{n}.bifxml"))
        rng = np.random.default_rng(zlib.crc32(f"{cfg}/{model}/{args.seed}".encode()))
        ent.append((cfg, model, *entropy_mc(bn, args.n_draws, rng)))
    ent = pd.DataFrame(ent, columns=["config", "model", "em_entropy_mc", "se"])
    out = out.merge(ent.drop(columns="se"), on=["config", "model"])
    out.to_csv(args.output_csv, sep=";", index=False)

    lab = out[out.gt_winner != "unresolved"]
    both = ent.merge(gt[["config", "model", "em_entropy"]].drop_duplicates(["config", "model"]))
    both = both.dropna()
    L = [f"Guided ground truth: {len(out)} non-trivial queries, {len(ent)} models",
         f"labels: {out.gt_winner.value_counts().to_dict()}", "",
         "uniform label (rows) vs guided label (columns):",
         pd.crosstab(out.gt_winner_uniform, out.gt_winner).to_string(), "",
         f"em_entropy_mc ({args.n_draws} draws): standard error median "
         f"{ent.se.median():.4f}, max {ent.se.max():.4f} bits",
         f"vs em_entropy on {len(both)} models: Pearson r "
         f"{np.corrcoef(both.em_entropy_mc, both.em_entropy)[0, 1]:.4f}, "
         f"mean |diff| {(both.em_entropy_mc - both.em_entropy).abs().mean():.4f} bits, "
         f"max |diff| {(both.em_entropy_mc - both.em_entropy).abs().max():.4f} bits",
         f"labelled rows for the tree: {len(lab)} "
         f"({(lab.gt_winner == 'worlds').sum()} worlds, "
         f"{(lab.gt_winner == 'programs').sum()} programs)"]
    open(os.path.splitext(args.output_csv)[0] + "_report.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
