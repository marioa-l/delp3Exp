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

Usage:
    python sampling/guided_ground_truth.py
    python sampling/final_tree.py --input_csv results_tte/all_results_guided.csv \\
        --depth 3 --ban_tiers 2,3 --min_leaf 25 \\
        --output_dir results_tte/analysis_guided/d3
"""

from __future__ import annotations

import argparse
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
from sampling.leaf_heuristic_experiment import combine, find_models_root  # noqa: E402

Q = ["config", "model", "literal"]


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="results_tte/all_results.csv")
    p.add_argument("--results_dir", default="results_tte/leaf_heuristics_v5")
    p.add_argument("--output_csv", default="results_tte/all_results_guided.csv")
    p.add_argument("--models_root", default=None)
    p.add_argument("--n_draws", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

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
