"""
Figures for the sample-by-sample convergence experiment
(leaf_heuristic_experiment.py, v5): four curves per query -- worlds and
programs, each uniform and mass-guided -- grouped by the leaves of the trees.

Writes PDF and PNG to --output_dir:
    conv_mass_vs_samples    mass visited after k distinct samples, uniform vs
                            guided; A worlds, B programs; all queries
    conv_gap_vs_time        distance to the exact interval over time, four
                            curves; A queries the tree sends to worlds,
                            B queries it sends to programs
    conv_exact_vs_time      fraction of queries at the exact interval over
                            time, same split
    conv_leaf_exact_<tree>  fraction at the exact interval within the cap,
                            per leaf of the tree
    conv_leaf_gap_<tree>    distance to the exact interval over time, per leaf
    conv_leaf_mass_<tree>   mass visited after k samples, per leaf
    conv_example_intervals  l and u sample by sample for one query per
                            recommendation (A worlds, B programs)
and conv_curves.csv with every plotted mean and standard error.

Compare the order within a method against k, and the methods against time:
a program groups every world that induces it, so per sample it always holds
more mass. Curves still missing from the results (a failed run) are skipped.

    python sampling/generate_leaf_heuristic_figures.py \\
        --input_dir results_tte/leaf_heuristics_v5 --output_dir results_tte/figures
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from sampling.leaf_heuristic_experiment import CURVES, KEY, combine  # noqa: E402

COLOR = {"worlds": "#0072B2", "programs": "#D55E00"}      # Okabe-Ito
STYLE = {"random": "--", "guided": "-"}
LABEL = {("worlds", "random"): "Worlds, uniform",
         ("worlds", "guided"): "Worlds, ancestral",
         ("programs", "random"): "Programs, uniform",
         ("programs", "guided"): "Programs, marginal order"}
EPS = 1e-6                    # the stopping tolerance: gaps below it count as 0
K_MAX = 1000                  # every sample up to here is in the traces
K_GRID = np.unique(np.round(np.logspace(0, 3, 40)).astype(int))
ROWS = []                     # everything plotted, for conv_curves.csv


def setup():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11, "axes.grid": False,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def letter(ax, s):
    ax.text(-0.13, 1.02, s, transform=ax.transAxes, fontsize=14,
            fontweight="bold", va="bottom")


def stats(p):
    """Mean and standard error per column of a (query x grid) frame."""
    n = p.count()
    return p.mean(), p.std(ddof=1) / np.sqrt(n.clip(lower=1)), n


def record(fig, group, m, f, xname, x, mean, se, n):
    ROWS.extend(dict(figure=fig, group=group, method=m, family=f, x_name=xname,
                     x=a, mean=b, sem=c, n=d)
                for a, b, c, d in zip(x, mean, se, n))


def line(ax, x, mean, se, m, f, label=True):
    ax.plot(x, mean, STYLE[f], color=COLOR[m], lw=2,
            label=LABEL[(m, f)] if label else None)
    ax.fill_between(x, mean - se, mean + se, color=COLOR[m], alpha=0.15, lw=0)


# ── Data ─────────────────────────────────────────────────────────────────────
def load(d):
    res = combine(pd.read_csv(os.path.join(d, "results.csv"), sep=";"))
    tr = pd.concat([pd.read_csv(f, sep=";")
                    for f in sorted(glob.glob(os.path.join(d, "traces", "*.csv")))])
    tr = tr.drop_duplicates(KEY + ["method", "family", "k"], keep="last")
    present = [(m, f) for m, f, _ in CURVES if res[f"{m}_{f}_k"].notna().any()]
    missing = [f"{m}_{f}" for m, f, _ in CURVES if (m, f) not in present]
    if missing:
        print("[warn] curves with no results yet, left out:", missing)
    return res, tr, present


def mass_at_k(res, tr, m, f):
    """(query x k) mass visited. Every sample up to K_MAX is recorded, so a
    missing k lies after the run's end: carried forward if it ended at the
    exact interval or exhausted the space, censored (NaN) if the cap cut it."""
    c = f"{m}_{f}"
    t = tr[(tr.method == m) & (tr.family == f) & (tr.k <= K_MAX)]
    p = t.pivot_table(index=KEY, columns="k", values="mass", aggfunc="last")
    p = p.reindex(columns=range(1, K_MAX + 1)).ffill(axis=1)
    info = res.set_index(KEY)[[f"{c}_k", f"{c}_end"]].reindex(p.index)
    cut = ((info[f"{c}_end"] == "cap").values[:, None]
           & (np.arange(1, K_MAX + 1)[None, :] > info[f"{c}_k"].values[:, None]))
    return p.mask(cut)[K_GRID]


def gap_at_t(res, tr, m, f, grid):
    """(query x t) distance to the exact interval, (exact_l - l) + (u - exact_u),
    from the last record at or before each t; floored at EPS."""
    c = f"{m}_{f}"
    keys = res.loc[res[f"{c}_k"].notna(), KEY + ["exact_l", "exact_u"]]
    g = keys.merge(pd.DataFrame({"t": grid}), how="cross").sort_values("t")
    t = tr.loc[(tr.method == m) & (tr.family == f), KEY + ["t", "l", "u"]].sort_values("t")
    s = pd.merge_asof(g, t, on="t", by=KEY, direction="backward")
    s["l"], s["u"] = s.l.fillna(0.0), s.u.fillna(1.0)
    s["gap"] = ((s.exact_l - s.l) + (s.u - s.exact_u)).clip(lower=EPS)
    return s.pivot_table(index=KEY, columns="t", values="gap")


def exact_at_t(res, m, f, grid):
    """(query x t) 1 once the curve has reached the exact interval."""
    c = f"{m}_{f}"
    r = res[res[f"{c}_k"].notna()]
    tte = r[f"{c}_tte"].fillna(np.inf).values
    return pd.DataFrame((tte[:, None] <= grid[None, :]).astype(float),
                        index=pd.MultiIndex.from_frame(r[KEY]), columns=grid)


def subset(p, res, mask):
    idx = pd.MultiIndex.from_frame(res.loc[mask, KEY])
    return p.reindex(idx.intersection(p.index))


# ── Figures ──────────────────────────────────────────────────────────────────
def fig_mass(res, tr, present, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, m, s in zip(axes, ("worlds", "programs"), "AB"):
        for f in ("random", "guided"):
            if (m, f) not in present:
                continue
            mean, se, n = stats(mass_at_k(res, tr, m, f))
            line(ax, K_GRID, mean, se, m, f)
            record("conv_mass_vs_samples", "all", m, f, "k", K_GRID, mean, se, n)
            for k in (1, 10, 100):
                ax.plot(k, mean[k], "o", color=COLOR[m], ms=6)
                below = f == "random" and mean[k] > 0.1   # near 0 it would hit the axis
                ax.annotate(f"{mean[k]:.2f}", (k, mean[k]), textcoords="offset points",
                            xytext=(6, -14 if below else 6), fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(1, K_MAX)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Distinct samples visited, k")
        letter(ax, s)
    axes[0].set_ylabel("Probability mass visited")
    fig_legend(fig, present, ncol=len(present), y=1.08)
    save(fig, out, "conv_mass_vs_samples")


def fig_legend(fig, present, ncol, y):
    handles = [plt.Line2D([], [], color=COLOR[m], ls=STYLE[f], lw=2) for m, f in present]
    fig.legend(handles, [LABEL[c] for c in present], loc="upper center",
               ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


def _by_recommendation(res, tr, present, out, name, ylabel, value, logy):
    cap = float(res.time_limit.iloc[0])
    grid = np.logspace(-1, np.log10(cap), 60)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, rec, s in zip(axes, ("worlds", "programs"), "AB"):
        mask = res.rec_d3 == rec
        for m, f in present:
            mean, se, n = stats(subset(value(res, tr, m, f, grid), res, mask))
            line(ax, grid, mean.values, se.values, m, f)
            record(name, f"recommends {rec}", m, f, "t", grid, mean.values, se.values, n.values)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlim(grid[0], cap)
        ax.set_xlabel("Time (s)")
        ax.text(0.98, 0.04 if not logy else 0.96, f"n = {int(mask.sum())}",
                transform=ax.transAxes, ha="right", va="bottom" if not logy else "top",
                fontsize=11)
        letter(ax, s)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(loc="lower left" if logy else "upper left", frameon=False)
    save(fig, out, name)


def fig_gap(res, tr, present, out):
    _by_recommendation(res, tr, present, out, "conv_gap_vs_time",
                       "Distance to the exact interval", gap_at_t, logy=True)


def fig_exact(res, tr, present, out):
    _by_recommendation(res, tr, present, out, "conv_exact_vs_time",
                       "Queries at the exact interval (fraction)",
                       lambda r, t, m, f, g: exact_at_t(r, m, f, g), logy=False)


def leaves_of(res, tag):
    lv = sorted(res[f"leaf_{tag}"].unique(), key=lambda s: int(s[1:]))
    return [(l, res.loc[res[f"leaf_{tag}"] == l, f"rec_{tag}"].iloc[0],
             int((res[f"leaf_{tag}"] == l).sum())) for l in lv]


def fig_leaf_exact(res, present, out, tag):
    lv = leaves_of(res, tag)
    w = 0.8 / len(present)
    fig, ax = plt.subplots(figsize=(max(6, 1.25 * len(lv)), 4.2))
    for i, (m, f) in enumerate(present):
        c = f"{m}_{f}"
        x = np.arange(len(lv)) + (i - (len(present) - 1) / 2) * w
        p = np.array([res.loc[res[f"leaf_{tag}"] == l, f"{c}_tte"].notna().mean() for l, _, _ in lv])
        n = np.array([k for _, _, k in lv])
        se = np.sqrt(p * (1 - p) / n)
        ax.bar(x, p, w * 0.92, yerr=se, capsize=2, color=COLOR[m],
               alpha=1.0 if f == "guided" else 0.45, edgecolor=COLOR[m],
               hatch=None if f == "guided" else "///", label=LABEL[(m, f)])
        record(f"conv_leaf_exact_{tag}", "leaf", m, f, "leaf", [l for l, _, _ in lv], p, se, n)
        if len(lv) <= 8:
            for xi, pi, si in zip(x, p, se):
                ax.text(xi, pi + si + 0.02, f"{pi:.2f}", ha="center", va="bottom",
                        fontsize=8, rotation=90)
    ax.set_xticks(np.arange(len(lv)))
    ax.set_xticklabels([f"{l}\n{r}\nn={k}" for l, r, k in lv], fontsize=10)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_ylim(0, 1.25 if len(lv) <= 8 else 1.05)
    ax.set_ylabel("Queries at the exact interval\nwithin the cap (fraction)")
    ax.set_xlabel(f"Leaf of the depth-{tag[1:]} tree and its recommendation")
    ax.legend(loc="upper center", ncol=len(present), frameon=False, fontsize=10,
              bbox_to_anchor=(0.5, 1.13))
    save(fig, out, f"conv_leaf_exact_{tag}")


def fig_leaf_grid(res, tr, present, out, tag, kind):
    lv = leaves_of(res, tag)
    ncol = 4
    nrow = int(np.ceil(len(lv) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    cap = float(res.time_limit.iloc[0])
    grid = np.logspace(-1, np.log10(cap), 50)
    cache = {(m, f): (gap_at_t(res, tr, m, f, grid) if kind == "gap"
                      else mass_at_k(res, tr, m, f)) for m, f in present}
    for ax, (l, r, k) in zip(axes.flat, lv):
        mask = res[f"leaf_{tag}"] == l
        for m, f in present:
            mean, se, n = stats(subset(cache[(m, f)], res, mask))
            x = grid if kind == "gap" else K_GRID
            line(ax, x, mean.values, se.values, m, f, label=False)
            record(f"conv_leaf_{kind}_{tag}", l, m, f, "t" if kind == "gap" else "k",
                   x, mean.values, se.values, n.values)
        ax.set_xscale("log")
        if kind == "gap":
            ax.set_yscale("log")
        ax.text(0.04, 0.05 if kind == "gap" else 0.95, f"{l} · {r} · n={k}",
                transform=ax.transAxes, fontsize=10,
                va="bottom" if kind == "gap" else "top")
    for ax in axes.flat[len(lv):]:
        ax.axis("off")
    # The lowest panel of each column carries the x labels, also when the
    # bottom row is incomplete.
    for i, ax in enumerate(axes.flat[:len(lv)]):
        if i + ncol >= len(lv):
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.set_xlabel("Time (s)" if kind == "gap" else "Distinct samples, k")
    for ax in axes[:, 0]:
        ax.set_ylabel("Distance to exact" if kind == "gap" else "Mass visited")
    fig.tight_layout()
    fig_legend(fig, present, ncol=len(present), y=1.04)
    save(fig, out, f"conv_leaf_{kind}_{tag}")


def fig_example(res, tr, present, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, s in zip(axes, ("worlds", "programs"), "AB"):
        if not {(m, "random"), (m, "guided")} <= set(present):
            ax.axis("off")
            continue
        q = res[(res.rec_d3 == m) & res[f"{m}_random_tte"].notna()
                & res[f"{m}_guided_tte"].notna() & (res[f"{m}_random_tte"] >= 10)]
        if q.empty:
            ax.axis("off")
            continue
        # The query of median speed-up, so the example is typical, not the best.
        q = q.assign(r=np.log(q[f"{m}_guided_tte"] / q[f"{m}_random_tte"])).sort_values("r")
        row = q.iloc[len(q) // 2]
        key = tuple(row[KEY])
        ax.axhspan(row.exact_l, row.exact_u, color="0.85", lw=0, label="Exact interval")
        if row.exact_u - row.exact_l < 0.01:
            ax.axhline(row.exact_l, color="0.5", lw=1)
        mine = ((tr.method == m) & (tr.config == key[0]) & (tr.model == key[1])
                & (tr.literal == key[2]) & (tr.run == key[3]))
        for f in ("random", "guided"):
            t = tr[mine & (tr.family == f)]
            for col in ("l", "u"):
                ax.step(t.k, t[col], STYLE[f], where="post", color=COLOR[m], lw=2,
                        label=f"{LABEL[(m, f)]}: l and u" if col == "l" else None)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Distinct samples visited, k")
        ax.text(1.0, 1.02, f"{key[0]}/{key[1]}, {key[2]}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=10)
        letter(ax, s)
        print(f"example {m}: {key}  tte uniform {row[f'{m}_random_tte']:.1f} s, "
              f"guided {row[f'{m}_guided_tte']:.1f} s")
    axes[0].set_ylabel("Probability")
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.85")] + \
        [plt.Line2D([], [], color=COLOR[m], ls=STYLE[f], lw=2) for m, f in present]
    labels = ["Exact interval"] + [f"{LABEL[c]}: l and u" for c in present]
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), fontsize=10)
    save(fig, out, "conv_example_intervals")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="results_tte/leaf_heuristics_v5")
    p.add_argument("--output_dir", default="results_tte/figures")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup()
    res, tr, present = load(args.input_dir)
    tags = [c[5:] for c in res.columns if c.startswith("leaf_")]
    fig_mass(res, tr, present, args.output_dir)
    fig_gap(res, tr, present, args.output_dir)
    fig_exact(res, tr, present, args.output_dir)
    fig_example(res, tr, present, args.output_dir)
    for tag in tags:
        fig_leaf_exact(res, present, args.output_dir, tag)
        fig_leaf_grid(res, tr, present, args.output_dir, tag, "gap")
        fig_leaf_grid(res, tr, present, args.output_dir, tag, "mass")
    pd.DataFrame(ROWS).to_csv(os.path.join(args.output_dir, "conv_curves.csv"), index=False)
    print("wrote conv_curves.csv")


if __name__ == "__main__":
    main()
