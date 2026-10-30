"""
Convergence of the approximate interval over time, per sampling family.

The final state at the time cap says where each family ends up but not how it
got there. These curves answer the question the cap hides: whether the
leaf-guided heuristics close the interval faster in the first seconds, even
when the three families are indistinguishable once the budget is spent.

Two corrections matter for the curves to mean anything:

  * A run that ends before the cap -- because it reached the exact interval,
    or because it exhausted its space -- keeps its final interval afterwards.
    Dropping it from later time points instead would leave only the hard
    queries in the average and make every family look worse over time.
  * The final state is taken from the results table rather than from the last
    point that happened to land on the trace grid, since a run that reaches
    the exact interval between two grid points would otherwise be recorded as
    still short of it.

Reads:
    results_tte/leaf_heuristics_v4/leaf_heuristic_traces.csv
    results_tte/leaf_heuristics_v4/leaf_heuristic_results.csv

Writes PDF and PNG to --output_dir (default results_tte/figures/):
    convergence.{pdf,png}

Usage:
    python sampling/generate_convergence_figures.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FONT = "DejaVu Sans"
C = {"random": "#8C8C8C", "generic": "#D55E00", "guided": "#0072B2"}
# The comparison the paper makes is random against guided; generic is reported
# in the text only, so it is read from the data but not drawn.
SHOWN = ["random", "guided"]
M_COL = {"worlds": "#0072B2", "programs": "#D55E00"}
KEY = ["config", "model", "literal"]


def style(ax, xlabel="", ylabel=""):
    ax.set_xlabel(xlabel, fontsize=7.5, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=7.5, fontname=FONT)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=7)
    ax.minorticks_on()


def build(trace_csv, results_csv):
    t = pd.read_csv(trace_csv, sep=";")
    r = pd.read_csv(results_csv, sep=";")
    r = r[r["ok"] == True]  # noqa: E712
    grid = np.sort(t["t"].unique())

    piv = (t.pivot_table(index=KEY + ["family"], columns="t", values="quality")
             .reindex(columns=grid).ffill(axis=1).fillna(0.0).reset_index())
    m = piv.melt(id_vars=KEY + ["family"], var_name="t", value_name="q")

    ends = []
    for fam in C:
        sub = r[KEY + [f"{fam}_quality", f"{fam}_wall"]].copy()
        sub.columns = KEY + ["fq", "wall"]
        sub["family"] = fam
        ends.append(sub)
    m = m.merge(pd.concat(ends), on=KEY + ["family"], how="left")
    m["q"] = np.where(m["t"] >= m["wall"], m["fq"], m["q"])
    return m.merge(t[KEY + ["method"]].drop_duplicates(), on=KEY)


def panel(ax, df, label):
    n = df.groupby("family").size().iloc[0] // df["t"].nunique()
    for fam in SHOWN:
        col = C[fam]
        s = df[df.family == fam].groupby("t")["q"]
        mean, sem = s.mean(), s.std(ddof=1) / np.sqrt(s.size())
        ax.plot(mean.index, mean.values, color=col, lw=1.4, label=fam)
        ax.fill_between(mean.index, mean - sem, mean + sem, color=col,
                        alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 300)
    ax.set_ylim(0, 1.02)
    ax.text(0.03, 0.95, f"{label}  (n={n})", transform=ax.transAxes,
            fontsize=7.5, fontname=FONT, va="top")
    style(ax, "time (s, log scale)", "mean interval quality")


def panel_delta(ax, m):
    """guided minus random, paired per query, one line per sampling method."""
    ax.axhline(0, color="black", lw=0.8, ls=":")
    for meth, col in M_COL.items():
        df = m[m.method == meth]
        p = (df.pivot_table(index=KEY + ["t"], columns="family", values="q")
               .dropna().reset_index())
        p["delta"] = p["guided"] - p["random"]
        g = p.groupby("t")["delta"].agg(d="mean", e="sem")
        ax.plot(g.index, g["d"], color=col, lw=1.4, label=meth)
        ax.fill_between(g.index, g["d"] - g["e"], g["d"] + g["e"],
                        color=col, alpha=0.18, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 300)
    style(ax, "time (s, log scale)", "guided $-$ random")
    ax.legend(loc="upper left", fontsize=5.5, framealpha=0.9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="results_tte/leaf_heuristics_v4")
    p.add_argument("--output_dir", default="results_tte/figures")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    m = build(os.path.join(args.run_dir, "leaf_heuristic_traces.csv"),
              os.path.join(args.run_dir, "leaf_heuristic_results.csv"))
    print(f"queries: {m.groupby(KEY).ngroups}  grid points: {m.t.nunique()}")

    # Two panels, not three. A paired-difference panel was useful while we
    # believed there was an early advantage the overlapping curves hid; with
    # the corrected baseline the difference is noise about zero, and drawing
    # it invites reading a trend into bands that straddle zero throughout.
    fig, axes = plt.subplots(1, 2, figsize=(4.79, 2.0))
    panel(axes[0], m[m.method == "worlds"], "worlds")
    panel(axes[1], m[m.method == "programs"], "programs")
    axes[0].legend(loc="lower right", fontsize=6, framealpha=0.9)
    axes[1].set_ylabel("")
    fig.tight_layout(pad=0.4)

    for ext in ("pdf", "png"):
        out = os.path.join(args.output_dir, f"convergence.{ext}")
        fig.savefig(out, format=ext, bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
