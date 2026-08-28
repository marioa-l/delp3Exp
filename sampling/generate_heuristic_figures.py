"""
Figures for the heuristic experiment: does adding a heuristic to the method
the abstract tree recommends beat running that method plainly?

The comparison is paired (both modes run on the same query), so every panel
works on per-query deltas rather than independent group means.

Two things have to be read together, which is why they share a figure:
the heuristic can tighten the interval it reports while simultaneously
reaching the exact interval far less often. A panel showing only quality
would overstate the result.

Reads:
    results_tte/heuristics/heuristic_abstract_results.csv

Writes to --output_dir (default results_tte/figures/):
    heuristic_overview.pdf      4-panel summary (the figure to show)
    heuristic_per_leaf.pdf      per-leaf delta distribution, standalone
    heuristic_exact_tradeoff.pdf  exactness cost, standalone

Usage:
    python sampling/generate_heuristic_figures.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # scipy is optional; p-values are simply omitted
    wilcoxon = None

FONT = "DejaVu Sans"
LABEL_SIZE, TITLE_SIZE = 12, 14
COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"
COLOR_BASE     = "#8C8C8C"
COLOR_WIN      = "#27AE60"
COLOR_LOSS     = "#C0392B"


def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontname=FONT)
    ax.set_title(title, fontsize=TITLE_SIZE, fontname=FONT, pad=8)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=10)
    ax.minorticks_on()


def method_color(m):
    return COLOR_WORLDS if m == "worlds" else COLOR_PROGRAMS


def leaf_table(df):
    """Per-leaf aggregates, ordered by median delta (worst first)."""
    rows = []
    for (leaf, method), g in df.groupby(["leaf", "method"]):
        x = g["quality_delta"].values
        p = np.nan
        if wilcoxon is not None and len(x) >= 5 and not np.allclose(x, 0):
            try:
                p = wilcoxon(x).pvalue
            except ValueError:
                p = np.nan
        rows.append({
            "leaf": leaf.replace("abs_leaf_", "leaf "), "method": method,
            "n": len(g),
            "mean": x.mean(), "median": np.median(x),
            "sem": x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
            "win_rate": (x > 0).mean(),
            "base_q": g["base_quality"].mean(),
            "heur_q": g["heur_quality"].mean(),
            "base_q_sem": g["base_quality"].std(ddof=1) / np.sqrt(len(g)),
            "heur_q_sem": g["heur_quality"].std(ddof=1) / np.sqrt(len(g)),
            "base_exact": int(g["base_time_to_exact"].notna().sum()),
            "heur_exact": int(g["heur_time_to_exact"].notna().sum()),
            "p": p,
        })
    return pd.DataFrame(rows).sort_values("median").reset_index(drop=True)


# ── Panels ──────────────────────────────────────────────────────────────────
def panel_delta(ax, df, tab):
    """Per-leaf distribution of the paired per-query delta."""
    order = tab["leaf"].tolist()
    data, colors = [], []
    for leaf in order:
        m = tab.loc[tab.leaf == leaf, "method"].iloc[0]
        key = "abs_leaf_" + leaf.split()[1]
        data.append(df.loc[df.leaf == key, "quality_delta"].values)
        colors.append(method_color(m))

    pos = np.arange(len(order))
    bp = ax.boxplot(data, positions=pos, vert=False, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=1.6))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    for i, arr in enumerate(data):
        j = np.random.uniform(-0.16, 0.16, size=len(arr))
        ax.scatter(arr, pos[i] + j, s=5, color=colors[i], alpha=0.35,
                   edgecolors="none", zorder=3)

    ax.axvline(0, color="black", lw=1.0, ls=":")
    labels = [f"{r.leaf}  ({r.method[:4]}, n={r.n})"
              for r in tab.itertuples()]
    ax.set_yticks(pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontname=FONT)
    ax.set_xlim(-1.05, 1.05)
    for i, r in enumerate(tab.itertuples()):
        star = ""
        if not np.isnan(r.p):
            star = "***" if r.p < 1e-3 else "**" if r.p < 1e-2 \
                   else "*" if r.p < 0.05 else "n.s."
        ax.text(1.02, pos[i], f"{r.win_rate:.0%} wins  {star}",
                va="center", fontsize=8.5, fontname=FONT, color="#333333")
    style_ax(ax, xlabel=r"quality delta  =  heuristic $-$ classic",
             title="A. Per-query gain, by tree leaf (positive = heuristic better)")


def panel_quality(ax, df, tab):
    """Mean interval quality, classic vs heuristic, per leaf."""
    pos = np.arange(len(tab))
    w = 0.38
    ax.barh(pos - w / 2, tab["base_q"], w, xerr=tab["base_q_sem"],
            color=COLOR_BASE, edgecolor="white", label="classic (uniform)",
            error_kw=dict(lw=0.9, ecolor="#444444"))
    ax.barh(pos + w / 2, tab["heur_q"], w, xerr=tab["heur_q_sem"],
            color=[method_color(m) for m in tab["method"]],
            edgecolor="white", label="heuristic",
            error_kw=dict(lw=0.9, ecolor="#444444"))
    ax.set_yticks(pos)
    ax.set_yticklabels(tab["leaf"], fontsize=9.5, fontname=FONT)
    ax.set_xlim(0, 1.19)
    for i, r in enumerate(tab.itertuples()):
        ax.text(r.base_q + 0.015, i - w / 2, f"{r.base_q:.2f}",
                va="center", fontsize=8, fontname=FONT)
        ax.text(r.heur_q + 0.015, i + w / 2, f"{r.heur_q:.2f}",
                va="center", fontsize=8, fontname=FONT)
    style_ax(ax, xlabel="Mean interval quality  (1.0 = exact interval)",
             title="B. Interval quality at the time cap")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def panel_exact(ax, tab):
    """How often each mode actually reached the exact interval."""
    pos = np.arange(len(tab))
    w = 0.38
    base_pct = tab["base_exact"] / tab["n"]
    heur_pct = tab["heur_exact"] / tab["n"]
    ax.barh(pos - w / 2, base_pct, w, color=COLOR_BASE, edgecolor="white",
            label="classic (uniform)")
    ax.barh(pos + w / 2, heur_pct, w,
            color=[method_color(m) for m in tab["method"]],
            edgecolor="white", label="heuristic")
    for i, r in enumerate(tab.itertuples()):
        ax.text(base_pct[i] + 0.015, i - w / 2,
                f"{r.base_exact}/{r.n}", va="center", fontsize=8,
                fontname=FONT)
        ax.text(heur_pct[i] + 0.015, i + w / 2,
                f"{r.heur_exact}/{r.n}", va="center", fontsize=8,
                fontname=FONT)
    ax.set_yticks(pos)
    ax.set_yticklabels(tab["leaf"], fontsize=9.5, fontname=FONT)
    ax.set_xlim(0, 1.25)
    style_ax(ax, xlabel="Fraction of queries reaching the exact interval",
             title="C. The cost: exactness is reached far less often")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def panel_scatter(ax, df):
    """Per-query classic vs heuristic quality."""
    for m in ("worlds", "programs"):
        s = df[df.method == m]
        ax.scatter(s["base_quality"], s["heur_quality"], s=14, alpha=0.45,
                   color=method_color(m), edgecolors="none",
                   label=f"{m} (n={len(s)})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="no change")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    box = dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5)
    ax.text(0.05, 0.95, "heuristic better", fontsize=9, fontname=FONT,
            color=COLOR_WIN, va="top", bbox=box)
    ax.text(0.97, 0.12, "classic better", fontsize=9, fontname=FONT,
            color=COLOR_LOSS, ha="right", bbox=box)
    style_ax(ax, xlabel="Classic (uniform) quality",
             ylabel="Heuristic quality",
             title="D. Per-query comparison")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)


# ── Figures ─────────────────────────────────────────────────────────────────
def fig_overview(df, tab, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11))
    panel_delta(axes[0, 0], df, tab)
    panel_quality(axes[0, 1], df, tab)
    panel_exact(axes[1, 0], tab)
    panel_scatter(axes[1, 1], df)

    n = len(df)
    med = df["quality_delta"].median()
    win = (df["quality_delta"] > 0).mean()
    be = int(df["base_time_to_exact"].notna().sum())
    he = int(df["heur_time_to_exact"].notna().sum())
    fig.suptitle(
        "Heuristic vs classic sampling, within the method the abstract tree "
        f"recommends   (n={n} queries, 300 s cap, paired)\n"
        f"Overall: median delta {med:+.3f}, heuristic wins {win:.1%}; "
        f"reached exact {be} -> {he}",
        fontsize=14.5, fontname=FONT, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_per_leaf(df, tab, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 6))
    panel_delta(ax, df, tab)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_exact(df, tab, out_path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    panel_exact(ax, tab)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_quality_and_time(df, out_path):
    """
    Quality and time, heuristic vs base, plus the regime where the heuristic
    actually pays off.

    Averages alone mislead here: the heuristic is a large win on the queries
    where plain sampling is failing and a small loss everywhere else, so the
    mean hides both effects. The split by base quality is the real result.
    """
    d = df.copy()
    d["base_width"] = d["base_u"] - d["base_l"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # A. quality by regime
    ax = axes[0]
    bins = [(-0.01, 0.2, "plain method\nfailing\n(q<=0.2)"),
            (0.2, 0.8, "plain method\nstruggling\n(0.2-0.8)"),
            (0.8, 1.01, "plain method\nalready fine\n(q>0.8)")]
    xs = np.arange(len(bins))
    w = 0.38
    bq = [d[(d.base_quality > lo) & (d.base_quality <= hi)].base_quality.mean()
          for lo, hi, _ in bins]
    hq = [d[(d.base_quality > lo) & (d.base_quality <= hi)].heur_quality.mean()
          for lo, hi, _ in bins]
    ns = [int(((d.base_quality > lo) & (d.base_quality <= hi)).sum())
          for lo, hi, _ in bins]
    ax.bar(xs - w / 2, bq, w, color=COLOR_BASE, edgecolor="white",
           label="plain (uniform)")
    ax.bar(xs + w / 2, hq, w, color=COLOR_WIN, edgecolor="white",
           label="with heuristic")
    for i, (b, h, n) in enumerate(zip(bq, hq, ns)):
        ax.text(i - w / 2, b + 0.015, f"{b:.2f}", ha="center", fontsize=9.5,
                fontname=FONT)
        ax.text(i + w / 2, h + 0.015, f"{h:.2f}", ha="center", fontsize=9.5,
                fontname=FONT)
        ax.text(i, -0.055, f"n={n}", ha="center", va="top", fontsize=9,
                color="dimgray", fontname=FONT)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[2] for b in bins], fontsize=9, fontname=FONT)
    ax.set_ylim(-0.12, 1.12)
    style_ax(ax, ylabel="Mean interval quality (1.0 = exact)",
             title="A. Quality: the heuristic rescues\nthe cases plain sampling fails")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # B. time to exact
    ax = axes[1]
    both = d.dropna(subset=["base_time_to_exact", "heur_time_to_exact"])
    ax.scatter(both.base_time_to_exact, both.heur_time_to_exact, s=16,
               alpha=0.5, color=COLOR_WORLDS, edgecolors="none",
               label=f"both reached exact (n={len(both)})")
    lim = [1, max(both.base_time_to_exact.max(),
                  both.heur_time_to_exact.max()) * 1.15]
    ax.plot(lim, lim, "k--", lw=0.9, label="equal time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    sp = (both.base_time_to_exact / both.heur_time_to_exact).median()
    ax.text(0.04, 0.96, f"heuristic slower\nfor {(sp<1)*100:.0f}% of these\n"
                        f"median speedup {sp:.2f}x",
            transform=ax.transAxes, va="top", fontsize=9.5, fontname=FONT,
            color=COLOR_LOSS,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))
    nb = int(d.base_time_to_exact.notna().sum())
    nh = int(d.heur_time_to_exact.notna().sum())
    style_ax(ax, xlabel="Plain: time to exact (s, log)",
             ylabel="Heuristic: time to exact (s, log)",
             title=f"B. Time: no saving\nreached exact {nb} -> {nh} of {len(d)}")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # C. switching policy payoff
    ax = axes[2]
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
    vals, labels = [], []
    for T in thresholds:
        applied = d.base_width > T if T > 0 else pd.Series(True, index=d.index)
        q = np.where(applied, d.heur_quality, d.base_quality)
        vals.append(q.mean())
        labels.append("always" if T == 0 else
                      ("never" if T > 1 else f">{T:g}"))
    base_only = d.base_quality.mean()
    colors = [COLOR_WIN if v > base_only else COLOR_LOSS for v in vals]
    ax.bar(np.arange(len(vals)), vals, color=colors, edgecolor="white",
           width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=9.5,
                fontname=FONT)
    ax.axhline(base_only, color="black", lw=1.1, ls="--",
               label=f"plain only ({base_only:.3f})")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5, fontname=FONT)
    ax.set_ylim(0.78, 1.005)
    style_ax(ax, xlabel="apply the heuristic when the plain interval is wider than",
             ylabel="Mean final quality",
             title="C. Switching beats both\npure strategies")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle(
        "Heuristic vs plain sampling: quality, time, and when it is worth "
        f"switching   (n={len(d)} queries, 300 s cap, paired runs)",
        fontsize=14, fontname=FONT, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv",
                        default="results_tte/heuristics/"
                                "heuristic_abstract_results.csv")
    parser.add_argument("--output_dir", default="results_tte/figures")
    parser.add_argument("--min_n", type=int, default=5,
                        help="Drop leaves with fewer queries than this")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT
    np.random.seed(0)  # jitter reproducibility

    df = pd.read_csv(args.input_csv, sep=";")
    df = df[df["ok"] == True].dropna(subset=["quality_delta"])  # noqa: E712
    counts = df.groupby("leaf").size()
    keep = counts[counts >= args.min_n].index
    dropped = sorted(set(counts.index) - set(keep))
    if dropped:
        print(f"  Dropped leaves with n < {args.min_n}: {dropped}")
    df = df[df.leaf.isin(keep)]
    print(f"Queries: {len(df)}  |  leaves: {df.leaf.nunique()}")

    tab = leaf_table(df)
    fig_overview(df, tab, os.path.join(args.output_dir,
                                       "heuristic_overview.pdf"))
    fig_per_leaf(df, tab, os.path.join(args.output_dir,
                                       "heuristic_per_leaf.pdf"))
    fig_exact(df, tab, os.path.join(args.output_dir,
                                    "heuristic_exact_tradeoff.pdf"))
    fig_quality_and_time(df, os.path.join(args.output_dir,
                                          "heuristic_quality_and_time.pdf"))

    out = os.path.join(args.output_dir, "heuristic_leaf_table.csv")
    tab.to_csv(out, sep=";", index=False)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
