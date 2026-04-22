"""
Real-data correlation figures for a single DeLP3E model.

Given exact bounds, Worlds sampling results, and Programs sampling results,
generates three publication-ready PDF figures:

  1. correlation_per_approach.pdf
     Two side-by-side correlation heatmaps (one per approach) of per-literal
     metrics: [exact_l, exact_u, exact_width, approx_l, approx_u, approx_width,
     approx_time]. Shows how well each approach tracks the exact structure.

  2. cross_approach_correlation.pdf
     Rectangular heatmap: Worlds metrics (rows) × Programs metrics (cols).
     Shows whether both approaches agree on which literals are "hard" or "easy".

  3. quality_comparison.pdf
     Grouped bar chart: worlds_quality vs progs_quality per literal,
     colour-coded by winner (worlds / programs / tie).

Usage:
    cd clean_workspace
    python sampling/plot_real_correlations.py \
        --worlds  data/results/0model/120s/0model_s_w_timeNEW.json \
        --progs   data/results/0model/120s/0model_s_p_timeNEW.json \
        --exact   data/test_models/exact/0model_e_wNEW.json \
        --model   data/test_models/isolated/0model.json \
        --output_dir data/results/real_correlations/
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Aesthetics ──────────────────────────────────────────────────────────────
FONT = "DejaVu Sans"
DPI = 300
C_WORLDS = "#1f77b4"
C_PROGS = "#ff7f0e"
C_TIE = "#2ca02c"
C_EXACT = "#444444"
plt.rcParams["font.family"] = FONT


# ── I/O helpers ─────────────────────────────────────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)


def quality_metric(approx_l, approx_u, exact_l, exact_u):
    w_a = approx_u - approx_l
    w_e = exact_u - exact_l
    r_e = 1 - w_e
    if r_e == 0:
        return 0.0
    return (1 - w_a) / r_e


# ── Build per-literal DataFrame ─────────────────────────────────────────────
def build_df(worlds_data, progs_data, exact_data):
    exact_lits = set(exact_data["status"].keys())
    worlds_lits = set(worlds_data["status"].keys())
    progs_lits = set(progs_data["status"].keys())
    common = exact_lits & worlds_lits & progs_lits

    rows = []
    for lit in sorted(common):
        ex = exact_data["status"][lit]
        w = worlds_data["status"][lit]
        p = progs_data["status"][lit]

        ex_l = ex.get("l", ex.get("pyes", 0))
        ex_u = ex.get("u", 1 - ex.get("pno", 0))

        w_l = w.get("l", w.get("pyes", 0))
        w_u = w.get("u", 1 - w.get("pno", 0))

        p_l = p.get("l", p.get("pyes", 0))
        p_u = p.get("u", 1 - p.get("pno", 0))

        q_w = quality_metric(w_l, w_u, ex_l, ex_u)
        q_p = quality_metric(p_l, p_u, ex_l, ex_u)

        rows.append(
            {
                "literal": lit,
                # exact
                "exact_l": ex_l,
                "exact_u": ex_u,
                "exact_width": ex_u - ex_l,
                "exact_pyes": ex.get("pyes", 0),
                "exact_pno": ex.get("pno", 0),
                "exact_pundecided": ex.get("pundecided", 0),
                # worlds
                "worlds_l": w_l,
                "worlds_u": w_u,
                "worlds_width": w_u - w_l,
                "worlds_time": w.get("time", 0),
                "worlds_pyes": w.get("pyes", 0),
                "worlds_pno": w.get("pno", 0),
                "worlds_pundecided": w.get("pundecided", 0),
                "worlds_quality": q_w,
                # programs
                "progs_l": p_l,
                "progs_u": p_u,
                "progs_width": p_u - p_l,
                "progs_time": p.get("time", 0),
                "progs_pyes": p.get("pyes", 0),
                "progs_pno": p.get("pno", 0),
                "progs_pundecided": p.get("pundecided", 0),
                "progs_quality": q_p,
                # derived
                "quality_diff": q_w - q_p,
                "winner": (
                    "worlds"
                    if q_w > q_p
                    else ("programs" if q_p > q_w else "tie")
                ),
            }
        )

    return pd.DataFrame(rows)


# ── Figure 1: per-approach correlation heatmaps ──────────────────────────────
def _heatmap(ax, corr, title, annot_size=9):
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(corr.index, fontsize=10)
    ax.tick_params(which="both", length=0)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            c = "white" if abs(v) > 0.65 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=annot_size, color=c, fontweight="bold")
    ax.set_title(title, fontsize=13, pad=8, fontweight="bold")
    return im


def fig_per_approach(df, output_path, model_info):
    cols_w = ["exact_l", "exact_u", "exact_width",
              "worlds_l", "worlds_u", "worlds_width", "worlds_time"]
    cols_p = ["exact_l", "exact_u", "exact_width",
              "progs_l", "progs_u", "progs_width", "progs_time"]

    nice_w = ["exact $l$", "exact $u$", "exact width",
              "worlds $l$", "worlds $u$", "worlds width", "worlds time"]
    nice_p = ["exact $l$", "exact $u$", "exact width",
              "progs $l$", "progs $u$", "progs width", "progs time"]

    corr_w = df[cols_w].rename(columns=dict(zip(cols_w, nice_w))).corr()
    corr_p = df[cols_p].rename(columns=dict(zip(cols_p, nice_p))).corr()

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    im_w = _heatmap(axes[0], corr_w, "Worlds Sampling")
    im_p = _heatmap(axes[1], corr_p, "Programs Sampling")

    cbar = fig.colorbar(im_p, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Pearson $r$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Per-Literal Metric Correlations — {model_info}  ($n = {len(df)}$ literals)",
        fontsize=14,
    )
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 2: cross-approach correlation ─────────────────────────────────────
def fig_cross_approach(df, output_path, model_info):
    w_vars = ["worlds_l", "worlds_u", "worlds_width", "worlds_time", "worlds_quality"]
    p_vars = ["progs_l", "progs_u", "progs_width", "progs_time", "progs_quality"]

    w_labels = ["worlds $l$", "worlds $u$", "worlds width", "worlds time", "worlds quality"]
    p_labels = ["progs $l$", "progs $u$", "progs width", "progs time", "progs quality"]

    corr_mat = np.zeros((len(w_vars), len(p_vars)))
    for i, wv in enumerate(w_vars):
        for j, pv in enumerate(p_vars):
            if df[wv].std() == 0 or df[pv].std() == 0:
                corr_mat[i, j] = np.nan
            else:
                corr_mat[i, j] = df[wv].corr(df[pv])

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    masked = np.ma.masked_invalid(corr_mat)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(p_vars)))
    ax.set_yticks(range(len(w_vars)))
    ax.set_xticklabels(p_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticklabels(w_labels, fontsize=11)
    ax.tick_params(which="both", length=0)

    for i in range(len(w_vars)):
        for j in range(len(p_vars)):
            v = corr_mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                c = "white" if abs(v) > 0.65 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=10, color=c, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Pearson $r$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"Cross-Approach Correlation: Worlds × Programs\n{model_info}  ($n={len(df)}$ literals)",
        fontsize=13,
        pad=10,
    )
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 3: quality comparison bar chart ───────────────────────────────────
def fig_quality_comparison(df, output_path, model_info):
    df_sorted = df.sort_values("quality_diff")
    lits = df_sorted["literal"].tolist()
    n = len(lits)
    x = np.arange(n)
    w = 0.35

    winner_colors_w = [
        C_WORLDS if row.winner == "worlds" else (C_TIE if row.winner == "tie" else "#bbbbbb")
        for _, row in df_sorted.iterrows()
    ]
    winner_colors_p = [
        C_PROGS if row.winner == "programs" else (C_TIE if row.winner == "tie" else "#dddddd")
        for _, row in df_sorted.iterrows()
    ]

    plt.style.use("seaborn-v0_8-paper")
    fig, axes = plt.subplots(2, 1, figsize=(max(12, n * 0.55), 10),
                             gridspec_kw={"height_ratios": [3, 1]},
                             constrained_layout=True)

    ax = axes[0]
    bars_w = ax.bar(x - w / 2, df_sorted["worlds_quality"], w,
                    color=winner_colors_w, label="Worlds", edgecolor="white", linewidth=0.5)
    bars_p = ax.bar(x + w / 2, df_sorted["progs_quality"], w,
                    color=winner_colors_p, label="Programs", edgecolor="white", linewidth=0.5)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Perfect (q=1)")

    # Exact width reference line on secondary y (optional)
    ax.set_xticks(x)
    ax.set_xticklabels(lits, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Quality Metric", fontsize=13)
    ax.set_ylim(0, 1.15)
    ax.tick_params(which="both", direction="in")
    ax.minorticks_on()

    # Value labels on bars (only when bar is tall enough)
    for bar in list(bars_w) + list(bars_p):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7,
                    rotation=70, fontfamily=FONT)

    # Legend with winner colours
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_WORLDS, label="Worlds wins"),
        Patch(facecolor=C_PROGS, label="Programs wins"),
        Patch(facecolor=C_TIE, label="Tie"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=0.8, label="Perfect (q=1)"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper left", framealpha=0.9)
    ax.set_title(
        f"Quality Metric per Literal — {model_info}", fontsize=14, pad=8
    )

    # Bottom panel: quality difference
    ax2 = axes[1]
    colors_diff = [C_WORLDS if v > 0 else (C_TIE if v == 0 else C_PROGS)
                   for v in df_sorted["quality_diff"]]
    ax2.bar(x, df_sorted["quality_diff"], color=colors_diff, edgecolor="white", linewidth=0.5)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(lits, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Δ Quality\n(Worlds − Progs)", fontsize=11)
    ax2.tick_params(which="both", direction="in")
    ax2.minorticks_on()

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 4: program-feature × approach-metric rectangular heatmap ──────────
def fig_program_metrics_correlation(df, output_path, model_info):
    """
    Single rectangular heatmap:
      rows  → literal/program features derived from the exact computation
      cols  → 3 performance metrics × 2 approaches  (6 columns total)

    'Solver queries' is represented as P(yes) — the fraction of DeLP calls
    that returned YES for each literal. Total query count is constant within
    each run (all literals share the same time budget), so P(yes) is the
    only query-level metric that varies meaningfully per literal.
    """
    # ── Row features (from exact computation — ground truth per literal) ──
    row_vars = [
        "exact_l",
        "exact_u",
        "exact_width",
        "exact_pyes",
        "exact_pno",
        "exact_pundecided",
    ]
    row_labels = [
        "exact $l$",
        "exact $u$",
        "exact width",
        "$P_{exact}$(yes)",
        "$P_{exact}$(no)",
        "$P_{exact}$(undecided)",
    ]

    # ── Column metrics (3 per approach) ──
    col_vars = [
        "worlds_quality",
        "worlds_time",
        "worlds_pyes",
        "progs_quality",
        "progs_time",
        "progs_pyes",
    ]
    col_labels = [
        "quality",
        "solver\ntime (s)",
        "P(yes)\n≈ n queries",
        "quality",
        "solver\ntime (s)",
        "P(yes)\n≈ n queries",
    ]
    n_rows, n_cols = len(row_vars), len(col_vars)

    # ── Build matrix ──
    mat = np.full((n_rows, n_cols), np.nan)
    for i, rv in enumerate(row_vars):
        for j, cv in enumerate(col_vars):
            if rv not in df or cv not in df:
                continue
            s1, s2 = df[rv], df[cv]
            if s1.std() == 0 or s2.std() == 0:
                mat[i, j] = 0.0
            else:
                mat[i, j] = s1.corr(s2)

    # ── Plot ──
    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)
    ax.tick_params(which="both", length=0)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
            else:
                color = "white" if abs(v) > 0.62 else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=10, color=color, fontweight="bold")

    # Vertical separator between Worlds block and Programs block
    ax.axvline(2.5, color="black", linewidth=2.0, linestyle="-")

    # Approach headers above the columns
    header_y = -0.9  # in axes coords
    for col_center, label, color in [
        (1.0, "Worlds Sampling", C_WORLDS),
        (4.0, "Programs Sampling", C_PROGS),
    ]:
        ax.text(
            col_center, -1.15,
            label,
            ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=color,
            transform=ax.transData,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson $r$", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        f"Program Features × Approach Metrics — {model_info}\n"
        f"($n = {len(df)}$ literals  |  "
        r"$\dagger$ P(yes) = fraction of YES solver calls per literal)",
        fontsize=12,
        pad=14,
    )

    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def model_info_str(model_path):
    try:
        d = load(model_path)
        em = d["em_var"]
        af = d["af"]
        n_rules = len(af)
        trivial = {"True", "", "not True"}
        n_annots = sum(1 for r in af if r[1] not in trivial)
        name = os.path.basename(model_path).replace(".json", "")
        return (
            f"{name} — "
            f"$n_{{EM}}={em}$, $n_{{annots}}={n_annots}$, "
            f"$n_{{annots}}-n_{{EM}}={n_annots - em:+d}$, "
            f"$n_{{rules}}={n_rules}$"
        )
    except Exception:
        return os.path.basename(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate real-data correlation figures for a single model."
    )
    parser.add_argument("--worlds", required=True, help="Worlds sampling result JSON")
    parser.add_argument("--progs", required=True, help="Programs sampling result JSON")
    parser.add_argument("--exact", required=True, help="Exact bounds JSON")
    parser.add_argument("--model", required=True, help="DeLP3E model JSON")
    parser.add_argument("--output_dir", required=True, help="Directory for output PDFs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    worlds_data = load(args.worlds)
    progs_data = load(args.progs)
    exact_data = load(args.exact)
    info = model_info_str(args.model)

    df = build_df(worlds_data, progs_data, exact_data)
    print(f"Loaded {len(df)} common literals.")
    print(f"Winner distribution:\n{df['winner'].value_counts().to_string()}\n")
    print(f"Quality diff stats:")
    print(f"  mean : {df['quality_diff'].mean():+.4f}")
    print(f"  std  : {df['quality_diff'].std():.4f}")

    fig_per_approach(df, os.path.join(args.output_dir, "correlation_per_approach.pdf"), info)
    fig_cross_approach(df, os.path.join(args.output_dir, "cross_approach_correlation.pdf"), info)
    fig_quality_comparison(df, os.path.join(args.output_dir, "quality_comparison.pdf"), info)
    fig_program_metrics_correlation(
        df, os.path.join(args.output_dir, "program_metrics_correlation.pdf"), info
    )
