import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def compute_metric(approximate, exact):
    width_approximate = approximate[1] - approximate[0]
    width_exact = exact[1] - exact[0]
    remainder_approximate = 1 - width_approximate
    remainder_exact = 1 - width_exact

    if remainder_exact == 0:
        return 0.0
    return remainder_approximate / remainder_exact


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def generate_summary_plots(exact_path, worlds_path, progs_path, output_dir, timeout):
    exact_data = load_json(exact_path)
    worlds_data = load_json(worlds_path)
    progs_data = load_json(progs_path)

    literals = sorted(list(exact_data["status"].keys()))

    metrics_quality_w = []
    metrics_quality_p = []

    times_w = []
    times_p = []

    global_calls_w = worlds_data["data"]["delp_calls"]
    global_calls_p = progs_data["data"]["delp_calls"]

    calls_w_lit = [global_calls_w for _ in literals]
    calls_p_lit = [global_calls_p for _ in literals]

    for lit in literals:
        # Exact
        ex_s = exact_data["status"][lit]
        ex_l, ex_u = (
            ex_s.get("l", ex_s.get("pyes", 0)),
            ex_s.get("u", 1 - ex_s.get("pno", 0)),
        )

        # Worlds
        w_s = worlds_data["status"].get(lit, {})
        w_l, w_u = w_s.get("l", w_s.get("pyes", 0)), w_s.get("u", 1 - w_s.get("pno", 0))
        metrics_quality_w.append(compute_metric([w_l, w_u], [ex_l, ex_u]))
        times_w.append(w_s.get("time", 0.0))
        calls_w_lit.append(w_s.get("delp_calls", 0))

        # Programs
        p_s = progs_data["status"].get(lit, {})
        p_l, p_u = p_s.get("l", p_s.get("pyes", 0)), p_s.get("u", 1 - p_s.get("pno", 0))
        metrics_quality_p.append(compute_metric([p_l, p_u], [ex_l, ex_u]))
        times_p.append(p_s.get("time", 0.0))
        calls_p_lit.append(p_s.get("delp_calls", 0))

    # Apply publication style
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    x = np.arange(len(literals))
    width = 0.35

    # ---------------------------------------------------------
    # 1. PER LITERAL PLOTS
    # ---------------------------------------------------------
    # 1A. Quality Metric Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, metrics_quality_w, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, metrics_quality_p, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Quality Metric (Closer to 1 is better)")
    ax.set_title(f"Approximation Quality per Literal ({timeout}s Timeout)")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_quality_{timeout}s.png"), dpi=300)
    plt.close()

    # 1B. Time Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, times_w, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, times_p, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Solver Time (s)")
    ax.set_title(f"DeLP Solver Time per Literal ({timeout}s Timeout)")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"summary_time_{timeout}s.png"), dpi=300)
    plt.close()

    # 1C. Solver Calls Plot per Literal
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, calls_w_lit, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, calls_p_lit, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Evaluations (Count)")
    ax.set_title(f"Solver Evaluations per Literal ({timeout}s Timeout)")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"summary_calls_per_literal_{timeout}s.png"), dpi=300
    )
    plt.close()

    # ---------------------------------------------------------
    # 2. AVERAGES (GLOBAL) PLOTS
    # ---------------------------------------------------------

    # Calculate means and std devs
    avg_quality_w = np.mean(metrics_quality_w)
    std_quality_w = np.std(metrics_quality_w)
    avg_quality_p = np.mean(metrics_quality_p)
    std_quality_p = np.std(metrics_quality_p)

    avg_time_w = np.mean(times_w)
    std_time_w = np.std(times_w)
    avg_time_p = np.mean(times_p)
    std_time_p = np.std(times_p)

    avg_calls_w = np.mean(calls_w_lit)
    std_calls_w = np.std(calls_w_lit)
    avg_calls_p = np.mean(calls_p_lit)
    std_calls_p = np.std(calls_p_lit)

    approaches = ["Worlds", "Programs"]
    x_pos = np.arange(len(approaches))

    # 2A. Average Quality
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x_pos,
        [avg_quality_w, avg_quality_p],
        yerr=[std_quality_w, std_quality_p],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
        error_kw={"capthick": 2, "elinewidth": 2},
        alpha=0.9,
    )
    ax.set_ylabel("Average Quality Metric")
    ax.set_title(f"Average Quality ({timeout}s Timeout)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)

    # Data labels
    for bar in bars:
        height = bar.get_height()
        y_pos = height / 2 if height > 0.1 else height + 0.05
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"average_quality_{timeout}s.png"), dpi=300)
    plt.close()

    # 2B. Average Time
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x_pos,
        [avg_time_w, avg_time_p],
        yerr=[std_time_w, std_time_p],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
        error_kw={"capthick": 2, "elinewidth": 2},
        alpha=0.9,
    )
    ax.set_ylabel("Average Solver Time (s)")
    ax.set_title(f"Average Solver Time ({timeout}s Timeout)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)

    for bar in bars:
        height = bar.get_height()
        y_pos = height / 2 if height > 0.1 else height + 0.05
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"average_time_{timeout}s.png"), dpi=300)
    plt.close()

    # 2C. Total Global Calls (Evaluations)
    global_calls_w = worlds_data["data"]["delp_calls"]
    global_calls_p = progs_data["data"]["delp_calls"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x_pos,
        [global_calls_w, global_calls_p],
        color=["#1f77b4", "#ff7f0e"],
        capsize=5,
        alpha=0.9,
    )
    ax.set_ylabel("Total Global Evaluations")
    ax.set_title(f"Total Solver Evaluations ({timeout}s Timeout)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"average_calls_{timeout}s.png"), dpi=300)
    plt.close()

    print(f"Summary and average plots generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot summary metrics for all literals."
    )
    parser.add_argument("--exact", required=True)
    parser.add_argument("--worlds", required=True)
    parser.add_argument("--progs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    generate_summary_plots(args.exact, args.worlds, args.progs, args.out, args.timeout)
