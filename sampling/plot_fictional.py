import os

import matplotlib.pyplot as plt
import numpy as np


def generate_fictional_plots(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    literals = [f"lit_{i}" for i in range(1, 16)]
    n_lits = len(literals)

    # Fictional Data where Worlds is significantly better than Programs
    # 1. Quality Metric (Closer to 1 is better)
    quality_w = np.random.normal(loc=0.92, scale=0.03, size=n_lits)
    quality_w = np.clip(quality_w, 0, 1.0)
    quality_p = np.random.normal(loc=0.55, scale=0.15, size=n_lits)
    quality_p = np.clip(quality_p, 0, 1.0)

    # 2. Time (Lower is better)
    time_w = np.random.normal(loc=0.8, scale=0.2, size=n_lits)
    time_p = np.random.normal(loc=3.5, scale=0.8, size=n_lits)

    # 3. Solver Calls per literal (Lower is better)
    calls_w = np.random.normal(loc=120, scale=15, size=n_lits).astype(int)
    calls_p = np.random.normal(loc=450, scale=40, size=n_lits).astype(int)

    # Style
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

    x = np.arange(n_lits)
    width = 0.35

    # --- Plot 1: Quality per Literal ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, quality_w, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, quality_p, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Quality Metric (Closer to 1 is better)")
    ax.set_title("Approximation Quality per Literal")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fictional_quality_per_literal.png"), dpi=300)
    plt.close()

    # --- Plot 2: Average Quality ---
    fig, ax = plt.subplots(figsize=(6, 5))
    approaches = ["Worlds", "Programs"]
    x_pos = np.arange(len(approaches))
    bars = ax.bar(
        x_pos,
        [np.mean(quality_w), np.mean(quality_p)],
        yerr=[np.std(quality_w), np.std(quality_p)],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
        error_kw={"capthick": 2, "elinewidth": 2},
        alpha=0.9,
    )

    ax.set_ylabel("Average Quality Metric")
    ax.set_title("Average Quality")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)
    ax.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5)

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
    plt.savefig(os.path.join(output_dir, "fictional_average_quality.png"), dpi=300)
    plt.close()

    # --- Plot 3: Time per Literal ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, time_w, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, time_p, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Solver Time (s)")
    ax.set_title("DeLP Solver Time per Literal")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fictional_time_per_literal.png"), dpi=300)
    plt.close()

    # --- Plot 4: Average Time ---
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x_pos,
        [np.mean(time_w), np.mean(time_p)],
        yerr=[np.std(time_w), np.std(time_p)],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
        error_kw={"capthick": 2, "elinewidth": 2},
        alpha=0.9,
    )

    ax.set_ylabel("Average Solver Time (s)")
    ax.set_title("Average Solver Time")
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
    plt.savefig(os.path.join(output_dir, "fictional_average_time.png"), dpi=300)
    plt.close()

    # --- Plot 5: Calls per Literal ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, calls_w, width, label="Worlds", color="#1f77b4")
    ax.bar(x + width / 2, calls_p, width, label="Programs", color="#ff7f0e")

    ax.set_ylabel("Evaluations (Count)")
    ax.set_title("Solver Evaluations per Literal")
    ax.set_xticks(x)
    ax.set_xticklabels(literals, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fictional_calls_per_literal.png"), dpi=300)
    plt.close()

    # --- Plot 6: Average Calls ---
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        x_pos,
        [np.mean(calls_w), np.mean(calls_p)],
        yerr=[np.std(calls_w), np.std(calls_p)],
        color=["#1f77b4", "#ff7f0e"],
        capsize=8,
        error_kw={"capthick": 2, "elinewidth": 2},
        alpha=0.9,
    )

    ax.set_ylabel("Average Evaluations")
    ax.set_title("Average Solver Evaluations")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(approaches)

    for bar in bars:
        height = bar.get_height()
        y_pos = height / 2 if height > 0.1 else height + 0.05
        ax.annotate(
            f"{height:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fictional_average_calls.png"), dpi=300)
    plt.close()

    print(f"Fictional plots generated in {output_dir}")


if __name__ == "__main__":
    generate_fictional_plots("./data/results/fictional/")
