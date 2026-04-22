import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.byProgramSampling import Programs
from sampling.byWorldSampling import Worlds
from utils.utils import natural_key


def print_results(model_path, output_path, suffix):
    print("-" * 50)


def plot_convergence(history_w, history_p, model_path, output_path):
    """Genera gráficos de convergencia para cada literal."""
    if not history_w and not history_p:
        return

    model_name = os.path.basename(model_path).replace(".json", "")
    # Obtener literales de la primera entrada disponible
    literals = (
        history_w[0]["values"].keys() if history_w else history_p[0]["values"].keys()
    )

    for lit in literals:
        plt.figure(figsize=(10, 6))

        # Plot World Sampling
        if history_w:
            times_w = [h["time"] for h in history_w]
            l_w = [h["values"][lit]["l"] for h in history_w]
            u_w = [h["values"][lit]["u"] for h in history_w]
            plt.plot(times_w, l_w, "b-", label="Worlds Lower")
            plt.plot(times_w, u_w, "b--", label="Worlds Upper")

        # Plot Program Sampling
        if history_p:
            times_p = [h["time"] for h in history_p]
            l_p = [h["values"][lit]["l"] for h in history_p]
            u_p = [h["values"][lit]["u"] for h in history_p]
            plt.plot(times_p, l_p, "r-", label="Programs Lower")
            plt.plot(times_p, u_p, "r--", label="Programs Upper")

        plt.xlabel("Time (s)")
        plt.ylabel("Probability Interval [l, u]")
        plt.title(f"Convergence for {lit} in {model_name}")
        plt.legend()
        plt.grid(True)

        # Guardar gráfico
        filename = f"{model_name}_{lit.replace('~', 'neg_')}_convergence.png"
        plt.savefig(os.path.join(output_path, filename))
        plt.close()
        print(f"Gráfico guardado: {filename}")


def plot_convergence_by_time(
    history_w, history_p, model_path, output_path, exact_data=None
):
    """Genera gráficos de convergencia (L y U) por tiempo para cada literal, con estilo de publicación."""
    if not history_w and not history_p:
        return

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

    model_name = os.path.basename(model_path).replace(".json", "")
    literals = list(history_w.keys()) if history_w else list(history_p.keys())

    for lit in literals:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Exact lines
        if exact_data and lit in exact_data["status"]:
            exact_s = exact_data["status"][lit]
            e_l = exact_s.get("l", exact_s.get("pyes", 0))
            e_u = exact_s.get("u", 1 - exact_s.get("pno", 0))
            ax.axhline(
                y=e_l, color="black", linestyle=":", linewidth=1.5, label="Exact L"
            )
            ax.axhline(
                y=e_u, color="black", linestyle="-.", linewidth=1.5, label="Exact U"
            )

        if history_w and lit in history_w:
            times_w = [h["time"] for h in history_w[lit]]
            l_w = [h["l"] for h in history_w[lit]]
            u_w = [h["u"] for h in history_w[lit]]
            ax.plot(
                times_w,
                l_w,
                color="#1f77b4",
                linestyle="-",
                linewidth=2,
                label="Worlds L",
            )
            ax.plot(
                times_w,
                u_w,
                color="#1f77b4",
                linestyle="--",
                linewidth=2,
                label="Worlds U",
            )

        if history_p and lit in history_p:
            times_p = [h["time"] for h in history_p[lit]]
            l_p = [h["l"] for h in history_p[lit]]
            u_p = [h["u"] for h in history_p[lit]]
            ax.plot(
                times_p,
                l_p,
                color="#ff7f0e",
                linestyle="-",
                linewidth=2,
                label="Programs L",
            )
            ax.plot(
                times_p,
                u_p,
                color="#ff7f0e",
                linestyle="--",
                linewidth=2,
                label="Programs U",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Probability Interval [L, U]")
        ax.set_title(f"Convergence by Time: {lit}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filename = f"{model_name}_{lit.replace('~', 'neg_')}_time_conv.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300)
        plt.close(fig)
        print(f"  Gráfico guardado: {filename}")


def plot_convergence_by_calls(
    history_w, history_p, model_path, output_path, exact_data=None
):
    """Genera gráficos de convergencia por llamadas DeLP para cada literal."""
    if not history_w and not history_p:
        return

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

    model_name = os.path.basename(model_path).replace(".json", "")
    literals = list(history_w.keys()) if history_w else list(history_p.keys())

    for lit in literals:
        fig, ax = plt.subplots(figsize=(8, 5))

        # Exact lines
        if exact_data and lit in exact_data["status"]:
            exact_s = exact_data["status"][lit]
            e_l = exact_s.get("l", exact_s.get("pyes", 0))
            e_u = exact_s.get("u", 1 - exact_s.get("pno", 0))
            ax.axhline(
                y=e_l, color="black", linestyle=":", linewidth=1.5, label="Exact L"
            )
            ax.axhline(
                y=e_u, color="black", linestyle="-.", linewidth=1.5, label="Exact U"
            )

        if history_w and lit in history_w:
            calls_w = [h["calls"] for h in history_w[lit]]
            l_w = [h["l"] for h in history_w[lit]]
            u_w = [h["u"] for h in history_w[lit]]
            ax.plot(
                calls_w,
                l_w,
                color="#1f77b4",
                linestyle="-",
                linewidth=2,
                label="Worlds L",
            )
            ax.plot(
                calls_w,
                u_w,
                color="#1f77b4",
                linestyle="--",
                linewidth=2,
                label="Worlds U",
            )

        if history_p and lit in history_p:
            calls_p = [h["calls"] for h in history_p[lit]]
            l_p = [h["l"] for h in history_p[lit]]
            u_p = [h["u"] for h in history_p[lit]]
            ax.plot(
                calls_p,
                l_p,
                color="#ff7f0e",
                linestyle="-",
                linewidth=2,
                label="Programs L",
            )
            ax.plot(
                calls_p,
                u_p,
                color="#ff7f0e",
                linestyle="--",
                linewidth=2,
                label="Programs U",
            )

        ax.set_xlabel("DeLP Solver Calls")
        ax.set_ylabel("Probability Interval [L, U]")
        ax.set_title(f"Convergence by DeLP Calls: {lit}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="center right", bbox_to_anchor=(1.25, 0.5))
        ax.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        filename = f"{model_name}_{lit.replace('~', 'neg_')}_calls_conv.png"
        plt.savefig(os.path.join(output_path, filename), dpi=300)
        plt.close(fig)
        print(f"  Gráfico guardado: {filename}")


def run_experiment(models_path, base_output_path, time_limit):
    models = sorted(glob.glob(models_path + "*model.json"), key=natural_key)
    print(f"Found {len(models)} models to process.")

    for model in models:
        model_name = os.path.basename(model).replace(".json", "")
        output_path = os.path.join(base_output_path, model_name, f"{time_limit}s")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        print(f"\n{'=' * 60}")
        print(f"Processing model: {model}")
        print(f"Output directory: {output_path}")
        print(f"{'=' * 60}")

        history_w = []
        history_p = []

        # Sampling by Worlds
        print(f"\n>>> Running World Sampling ({time_limit}s)...")
        try:
            sampler_w = Worlds(model, output_path + "/")
            # percentile_samples is ignored if max_seconds is set
            history_w = sampler_w.start_sampling(
                percentile_samples=0,
                source="random",
                info="_s_w_time",
                max_seconds=time_limit,
            )
            print_results(model, output_path + "/", "_s_w_time")
        except Exception as e:
            print(f"Error in World Sampling: {e}")

        # Sampling by Programs
        print(f"\n>>> Running Program Sampling ({time_limit}s)...")
        try:
            sampler_p = Programs(model, output_path + "/")
            # percentile_samples is ignored if max_seconds is set
            history_p = sampler_p.start_sampling(
                percentile_samples=0, info="_s_p_time", max_seconds=time_limit
            )
            print_results(model, output_path + "/", "_s_p_time")
        except Exception as e:
            print(f"Error in Program Sampling: {e}")

        # Intentar cargar datos exactos para el gráfico
        exact_data = None
        exact_path = os.path.join(
            os.path.dirname(model),
            "exact",
            os.path.basename(model).replace(".json", "_e_wNEW.json"),
        )
        if os.path.exists(exact_path):
            with open(exact_path, "r") as f:
                exact_data = json.load(f)

        # Generar gráficos de convergencia
        print("\n>>> Generating Convergence Plots...")
        print("  By Time:")
        plot_convergence_by_time(history_w, history_p, model, output_path, exact_data)
        print("  By DeLP Calls:")
        plot_convergence_by_calls(history_w, history_p, model, output_path, exact_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare sampling methods by time.")
    parser.add_argument(
        "--config", help="Path to configuration file", default="config.json"
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = json.load(f)

    input_path = config.get("input_path", "./data/models/")
    output_path = config.get("output_path", "./data/results/")
    time_limit = config.get("time_limit", 60)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    run_experiment(input_path, output_path, time_limit)
