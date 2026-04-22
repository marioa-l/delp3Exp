import argparse
import glob
import json
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.byProgramSampling import Programs
from sampling.byWorldSampling import Worlds
from utils.utils import natural_key, is_trivial_annot


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


def extract_model_features(model_path):
    """Extract structural features directly from a model JSON (no BN loading)."""
    data = load_json(model_path)
    em_vars = data["em_var"]
    af = data["af"]
    n_rules = len(af)
    n_annots = sum(1 for rule in af if not is_trivial_annot(rule[1]))
    n_worlds = 2 ** em_vars
    n_programs = 2 ** n_annots
    ratio_prog_world = n_programs / n_worlds if n_worlds > 0 else float("inf")
    literals_all = set()
    for level in data["literals"].values():
        for lit in level:
            literals_all.add(lit)
    return {
        "n_em_vars": em_vars,
        "n_annots": n_annots,
        "n_rules": n_rules,
        "n_worlds": n_worlds,
        "n_programs": n_programs,
        "ratio_prog_world": ratio_prog_world,
        "n_literals_total": len(literals_all),
    }


def run_mmm_experiment(models_path, output_csv, time_limit):
    models = sorted(
        glob.glob(os.path.join(models_path, "*model.json")), key=natural_key
    )[:10]
    print(f"Found {len(models)} models to process in {models_path}.")

    all_results = []

    for model in models:
        model_name = os.path.basename(model).replace(".json", "")
        print(f"\n{'=' * 60}")
        print(f"Processing model: {model_name}")
        print(f"{'=' * 60}")

        model_features = extract_model_features(model)

        exact_path = os.path.join(
            os.path.dirname(model), "exact", f"{model_name}_e_w.json"
        )
        if not os.path.exists(exact_path):
            exact_path = os.path.join(
                os.path.dirname(model), "exact", f"{model_name}_e_wNEW.json"
            )

        if not os.path.exists(exact_path):
            print(f"Warning: Exact data not found for {model_name}. Skipping.")
            continue

        exact_data = load_json(exact_path)
        literals_to_query = list(exact_data["status"].keys())

        print(f"Literals to evaluate: {literals_to_query}")

        # Ensure utils get_interest_lit logic uses the exact file properly (mocked via exact_data)

        # Sampling by Worlds
        print(f"\n>>> Running World Sampling ({time_limit}s)...")
        sampler_w = Worlds(model, os.path.dirname(output_csv) + "/")
        # Overwrite utils behavior directly for this isolated run
        sampler_w.utils.get_interest_lit = lambda: literals_to_query

        try:
            sampler_w.start_sampling(
                percentile_samples=0,
                source="random",
                info=f"_{model_name}_w",
                max_seconds=time_limit,
            )
            worlds_results = sampler_w.results
        except Exception as e:
            print(f"Error in World Sampling: {e}")
            worlds_results = None

        # Sampling by Programs
        print(f"\n>>> Running Program Sampling ({time_limit}s)...")
        sampler_p = Programs(model, os.path.dirname(output_csv) + "/")
        sampler_p.utils.get_interest_lit = lambda: literals_to_query
        try:
            sampler_p.start_sampling(
                percentile_samples=0, info=f"_{model_name}_p", max_seconds=time_limit
            )
            progs_results = sampler_p.results
        except Exception as e:
            print(f"Error in Program Sampling: {e}")
            progs_results = None

        # Gather data for CSV
        for lit in literals_to_query:
            ex_s = exact_data["status"][lit]
            ex_l, ex_u = (
                ex_s.get("l", ex_s.get("pyes", 0)),
                ex_s.get("u", 1 - ex_s.get("pno", 0)),
            )

            row = {
                "model": model_name,
                "literal": lit,
                "exact_l": ex_l,
                "exact_u": ex_u,
                **model_features,
            }

            if worlds_results and lit in worlds_results["status"]:
                w_s = worlds_results["status"][lit]
                w_l, w_u = (
                    w_s.get("l", w_s.get("pyes", 0)),
                    w_s.get("u", 1 - w_s.get("pno", 0)),
                )
                row["worlds_l"] = w_l
                row["worlds_u"] = w_u
                row["worlds_metric"] = compute_metric([w_l, w_u], [ex_l, ex_u])
                row["worlds_time"] = w_s.get("time", 0)
                row["worlds_calls"] = w_s.get(
                    "delp_calls", worlds_results["data"]["delp_calls"]
                )
            else:
                row["worlds_l"] = None
                row["worlds_u"] = None
                row["worlds_metric"] = None
                row["worlds_time"] = None
                row["worlds_calls"] = None

            if progs_results and lit in progs_results["status"]:
                p_s = progs_results["status"][lit]
                p_l, p_u = (
                    p_s.get("l", p_s.get("pyes", 0)),
                    p_s.get("u", 1 - p_s.get("pno", 0)),
                )
                row["progs_l"] = p_l
                row["progs_u"] = p_u
                row["progs_metric"] = compute_metric([p_l, p_u], [ex_l, ex_u])
                row["progs_time"] = p_s.get("time", 0)
                row["progs_calls"] = p_s.get(
                    "delp_calls", progs_results["data"]["delp_calls"]
                )
            else:
                row["progs_l"] = None
                row["progs_u"] = None
                row["progs_metric"] = None
                row["progs_time"] = None
                row["progs_calls"] = None

            all_results.append(row)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_csv, index=False)
    print(f"\nFinished processing. Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run sampling experiments and output CSV."
    )
    parser.add_argument("--models_path", help="Path to models directory", required=True)
    parser.add_argument("--output_csv", help="Path to output CSV file", required=True)
    parser.add_argument(
        "--time_limit",
        help="Time limit in seconds per literal/approach",
        type=int,
        default=60,
    )

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    run_mmm_experiment(args.models_path, args.output_csv, args.time_limit)
