import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.byWorldSampling import Worlds
from utils.utils import natural_key


def compute_exact(input_path, output_path):
    models = sorted(glob.glob(os.path.join(input_path, "*model.json")), key=natural_key)
    print(f"Found {len(models)} models to compute exact intervals.")

    for model in models:
        model_name = os.path.basename(model).replace(".json", "")
        exact_file = os.path.join(output_path, f"{model_name}_e_wNEW.json")

        if os.path.exists(exact_file):
            print(
                f"\nSkipping model {model_name} because exact results already exist: {exact_file}"
            )
            continue

        print(f"\n{'=' * 60}")
        print(f"Computing exact intervals for model: {model}")
        print(f"{'=' * 60}")

        try:
            sampler = Worlds(model, output_path + "/")
            # percentile_samples=100 and max_seconds=None will trigger exact computation in byWorldSampling
            history = sampler.start_sampling(
                percentile_samples=100, source="exact", info="_e_w"
            )
            print(f"Exact computation finished successfully.")
        except Exception as e:
            print(f"Error computing exact intervals: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute exact probability intervals.")
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
    output_path = config.get("exact_output_path", "./data/exact/")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    compute_exact(input_path, output_path)
