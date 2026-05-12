"""
Enrich `results/all_results.csv` with AM, EM and AF complexity features.

These features are STATIC for a given model — computed once per (config, model)
and merged back into every row of that model. A JSON cache is kept on disk so
a re-run only computes models that are missing.

Usage:
    python sampling/enrich_dataset.py
    python sampling/enrich_dataset.py --models_root /path/to/delp3e_models
    python sampling/enrich_dataset.py --workers 4

The cache lives at `results/complexity_cache.json` and the enriched CSV is
written back to `results/all_results.csv` (the original is backed up first).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampling.complexity_features import compute_all  # noqa: E402

DEFAULT_INPUT_CSV  = "results/all_results.csv"
DEFAULT_CACHE      = "results/complexity_cache.json"
DEFAULT_DGRAPH_DIR = "results/dgraphs"

# Default model repo locations (try in order); can be overridden via --models_root
DEFAULT_ROOTS = [
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
]


def find_models_root(override: str | None) -> str:
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(
        f"Models repo not found in {DEFAULT_ROOTS}. Use --models_root."
    )


def load_cache(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_cache(cache: dict, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2)
    os.replace(tmp, path)


def _worker(args):
    config, model_name, model_path, dgraph_path = args
    try:
        return (config, model_name,
                compute_all(model_path, save_raw_to=dgraph_path), None)
    except Exception as e:  # pragma: no cover
        return (config, model_name, None, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_root", default=None,
                        help="Path to delp3e_models / exp-delp3e directory")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1 = sequential)")
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV,
                        help="CSV to enrich (will be overwritten; backed up first)")
    parser.add_argument("--cache", default=DEFAULT_CACHE,
                        help="Cache file for parsed complexity features")
    parser.add_argument("--dgraph_dir", default=DEFAULT_DGRAPH_DIR,
                        help="Where to read/write raw dGraph JSONs")
    args = parser.parse_args()

    root = find_models_root(args.models_root)
    print(f"Using models root: {root}")
    input_csv = args.input_csv
    cache_path = args.cache
    dgraph_dir = args.dgraph_dir

    df = pd.read_csv(input_csv, sep=";")
    pairs = df[["config", "model"]].drop_duplicates().values.tolist()
    print(f"Unique (config, model) pairs in dataset: {len(pairs)}")

    cache = load_cache(cache_path)
    print(f"Cached pairs: {len(cache)}")

    # Build job list for missing pairs only
    todo = []
    for cfg, model_name in pairs:
        key = f"{cfg}/{model_name}"
        dgraph_path = os.path.join(dgraph_dir, cfg, f"{model_name}_dgraph.json")
        # Skip only if BOTH the cached features and the raw dgraph already exist
        if (key in cache and cache[key].get("am_n_arguments") is not None
                and os.path.exists(dgraph_path)):
            continue
        path = os.path.join(root, cfg, f"{model_name}.json")
        if not os.path.exists(path):
            print(f"  [warn] missing model file: {path}")
            continue
        todo.append((cfg, model_name, path, dgraph_path))

    print(f"Pairs to compute: {len(todo)}")

    if not todo:
        print("Nothing to do. Merging cache into CSV directly.")
    else:
        t0 = time.time()
        if args.workers <= 1:
            for i, job in enumerate(todo):
                cfg, model_name, features, err = _worker(job)
                key = f"{cfg}/{model_name}"
                if err or features is None:
                    print(f"  [{i+1}/{len(todo)}] ERROR {key}: {err}")
                    continue
                cache[key] = features
                if (i + 1) % 25 == 0:
                    save_cache(cache, cache_path)
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    remaining = (len(todo) - i - 1) / rate
                    print(f"  [{i+1}/{len(todo)}] saved cache "
                          f"({elapsed/60:.1f} min elapsed, "
                          f"~{remaining/60:.1f} min remaining)")
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as exe:
                futures = {exe.submit(_worker, j): j for j in todo}
                for fut in as_completed(futures):
                    cfg, model_name, features, err = fut.result()
                    done += 1
                    key = f"{cfg}/{model_name}"
                    if err or features is None:
                        print(f"  [{done}/{len(todo)}] ERROR {key}: {err}")
                        continue
                    cache[key] = features
                    if done % 25 == 0:
                        save_cache(cache, cache_path)
                        elapsed = time.time() - t0
                        rate = done / elapsed
                        remaining = (len(todo) - done) / rate
                        print(f"  [{done}/{len(todo)}] saved cache "
                              f"({elapsed/60:.1f} min elapsed, "
                              f"~{remaining/60:.1f} min remaining)")
        save_cache(cache, cache_path)
        print(f"All done in {(time.time() - t0)/60:.1f} min.")

    # Merge cache features into the CSV
    print("\nMerging features into CSV...")
    feat_keys = set()
    for v in cache.values():
        feat_keys.update(v.keys())
    feat_keys = sorted(feat_keys)

    feature_rows = []
    for cfg, model_name in df[["config", "model"]].values:
        key = f"{cfg}/{model_name}"
        feats = cache.get(key, {})
        feature_rows.append({k: feats.get(k) for k in feat_keys})

    feat_df = pd.DataFrame(feature_rows)
    # Drop columns from feat_df that already exist in df (avoid duplicates)
    feat_df = feat_df[[c for c in feat_df.columns if c not in df.columns]]
    enriched = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    backup = input_csv + ".pre_enrich"
    if not os.path.exists(backup):
        os.rename(input_csv, backup)
        print(f"  Backed up original to: {backup}")
    enriched.to_csv(input_csv, sep=";", index=False)
    print(f"  Saved enriched CSV: {input_csv}")
    print(f"  Total columns: {len(enriched.columns)}  (added {len(feat_df.columns)} new)")


if __name__ == "__main__":
    main()
