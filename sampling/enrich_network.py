"""
Add network / graph-theoretic features to the complexity cache and merge them
into the enriched CSV.

Reuses the already-saved dGraphs (`results/dgraphs/<cfg>/<model>_dgraph.json`)
and BN .bifxml files, so the DeLP solver is NOT called again. The work is
purely in-memory graph computations and is fast (~minutes for the whole
dataset).

Usage:
    python sampling/enrich_network.py \
        --input_csv results_5min/all_results.csv \
        --cache    results/complexity_cache.json \
        --dgraph_dir results/dgraphs \
        --workers 4
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
from sampling.network_features import compute_all_network  # noqa: E402

DEFAULT_INPUT_CSV  = "results/all_results.csv"
DEFAULT_CACHE      = "results/complexity_cache.json"
DEFAULT_DGRAPH_DIR = "results/dgraphs"

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
    raise FileNotFoundError(f"Models repo not found. Use --models_root.")


def _worker(args):
    cfg, model_name, model_path, bn_path, dgraph_path = args
    try:
        return (cfg, model_name,
                compute_all_network(model_path, bn_path=bn_path,
                                    dgraph_path=dgraph_path), None)
    except Exception as e:
        return (cfg, model_name, None, str(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=DEFAULT_INPUT_CSV)
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--dgraph_dir", default=DEFAULT_DGRAPH_DIR)
    parser.add_argument("--models_root", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    root = find_models_root(args.models_root)
    print(f"Using models root: {root}")
    print(f"Reading CSV: {args.input_csv}")
    print(f"Cache: {args.cache}")
    print(f"DGraph dir: {args.dgraph_dir}")

    df = pd.read_csv(args.input_csv, sep=";")
    pairs = df[["config", "model"]].drop_duplicates().values.tolist()
    print(f"Unique (config, model) pairs: {len(pairs)}")

    # Load cache (may already contain complexity features from previous step)
    cache = {}
    if os.path.exists(args.cache):
        with open(args.cache) as f:
            cache = json.load(f)
    print(f"Cached pairs: {len(cache)}")

    # Build jobs for pairs that don't have network features yet
    todo = []
    for cfg, model_name in pairs:
        key = f"{cfg}/{model_name}"
        existing = cache.get(key, {})
        if "em_graph_density" in existing and "am_attack_density" in existing:
            continue  # already has network features
        model_path = os.path.join(root, cfg, f"{model_name}.json")
        if not os.path.exists(model_path):
            print(f"  [warn] missing model: {model_path}")
            continue
        import re
        n = re.search(r"(\d+)model", model_name)
        bn_path = os.path.join(root, cfg, f"BN{n.group(1)}.bifxml") if n else None
        dgraph_path = os.path.join(args.dgraph_dir, cfg,
                                   f"{model_name}_dgraph.json")
        todo.append((cfg, model_name, model_path, bn_path, dgraph_path))

    print(f"Pairs to compute: {len(todo)}")
    if todo:
        t0 = time.time()
        if args.workers <= 1:
            for i, job in enumerate(todo):
                cfg, model_name, features, err = _worker(job)
                key = f"{cfg}/{model_name}"
                if err or features is None:
                    print(f"  [{i+1}/{len(todo)}] ERROR {key}: {err}")
                    continue
                cache.setdefault(key, {}).update(features)
                if (i + 1) % 50 == 0:
                    with open(args.cache, "w") as f:
                        json.dump(cache, f, indent=2)
                    print(f"  [{i+1}/{len(todo)}] cache saved")
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
                    cache.setdefault(key, {}).update(features)
                    if done % 50 == 0:
                        with open(args.cache, "w") as f:
                            json.dump(cache, f, indent=2)
                        print(f"  [{done}/{len(todo)}] cache saved")
        with open(args.cache, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Done in {(time.time() - t0)/60:.1f} min.")

    # Merge into CSV. Cache values are the source of truth for every feature
    # key in the cache: overwrite pre-existing columns (which may be all-NaN
    # from a previous enrich pass) rather than silently keeping the empties.
    print("\nMerging into CSV...")
    feat_keys = set()
    for v in cache.values():
        feat_keys.update(v.keys())
    feat_keys = sorted(feat_keys)
    feat_rows = [
        {k: cache.get(f"{cfg}/{m}", {}).get(k) for k in feat_keys}
        for cfg, m in df[["config", "model"]].values
    ]
    feat_df = pd.DataFrame(feat_rows).reset_index(drop=True)
    df = df.reset_index(drop=True)

    added, overwritten = 0, 0
    for col in feat_df.columns:
        if col in df.columns:
            df[col] = feat_df[col]
            overwritten += 1
        else:
            df[col] = feat_df[col]
            added += 1

    backup = args.input_csv + ".pre_network"
    if not os.path.exists(backup):
        os.rename(args.input_csv, backup)
        print(f"  Backed up original to: {backup}")
    df.to_csv(args.input_csv, sep=";", index=False)
    print(f"  Saved enriched CSV: {args.input_csv}")
    print(f"  Columns: {len(df.columns)}  "
          f"(added {added}, overwritten {overwritten})")


if __name__ == "__main__":
    main()
