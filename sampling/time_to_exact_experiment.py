"""
Ground-truth experiment: for every INTEREST literal of every model in a
configuration, run each sampler until it reaches the exact interval or
until a safety cap is hit, and record ``time_to_exact`` for both methods.

The recommended method (from the decision tree) will later be validated
against the ground-truth winner, defined as:

    - both methods reached the exact interval  -> winner = the faster one
    - only one reached                          -> winner = the one that did
    - neither reached (safety cap on both)      -> "unresolved"

Reads:
    Exact files under <models_root>/<config>/exact/<N>model_e_w.json
    Model files under <models_root>/<config>/<N>model.json
    dGraph cache under <dgraph_dir>/<config>/<N>model_dgraph.json (optional)

Writes to <output_dir>/<config>/:
    <config>_tte.csv          — one row per (model, literal)
    <model>/                  — per-model artefacts (kept minimal)

Usage:
    python sampling/time_to_exact_experiment.py \\
        --configs sms,sss \\
        --n_models 100 \\
        --cap 900 \\
        --workers 8 \\
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \\
        --output_root results_tte
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byProgramSampling import Programs  # noqa: E402
from sampling.byWorldSampling import Worlds      # noqa: E402
from utils.utils import natural_key              # noqa: E402


ALL_CONFIGS = ["sms", "sss", "scs", "smm", "mss", "ssm", "mms", "mcs",
               "ccs", "css", "cms", "scm", "ssc", "smc", "cmm"]

DEFAULT_ROOTS = [
    "/home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e",
    "/Users/marioleiva/Documents/desarrollo/delp3e_models",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def find_models_root(override: str | None) -> str:
    if override and os.path.isdir(override):
        return override
    for r in DEFAULT_ROOTS:
        if os.path.isdir(r):
            return r
    raise FileNotFoundError(f"Models root not found; tried {DEFAULT_ROOTS}")


def exact_path_for(cfg_dir: str, model_name: str) -> str:
    return os.path.join(cfg_dir, "exact", f"{model_name}_e_w.json")


def load_interest_targets(exact_path: str) -> dict:
    """
    Return ``{literal: (l, u)}`` for every literal flagged INTEREST in the
    exact file.
    """
    with open(exact_path) as f:
        data = json.load(f)
    out = {}
    for lit, s in data.get("status", {}).items():
        if s.get("flag") != "INTEREST":
            continue
        l = s.get("l", s.get("pyes", 0.0))
        u = s.get("u", 1 - s.get("pno", 0.0))
        out[lit] = (float(l), float(u))
    return out


# ── Worker ───────────────────────────────────────────────────────────────────
def _run_one_model(job: tuple) -> list:
    """
    Process a single model: for each INTEREST literal, run both samplers with
    the safety cap and stop_at_exact. Returns a list of result dicts.
    """
    (cfg, model_name, model_path, exact_path,
     save_dir, cap, dgraph_path, epsilon) = job

    rows = []
    try:
        targets = load_interest_targets(exact_path)
    except Exception as e:
        return [{
            "config": cfg, "model": model_name, "literal": None,
            "ok": False, "error": f"exact_load: {e}",
        }]
    if not targets:
        return [{
            "config": cfg, "model": model_name, "literal": None,
            "ok": True, "note": "no INTEREST literals",
        }]

    os.makedirs(save_dir, exist_ok=True)

    for lit, (exact_l, exact_u) in targets.items():
        row = {
            "config": cfg, "model": model_name, "literal": lit,
            "exact_l": exact_l, "exact_u": exact_u,
            "safety_cap": cap,
        }

        # ── Worlds ────────────────────────────────────────────────────────
        w_start = time.time()
        try:
            wsampler = Worlds(model_path, save_dir + "/")
            wsampler.utils.get_interest_lit = lambda: [lit]
            wsampler.start_sampling(
                percentile_samples=0, source="random",
                info=f"_tte_w",
                max_seconds=cap, mode="random",
                dgraph_path=dgraph_path,
                stop_at_exact={lit: (exact_l, exact_u)},
                stop_epsilon=epsilon,
            )
            w_status = wsampler.results["status"].get(lit, {})
            row["time_worlds"] = w_status.get("time_to_exact")
            row["worlds_wall_time"] = time.time() - w_start
            row["worlds_l"] = w_status.get("pyes")
            row["worlds_u"] = 1 - w_status.get("pno", 0.0)
            row["worlds_delp_calls"] = w_status.get("delp_calls", 0)
            row["worlds_ok"] = True
        except Exception as e:
            row.update({"worlds_ok": False,
                        "worlds_error": str(e),
                        "worlds_wall_time": time.time() - w_start})

        # ── Programs ─────────────────────────────────────────────────────
        p_start = time.time()
        try:
            psampler = Programs(model_path, save_dir + "/")
            psampler.utils.get_interest_lit = lambda: [lit]
            psampler.start_sampling(
                percentile_samples=0, info=f"_tte_p",
                max_seconds=cap, mode="random",
                stop_at_exact={lit: (exact_l, exact_u)},
                stop_epsilon=epsilon,
            )
            p_status = psampler.results["status"].get(lit, {})
            row["time_programs"] = p_status.get("time_to_exact")
            row["programs_wall_time"] = time.time() - p_start
            row["programs_l"] = p_status.get("pyes")
            row["programs_u"] = 1 - p_status.get("pno", 0.0)
            row["programs_delp_calls"] = p_status.get("delp_calls", 0)
            row["programs_ok"] = True
        except Exception as e:
            row.update({"programs_ok": False,
                        "programs_error": str(e),
                        "programs_wall_time": time.time() - p_start})

        # ── Ground-truth winner from times ───────────────────────────────
        tw = row.get("time_worlds")
        tp = row.get("time_programs")
        if tw is not None and tp is not None:
            row["gt_winner"] = "worlds" if tw < tp else "programs"
        elif tw is not None:
            row["gt_winner"] = "worlds"
        elif tp is not None:
            row["gt_winner"] = "programs"
        else:
            row["gt_winner"] = "unresolved"
        row["ok"] = True
        rows.append(row)

    return rows


def _run_one_model_safe(job):
    """Wrapper so an unexpected crash inside one worker doesn't kill the pool."""
    try:
        return _run_one_model(job)
    except Exception as e:  # pragma: no cover
        cfg, name, *_ = job
        return [{"config": cfg, "model": name, "literal": None,
                 "ok": False, "error": str(e)}]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs",
                        default=",".join(ALL_CONFIGS),
                        help="Comma-separated configs (default: all 15)")
    parser.add_argument("--n_models", type=int, default=100,
                        help="Models with an exact file to include per config")
    parser.add_argument("--cap", type=int, default=900,
                        help="Safety cap in seconds per sampler per literal "
                             "(default 900 = 15 minutes)")
    parser.add_argument("--epsilon", type=float, default=1e-6,
                        help="Tolerance for the exact-interval match")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--models_root", default=None)
    parser.add_argument("--dgraph_dir", default="results/dgraphs")
    parser.add_argument("--output_root", default="results_tte")
    args = parser.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    models_root = find_models_root(args.models_root)
    os.makedirs(args.output_root, exist_ok=True)

    print(f"Models root:   {models_root}")
    print(f"Configs:       {configs}")
    print(f"Models each:   {args.n_models}")
    print(f"Safety cap:    {args.cap}s")
    print(f"Epsilon:       {args.epsilon}")
    print(f"Workers:       {args.workers}")
    print(f"Output root:   {args.output_root}\n")

    grand_total = 0
    for cfg in configs:
        cfg_dir = os.path.join(models_root, cfg)
        if not os.path.isdir(cfg_dir):
            print(f"[skip] {cfg}: directory not found")
            continue
        model_paths = sorted(
            glob.glob(os.path.join(cfg_dir, "*model.json")),
            key=natural_key,
        )
        # Keep only models with an exact file
        model_paths = [p for p in model_paths
                       if os.path.exists(exact_path_for(cfg_dir,
                                                       os.path.basename(p)[:-5]))]
        model_paths = model_paths[: args.n_models]
        if not model_paths:
            print(f"[skip] {cfg}: no models with exact files")
            continue

        cfg_out = os.path.join(args.output_root, cfg)
        os.makedirs(cfg_out, exist_ok=True)
        csv_path = os.path.join(cfg_out, f"{cfg}_tte.csv")
        write_header = not os.path.exists(csv_path)

        # Build one job per model
        jobs = []
        for model_path in model_paths:
            model_name = os.path.basename(model_path)[:-5]
            n = ''.join(ch for ch in model_name if ch.isdigit())
            dgraph_path = os.path.join(args.dgraph_dir, cfg,
                                       f"{model_name}_dgraph.json")
            save_dir = os.path.join(cfg_out, model_name)
            jobs.append((cfg, model_name, model_path,
                         exact_path_for(cfg_dir, model_name),
                         save_dir, args.cap, dgraph_path, args.epsilon))

        print(f"\n=== {cfg}: {len(jobs)} models ===")
        t0 = time.time()
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = {exe.submit(_run_one_model_safe, j): j for j in jobs}
            for fut in as_completed(futures):
                done += 1
                rows = fut.result()
                if rows:
                    df_chunk = pd.DataFrame(rows)
                    df_chunk.to_csv(csv_path, mode="a", header=write_header,
                                    index=False, sep=";")
                    write_header = False
                gt = [r.get("gt_winner") for r in rows if r.get("ok")]
                summary = f"{sum(1 for w in gt if w=='worlds')}w/" \
                          f"{sum(1 for w in gt if w=='programs')}p/" \
                          f"{sum(1 for w in gt if w=='unresolved')}u"
                job = futures[fut]
                elapsed_min = (time.time() - t0) / 60
                print(f"  [{done}/{len(jobs)}] {job[1]:>10s}  {summary:<12s}"
                      f"  ({elapsed_min:.1f} min)",
                      flush=True)

        elapsed_min = (time.time() - t0) / 60
        print(f"  {cfg} finished in {elapsed_min:.1f} min "
              f"(CSV: {csv_path})")
        grand_total += elapsed_min

    print(f"\nTotal wall time: {grand_total:.1f} min "
          f"({grand_total/60:.1f} h)")


if __name__ == "__main__":
    main()
