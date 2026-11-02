"""
Re-run the queries that stayed "unresolved" in the time-to-exact experiment
(neither sampler reached the exact interval before the safety cap) using a
longer cap.

Only the (config, model, literal) triples flagged ``gt_winner == unresolved``
in the input CSV are re-executed, so the cost stays proportional to what is
actually missing. ``--all_queries`` lifts that filter, ``--models_csv``
restricts to listed models and ``--methods`` runs a single sampler, which
together re-run one half of the ground truth where a bug invalidated it:

    python sampling/rerun_unresolved.py \\
        --input_csv results_tte/all_results.csv \\
        --all_queries --methods programs \\
        --models_csv results_tte/bitshift_affected_models.csv \\
        --cap 900 --workers 16 \\
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \\
        --output_root results_tte_bitshift

Designed for a multi-day HPC run:
    - resumable: on restart, triples already present in the output CSVs are
      skipped, so the job can be killed and relaunched freely,
    - incremental: every finished model is appended immediately,
    - parallel: one process per model.

Output uses the same 20-column schema as time_to_exact_experiment.py, so
``merge_tte.py`` can merge these results with the originals.

Usage (icic, 1-hour cap):
    python sampling/rerun_unresolved.py \\
        --input_csv results_tte/all_results.csv \\
        --cap 3600 \\
        --workers 16 \\
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \\
        --output_root results_tte_long

Resume after an interruption: rerun the exact same command.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.byProgramSampling import Programs          # noqa: E402
from sampling.byWorldSampling import Worlds              # noqa: E402
from sampling.time_to_exact_experiment import (          # noqa: E402
    find_models_root, exact_path_for, load_interest_targets,
)

# Same column order as time_to_exact_experiment._run_one_model
FULL_COLUMNS = [
    "config", "model", "literal", "exact_l", "exact_u", "safety_cap",
    "time_worlds", "worlds_wall_time", "worlds_l", "worlds_u",
    "worlds_delp_calls", "worlds_ok",
    "time_programs", "programs_wall_time", "programs_l", "programs_u",
    "programs_delp_calls", "programs_ok",
    "gt_winner", "ok",
]


# ── Worker ───────────────────────────────────────────────────────────────────
def _run_one_model(job: tuple) -> list:
    """Re-run only the requested literals of a single model."""
    (cfg, model_name, model_path, exact_path,
     save_dir, cap, dgraph_path, epsilon, literals, methods) = job

    try:
        targets = load_interest_targets(exact_path)
    except Exception:
        return []
    targets = {lit: v for lit, v in targets.items() if lit in set(literals)}
    if not targets:
        return []

    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for lit, (exact_l, exact_u) in targets.items():
        row = {
            "config": cfg, "model": model_name, "literal": lit,
            "exact_l": exact_l, "exact_u": exact_u, "safety_cap": cap,
        }

        w_start = time.time()
        try:
            if "worlds" not in methods:
                raise StopIteration  # leave the worlds columns empty
            wsampler = Worlds(model_path, save_dir + "/")
            wsampler.utils.get_interest_lit = lambda: [lit]
            wsampler.start_sampling(
                percentile_samples=0, source="random", info="_tte_w",
                max_seconds=cap, mode="random", dgraph_path=dgraph_path,
                stop_at_exact={lit: (exact_l, exact_u)},
                stop_epsilon=epsilon,
            )
            st = wsampler.results["status"].get(lit, {})
            row["time_worlds"] = st.get("time_to_exact")
            row["worlds_wall_time"] = time.time() - w_start
            row["worlds_l"] = st.get("pyes")
            row["worlds_u"] = 1 - st.get("pno", 0.0)
            row["worlds_delp_calls"] = st.get("delp_calls", 0)
            row["worlds_ok"] = True
        except StopIteration:
            pass
        except Exception as e:
            row.update({"worlds_ok": False, "worlds_error": str(e),
                        "worlds_wall_time": time.time() - w_start})

        p_start = time.time()
        try:
            if "programs" not in methods:
                raise StopIteration
            psampler = Programs(model_path, save_dir + "/")
            psampler.utils.get_interest_lit = lambda: [lit]
            psampler.start_sampling(
                percentile_samples=0, info="_tte_p",
                max_seconds=cap, mode="random",
                stop_at_exact={lit: (exact_l, exact_u)},
                stop_epsilon=epsilon,
            )
            st = psampler.results["status"].get(lit, {})
            row["time_programs"] = st.get("time_to_exact")
            row["programs_wall_time"] = time.time() - p_start
            row["programs_l"] = st.get("pyes")
            row["programs_u"] = 1 - st.get("pno", 0.0)
            row["programs_delp_calls"] = st.get("delp_calls", 0)
            row["programs_ok"] = True
        except StopIteration:
            pass
        except Exception as e:
            row.update({"programs_ok": False, "programs_error": str(e),
                        "programs_wall_time": time.time() - p_start})

        # With a single method the winner below is provisional; merge_tte.py
        # recomputes it once the other method's columns are filled in from
        # the earlier run.
        tw, tp = row.get("time_worlds"), row.get("time_programs")
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
    try:
        return _run_one_model(job)
    except Exception as e:  # pragma: no cover
        cfg, name, *_ = job
        print(f"  [error] {cfg}/{name}: {e}", flush=True)
        return []


# ── Resume support ───────────────────────────────────────────────────────────
def already_done(output_root: str) -> set:
    """Return the (config, model, literal) triples already present on disk."""
    done = set()
    if not os.path.isdir(output_root):
        return done
    for cfg in sorted(os.listdir(output_root)):
        path = os.path.join(output_root, cfg, f"{cfg}_tte.csv")
        if not os.path.exists(path):
            continue
        # Read defensively: the file is appended to by many runs.
        with open(path) as fh:
            for line in fh:
                f = line.rstrip("\n").split(";")
                if len(f) != len(FULL_COLUMNS) or f[0] == "config":
                    continue
                done.add((f[0], f[1], f[2]))
    return done


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="results_tte/all_results.csv",
                        help="CSV holding the gt_winner column to filter on")
    parser.add_argument("--cap", type=int, default=3600,
                        help="Safety cap in seconds per sampler per literal "
                             "(default 3600 = 1 hour)")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--configs", default=None,
                        help="Restrict to these configs (comma-separated)")
    parser.add_argument("--models_root", default=None)
    parser.add_argument("--dgraph_dir", default="results/dgraphs")
    parser.add_argument("--output_root", default="results_tte_long")
    parser.add_argument("--dry_run", action="store_true",
                        help="Report what would be run and exit")
    parser.add_argument("--methods", default="worlds,programs",
                        help="Which samplers to run (comma-separated). With "
                             "one method the other's columns stay empty and "
                             "merge_tte.py --supersede fills them from the "
                             "earlier root.")
    parser.add_argument("--all_queries", action="store_true",
                        help="Run every query of --input_csv, not only the "
                             "unresolved ones")
    parser.add_argument("--models_csv", default=None,
                        help="Restrict to the (config, model) pairs listed in "
                             "this CSV")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    models_root = find_models_root(args.models_root)
    df = pd.read_csv(args.input_csv, sep=";")
    if not args.all_queries:
        df = df[df["gt_winner"] == "unresolved"]
    todo = df[["config", "model", "literal"]].drop_duplicates()
    if args.configs:
        keep = {c.strip() for c in args.configs.split(",") if c.strip()}
        todo = todo[todo["config"].isin(keep)]
    if args.models_csv:
        sel = pd.read_csv(args.models_csv)[["config", "model"]].drop_duplicates()
        todo = todo.merge(sel, on=["config", "model"])

    done = already_done(args.output_root)
    if done:
        before = len(todo)
        mask = [(r.config, r.model, r.literal) not in done
                for r in todo.itertuples()]
        todo = todo[mask]
        print(f"Resuming: {before - len(todo)} of {before} already on disk")

    if todo.empty:
        print("Nothing left to run.")
        return

    print(f"Models root : {models_root}")
    print(f"Methods     : {methods}")
    print(f"Safety cap  : {args.cap}s ({args.cap/60:.0f} min) per sampler")
    print(f"Workers     : {args.workers}")
    print(f"Output root : {args.output_root}")
    print(f"\nQueries to re-run: {len(todo)} across "
          f"{todo.groupby(['config','model']).ngroups} models")
    print(todo.groupby("config").size().to_string())

    worst_h = len(todo) * len(methods) * args.cap / 3600 / max(args.workers, 1)
    print(f"\nWorst case (every run hits the cap): {worst_h:.1f} h "
          f"of wall time at {args.workers} workers")

    if args.dry_run:
        return

    # Build one job per model
    jobs = []
    for (cfg, model_name), grp in todo.groupby(["config", "model"]):
        cfg_dir = os.path.join(models_root, cfg)
        model_path = os.path.join(cfg_dir, f"{model_name}.json")
        if not os.path.exists(model_path):
            print(f"  [skip] missing model file: {model_path}")
            continue
        jobs.append((
            cfg, model_name, model_path, exact_path_for(cfg_dir, model_name),
            os.path.join(args.output_root, cfg, model_name),
            args.cap,
            os.path.join(args.dgraph_dir, cfg, f"{model_name}_dgraph.json"),
            args.epsilon,
            grp["literal"].tolist(),
            methods,
        ))

    for cfg in todo["config"].unique():
        os.makedirs(os.path.join(args.output_root, cfg), exist_ok=True)

    t0 = time.time()
    n_done = 0
    tally = {"worlds": 0, "programs": 0, "unresolved": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_run_one_model_safe, j): j for j in jobs}
        for fut in as_completed(futures):
            n_done += 1
            job = futures[fut]
            cfg = job[0]
            rows = fut.result()
            if rows:
                csv_path = os.path.join(args.output_root, cfg, f"{cfg}_tte.csv")
                header = not os.path.exists(csv_path)
                # reindex, not [], so the columns of a method that did not
                # run come out empty instead of raising
                pd.DataFrame(rows).reindex(columns=FULL_COLUMNS).to_csv(
                    csv_path, mode="a", header=header, index=False, sep=";")
                for r in rows:
                    tally[r.get("gt_winner", "unresolved")] += 1
            elapsed_h = (time.time() - t0) / 3600
            print(f"  [{n_done}/{len(jobs)}] {cfg}/{job[1]:<12s} "
                  f"resolved so far: {tally['worlds']}w/"
                  f"{tally['programs']}p/{tally['unresolved']}u "
                  f"({elapsed_h:.2f} h)", flush=True)

    print(f"\nFinished in {(time.time()-t0)/3600:.2f} h")
    print(f"Newly resolved: {tally['worlds']} worlds, "
          f"{tally['programs']} programs, "
          f"still unresolved: {tally['unresolved']}")
    print(f"\nMerge with the originals via: python sampling/merge_tte.py")


if __name__ == "__main__":
    main()
