"""
Experiment runner: compare World Sampling vs Program Sampling for a given
model configuration.

For each model that has an exact file, runs both sampling approaches with the
configured time limit per literal, then writes per-literal results to a CSV.

Usage:
    python sampling/run_experiment.py --config config_sss.json
    python sampling/run_experiment.py --config config_sss.json --max_models 20 --workers 4
"""

import argparse
import glob
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.byProgramSampling import Programs
from sampling.byWorldSampling import Worlds
from utils.utils import natural_key


def load_json(path):
    with open(path) as f:
        return json.load(f)


def exact_path_for(model_path):
    base = os.path.dirname(model_path)
    name = os.path.basename(model_path)[:-5]
    return os.path.join(base, "exact", f"{name}_e_w.json")


def _parse_rule(rule_str):
    if " -< " in rule_str:
        rule_type = "defeasible"
        sep = " -< "
    elif " <- " in rule_str:
        rule_type = "strict"
        sep = " <- "
    else:
        return None, None, []
    head, rest = rule_str.split(sep, 1)
    body_lits = [b.strip() for b in rest.rstrip(";").strip().split(",") if b.strip()]
    return head.strip(), rule_type, body_lits


def _complement(lit):
    return lit[1:] if lit.startswith("~") else f"~{lit}"


def _is_trivial(annot_str):
    return annot_str.strip() in {"", "True", "not True"}


def model_features(model_path):
    data = load_json(model_path)
    em_var = data["em_var"]
    n_rules = len(data["af"])
    n_annots = sum(1 for r in data["af"] if not _is_trivial(r[1]))
    return {
        "em_var": em_var,
        "n_annots": n_annots,
        "n_rules": n_rules,
        "annots_minus_em": n_annots - em_var,
        "n_worlds": 2 ** em_var,
        "n_programs": 2 ** n_annots,
    }


def literal_features(literal, data):
    af = data["af"]
    complement = _complement(literal)
    lit_head_def = lit_head_strict = lit_body_count = 0
    lit_is_fact = lit_is_ann_fact = False
    lit_complement_body = lit_complement_head = 0
    body_sizes, annot_vars_list, annot_connector_list, n_annotated = [], [], [], 0

    for rule_str, annot_str in af:
        head, rule_type, body_lits = _parse_rule(rule_str)
        if head is None:
            continue
        body_is_true = body_lits == ["true"]
        has_annot = not _is_trivial(annot_str)

        if has_annot:
            n_annotated += 1
            annot_vars_list.append(len(re.findall(r"\d+", annot_str)))
            annot_connector_list.append(len(re.findall(r"\b(and|or|not)\b", annot_str)))
        if not body_is_true:
            body_sizes.append(len(body_lits))

        if head == literal:
            if rule_type == "defeasible":
                lit_head_def += 1
            else:
                lit_head_strict += 1
            if body_is_true:
                lit_is_ann_fact = has_annot
                lit_is_fact = not has_annot
        if head == complement:
            lit_complement_head += 1
        if literal in body_lits:
            lit_body_count += 1
        if complement in body_lits:
            lit_complement_body += 1

    n_rules = len(af)
    return {
        "lit_is_negated": int(literal.startswith("~")),
        "lit_head_def": lit_head_def,
        "lit_head_strict": lit_head_strict,
        "lit_body_count": lit_body_count,
        "lit_is_fact": int(lit_is_fact),
        "lit_is_ann_fact": int(lit_is_ann_fact),
        "lit_complement_body": lit_complement_body,
        "lit_complement_head": lit_complement_head,
        "af_pct_annotated": n_annotated / n_rules if n_rules > 0 else 0.0,
        "af_avg_annot_vars": sum(annot_vars_list) / len(annot_vars_list) if annot_vars_list else 0.0,
        "af_avg_connectors": sum(annot_connector_list) / len(annot_connector_list) if annot_connector_list else 0.0,
        "af_avg_body_size": sum(body_sizes) / len(body_sizes) if body_sizes else 0.0,
        "af_max_body_size": max(body_sizes) if body_sizes else 0,
    }


def quality_metric(approx_l, approx_u, exact_l, exact_u):
    remainder = 1 - (exact_u - exact_l)
    if remainder == 0:
        return 0.0
    return (1 - (approx_u - approx_l)) / remainder


def run_one_model(args):
    """
    Worker function. Accepts a single tuple so it works with ProcessPoolExecutor.
    Returns (model_name, rows, error_msg).
    """
    model_path, exact_path, save_path, time_limit, config_name = args
    model_name = os.path.basename(model_path)[:-5]

    try:
        exact_data = load_json(exact_path)
        model_data = load_json(model_path)

        literals = [
            lit for lit, s in exact_data["status"].items()
            if s.get("flag") == "INTEREST"
        ]
        if not literals:
            return model_name, [], None

        feats = model_features(model_path)
        os.makedirs(save_path, exist_ok=True)

        # World Sampling
        print(f"  [Worlds]   {model_name} ...", flush=True)
        sampler_w = Worlds(model_path, save_path + "/")
        sampler_w.utils.get_interest_lit = lambda: literals
        try:
            sampler_w.start_sampling(percentile_samples=0, source="random",
                                     info="_w", max_seconds=time_limit)
            w_status = sampler_w.results["status"]
        except Exception as e:
            print(f"    ERROR Worlds {model_name}: {e}", flush=True)
            w_status = {}

        # Program Sampling
        print(f"  [Programs] {model_name} ...", flush=True)
        sampler_p = Programs(model_path, save_path + "/")
        sampler_p.utils.get_interest_lit = lambda: literals
        try:
            sampler_p.start_sampling(percentile_samples=0, info="_p",
                                     max_seconds=time_limit)
            p_status = sampler_p.results["status"]
        except Exception as e:
            print(f"    ERROR Programs {model_name}: {e}", flush=True)
            p_status = {}

        rows = []
        for lit in literals:
            ex = exact_data["status"][lit]
            ex_l = ex.get("l", ex.get("pyes", 0.0))
            ex_u = ex.get("u", 1 - ex.get("pno", 0.0))

            row = {"config": config_name, "model": model_name, "literal": lit,
                   **feats, "exact_l": ex_l, "exact_u": ex_u,
                   "exact_width": ex_u - ex_l}

            if lit in w_status:
                ws = w_status[lit]
                w_l = ws.get("l", ws.get("pyes", 0.0))
                w_u = ws.get("u", 1 - ws.get("pno", 0.0))
                row.update({"worlds_l": w_l, "worlds_u": w_u,
                            "worlds_quality": quality_metric(w_l, w_u, ex_l, ex_u),
                            "worlds_solver_time": ws.get("time", 0.0),
                            "worlds_delp_calls": ws.get("delp_calls", 0)})
            else:
                row.update({"worlds_l": None, "worlds_u": None,
                            "worlds_quality": None,
                            "worlds_solver_time": None, "worlds_delp_calls": None})

            if lit in p_status:
                ps = p_status[lit]
                p_l = ps.get("l", ps.get("pyes", 0.0))
                p_u = ps.get("u", 1 - ps.get("pno", 0.0))
                row.update({"progs_l": p_l, "progs_u": p_u,
                            "progs_quality": quality_metric(p_l, p_u, ex_l, ex_u),
                            "progs_solver_time": ps.get("time", 0.0),
                            "progs_delp_calls": ps.get("delp_calls", 0)})
            else:
                row.update({"progs_l": None, "progs_u": None,
                            "progs_quality": None,
                            "progs_solver_time": None, "progs_delp_calls": None})

            wq = row["worlds_quality"]
            pq = row["progs_quality"]
            if wq is not None and pq is not None:
                row["winner"] = "worlds" if wq > pq else ("programs" if pq > wq else "tie")
            else:
                row["winner"] = None

            row.update(literal_features(lit, model_data))
            rows.append(row)

        return model_name, rows, None

    except Exception as e:
        return model_name, [], str(e)


def run_config(input_path, output_path, time_limit, config_name,
               max_models=None, workers=1):

    all_models = sorted(glob.glob(os.path.join(input_path, "*model.json")),
                        key=natural_key)
    models_with_exact = [m for m in all_models if os.path.exists(exact_path_for(m))]
    if max_models is not None:
        models_with_exact = models_with_exact[:max_models]

    print(f"\n{'='*60}")
    print(f"Config: {config_name}  |  Models: {len(models_with_exact)}  |  Workers: {workers}")
    print(f"Time limit: {time_limit}s per literal")
    print(f"{'='*60}\n")

    csv_path = os.path.join(output_path, f"{config_name}_results.csv")
    write_header = not os.path.exists(csv_path)
    completed = 0
    all_rows = []

    job_args = [
        (m, exact_path_for(m),
         os.path.join(output_path, os.path.basename(m)[:-5]),
         time_limit, config_name)
        for m in models_with_exact
    ]

    if workers == 1:
        # Sequential — simpler, easier to debug
        for i, args in enumerate(job_args):
            model_name = os.path.basename(args[0])[:-5]
            print(f"[{i+1}/{len(job_args)}] {model_name}")
            name, rows, err = run_one_model(args)
            if err:
                print(f"  SKIPPED {name}: {err}")
            elif rows:
                df_chunk = pd.DataFrame(rows)
                df_chunk.to_csv(csv_path, mode="a", header=write_header,
                                index=False, sep=";")
                write_header = False
                all_rows.extend(rows)
                completed += 1
                print(f"  -> {len(rows)} literal(s) saved  [{completed}/{len(job_args)} done]")
    else:
        # Parallel — models run concurrently, CSV written as each finishes
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(run_one_model, a): a for a in job_args}
            total = len(futures)
            for future in as_completed(futures):
                name, rows, err = future.result()
                completed += 1
                if err:
                    print(f"  [{completed}/{total}] SKIPPED {name}: {err}", flush=True)
                elif not rows:
                    print(f"  [{completed}/{total}] {name} — no interesting literals", flush=True)
                else:
                    df_chunk = pd.DataFrame(rows)
                    df_chunk.to_csv(csv_path, mode="a", header=write_header,
                                    index=False, sep=";")
                    write_header = False
                    all_rows.extend(rows)
                    print(f"  [{completed}/{total}] {name} done — {len(rows)} literal(s) saved",
                          flush=True)

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run World vs Program sampling experiments and save CSV."
    )
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON file")
    parser.add_argument("--max_models", type=int, default=None,
                        help="Max number of models to process (default: all)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default: 1). Use 4 on icic-server.")
    args = parser.parse_args()

    cfg = load_json(args.config)
    input_path  = cfg["input_path"]
    output_path = cfg["output_path"]
    time_limit  = cfg["time_limit"]
    config_name = os.path.basename(os.path.normpath(input_path))

    os.makedirs(output_path, exist_ok=True)

    rows = run_config(input_path, output_path, time_limit, config_name,
                      args.max_models, args.workers)

    csv_path = os.path.join(output_path, f"{config_name}_results.csv")
    if rows:
        df = pd.DataFrame(rows)
        print(f"\nResults saved to: {csv_path}")
        try:
            print(f"Total rows: {len(df)}  ({df['model'].nunique()} models)")
            print(f"\nWinner distribution:\n{df['winner'].value_counts().to_string()}")
            df["quality_diff"] = df["worlds_quality"] - df["progs_quality"]
            print(f"\nQuality diff (worlds − programs):")
            print(f"  mean : {df['quality_diff'].mean():+.4f}")
            print(f"  std  : {df['quality_diff'].std():.4f}")
        except Exception as e:
            print(f"  (summary unavailable: {e})")
    else:
        print("No results generated.")
