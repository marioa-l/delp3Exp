"""
Experiment runner: compare World Sampling vs Program Sampling for a given
model configuration.

For each model that has an exact file, runs both sampling approaches with the
configured time limit per literal, then writes per-literal results to a CSV.

Output CSV columns:
  config, model, literal,
  em_var, n_annots, n_rules, annots_minus_em, n_worlds, n_programs,
  exact_l, exact_u, exact_width,
  worlds_l, worlds_u, worlds_quality, worlds_solver_time, worlds_delp_calls,
  progs_l,  progs_u,  progs_quality,  progs_solver_time,  progs_delp_calls,
  winner,
  -- literal features --
  lit_is_negated, lit_head_def, lit_head_strict, lit_body_count,
  lit_is_fact, lit_is_ann_fact, lit_complement_body, lit_complement_head,
  -- program (AF) features --
  af_pct_annotated, af_avg_annot_vars, af_avg_connectors, af_avg_body_size,
  af_max_body_size

Usage:
    cd clean_workspace
    python sampling/run_experiment.py --config config.json
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.byProgramSampling import Programs
from sampling.byWorldSampling import Worlds
from utils.utils import is_trivial_annot, natural_key


def load_json(path):
    with open(path) as f:
        return json.load(f)


def exact_path_for(model_path):
    """Return path to the exact file (no NEW suffix)."""
    base = os.path.dirname(model_path)
    name = os.path.basename(model_path)[:-5]  # strip .json
    return os.path.join(base, "exact", f"{name}_e_w.json")


def _parse_rule(rule_str):
    """
    Parse a rule string into (head, rule_type, body_literals).

    rule_str examples:
        "a_7 -< ~d_15,~a_4,d_18;"
        "a_0 <- true;"
        "~a_8 -< ~a_4,~d_9;"
    Returns:
        head       : str  e.g. "a_7"
        rule_type  : str  "defeasible" | "strict"
        body_lits  : list[str]  e.g. ["~d_15", "~a_4", "d_18"]
    """
    if " -< " in rule_str:
        rule_type = "defeasible"
        sep = " -< "
    elif " <- " in rule_str:
        rule_type = "strict"
        sep = " <- "
    else:
        return None, None, []

    head, rest = rule_str.split(sep, 1)
    head = head.strip()
    body_str = rest.rstrip(";").strip()
    body_lits = [b.strip() for b in body_str.split(",") if b.strip()]
    return head, rule_type, body_lits


def _complement(lit):
    """Return the complement of a literal."""
    return lit[1:] if lit.startswith("~") else f"~{lit}"


def _annot_vars(annot_str):
    """Count BN variable indices referenced in an annotation formula."""
    return len(re.findall(r"\d+", annot_str))


def _annot_connectors(annot_str):
    """Count logical connectors (and / or / not) in an annotation formula."""
    tokens = re.findall(r"\b(and|or|not)\b", annot_str)
    return len(tokens)


def _is_trivial(annot_str):
    """True if annotation is empty, 'True', or 'not True'."""
    return annot_str.strip() in {"", "True", "not True"}


def model_features(model_path):
    """Structural features derivable from the model JSON alone."""
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
    """
    Per-literal and AF-global features extracted from the model JSON.

    Parameters
    ----------
    literal : str   e.g. "a_7" or "~a_8"
    data    : dict  parsed model JSON

    Returns
    -------
    dict with all literal + AF features
    """
    af = data["af"]
    complement = _complement(literal)

    # --- literal-level counters ---
    lit_head_def = 0
    lit_head_strict = 0
    lit_body_count = 0
    lit_is_fact = False
    lit_is_ann_fact = False
    lit_complement_body = 0
    lit_complement_head = 0

    # --- AF-global accumulators ---
    body_sizes = []
    annot_vars_list = []
    annot_connector_list = []
    n_annotated = 0

    for rule_str, annot_str in af:
        head, rule_type, body_lits = _parse_rule(rule_str)
        if head is None:
            continue

        body_is_true = body_lits == ["true"]
        has_annot = not _is_trivial(annot_str)

        # AF global features
        if has_annot:
            n_annotated += 1
            annot_vars_list.append(_annot_vars(annot_str))
            annot_connector_list.append(_annot_connectors(annot_str))

        if not body_is_true:
            body_sizes.append(len(body_lits))

        # Literal head features
        if head == literal:
            if rule_type == "defeasible":
                lit_head_def += 1
            else:
                lit_head_strict += 1
            if body_is_true:
                if not has_annot:
                    lit_is_fact = True
                else:
                    lit_is_ann_fact = True

        # Complement head
        if head == complement:
            lit_complement_head += 1

        # Body appearances
        if literal in body_lits:
            lit_body_count += 1
        if complement in body_lits:
            lit_complement_body += 1

    n_rules = len(af)
    af_pct_annotated = n_annotated / n_rules if n_rules > 0 else 0.0
    af_avg_annot_vars = sum(annot_vars_list) / len(annot_vars_list) if annot_vars_list else 0.0
    af_avg_connectors = sum(annot_connector_list) / len(annot_connector_list) if annot_connector_list else 0.0
    af_avg_body_size = sum(body_sizes) / len(body_sizes) if body_sizes else 0.0
    af_max_body_size = max(body_sizes) if body_sizes else 0

    return {
        # literal features
        "lit_is_negated": int(literal.startswith("~")),
        "lit_head_def": lit_head_def,
        "lit_head_strict": lit_head_strict,
        "lit_body_count": lit_body_count,
        "lit_is_fact": int(lit_is_fact),
        "lit_is_ann_fact": int(lit_is_ann_fact),
        "lit_complement_body": lit_complement_body,
        "lit_complement_head": lit_complement_head,
        # AF global features
        "af_pct_annotated": af_pct_annotated,
        "af_avg_annot_vars": af_avg_annot_vars,
        "af_avg_connectors": af_avg_connectors,
        "af_avg_body_size": af_avg_body_size,
        "af_max_body_size": af_max_body_size,
    }


def quality_metric(approx_l, approx_u, exact_l, exact_u):
    remainder_exact = 1 - (exact_u - exact_l)
    if remainder_exact == 0:
        return 0.0
    return (1 - (approx_u - approx_l)) / remainder_exact


def run_one_model(model_path, exact_path, save_path, time_limit, config_name):
    """Run both samplers on one model. Returns list of per-literal result dicts."""
    model_name = os.path.basename(model_path)[:-5]
    exact_data = load_json(exact_path)
    model_data = load_json(model_path)

    # Only query literals with an interesting exact interval
    literals = [
        lit for lit, s in exact_data["status"].items()
        if s.get("flag") == "INTEREST"
    ]
    if not literals:
        print(f"    No interesting literals — skipping {model_name}")
        return []

    feats = model_features(model_path)
    os.makedirs(save_path, exist_ok=True)

    # ── World Sampling ────────────────────────────────────────────────────────
    print(f"\n  [Worlds] {model_name} ({time_limit}s/literal) ...")
    sampler_w = Worlds(model_path, save_path + "/")
    sampler_w.utils.get_interest_lit = lambda: literals
    try:
        sampler_w.start_sampling(
            percentile_samples=0,
            source="random",
            info=f"_w",
            max_seconds=time_limit,
        )
        w_status = sampler_w.results["status"]
    except Exception as e:
        print(f"    ERROR (Worlds): {e}")
        w_status = {}

    # ── Program Sampling ──────────────────────────────────────────────────────
    print(f"  [Programs] {model_name} ({time_limit}s/literal) ...")
    sampler_p = Programs(model_path, save_path + "/")
    sampler_p.utils.get_interest_lit = lambda: literals
    try:
        sampler_p.start_sampling(
            percentile_samples=0,
            info=f"_p",
            max_seconds=time_limit,
        )
        p_status = sampler_p.results["status"]
    except Exception as e:
        print(f"    ERROR (Programs): {e}")
        p_status = {}

    # ── Build rows ────────────────────────────────────────────────────────────
    rows = []
    for lit in literals:
        ex = exact_data["status"][lit]
        ex_l = ex.get("l", ex.get("pyes", 0.0))
        ex_u = ex.get("u", 1 - ex.get("pno", 0.0))

        lit_feats = literal_features(lit, model_data)

        row = {
            "config": config_name,
            "model": model_name,
            "literal": lit,
            **feats,
            "exact_l": ex_l,
            "exact_u": ex_u,
            "exact_width": ex_u - ex_l,
        }

        if lit in w_status:
            ws = w_status[lit]
            w_l = ws.get("l", ws.get("pyes", 0.0))
            w_u = ws.get("u", 1 - ws.get("pno", 0.0))
            row.update({
                "worlds_l": w_l,
                "worlds_u": w_u,
                "worlds_quality": quality_metric(w_l, w_u, ex_l, ex_u),
                "worlds_solver_time": ws.get("time", 0.0),
                "worlds_delp_calls": ws.get("delp_calls", 0),
            })
        else:
            row.update({
                "worlds_l": None, "worlds_u": None,
                "worlds_quality": None,
                "worlds_solver_time": None, "worlds_delp_calls": None,
            })

        if lit in p_status:
            ps = p_status[lit]
            p_l = ps.get("l", ps.get("pyes", 0.0))
            p_u = ps.get("u", 1 - ps.get("pno", 0.0))
            row.update({
                "progs_l": p_l,
                "progs_u": p_u,
                "progs_quality": quality_metric(p_l, p_u, ex_l, ex_u),
                "progs_solver_time": ps.get("time", 0.0),
                "progs_delp_calls": ps.get("delp_calls", 0),
            })
        else:
            row.update({
                "progs_l": None, "progs_u": None,
                "progs_quality": None,
                "progs_solver_time": None, "progs_delp_calls": None,
            })

        if row["worlds_quality"] is not None and row["progs_quality"] is not None:
            if row["worlds_quality"] > row["progs_quality"]:
                row["winner"] = "worlds"
            elif row["progs_quality"] > row["worlds_quality"]:
                row["winner"] = "programs"
            else:
                row["winner"] = "tie"
        else:
            row["winner"] = None

        row.update(lit_feats)
        rows.append(row)

    return rows


def run_config(input_path, output_path, time_limit, config_name, max_models=None):
    all_models = sorted(
        glob.glob(os.path.join(input_path, "*model.json")), key=natural_key
    )

    models_with_exact = [m for m in all_models if os.path.exists(exact_path_for(m))]
    if max_models is not None:
        models_with_exact = models_with_exact[:max_models]

    print(f"\n{'='*60}")
    print(f"Config: {config_name}  |  Models with exact: {len(models_with_exact)}")
    print(f"Time limit: {time_limit}s per literal")
    print(f"{'='*60}")

    csv_path = os.path.join(output_path, f"{config_name}_results.csv")
    write_header = not os.path.exists(csv_path)

    all_rows = []
    for i, model_path in enumerate(models_with_exact):
        model_name = os.path.basename(model_path)[:-5]
        ex_path = exact_path_for(model_path)
        save_path = os.path.join(output_path, model_name)
        print(f"\n[{i+1}/{len(models_with_exact)}] {model_name}")
        try:
            rows = run_one_model(model_path, ex_path, save_path, time_limit, config_name)
            if rows:
                df_chunk = pd.DataFrame(rows)
                df_chunk.to_csv(csv_path, mode="a", header=write_header, index=False, sep=";")
                write_header = False
                all_rows.extend(rows)
        except Exception as e:
            print(f"  SKIPPED — {e}")

    return all_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run World vs Program sampling experiments and save CSV."
    )
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config.json (default: config.json)"
    )
    parser.add_argument(
        "--max_models", type=int, default=None,
        help="Limit number of models processed (default: all with exact files)"
    )
    args = parser.parse_args()

    cfg = load_json(args.config)
    input_path = cfg["input_path"]
    output_path = cfg["output_path"]
    time_limit = cfg["time_limit"]
    config_name = os.path.basename(os.path.normpath(input_path))

    os.makedirs(output_path, exist_ok=True)

    rows = run_config(input_path, output_path, time_limit, config_name, args.max_models)

    csv_path = os.path.join(output_path, f"{config_name}_results.csv")
    if rows:
        df = pd.read_csv(csv_path, sep=";")
        print(f"\nResults saved incrementally to: {csv_path}")
        print(f"Total rows: {len(df)}  ({df['model'].nunique()} models, {len(df)} literal–model pairs)")
        print(f"\nWinner distribution:\n{df['winner'].value_counts().to_string()}")
        df["quality_diff"] = df["worlds_quality"] - df["progs_quality"]
        print(f"\nQuality diff (worlds − programs):")
        print(f"  mean : {df['quality_diff'].mean():+.4f}")
        print(f"  std  : {df['quality_diff'].std():.4f}")
    else:
        print("No results generated.")
