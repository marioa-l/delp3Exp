"""
Compute complexity features for a DeLP3E model: AM, EM and AF metrics.

These features are STATIC for a given model — they do not depend on which
literal is queried, so they can be cached per (config, model).

AM (DeLP program) metrics — computed by parsing the solver's full output:
    - am_n_arguments        : number of arguments in the dialectical graph
    - am_n_defeaters        : total defeat relations
    - am_n_trees            : number of dialectical trees
    - am_avg_def_rules      : avg defeasible rules per argument (ALE)
    - am_avg_arg_lines      : avg argument lines per tree (AWI)
    - am_avg_height_lines   : avg height of argument lines (AHE)

EM (Bayesian Network) metrics — derived from the .bifxml file:
    - em_n_arcs             : number of directed arcs
    - em_treewidth          : treewidth of the moralized undirected graph
    - em_avg_in_degree      : average number of parents per node
    - em_max_in_degree      : maximum number of parents per node
    - em_entropy            : entropy via pyAgrum factorisation (sum of H(X|Pa(X)))

AF (Annotation Framework) metrics — already partly computed in the experiment;
adds richer counts:
    - af_n_em_vars_used     : distinct BN vars referenced across all annotations
    - af_avg_complexity     : avg (vars + connectors) per non-trivial annotation
    - af_max_complexity     : max (vars + connectors) per annotation

The functions return None on failure (e.g. binary missing, BN file missing) so
the caller can decide to skip or retry.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
from typing import Optional

import networkx as nx
from networkx.algorithms.approximation.treewidth import treewidth_min_degree


# ── Binary selection (matches consultDeLP.py) ─────────────────────────────────
_BINARY = ('delp/coremac' if platform.system() == 'Darwin'
           else 'delp/globalCore')


def _is_trivial(annot: str) -> bool:
    return annot.strip() in {"", "True", "not True"}


def _model_to_delp_text(model_data: dict) -> str:
    """Produce a Prolog-readable .delp file text from the model JSON."""
    rules = [r[0].rstrip(';') + '.' for r in model_data["af"]]
    return "\n".join(rules) + "\nuse_criterion(more_specific).\n"


# ── AM metrics ────────────────────────────────────────────────────────────────
def _count_lines(root_id: int, lines: list, level: int, heights: list) -> int:
    children = [d[3] for d in lines if d[2] == root_id]
    if not children:
        heights.append(level)
        return 1
    return sum(_count_lines(c, lines, level + 1, heights) for c in children)


def _parse_am_metrics(result: dict) -> dict:
    """Compute AM metrics from the solver's `status`+`dGraph` JSON output."""
    n_arguments = 0
    n_defeaters = 0
    n_def_rules = 0
    presumption_re = re.compile(r'\[\([a-zA-Z_0-9~]*\-\<[tT]rue\)\]')

    for entry in result.get('dGraph', []):
        lit_key = next(iter(entry))
        for argument in entry[lit_key]:
            arg_key = next(iter(argument))
            if ',' not in arg_key:
                continue
            subs = argument[arg_key].get('subarguments', [])
            if not subs:
                continue
            presumptions = sum(1 for s in subs if presumption_re.match(s))
            n_arguments += 1
            n_def_rules += len(subs) - presumptions
            n_defeaters += len(argument[arg_key].get('defeats', []))

    avg_def_rules = (n_def_rules / n_arguments) if n_arguments else 0.0

    n_arg_lines = 0
    tree_numbers = 0
    heights: list = []

    for entry in result.get('status', []):
        lit_key = next(iter(entry))
        trees = entry[lit_key].get('trees', [])
        roots = [t for t in trees if len(t) == 2]
        lines = [t for t in trees if len(t) == 4]
        if not lines:
            continue
        for root in roots:
            if '-<' not in root[0]:
                continue
            children = [d[3] for d in lines if d[2] == root[1]]
            if children:
                n_arg_lines += _count_lines(root[1], lines, 0, heights)
                tree_numbers += 1

    avg_height_lines = (sum(heights) / n_arg_lines) if n_arg_lines else 0.0
    avg_arg_lines = (n_arg_lines / tree_numbers) if tree_numbers else 0.0

    return {
        'am_n_arguments': n_arguments,
        'am_n_defeaters': n_defeaters,
        'am_n_trees': tree_numbers,
        'am_avg_def_rules': round(avg_def_rules, 3),
        'am_avg_arg_lines': round(avg_arg_lines, 3),
        'am_avg_height_lines': round(avg_height_lines, 3),
    }


def compute_am_metrics(model_path: str, timeout: int = 900,
                       save_raw_to: Optional[str] = None) -> Optional[dict]:
    """
    Run the DeLP solver on the program and extract AM complexity metrics.

    If `save_raw_to` is given, the full solver JSON output (status + dGraph)
    is also written to that path so future feature extraction can avoid
    re-running the solver.
    """
    with open(model_path) as f:
        model_data = json.load(f)
    delp_text = _model_to_delp_text(model_data)

    tmp_path = f"/tmp/_metrics_{os.getpid()}.delp"
    with open(tmp_path, 'w') as f:
        f.write(delp_text)

    try:
        proc = subprocess.run([_BINARY, 'file', tmp_path, 'all'],
                              capture_output=True, timeout=timeout)
        if proc.returncode != 0:
            return None
        result = json.loads(proc.stdout.decode('ascii', errors='replace'))
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if save_raw_to is not None:
        os.makedirs(os.path.dirname(save_raw_to), exist_ok=True)
        with open(save_raw_to, 'w') as f:
            json.dump(result, f)

    return _parse_am_metrics(result)


# ── EM metrics ────────────────────────────────────────────────────────────────
_ENUMERATE_MAX_WORLDS = 2 ** 18  # exact enumeration up to 262 144 worlds
_SAMPLE_SIZE          = 20000     # Monte Carlo sample size for larger BNs


def _compute_bn_entropy(bn, gum, n_nodes: int) -> float:
    """
    Compute (or estimate) the joint entropy of the Bayesian Network.
    Exact enumeration when feasible, otherwise sample-based estimate.
    """
    import math
    import itertools as it

    ie = gum.LazyPropagation(bn)
    ie.makeInference()
    nodes = list(bn.nodes())

    def joint_prob(values):
        ev = {nodes[i]: int(values[i]) for i in range(len(nodes))}
        ie.setEvidence(ev)
        try:
            return ie.evidenceProbability()
        except Exception:
            return 0.0

    if 2 ** n_nodes <= _ENUMERATE_MAX_WORLDS:
        h = 0.0
        for combo in it.product([1, 0], repeat=n_nodes):
            p = joint_prob(combo)
            if p > 0:
                h -= p * math.log2(p)
        return round(h, 4)

    # Sample-based estimate: H ≈ -E[log2 p(X)]  via samples drawn from the BN
    try:
        gen = gum.BNDatabaseGenerator(bn)
        gen.drawSamples(_SAMPLE_SIZE)
        h_sum = 0.0
        used = 0
        for i in range(_SAMPLE_SIZE):
            ev = {n: gen.samplesAt(i, j) for j, n in enumerate(bn.nodes())}
            ie.setEvidence(ev)
            try:
                p = ie.evidenceProbability()
            except Exception:
                continue
            if p > 0:
                h_sum -= math.log2(p)
                used += 1
        return round(h_sum / used, 4) if used else float('nan')
    except Exception:
        return float('nan')


def compute_em_metrics(bn_path: str) -> Optional[dict]:
    """
    Load the .bifxml file and compute structural + entropy metrics.
    Imports pyAgrum lazily — if not installed, returns only graph-based metrics.
    """
    try:
        import pyAgrum as gum  # type: ignore
    except ImportError:
        gum = None

    if gum is None or not os.path.exists(bn_path):
        return None

    try:
        bn = gum.loadBN(bn_path)
        nodes = list(bn.nodes())
        arcs = list(bn.arcs())
    except Exception:
        return None

    n_arcs = len(arcs)
    in_degrees = [len(bn.parents(n)) for n in nodes]
    avg_in = sum(in_degrees) / len(in_degrees) if in_degrees else 0.0
    max_in = max(in_degrees) if in_degrees else 0

    # Treewidth on moralised undirected graph
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for a, b in arcs:
        G.add_edge(a, b)
    # Moralise: connect co-parents
    for n in nodes:
        parents = list(bn.parents(n))
        for i in range(len(parents)):
            for j in range(i + 1, len(parents)):
                G.add_edge(parents[i], parents[j])
    try:
        tw, _ = treewidth_min_degree(G)
    except Exception:
        tw = -1

    # Entropy: enumerate worlds for small BNs, otherwise Monte Carlo estimate.
    em_entropy = _compute_bn_entropy(bn, gum, n_nodes=len(nodes))

    return {
        'em_n_arcs': n_arcs,
        'em_treewidth': tw,
        'em_avg_in_degree': round(avg_in, 3),
        'em_max_in_degree': max_in,
        'em_entropy': em_entropy,
    }


# ── AF metrics (additional to those already in the dataset) ───────────────────
def compute_af_extra(model_data: dict) -> dict:
    """
    Richer AF metrics not already present in the experiment CSV.
    """
    em_vars_used = set()
    complexities = []

    for _, annot in model_data["af"]:
        if _is_trivial(annot):
            continue
        vars_in = re.findall(r'\d+', annot)
        connectors = re.findall(r'\b(and|or|not)\b', annot)
        em_vars_used.update(vars_in)
        complexities.append(len(vars_in) + len(connectors))

    return {
        'af_n_em_vars_used': len(em_vars_used),
        'af_avg_complexity': (round(sum(complexities) / len(complexities), 3)
                              if complexities else 0.0),
        'af_max_complexity': max(complexities) if complexities else 0,
    }


# ── Top-level convenience wrapper ─────────────────────────────────────────────
def compute_all(model_path: str, bn_path: Optional[str] = None,
                save_raw_to: Optional[str] = None) -> dict:
    """
    Compute all complexity features for a single model.

    `bn_path` defaults to <dir>/BN<N>.bifxml beside the model.
    `save_raw_to`, if set, is the path where the raw solver JSON output is
    written so future feature extraction can avoid calling the solver again.
    Missing data is filled with NaN-like sentinels.
    """
    with open(model_path) as f:
        model_data = json.load(f)

    if bn_path is None:
        d = os.path.dirname(model_path)
        n = re.search(r'(\d+)model\.json$', os.path.basename(model_path))
        if n:
            bn_path = os.path.join(d, f"BN{n.group(1)}.bifxml")

    out = {}

    am = compute_am_metrics(model_path, save_raw_to=save_raw_to)
    if am is None:
        am = {k: None for k in (
            'am_n_arguments', 'am_n_defeaters', 'am_n_trees',
            'am_avg_def_rules', 'am_avg_arg_lines', 'am_avg_height_lines')}
    out.update(am)

    em = compute_em_metrics(bn_path) if bn_path else None
    if em is None:
        em = {k: None for k in (
            'em_n_arcs', 'em_treewidth', 'em_avg_in_degree',
            'em_max_in_degree', 'em_entropy')}
    out.update(em)

    out.update(compute_af_extra(model_data))
    return out
