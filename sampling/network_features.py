"""
Network / graph-theoretic features for DeLP3E models.

For each of the three components (EM, AM, AF) we view the structure as a
graph and compute centrality and global descriptors. All metrics aim to be
SCALE-INVARIANT so that they generalise across model sizes and across
different generators.

Public functions:
    compute_em_network_metrics(bn_path)
    compute_am_network_metrics(dgraph_path)
    compute_af_network_metrics(model_data)

Each returns a dict (or None on failure).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from statistics import mean, pstdev
from typing import Optional

import networkx as nx


# ── Small helpers ─────────────────────────────────────────────────────────────
def _gini(values) -> float:
    """Gini coefficient of a non-negative distribution; 0 = equal, 1 = concentrated."""
    arr = sorted(float(v) for v in values if v >= 0)
    n = len(arr)
    if n == 0 or sum(arr) == 0:
        return 0.0
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(arr, start=1):
        cum += v
        weighted += i * v
    return (2 * weighted) / (n * cum) - (n + 1) / n


def _safe_avg(d) -> float:
    vals = list(d.values()) if isinstance(d, dict) else list(d)
    return float(mean(vals)) if vals else 0.0


def _safe_max(d) -> float:
    vals = list(d.values()) if isinstance(d, dict) else list(d)
    return float(max(vals)) if vals else 0.0


def _safe_std(d) -> float:
    vals = list(d.values()) if isinstance(d, dict) else list(d)
    return float(pstdev(vals)) if len(vals) > 1 else 0.0


# ── EM (Bayesian Network) graph metrics ───────────────────────────────────────
def compute_em_network_metrics(bn_path: str) -> Optional[dict]:
    """Centrality and global descriptors of the BN directed graph."""
    try:
        import pyAgrum as gum  # type: ignore
    except ImportError:
        return None
    if not os.path.exists(bn_path):
        return None
    try:
        bn = gum.loadBN(bn_path)
    except Exception:
        return None

    nodes = list(bn.nodes())
    arcs = list(bn.arcs())
    n = len(nodes)
    if n == 0:
        return None

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(arcs)
    U = G.to_undirected()

    # Density already accounts for n
    density = nx.density(G)

    # Degree centrality (uses normalised in/out degree)
    deg_c = nx.degree_centrality(G)
    in_c  = nx.in_degree_centrality(G)
    out_c = nx.out_degree_centrality(G)

    # Closeness on undirected graph
    try:
        clos_c = nx.closeness_centrality(U)
    except Exception:
        clos_c = {n_: 0.0 for n_ in nodes}

    # Betweenness
    try:
        betw_c = nx.betweenness_centrality(G)
    except Exception:
        betw_c = {n_: 0.0 for n_ in nodes}

    # Eigenvector centrality (may not converge)
    try:
        eig_c = nx.eigenvector_centrality_numpy(G)
    except Exception:
        try:
            eig_c = nx.eigenvector_centrality(U, max_iter=2000)
        except Exception:
            eig_c = {n_: 0.0 for n_ in nodes}

    # Global structural descriptors
    try:
        clustering = nx.average_clustering(U)
    except Exception:
        clustering = 0.0
    # Average shortest path length on the largest weakly connected component
    try:
        if nx.is_connected(U):
            avg_path = nx.average_shortest_path_length(U)
            diameter = nx.diameter(U)
        else:
            largest_cc = max(nx.connected_components(U), key=len)
            sub = U.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(sub) if len(sub) > 1 else 0.0
            diameter = nx.diameter(sub) if len(sub) > 1 else 0
    except Exception:
        avg_path, diameter = 0.0, 0

    return {
        # Global graph descriptors
        "em_graph_density":      round(density, 4),
        "em_clustering_coef":    round(clustering, 4),
        "em_avg_shortest_path":  round(avg_path, 4),
        "em_diameter":           int(diameter),
        # Centrality aggregates (degree)
        "em_deg_centrality_avg": round(_safe_avg(deg_c), 4),
        "em_deg_centrality_max": round(_safe_max(deg_c), 4),
        "em_deg_centrality_std": round(_safe_std(deg_c), 4),
        # Closeness
        "em_closeness_avg":      round(_safe_avg(clos_c), 4),
        "em_closeness_max":      round(_safe_max(clos_c), 4),
        # Betweenness
        "em_betweenness_avg":    round(_safe_avg(betw_c), 4),
        "em_betweenness_max":    round(_safe_max(betw_c), 4),
        # Eigenvector
        "em_eigenvector_avg":    round(_safe_avg(eig_c), 4),
        "em_eigenvector_max":    round(_safe_max(eig_c), 4),
        # Concentration of in-degree (Gini): 0 = uniform, 1 = star-like
        "em_in_degree_gini":     round(_gini(in_c.values()), 4),
    }


# ── AM (DeLP attack graph) metrics ────────────────────────────────────────────
def _parse_attack_graph(dgraph_data: list) -> nx.DiGraph:
    """
    Build the argument attack graph from the solver's dGraph output.

    Nodes  = argument ids (the bracketed string form)
    Edges  = (attacker -> defended)
    """
    G = nx.DiGraph()
    seen = set()
    for entry in dgraph_data:
        lit_key = next(iter(entry))
        for argument in entry[lit_key]:
            arg_key = next(iter(argument))
            arg_data = argument[arg_key]
            arg_id = arg_data.get("id", arg_key)
            if arg_id in seen:
                continue
            seen.add(arg_id)
            G.add_node(arg_id)
            for d in arg_data.get("defeats", []):
                target = d.get("defeat") if isinstance(d, dict) else d
                if target:
                    G.add_edge(arg_id, target)
    return G


def compute_am_network_metrics(dgraph_path: str) -> Optional[dict]:
    """Centrality and global descriptors of the AM attack graph."""
    if not os.path.exists(dgraph_path):
        return None
    try:
        with open(dgraph_path) as f:
            raw = json.load(f)
    except Exception:
        return None

    G = _parse_attack_graph(raw.get("dGraph", []))
    n = G.number_of_nodes()
    if n == 0:
        return {
            "am_attack_density":      0.0,
            "am_attack_clustering":   0.0,
            "am_attack_avg_path":     0.0,
            "am_attack_diameter":     0,
            "am_attack_reciprocity":  0.0,
            "am_attack_deg_avg":      0.0,
            "am_attack_deg_max":      0.0,
            "am_attack_deg_std":      0.0,
            "am_attack_closeness_avg": 0.0,
            "am_attack_closeness_max": 0.0,
            "am_attack_betweenness_avg": 0.0,
            "am_attack_betweenness_max": 0.0,
            "am_attack_eigenvector_avg": 0.0,
            "am_attack_eigenvector_max": 0.0,
            "am_attack_in_degree_gini": 0.0,
        }

    U = G.to_undirected()
    density = nx.density(G)
    reciprocity = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0

    deg_c = nx.degree_centrality(G)
    in_c  = nx.in_degree_centrality(G)
    try:
        clos_c = nx.closeness_centrality(U)
    except Exception:
        clos_c = {nd: 0.0 for nd in G.nodes()}
    try:
        betw_c = nx.betweenness_centrality(G)
    except Exception:
        betw_c = {nd: 0.0 for nd in G.nodes()}
    try:
        eig_c = nx.eigenvector_centrality_numpy(G)
    except Exception:
        eig_c = {nd: 0.0 for nd in G.nodes()}

    try:
        clustering = nx.average_clustering(U)
    except Exception:
        clustering = 0.0
    try:
        if nx.is_connected(U):
            avg_path = nx.average_shortest_path_length(U)
            diameter = nx.diameter(U)
        else:
            largest = max(nx.connected_components(U), key=len)
            sub = U.subgraph(largest)
            avg_path = nx.average_shortest_path_length(sub) if len(sub) > 1 else 0.0
            diameter = nx.diameter(sub) if len(sub) > 1 else 0
    except Exception:
        avg_path, diameter = 0.0, 0

    return {
        "am_attack_density":          round(density, 4),
        "am_attack_clustering":       round(clustering, 4),
        "am_attack_avg_path":         round(avg_path, 4),
        "am_attack_diameter":         int(diameter),
        "am_attack_reciprocity":      round(float(reciprocity), 4),
        "am_attack_deg_avg":          round(_safe_avg(deg_c), 4),
        "am_attack_deg_max":          round(_safe_max(deg_c), 4),
        "am_attack_deg_std":          round(_safe_std(deg_c), 4),
        "am_attack_closeness_avg":    round(_safe_avg(clos_c), 4),
        "am_attack_closeness_max":    round(_safe_max(clos_c), 4),
        "am_attack_betweenness_avg":  round(_safe_avg(betw_c), 4),
        "am_attack_betweenness_max":  round(_safe_max(betw_c), 4),
        "am_attack_eigenvector_avg":  round(_safe_avg(eig_c), 4),
        "am_attack_eigenvector_max":  round(_safe_max(eig_c), 4),
        "am_attack_in_degree_gini":   round(_gini(in_c.values()), 4),
    }


# ── AF (bipartite annotation graph) metrics ───────────────────────────────────
def _is_trivial(annot: str) -> bool:
    return annot.strip() in {"", "True", "not True"}


def compute_af_network_metrics(model_data: dict) -> dict:
    """
    Bipartite graph: annotated rules <-> BN variables referenced in annotations.

    Returns concentration / coverage descriptors on the variable-usage side.
    """
    # rule_idx -> set of BN var indices
    rule_vars: list[set[str]] = []
    var_usage: Counter = Counter()

    for rule_str, annot in model_data.get("af", []):
        if _is_trivial(annot):
            continue
        vars_in = set(re.findall(r"\d+", annot))
        rule_vars.append(vars_in)
        for v in vars_in:
            var_usage[v] += 1

    if not rule_vars:
        return {
            "af_bipartite_density":  0.0,
            "af_var_usage_gini":     0.0,
            "af_var_coverage":       0.0,
            "af_avg_var_usage":      0.0,
            "af_max_var_usage":      0,
            "af_rule_to_var_ratio":  0.0,
        }

    n_rules = len(rule_vars)
    n_vars = len(var_usage)
    em_var = model_data.get("em_var", n_vars)

    # Total possible bipartite edges = n_rules * em_var
    total_edges = sum(len(s) for s in rule_vars)
    bipartite_density = (total_edges / (n_rules * em_var)) if em_var > 0 else 0.0

    return {
        "af_bipartite_density": round(bipartite_density, 4),
        "af_var_usage_gini":    round(_gini(var_usage.values()), 4),
        "af_var_coverage":      round(n_vars / em_var, 4) if em_var > 0 else 0.0,
        "af_avg_var_usage":     round(float(mean(var_usage.values())), 4),
        "af_max_var_usage":     int(max(var_usage.values())),
        "af_rule_to_var_ratio": round(n_rules / n_vars, 4) if n_vars > 0 else 0.0,
    }


# ── Convenience ───────────────────────────────────────────────────────────────
def compute_all_network(model_path: str,
                        bn_path: Optional[str] = None,
                        dgraph_path: Optional[str] = None) -> dict:
    """Compute all network features for one model and return a flat dict."""
    with open(model_path) as f:
        model_data = json.load(f)
    out: dict = {}

    em = compute_em_network_metrics(bn_path) if bn_path else None
    if em is None:
        em = {k: None for k in (
            "em_graph_density", "em_clustering_coef", "em_avg_shortest_path",
            "em_diameter", "em_deg_centrality_avg", "em_deg_centrality_max",
            "em_deg_centrality_std", "em_closeness_avg", "em_closeness_max",
            "em_betweenness_avg", "em_betweenness_max", "em_eigenvector_avg",
            "em_eigenvector_max", "em_in_degree_gini")}
    out.update(em)

    am = compute_am_network_metrics(dgraph_path) if dgraph_path else None
    if am is None:
        am = {k: None for k in (
            "am_attack_density", "am_attack_clustering", "am_attack_avg_path",
            "am_attack_diameter", "am_attack_reciprocity",
            "am_attack_deg_avg", "am_attack_deg_max", "am_attack_deg_std",
            "am_attack_closeness_avg", "am_attack_closeness_max",
            "am_attack_betweenness_avg", "am_attack_betweenness_max",
            "am_attack_eigenvector_avg", "am_attack_eigenvector_max",
            "am_attack_in_degree_gini")}
    out.update(am)

    out.update(compute_af_network_metrics(model_data))
    return out
