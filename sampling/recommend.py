"""
Sampling-method recommenders.

Two independent classifiers are exposed. Both take a dict of features for
one (model, literal) pair and return ``(method, region_label)`` where
``method`` is ``"worlds"`` or ``"programs"`` and ``region_label`` names the
branch of the tree that produced the decision.

    recommend_comparison(features)
        Hardcoded rules of the depth-3 comparison tree
        (sampling/comparison_tree.py). No external artefact needed.

    recommend_abstract(features, pkl_path)
        Loads the pickle produced by abstract_tree.py (with the fitted
        tree, encoders and numeric bin thresholds) and applies it to the
        features exactly as during training.

Both recommenders share the same input contract, so they can be used
interchangeably by validate_trees.py and by the heuristic sampler.
"""

from __future__ import annotations

import pickle
from typing import Any


# ── Comparison tree (depth 3) ────────────────────────────────────────────────
def recommend_comparison(features: dict) -> tuple[str, str]:
    """
    Apply the four rules of the depth-3 comparison tree.

    Rule regions:
        A       — Worlds  (n=901, purity 100%)
        B       — Programs (n=352, purity 99%)
        C       — Worlds  (n=930, purity 72%)
        D       — Programs (n=148, purity 68%)
        A_tiny  — noise leaves under the R1 branch (Programs, n=2)
        B_tiny  — noise leaf under the R2 branch (Worlds,  n=4)

    The four ``*_tiny`` leaves are also returned by their proper
    ``method`` label because that's what the trained tree emits.
    """
    def g(k):  # tolerant getter — allows None to short-circuit
        v = features.get(k)
        return None if v is None else float(v)

    af_body   = g("af_avg_body_size")
    af_ratio  = g("af_rule_to_var_ratio")
    em_var    = g("em_var")
    n_annots  = g("n_annots")
    em_clust  = g("em_clustering_coef")
    em_deg_std = g("em_deg_centrality_std")
    am_close  = g("am_attack_closeness_avg")
    em_dens   = g("em_graph_density")
    am_args   = g("am_n_arguments")
    em_ent    = g("em_entropy")
    af_var_use = g("af_avg_var_usage")
    af_pct    = g("af_pct_annotated")
    am_gini   = g("am_attack_in_degree_gini")

    if af_body is None or af_ratio is None:
        return ("worlds", "unknown")

    if af_body < af_ratio:
        # Left branch of the root
        if em_var is None or n_annots is None:
            return ("worlds", "A")
        if em_var < n_annots:
            # Both children of this node recommend WORLDS regardless of
            # the am_attack_closeness_avg vs em_graph_density test.
            return ("worlds", "A")
        else:
            # em_var >= n_annots — two tiny noise leaves.
            if em_clust is None or em_deg_std is None:
                return ("worlds", "A_tiny")
            if em_clust < em_deg_std:
                return ("worlds", "A_tiny")
            else:
                return ("programs", "A_tiny")
    else:
        # Right branch of the root
        if am_args is None or em_ent is None:
            return ("programs", "B")
        if am_args < em_ent:
            if af_body < (af_var_use if af_var_use is not None else 0):
                return ("worlds", "B_tiny")
            return ("programs", "B")
        else:
            if af_pct is None or am_gini is None:
                return ("worlds", "C")
            if af_pct < am_gini:
                return ("worlds", "C")
            else:
                return ("programs", "D")


# ── Abstract tree (from pickle) ──────────────────────────────────────────────
class AbstractRecommender:
    """
    Loads the pickle emitted by abstract_tree.py and applies the same
    binning + prediction pipeline to new rows.

    The pickle contains: ``tree`` (fitted DecisionTreeClassifier),
    ``encoders`` (dict feat -> ordered list of categorical values),
    ``bins_numeric`` (dict feat -> {method, bins, labels}),
    ``predictors_categorical`` (list of columns the tree expects),
    ``feature_columns`` (same as predictors_categorical, kept for
    forward compatibility).
    """

    def __init__(self, pkl_path: str):
        with open(pkl_path, "rb") as f:
            state = pickle.load(f)
        self.tree = state["tree"]
        self.encoders = state["encoders"]
        self.bins_numeric = state["bins_numeric"]
        self.predictors = state["predictors_categorical"]
        self.n_bins = state["n_bins"]

    # ---- Derived / sign features (kept in sync with abstract_tree.py) ----
    @staticmethod
    def _safe_div(num, den):
        try:
            if den is None or den == 0:
                return None
            return float(num) / float(den)
        except (TypeError, ValueError):
            return None

    def _derived(self, r: dict) -> dict:
        out = dict(r)
        out["em_density"]           = self._safe_div(r.get("em_n_arcs"), r.get("em_var"))
        out["em_entropy_per_var"]   = self._safe_div(r.get("em_entropy"), r.get("em_var"))
        out["em_treewidth_ratio"]   = self._safe_div(r.get("em_treewidth"), r.get("em_var"))
        out["em_max_indeg_ratio"]   = self._safe_div(r.get("em_max_in_degree"), r.get("em_var"))
        out["am_arg_density"]       = self._safe_div(r.get("am_n_arguments"), r.get("n_rules"))
        out["am_defeat_ratio"]      = self._safe_div(r.get("am_n_defeaters"), r.get("am_n_arguments"))
        out["am_tree_density"]      = self._safe_div(r.get("am_n_trees"), r.get("am_n_arguments"))
        out["af_em_coverage"]       = self._safe_div(r.get("af_n_em_vars_used"), r.get("em_var"))
        out["lit_head_def_rate"]    = self._safe_div(r.get("lit_head_def"), r.get("n_rules"))
        out["lit_head_strict_rate"] = self._safe_div(r.get("lit_head_strict"), r.get("n_rules"))
        out["lit_body_rate"]        = self._safe_div(r.get("lit_body_count"), r.get("n_rules"))
        out["lit_complement_body_rate"] = self._safe_div(r.get("lit_complement_body"), r.get("n_rules"))
        out["lit_complement_head_rate"] = self._safe_div(r.get("lit_complement_head"), r.get("n_rules"))
        # Sign feature: space orientation
        v = r.get("annots_minus_em")
        if v is None:
            out["space_orientation"] = "unknown"
        elif v < -1:
            out["space_orientation"] = "programs-favored"
        elif v > 1:
            out["space_orientation"] = "worlds-favored"
        else:
            out["space_orientation"] = "balanced"
        return out

    def _bin_value(self, col: str, value: Any) -> str:
        info = self.bins_numeric.get(col)
        if info is None or info["method"] in {"constant", "all-NaN", "failed"}:
            return "medium"
        bins = info["bins"]
        labels = info["labels"]
        if value is None:
            return "medium"
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "medium"
        for i in range(1, len(bins)):
            if v <= bins[i]:
                return labels[i - 1]
        return labels[-1]

    def _encode(self, col: str, cat: str) -> int:
        order = self.encoders.get(col, [])
        if cat in order:
            return order.index(cat)
        if "missing" in order:
            return order.index("missing")
        return 0

    def predict(self, features: dict) -> tuple[str, str]:
        """Return (method, region_label) where region_label is 'abs_leaf_<id>'."""
        r = self._derived(features)
        # Bin every predictor into its categorical label
        x = []
        for col in self.predictors:
            raw = r.get(col)
            # Binary/categorical features pass through if already a string,
            # numeric ones go through _bin_value.
            if col in ("lit_is_negated", "lit_is_fact", "lit_is_ann_fact"):
                cat = "medium" if raw is None else str(int(raw))
            elif col == "space_orientation":
                cat = raw or "unknown"
            else:
                cat = self._bin_value(col, raw)
            x.append(self._encode(col, cat))
        pred = int(self.tree.predict([x])[0])
        # Also identify the leaf so we can report region purity later.
        leaf_id = int(self.tree.apply([x])[0])
        method = "worlds" if pred == 1 else "programs"
        return (method, f"abs_leaf_{leaf_id}")
