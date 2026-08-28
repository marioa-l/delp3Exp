"""
The tree to present: every split compares two model attributes against each
other (``A < B``), never against a numeric threshold, and any question whose
two answers lead to the same recommendation is removed.

Trained and validated on the time-to-exact ground truth — for each query,
both samplers run until one reaches the exact interval, and the winner is
whichever got there first. That criterion does not depend on a chosen
timeout, unlike "best interval within N seconds".

What this script does:
    1. Builds every pairwise comparison A < B over the measurable features.
    2. Fits a comparison-only tree on ``gt_winner``.
    3. Prunes it: a node whose two branches end in the same recommendation
       is replaced by that recommendation, repeatedly until nothing changes.
       This is what shortens the branches without changing a single answer.
    4. Validates with GroupKFold grouped by (config, model), so no literal
       of a model is ever in both train and test, plus leave-one-config-out
       to show how it travels to an unseen configuration.

Writes to --output_dir (default results_tte/analysis/):
    final_tree.pdf          the figure to present
    final_tree_rules.txt    the same tree as natural-language rules
    final_tree.json         portable model (no sklearn needed to apply it)
    final_tree_report.txt   accuracy, per config, pruning effect

Usage:
    python sampling/final_tree.py
    python sampling/final_tree.py --depth 4        # try a deeper tree
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from sampling.comparison_tree import (  # noqa: E402
    select_measurable_features, impute_median, make_pair_comparisons,
)

FONT = "DejaVu Sans"
COLOR_WORLDS = "#0072B2"
COLOR_PROGRAMS = "#D55E00"
COLOR_NODE = "#F2F2F2"
EDGE = "#333333"

# Ground-truth / outcome columns that must never become predictors.
LEAK = {
    "gt_winner", "winner", "winner_binary", "ok", "safety_cap",
    "time_worlds", "time_programs", "worlds_wall_time", "programs_wall_time",
    "worlds_l", "worlds_u", "programs_l", "programs_u",
    "worlds_delp_calls", "programs_delp_calls", "worlds_ok", "programs_ok",
    "exact_l", "exact_u", "exact_width",
    "n_worlds", "n_programs", "annots_minus_em",
}


# ── What each attribute costs to obtain ─────────────────────────────────────
# Tier 0  read straight off the model JSON (counting rules, annotations, vars)
# Tier 1  graph metrics over the BN DAG (networkx on a small graph)
# Tier 2  BN inference / NP-hard structure: entropy needs LazyPropagation,
#         treewidth needs the moralised graph decomposition
# Tier 3  requires running the DeLP solver as a subprocess to build the
#         dialectical graph — by far the most expensive thing here, and the
#         very work the recommendation is supposed to help us avoid
TIER_NAMES = {0: "free (parse JSON)", 1: "cheap (BN graph)",
              2: "costly (BN inference)", 3: "very costly (DeLP solver)"}
# Expensive questions get a loud outline so the cost is visible in the figure.
TIER_EDGE = {0: "#333333", 1: "#333333", 2: "#B7791F", 3: "#B03A2E"}


def feature_tier(name: str) -> int:
    if name.startswith("am_"):
        return 3
    if name in ("em_entropy", "em_treewidth"):
        return 2
    if name.startswith("em_"):
        return 1
    return 0


def pair_tier(col: str) -> int:
    a, b = col.split("_LT_")
    return max(feature_tier(a), feature_tier(b))


def parse_policy(spec: str, max_depth: int) -> list[int]:
    """
    '0,1,2,3' -> at depth 0 only tier<=0 may be used, at depth 1 tier<=1, ...
    The last value is repeated for any deeper level.
    """
    vals = [int(v) for v in spec.split(",") if v.strip() != ""]
    if not vals:
        vals = [3]
    while len(vals) < max_depth:
        vals.append(vals[-1])
    return vals


# ── Cost-aware greedy tree ──────────────────────────────────────────────────
def _gini_impurity(w_pos, w_tot):
    p = np.divide(w_pos, w_tot, out=np.full_like(w_tot, 0.5, dtype=float),
                  where=w_tot > 0)
    return 1.0 - p ** 2 - (1.0 - p) ** 2


def _best_split(X, y, w, idx, cols):
    """Return (col_index, weighted_impurity) of the best allowed split."""
    if len(cols) == 0:
        return None, None
    Xs = X[np.ix_(idx, cols)].astype(np.float64)
    ws = w[idx]
    pos = ws * y[idx]
    w_tot = ws.sum()
    w_right = ws @ Xs
    p_right = pos @ Xs
    w_left = w_tot - w_right
    p_left = pos.sum() - p_right
    imp = (w_left * _gini_impurity(p_left, w_left)
           + w_right * _gini_impurity(p_right, w_right)) / w_tot
    # A split that sends everything one way is useless.
    imp[(w_left <= 0) | (w_right <= 0)] = np.inf
    k = int(np.argmin(imp))
    if not np.isfinite(imp[k]):
        return None, None
    return cols[k], float(imp[k])


def grow_cost_aware(X, y, w, columns, policy, max_depth, min_leaf,
                    idx=None, depth=0):
    """
    Greedy tree where the attributes usable at each level are limited by the
    cost policy, so expensive questions can only appear deeper down. Every
    query decided above them never pays for them at all.
    """
    if idx is None:
        idx = np.arange(len(y))
    n_p = int((y[idx] == 0).sum())
    n_w = int((y[idx] == 1).sum())
    # Weighted majority, so the minority class is not silently swallowed.
    wp = w[idx][y[idx] == 0].sum()
    ww = w[idx][y[idx] == 1].sum()
    leaf = {"kind": "leaf", "cls": "worlds" if ww >= wp else "programs",
            "n_p": n_p, "n_w": n_w}
    if depth >= max_depth or len(idx) < 2 * min_leaf or n_p == 0 or n_w == 0:
        return leaf

    budget = policy[depth] if depth < len(policy) else policy[-1]
    cols = [i for i, c in enumerate(columns) if pair_tier(c) <= budget]
    col, _ = _best_split(X, y, w, idx, cols)
    if col is None:
        return leaf

    mask = X[idx, col] == 1
    right, left = idx[mask], idx[~mask]
    if len(right) < min_leaf or len(left) < min_leaf:
        return leaf

    a, b = columns[col].split("_LT_")
    return {
        "kind": "node", "a": a, "b": b, "tier": pair_tier(columns[col]),
        "yes": grow_cost_aware(X, y, w, columns, policy, max_depth, min_leaf,
                               right, depth + 1),
        "no": grow_cost_aware(X, y, w, columns, policy, max_depth, min_leaf,
                              left, depth + 1),
    }


def cost_profile(node, df):
    """
    For each query, the highest tier actually consulted before a decision was
    reached. This is the number that matters: a query decided high in the tree
    never pays for the expensive attributes below.
    """
    tiers = []
    for row in df.to_dict("records"):
        n, worst = node, 0
        while n["kind"] != "leaf":
            worst = max(worst, n.get("tier", pair_tier(f"{n['a']}_LT_{n['b']}")))
            a, b = row.get(n["a"]), row.get(n["b"])
            went = (a is not None and b is not None
                    and not pd.isna(a) and not pd.isna(b)
                    and float(a) < float(b))
            n = n["yes"] if went else n["no"]
        tiers.append(worst)
    return np.array(tiers)


# ── Plain tree representation (so pruning and export are trivial) ───────────
def to_plain(sk, node_id, predictors, counts):
    """Convert an sklearn node into a plain dict, carrying real class counts."""
    t = sk.tree_
    if t.children_left[node_id] == -1:
        n_p, n_w = counts.get(node_id, (0, 0))
        return {"kind": "leaf", "cls": "worlds" if n_w >= n_p else "programs",
                "n_p": int(n_p), "n_w": int(n_w)}
    a, b = predictors[t.feature[node_id]].split("_LT_")
    # A comparison feature is 1 when A < B; sklearn sends value>threshold right.
    return {
        "kind": "node", "a": a, "b": b,
        "yes": to_plain(sk, t.children_right[node_id], predictors, counts),
        "no": to_plain(sk, t.children_left[node_id], predictors, counts),
    }


def leaf_counts(sk, X, y):
    ids = sk.apply(X)
    out: dict[int, list[int]] = {}
    for lid, label in zip(ids, y):
        c = out.setdefault(int(lid), [0, 0])
        c[1 if label == 1 else 0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def prune(node):
    """
    Collapse any question whose two answers give the same recommendation.

    Applied bottom-up and repeated implicitly by the recursion, so a chain of
    redundant questions disappears in one pass.
    """
    if node["kind"] == "leaf":
        return node
    node["yes"] = prune(node["yes"])
    node["no"] = prune(node["no"])
    y, n = node["yes"], node["no"]
    if y["kind"] == "leaf" and n["kind"] == "leaf" and y["cls"] == n["cls"]:
        return {"kind": "leaf", "cls": y["cls"],
                "n_p": y["n_p"] + n["n_p"], "n_w": y["n_w"] + n["n_w"]}
    return node


def count_nodes(node):
    if node["kind"] == "leaf":
        return 0, 1
    qi, li = count_nodes(node["yes"])
    qj, lj = count_nodes(node["no"])
    return qi + qj + 1, li + lj


def apply_plain(node, row):
    """Walk the plain tree for one feature dict; returns (method, path)."""
    path = []
    while node["kind"] != "leaf":
        a, b = row.get(node["a"]), row.get(node["b"])
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            went = False           # undecidable -> take the ELSE branch
        else:
            went = float(a) < float(b)
        path.append(f"{node['a']} {'<' if went else '>='} {node['b']}")
        node = node["yes"] if went else node["no"]
    return node["cls"], path


def predict_plain(node, X_pair_row_lookup):
    """Vectorised apply over the prebuilt comparison matrix."""
    raise NotImplementedError  # not needed; see predict_frame


def predict_frame(node, df):
    """Predict for a whole DataFrame of raw features."""
    return np.array([apply_plain(node, r)[0]
                     for r in df.to_dict("records")])


# ── Figure ──────────────────────────────────────────────────────────────────
def layout(node, depth=0, x_counter=None, pos=None):
    if x_counter is None:
        x_counter, pos = [0], {}
    node["_id"] = id(node)
    if node["kind"] == "leaf":
        pos[node["_id"]] = (x_counter[0], depth)
        x_counter[0] += 1
    else:
        # "yes" on the left, so the tree reads the way the rules are written.
        layout(node["yes"], depth + 1, x_counter, pos)
        layout(node["no"], depth + 1, x_counter, pos)
        xl = pos[node["yes"]["_id"]][0]
        xr = pos[node["no"]["_id"]][0]
        pos[node["_id"]] = ((xl + xr) / 2, depth)
    return pos


def draw(node, pos, ax, max_depth):
    if node["kind"] != "leaf":
        x, d = pos[node["_id"]]
        for child, label, dx in ((node["yes"], "yes", -0.06),
                                 (node["no"], "no", 0.06)):
            cx, cd = pos[child["_id"]]
            ax.plot([x, cx], [-d - 0.16, -cd + 0.16], color="#777777",
                    lw=1.0, zorder=1)
            ax.text((x + cx) / 2 + dx, (-d - 0.16 + -cd + 0.16) / 2, label,
                    fontsize=8.5, fontname=FONT, ha="center", va="center",
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.9, pad=1.0))
            draw(child, pos, ax, max_depth)

    x, d = pos[node["_id"]]
    if node["kind"] == "leaf":
        total = node["n_p"] + node["n_w"]
        purity = (max(node["n_p"], node["n_w"]) / total) if total else 0
        color = COLOR_WORLDS if node["cls"] == "worlds" else COLOR_PROGRAMS
        txt = (f"sample\n{node['cls'].upper()}\n"
               f"n={total}  {purity:.0%}")
        w, h = 0.78, 0.30
        ax.add_patch(FancyBboxPatch((x - w / 2, -d - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=color, ec=EDGE, lw=1.3, zorder=2))
        ax.text(x, -d, txt, ha="center", va="center", color="white",
                fontsize=8.5, fontname=FONT, zorder=3, linespacing=1.35)
    else:
        txt = f"{node['a']}\n<\n{node['b']}"
        w, h = 0.92, 0.32
        tier = node.get("tier", pair_tier(f"{node['a']}_LT_{node['b']}"))
        ax.add_patch(FancyBboxPatch((x - w / 2, -d - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    fc=COLOR_NODE, ec=TIER_EDGE[tier],
                                    lw=2.0 if tier >= 2 else 1.3, zorder=2))
        ax.text(x, -d, txt, ha="center", va="center", fontsize=8.5,
                fontname=FONT, zorder=3, linespacing=1.2)
        if tier >= 2:
            ax.text(x, -d - h / 2 - 0.045, TIER_NAMES[tier], ha="center",
                    va="top", fontsize=7.5, fontname=FONT,
                    color=TIER_EDGE[tier])


def figure(node, out_path, acc, acc_std, n, n_leaves, n_q, pruned_from):
    pos = layout(node)
    xs = [p[0] for p in pos.values()]
    ds = [p[1] for p in pos.values()]
    fig_w = max(8.5, (max(xs) - min(xs) + 1) * 2.1)
    fig_h = max(3.6, (max(ds) + 1) * 1.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw(node, pos, ax, max(ds))
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(-max(ds) - 0.32, 0.32)
    ax.axis("off")
    ax.set_title(
        "Which sampler reaches the exact interval first\n"
        f"every question compares two attributes — no numeric thresholds",
        fontsize=14, fontname=FONT, pad=10)
    ax.text(0.5, -0.02,
            f"held-out accuracy {acc:.3f} ± {acc_std:.3f}  "
            f"(n={n}, grouped by model)   ·   "
            f"{n_q} questions, {n_leaves} leaves "
            f"(pruned from {pruned_from})",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5, fontname=FONT, color="#333333")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def rules_text(node, indent=0):
    pad = "    " * indent
    if node["kind"] == "leaf":
        total = node["n_p"] + node["n_w"]
        purity = (max(node["n_p"], node["n_w"]) / total) if total else 0
        return [f"{pad}-> sample {node['cls'].upper()}  "
                f"(n={total}, purity={purity:.0%}, "
                f"programs={node['n_p']}, worlds={node['n_w']})"]
    out = [f"{pad}IF {node['a']} < {node['b']}:"]
    out += rules_text(node["yes"], indent + 1)
    out.append(f"{pad}ELSE:")
    out += rules_text(node["no"], indent + 1)
    return out


def strip_ids(node):
    node.pop("_id", None)
    if node["kind"] != "leaf":
        strip_ids(node["yes"])
        strip_ids(node["no"])
    return node


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", default="results_tte/all_results.csv")
    p.add_argument("--output_dir", default="results_tte/analysis")
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--variance_threshold", type=float, default=0.05)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cost_policy", default=None,
                   help="Max attribute tier usable at each depth, e.g. "
                        "'0,1,2,3'. Tiers: 0 parse JSON, 1 BN graph, "
                        "2 BN inference (entropy/treewidth), 3 DeLP solver "
                        "(am_*). Omit for the plain sklearn tree.")
    p.add_argument("--min_leaf", type=int, default=20,
                   help="Minimum queries per leaf")
    p.add_argument("--ban_tiers", default="",
                   help="Tiers to forbid outright, e.g. '2' to drop entropy "
                        "and treewidth, '2,3' to also drop everything that "
                        "needs the DeLP solver.")
    p.add_argument("--ban_features", default="",
                   help="Comma-separated attribute names to forbid outright.")
    p.add_argument("--no_prune", action="store_true",
                   help="Keep questions whose two answers agree")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plt.rcParams["font.family"] = FONT

    df = pd.read_csv(args.input_csv, sep=";")
    df = df.dropna(subset=["gt_winner"])
    df = df[df["gt_winner"].isin(["worlds", "programs"])].copy()
    df = df.reset_index(drop=True)
    y = (df["gt_winner"] == "worlds").astype(int).values
    groups = (df["config"] + "/" + df["model"]).values
    print(f"Rows: {len(df)}  | configs: {df.config.nunique()} "
          f"| models: {len(set(groups))}")
    print(f"  worlds={int(y.sum())}  programs={int((1-y).sum())}")

    feats = [c for c in select_measurable_features(df) if c not in LEAK]
    banned_tiers = {int(t) for t in args.ban_tiers.split(",") if t.strip()}
    banned_feats = {f.strip() for f in args.ban_features.split(",") if f.strip()}
    if banned_tiers or banned_feats:
        before = len(feats)
        feats = [f for f in feats
                 if feature_tier(f) not in banned_tiers
                 and f not in banned_feats]
        print(f"Banned {before - len(feats)} attributes "
              f"(tiers={sorted(banned_tiers) or '-'}, "
              f"names={sorted(banned_feats) or '-'})")
    df = impute_median(df, feats)
    X_pair, pairs = make_pair_comparisons(df, feats, args.variance_threshold)
    print(f"Base features: {len(feats)}  ->  "
          f"pairwise comparisons kept: {X_pair.shape[1]}")

    cols = list(X_pair.columns)
    Xv = X_pair.values
    policy = parse_policy(args.cost_policy, args.depth) if args.cost_policy \
        else None
    # Balanced weights, so the 10.7% minority class still drives splits.
    w = np.where(y == 1, len(y) / (2.0 * max((y == 1).sum(), 1)),
                 len(y) / (2.0 * max((y == 0).sum(), 1)))

    def fit_tree(tr_idx):
        """Fit on the given row indices and return the plain (pruned) tree."""
        if policy is not None:
            node = grow_cost_aware(Xv, y, w, cols, policy, args.depth,
                                   args.min_leaf, np.asarray(tr_idx), 0)
        else:
            m = DecisionTreeClassifier(max_depth=args.depth,
                                       random_state=args.seed,
                                       min_samples_leaf=args.min_leaf,
                                       class_weight="balanced").fit(
                                           Xv[tr_idx], y[tr_idx])
            node = to_plain(m, 0, cols, leaf_counts(m, Xv[tr_idx], y[tr_idx]))
        return node if args.no_prune else prune(node)

    if policy:
        print("Cost policy by depth: "
              + ", ".join(f"d{i}<=tier{t}" for i, t in enumerate(policy)))

    # ── Honest validation, grouped by model ──────────────────────────────
    gkf = GroupKFold(n_splits=args.n_splits)
    accs, oof = [], np.empty(len(df), dtype=object)
    for tr, te in gkf.split(Xv, y, groups):
        node = fit_tree(tr)
        pred = predict_frame(node, df.iloc[te])
        oof[te] = pred
        accs.append((pred == df["gt_winner"].values[te]).mean())
    acc, acc_std = float(np.mean(accs)), float(np.std(accs))
    print(f"\nHeld-out accuracy (GroupKFold by model): {acc:.4f} +/- {acc_std:.4f}")

    # Persist out-of-fold predictions so the model-comparison figure can use
    # this tree without refitting it.
    oof_path = os.path.join(args.output_dir, "final_tree_predictions.csv")
    df.assign(pred_final=oof,
              match_final=(oof == df["gt_winner"].values))[
        ["config", "model", "literal", "gt_winner",
         "pred_final", "match_final"]].to_csv(oof_path, sep=";", index=False)

    # Leave-one-config-out
    loco = []
    for cfg in sorted(df.config.unique()):
        te = (df.config == cfg).values
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        node = fit_tree(np.where(tr)[0])
        pred = predict_frame(node, df[te])
        loco.append((cfg, int(te.sum()),
                     (pred == df["gt_winner"].values[te]).mean()))
    loco_df = pd.DataFrame(loco, columns=["config", "n", "acc"])
    loco_w = float(np.average(loco_df.acc, weights=loco_df.n))

    # ── Final tree on all rows, then prune ───────────────────────────────
    all_idx = np.arange(len(df))
    if policy is not None:
        raw = grow_cost_aware(Xv, y, w, cols, policy, args.depth,
                              args.min_leaf, all_idx, 0)
    else:
        m = DecisionTreeClassifier(max_depth=args.depth, random_state=args.seed,
                                   min_samples_leaf=args.min_leaf,
                                   class_weight="balanced").fit(Xv, y)
        raw = to_plain(m, 0, cols, leaf_counts(m, Xv, y))
    q_before, l_before = count_nodes(raw)
    tree = raw if args.no_prune else prune(raw)
    q_after, l_after = count_nodes(tree)
    print(f"Pruning: {q_before} questions / {l_before} leaves "
          f"->  {q_after} questions / {l_after} leaves")

    # What each decision actually costs
    tiers = cost_profile(tree, df)
    cost_lines = []
    for t in sorted(TIER_NAMES):
        frac = float((tiers == t).mean())
        if frac:
            cost_lines.append(f"  decided using at most tier {t} "
                              f"[{TIER_NAMES[t]}]: {frac:.1%}")
    never_solver = float((tiers < 3).mean())
    never_infer = float((tiers < 2).mean())
    cost_lines.append(f"  never runs the DeLP solver : {never_solver:.1%}")
    cost_lines.append(f"  never runs BN inference    : {never_infer:.1%}")
    print("\n".join(cost_lines))

    in_sample = (predict_frame(tree, df) == df["gt_winner"].values).mean()
    baseline = max((y == 1).mean(), (y == 0).mean())

    # ── Outputs ──────────────────────────────────────────────────────────
    figure(tree, os.path.join(args.output_dir, "final_tree.pdf"),
           acc, acc_std, len(df), l_after, q_after, l_before)

    rules = "\n".join(rules_text(tree))
    with open(os.path.join(args.output_dir, "final_tree_rules.txt"), "w") as f:
        f.write("Recommendation rules — every question compares two "
                "attributes.\n")
        f.write("Target: which sampler reaches the exact interval first.\n\n")
        f.write(rules + "\n")

    with open(os.path.join(args.output_dir, "final_tree.json"), "w") as f:
        json.dump({"format": "comparison-tree-v1",
                   "target": "gt_winner",
                   "depth": args.depth,
                   "held_out_accuracy": acc,
                   "tree": strip_ids(json.loads(json.dumps(tree)))}, f,
                  indent=1)

    L = ["=== Comparison tree on the time-to-exact ground truth ===", ""]
    L.append(f"Rows            : {len(df)}  "
             f"({int((1-y).sum())} programs, {int(y.sum())} worlds)")
    L.append(f"Configurations  : {df.config.nunique()}")
    L.append(f"Distinct models : {len(set(groups))}")
    L.append(f"Base features   : {len(feats)}")
    L.append(f"Comparisons kept: {X_pair.shape[1]}")
    L.append("")
    L.append(f"Held-out accuracy (GroupKFold by model) : "
             f"{acc:.4f} +/- {acc_std:.4f}")
    L.append(f"Leave-one-config-out (weighted)         : {loco_w:.4f}")
    L.append(f"Majority baseline (always worlds)       : {baseline:.4f}")
    L.append(f"In-sample accuracy of the shown tree    : {in_sample:.4f}")
    L.append("")
    L.append(f"Structure: {q_before} questions / {l_before} leaves before "
             f"pruning, {q_after} / {l_after} after.")
    L.append("Pruning removes questions whose two answers give the same")
    L.append("recommendation, so no answer changes — only the drawing shrinks.")
    L.append("")
    L.append("-- Leave-one-config-out detail --")
    for r in loco_df.itertuples():
        L.append(f"  {r.config:<8s} n={r.n:>5d}  acc={r.acc:.3f}")
    L.append("")
    L.append("-- Rules --")
    L.append(rules)
    report = os.path.join(args.output_dir, "final_tree_report.txt")
    with open(report, "w") as f:
        f.write("\n".join(L))
    print(f"  Saved: {report}")
    print()
    print("\n".join(L[:16]))
    print()
    print(rules)


if __name__ == "__main__":
    main()
