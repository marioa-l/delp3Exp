"""
Two-panel figure for discussing the abstract tree with the ground-truth
validation:

    Left  panel:  the abstract decision tree (drawn manually so bin
                  thresholds are shown as their semantic labels — 'low',
                  'medium', 'high', 'balanced', 'programs-favored',
                  'worlds-favored' — instead of the raw ordinal indices
                  used internally by sklearn).
    Right panel:  per-leaf accuracy on the time-to-exact validation set,
                  sorted from worst to best. Leaves are colour-matched
                  with the tree to make it easy to spot the failing ones.

Both panels use the abs_leaf_<ID> identifier that ``validate_trees_tte.py``
already writes into ``pred_abs_region``.

Reads:
    results_5min/analysis/abstract_tree_n3_d3.pkl
    results_tte/analysis/validation_tte_predictions.csv

Writes:
    results_tte/figures/tte_abstract_tree_and_validation.pdf
"""

from __future__ import annotations

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

FONT = "DejaVu Sans"

COLOR_WORLDS   = "#0072B2"
COLOR_PROGRAMS = "#D55E00"
COLOR_OK       = "#27AE60"
COLOR_BAD      = "#C0392B"
COLOR_NODE_BG  = "#F4F4F4"
COLOR_NODE_EDGE = "#333333"


def edge_label_for(node_id: int, is_left: bool, feature_name: str,
                   threshold: float, encoders: dict) -> str:
    """
    Translate an sklearn split (`feature <= threshold`) back to the
    semantic bin labels that abstract_tree.py encoded before fitting.

    ``encoders[feature_name]`` is the ordered list of labels used to turn
    the categorical feature into integers (0, 1, 2, ...). sklearn splits
    on ``<= 0.5`` / ``<= 1.5`` / ``<= 2.5``, so the split cutoff floors
    to the index of the last label kept on the left branch.
    """
    labels = encoders.get(feature_name)
    if not labels:
        return "<=" if is_left else ">"
    # sklearn threshold X.5 → left keeps indices 0..floor(X.5)=X
    cut = int(np.floor(threshold))
    left = labels[: cut + 1]
    right = labels[cut + 1:]
    picked = left if is_left else right
    if len(picked) == 1:
        return picked[0]
    return " / ".join(picked)


def draw_tree_panel(ax, tree_state, val_stats, color_by="accuracy"):
    tree = tree_state["tree"]
    feats = tree_state["predictors_categorical"]
    encoders = tree_state["encoders"]
    tt = tree.tree_

    # class 0 == programs, class 1 == worlds (verified against
    # validate_trees_tte report)
    CLASS_TO_METHOD = {0: "programs", 1: "worlds"}

    # ── Layout: perfect binary tree of depth 3 ────────────────────────────
    depth = 3
    positions = {}
    def _place(node_id, level, x_low, x_high):
        x_mid = (x_low + x_high) / 2
        y = 1.0 - level / depth
        positions[node_id] = (x_mid, y)
        left, right = tt.children_left[node_id], tt.children_right[node_id]
        if left != -1:
            _place(left, level + 1, x_low, x_mid)
            _place(right, level + 1, x_mid, x_high)
    _place(0, 0, 0.0, 1.0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.18, 1.12)
    ax.axis("off")

    NODE_W, NODE_H = 0.19, 0.09
    LEAF_W, LEAF_H = 0.115, 0.13

    def _box(x, y, w, h, text, face, edge, text_color="black", fontsize=8.5):
        patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.008,rounding_size=0.015",
                               linewidth=1.3, edgecolor=edge, facecolor=face)
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center",
                fontname=FONT, fontsize=fontsize, color=text_color,
                linespacing=1.25)

    # ── Draw edges first (behind the boxes) ───────────────────────────────
    for node_id, (x, y) in positions.items():
        left, right = tt.children_left[node_id], tt.children_right[node_id]
        if left == -1:
            continue
        fname = feats[tt.feature[node_id]]
        thr = tt.threshold[node_id]
        for child, is_left in ((left, True), (right, False)):
            cx, cy = positions[child]
            ax.plot([x, cx], [y - NODE_H / 2, cy + NODE_H / 2],
                    color="#666666", lw=1.0, zorder=1)
            # label mid-edge
            lx, ly = (x + cx) / 2, (y + cy) / 2
            label = edge_label_for(node_id, is_left, fname, thr, encoders)
            ax.text(lx + (-0.015 if is_left else 0.015), ly,
                    label, ha="right" if is_left else "left",
                    va="center", fontname=FONT, fontsize=8,
                    color="#333333",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.2))

    # ── Draw nodes ────────────────────────────────────────────────────────
    for node_id, (x, y) in positions.items():
        is_leaf = tt.children_left[node_id] == -1
        if is_leaf:
            pred_class = int(np.argmax(tt.value[node_id][0]))
            method = CLASS_TO_METHOD[pred_class]
            leaf_key = f"abs_leaf_{node_id}"
            stat = val_stats.get(leaf_key, {"n": 0, "acc": float("nan")})
            if color_by == "prediction":
                face = COLOR_WORLDS if method == "worlds" else COLOR_PROGRAMS
            else:
                face = COLOR_OK if stat["acc"] >= 0.6 else COLOR_BAD
            txt = (f"$\\bf{{leaf\\ {node_id}}}$\n"
                   f"→ {method}\n"
                   f"acc={stat['acc']:.2f}\n"
                   f"n={stat['n']}")
            _box(x, y, LEAF_W, LEAF_H, txt, face=face,
                 edge=COLOR_NODE_EDGE, text_color="white", fontsize=8.0)
        else:
            fname = feats[tt.feature[node_id]]
            txt = f"{fname}"
            _box(x, y, NODE_W, NODE_H, txt, face=COLOR_NODE_BG,
                 edge=COLOR_NODE_EDGE, fontsize=8.5)

    ax.set_title("Abstract decision tree (depth 3, n_bins=3)",
                 fontsize=14, fontname=FONT, pad=6)


def draw_validation_panel(ax, val_stats):
    """Horizontal bar chart of per-leaf accuracy, sorted worst to best."""
    items = sorted(val_stats.items(), key=lambda kv: kv[1]["acc"])
    labels = [k for k, _ in items]
    accs = [v["acc"] for _, v in items]
    ns = [v["n"] for _, v in items]
    preds = [v["pred"] for _, v in items]
    colors = [COLOR_OK if a >= 0.6 else COLOR_BAD for a in accs]

    y = np.arange(len(labels))
    ax.barh(y, accs, color=colors, edgecolor="white", height=0.65)
    for i, (a, n, pr) in enumerate(zip(accs, ns, preds)):
        ax.text(min(a + 0.02, 1.02), i,
                f"{a:.2f}   (n={n}, pred={pr})",
                va="center", fontname=FONT, fontsize=10)
    ax.axvline(0.6, color="black", lw=0.7, ls="--",
               label="Threshold (0.60)")
    ax.axvline(0.5, color="gray", lw=0.6, ls=":", label="Chance (0.50)")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontname=FONT, fontsize=11)
    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Accuracy (time-to-exact ground truth)",
                  fontsize=12, fontname=FONT)
    ax.set_title("Per-leaf accuracy on the validation set",
                 fontsize=14, fontname=FONT, pad=6)
    ax.tick_params(which="both", direction="in", top=True, right=True,
                   labelsize=10)
    ax.minorticks_on()
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)


def compute_val_stats(val_df: pd.DataFrame) -> dict:
    stats = {}
    for reg, grp in val_df.groupby("pred_abs_region"):
        stats[reg] = {
            "n": int(len(grp)),
            "acc": float(grp["match_abs"].mean()),
            "pred": grp["pred_abs_method"].mode().iloc[0],
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",
                        default="results_5min/analysis/"
                                "abstract_tree_n3_d3.pkl")
    parser.add_argument("--val_csv",
                        default="results_tte/analysis/"
                                "validation_tte_predictions.csv")
    parser.add_argument("--out",
                        default="results_tte/figures/"
                                "tte_abstract_tree_and_validation.pdf")
    parser.add_argument("--color_by",
                        choices=["accuracy", "prediction"],
                        default="accuracy",
                        help="'accuracy' (green/red by acc >= 0.6) or "
                             "'prediction' (blue=worlds, orange=programs).")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        tree_state = pickle.load(f)

    val_df = pd.read_csv(args.val_csv, sep=";")
    val_df = val_df.dropna(subset=["gt_winner"])
    val_df = val_df[val_df["gt_winner"].isin(["worlds", "programs"])]

    val_stats = compute_val_stats(val_df)

    plt.rcParams["font.family"] = FONT
    fig, (ax_tree, ax_val) = plt.subplots(
        1, 2, figsize=(18, 7.0),
        gridspec_kw={"width_ratios": [1.7, 1.0]})

    draw_tree_panel(ax_tree, tree_state, val_stats, color_by=args.color_by)
    draw_validation_panel(ax_val, val_stats)

    overall = val_df["match_abs"].mean()
    fig.suptitle(
        f"Abstract tree structure and its validation on the "
        f"time-to-exact ground truth  (overall accuracy = {overall:.3f})",
        fontsize=15, fontname=FONT, y=1.01)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
