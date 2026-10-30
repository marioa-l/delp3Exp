# Where this is and what comes next

Written 2026-10-30, after finding the program-id bit shift in `SamplAM`.
Supersedes the 2026-10-19 note on the with-replacement bug, whose steps were
completed (ground truth recomputed, tree refitted, heuristics re-run, paper
updated) before this one was found.

## What happened

A program id carries one bit per rule (`map_world_to_prog`), but
`id_prog_to_bin` padded it to `n_annots` digits. Whenever the first rule of
the model was absent the binary string lost its leading zero, came out shorter
than the rule list, and `byProgramSampling` read every bit one rule off: the
mass expression and the program text sent to the solver belonged to another
program, or to none. In `sms/23model` 138 of 276 programs had a short id, 64
came out "inconsistent", and the mass the sampler could account for summed to
0.001304 -- exactly the `random_quality` the heuristic run reported.

Fixed in `utils/utils.py` (pad to `am_rules_dim`). Verified: `sms/23model ~a_6`
now reaches the exact interval in 12 s with 0 inconsistent programs,
`smm/32model a_8` in 17 s. Neither closed before.

Not affected: world sampling (uses the id only as a cache key), the exact
intervals (computed by worlds), the `bn_marginal` mode (builds programs from
the full `prog_in_bin`).

## What this invalidates

The bug bites a model whenever rule 0 carries a non-trivial annotation:
95-100% of the models with AF size `s` or `c`, ~40% with `m`. Over the
ground truth:

| models | queries | programs | worlds | unresolved |
|---|---|---|---|---|
| rule 0 trivial (sound) | 776 | 339 | 307 | 130 |
| rule 0 annotated (affected) | 1581 | 84 | 1209 | 288 |

In the sound models the two algorithms split evenly; in the affected ones
`SamplAM` almost never wins. So:

- the `programs` half of the ground truth on the 1581 affected queries
  (`results_tte/bitshift_affected_models.csv` lists the 880 models),
- the decision tree and every number derived from it,
- the programs branch of the heuristic experiment (87 of its 282 queries are
  on affected models: random exact rate 0.26 there vs 0.84 on sound ones).

The `worlds` half of every row is kept.

## Step 1 -- recompute the programs half (icic)

`rerun_unresolved.py` gained `--all_queries`, `--models_csv` and `--methods`;
with a single method the other method's columns stay empty and the merge
fills them from the earlier root.

    python sampling/rerun_unresolved.py \
        --input_csv results_tte/all_results.csv \
        --all_queries --methods programs \
        --models_csv results_tte/bitshift_affected_models.csv \
        --cap 900 --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_root results_tte_bitshift

1581 queries, worst case 24.7 h at 16 workers; resumable by relaunching the
same command. Then merge, with later roots overriding column by column and the
winner recomputed from the two times:

    python sampling/merge_tte.py --supersede \
        --roots results_tte,results_tte_long,results_tte_fixed,results_tte_fixed_long,results_tte_bitshift

All four existing roots must be listed, in that order: they reproduce the
current `all_results.csv` at 99.9% (two `cmm` rows flip from unresolved to
worlds, because the 3600 s run had reached the interval and the earlier
merge missed it). Then re-run what stays unresolved at the long cap, programs
only, same model list -- the input CSV must be the merged one:

    python sampling/rerun_unresolved.py \
        --input_csv results_tte/all_results.csv \
        --methods programs \
        --models_csv results_tte/bitshift_affected_models.csv \
        --cap 3600 --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_root results_tte_bitshift_long

    python sampling/merge_tte.py --supersede \
        --roots results_tte,results_tte_long,results_tte_fixed,results_tte_fixed_long,results_tte_bitshift,results_tte_bitshift_long
    python sampling/add_base_features.py --input_csv results_tte/all_results.csv

## Step 2 -- refit and revalidate the tree

    python sampling/final_tree.py --depth 5 --ban_tiers 2,3 --min_leaf 25 --cascade

Reference, measured under the bug:

| | current (biased) |
|---|---|
| held-out accuracy | 0.8386 +/- 0.0098 |
| leave-one-config-out | 0.7803 |
| majority baseline | 0.7818 |
| class split | 1516 worlds / 423 programs / 418 unresolved |

Expect the split to move a long way towards programs and the root question
(`em_var < n_annots`) to change. Section 5 is written around the current
questions.

## Step 3 -- heuristics, redesigned

### What was measured

Probability mass in these models is highly concentrated. Joint computed for
every world (CPT product, vectorised; `toarray()` axes run in the reverse of
`pot.names`):

| model | 99% of the mass in | 1-1e-6 of the mass in |
|---|---|---|
| ssm/26 (|E|=18, 262k worlds) | 0.5% of worlds | 23% of worlds |
| ssm/0 (|E|=18) | 0.3% | 19% |
| scm/49 (|E|=18) | 0.4% | 21% |
| ssc/15 (|E|=24, 16.7M worlds) | 7% | 69% |

Replaying visiting orders with real solver statuses (one call per distinct
program, then offline replay) gives the ceiling of what an order can buy:

| model, space | uniform: quality@1% -> exact at | mass order: quality@1% -> exact at |
|---|---|---|
| ssm/26, worlds | 0.002 -> 100% | 0.997 -> 21% |
| ssm/0, worlds | 0.001 -> 99.9% | 0.998 -> 16% |
| smm/32, worlds | 0.002 -> 100% | 0.999 -> 11% |
| ssm/26, programs | 0.001 -> 99.9% | 0.970 -> 43% |
| sms/23, programs | 0.000 -> 99.6% | 0.706 -> 50% |

Scripts: `diag_mass.py`, `sim_order.py` (session scratchpad; rewrite under
`sampling/` if the numbers go in the paper).

### What is off limits

Visiting worlds in decreasing P(w) requires computing P(w) for all 2^|E|
worlds. That is the exact algorithm with a good visiting order; it is only
feasible here because the validation dataset is enumerable (it has to be, or
there would be no exact intervals to validate against). Same for computing
the exact mass of every program up front. Neither is a sampling heuristic and
either makes the tree and the uniform baseline meaningless. Rule: polynomial
work per world is fine, exponential work in total is not.

### The two heuristics to implement

- **Worlds: ancestral sampling from the EM.** Walk the variables in
  topological order and draw each from its CPT given the parents. O(|E|) per
  world, no inference, no enumeration, and P(w) comes for free as the product
  of the probabilities drawn -- which also removes the 0.8 ms engine call the
  current sampler pays per world. Skip repeats (as `random` does). Heavy
  worlds come first in expectation; the tail (a world of mass 1e-5 needs
  ~1e5 draws) is the price.
- **Programs: order the enumerated list by the product of annotation
  marginals.** `SamplAM` already enumerates the 2^|used| assignments; ordering
  that list by `_compute_annotation_marginals()` (one inference per
  annotation, not per program) and drawing without replacement adds nothing
  exponential. It is `bn_marginal` used as an order instead of as a
  with-replacement distribution, which is what sank it.

Expected: both win by a wide margin on quality at any practical budget. On
time-to-exact the worlds heuristic may lose on the tail, as `hub_strata` did.
That is a property of the criterion (closing at 1e-6 means visiting the
1e-7-mass worlds, which no weighted sampler finds quickly), to be reported as
such.

### Also worth trying

An attribute the tree can ask that defines an order. The tree's original
purpose was that; it ended up using structural attributes only. Candidates
would be cheap summaries of mass concentration (e.g. the entropy of the
annotation marginals, or the spread of the per-variable marginals), which say
how much an order can buy rather than what the order is.

### Check before implementing

`generic` on worlds (`bn_centrality`) already drew from the EM with repeats
skipped and scored 0.002 on `ssm/26` at 280 s. With 90% of the mass in 101
worlds that is not possible if it works. Find out why before building the
ancestral sampler on the same path (`gen_samples` -> `bn_buffer`).

## Paper state

`KBS_.../main.tex` compiles clean (0 errors, 35 pages; 36 undefined
references, all pre-existing `\linelabel` targets). Results and Conclusions
are written against the biased ground truth and will need revisiting after
step 2; the heuristics subsection (negative result) will be replaced after
step 3. `images/final_tree_abst.pdf` still shows the 5-question tree;
`final_tree_abst_v2.pptx` is the 8-question one, to restyle and export once
the tree is refitted (it will change again).
