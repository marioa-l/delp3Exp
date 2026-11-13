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
    python sampling/enrich_dataset.py --input_csv results_tte/all_results.csv

Both feature scripts are needed after every merge: the merge keeps only the
result columns, `add_base_features` adds the 19 size/literal/AF attributes and
`enrich_dataset` the 49 AM/EM/AF complexity and graph attributes (from the
cache in `results/complexity_cache.json`). The tree sees 41 of the 68 with
tiers 2-3 banned.

## Step 2 -- the final trees: depth 1, 3 and 5 (decided 2026-11-05)

Three trees are the final result, not one. Pruning is off, because it
collapses every depth to the root question (no deeper question changes the
recommendation); the deeper trees are kept for their leaves, whose purity and
attributes are what the heuristics will read.

    for d in 1 3 5; do
      python sampling/final_tree.py --depth $d --ban_tiers 2,3 --min_leaf 25 \
          --cascade --no_prune --output_dir results_tte/analysis/d$d
    done

Then draw the three as editable slides (needs `pptxgenjs` on NODE_PATH, e.g.
`npm install pptxgenjs` in any folder and `NODE_PATH=<folder>/node_modules`):

    node sampling/trees_to_pptx.js results_tte/figures/final_trees.pptx \
        results_tte/analysis/d1/final_tree.json \
        results_tte/analysis/d3/final_tree.json \
        results_tte/analysis/d5/final_tree.json

### Final ground truth (2026-11-06)

All six roots merged; the 3600 s programs-only re-run resolved 14 of the 88
queries left open on affected models (8 smc, 6 ssc, all programs).

| | worlds | programs | unresolved | total |
|---|---|---|---|---|
| final | 1308 | 846 | 203 (178 cmm, 14 smc, 11 scm) | 2357 |

Of the 2154 resolved: both samplers reach the exact interval on 1206 (56%),
only worlds on 590, only programs on 358. Where both reach, the slower takes
a median 1.68x the faster, >2x in 41.3%, >10x in 6.6%.

### Final trees, on that ground truth

| depth | held-out (GroupKFold by model) | LOCO | in-sample | questions |
|---|---|---|---|---|
| 1 | 0.9257 +/- 0.0207 | 0.9248 | 0.9276 | 1 |
| 3 | 0.9345 +/- 0.0212 | 0.9271 | 0.9378 | 6 |
| 5 | 0.9206 +/- 0.0228 | 0.9025 | 0.9378 | 14 |
| majority baseline | 0.6072 | | | |

Outputs in `results_tte/analysis/d{1,3,5}/`, drawn in
`results_tte/figures/final_trees.pptx`. Accuracy does not drop with depth;
it is flat within one standard error. What changes is transfer: depth 5
loses 2 points under LOCO.

(On the 900 s ground truth the same trees gave 0.9238 / 0.9318 / 0.9271
held-out with fold std ~0.005. The trees differ only in leaf counts and one
depth-5 question, so the larger std is probably the GroupKFold assignment
shifting with the added rows; not checked.) Under either
root branch every leaf carries the same recommendation; the leaves differ in
purity (55% to 100%), i.e. in how close the two algorithms are.

Reference, measured under the bit-shift bug: held-out 0.8386, LOCO 0.7803,
baseline 0.7818, split 1516 / 423 / 418.

The heuristics are to be designed from the information in the depth-3 and
depth-5 trees (which attributes a query's branch consulted, and the purity
of its leaf), on top of the mass-based mechanisms below.

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

### Checked: `bn_centrality` never sampled from the EM (2026-11-12)

`em.gen_samples` raises on every call under pyAgrum 1.15 (`drawSamples`
returns the log-likelihood, not the samples); the world sampler catches it
and falls back to a uniform draw, paying 0.34 ms per draw. The v4 `generic`
worlds numbers are uniform sampling made slower: in the 18- and 24-variable
configs it visited 27-56% of the worlds `random` did. Uniform itself scored
0.003 on `ssm/26`. `utils/umda.py` and `bn.py:302` call the same function.
Paper: L1032 says the scenario heuristic draws from the joint distribution,
and L1569-1577 mixes this artefact into the generic results; both must go.

### Implemented (2026-11-12)

- `ancestral` (worlds, `UniqueAncestral` in `byWorldSampling.py`): ancestral
  sampling without replacement over a trie of unvisited mass. Validated:
  exhausts the space with no repeats, total mass 1.000000000000, P(w) equal
  to pyAgrum inference to 6e-17, 100-160 us per draw.
- `marginal_order` (programs): the enumerated list sorted by the product of
  annotation marginals, without replacement.
- Smoke test, 20 s: `ssm/26 a_6` mass visited worlds 0.002 -> 0.991,
  programs 0.009 -> 0.940.

### Step 3 run -- sample-by-sample convergence on the tree leaves (icic)

Four curves per non-trivial query (worlds and programs, each uniform and
guided), routed by the depth-3 and depth-5 trees, 60 queries per leaf of
each tree (618 queries), 280 s cap per curve:

    python sampling/leaf_heuristic_experiment.py --n_per_leaf 60 \
        --time_limit 280 --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_dir results_tte/leaf_heuristics_v5

Worst case 12 h at 16 workers, plus the program enumeration each query pays
before its clock starts. Resumable by relaunching the same command;
`--summary_only` rebuilds `summary.txt` / `summary.csv` from what is on disk.
Read mass against k to compare the order within a method, and the gap to the
exact interval against time to compare the methods.

Results (2026-11-13, all four curves on the 618 queries, `summary.txt`):
ancestral beats uniform worlds on every query (a third of the samples, 0.41x
the time); marginal order beats uniform programs on 91% (0.90x). Tree +
guided: 577/618 exact, 65.2 +/- 3.4 s mean capped time, against 92.9 +/- 4.0
for tree + uniform and 57.7 +/- 3.1 for the oracle. But where the tree says
programs, the guided programs sampler is first only 0.52 of the time (d3 L6
flips to worlds): ancestral helped worlds far more than marginal order helped
programs, so the uniform labels no longer describe the guided samplers.

## Step 4 -- guided ground truth and a new tree

The current tree is, in effect, one question (`em_var < n_annots`, sample the
smaller space): with uniform samplers the cost follows the size of the space.
With guided samplers it follows how concentrated the mass is, so the labels
change (618 queries: 138 programs -> worlds, 8 the other way). A trial on
those 618, cheap attributes plus `em_entropy_mc`, puts
`af_n_em_vars_used < em_entropy_mc -> programs` (89/89) at the root.
`em_entropy_mc` is the mean -log2 P(w) over ancestral draws: no inference,
r = 0.9999 against `em_entropy`, 2 s for 431 models.

icic, guided curves on every non-trivial query (1770; the 618 are skipped,
1152 jobs, worst case 11.2 h at 16 workers):

    python sampling/leaf_heuristic_experiment.py --n_per_leaf 0 \
        --include_unresolved --curves worlds_guided,programs_guided \
        --time_limit 280 --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_dir results_tte/leaf_heuristics_v5

Optional, after it: the same command with
`--curves worlds_random,programs_random` (another 11.2 h worst case) gives
all four curves on all 1770, so the convergence figures and the tree +
uniform baseline cover every query, measured on the same machine. The
figures only average over queries that have every curve.

Then, locally:

    python sampling/guided_ground_truth.py
    for d in 1 3 5; do
      python sampling/final_tree.py --input_csv results_tte/all_results_guided.csv \
          --depth $d --ban_tiers 2,3 --min_leaf 25 --cascade --no_prune \
          --output_dir results_tte/analysis_guided/d$d
    done

and compare the old and new trees under guided sampling by policy cost (mean
capped time, queries at the exact interval), not only accuracy.

## Paper state

Superseded: a new version of the paper will be supplied; edit that one, not
the `KBS_...` copy described below.

`KBS_.../main.tex` compiles clean (0 errors, 35 pages; 36 undefined
references, all pre-existing `\linelabel` targets). Results and Conclusions
are written against the biased ground truth and will need revisiting after
step 2; the heuristics subsection (negative result) will be replaced after
step 3. `images/final_tree_abst.pdf` still shows the 5-question tree;
`final_tree_abst_v2.pptx` is the 8-question one, to restyle and export once
the tree is refitted (it will change again).
