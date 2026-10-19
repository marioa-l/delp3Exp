# Where this is and what comes next

Written 2026-10-19, after finding that program sampling drew with replacement
while world sampling did not.

## What happened

`SamplAM` drew from the enumerated program space with replacement and queried
every draw. A program already seen cost another solver call that could not
move the interval, since its scenarios were already in `known_evidences`.
`SamplEM` has always skipped repeats before calling the solver.

Measured on `smm/39model`, whose space holds 68 distinct consistent programs:
the old path spent 944 solver calls over 20 s to reach the interval the fixed
one reaches in 68 calls and 2 s.

Fixed in `c09fba70`. The consequence is that every result computed before that
commit charged the two algorithms on different terms.

## What this invalidates

- **The time-to-exact ground truth** (`results_tte/all_results.csv`). Both
  algorithms ran in `random` mode, so `SamplAM` was handicapped by roughly an
  order of magnitude in solver calls and `SamplEM` was not. "Which converges
  first" leaned towards `SamplEM`. The 1687/203 split is measured under that
  bias.
- **The decision tree**, which was fitted against those labels
  (`results_tte/analysis/final_tree.*`).
- **Everything downstream**: the per-configuration accuracies, the heuristic
  experiments, and the Results and Conclusions sections of the paper.

Not affected: the two sampling algorithms themselves, their soundness
propositions, the generators, and the dataset of \D3E programs.

## Step 1 — recompute the ground truth

Only where the program space is small enough for the waste to matter. With
2^40 programs or more a few thousand draws practically never collide, so the
fix changes nothing there, and world sampling wins unanimously in those
configurations anyway.

    python sampling/time_to_exact_experiment.py \
        --configs sms,smc,smm,sss,ssc,ssm,scm,scs \
        --n_models 100 \
        --cap 900 \
        --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_root results_tte_fixed

1366 queries across the eight configurations. Worst case 42 h at 16 workers,
but far less in practice: the program side now exhausts small spaces in
seconds instead of grinding through duplicates.

Then re-run whatever stays unresolved at the larger cap, as before:

    python sampling/rerun_unresolved.py \
        --input_csv results_tte_fixed/all_results.csv \
        --cap 3600 \
        --workers 16 \
        --models_root /home/jupyter-mario.leiva.al@gma-57ad0/exp-delp3e \
        --output_root results_tte_fixed_long

Merge, keeping the unaffected configurations from the old run:

    python sampling/merge_tte.py --roots results_tte,results_tte_fixed,results_tte_fixed_long
    python sampling/add_base_features.py --input_csv results_tte/all_results.csv

`merge_tte.py` keeps the row with the largest `safety_cap` per query, so the
recomputed rows supersede the old ones only where they exist.

## Step 2 — refit and revalidate the tree

    python sampling/final_tree.py --depth 5 --ban_tiers 2,3 --min_leaf 25 --cascade

Compare against what the biased labels gave, and expect the comparison to be
part of the story rather than a formality:

| | old (biased) |
|---|---|
| held-out accuracy | 0.9413 +/- 0.0271 |
| leave-one-config-out | 0.8952 |
| majority baseline | 0.8926 |
| class split | 1687 worlds / 203 programs |

If the split moves materially towards programs, the tree's questions will
change and Section 5 has to be rewritten around the new ones.

## Step 3 — heuristics, last

Only once the ground truth and the tree are settled. Current state:

- **H1 (`hub_strata`, worlds) is a no-op by construction.** It draws a uniform
  world and overwrites k hub variables with a round-robin stratum. The non-hub
  bits are uniform and the hub bits cycle through all 2^k combinations, so the
  induced distribution over worlds is exactly uniform. Discard it; there is no
  redesign inside that idea.
- **H2 was redesigned** (`program_strata`) to partition the enumerated program
  space by the annotations with the most structural leverage, instead of
  drawing a world and deriving its program, which visited programs in
  proportion to how many worlds induce them. With both modes now drawing
  without replacement, H2 and `random` are identical on spaces small enough to
  exhaust — order does not change the endpoint, only the trajectory. Whether
  it helps on large program spaces is open, and the convergence traces are
  what would show it.
- `generic` (`bn_centrality`, `bn_marginal`) stays as a paragraph in the paper:
  it performs worse and it re-pays the costs the tree avoids, but it is the
  clearest demonstration of a sampler reporting a tighter interval while almost
  never reaching the exact one.

## Paper state

`KBS_.../main.tex`, 37 pages, compiles clean (0 LaTeX errors, 0 undefined
citations, 25 undefined references all pre-existing `\linelabel` targets).
Sections 5 and 6 and the Conclusions are written against the biased numbers
and will need revisiting after step 2.

The heuristics subsection text was handed over in LaTeX but not yet pasted
into Overleaf.
