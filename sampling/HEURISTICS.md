# Sampling heuristics

Two generations of heuristics live in the samplers. They differ in where
their guidance comes from.

## Generic heuristics (earlier)

Independent of the decision tree: they rebuild their own state on every run
and behave identically whichever leaf the query landed in.

| mode | method | what it does | cost it re-pays |
|---|---|---|---|
| `bn_centrality` | worlds | Draws worlds from the BN's joint distribution instead of uniformly, then accepts each with probability proportional to an *influence* score — the attack-graph in-degree of the arguments whose rules the world activates. | Needs the solver's dGraph (tier 3) |
| `bn_marginal` | programs | Computes, per annotated rule, the BN marginal probability that its annotation holds (SymPy enumerates satisfying assignments, LazyPropagation sums them), then includes each rule independently at that rate. Adds a syntactic pre-filter that drops programs with direct strict contradictions before calling the solver. | Needs BN inference (tier 2) |

Both reintroduce exactly the computations the comparison tree was built to
avoid, which is the main argument against them.

`bn_centrality` never drew from the BN. Under pyAgrum 1.15
`BNDatabaseGenerator.drawSamples` returns the log-likelihood of the draw, not
the samples, so `em.gen_samples` raises on every call; the sampler catches it
and falls back to a uniform draw, paying the failed call (0.34 ms) on every
draw, repeats included. Its v4 numbers are uniform sampling made slower, not
a heuristic. `bn_marginal` does not go through `gen_samples` and is not
affected; it fails to close the interval because it samples with
replacement.

## Mass-based heuristics (v5)

Both order the visits by probability mass while doing only polynomial work
per sample: nothing computes P for the whole space up front.

| mode | method | what it does |
|---|---|---|
| `ancestral` | worlds | Ancestral sampling from the EM in topological order, without replacement (`UniqueAncestral`): a trie over the order holds the unvisited mass below each branch, every draw is proportional to P(w) among the unvisited worlds, and P(w) is the product of the CPT entries drawn. O(\|E\|) per world, no inference, no repeats, ends when every world of positive mass has been visited. |
| `marginal_order` | programs | The enumerated program list, visited without replacement in decreasing product of the annotation marginals (the program's mass if the annotations were independent). One inference per annotation, charged to the run's clock. |

## Leaf-guided heuristics (new)

These reuse attributes the tree already consulted — tier 0 and tier 1 only,
so reaching the leaf and running the sampler share the same information and
nothing expensive is computed twice.

### H1 — `hub_strata` (world sampling)

The BN rarely spreads influence evenly: a few variables sit at the centre of
the graph and dominate which worlds matter. Uniform sampling leaves coverage
of those variables to chance.

H1 ranks the EM variables by degree centrality (parents + children, read off
the BN graph — no inference), takes the top *k*, and enumerates all 2^*k*
assignments of them. Sampling then cycles round-robin through those strata:
the remaining variables come from the BN as usual, but the hub variables are
pinned to the current stratum. Every hub configuration is therefore visited
early instead of eventually.

The estimator still weights each world by its true probability, so the
bounds stay sound — only the order in which the space is covered changes.

### H2 — `usage_strata` (program sampling)

The tree reaches a "sample programs" leaf through facts about annotation
structure (body size, connectors, rule-to-variable ratio), which say that a
few EM variables drive many rules.

H2 counts, for each EM variable, how many annotations reference it, takes
the top *k*, and enumerates their 2^*k* assignments. For each proposal the
remaining EM variables are drawn at random, and the annotations are then
*evaluated* under the resulting assignment to obtain the program.

Deriving the program from an assignment rather than forcing annotation bits
directly is what keeps the proposal sound. Annotations share variables, so
fixing their truth values independently produces conjunctions no world
satisfies; each such program is discarded after the solver has already been
called. An earlier version of H2 did exactly that and wasted 42 of every 45
samples. The current form produces zero inconsistent programs.

### H3 — adaptive stratification depth

Both H1 and H2 need a *k*. A fixed constant is wrong in both directions: a
small space is covered fine without stratification, and a large one deserves
more coverage insurance.

H3 derives *k* from the size of the space being sampled, `log2(dims) - 1`
capped at 4, and adds one more level when the caller passes `hint_ratio > 0`
— meaning this method's space is the larger of the two. That ratio is
`n_annots - em_var`, the same quantity the tree's root question compares, so
it is already known by the time the sampler starts.

`strata_k` overrides it explicitly when you want a fixed depth.

## Interface

Both samplers accept:

```python
start_sampling(..., mode="hub_strata" | "usage_strata",
               strata_k=None,        # None -> H3 picks it
               hint_ratio=None)      # log2 size ratio, positive = larger space
```

`sampling/leaf_heuristic_experiment.py` (v5) runs four curves on every
non-trivial query -- worlds and programs, each uniform and mass-guided -- and
records the interval and the mass visited sample by sample, grouped by the
leaves of the depth-3 and depth-5 trees. The v4 version (plain, generic and
guided on the recommended method only, quality at the cap) is the one at
commit c09fba70; its results are in `results_tte/leaf_heuristics_v4/`.
