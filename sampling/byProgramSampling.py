import copy
import re
import time

import numpy as np
from progress.counter import Counter
from sympy import *

from delp.consultDeLP import *
from utils.utils import *


def def_vars_solver(annots: dict) -> None:
    """To define vars used in the model as symbols to use in the solver"""
    aux = [re.findall(r"\d+", annot) for key, annot in annots.items()]
    used_vars = [item for sublist in aux for item in sublist]
    in_list = list(set(used_vars))
    var(["x" + em_var for em_var in in_list])
    return in_list


def adapt_annots(annots: dict) -> dict:
    """To adapt annotations into valid expressions to use in the solver"""
    adapted_annots = {}
    for index, annot in enumerate(annots):
        if annots[annot].isdigit():
            adapted_annots[annot] = {
                "True": "(x" + annots[annot] + ")",
                "False": "~(x" + annots[annot] + ")",
            }
        else:
            op_repl = (
                annots[annot].replace("and", "&").replace("or", "|").replace("not", "~")
            )
            in_list = op_repl.split(" ")
            expr = " ".join(
                ["x" + element if element.isdigit() else element for element in in_list]
            )
            adapted_annots[annot] = {
                "True": "(" + expr + ")",
                "False": "~(" + expr + ")",
            }
    return adapted_annots


def to_evidence(model: json) -> json:
    """To transform a model (values that satisfied a formula) in the format of evidences"""
    evidence = {}
    for k, v in model.items():
        if v:
            evidence[str(k)[1:]] = 1
        else:
            evidence[str(k)[1:]] = 0
    return evidence


class Programs:
    def __init__(self, model_path: str, save_path: str):
        # Utils to handle model
        self.utils = Model(model_path, save_path)
        # Define all variables in the programs to use in the solver
        # and return the list of that variables
        self.used_vars = def_vars_solver(self.utils.annotations)
        # Adapt annotations to use in the solver
        self.adapted_annots = adapt_annots(self.utils.annotations)
        # To save all results
        self.results = {}
        # To control repeated evidences
        self.known_evidences = []
        # to save programs id and bin
        self.local_n_programs = []

    # ── Leaf-guided helpers (H2 usage strata, H3 adaptive depth) ────────────
    def _annotation_vars(self) -> dict:
        """
        ``{annotation_index -> set(EM variable indices it references)}``,
        read straight off the annotation strings. Tier-0 information: no BN
        inference and no solver, just parsing what is already in the model.
        """
        # The dict keys are the rule indices, i.e. the positions of "x" in
        # prog_in_bin — not 0,1,2,... Keying by enumeration order instead
        # silently writes each annotation's bit into the wrong slot.
        return {key: {int(t) for t in re.findall(r"\d+", str(text))}
                for key, text in self.utils.annotations.items()}

    def _high_leverage_annots(self, k: int) -> list:
        """
        H2: the annotation positions worth covering systematically.

        An annotation scores high when it references EM variables that many
        other annotations also depend on, so pinning its value down settles a
        large slice of the program space. These are facts about annotation
        structure, which is what the branch leading to a "sample programs"
        leaf compares (body size, connectors, rule-to-variable ratio).
        """
        avars = self._annotation_vars()
        usage = {}
        for vs in avars.values():
            for v in vs:
                usage[v] = usage.get(v, 0) + 1
        score = {pos: sum(usage.get(v, 0) for v in vs)
                 for pos, vs in avars.items()}
        return sorted(score, key=lambda i: -score[i])[:k]

    def _partition_programs(self, programs: list, positions: list) -> list:
        """
        Split the enumerated consistent programs into cells by the values the
        chosen annotations take in them.

        Partitioning the program space itself is what distinguishes this from
        drawing a world and deriving its program: the latter visits programs
        in proportion to how many worlds induce them, which over-samples the
        few programs that already carry most of the mass and leaves the rest
        of the space uncovered. Here every program keeps equal weight within
        its cell, and the cells are visited in turn.
        """
        cells = {}
        for pid in programs:
            bits = self.utils.id_prog_to_bin(pid)
            key = tuple(bits[p] if p < len(bits) else 0 for p in positions)
            cells.setdefault(key, []).append(pid)
        out = list(cells.values())
        for c in out:
            np.random.shuffle(c)
        return out

    @staticmethod
    def _adaptive_k(n_dims: int, hint_ratio: float = None,
                    k_max: int = 4) -> int:
        """H3: stratification depth chosen from the size of the space."""
        if n_dims <= 2:
            return 0
        k = int(min(k_max, max(1, round(np.log2(max(n_dims, 2)) - 1))))
        if hint_ratio is not None and hint_ratio > 0:
            k = int(min(k_max, k + 1))
        return min(k, n_dims)

    # ── Heuristic P helpers ──────────────────────────────────────────────────
    def _compute_annotation_marginals(self):
        """
        For every annotated rule, compute the BN marginal probability that
        its annotation formula evaluates to True. Returns a dict
        ``{annotation_index -> marginal in [0, 1]}``.

        Uses SymPy to enumerate satisfying assignments and the BN's
        LazyPropagation engine to sum their probabilities. Values are
        clamped to [0, 1] to guard against numerical drift.
        """
        marginals = {}
        for annot_idx, forms in self.adapted_annots.items():
            try:
                models = list(satisfiable(eval(forms["True"]), all_models=True))
            except Exception:
                marginals[annot_idx] = 0.5
                continue
            prob = 0.0
            for model in models:
                if model:
                    evidence = to_evidence(model)
                    prob += self.utils.em.get_sampling_prob(evidence)
            marginals[annot_idx] = min(1.0, max(0.0, prob))
        return marginals

    def _sample_program_bin_from_marginals(self, marginals):
        """
        Draw a program by including each annotated rule independently
        according to its marginal probability. Returns the binary
        program string used by ``map_bin_to_prog``.
        """
        out = list(self.utils.prog_in_bin)
        for idx, marg in marginals.items():
            out[idx] = 1 if np.random.random() < marg else 0
        return out

    def _has_direct_strict_contradiction(self, program_text: str) -> bool:
        """
        Cheap syntactic pre-filter: return True if the program contains
        a strict fact and the strict fact for its complement literal.
        Only inspects lines that look like strict facts (``<- true``) —
        which are the ones the LLM-labelling experiment showed can flip
        polarity when annotations disagree.
        """
        facts = set()
        for line in program_text.split(";"):
            line = line.strip()
            if not line.endswith("<- true") and " <- true" not in line:
                continue
            head = line.split(" <- ", 1)[0].strip()
            if head:
                facts.add(head)
        for f in facts:
            comp = f[1:] if f.startswith("~") else "~" + f
            if comp in facts:
                return True
        return False

    def start_sampling(
        self, percentile_samples: int, info: str, max_seconds: int = None,
        mode: str = "random",
        stop_at_exact: dict = None,
        stop_epsilon: float = 1e-6,
        strata_k: int = None,
        hint_ratio: float = None,
    ) -> list:
        """
        `stop_at_exact` is an optional mapping ``{literal: (exact_l, exact_u)}``.
        When provided, per-literal sampling exits early as soon as the running
        (l, u) matches the exact bounds within `stop_epsilon`. The observed
        wall-clock ``time_to_exact`` is stored in
        ``self.results["status"][lit]``.
        """
        """To run exact compute of the interval or select randomly a subset of all
        possible programs (combining the values of its annotations) to perform an
        approximation of the exact interval"""
        ##To find all possible consistent programs
        # local_n_programs = {}
        n_used_vars = len(self.used_vars)
        poss_prog_format = "{0:0" + str(n_used_vars) + "b}"
        poss_asignations = pow(2, n_used_vars)
        counter = Counter(
            "Processing possible programs (%d): " % (poss_asignations),
            max=poss_asignations,
        )
        for asignation in range(poss_asignations):
            asign_list = list(poss_prog_format.format(asignation))
            unique_world_program = ["x"] * self.utils.em_vars
            for index, value in enumerate(asign_list):
                unique_world_program[int(self.used_vars[index])] = int(value)
            prog, id_prog = self.utils.map_world_to_prog(unique_world_program)
            if id_prog not in self.local_n_programs:
                self.local_n_programs.append(id_prog)
                # evidence = {str(self.used_vars[index]):int(val) for index, val in enumerate(asign_list)}
                # self.local_n_programs[id_prog] = self.utils.em.get_sampling_prob(evidence)
            counter.next()
        counter.finish()
        ##
        # n_programs = self.utils.get_n_programs()
        n_programs = len(self.local_n_programs)
        print("Number of programs: " + str(n_programs))

        repeated_programs = 0
        inconsistent_programs = 0
        worlds_consulted = 0
        history = []

        # Pre-compute annotation marginals if we're using the heuristic.
        annotation_marginals = None
        if mode == "bn_marginal":
            annotation_marginals = self._compute_annotation_marginals()

        # H2: cells of the program space to cover systematically.
        prog_cells, cell_pos = [], []
        if mode == "program_strata":
            k = strata_k if strata_k is not None else self._adaptive_k(
                self.utils.n_annots, hint_ratio)
            cell_pos = self._high_leverage_annots(k)
            if cell_pos:
                prog_cells = self._partition_programs(
                    self.local_n_programs, cell_pos)
            print(f"    [H2] {len(self.local_n_programs)} programs split by "
                  f"{len(cell_pos)} annotations -> {len(prog_cells)} cells")

        if max_seconds is not None:
            lit_to_query = self.utils.get_interest_lit()
            self.results["status"] = {lit: copy.copy(STATUS) for lit in lit_to_query}
            history = {}
            total_n_samples = 0
            total_delp_calls = 0
            total_time = 0
            total_inconsistent = 0
            total_worlds_consulted = 0

            for lit in lit_to_query:
                print(f"  -> Evaluando literal: {lit}  (mode={mode})")
                self.known_evidences = []  # reset
                exact_target = None
                if stop_at_exact is not None and lit in stop_at_exact:
                    exact_target = stop_at_exact[lit]
                (
                    lit_time,
                    lit_inconsistent,
                    lit_samples,
                    lit_worlds,
                    lit_history,
                    lit_delp_calls,
                    lit_time_to_exact,
                ) = self.consult_single_literal_time(
                    self.local_n_programs, self.adapted_annots, lit, max_seconds,
                    mode, annotation_marginals,
                    stop_at_exact=exact_target,
                    stop_epsilon=stop_epsilon,
                    prog_cells=prog_cells,
                )
                self.results["status"][lit]["time_to_exact"] = lit_time_to_exact

                total_time += lit_time
                total_inconsistent += lit_inconsistent
                total_n_samples += lit_samples
                total_worlds_consulted += lit_worlds
                total_delp_calls += lit_delp_calls
                history[lit] = lit_history

            execution_time = total_time
            inconsistent_programs = total_inconsistent
            n_samples = total_n_samples
            worlds_consulted = total_worlds_consulted
            delp_calls = total_delp_calls
            repeated_programs = 0
        elif percentile_samples == 100:
            # To compute the exact interval
            lit_to_query = self.utils.search_lit_to_consult()
            n_samples = n_programs
            unique_programs = self.local_n_programs
            # unique_programs = range(n_programs)
            repeated_programs = 0  # ????
            prog_data = self.consult_programs(
                unique_programs, self.adapted_annots, lit_to_query
            )
            execution_time, inconsistent_programs, delp_calls = prog_data
            worlds_consulted = len(self.known_evidences)
            self.results["data"] = {
                "n_samples": n_samples,
                "time": execution_time,
                "repeated_programs": repeated_programs,
                "inconsistent_programs": inconsistent_programs,
                "worlds_consulted": worlds_consulted,
                "delp_calls": delp_calls,
            }
        else:
            lit_to_query = self.utils.get_interest_lit()
            n_samples = int(get_percentile(percentile_samples, n_programs))
            sampled_programs = np.random.choice(
                self.local_n_programs, n_samples, replace=True
            )
            # sampled_programs = np.random.choice(n_programs, n_samples, replace=True)
            unique_programs = list(set(sampled_programs))
            repeated_programs = n_samples - len(unique_programs)
            prog_data = self.consult_programs(
                unique_programs, self.adapted_annots, lit_to_query
            )
            execution_time, inconsistent_programs, delp_calls = prog_data
            worlds_consulted = len(self.known_evidences)
            self.results["data"] = {
                "n_samples": n_samples,
                "time": execution_time,
                "repeated_programs": repeated_programs,
                "inconsistent_programs": inconsistent_programs,
                "worlds_consulted": worlds_consulted,
                "delp_calls": delp_calls,
            }

        if max_seconds is None:
            write_results(self.results, self.utils.save_path, info)
        else:
            self.results["data"] = {
                "n_samples": n_samples,
                "time": execution_time,
                "repeated_programs": repeated_programs,
                "inconsistent_programs": inconsistent_programs,
                "worlds_consulted": worlds_consulted,
                "delp_calls": delp_calls,
            }
            write_results(self.results, self.utils.save_path, info)
        print(self.results["data"])
        return history

    def consult_single_literal_time(
        self,
        programs: list,
        adapted_annots: dict,
        lit: str,
        max_seconds: int,
        mode: str = "random",
        annotation_marginals: dict = None,
        stop_at_exact: tuple = None,
        stop_epsilon: float = 1e-6,
        prog_cells: list = None,
    ) -> tuple:
        """
        Iterate sampling programs until the time budget is exhausted.
        Supported modes:
            "random"       — pick programs uniformly from the enumerated
                             list (current baseline).
            "bn_marginal"  — draw each rule's presence independently with
                             probability equal to its annotation's BN
                             marginal, then reject the candidate before
                             the solver call if it contains a direct
                             strict contradiction (Heuristic P).
            "program_strata" — partition the enumerated programs by the
                             annotations with the most structural leverage and
                             visit the cells in turn, drawing uniformly without
                             replacement inside each (Heuristic H2).
        """
        initial_time = time.time()
        lit_history = []
        delp_calls = 0
        inconsistent_programs = 0
        prefiltered = 0
        sampled_count = 0
        cell_cursor = 0          # round-robin position over the program cells
        pool = list(programs)    # drawn from without replacement by "random"
        np.random.shuffle(pool)
        time_to_exact = None

        while True:
            current_time = time.time() - initial_time
            if current_time >= max_seconds:
                break

            sampled_count += 1

            if mode == "program_strata" and prog_cells:
                # Round-robin over the cells, drawing uniformly without
                # replacement inside the current one. Every program keeps
                # equal weight within its cell, so the estimator is unchanged;
                # what the cells add is even coverage of the partition.
                cell = None
                for off in range(len(prog_cells)):
                    c = prog_cells[(cell_cursor + off) % len(prog_cells)]
                    if c:
                        cell, cell_cursor = c, (cell_cursor + off + 1) % len(prog_cells)
                        break
                if cell is None:
                    break                      # every cell exhausted
                sampled_prog = cell.pop()
                sampled_in_bin = self.utils.id_prog_to_bin(sampled_prog)
            elif mode == "bn_marginal" and annotation_marginals is not None:
                sampled_in_bin = self._sample_program_bin_from_marginals(
                    annotation_marginals)
            else:
                # Without replacement. The program space is enumerated and
                # finite, and re-drawing a program already queried costs a
                # solver call that cannot move the interval: its scenarios are
                # in known_evidences and contribute no further probability.
                if not pool:
                    break                      # space exhausted
                sampled_prog = pool.pop()
                sampled_in_bin = self.utils.id_prog_to_bin(sampled_prog)

            expression = ""
            for index, value in enumerate(sampled_in_bin):
                if self.utils.prog_in_bin[index] == "x":
                    if value == 1:
                        expression += adapted_annots[index]["True"] + " & "
                    else:
                        expression += adapted_annots[index]["False"] + " & "
            flag = False
            program = self.utils.map_bin_to_prog(sampled_in_bin)

            # Cheap syntactic pre-filter for Heuristic P: skip candidates
            # with obvious strict contradictions before touching the solver.
            if mode == "bn_marginal":
                if self._has_direct_strict_contradiction(program):
                    prefiltered += 1
                    inconsistent_programs += 1
                    continue

            status = query_to_delp(program, [lit])
            delp_calls += 1

            prob = float(0.0)
            models = satisfiable(eval(expression[:-3]), all_models=True)
            for model in models:
                if model:
                    evidence = to_evidence(model)
                    if evidence not in self.known_evidences:
                        prob += self.utils.em.get_sampling_prob(evidence)
                        self.known_evidences.append(evidence)
                else:
                    inconsistent_programs += 1
                    flag = True

            if not flag:
                response = status[lit]
                self.results["status"][lit][response["status"]] += 1
                self.results["status"][lit]["p" + response["status"]] += prob
                self.results["status"][lit]["time"] += response["time"]

            s = self.results["status"][lit]
            s["delp_calls"] = delp_calls
            elapsed_now = time.time() - initial_time
            l_now = s["pyes"]
            u_now = 1.0 - s["pno"]
            lit_history.append(
                {
                    "time": elapsed_now,
                    "calls": delp_calls,
                    "l": l_now,
                    "u": u_now,
                }
            )
            # Stop as soon as the running interval matches the exact one.
            if stop_at_exact is not None:
                ex_l, ex_u = stop_at_exact
                if (abs(l_now - ex_l) < stop_epsilon
                        and abs(u_now - ex_u) < stop_epsilon):
                    time_to_exact = elapsed_now
                    break

        execution_time = time.time() - initial_time
        return (
            execution_time,
            inconsistent_programs,
            sampled_count,
            len(self.known_evidences),
            lit_history,
            delp_calls,
            time_to_exact,
        )

    def consult_programs(
        self, unique_programs: list, adapted_annots: dict, lit_to_query: list
    ) -> list:
        """To iterate over sampled programs consulting for literals"""
        self.results["status"] = {lit: copy.copy(STATUS) for lit in lit_to_query}
        # To count the number of inconsistent programs sampled
        inconsistent_programs = 0
        delp_calls = 0
        counter = Counter(
            "Processing programs (%d): " % (len(unique_programs)),
            max=len(unique_programs),
        )
        initial_time = time.time()
        for sampled_prog in unique_programs:
            sampled_in_bin = self.utils.id_prog_to_bin(sampled_prog)
            # Build the program from the sampled annotations
            # self.replace_in_program(sampled_in_bin)
            # To create the expression that generate a sampled program
            expression = ""
            for index, value in enumerate(sampled_in_bin):
                if self.utils.prog_in_bin[index] == "x":
                    if value == 1:
                        expression += adapted_annots[index]["True"] + " & "
                    else:
                        expression += adapted_annots[index]["False"] + " & "
            flag = False
            # program = self.utils.map_bin_to_prog(self.utils.prog_in_bin)
            program = self.utils.map_bin_to_prog(sampled_in_bin)
            status = query_to_delp(program, lit_to_query)
            delp_calls += 1  # Count DeLP call
            prob = float(0.0)
            models = satisfiable(eval(expression[:-3]), all_models=True)
            for model in models:
                if model:
                    # The sampled program is consistent, is a valid program
                    evidence = to_evidence(model)
                    if evidence not in self.known_evidences:
                        # Get probability of the new evidence
                        prob += self.utils.em.get_sampling_prob(evidence)
                        self.known_evidences.append(evidence)
                else:
                    # To sampled program is inconsistent
                    inconsistent_programs += 1
                    flag = True
            if not flag:
                self.update_results(status, prob)
            counter.next()
        counter.finish()
        print(self.utils.model_path + " <<Complete>>")
        execution_time = time.time() - initial_time
        return [execution_time, inconsistent_programs, delp_calls]

    def replace_in_program(self, sample: list) -> None:
        """To replace annotations status in the program"""
        for index, annot in enumerate(self.utils.annotations):
            self.utils.prog_in_bin[annot] = int(sample[index])

    def update_results(self, status: dict, prob: float) -> None:
        """To update results"""
        for literal, response in status.items():
            # Update number of worlds
            self.results["status"][literal][response["status"]] += 1
            # Update probabilities
            self.results["status"][literal]["p" + response["status"]] += prob
            # Save time to compute the query in the world
            self.results["status"][literal]["time"] += response["time"]
