import copy
import json
import os
import re
import time

import numpy as np
from progress.counter import Counter

from delp.consultDeLP import *
from utils.utils import *


class Worlds:
    def __init__(self, model_path: str, save_path: str):
        # Utils to handle model
        self.utils = Model(model_path, save_path)
        # To save all results
        self.results = {}
        # To control repeated programs generates by worlds
        self.known_progs = KnownSamples()
        # Optional cache for the argument-influence table (populated lazily
        # when a heuristic mode needs it).
        self._influence_cache = None

    # ── Heuristic W helpers ──────────────────────────────────────────────────
    def _load_influence_from_dgraph(self, dgraph_path: str):
        """
        Build a mapping ``rule_index -> influence_score`` from a cached
        solver dGraph. The influence score of a rule is the in-degree, in
        the AM attack graph, of the argument that has the rule's head as
        its conclusion. Rules whose head does not appear in any attacked
        argument get score 0.
        """
        if not os.path.exists(dgraph_path):
            return {}
        with open(dgraph_path) as f:
            raw = json.load(f)
        # Build a directed attack graph: node = argument id (string),
        # edge = defeat (attacker -> defended). Count in-degrees.
        in_degree = {}
        for entry in raw.get("dGraph", []):
            lit = next(iter(entry))
            for argument in entry[lit]:
                arg_key = next(iter(argument))
                ad = argument[arg_key]
                arg_id = ad.get("id", arg_key)
                in_degree.setdefault(arg_id, 0)
                for d in ad.get("defeats", []):
                    target = d.get("defeat") if isinstance(d, dict) else d
                    if target:
                        in_degree[target] = in_degree.get(target, 0) + 1
        # For each rule of the AF (annotated OR not — we key by index into
        # utils.model), look up the in-degree of the argument concluding its
        # head literal.
        rule_scores = {}
        for i, entry in enumerate(self.utils.model):
            rule_str = entry[0]
            head = self._rule_head(rule_str)
            if head is None:
                rule_scores[i] = 0
                continue
            best = 0
            for arg_id, deg in in_degree.items():
                if head in arg_id:
                    best = max(best, deg)
            rule_scores[i] = best
        return rule_scores

    @staticmethod
    def _rule_head(rule_str: str):
        if " -< " in rule_str:
            return rule_str.split(" -< ", 1)[0].strip()
        if " <- " in rule_str:
            return rule_str.split(" <- ", 1)[0].strip()
        return None

    def _influence_of_world(self, world_bin) -> float:
        """
        Sum the influence score of every annotated rule active under the
        world. If no influence table is available, returns 1 so the
        centrality factor becomes neutral.
        """
        if not self._influence_cache:
            return 1.0
        score = 0.0
        for i, bit in enumerate(world_bin):
            if i in self._influence_cache and bit == 1:
                score += self._influence_cache[i]
        return max(score, 1.0)  # keep it strictly positive

    def start_sampling(
        self, percentile_samples: int, source: str, info: str,
        max_seconds: int = None, mode: str = "random",
        dgraph_path: str = None,
        stop_at_exact: dict = None,
        stop_epsilon: float = 1e-6,
    ) -> dict:
        """
        `stop_at_exact` is an optional mapping ``{literal: (exact_l, exact_u)}``.
        When set, per-literal sampling stops as soon as the running interval
        matches the exact bounds within `stop_epsilon`. The observed
        `time_to_exact` is stored in ``self.results["status"][lit]``.
        """
        print(f"--- Sampleando modelo/programa: {self.utils.model_path} ---")
        """Permite samplear por porcentaje o por tiempo (en segundos)"""
        n_worlds = self.utils.get_n_worlds()
        lit_to_query = None
        n_samples = 0
        unique_worlds = []
        repeated_worlds = 0
        history = {}

        # If a heuristic mode was requested, prepare its state.
        if mode == "bn_centrality":
            if dgraph_path is not None:
                self._influence_cache = self._load_influence_from_dgraph(dgraph_path)
            else:
                self._influence_cache = {}
        else:
            self._influence_cache = None

        if max_seconds is not None:
            # Samplear por tiempo de forma aislada por cada literal
            lit_to_query = self.utils.get_interest_lit()
            self.results["status"] = {lit: copy.copy(STATUS) for lit in lit_to_query}

            total_n_samples = 0
            total_time = 0
            total_delp_calls = 0
            total_repeated_worlds = 0
            total_unique_progs = 0

            for lit in lit_to_query:
                print(f"  -> Evaluando literal: {lit}  (mode={mode})")
                self.known_progs = (
                    KnownSamples()
                )  # Reset cache para aislar el experimento
                exact_target = None
                if stop_at_exact is not None and lit in stop_at_exact:
                    exact_target = stop_at_exact[lit]
                (lit_n_samples, lit_time, lit_history, lit_delp_calls,
                 lit_rep_worlds, lit_time_to_exact) = (
                    self.consult_single_literal_time(
                        n_worlds, lit, max_seconds, mode,
                        stop_at_exact=exact_target,
                        stop_epsilon=stop_epsilon,
                    )
                )
                self.results["status"][lit]["time_to_exact"] = lit_time_to_exact

                total_n_samples += lit_n_samples
                total_time += lit_time
                total_delp_calls += lit_delp_calls
                total_repeated_worlds += lit_rep_worlds
                total_unique_progs += self.known_progs.get_unique_samples()
                history[lit] = lit_history

            execution_time = total_time
            n_samples = total_n_samples
            repeated_worlds = total_repeated_worlds
            delp_calls = total_delp_calls
            unique_progs_count = total_unique_progs
        else:
            if percentile_samples == 100:
                # To compute the exact interval
                lit_to_query = self.utils.search_lit_to_consult()
                n_samples = n_worlds
                unique_worlds = range(n_samples)
                repeated_worlds = 0
                execution_time = self.consult_worlds(unique_worlds, lit_to_query)
                delp_calls = self.known_progs.get_unique_samples()
                unique_progs_count = self.known_progs.get_unique_samples()
            else:
                lit_to_query = self.utils.get_interest_lit()
                n_samples = int(get_percentile(percentile_samples, n_worlds))
                if source == "distribution":
                    # Sample from Probability Distribution Function
                    unique_worlds, repeated_worlds = self.utils.em.gen_samples(
                        n_samples
                    )
                else:
                    # Sample worlds randomly
                    sampled_worlds = np.random.choice(n_worlds, n_samples, replace=True)
                    unique_worlds = list(set(sampled_worlds))
                    repeated_worlds = n_samples - len(unique_worlds)
                    # Consult in each sampled world
                    execution_time = self.consult_worlds(unique_worlds, lit_to_query)
                unique_progs_count = self.known_progs.get_unique_samples()
                delp_calls = unique_progs_count

        self.results["data"] = {
            "n_samples": n_samples,
            "time": execution_time,
            "repeated_worlds": repeated_worlds,
            "repeated_progs": n_samples - unique_progs_count,
            "unique_progs": unique_progs_count,
            "delp_calls": delp_calls,
        }
        write_results(self.results, self.utils.save_path, info)
        print(f"Resultados guardados en {self.utils.save_path}")
        print("Resumen de resultados:")
        print(self.results["data"])
        return history

    def consult_single_literal_time(
        self, n_worlds: int, lit: str, max_seconds: int,
        mode: str = "random",
        stop_at_exact: tuple = None,
        stop_epsilon: float = 1e-6,
    ) -> tuple:
        """
        Iterate sampling worlds until the time budget is exhausted.
        Two modes are supported:
            "random"        — the current uniform random sampling of world IDs.
            "bn_centrality" — importance sampling: worlds are drawn directly
                              from the BN's joint distribution and accepted
                              proportionally to their argument-influence
                              score (Heuristic W). Each unique world is
                              still processed at most once; the influence
                              score modulates rejection so decisive worlds
                              are visited first.
        """
        initial_time = time.time()
        lit_history = []
        delp_calls = 0
        sampled_count = 0
        sampled_ids = set()
        time_to_exact = None  # populated when the running interval matches exact

        # Buffer for BN-drawn worlds when using the heuristic.
        bn_buffer = []
        BN_BATCH = 200
        influence_norm = None  # highest observed influence, used to normalize

        while True:
            current_time = time.time() - initial_time
            if current_time >= max_seconds:
                break

            sampled_count += 1

            if mode == "bn_centrality":
                if not bn_buffer:
                    try:
                        unique_batch, _ = self.utils.em.gen_samples(BN_BATCH)
                    except Exception:
                        unique_batch = []
                    bn_buffer.extend(unique_batch)
                if not bn_buffer:
                    # Fallback to uniform if BN sampler failed
                    sampled_world = int(np.random.choice(n_worlds, 1, replace=True)[0])
                    if sampled_world in sampled_ids:
                        continue
                    sampled_ids.add(sampled_world)
                    world, evidence = self.utils.id_world_to_bin(sampled_world)
                else:
                    entry = bn_buffer.pop(0)
                    world = entry[0]
                    world_key = tuple(world)
                    if world_key in sampled_ids:
                        continue
                    # Compute the influence-based acceptance probability.
                    infl = self._influence_of_world(world)
                    if influence_norm is None or infl > influence_norm:
                        influence_norm = max(infl, 1.0)
                    accept_prob = infl / influence_norm
                    if np.random.random() > accept_prob:
                        continue
                    sampled_ids.add(world_key)
                    evidence = {i: int(v) for i, v in enumerate(world)}
            else:
                sampled_world = int(np.random.choice(n_worlds, 1, replace=True)[0])
                if sampled_world in sampled_ids:
                    continue
                sampled_ids.add(sampled_world)
                world, evidence = self.utils.id_world_to_bin(sampled_world)

            prob_world = self.utils.em.get_sampling_prob(evidence)
            program, id_program = self.utils.map_world_to_prog(world)

            status = self.known_progs.search_sample(id_program)
            if status == -1:
                status = query_to_delp(program, [lit])
                delp_calls += 1
                self.known_progs.save_sample(id_program, status)

            response = status[lit]
            self.results["status"][lit][response["status"]] += 1
            self.results["status"][lit]["p" + response["status"]] += prob_world
            self.results["status"][lit]["time"] += response["time"]

            s = self.results["status"][lit]
            s["delp_calls"] = delp_calls
            # l = pyes, u = 1 - pno
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
        repeated_worlds = sampled_count - len(sampled_ids)
        return (sampled_count, execution_time, lit_history, delp_calls,
                repeated_worlds, time_to_exact)

    def consult_worlds(self, worlds: list, lit_to_query: list) -> float:
        """To iterate over sampled worlds consulting for literals"""
        self.results["status"] = {lit: copy.copy(STATUS) for lit in lit_to_query}
        # To control if worlds are sampled or generated
        if isinstance(worlds[0], (int, np.int64)):
            to_convert = "self.utils.id_world_to_bin(sampled_world)"
        else:
            to_convert = "sampled_world"
        counter = Counter("Processing worlds: ", max=len(worlds))
        initial_time = time.time()
        for sampled_world in worlds:
            # Get world in list format
            world, evidence = eval(to_convert)
            # Get the probability of the world
            prob_world = self.utils.em.get_sampling_prob(evidence)
            # Build the program for world
            program, id_program = self.utils.map_world_to_prog(world)
            status = self.known_progs.search_sample(id_program)
            if status == -1:
                # New program
                status = query_to_delp(program, lit_to_query)
                self.known_progs.save_sample(id_program, status)
                for literal, response in status.items():
                    # Update number of worlds
                    self.results["status"][literal][response["status"]] += 1
                    # Update probabilities
                    self.results["status"][literal]["p" + response["status"]] += (
                        prob_world
                    )
                    # Save time to compute the query in the world
                    self.results["status"][literal]["time"] += response["time"]
            else:
                # Known program
                for literal, response in status.items():
                    # Update number of worlds
                    self.results["status"][literal][response["status"]] += 1
                    # Update probabilities
                    self.results["status"][literal]["p" + response["status"]] += (
                        prob_world
                    )
                    # Save time to compute the query in the world
                    self.results["status"][literal]["time"] += 0
            counter.next()
        counter.finish()
        print(self.utils.model_path + " <<Complete>>")
        execution_time = time.time() - initial_time
        return execution_time
