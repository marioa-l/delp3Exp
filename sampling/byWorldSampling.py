import copy
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

    def start_sampling(
        self, percentile_samples: int, source: str, info: str, max_seconds: int = None
    ) -> dict:
        print(f"--- Sampleando modelo/programa: {self.utils.model_path} ---")
        """Permite samplear por porcentaje o por tiempo (en segundos)"""
        n_worlds = self.utils.get_n_worlds()
        lit_to_query = None
        n_samples = 0
        unique_worlds = []
        repeated_worlds = 0
        history = {}

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
                print(f"  -> Evaluando literal: {lit}")
                self.known_progs = (
                    KnownSamples()
                )  # Reset cache para aislar el experimento
                lit_n_samples, lit_time, lit_history, lit_delp_calls, lit_rep_worlds = (
                    self.consult_single_literal_time(n_worlds, lit, max_seconds)
                )

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
        self, n_worlds: int, lit: str, max_seconds: int
    ) -> tuple:
        """Itera sobre worlds generados al azar consultando un único literal hasta agotar el tiempo"""
        initial_time = time.time()
        lit_history = []
        delp_calls = 0
        sampled_count = 0
        sampled_ids = set()

        while True:
            current_time = time.time() - initial_time
            if current_time >= max_seconds:
                break

            sampled_world = np.random.choice(n_worlds, 1, replace=True)[0]
            sampled_count += 1

            # Skip repeated worlds — only process each world once
            if sampled_world in sampled_ids:
                continue
            sampled_ids.add(sampled_world)

            if isinstance(sampled_world, (int, np.int64)):
                world, evidence = self.utils.id_world_to_bin(sampled_world)
            else:
                world, evidence = sampled_world

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
            lit_history.append(
                {
                    "time": time.time() - initial_time,
                    "calls": delp_calls,
                    "l": s["pyes"],
                    "u": 1.0 - s["pno"],
                }
            )

        execution_time = time.time() - initial_time
        repeated_worlds = sampled_count - len(sampled_ids)
        return sampled_count, execution_time, lit_history, delp_calls, repeated_worlds

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
