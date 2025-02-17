import copy
from utils.utils import *
from progress.counter import Counter
from delp.consultDeLP import *
import time
import numpy as np


class Worlds:
    def __init__(self, model_path: str, save_path: str):
        # Utils to handle model
        self.utils = Model(model_path, save_path)
        # To save all results
        self.results = {}
        # To control repeated programs generates by worlds
        self.known_progs = KnownSamples()

    def start_sampling(self, percentile_samples: int, source: str, info: str) -> None:
        """To run exact compute of the interval or select randomly a subset of all
        possible worlds to perform an approximation of the exact interval"""
        # Total number of possible worlds
        n_worlds = self.utils.get_n_worlds()
        if percentile_samples == 100:
            # To compute the exact interval
            lit_to_query = self.utils.search_lit_to_consult()
            n_samples = n_worlds
            unique_worlds = range(n_samples)
            repeated_worlds = 0
        else:
            lit_to_query = self.utils.get_interest_lit()
            n_samples = int(get_percentile(percentile_samples, n_worlds))
            if source == 'distribution':
                # Sample from Probability Distribution Function
                unique_worlds, repeated_worlds = self.utils.em.gen_samples(n_samples)
            else:
                # Sample worlds randomly
                sampled_worlds = np.random.choice(n_worlds, n_samples, replace=True)
                unique_worlds = list(set(sampled_worlds))
                repeated_worlds = n_samples - len(unique_worlds)
        # Consult in each sampled world
        execution_time = self.consult_worlds(unique_worlds, lit_to_query)
        self.results['data'] = {
            'n_samples': n_samples,
            'time': execution_time,
            'repeated_worlds': repeated_worlds,
            'repeated_progs': n_samples - self.known_progs.get_unique_samples(),
            'unique_progs': self.known_progs.get_unique_samples()
        }
        write_results(self.results, self.utils.save_path, info)

    def consult_worlds(self, worlds: list, lit_to_query: list) -> float:
        """To iterate over sampled worlds consulting for literals"""
        self.results['status'] = {lit: copy.copy(STATUS) for lit in lit_to_query}
        # To control if worlds are sampled or generated
        if isinstance(worlds[0], (int, np.int64)):
            to_convert = 'self.utils.id_world_to_bin(sampled_world)'
        else:
            to_convert = 'sampled_world'
        counter = Counter('Processing worlds: ', max=len(worlds))
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
                    self.results['status'][literal][response['status']] += 1
                    # Update probabilities
                    self.results['status'][literal]['p' + response['status']] += prob_world
                    # Save time to compute the query in the world 
                    self.results['status'][literal]['time'] += response['time']
            else:
                # Known program
                for literal, response in status.items():
                    # Update number of worlds
                    self.results['status'][literal][response['status']] += 1
                    # Update probabilities
                    self.results['status'][literal]['p' + response['status']] += prob_world
                    # Save time to compute the query in the world 
                    self.results['status'][literal]['time'] += 0 
            counter.next()
        counter.finish()
        print(self.utils.model_path + " <<Complete>>")
        execution_time = time.time() - initial_time
        return execution_time
