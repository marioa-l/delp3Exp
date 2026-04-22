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

    def start_sampling(
        self, percentile_samples: int, info: str, max_seconds: int = None
    ) -> list:
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
                print(f"  -> Evaluando literal: {lit}")
                self.known_evidences = []  # reset
                (
                    lit_time,
                    lit_inconsistent,
                    lit_samples,
                    lit_worlds,
                    lit_history,
                    lit_delp_calls,
                ) = self.consult_single_literal_time(
                    self.local_n_programs, self.adapted_annots, lit, max_seconds
                )

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
    ) -> tuple:
        initial_time = time.time()
        lit_history = []
        delp_calls = 0
        inconsistent_programs = 0
        sampled_count = 0

        while True:
            current_time = time.time() - initial_time
            if current_time >= max_seconds:
                break

            sampled_prog = np.random.choice(programs, 1, replace=True)[0]
            sampled_count += 1
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
            lit_history.append(
                {
                    "time": time.time() - initial_time,
                    "calls": delp_calls,
                    "l": s["pyes"],
                    "u": 1.0 - s["pno"],
                }
            )

        execution_time = time.time() - initial_time
        return (
            execution_time,
            inconsistent_programs,
            sampled_count,
            len(self.known_evidences),
            lit_history,
            delp_calls,
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
