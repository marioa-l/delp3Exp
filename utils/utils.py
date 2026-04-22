import json
import os
import re
from collections import OrderedDict

import numpy as np

from probmodels.bn import BayesNetwork

"""Generals Names and Values"""
BN_NAMES = "BN"
WIDTH_OF_INTEREST = 0.5
STATUS = {
    "yes": 0,
    "no": 0,
    "undecided": 0,
    "unknown": 0,
    "pyes": 0.0,
    "pno": 0.0,
    "pundecided": 0.0,
    "punknown": 0.0,
    "time": 0.0,
}
"""General Utils"""


def read_json_file(path_file: str) -> json:
    """To read a json file"""
    try:
        file = open(path_file, "r")
        to_dict = json.load(file)
        file.close()
        return to_dict
    except IOError:
        print("Error trying to open the file %s" % path_file)
        exit()
    except ValueError:
        print("JSON incorrect format: %s" % path_file)
        exit()


def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def gfn(path: str) -> str:
    """Get the file name of a model specified in path"""
    return os.path.basename(path)


def gdn(path: str) -> str:
    """Get the directory name in a specified path"""
    return os.path.dirname(path) + "/"


def gfnexact(path: str) -> str:
    """Get the file name that contains the exact values of the model specified in path"""
    dir_name = os.path.dirname(path)
    model_name = gfn(path)[:-5]
    return dir_name + "/exact/" + model_name + "_e_wNEW.json"


def gfnexact_from_sampling(path: str) -> str:
    dir_name = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    model_name = gfn(path)[:-9]
    return dir_name + "/exact/" + model_name + "_e_wNEW.json"


def to_decimal_format(number: float, decimals: int) -> str:
    """To format a float number"""
    return str(format(number, "." + str(decimals) + "f")).replace(".", ",")


def bin_to_int(in_binary: str) -> int:
    return int(in_binary, 2)


def gbn(index: str) -> str:
    """To return the Bayesian Network name of a model"""
    return BN_NAMES + index


def get_percentile(percentile: int, total: int) -> float:
    """Return the <percentile> percentile of <total> (the integer part)"""
    return (percentile * total) / 100


def write_results(results: json, path: str, approach: str) -> None:
    """To compute and save results"""
    n_samples = results["data"]["n_samples"]
    for lit, status in results["status"].items():
        status["percY"] = get_percentile(status["yes"], n_samples)
        status["percN"] = get_percentile(status["no"], n_samples)
        status["percU"] = get_percentile(status["undecided"], n_samples)
        status["percUNK"] = get_percentile(status["unknown"], n_samples)
        status["l"] = status["pyes"]
        status["u"] = 1 - status["pno"]
        if status["u"] - status["l"] <= WIDTH_OF_INTEREST:
            status["flag"] = "INTEREST"
    with open(path + approach + "NEW.json", "w") as outfile:
        json.dump(results, outfile, indent=4)


def compute_metric(approximate: list, exact: list):
    """Compute the metric"""
    # approximate = [l,u]
    # exact = [l,u]
    width_approximate = approximate[1] - approximate[0]
    width_exact = exact[1] - exact[0]
    remainder_approximate = 1 - width_approximate
    remainder_exact = 1 - width_exact
    if remainder_exact == 0:
        metric = 0
    else:
        metric = remainder_approximate / remainder_exact
    return "{:.4f}".format(metric)


def format_annot(annot: str, world: list) -> str:
    """To transform annot into an expression from the values of a world"""
    to_eval = ""
    aux = annot.strip().split(" ")
    for element in aux:
        try:
            if world[int(element)] == 1:
                var_status = "True"
            else:
                var_status = "False"
        except ValueError:
            var_status = element

        to_eval = to_eval + " " + var_status

    return to_eval


def eval_annot(annot: str, world: list) -> bool:
    """To evaluate an annotation"""
    if annot == "" or annot == "True":
        return True
    elif annot == "not True":
        return False
    else:
        formatted_annot = format_annot(annot, world)
        return eval(formatted_annot)


def is_trivial_annot(annot: str) -> bool:
    """To evaluate if an annotation is 'trivial'"""
    if annot == "True" or annot == "" or annot == "not True":
        return True
    else:
        return False


def get_prog_info(model: list) -> list:
    """To return program info:
    - Number of possible programs
    - A dictionary that indicate the positions of annotations in the program"""
    annotations = OrderedDict()
    program_in_bin = []
    for index, annot in enumerate(model):
        if not is_trivial_annot(annot[1]):
            annotations[index] = annot[1]
            program_in_bin.append("x")
        else:
            if annot[1] == "True" or annot[1] == "":
                program_in_bin.append(1)
            else:
                program_in_bin.append(0)
    n_annots = len(annotations)
    return [n_annots, annotations, program_in_bin]


class Model:
    """Class for model handling"""

    def __init__(self, model_path: str, save_path: str):
        self.model_path = model_path
        model_data = read_json_file(model_path)
        # The rule and annotations
        self.model = model_data["af"]
        # The number of EM variables in the model
        self.em_vars = model_data["em_var"]
        # The number of rules in the model
        self.am_rules_dim = len(self.model)
        # Info of the program in the model
        self.n_annots, self.annotations, self.prog_in_bin = get_prog_info(self.model)
        # All literals used in the AM model
        self.literals_in_model = model_data["literals"]
        # To load the Bayesian Network of the model
        index_model = re.search(r"\d+", gfn(model_path)).group()
        self.em = BayesNetwork(gbn(index_model), gdn(model_path))
        self.em.load_bn()
        self.save_path = save_path + gfn(model_path)[:-5]
        self.to_bin_prog_format = "{0:0" + str(self.n_annots) + "b}"
        self.to_bin_world_format = "{0:0" + str(self.em_vars) + "b}"

    def get_n_worlds(self) -> int:
        """Return the total number of possible worlds"""
        return pow(2, self.em_vars)

    def get_n_programs(self) -> int:
        """Return the total number of possible programs"""
        return pow(2, self.n_annots)

    def id_prog_to_bin(self, id_prog: int) -> list:
        """To convert the id of a prog into a binary array"""
        prog = [int(digit) for digit in list(self.to_bin_prog_format.format(id_prog))]
        return prog

    def id_world_to_bin(self, id_world: int) -> list:
        """To convert the id of a world into a binary array and it's evidence"""
        world = [
            int(digit) for digit in list(self.to_bin_world_format.format(id_world))
        ]
        evidence = {i: world[i] for i in range(len(world))}
        return [world, evidence]

    def map_bin_to_prog(self, bin_array: list) -> str:
        """Map a binary array into a prog"""
        prog = ""
        for index, value in enumerate(bin_array):
            if value == 1:
                # Add the rule
                prog += self.model[index][0]
        return prog

    def map_world_to_prog(self, world: list) -> list:
        """Map a world (in binary representation) into a prog"""
        prog = ""
        prog_in_bin = "0b"
        for rule, annot in self.model:
            check_annot = eval_annot(annot, world)
            if check_annot:
                prog += rule
                prog_in_bin += "1"
            else:
                prog_in_bin += "0"
        id_prog = bin_to_int(prog_in_bin)
        return [prog, id_prog]

    def get_interest_lit(self) -> list:
        """Return the literals with interest intervals of the model
        (for sampling)"""
        exact_file_name = gfnexact(self.model_path)
        literals = read_json_file(exact_file_name)["status"].keys()
        return literals

    def search_lit_to_consult(self):
        """Return all unique literals present in the model"""
        literals = set()
        for level in self.literals_in_model.values():
            for lit in level:
                literals.add(lit)
        return list(literals)


class KnownSamples:
    def __init__(self):
        self.samples = {}

    def search_sample(self, id_sample: int):
        try:
            return self.samples[id_sample]
        except KeyError:
            return -1

    def save_sample(self, id_sample: int, data: json) -> None:
        self.samples[id_sample] = data

    def get_unique_samples(self) -> int:
        return len(self.samples)
