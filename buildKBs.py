import sys
sys.path.insert(1, './Utils/')
sys.path.insert(2, './EM/BNs/')
from utilsExp import *
import argparse
import numpy as np
from bn import *
import pyAgrum as gum

var_to_use_probs = {}
neg_probs = [0.80, 0.20] # [Prob for literal, Prob for negated literal]

def assign_labels_formulas(rules, to_all_rules):
    #to_all is for assign the labels only to defeasible rules or all rules
    af = []
    if to_all_rules:
        for rule in rules:
            label = np.random.random(1)[0]  # The label
            formula = get_simple_formula()
            af.append(
                ['('+ rule +')::' + str(label)[:4] + ';', formula])
    else:
        for rule in rules:
            if '-<' in rule:
                label = np.random.random(1)[0]
            else:
                label = 1.00
            formula = get_simple_formula()
            af.append(
                ['('+ rule +')::' + str(label)[:4] + ';', formula])
    return af


def assign_formulas(rules):
    af = []
    for rule in rules:
        formula = get_simple_formula()
        af.append([rule + ';', formula])
    return af


def build_BN(nodes, arcs, alpha_entropy, path_to_save):
    #For build a random Bayesian Network
    bayesian_network = BayesNetwork('BN',path_to_save)
    bayesian_network.build_save_random_BN(nodes, arcs, True)
    if alpha_entropy != 0:
        # To change the entropy
        bayesian_network.make_CPTs(bayesian_network.structure[0], alpha_entropy)


def main(data, nvar, nvaruse, with_labels, path_to_save):
    if(nvaruse <= nvar):
        # Generate variables
        randomVar = [str(var) for var in list(range(nvar))]
        # Get all rules from the delp program
        rules = [rule[:-1] for rule in data["delp"][1:]]
        # Get the first nvaruse from randomVar
        randomVarToUse = randomVar[:nvaruse]
        randomVarToUse.append('True')
        # Assign probabilities to each variables to use
        probs = []
        var_to_use_probs["variables"] = randomVarToUse
        var_to_use_probs["probs"] = probs
        if with_labels:
            af = assign_labels_formulas(rules, False)
        else:
            af = assign_formulas(rules)

        program = {
            "random_var": randomVar,
            "var_used": randomVarToUse,
            "af": af
        }

        write_json_file(program, path_to_save + 'KB')   # Save the KB

        nodes = nvar #  Number of nodes for the Bayesian Network
        arcs = nvar #   Max number of arcs for the Bayesian Network
        alpha_entropy = 0 # 0 Max entropy, 1 Min entropy
        build_BN(nodes, arcs, alpha_entropy, path_to_save)
        print("KB generated")
    else:
        print_error_msj("Error: nvaruse > nvar")
        exit()

def get_simple_formula():
    # To build simple formulas with one atom
    if len(var_to_use_probs["probs"]) != 0:
        variable = np.random.choice(var_to_use_probs["variables"],
                                    1,
                                    p = var_to_use_probs["probs"],
                                    replace = True)
    else:
        variable = np.random.choice(var_to_use_probs["variables"],
                                    1,
                                    replace = True)
    negation = np.random.choice(["","not "], 1, p = neg_probs, replace = True)
    formula = str(negation[0]) +str(variable[0])
    return formula

def get_formula():
    # To build formulas with two atoms and one operator
    if(len(probs) > 0):
        atoms = np.random.choice(variables, 2, p= probs, replace=True)
    else:
        atoms = np.random.choice(variables, 2, replace=True)

    if 'True' in atoms:
        return 'True'
    else:
        operator = np.random.choice(['and','or'], 1, replace=True)
        return str(atoms[0] + ' ' + operator[0] + ' ' + atoms[1])


parser = argparse.ArgumentParser(description='Script to generate annotations \
                                                randomly for a del3e program')
parser.add_argument('-delppath',
                    action='store',
                    help="The delp3e program",
                    dest="program",
                    type=getDataFromFile,
                    required=True)
parser.add_argument('-nvar',
                    help='Number of elements',
                    dest="nvar",
                    type=int,
                    required=True)
parser.add_argument('-nvaruse',
                    help='Number of elements to use in each formula',
                    dest="nvaruse",
                    type=int,
                    required=True)
parser.add_argument('-labels',
                    help='To generate labels for each rule',
                    dest='label',
                    type=bool,
                    required=False)
parser.add_argument('-operators',
                    help='Operator to use (NOT IMPLEMENTED)',
                    dest="operators",
                    required=False)
parser.add_argument('-outpath',
                    help='Path for the output files',
                    dest="path_to_save",
                    required=True)
arguments = parser.parse_args(sys.argv)


main(arguments.program, arguments.nvar,
    arguments.nvaruse, False, arguments.path_to_save)

##### To call from other script #####
# sys.argv = [
# '-delppath', 'DeLP Program path',
# '-nvar', 'Int',
# '-nvaruse', 'Int',
# '-outpath', 'Result path'
# ]