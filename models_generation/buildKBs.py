import sys
sys.path.insert(1, './Utils/')
sys.path.insert(2, './EM/BNs/')
import argparse
import numpy as np
from probmodels.bn import *
import pyAgrum as gum
from utils.utils import *


class CreateDeLP3E: 
    neg_prob = 0.05 # Prob to negate a variable in annotation construction
    operators = ['and', 'or'] # Operators for annotation construction

    def __init__(self, am: list, fa_ann: int, 
                        var_use_ann: int, em_var: int,
                        em_var_use_ann: int, 
                        arcs: int, alpha: int, tau: int, 
                        path_to_save: str) -> None:
        self.delp_programs = am
        self.fa_ann = fa_ann 
        self.var_use_ann = var_use_ann
        self.em_var = em_var
        self.em_var_use_ann = list(range(em_var_use_ann))
        # To consider the "True" as a possible anotation
        #self.em_var_use_ann.append("True")
        self.arcs = arcs
        self.alpha= alpha
        self.tau = tau
        self.save = path_to_save 


    def assign_labels_formulas(self, rules, to_all_rules):
        #to_all is for assign the labels only to defeasible rules or all rules
        af = []
        if to_all_rules:
            for rule in rules:
                label = np.random.random(1)[0]  # The label
                formula = self.get_simple_formula()
                af.append(
                    ['('+ rule +')::' + str(label)[:4] + ';', formula])
        else:
            for rule in rules:
                if '-<' in rule:
                    label = np.random.random(1)[0]
                else:
                    label = 1.00
                formula = self.get_simple_formula()
                af.append(
                    ['('+ rule +')::' + str(label)[:4] + ';', formula])
        return af
    

    def filter_rules(self, rules: list) -> list:
        filtered_rules = []

        if self.fa_ann == 0:
            # Only the facts and presumptions are annotated
            filtered_rules = [rule for rule in rules if '<- true' in rule or 
                                                        '-< true' in rule]
        elif self.fa_ann < 100:
            perc = int((self.fa_ann * len(rules)) / 100)
            filtered_rules = np.random.choice(rules, perc)
        else:
            # All delp program is annotated
            filtered_rules = rules

        return filtered_rules


    def get_variables(self, number: int):
        variables = []
        if number != 1:
            rnd_vars = np.random.choice(self.em_var_use_ann, number, replace=False)
        else:
            rnd_vars = np.random.choice(self.em_var_use_ann + ['True'], number)
        for var in rnd_vars:
            rnd_negation = np.random.random()
            if rnd_negation < self.neg_prob:
                variables.append("not " + str(var))
            else:
                variables.append(str(var))
        return variables


    def get_annotation(self) -> str:
        if self.var_use_ann == 1:
            # Build annotation with one variable (0 operators)
            annotation = self.get_variables(1)[0]
        elif self.var_use_ann == 2:
            # Build annotation with two variables (1 operator)
            variables = self.get_variables(2)
            operator = np.random.choice(self.operators, 1)[0]
            annotation = " ".join([variables[0], operator, variables[1]])
        elif self.var_use_ann == 3:
            # Build annotation with three variables (2 operators)
            variables = self.get_variables(3)
            operators = np.random.choice(self.operators, 2, replace=True)
            annotation = " ".join([variables[0], operators[0], variables[1], 
                                                operators[1], variables[2]])
        return annotation


    def build_BN(self, am_name: str) -> None:
        nodes = self.em_var
        arcs = self.arcs
        alpha = self.alpha
        tau = self.tau  # ???
        #For build a random Bayesian Network
        bayesian_network = BayesNetwork('BN' + am_name, self.save)
        bayesian_network.build_save_random_BN(nodes, arcs, True)
        if alpha != 0:
            # To change the entropy
            bayesian_network.make_CPTs(bayesian_network.structure[0], alpha)


    def create(self) -> None:
        created_models = 0
        print("Building models...")
        for p_index, path_delp in enumerate(self.delp_programs):
            delp_program = read_json_file(path_delp)
            delp_rules = [rule[:-1] for rule in delp_program['delp']]
            rule_to_annot = self.filter_rules(delp_rules)
            af = []
            for rule in delp_rules:
                if rule in rule_to_annot:
                    annotation = self.get_annotation()
                    af.append([rule + ';', annotation])
                else:
                    af.append([rule + ';', ""])
            
            # Create the Environmental Model
            if self.arcs != 0:
                # Create the em with a Bayesian Network
                self.build_BN(gfn(path_delp)[:-9])
            else:
                # Create the em with a Tuple-Independence Model
                self.build_BN(gfn(path_delp)[:-9])
                pass

            # To save the delp3e model
            delp3e_model = {
                    "em_var": self.em_var,
                    "em_var_use_ann": self.em_var_use_ann,
                    "literals": delp_program['literals'],
                    "af": af
                    }
            
            # Save the delp3e file
            with open(self.save + str(p_index) + 'model' + '.json','w') as outfile:
                json.dump(delp3e_model, outfile, indent = 4)
            created_models += 1   
        print("Models created: " + str(created_models))
