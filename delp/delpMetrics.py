import csv
import sys
import time
from subprocess import STDOUT, check_output
from utils import *
import re
import random


def get_random_queries(literals_dicts: dict, perc: int):
    """
    Randomly select a percentage of literals per level to query.

    :param literals_dicts: A dictionary with all literals per level
    :param perc: Percentile to select
    """
    literals = []
    in_string_literals = '['
    for level, lits in literals_dicts.items():
        n_literals_to_choose = int((float(perc) * float(len(lits))) / 100)
        literals_to_consult = random.sample(lits, n_literals_to_choose)
        literals.append(literals_to_consult)
        for lit in literals_to_consult:
            in_string_literals += lit + ','
    in_string_literals = in_string_literals[:-1] + ']'
    print(in_string_literals)
    return [in_string_literals, literals]


class ComputeMetrics:

    def __init__(self,
                 path_file_results: str,
                 file_results_name: str,
                 path_dataset: str,
                 path_delp: str) -> None:
        """
        The constructor for the experiment class 
        Args:
            -path_file_results: The path for save results files
            -file_results_name: The name of the result file
        """
        self.path_file_results = path_file_results
        self.file_results_name = file_results_name
        self.path_dataset = path_dataset
        self.path_delp = path_delp
        self.aux_height = []
        self.times = []
        self.rules = []
        self.facts_presumptions = []
        # self.patter_presumption = \[\([a-zA-Z]*\-\<[t|T]rue\)\]

    def show_setting(self) -> None:
        """
        Show experiment settings
        """
        print_info("Output path: " + self.path_file_results)
        print_info("Result file: " + self.file_results_name)
        print_info("Dataset: " + self.path_dataset)
        print_info("Delp program: " + self.path_delp)

    def build_path_result(self) -> str:
        """
        Build the path where the results will be saved
        """
        return self.path_file_results + self.file_results_name + '.json'

    def query_literal_solver(self, literal: str) -> str:
        """
        Call to delp solver to query for one literal
        Args:
            -literal: The literal to consult
        """
        delp_program = self.path_delp
        cmd = ['./globalCore', 'file', delp_program, 'answ', literal]
        try:
            output = check_output(cmd, stderr=STDOUT, timeout=60). \
                decode(sys.stdout.encoding)
            result = output
            return result
        except Exception as e:
            print(e)
            exit()

    def query_delp_solver_approximation(self, perc) -> json:
        delp_program = self.path_delp
        print("\nProgram: ", delp_program)
        delp_program_json = delp_program.replace(".delp", ".json")
        program_literals = get_data_from_file(delp_program_json)
        program_literals = program_literals["literals"]
        literals_to_query = get_random_queries(program_literals, perc)
        cmd = ['./globalCore', 'file', delp_program, literals_to_query[0]]
        try:
            # TimeOut 15 minutes
            output = check_output(cmd, stderr=STDOUT, timeout=900). \
                decode(sys.stdout.encoding)
            result = json.loads(output)
            return result
        except Exception as e:
            print("Exception: ", e)
            return json.loads('{"status":"","dGraph":""}')

    def query_delp_solver(self, perc: float) -> json:
        """
        Call to delp solver to get all answers for the delp program
        in self.path_delp
        """
        delp_program = self.path_delp
        print("\nProgram: ", delp_program)
        cmd = ['./globalCore', 'file', delp_program, 'all']
        try:
            # TimeOut 15 minutes
            output = check_output(cmd, stderr=STDOUT, timeout=900). \
                decode(sys.stdout.encoding)
            result = json.loads(output)
            return result
        except Exception as e:
            print("TimeOut:")
            return json.loads('{"status":"","dGraph":""}')

    def get_size_metrics(self) -> list:
        """
        Count the number of rules and facts and presumptions in the program
        Output:
            -list: [#rules, #factsandpresum]
        """
        delp = open(self.path_delp, 'r').read().replace('\n', '')
        facts = len(re.findall('<- true.', delp))
        presumptions = len(re.findall('-< true.', delp))
        drules = len(re.findall('-<', delp)) - presumptions
        srules = len(re.findall('<-', delp)) - facts
        return [srules + drules, facts + presumptions]

    def count_lines(self, root: int, lines: list, level=0) -> int:
        """
        Count the number of lines of a dialectical tree with root <root>
        Args:
            -root: The id of the root argument
            -lines: List of all defeat relationships to build all arg lines
                    [[arg, defeater, id_arg, id_defeater],...]
        """
        children = [defeaters[3] for defeaters in lines if defeaters[2] == root]
        if len(children) == 0:
            # is a leaf
            self.aux_height.append(level)
            return 1  # Line, Height
        line = 0
        for child in children:
            line += self.count_lines(child, lines, level + 1)
        return line

    def analyze_results(self, result, defs, id_p):
        n_arguments = 0.0
        n_defeaters = 0.0
        avg_def_rules = 0.0
        n_arg_lines = 0.0
        avg_height_lines = 0.0
        avg_arg_lines = 0.0
        tree_numbers = 0
        args_defs = {}
        # Arguments, ADDL and Defeaters section
        d_graphs_data = result['dGraph']
        n_def_rules = 0  # To compute the average of defeasible rules
        for literal in d_graphs_data:
            literal_key = list(literal.keys())[0]
            arguments = literal[literal_key]
            for argument in arguments:
                argument_key = list(argument.keys())[0]
                if ',' in argument_key:
                    def_rules_in_body = len(argument[argument_key]["subarguments"])
                    if def_rules_in_body != 0:
                        delete_presumptions = sum(1 if re.match('\[\([a-zA-Z]*\-\<[t|T]rue\)\]', subs) else 0 for subs
                                                  in argument[argument_key]["subarguments"])
                        n_arguments += 1
                        n_def_rules += def_rules_in_body - delete_presumptions
                        defeaters = argument[argument_key]['defeats']
                        if defs:
                            args_defs[argument_key] = defeaters
                        n_defeaters += len(defeaters)

        if n_arguments != 0:
            avg_def_rules = n_def_rules / n_arguments

        # Trees section
        trees_data = result['status']
        for literal in trees_data:
            literal_key = list(literal.keys())[0]
            trees = literal[literal_key]['trees']
            roots = [root for root in trees if len(root) == 2]
            lines = [attacks for attacks in trees if len(attacks) == 4]
            if len(lines) != 0:
                for root in roots:
                    if '-<' in root[0]:
                        children = [defeaters[3] for defeaters in lines if defeaters[2] == root[1]]
                        if len(children) != 0:
                            n_arg_lines += self.count_lines(root[1], lines)
                            tree_numbers += 1
            else:
                pass
        sum_height_lines = sum(self.aux_height)
        self.aux_height = []
        if n_arg_lines != 0.0:
            avg_height_lines = sum_height_lines / n_arg_lines
        if tree_numbers != 0.0:
            avg_arg_lines = n_arg_lines / tree_numbers  # N° lines / N° Trees
        # To save the arguments and its defeaters
        if defs:
            write_result(self.path_file_results + id_p + 'OUTPUT.json', args_defs)
        return {
            'n_programs': id_p,
            'n_arguments': int(n_arguments),
            'n_defeaters': int(n_defeaters),
            'n_trees': tree_numbers,
            'avg_def_rules': float('{0:.2f}'.format(avg_def_rules)),
            'avg_arg_lines': float('{0:.2f}'.format(avg_arg_lines)),
            'avg_height_lines': float('{0:.2f}'.format(avg_height_lines))
        }

    def compute_metrics(self, defs, id_p, method, perc):
        initial_time = time.time()
        core_response = method(perc)
        end_time = time.time()
        query_time = end_time - initial_time
        self.times.append(query_time)
        if core_response != "Error":
            size_metrics = self.get_size_metrics()
            result = self.analyze_results(core_response, defs, id_p)
            self.rules.append(size_metrics[0])
            self.facts_presumptions.append(size_metrics[1])
            return result
        else:
            return {
                'n_programs':'0',
                'n_arguments': 0,
                'n_defeaters': 0,
                'n_trees': 0,
                'avg_def_rules': 0.0,
                'avg_arg_lines': 0,
                'avg_height_lines': 0.0
            }

    def compute_one(self, defs) -> None:
        self.aux_height = []
        n_programs = []
        arguments = []
        defeaters = []
        n_trees = []
        def_rules = []
        arg_lines = []
        height_lines = []
        data = self.compute_metrics(defs, '0', self.query_delp_solver, '0')
        n_programs.append(data['n_programs'])
        arguments.append(data['n_arguments'])
        defeaters.append(data['n_defeaters'])
        n_trees.append(data['n_trees'])
        def_rules.append(data['avg_def_rules'])
        arg_lines.append(data['avg_arg_lines'])
        height_lines.append(data['avg_height_lines'])

        self.compute_save_metrics(n_programs, arguments, def_rules, n_trees, arg_lines, height_lines, self.rules,
                                  self.facts_presumptions)

    def compute_save_metrics(self, n_programs:list, arguments: list, def_rules: list, n_trees: list, arg_lines: list, height_lines: list,
                             rules: list, facts_presumptions: list) -> None:
        min_time = min(self.times)
        max_time = max(self.times)
        results = {
            'args':
                {
                    'mean': my_round(np.mean(arguments)),
                    'std': my_round(np.std(arguments))
                },
            'addl':
                {
                    'mean': my_round(np.mean(def_rules)),
                    'std': my_round(np.std(def_rules))
                },
            't':
                {
                    'mean': my_round(np.mean(n_trees)),
                    'std': my_round(np.std(n_trees))
                },
            'b':
                {
                    'mean': my_round(np.mean(arg_lines)),
                    'std': my_round(np.std(arg_lines))
                },
            'h':
                {
                    'mean': my_round(np.mean(height_lines)),
                    'std': my_round(np.std(height_lines))
                },
            'times': {
                'min': float('{:0.2f}'.format(min_time)),
                'max': float('{:0.2f}'.format(max_time)),
                'mean': my_round(np.mean(self.times)),
                'std': my_round(np.std(self.times))
                    },
            'rules': {
                'mean': my_round(np.mean(rules)),
                'std': my_round(np.std(rules))
                    },
            'base': {
                'mean': my_round(np.mean(facts_presumptions)),
                'std': my_round(np.std(facts_presumptions))
                    }
        }
        write_result(self.build_path_result(), results)
        # To write the final csv with the results
        csv_path = self.path_file_results + 'variation_metrics.csv'
        with open(csv_path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['program', 'args', 'addl', 't', 'b', 'h', 'rules', 'base', 'times'])
            for program_result in range(len(arguments)):
                writer.writerow([
                    n_programs[program_result],
                    arguments[program_result],
                    def_rules[program_result],
                    n_trees[program_result],
                    arg_lines[program_result],
                    height_lines[program_result],
                    rules[program_result],
                    facts_presumptions[program_result],
                    my_round(self.times[program_result])])
            file.close()

    def compute_dataset(self, dataset_length, defs, approx_metrics, perc):
        """
        Computes the metrics of a set of DeLP programs.
        Args:
            dataset_length: Number of programs in the dataset
            # TODO: Complete this parameters
            defs:
            approx_metrics:
            perc:

        Returns:
            Computes and generates a file with the value of the metrics of a dataset
        """
        if approx_metrics:
            method_compute_metrics = self.query_delp_solver_approximation
        else:
            method_compute_metrics = self.query_delp_solver
        n_programs = []
        arguments = []
        defeaters = []
        n_trees = []
        def_rules = []
        arg_lines = []
        height_lines = []
        # TODO: Refactor the way to load the dataset (glob with order)
        for count in range(dataset_length):
            file_path = self.path_dataset + str(count) + 'delp' + '.delp'
            self.path_delp = file_path
            data = self.compute_metrics(defs, str(count), method_compute_metrics, perc)
            n_programs.append(data['n_programs'])
            arguments.append(data['n_arguments'])
            defeaters.append(data['n_defeaters'])
            n_trees.append(data['n_trees'])
            def_rules.append(data['avg_def_rules'])
            arg_lines.append(data['avg_arg_lines'])
            height_lines.append(data['avg_height_lines'])
        self.compute_save_metrics(n_programs, arguments, def_rules, n_trees, arg_lines, height_lines, self.rules,
                                  self.facts_presumptions)
