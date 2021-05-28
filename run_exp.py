import sys
import csv
import re
from utilsExp import *
from exactSampling import *
from sampleRandomSampling import *
import argparse
from multiprocessing import Process

class Experiment:
    def __init__(self):
        pass

    
    def exact_sampling(self, models: list, models_path: str, output: str):
        for model in models:
            index = int(re.search(r'\d+', os.path.basename(model)).group())
            exact = Exact(model, models_path, 'BNdelp' + str(index), output)
            exact.start_exact_sampling()
    
    
    def dist_sampling(self, models: list, models_path: str, output: str, samples: int):
        for model in models:
            index = int(re.search(r'\d+', os.path.basename(model)).group())
            exact_values = read_json_file(os.path.dirname(model)+'/par/'+ os.path.basename(model)[:-5] + 'output.json')
            world_sampling = WorldSampling(model, models_path, 'BNdelp' + str(index), output, exact_values["status"].keys())
            
            world_sampling.start_distribution_sampling(samples)

    def analyze_results(self, files_path: str):
        results = glob.glob(files_path + 'modeldelp*output.json')
        total_time = 0.0
        interest = 0
        for result in results:    
            data = read_json_file(result)
            total_time += data["data"]["time"]
            for key, val in data["status"].items():
                if "flag" in val:
                    interest += 1
        print_ok("Total time: " + str(total_time))
        print_ok("Average: " + str(total_time / len(results)))
        print_ok("Interest: " + str(interest))


    def write_csv(self, results_path: str) -> None:
        results = glob.glob(results_path + 'modeldelp*output.json')
        fieldnames = ['Prog|Lit', 'Exact', 'Time']
        rows =[]
        for result in results:
            n_program = int(re.search(r'\d+', os.path.basename(result)).group())
            data = read_json_file(result)
            for lit, status in data["status"].items():
                if "flag" in status:
                    rows.append(
                        {
                            'Prog|Lit': str(n_program) + '|' + lit,
                            'Exact': '[' + format(status["l"],'.4f')  +'-'+ format(status["u"],'.4f')  +']',
                            'Time': format(status["time"],'.2f')
                        }
                    )
        with open(results_path + 'csvResults.csv', 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def exact_parallel(models: list):
    experiment.exact_sampling(models, arguments.path, arguments.out_path) 


def sampling_parallel(models: list):
    experiment.dist_sampling(models, arguments.path, arguments.out_path, int(arguments.sampling))

parser = argparse.ArgumentParser(description = " Script for all experiment")
parser.add_argument('-exact',
                    help="(bool) To compute the exact values",
                    action='store_true',
                    dest="exact")
parser.add_argument('-sampling',
                    help='To compute world sampling approximation',
                    action='store',
                    dest="sampling")
parser.add_argument('-path',
                    help="Path to read models",
                    action='store',
                    dest="path",
                    required=True)
parser.add_argument('-analyze',
                    help="(bool) To analyze time and number of interesting literals",
                    action='store_true',
                    dest='analyze')
parser.add_argument('-out',
                    help="Path to save the results",
                    action='store',
                    dest='out_path',
                    required=True)
parser.add_argument('-parallel',
                    help="(bool) To run in parallel",
                    action="store_true",
                    dest="parallel")
parser.add_argument('-tocsv',
                    help="(bool) To generate the results in csv format",
                    action='store_true',
                    dest="tocsv")

arguments = parser.parse_args()

experiment = Experiment()
models = glob.glob(arguments.path + 'modeldelp*.json')
mid = int(len(models)/2)
total_models = len(models)
part1 = models[:mid]
part2 = models[mid:]

if arguments.tocsv:
    experiment.write_csv(arguments.path)
elif arguments.exact:
    if arguments.parallel:
        init_time = time.time()
        p1 = Process(target=exact_parallel, args=(part1,))
        p2 = Process(target=exact_parallel, args=(part2,))
        print_info("Starting in parallel...")
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        end_time = time.time() - init_time
        print("Time to running in parallel: ", end_time)
    else:
        init_time = time.time()
        print_info("Starting in sequencial...")
        experiment.exact_sampling(models, arguments.path, arguments.out_path)
        end_time = time.time() - init_time
        print("Time to run in sequencial: ", end_time)
elif arguments.sampling:
    if arguments.parallel:
        init_time = time.time()
        p1 = Process(target=sampling_parallel, args=(part1,))
        p2 = Process(target=sampling_parallel, args=(part2,))
        print_info("Starting in parallel...")
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        end_time = time.time() - init_time
        print("Time to running in parallel: ", end_time)
    else:
        init_time = time.time()
        print_info("Starting in sequencial...")
        experiment.dist_sampling(models, arguments.path, arguments.out_path, int(arguments.sampling))
        end_time = time.time() - init_time
        print("Time to run in sequencial: ", end_time)
    
elif arguments.analyze:
    experiment.analyze_results(arguments.path)
