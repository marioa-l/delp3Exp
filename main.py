import sys
import csv
import glob
import re
from sampling import Programs, Worlds
from sampling.byProgramSampling import *
import argparse
from multiprocessing import Process


class Experiment:
    def __init__(self):
        pass

    def exact_sampling(self, models, output, approach):
        """
        :params
            -models: list
            -output: str
            -approach: str = ['worlds', 'programs']
        """
        if approach == 'worlds':
            # By worlds
            for model in models:
                exact = Worlds(model, output)
                exact.start_sampling(100, '-', '_e_w')
        else:
            # By programs
            for model in models:
                exact = Programs(model, output)
                exact.start_sampling(100, '_e_p')

    def by_world_sampling(self, models, output, samples, source):
        """
        :params
            -models: list
            -models_path: list
            -output: list
            -samples: int
            -source: str = ['dist', 'random']
        """
        if source == 'info':
            # Sampling from probability distribution
            for model in models:
                world_sampling = Worlds(model, output)
                world_sampling.start_sampling(samples, 'distribution', '_s_w')
        else:
            # Random sampling
            for model in models:
                world_sampling = Worlds(model, output)
                world_sampling.start_sampling(samples, 'random', '_s_w')

    def by_delp_sampling(self, models, output, samples):
        """
        :params
            -models: list
            -models_path: list
            -output: list
            -samples: int
            -amfilter: bool 
        """
        for model in models:
            program_sampling = Programs(model, output)
            program_sampling.start_sampling(samples, '_s_p')

    def analyze_results(self, files_path):
        """
        :params
            -files_path: str
        """
        results = glob.glob(files_path + '*model*.json')
        total_time = 0.0
        interest = 0
        for result in results:
            data = read_json_file(result)
            total_time += data["data"]["time"]
            for key, val in data["status"].items():
                if "flag" in val:
                    interest += 1
        print("Number of files: " + str(len(results)))
        print("Total time: " + str(total_time))
        print("Average: " + str(total_time / len(results)))
        print("Interest: " + str(interest))

    def write_exact_csv(self, results_path):
        """
        :params
            -results_path: str
        """
        times = []
        unique_programs = []
        n_worlds = 0
        results = glob.glob(results_path + '*model_e_*.json')
        #results = glob.glob(results_path + 'model*.json')
        fieldnames = ['Prog', 'Lit', 'Exact', 'Time']
        rows = []
        for result in results:
            n_program = re.search(r'\d+', gfn(result)).group()
            data = read_json_file(result)
            for lit, status in data["status"].items():
                if "flag" in status:
                    interval = '[' + to_decimal_format(status["l"], 4) + '-' + to_decimal_format(status["u"], 4) + ']'
                    rows.append(
                        {
                            'Prog': int(n_program),
                            'Lit': lit,
                            'Exact': interval,
                            'Time': format(status["time"], '.2f')
                        }
                    )
            times.append(data['data']['time'])
            unique_programs.append(data['data']['unique_progs'])
            n_worlds = data['data']['n_samples']
        ordered_rows = sorted(rows, key=lambda k: k['Prog'])
        with open(results_path + 'csvE_Results.csv', 'w', encoding='utf-8',
                  newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)
        times_mean, times_sd = np.mean(times), np.std(times)
        unique_programs_mean, unique_programs_sd = np.mean(unique_programs), np.std(unique_programs)
        with open(results_path + 'values_e.json', 'w') as output:
            json.dump({'time_mean': times_mean,
                        'time_sd': times_sd,
                        'unique_programs_mean': unique_programs_mean,
                        'unique_programs_sd': unique_programs_sd,
                        'tcm': times_mean / n_worlds,
                        'te': times_mean / 3600 }, output, indent=4)

    def write_sampling_csv(self, results_path: str) -> None:
        metrics = []
        time = []
        masses = []
        n_samples = []
        worlds_consulted = []
        results = glob.glob(results_path + '*model_s_*.json')
        fieldnames = ['Prog', 'Lit', 'Intervalo', 'Metric', 'Time', 'Mass']
        rows = []
        for result in results:
            program_name = gfn(result)
            n_program = re.search(r'\d+', program_name).group()
            data_sampling = read_json_file(result)
            exact = read_json_file(gfnexact_from_sampling(result))
            for lit, lit_e in exact["status"].items():
                if "flag" in lit_e:
                    lit_s = data_sampling["status"][lit]
                    intervalo = '[' + to_decimal_format(lit_s["l"], 4) + '-' + to_decimal_format(lit_s["u"], 4) + ']'
                    metric = compute_metric([lit_s["l"], lit_s["u"]],
                                            [lit_e["l"], lit_e["u"]])
                    mass = (lit_s["pyes"] + lit_s["pno"] +
                            lit_s["pundecided"] + lit_s["punknown"])
                    rows.append(
                        {
                            'Prog': int(n_program),
                            'Lit': lit,
                            'Intervalo': intervalo,
                            'Metric': metric,
                            'Time': format(lit_s["time"], '.2f'),
                            'Mass': format(mass, '.4f')
                        }
                    )
                    metrics.append(float(metric))
                    masses.append(mass)
            time.append(data_sampling['data']['time'])
            n_samples.append(data_sampling['data']['n_samples'])
            if '_s_p' in program_name:
                # is a program based sample
                worlds_consulted.append(data_sampling['data']['worlds_consulted'])
            else:
                # is a world based sample
                worlds_consulted.append(data_sampling['data']['n_samples'] - data_sampling['data']['repeated_worlds'])
        ordered_rows = sorted(rows, key=lambda k: k['Prog'])
        with open(results_path + 'csvS_Results.csv', 'w', encoding='utf-8',
                  newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ordered_rows)
        metric_mean, metric_sd = np.mean(metrics), np.std(metrics)
        time_mean, time_sd = np.mean(time), np.std(time)
        masses_mean, masses_sd = np.mean(masses), np.std(masses)
        n_samples_mean, n_samples_sd = np.mean(n_samples), np.std(n_samples)
        worlds_consulted_mean, worlds_consulted_sd = np.mean(worlds_consulted), np.std(worlds_consulted)
        
        with open(results_path + 'values_s.json', 'w') as output:
            json.dump({
                'metric_mean': metric_mean,
                'metric_sd': metric_sd,
                'time_mean': time_mean,
                'time_sd': time_sd,
                'mass_mean': masses_mean,
                'mass_sd': masses_sd,
                'n_samples_mean': n_samples_mean,
                'n_samples_sd': n_samples_sd,
                'worlds_consulted_mean': worlds_consulted_mean,
                'worlds_consulted_sd': worlds_consulted_sd
                }, output, indent=4)


def run_parallel(models, obj_exp, func, params):
    # Params is a tuple
    mid = int(len(models) / 2)
    total_models = len(models)
    models_1 = models[:mid]
    models_2 = models[mid:]
    init_time = time.time()
    p1 = Process(target=getattr(obj_exp, str(func)), args=(models_1,) + params)
    p2 = Process(target=getattr(obj_exp, str(func)), args=(models_2,) + params)
    print("Starting in parallel...")
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    end_time = time.time() - init_time
    print("Time to running in parallel: ", end_time)


# Params definition:
parser = argparse.ArgumentParser(description="Script for all experiment")

parser.add_argument('-path',
                    help="Path to read files",
                    action='store',
                    dest="path",
                    required=True)

parser.add_argument('-out',
                    help="Path to save the results",
                    action='store',
                    dest='out',
                    required=True)
## For sampling
parser.add_argument('-approach',
                    help="Approach to use",
                    choices=['worlds', 'programs'],
                    action='store',
                    dest='approach',
                    required=True)

parser.add_argument('-exact',
                    help="(bool) To compute the exact values",
                    action='store_true',
                    dest="exact")

parser.add_argument('-sampling',
                    help='To compute sampling approximation',
                    choices=['info', 'random'],
                    action='store',
                    dest="sampling")

parser.add_argument('-size',
                    help='Percentage of samples to generate',
                    type=int,
                    action='store',
                    dest='size')

parser.add_argument('-parallel',
                    help="(bool) To run in parallel",
                    action="store_true",
                    dest="parallel")

parser.add_argument('-i',
                    help="Index of model where start",
                    action='store',
                    dest='i',
                    required=True)

parser.add_argument('-j',
                    help="Index of model where stop",
                    action='store',
                    dest='j',
                    required=True)
## For analyze results
parser.add_argument('-analyze',
                    help="(bool) To analyze the results",
                    action='store_true',
                    dest='analyze')

parser.add_argument('-tocsv',
                    help="To generate the results in csv format",
                    choices=['exact', 'sampling'],
                    action='store',
                    dest="tocsv")
## For test one particular models
parser.add_argument('-test',
                    help="Path of one model",
                    action='store',
                    dest='one_path')

args = parser.parse_args()

exp = Experiment()
# Get all models
if args.one_path:
    models = [args.one_path]
else:
    models = sorted(glob.glob(args.path + '*model.json'), key=natural_key)[int(args.i):int(args.j)]
# To generate the csv files:
if args.tocsv:
    if args.tocsv == 'exact':
        exp.write_exact_csv(args.path)
    else:
        exp.write_sampling_csv(args.path)
# To analyze the results:
elif args.analyze:
    exp.analyze_results(args.path)
# To run exact:
elif args.exact:
    if args.parallel:
        run_parallel(models, exp, 'exact_sampling', (args.out,
                                                     args.approach))
    else:
        exp.exact_sampling(models, args.out, args.approach)
# To run sampling
elif args.sampling:
    if args.parallel:
        if args.approach == 'worlds':
            # By worlds in parallel
            run_parallel(models, exp, 'by_world_sampling', (args.path, args.out,
                                                            args.size,
                                                            args.sampling))
        else:
            # By delp in parallel
            run_parallel(models, exp, 'by_delp_sampling', (args.path, args.out,
                                                           args.size,
                                                           args.sampling))
    else:
        if args.approach == 'worlds':
            # By worlds in sequential
            exp.by_world_sampling(models, args.out, args.size,
                                  args.sampling)
        else:
            # By delp in sequential
            exp.by_delp_sampling(models, args.out, args.size)
