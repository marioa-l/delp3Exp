import numpy as np
from progress.bar import IncrementalBar
from progress.spinner import Spinner
from delp.consultDeLP import queryToProgram
from em.bn import *
from utilsExp import *
import time

bayesianNetwork, rules, used_atoms, em_atoms, af = '', '', '', '', ''
uniquePrograms, uniqueOnlyPrograms, uniqueEvidences, uniquesWorlds = set(), set(), set(), set()
inconsistent_program, repeated_program, repeated_evidence, dimension = 0, 0, 0, 0

results = {
    'yes': {
        'total': 0,
        'perc': 0.00,
        'prob': 0.00
    },
    'no': {
        'total': 0,
        'perc': 0.00,
        'prob': 0.00
    },
    'und': {
        'total': 0,
        'perc': 0.00,
        'prob': 0.00
    },
    'unk': {
        'total': 0,
        'perc': 0.00,
        'prob': 0.00
    },
    'inconsistent': {
        'total': 0,
        'perc': 0.00
    },
    'repeated': {
        'total': 0,
        'perc': 0.00
    },
    'l': 0.00,
    'u': 1.00,
}

def checkEqual(lst):
    return lst[1:] == lst[:-1]


def build_delp_from_binaries(binaries):
    delp = ''
    for index, item in enumerate(binaries):
        if item == 1:
            delp += rules[index]
    return delp


def get_evidence(program):
    evidence = {}
    for index, rule in enumerate(program):
        atom = used_atoms[index]  # Atom of the program rule 'rule'
        if rule == 1:
            # The atom must be True
            if 'not' in atom:
                atomInself = str(int(atom.split(' ')[1]) - 1)
                if atomInself in evidence:
                    if evidence[atomInself] == 0:
                        pass
                    else:
                        evidence = 'incorrect_program'

                        break
                else:
                    evidence[atomInself] = 0
            elif atom == 'True':
                continue
            else:
                atom = str(int(atom) - 1)
                if atom in evidence:
                    if evidence[atom] == 1:
                        pass
                    else:
                        evidence = 'incorrect_program'

                        break
                else:
                    evidence[atom] = 1
        else:
            # The atom must be false
            if 'not' in atom:
                atomInself = str(int(atom.split(' ')[1]) - 1)
                if atomInself in evidence:
                    if evidence[atomInself] == 1:
                        pass
                    else:
                        evidence = 'incorrect_program'

                        break
                else:
                    evidence[atomInself] = 1
            elif atom == 'True':
                # Incorrect program
                evidence = 'incorrect_program'
                break
            else:
                atom = str(int(atom) - 1)
                if atom in evidence:
                    if evidence[atom] == 0:
                        pass
                    else:
                        evidence = 'incorrect_program'

                        break
                else:
                    evidence[atom] = 0

    return evidence


def get_interest_programs(programs, literal, bn):
    global inconsistent_program, repeated_program, uniquePrograms, uniqueOnlyPrograms, uniqueEvidences, repeated_evidence
    yes_programs = []
    no_programs = []
    for program in programs:
        tuple_program = tuple(program)
        if tuple_program not in uniqueOnlyPrograms:
            uniqueOnlyPrograms.add(tuple_program)
            delp = build_delp_from_binaries(program)
            evidence = get_evidence(program)
            if evidence != 'incorrect_program':
                tuple_evidence = tuple(evidence.items())
                if tuple_evidence not in uniqueEvidences:
                    uniqueEvidences.add(tuple_evidence)
                    status = queryToProgram([delp, program], literal, uniquePrograms)
                    prWorld = bn.get_sampling_prob(evidence)
                    if status[1] == 'yes':
                        yes_programs.append(program)
                        results['yes']['total'] += 1
                        results['yes']['prob'] = results['yes']['prob'] + prWorld
                    elif status[1] == 'no':
                        no_programs.append(program)
                        results['no']['total'] += 1
                        results['no']['prob'] = results['no']['prob'] + prWorld
                    elif status[1] == 'undecided':
                        results['und']['total'] += 1
                        results['und']['prob'] = results['und']['prob'] + prWorld
                    elif status[1] == 'unknown':
                        results['unk']['total'] += 1
                        results['unk']['prob'] = results['unk']['prob'] + prWorld
                else:
                    repeated_evidence += 1
            else:
                inconsistent_program += 1
        else:
            repeated_program += 1
    return [yes_programs, no_programs]


def get_interest_worlds(worlds, literal, bn):
    global uniquesWorlds, uniquePrograms
    yes_worlds = []
    no_worlds = []
    for world in worlds:
        worldAsTuple = tuple(world)
        if(not worldAsTuple in uniquesWorlds):
            uniquesWorlds.add(worldAsTuple)
            evidence = {i: world[i] for i in range(0, len(world))}  # Dict
            prWorld = bn.get_sampling_prob(evidence)
            # Build the PreDeLP Program for a world
            delpProgram = map_world_to_program(af, em_atoms, world)
            status = queryToProgram(delpProgram, literal, uniquePrograms)
            if status[1] == 'yes':
                yes_worlds.append(world)
                results['yes']['total'] += 1
                results['yes']['prob'] = results['yes']['prob'] + prWorld
            elif status[1] == 'no':
                no_worlds.append(world)
                results['no']['total'] += 1
                results['no']['prob'] = results['no']['prob'] + prWorld
            elif status[1] == 'undecided':
                results['und']['total'] += 1
                results['und']['prob'] = results['und']['prob'] + prWorld
            elif status[1] == 'unknown':
                results['unk']['total'] += 1
                results['unk']['prob'] = results['unk']['prob'] + prWorld
    return [yes_worlds, no_worlds]


def update_prob_vector(yes_programs, no_programs, prob_yes, prob_no):
    # Update the yes probability vector
    if len(yes_programs) != 0:
        new_yes_prob_vector = [sum(i) for i in zip(*yes_programs)]
        new_yes_prob_vector = [elem / len(yes_programs) for elem in new_yes_prob_vector]
    else:
        new_yes_prob_vector = prob_yes
    # Update the no probability vector
    if len(no_programs) != 0:
        new_no_prob_vector = [sum(i) for i in zip(*no_programs)]
        new_no_prob_vector = [elem / len(no_programs) for elem in new_no_prob_vector]
    else:
        new_no_prob_vector = prob_no

    #print_ok_ops("********Updating vectors********\n")
    #print("YES: ", new_yes_prob_vector)
    #print("NO: ", new_no_prob_vector)
    #print_ok_ops("********************************")
    return [new_yes_prob_vector, new_no_prob_vector]


def generate_population(prob_vector_yes, prob_vector_no, quantity):
    population = []
    for i in range(int(quantity/2)):
        individual_yes = []
        individual_no = []
        for j in range(dimension):
            var_value_yes = np.random.choice(2, p=[1 - prob_vector_yes[j], prob_vector_yes[j]])
            var_value_no = np.random.choice(2, p=[1 - prob_vector_no[j], prob_vector_no[j]])
            individual_yes.append(var_value_yes)
            individual_no.append(var_value_no)
        population.append(individual_yes)
        population.append(individual_no)
    return population


def generate_initial_population(population):
    current_population = []
    for i in range(population * 2):
        individual = []
        for j in range(dimension):
            if used_atoms[j] != 'True':
                var_value = np.random.choice(2)
            else:
                var_value = 1
            individual.append(var_value)
        current_population.append(individual)
    return current_population


def umda(literal, domain, iterations, population, bn):
    prob_vector_yes = [0.5] * dimension
    prob_vector_no = [0.5] * dimension

    if domain == 'am':
        current_population = generate_initial_population(population)
        bar = IncrementalBar('Building probability vectors...', max=iterations)
        for i in range(iterations):  # Iterations
            # Selected Population
            interest_programs = get_interest_programs(current_population, literal, bn)
            # Update the probability vector
            new_prob_vectors = update_prob_vector(interest_programs[0],
                                                  interest_programs[1],
                                                  prob_vector_yes,
                                                  prob_vector_no)
            prob_vector_yes = new_prob_vectors[0]
            prob_vector_no = new_prob_vectors[1]
            # Generate new population with probability vectors
            current_population = generate_population(prob_vector_yes, prob_vector_no, population)

            bar.next()
        bar.finish()
    else:
        #current_population = generate_population(prob_vector_yes, prob_vector_no, population)
        current_population = bn.gen_samples(population)
        current_population = [world[0] for world in current_population]
        #bar = IncrementalBar('Building probability vectors...', max=iterations)
        spinner = Spinner("Processing")
        for i in range(iterations):  # Iterations
            # Selected Population
            interest_worlds = get_interest_worlds(current_population, literal, bn)
            # Update the probability vector
            new_prob_vectors = update_prob_vector(interest_worlds[0],
                                                  interest_worlds[1],
                                                  prob_vector_yes,
                                                  prob_vector_no)
            prob_vector_yes = new_prob_vectors[0]
            prob_vector_no = new_prob_vectors[1]
            # Generate new population with probability vectors
            current_population = generate_population(prob_vector_yes, prob_vector_no, population)

            spinner.next()


    # print("Prob vector for Yes: ", prob_vector_yes)
    # print("Prob vector for No: ", prob_vector_no)
    #print("Incorrect programs: ", inconsistent_program)
    #print("Repeated programs: ", repeated_program)

    return [prob_vector_yes, prob_vector_no]


def sampling_umda_programs(literal, yes_prob_vector, no_prob_vector, quantity, bn):
    global uniquePrograms, inconsistent_program, repeated_program, uniqueOnlyPrograms
    print("Generating programs...")
    generated_programs = generate_population(yes_prob_vector, no_prob_vector, quantity)
    print("OK\n")

    bar = IncrementalBar('Analyzing generate programs...', max=quantity * 2)
    for program in generated_programs:
        if not tuple(program) in uniqueOnlyPrograms:
            uniqueOnlyPrograms.add(tuple(program))
            delp = build_delp_from_binaries(program)
            status = queryToProgram([delp, program], literal, uniquePrograms)
            evidence = get_evidence(program)
            if evidence != 'incorrect_program':
                tuple_evidence = tuple(evidence.items())
                if tuple_evidence not in uniqueEvidences:
                    uniqueEvidences.add(tuple_evidence)
                    prWorld = bn.get_sampling_prob(evidence)
                    if status[1] == 'yes':
                        results['yes']['total'] += 1
                        results['yes']['prob'] = results['yes']['prob'] + prWorld
                    elif status[1] == 'no':
                        results['no']['total'] += 1
                        results['no']['prob'] = results['no']['prob'] + prWorld
                    elif status[1] == 'undecided':
                        results['und']['total'] += 1
                        results['und']['prob'] = results['und']['prob'] + prWorld
                    elif status[1] == 'unknown':
                        results['unk']['total'] += 1
                        results['unk']['prob'] = results['unk']['prob'] + prWorld
                else:
                    repeated_evidence += 1
            else:
                inconsistent_program += 1
        else:
            repeated_program += 1

        bar.next()
    bar.finish()


def saampling_umda_worlds(literal, yes_prob_vector, no_prob_vector, quantity, bn):
    global uniquePrograms, uniquesWorlds
    print("Generating words...")
    generated_worlds = generate_population(yes_prob_vector, no_prob_vector, quantity)
    print("OK\n")

    bar = IncrementalBar("Analyzing generated worlds...", max=len(generated_worlds))
    for world in generated_worlds:
        worldAsTuple = tuple(world)
        if(not worldAsTuple in uniquesWorlds):
            uniquesWorlds.add(worldAsTuple)
            evidence = {i: world[i] for i in range(0, len(world))}  # Dict
            prWorld = bn.get_sampling_prob(evidence)
            # Build the PreDeLP Program for a world
            delpProgram = map_world_to_program(af, em_atoms, world)
            status = queryToProgram(delpProgram, literal, uniquePrograms)
            if status[1] == 'yes':
                results['yes']['total'] += 1
                results['yes']['prob'] = results['yes']['prob'] + prWorld
            elif status[1] == 'no':
                results['no']['total'] += 1
                results['no']['prob'] = results['no']['prob'] + prWorld
            elif status[1] == 'undecided':
                results['und']['total'] += 1
                results['und']['prob'] = results['und']['prob'] + prWorld
            elif status[1] == 'unknown':
                results['unk']['total'] += 1
                results['unk']['prob'] = results['unk']['prob'] + prWorld
        bar.next()
    bar.finish()


def umda_sampling(literal, quantity_to_sample, iterations, population, domain, bn, pathToResult):
    # 'domain' is for choice between UMDA with programs or worlds
    # domain = am --> to programs
    # domain = em --> to worlds
    results['total_sampling'] = iterations * population
    # results['generate_samples'] = quantity_to_sample
    # results['total_sampling'] = results['training_samples'] + results['generate_samples']
    initial_time = time.time()
    probability_vectors = umda(literal, domain, iterations, population, bn)

    # if domain == 'am':
    #     sampling_umda_programs(literal, probability_vectors[0], probability_vectors[1], quantity_to_sample, bn)
    #     print("Unique programs: ", end='')
    #     print_ok_ops("%s" % (len(uniqueOnlyPrograms)))
    #     results['unique_programs'] = len(uniqueOnlyPrograms)
    #     results['inconsistent']['total'] = inconsistent_program
    #     results['inconsistent']['perc'] = "{:.2f}".format((results['inconsistent']['total'] * 100) / results['total_sampling'])
    #     results['repeated']['total'] = repeated_program
    #     results['repeated']['perc'] = "{:.2f}".format((results['repeated']['total'] * 100) / results['total_sampling'])
    #     results['domain'] = 'programs'
    # else:
    #     saampling_umda_worlds(literal, probability_vectors[0], probability_vectors[1], quantity_to_sample, bn)
    # print("Unique worlds: ", end='')
    # print_ok_ops("%s" % (len(uniquesWorlds)))
    results['unique_worlds'] = len(uniquesWorlds)
    results['repeated']['total'] = results['total_sampling'] - results['unique_worlds']
    results['repeated']['perc'] = "{:.2f}".format((results['repeated']['total'] * 100) / results['total_sampling'])
    results['domain'] = 'worlds'

    #print(results)
    execution_time = time.time() - initial_time

    results['execution_time'] = execution_time

    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / results['total_sampling'])
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / results['total_sampling'])
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / results['total_sampling'])
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / results['total_sampling'])

    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']


    #print("Repeated programs: ", end='')
    #print_ok_ops("%s" % repeated_program)
    #print("Analyzed programs: ", (iterations * (2 * population) + quantity * 2))
    #print("Unique evidence: ", end='')
    #print_ok_ops("%s" % len(uniqueEvidences))

    with open(pathToResult + 'UMDA.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


def umda_brute_force_programs(literal, bn):
    global uniqueEvidence, uniquePrograms, repeated_evidence, inconsistent_program

    all_possible_programs = pow(2, dimension)
    bar = IncrementalBar("Analyzing programs...", max=all_possible_programs)
    initial_time = time.time()
    for int_value in range(all_possible_programs):
        program = int_to_bin_with_format(int_value, dimension)[0]  # Return [program, evidence] REVISAR
        evidence = get_evidence(program)
        if evidence != 'incorrect_program':
            tuple_evidence = tuple(evidence.items())
            if tuple_evidence not in uniqueEvidences:
                uniqueEvidences.add(tuple_evidence)
                delp = build_delp_from_binaries(program)
                status = queryToProgram([delp, program], literal, uniquePrograms)
                prWorld = bn.get_sampling_prob(evidence)
                if status[1] == 'yes':
                    results['yes']['total'] += 1
                    results['yes']['prob'] = results['yes']['prob'] + prWorld
                elif status[1] == 'no':
                    results['no']['total'] += 1
                    results['no']['prob'] = results['no']['prob'] + prWorld
                elif status[1] == 'undecided':
                    results['und']['total'] += 1
                    results['und']['prob'] = results['und']['prob'] + prWorld
                elif status[1] == 'unknown':
                    results['unk']['total'] += 1
                    results['unk']['prob'] = results['unk']['prob'] + prWorld
            else:
                repeated_evidence += 1
        else:
            inconsistent_program += 1
        bar.next()
    bar.finish()

    time_execution = time.time() - initial_time
    results['execution_time'] = time_execution
    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / all_possible_programs)
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / all_possible_programs)
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / all_possible_programs)
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / all_possible_programs)
    results['inconsistent']['total'] = inconsistent_program
    results['inconsistent']['perc'] = "{:.2f}".format((results['inconsistent']['total'] * 100) / all_possible_programs)
    results['programsAnalyzed'] = all_possible_programs
    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']
    print("Unique programs: ", end='')
    print_ok_ops("%s" % (int_value + 1))
    print("Unique evidence: ", end='')
    print_ok_ops("%s" % len(uniqueEvidences))
    print("Inconsistent programs: ", end='')
    print_ok_ops("%s" % inconsistent_program)

    with open('/home/mario/results/umda/UMDAForceBrutePrograms.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


def umda_brute_force_worlds(literal, bn):
    global uniquePrograms
    all_possible_worlds = pow(2, dimension)
    bar = IncrementalBar('Analyzing worlds...', max=all_possible_worlds)
    initial_time = time.time()
    for int_value in range(all_possible_worlds):
        worldData = int_to_bin_with_format(int_value, dimension)  # Return [program, evidence] REVISAR
        world = worldData[0]
        evidence = worldData[1]
        prWorld = bn.get_sampling_prob(evidence)

        # Build the delp program for a world
        delpProgram = map_world_to_program(af, em_atoms, world)
        # Compute the literal status
        status = queryToProgram(delpProgram, literal, uniquePrograms)
        if status[1] == 'yes':
            results['yes']['total'] += 1
            results['yes']['prob'] = results['yes']['prob'] + prWorld
        elif status[1] == 'no':
            results['no']['total'] += 1
            results['no']['prob'] = results['no']['prob'] + prWorld
        elif status[1] == 'undecided':
            results['und']['total'] += 1
            results['und']['prob'] = results['und']['prob'] + prWorld
        elif status[1] == 'unknown':
            results['unk']['total'] += 1
            results['unk']['prob'] = results['unk']['prob'] + prWorld
        bar.next()
    bar.finish()

    results['worldsAnalyzed'] = all_possible_worlds
    time_execution = time.time() - initial_time
    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / all_possible_worlds)
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / all_possible_worlds)
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / all_possible_worlds)
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / all_possible_worlds)
    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']
    results['time_execution'] = time_execution

    # Save file with results
    with open('/home/mario/results/umda/UMDAForceBruteWorlds.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


def random_sampling_program(literal, bn, samples):
    global inconsistent_program, repeated_program, uniquePrograms, uniqueOnlyPrograms, uniqueEvidences, repeated_evidence
    all_possible_programs = pow(2, dimension)
    bar = IncrementalBar("Random sampling programs...", max=samples)
    initial_time = time.time()
    for i in range(samples):
        int_program = np.random.choice(all_possible_programs + 1,1)
        program = int_to_bin_with_format(int_program, dimension)[0]
        tuple_program = tuple(program)
        if tuple_program not in uniqueOnlyPrograms:
            uniqueOnlyPrograms.add(tuple_program)
            delp = build_delp_from_binaries(program)
            evidence = get_evidence(program)
            if evidence != 'incorrect_program':
                tuple_evidence = tuple(evidence.items())
                if tuple_evidence not in uniqueEvidences:
                    uniqueEvidences.add(tuple_evidence)
                    status = queryToProgram([delp, program], literal, uniquePrograms)
                    prWorld = bn.get_sampling_prob(evidence)
                    if status[1] == 'yes':
                        results['yes']['total'] += 1
                        results['yes']['prob'] = results['yes']['prob'] + prWorld
                    elif status[1] == 'no':
                        results['no']['total'] += 1
                        results['no']['prob'] = results['no']['prob'] + prWorld
                    elif status[1] == 'undecided':
                        results['und']['total'] += 1
                        results['und']['prob'] = results['und']['prob'] + prWorld
                    elif status[1] == 'unknown':
                        results['unk']['total'] += 1
                        results['unk']['prob'] = results['unk']['prob'] + prWorld
                else:
                    repeated_evidence += 1
            else:
                inconsistent_program += 1
        else:
            repeated_program += 1
        bar.next()
    bar.finish()
    time_execution = time.time() - initial_time
    results['execution_time'] = time_execution
    results['unique_programs'] = len(uniqueOnlyPrograms)
    results['inconsistent']['total'] = inconsistent_program
    results['inconsistent']['perc'] = "{:.2f}".format((results['inconsistent']['total'] * 100) / samples)
    results['repeated']['total'] = repeated_program
    results['repeated']['perc'] = "{:.2f}".format((results['repeated']['total'] * 100) / samples)
    results['domain'] = 'programs'
    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / samples)
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / samples)
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / samples)
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / samples)
    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']
    results['total_sampling'] = samples
    print("Unique programs: ", end='')
    print_ok_ops("%s" % len(uniqueOnlyPrograms))
    print("Unique evidence: ", end='')
    print_ok_ops("%s" % len(uniqueEvidences))
    print("Inconsistent programs: ", end='')
    print_ok_ops("%s" % inconsistent_program)

    print("repeated evidence: ", repeated_evidence)

    with open('/home/mario/results/umda/UMDARandomPrograms.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)


def main(experiment, literal, samples, program, myBN, pathToResult):
    global rules, used_atoms, dimension, em_atoms, af

    # pathToProgram = "/home/mario/results/umda/9models.json"
    # myBN = BayesNetwork('9TEST', '/home/mario/results/umda/')
    # myBN.load_bn()
    # program = getDataFromFile(pathToProgram)
    em_atoms = program["randomVar"]
    af = program["af"]
    rules = [item[0] for item in program["af"]]
    used_atoms = [item[1] for item in program["af"]]
    #literal = 'l_3'  # Literal to consult
    # dimension = len(rules) # Rules or atoms [Program or Worlds]

    # samples = 10000
    # samples_random = 25000
    population = 50
    iterations = int(samples / population)
    if experiment == 'bfp':
        dimension = len(rules)
        umda_brute_force_programs(literal, myBN)
    elif experiment == 'bfw':
        dimension = len(em_atoms)
        umda_brute_force_worlds(literal, myBN)
    elif experiment == 'pvp':
        dimension = len(rules)
        umda_sampling(literal, samples, iterations, population, 'am', myBN)
    elif experiment == 'pvw':
        dimension = len(em_atoms)
        umda_sampling(literal, samples, iterations, population, 'em', myBN, pathToResult)
    elif experiment == 'rsp':
        dimension = len(rules)
        random_sampling_program(literal, myBN, samples_random)

    # umda_brute_force_programs(literal, myBN)
    # umda_brute_force_worlds(literal, myBN)
#main('pvp')

literal = sys.argv[0]
samples = sys.argv[1]
pathToProgram = sys.argv[2]
program = getDataFromFile(pathToProgram)
myBN = BayesNetwork(sys.argv[3] + 'TEST', sys.argv[4])
myBN.load_bn()
pathToResult = sys.argv[5]

main('pvw', literal, samples, program, myBN, pathToResult)
