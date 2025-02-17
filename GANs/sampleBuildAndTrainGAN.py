import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from buildYesGAN import *
from buildNoGAN import *
from utils.toCNF import *
tf.get_logger().setLevel('ERROR')
tf.get_logger().warning('test')

## For test the GAN's model ##
##existYesModel, existNoModel = True, True
incorrectProgram = 0

existYesModel, existNoModel = False, False
uniquesWorlds, uniquePrograms = set(), set()
bayesianNetworl, allWorlds, globalProgram, predicates, inCNF = '', '', '', '', ''
bayesianNetworl, allWorlds, globalProgram, predicates = '', '', '', ''
results = {
    'literal': '',
    'yes':{
        'total':0,
        'perc':0.00,
        'prob':0.00
    },
    'no':{
        'total':0,
        'perc':0.00,
        'prob':0.00
    },
    'und':{
        'total':0,
        'perc':0.00,
        'prob':0.00
    },
    'unk':{
        'total':0,
        'perc':0.00,
        'prob':0.00
    },
    'l':0.00,
    'u':1.00,
    'timeExecution': [],
    'worldsAnalyzed': 0,
    'repeatedWorlds': 0,
    'program': '',
    'atomos': '',
    'atomUsed': ''
}

def searchTrainDataset(literal):
    global uniquePrograms, uniquesWorlds, results
    numberOfWorlds = allWorlds
    dim = len(predicates)
    numberOfAllWorlds = pow(2, dim)
    # If the training dataset exist
    #trainingDataSet = getTrainingDatasetForDB('training'+ literalStatus[0])
    datasetYes = []
    datasetNo = []
    print_ok_ops('Starting random sampling...')
    bar = IncrementalBar('Processing worlds', max=numberOfWorlds)
    initialTime = time.time()
    for i in range(numberOfWorlds):
        randomNum = np.random.choice(numberOfAllWorlds,1)
        worldData = int_to_bin_with_format(randomNum[0], dim)
        #worldData = allWorlds[i]
        worldAsTuple = tuple(worldData[0])
        if(not worldAsTuple in uniquesWorlds):
            uniquesWorlds.add(worldAsTuple)
            world = worldData[0]
            evidence = worldData[1]
            prWorld = bayesianNetwork.get_sampling_prob(evidence)
            # Build the PreDeLP Program for a world
            # delpProgram = [[rules], [binary]]
            delpProgram = mapWorldToProgram(globalProgram, predicates, world)
            delpProgram = mapWorldToProgram(globalProgram, predicates, world)
            status = queryToProgram(delpProgram, literal, uniquePrograms)
            if status[1] == 'yes':
                results['yes']['total'] += 1
                results['yes']['prob'] = results['yes']['prob'] + prWorld
                datasetYes.append(delpProgram[1])
                results['yes']['prob'] = results['yes']['prob'] + prWorld
                datasetYes.append(world)
            elif status[1] == 'no':
                results['no']['total'] += 1
                results['no']['prob'] = results['no']['prob'] + prWorld
                datasetNo.append(delpProgram[1])
                results['no']['prob'] = results['no']['prob'] + prWorld
                datasetNo.append(world)
            elif status[1] == 'undecided':
                results['und']['total'] += 1
                results['und']['prob'] = results['und']['prob'] + prWorld
            elif status[1] == 'unknown':
                results['unk']['total'] += 1
                results['unk']['prob'] = results['unk']['prob'] + prWorld
        bar.next()
    bar.finish()
    results['timeExecution'].append(time.time() - initialTime) # Time to sampling for find traininig dataset
    results['worldsAnalyzed'] = numberOfWorlds
    print_ok_ops('Length of training data set found (Yes): %s' % (len(datasetYes)))
    print_ok_ops('Length of training data set found (No): %s' % (len(datasetNo)))
    # Save the training dataset finded
    #saveTrainingDataset(dataset, literalStatus)
    return [datasetYes, datasetNo]

def samplingAndTraining(literal, pathResult):
    global existNoModel, existYesModel, results

    # Search a training dataset
    trainingDatasets = searchTrainDataset(literal)
    
    
    trainingDatasets = searchTrainDataset(literal)
    timeout = 0
    initialTime = time.time()
    if(len(trainingDatasets[0]) != 0):
        dataDim = len(trainingDatasets[0][0])
        # Yes training
        # trainingDatasetes[1] = 'no'
        configureTrainingYes(dataDim, trainingDatasets[0], pathResult, timeout)
        existYesModel = True
    else:
        print_error_msj("A 'yes' training dataset could not be found")
        existYesModel = False

    ################ For 'no' Training ################
    if(len(trainingDatasets[1]) != 0):
        dataDim = len(trainingDatasets[1][0])
        # No training
        # trainingDatasetes[1] = 'no'
        configureTrainingNo(dataDim, trainingDatasets[1], pathResult, timeout)
        existNoModel = True
    else:
        print_error_msj("A 'no' training dataset could not be found")
        existNoModel = False

    results['timeExecution'].append(time.time() - initialTime) # Time to training

def analyzeWorld(progInBin, literal):
    global results, uniquesWorlds, uniquePrograms, incorrectProgram, inconsistent
    global yesPrograms, noPrograms
    # Build the PreDeLP Program from the program in binary (if is a correct program)
    # delpProgram = [rules, progInBin, evidence]
    delpProgram = mapBinToProgram(globalProgram, progInBin)
    if(delpProgram != -1):
        aux = len(uniquePrograms)
def analyzeWorld(world, literal):
    global results, uniquesWorlds, uniquePrograms
    y = tuple(world)
    if y not in uniquesWorlds:
        uniquesWorlds.add(y)
        evidence = {i: world[i] for i in range(0, len(world))}  # Dict
        pr_world = bayesianNetwork.get_sampling_prob(evidence)
        # Build the PreDeLP Program for a world
        delpProgram = mapWorldToProgram(globalProgram, predicates, world)
        status = queryToProgram(delpProgram, literal, uniquePrograms)
        if aux != len(uniquePrograms):
            if status[1] == 'yes':
                probs = getSamplingProb(delpProgram[2])
                yesPrograms.append(probs)
                results['yes']['prob'] = results['yes']['prob'] + probs
                # worlds = get_worlds_by_program(delpProgram[2])
                # if(len(worlds) != 0):
                #     #toComplete = completeWorlds(len(predicates) - len(worlds[0]))
                #     #completedWorlds = [i[0] + i[1] for i in itertools.product(worlds, toComplete)]
                #     results['yes']['total'] += len(worlds)
                #     probs = getProbWorlds(worlds)
                #     results['yes']['prob'] = results['yes']['prob'] + probs
                # else:
                #     inconsistent += 1
            elif status[1] == 'no':
                probs = getSamplingProb(delpProgram[2])
                noPrograms.append(probs)
                results['no']['prob'] = results['no']['prob'] + probs
                # worlds = get_worlds_by_program(delpProgram[2])
                # if(len(worlds) != 0):
                #     #toComplete = completeWorlds(len(predicates) - len(worlds[0]))
                #     #completedWorlds = [i[0] + i[1] for i in itertools.product(worlds, toComplete)]
                #     results['no']['total'] += len(worlds)
                #     probs = getProbWorlds(worlds)
                #     results['no']['prob'] = results['no']['prob'] + probs
                # else:
                #     inconsistent += 1
            elif status[1] == 'undecided':
                # worlds = get_worlds_by_program(delpProgram[2])
                results['und']['total'] += 1
                # probs = getProbWorlds(worlds)
                # results['und']['prob'] = results['und']['prob'] + probs
            elif status[1] == 'unknown':
                # worlds = get_worlds_by_program(delpProgram[2])
                results['unk']['total'] += 1
                # probs = getProbWorlds(worlds)
                # results['unk']['prob'] = results['unk']['prob'] + probs
    else:
        #Incorrect Program
        incorrectProgram += 1
        if status[1] == 'yes':
            results['yes']['total'] += 1
            results['yes']['prob'] = results['yes']['prob'] + pr_world
        elif status[1] == 'no':
            results['no']['total'] += 1
            results['no']['prob'] = results['no']['prob'] + pr_world
        elif status[1] == 'undecided':
            results['und']['total'] += 1
            results['und']['prob'] = results['und']['prob'] + pr_world
        elif status[1] == 'unknown':
            results['unk']['total'] += 1
            results['unk']['prob'] = results['unk']['prob'] + pr_world


def samplingGan(samples, pathResult, literal):
    global results, uniquePrograms
    numberOfAllWorlds = allWorlds
    uniquePrograms = set()
    global results
    newYesWorldsGenerated = results["yes"]['total']
    newNoWorldsGenerated = results["no"]['total']

    print_error_msj("Starting GAN Sampling...")
    nSamples = int(samples)
    initialTime = time.time()
    # Check if models exists
    dataDim = len(globalProgram) 
    if existYesModel or existNoModel:
        noise = tf.random.normal([nSamples, dataDim]) # Controlar esto de normal o uniforme
        if existYesModel:
            new_modelYes = tf.keras.models.load_model(pathResult + 'my_model_yes/')
            modelsYes = new_modelYes(noise, training=False)
            modelsToBinYes = (modelsYes.numpy() > 0.5) * 1
            listModelYes = [model.tolist() for model in modelsToBinYes]
        else:
            randomNum = np.random.choice(numberOfAllWorlds, nSamples, replace=True)
            listModelYes = list(map(lambda ints: int_to_bin_with_format(ints, dataDim)[0], randomNum))
            #listModelYes = [world for [world, asDict] in genSamples(bayesianNetwork, nSamples, pathResult)] To work with BN
            listModelYes = [world for [world, asDict] in genSamples(bayesianNetwork, nSamples, pathResult)]
        
        noise = tf.random.normal([nSamples, dataDim]) # Controlar esto de normal o uniforme
        if existNoModel:
            new_modelNo = tf.keras.models.load_model(pathResult + 'my_model_no/')
            modelsNo = new_modelNo(noise, training=False)
            modelsToBinNo = (modelsNo.numpy() > 0.5) * 1
            listModelNo = [model.tolist() for model in modelsToBinNo]
        else:
            randomNum = np.random.choice(numberOfAllWorlds, nSamples, replace=True)
            listModelNo = list(map(lambda ints: int_to_bin_with_format(ints, dataDim)[0], randomNum))
            #listModelNo = [world for [world, asDict] in genSamples(bayesianNetwork, nSamples, pathResult)] To work with BN
            listModelNo = [world for [world, asDict] in genSamples(bayesianNetwork, nSamples, pathResult)]
        
        models = listModelYes + listModelNo
    else:
        randomNum = np.random.choice(numberOfAllWorlds, int(nSamples * 2), replace=True)
        models = list(map(lambda ints: int_to_bin_with_format(ints, dataDim)[0], randomNum))
        #models = [world['program'] for world in np.random.choice(allWorlds, int(nSamples * 2), replace=True)]
    
    bar = IncrementalBar('Processing generated programs...', max=len(models))

    for wAux in models:
        analyzeWorld(wAux, literal)
        bar.next()
    bar.finish()

    results['timeExecution'].append(time.time() - initialTime)  # Time for guided sampling
    results['worldsAnalyzed'] += nSamples*2
    results['repeatedWorlds'] = results['worldsAnalyzed'] - len(uniquesWorlds)
    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / results['worldsAnalyzed'])
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / results['worldsAnalyzed'])
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / results['worldsAnalyzed'])
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / results['worldsAnalyzed'])
    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']
    results['literal'] = literal
    results['program'] = globalProgram
    results['atomos'] = predicates

    # results['atomUsed'] = predUsed

    with open(pathResult + 'sampleGanResults.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

    newYesWorldsGenerated = results["yes"]['total'] - newYesWorldsGenerated
    newNoWorldsGenerated = results["no"]['total'] - newNoWorldsGenerated
    print_error_msj("New yes worlds generated by GAN: %s" % (newYesWorldsGenerated))
    print_error_msj("New no worlds generated by GAN: %s" % (newNoWorldsGenerated))

    # Save the unique worlds
    #save_unique_worlds(uniquesWorlds, 'GAN', pathResult)

    #print results
    print_ok_ops("Results: ")
    print("Unique worlds: ", end='')
    print_ok_ops("%s" % (len(uniquesWorlds)))
    print("Unique programs: ", end='')
    print_ok_ops("%s" % (len(uniquePrograms) - inconsistent))
    print("Incorrect Programs: ", end='')
    print_error_msj("%s" % (incorrectProgram))
    print("Inconsistent Programs: ", end='')
    print_error_msj("%s" % (inconsistent))
    print_ok_ops("Prob(%s) = [%.4f, %.4f]" % (literal, results['l'], results['u']))


def main(literal, models, bn, st, ss, pathResult):
    global allWorlds, globalProgram, predicates, numberOfWorlds, bayesianNetwork, inCNF
def main(literal, models, bn, st, ss, pathResult):
    global allWorlds, globalProgram, predicates, numberOfWorlds, bayesianNetwork

    bayesianNetwork = bn
    allWorlds = st
    #allWorlds = genSamples(bn, st, pathResult)
    globalProgram = models["af"]
    predicates = models["randomVar"]
    
    formulas = [form for [rules,form] in globalProgram]
    inCNF = generateClauses(formulas)
    bayesianNetwork = bn
    allWorlds = st
    #allWorlds = genSamples(bn, st, pathResult)
    globalProgram = models["af"]
    predicates = models["randomVar"]

    # Sampling and Training
    samplingAndTraining(literal, pathResult)
    # Guiaded Sampling
    samplingGan(int(ss/2), pathResult, literal)

parser = argparse.ArgumentParser(description="Script to perform the build and training of the GAN")

parser.add_argument('-l',
                    help = 'The literal to search the training dataset',
                    action = 'store',
                    dest = 'literal',
                    required = True)
# parser.add_argument('-p',
#                     help='The DeLP3E program path',
#                     action='store',
#                     dest='program',
#                     type=getDataFromFile,
#                     required=True)
# parser.add_argument('-bn',
#                     help='The Bayesian Network file path (only "bifxml" for now)',
#                     action='store',
#                     dest='bn',
#                     type=loadBN,
#                     required=True)
# parser.add_argument('-pathR',
#                     help='Path to save the results',
#                     dest='pathResult',
#                     required=True)
parser.add_argument('-st',
                    help='Number of samples to search a training dataset',
                    dest='samplesT',
                    type=int,
                    required=True)
parser.add_argument('-sg',
                    help='Number of guided samples',
                    dest='samplesS',
                    type=int,
                    required=True)
#arguments = parser.parse_args()

pathToProgram = "/home/mario/results/final/models.json"
pathResult = "/home/mario/results/final/"
program = getDataFromFile(pathToProgram)
myBN = BayesNetwork('TEST', '/home/mario/results/final/')
myBN.load_bn()
literal = 'l_5'
st = 300000
sg = 200000
main(literal, program, myBN, st, sg, pathResult)

