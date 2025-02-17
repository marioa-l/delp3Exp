import signal
from delp.consultDeLP import *
import argparse
from em.bnCode import *

uniquePrograms, uniquesWorlds = set(), set()
globalProgram, predicates, bayesianNetwork = '', '', ''
results = {
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
    'timeExecution':[],
    'worldsAnalyzed':0
}

def handlerTimer(signum, frame):
    raise Exception("Time over")

def startSampling(literal, timeout, pathResult):
    global uniquesWorlds, uniquePrograms

    signal.signal(signal.SIGALRM, handlerTimer)
    print_error_msj("\nTime setting: " + str(timeout) + " seconds")
    print_ok_ops('Starting random sampling...')
    worldsAnalyzed = 0
    #bar = IncrementalBar('Processing worlds', max=numberOfWorlds)
    spinner = Spinner('Processing worlds...')
    initialTime = time.time()
    signal.alarm(timeout)
    try:
        while(True):
            allWorlds = genSamples(bayesianNetwork, 10000, pathResult)
            for i in range(len(allWorlds)):
                worldData = allWorlds[i]
                worldAsTuple = tuple(worldData[0])
                if(not worldAsTuple in uniquesWorlds):
                    uniquesWorlds.add(worldAsTuple)
                    world = worldData[0]
                    evidence = worldData[1]
                    prWorld = getSamplingProb(evidence)
                    # Build the PreDeLP Program for a world
                    delpProgram = mapWorldToProgram(globalProgram, predicates, world)
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
                spinner.next()
                worldsAnalyzed += 1
            #bar.finish()
    except Exception as e:
        print('\n')
        print_error_msj(e)
        
    signal.alarm(0)
    results['timeExecution'].append(time.time() - initialTime)
    results['worldsAnalyzed'] = worldsAnalyzed
    results['yes']['perc'] = "{:.2f}".format((results['yes']['total'] * 100) / results['worldsAnalyzed'])
    results['no']['perc'] = "{:.2f}".format((results['no']['total'] * 100) / results['worldsAnalyzed'])
    results['und']['perc'] = "{:.2f}".format((results['und']['total'] * 100) / results['worldsAnalyzed'])
    results['unk']['perc'] = "{:.2f}".format((results['unk']['total'] * 100) / results['worldsAnalyzed'])
    results['l'] = results['yes']['prob']
    results['u'] = results['u'] - results['no']['prob']
    
    #Save file with results    
    with open(pathResult + 'timeRandomResults.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)
   
    # Print the results
    print_ok_ops("Results: ")
    print("Unique worlds: ", end='')
    print_ok_ops("%s" % (len(uniquesWorlds)))
    print("Unique programs: ", end='')
    print_ok_ops("%s" % (len(uniquePrograms)))
    print_ok_ops("Prob(%s) = [%.4f, %.4f]" % (literal, results['l'], results['u']))


def main(literal, timeout, models, bn, pathResult):
    global globalProgram, predicates, bayesianNetwork
    
    bayesianNetwork = bn
    globalProgram = models["af"]
    predicates = models["randomVar"]
    
    startSampling(literal, timeout, pathResult)
    

parser = argparse.ArgumentParser(description="Script to perform the Sampling experiment (Random)")

parser.add_argument('-l',
                    help = 'The literal to calculate the probability interval',
                    action = 'store',
                    dest = 'literal',
                    required = True)
parser.add_argument('-p',
                    help='The DeLP3E program path',
                    action='store',
                    dest='program',
                    type=getDataFromFile,
                    required=True)
parser.add_argument('-bn',
                    help='The Bayesian Network file path (only "bifxml" for now)',
                    action='store',
                    dest='bn',
                    type=loadBN,
                    required=True)
parser.add_argument('-pathR',
                    help='Path to save the results (with file name)',
                    dest='pathResult',
                    required=True)
parser.add_argument('-t',
                    help='Seconds to execute',
                    dest='timeout',
                    type=int,
                    required=True)
arguments = parser.parse_args()

main(arguments.literal, arguments.timeout, arguments.program, arguments.bn, arguments.pathResult)
