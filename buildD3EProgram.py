from utilsExp import *
import argparse
import numpy as np

def main(data, nvar, nvaruse, fileName):
    if(nvaruse <= nvar):
        af = []
        randomVar = [str(var) for var in list(range(nvar))] # Generate variables 
        rules = data["rules"]
        randomVarToUse = randomVar[:nvaruse] #Get the first nvaruse from randomVar
        randomVarToUse.append('True')
        trueProb = [0.5]
        otherValuesProb = [float(0.5) / float((len(randomVarToUse) - 1))] * (len(randomVarToUse) - 1)
        probs = otherValuesProb + trueProb
        print(randomVarToUse)
        print(probs)
        for rule in rules:
            form = getForm(randomVarToUse, probs)
            af.append([rule, form])

        program = {
            "randomVar":randomVar,
            "af":af
        }

        writeLebelledProgram(program, fileName)
    else:
        print_error_msj("Error")
        exit()

def getForm(variables, probs):
    form = np.random.choice(variables, 1, p = probs,replace=True)
    return str(form[0])


def checkNumberElems(nvar, nvaruse):
    if nvaruse > nvar:
        return True
    else:
        return False

parser = argparse.ArgumentParser(description='Script to generate formulas randomly for a del3e program')
parser.add_argument('-p',
                    action='store',
                    help="The delp3e program",
                    dest="data",
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
parser.add_argument('-op',
                    help='Operator to use (NOT IMPLEMENTED)',
                    dest="operators",
                    required=False)
parser.add_argument('-out',
                    help='Output file name',
                    dest="fileName",
                    required=True)                     
arguments = parser.parse_args()


main(arguments.data, arguments.nvar, arguments.nvaruse, arguments.fileName)