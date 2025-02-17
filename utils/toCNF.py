from pysat.formula import CNF
from pysat.solvers import Glucose3
from sympy.logic.boolalg import to_cnf
import string
import re
from sympy import *
import itertools


def mainTest(formulas):
    clauseForSolver = []
    for i in formulas:
        clauseForSolver.append(get_clause_for_solver(i[0],i[1]))
    toSolve = list(itertools.chain.from_iterable(clauseForSolver))
    print(toSolve)
    with Glucose3(bootstrap_with=toSolve) as g:
        for m in g.enum_models():
            print(m)

def toCNF(formula, state, symbols):
    if state == False:
        if "~" in formula:
            #formula = "~ ( " + formula + " )"
            formula = formula.replace("~","")
        else:
            formula = '~' + formula 
    #formula = formula.split(" ")
    #formula_with_symbols = " ".join([symbols[int(char) - 1] if char.isdigit() else char for char in formula])
    #formInCNF = str(to_cnf(formula))
    #formula_with_atoms = "".join([str(symbols.index(char) + 1) if char.isalpha() else char for char in formInCNF])
    return formula.replace("~","-")

def generateClauses(formulas):
    # 'formulas' is a list with formulas 
    # of the form 'not ( not 4 | 6 ) & 10', where the numbers represent atoms from the EM
    cluasesToSolver = [] #[[Variables for 'True' state],[Variables for 'False' state]...]
    symbols = list(string.ascii_lowercase) #The alphabet
    for formula in formulas:
        if formula != "True":
            formula = formula.replace("and", "&").replace("or", "|").replace("not", "~")
            cluasesToSolver.append([toCNF(formula, True, symbols), toCNF(formula, False, symbols)])
        else:
            cluasesToSolver.append("$")
    return(cluasesToSolver)

#generateClauses(["not ( not '4' or '6' ) and '10'", "'4' or '5'", "not '6'", "( '4' and '7' ) or '3'"])
#print(generateClauses(["not 4", "5", "not 6", "902"]))
