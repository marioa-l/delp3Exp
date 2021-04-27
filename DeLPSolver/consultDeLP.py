import sys
sys.path.insert(4, '../Utils/utilsExp.py')
import subprocess
#from utilsExp import *

def queryToProgram(delpProgram, literal, uniquePrograms, delpSolverName):
    if(not len(delpProgram[0]) == 0):
        iProgram = getIndexInList(tuple(delpProgram[1]), uniquePrograms)
        if(iProgram == -1):
            # Programa único
            # Add the preference criterion
            delpProgramString = delpProgram[0] + 'use_criterion(more_specific);'
            #delpProgramString = delpProgram[0] + "use_criterion(lab_comparison)'"
            #cmd = ['./expListICIC', delpProgramString, literal]
            cmd = ['./DeLPSolver/' + delpSolverName, 'stream', delpProgramString, 'answ', literal]
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            o, e = proc.communicate()

            if(proc.returncode == 0):
                uniquePrograms.add((tuple(delpProgram[1]), o.decode('ascii'))) # Save the new program
                return (delpProgram[1], o.decode('ascii'))
            else:
                print_error_msj("Error to consult literal")
                exit()
        else:
            # Programa repetido
            return iProgram
    else:
        return ((''), ('unknown'))

def getIndexInList(program, uniquePrograms):
    pos = -1
    for i,t in enumerate(uniquePrograms):
        if program in t:
            pos = t
            break
    return pos
    #return -1
