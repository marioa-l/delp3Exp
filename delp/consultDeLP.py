import sys
import time
import subprocess

def query_to_delp(delp_program, literals):
    # Add the preference criterion
    delp_string = delp_program + 'use_criterion(more_specific);'
    status_literals = {}
    for literal in literals:
        cmd = ['./globalCore', 'stream', delp_string, 'answ', literal]
        literal_time = time.time()
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        o, e = proc.communicate()
        end_literal_time = time.time() - literal_time
        if(proc.returncode == 0):
            status_literals[literal] = {'status': o.decode('ascii'), 'time': end_literal_time}
            #return o.decode('ascii')
        else:
            print("Error to consult literal")
            exit()
    return status_literals
