import os
import time
import subprocess

# Directory containing this file — used to resolve the binary and its libraries
_DELP_DIR = os.path.dirname(os.path.abspath(__file__))
_BINARY   = os.path.join(_DELP_DIR, "globalCore")

# Prepend delp/ to LD_LIBRARY_PATH so libswipl.so.10 is found at runtime
_env = os.environ.copy()
_env["LD_LIBRARY_PATH"] = _DELP_DIR + os.pathsep + _env.get("LD_LIBRARY_PATH", "")


def query_to_delp(delp_program, literals):
    delp_string = delp_program + 'use_criterion(more_specific);'
    status_literals = {}
    for literal in literals:
        cmd = [_BINARY, 'stream', delp_string, 'answ', literal]
        literal_time = time.time()
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                env=_env)
        o, e = proc.communicate()
        end_literal_time = time.time() - literal_time
        if proc.returncode == 0:
            status_literals[literal] = {'status': o.decode('ascii'), 'time': end_literal_time}
        else:
            print("Error to consult literal")
            print("Comando:", " ".join(cmd))
            print("Error:", e.decode('utf-8'))
            exit()
    return status_literals
