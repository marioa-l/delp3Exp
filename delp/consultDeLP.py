import os
import time
import tempfile
import subprocess


def query_to_delp(delp_program, literals):
    """
    Run the DeLP solver for each literal against the given program.

    The 'stream' interface broke in newer SWI-Prolog (term_string parsing
    changed), so we write the program to a temp .delp file and call the
    'file' interface, which the binary parses with full operator support.
    """
    # Convert ';' rule terminators to '.\n' so the file is a valid Prolog source
    program_text = delp_program.replace(';', '.\n') + 'use_criterion(more_specific).\n'

    status_literals = {}
    with tempfile.NamedTemporaryFile(mode='w', suffix='.delp', delete=False) as f:
        f.write(program_text)
        tmp_path = f.name

    try:
        for literal in literals:
            cmd = ['delp/globalCore', 'file', tmp_path, 'answ', f'[{literal}]']
            literal_time = time.time()
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            o, e = proc.communicate()
            end_literal_time = time.time() - literal_time
            if proc.returncode == 0:
                status_literals[literal] = {'status': o.decode('ascii'),
                                            'time': end_literal_time}
            else:
                print("Error to consult literal")
                print("Comando:", " ".join(cmd))
                print("Error:", e.decode('utf-8'))
                exit()
    finally:
        os.unlink(tmp_path)

    return status_literals
