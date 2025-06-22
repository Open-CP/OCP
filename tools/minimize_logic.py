try:
    from pyeda.inter import espresso_tts
except ImportError:
    print("PyEDA is not installed, installing it by 'pip3 install pyeda', https://pyeda.readthedocs.io/en/latest/")        
import subprocess
import os
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files/sbox_modeling/'))
if not os.path.exists(base_path): 
    os.makedirs(base_path, exist_ok=True)

def espresso_pattern_to_ineq(pattern): # Convert the Espresso output into a list of integer coefficients representing a linear inequality of the form: sum_i (coeff_i * x_i) >= rhs
    """    
    Parameters:
        pattern (str): A string consisting of characters '0', '1', or '-'. Each character corresponds to one variable:
                         '0' → +1 coefficient (positive)
                         '1' → -1 coefficient (negative)
                         '-' → 0 coefficient (ignored)

    Returns:
        List[int]: Coefficients followed by the right-hand side (RHS) constant.

    Example:
        pattern = '01-1'  # Suppose variables = ['x1', 'x2', 'x3', 'x4']
        Then:
            x1 → '0' → +1
            x2 → '1' → -1
            x3 → '-' →  0
            x4 → '1' → -1
            Coefficients = [1, -1, 0, -1], RHS = -2 (from two '1's) + 1 = -1 
            Inequality becomes: x1 - x2 - x4 >= -1
        Return: [1, -1, 0, -1, -1]
    """
    coeffs = []
    rhs = 0
    for ch in pattern:
        if ch == '0':
            coeffs.append(1)
        elif ch == '1':
            coeffs.append(-1)
            rhs -= 1
        else:
            coeffs.append(0)  # don't care
    return coeffs + [rhs + 1]


def ttb_to_ineq_logic(ttable, variables, mode=0): # Convert a truth table to CNF or MILP constraints using the Espresso logic minimization tool via PyEDA.
    # Prepare truth table in PLA (Programmable Logic Array) format
    cont_ttable = ''
    num_vars = len(variables)
    for n in range(2**num_vars):
        bit = '1' if ttable[n] == '0' else '0'
        cont_ttable += f'{bin(n)[2:].zfill(num_vars)} {bit}\n'
    file_contents = f".i {num_vars}\n"
    file_contents += ".o 1\n"
    file_contents += f".p {2**(num_vars)}\n"
    file_contents += ".ilb " + " ".join(variables) + "\n"     
    file_contents += ".ob F\n"
    file_contents += ".type fr\n" 
    file_contents += cont_ttable

    # Setup paths
    pla_file = os.path.join(base_path, 'ttable.txt')
    result_file = os.path.join(base_path, 'sttable.txt')

    # Write input PLA file
    with open(pla_file, "w") as fw:
        fw.write(file_contents)

    # Define espresso command-line options based on mode
    espresso_options =  [['-estrong', '-eonset'], [], ['-eonset']] # Espresso Script of Pyeda provides the parameters: "-e {fast,ness,nirr,nunwrap,onset,strong}"
    # espresso_options =  [[], ['-efast'], ['-estrong'], ['-eness'], ['-enirr'], ['-enunwrap'], ['-eonset'], ['-efast', '-eonset'], ['-estrong', '-eonset'], ['-estrong', '-eonset', '-eness'], ['-estrong', '-eonset', '-enirr'], ['-estrong', '-eonset', '-enunwrap']]
    espresso_command = ['espresso', *espresso_options[mode], pla_file] 
    
    # Run espresso via subprocess
    try:
        result = subprocess.run(espresso_command, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Espresso execution failed:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        print("Espresso execution exceeded the 3600s time limit.")
        return []
    
    
    # Save output and parse
    with open(result_file, 'w') as fw:
        fw.write(result.stdout)
    espresso_output = result.stdout.splitlines()
    raw_patterns = [line.strip() for line in espresso_output if line.strip() and not line.startswith('.')]
    
    # Convert logic lines to target constraints
    inequalities = [espresso_pattern_to_ineq(p[:len(variables)]) for p in raw_patterns]
    return inequalities
