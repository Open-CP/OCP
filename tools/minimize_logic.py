import subprocess
from pathlib import Path

try:
    from pyeda.inter import espresso_tts
except ImportError:
    print("[WARNING] PyEDA is not installed, installing it by 'pip3 install pyeda', https://pyeda.readthedocs.io/en/latest/")

ROOT = Path(__file__).resolve().parents[1] # this file -> tools -> <ROOT>
FILES_DIR = ROOT / "files" / "sbox_modeling"
FILES_DIR.mkdir(parents=True, exist_ok=True)

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


def ttb_to_ineq_logic(ttable, variables, mode=0, backend="espresso_pyeda", timeout=72000): # Convert a truth table to CNF or MILP constraints using the Espresso logic minimization tool via PyEDA.
    # Prepare truth table in PLA (Programmable Logic Array) format
    """
    Convert a truth table in PLA (Programmable Logic Array) format to inequalities using logic minimization.

    Args:
        ttable:
            Truth table values.
            ttable[n] == '0' means output bit 1 in PLA; otherwise output bit 0.
        variables (list[str]):
            Variable names in the truth table order.
        mode (int):
            Option preset for the external espresso binary.
        backend (str):
            "espresso_pyeda": Use the PyEDA library to call espresso internally.
            "espresso": Use the external espresso software.
        timeout (int):
            Timeout in seconds for espresso.

    Returns:
        list:
            A list of inequalities.
    """
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
    pla_file = str(FILES_DIR / 'ttable.txt')
    result_file = str(FILES_DIR / 'sttable.txt')

    # Write input PLA file
    with open(pla_file, "w") as fw:
        fw.write(file_contents)

    # Define espresso command-line options based on mode. Refer to Espresso documentation for details on these options.
    espresso_options =  [['-estrong', '-eonset'], [], ['-eonset']] # Espresso Script of Pyeda provides the parameters: "-e {fast,ness,nirr,nunwrap,onset,strong}"

    if backend == "espresso_pyeda":
        # Use PyEDA to call espresso internally
        espresso_command = ['espresso', *espresso_options[mode], pla_file]
    elif backend == "espresso":
        # Use external espresso software
        espresso_path = Path.home() / "espresso-logic" / "bin" / "espresso" # Adjust this path to where espresso is installed on your system
        if espresso_path is None:
            raise FileNotFoundError("Cannot find 'espresso' in PATH. Please install Espresso or switch backend='espresso_pyeda'.")
        espresso_command = [espresso_path, *espresso_options[mode], pla_file]
    try:
        result = subprocess.run(espresso_command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Espresso execution failed:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Espresso execution exceeded the timeout of {timeout} seconds.")

    # Save output and parse
    with open(result_file, 'w') as fw:
        fw.write(result.stdout)
    espresso_output = result.stdout.splitlines()
    raw_patterns = [line.strip() for line in espresso_output if line.strip() and not line.startswith('.')]

    # Convert logic lines to target constraints
    inequalities = [espresso_pattern_to_ineq(p[:len(variables)]) for p in raw_patterns]
    return inequalities
