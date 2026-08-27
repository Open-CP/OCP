"""Truth-table logic minimization into MILP/SAT inequalities via Espresso.

``ttb_to_ineq_logic`` converts an operator's truth table into a minimal set of linear
inequalities using the Espresso logic minimizer, with two backends selected by ``tool_type``:

- ``"minimize_logic"``: use PyEDA's espresso (the ``espresso`` executable PyEDA installs
  on PATH).
- ``"minimize_logic_espresso"``: use an external espresso binary (``~/espresso-logic/bin/espresso``).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from tools.paths import get_sbox_constraints_files_dir

try:
    import pyeda
except (ImportError, OSError):
    pyeda = None
    print("[WARNING] Failed to import PyEDA. Please check whether PyEDA is installed correctly. Install it by 'pip3 install pyeda', refer to https://pyeda.readthedocs.io/en/latest/")


def _find_external_espresso():
    """Locate the external espresso binary (independent of PyEDA's), or ``None``.

    Checks ``~/espresso-logic/bin`` (the documented install dir, not usually on PATH),
    then searches PATH -- excluding the Python environment's own bin, where PyEDA's
    ``espresso`` lives (that is the ``minimize_logic`` backend).
    """
    conventional = Path.home() / "espresso-logic" / "bin" / "espresso"
    if conventional.exists():
        return conventional

    env_bins = {str(Path(sys.prefix) / "bin"), str(Path(sys.exec_prefix) / "bin")}
    search_dirs = [d for d in os.environ.get("PATH", "").split(os.pathsep) if d and d not in env_bins]
    found = shutil.which("espresso", path=os.pathsep.join(search_dirs))
    return Path(found) if found else None


def espresso_pattern_to_ineq(pattern): # Convert the Espresso output into a list of integer coefficients representing a linear inequality of the form: sum_i (coeff_i * x_i) >= rhs
    """
    Parameters:
        pattern (str): A string consisting of characters '0', '1', or '-'. Each character corresponds to one variable:
                         '0' -> +1 coefficient (positive)
                         '1' -> -1 coefficient (negative)
                         '-' -> 0 coefficient (ignored)

    Returns:
        List[int]: Coefficients followed by the right-hand side (RHS) constant.

    Example:
        pattern = '01-1'  # Suppose variables = ['x1', 'x2', 'x3', 'x4']
        Then:
            x1 = '0' -> +1
            x2 = '1' -> -1
            x3 = '-' ->  0
            x4 = '1' -> -1
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


def ttb_to_ineq_logic(ttable, variables, mode=0, tool_type="minimize_logic", timeout=3600):
    """
    Convert a truth table in PLA (Programmable Logic Array) format to inequalities using logic minimization.

    Args:
        ttable:
            Truth table values.
            ttable[n] == '0' means output bit 1 in PLA; otherwise output bit 0.
        variables (list[str]):
            Variable names in the truth table order.
        mode (int):
            Espresso option preset (0, 1, or 2).
        tool_type (str):
            "minimize_logic": use PyEDA's espresso (the 'espresso' executable on PATH).
            "minimize_logic_espresso": use the external espresso, auto-discovered by
                ``_find_external_espresso`` in common install locations.
        timeout (int):
            Timeout in seconds for the espresso subprocess (default 3600 = 1 hour); a
            ``RuntimeError`` is raised if espresso does not finish within it.

    Returns:
        tuple: (inequalities, information).
    """
    # Define espresso command-line options based on mode. Refer to Espresso documentation for details on these options.
    espresso_options = [['-estrong', '-eonset'], [], ['-eonset']] # Espresso Script of Pyeda provides the parameters: "-e {fast,ness,nirr,nunwrap,onset,strong}"
    if not isinstance(mode, int) or not (0 <= mode < len(espresso_options)):
        raise ValueError(f"Invalid mode = {mode}. Expected an integer in [0, {len(espresso_options) - 1}].")

    # Prepare the truth table in PLA (Programmable Logic Array) format.
    num_vars = len(variables)
    pla_rows = [f'{bin(n)[2:].zfill(num_vars)} {"1" if ttable[n] == "0" else "0"}' for n in range(2 ** num_vars)]
    file_contents = (
        f".i {num_vars}\n"
        + ".o 1\n"
        + f".p {2 ** num_vars}\n"
        + ".ilb " + " ".join(variables) + "\n"
        + ".ob F\n"
        + ".type fr\n"
        + "\n".join(pla_rows) + "\n"
    )

    # Write the input PLA file.
    pla_file = str(get_sbox_constraints_files_dir() / 'ttable.txt')
    with open(pla_file, "w") as fw:
        fw.write(file_contents)

    if tool_type == "minimize_logic": # PyEDA's espresso (the 'espresso' executable on PATH).
        backend_name = "espresso_pyeda"
        espresso_path = shutil.which("espresso")
        if espresso_path is None:
            raise RuntimeError(
                "Cannot find PyEDA's espresso on PATH. Install PyEDA (pip install pyeda) "
                "and make sure its environment is active."
            )
        backend_version = getattr(pyeda, "__version__", "unknown")
        espresso_command = [espresso_path, *espresso_options[mode], pla_file]
    elif tool_type == "minimize_logic_espresso": # External Espresso software.
        backend_name = "espresso"
        espresso_path = _find_external_espresso()
        if espresso_path is None:
            raise RuntimeError(
                "Cannot find an external espresso binary. Install it first."
            )
        backend_version = "unknown"
        try:
            result = subprocess.run([str(espresso_path), "-v"], capture_output=True, text=True, check=False)
            version_text = (result.stdout + result.stderr).strip()
            if version_text:
                backend_version = version_text.splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            pass
        espresso_command = [str(espresso_path), *espresso_options[mode], pla_file]
    else:
        raise ValueError(f"unknown tool type {tool_type}")

    try:
        result = subprocess.run(espresso_command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"Espresso execution failed:\n{result.stderr}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Espresso execution exceeded the timeout of {timeout} seconds.")

    # Parse the espresso output into inequalities.
    espresso_output = result.stdout.splitlines()
    raw_patterns = [line.strip() for line in espresso_output if line.strip() and not line.startswith('.')]
    inequalities = [espresso_pattern_to_ineq(p[:len(variables)]) for p in raw_patterns]
    information = {"Backend": backend_name, "Backend version": backend_version, "Mode": espresso_options[mode]}
    return inequalities, information
