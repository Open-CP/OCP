"""
This module provides tools for solving MILP/SAT models. Supports multiple solvers and configurations.
    - MILP solvers: Gurobi, SCIP
    - SAT solvers: PySAT, OR-Tools CPSAT, external DIMACS binaries (e.g., PRS, CaDiCaL, CryptoMiniSat, Kissat)
"""
from importlib import import_module
from importlib.util import find_spec
from shutil import which
import os
import re
import subprocess
import tempfile
import time

from tools.resource_monitor import RuntimeResourceMonitor


DEFAULT_MILP_SOLVER = "GUROBI"
DEFAULT_SAT_BACKEND = "PySAT"
MILP_SOLVERS = ("GUROBI", "SCIP")
PYSAT_SOLVERS = (
    "Cadical103",
    "Cadical153",
    "Cadical195",
    "CryptoMinisat",
    "Gluecard3",
    "Gluecard4",
    "Glucose3",
    "Glucose4",
    "Lingeling",
    "MapleChrono",
    "MapleCM",
    "Maplesat",
    "Mergesat3",
    "Minicard",
    "Minisat22",
    "MinisatGH",
)
PYSAT_SOLVER_NAME_MAP = {name.lower(): name for name in PYSAT_SOLVERS}

# External SAT solvers run as standalone binaries on a DIMACS .cnf file.
EXTERNAL_SAT_SOLVERS = {
    "PRS": lambda cfg: [
        cfg.get("solver_bin", "prs"),
        "--mode=0",
        f"--threads={cfg.get('threads', 16)}",
        "--preprocess_threads=1",
        "--breakid_threads=1",
        "--kissat_sup_threads=1",
        "--kissat_mab_prop_threads=1",
        "--sbva_threads=1",
    ],
    "CaDiCaL": lambda cfg: [cfg.get("solver_bin", "cadical")],
    "CryptoMiniSat": lambda cfg: [cfg.get("solver_bin", "cryptominisat5")],
    "Kissat": lambda cfg: [cfg.get("solver_bin", "kissat")],
}
EXTERNAL_SAT_SOLVER_NAME_MAP = {name.lower(): name for name in EXTERNAL_SAT_SOLVERS}

SAT_SOLVERS = ("DEFAULT", *PYSAT_SOLVERS, "ORTools", *EXTERNAL_SAT_SOLVERS)

def _printable_config(config_solver):
    """Return config_solver without the bulky resource_usage entry, for the settings echo line."""
    return {k: v for k, v in config_solver.items() if k != "resource_usage"}


SOLVER_STATUSES = ("SAT", "UNSAT", "UNKNOWN")  # how a solve ended; only "UNSAT" proves no solution exists


def _set_solver_status(config_solver, status, reason=None):
    """Record how the solve ended, so a caller cannot read ``[]`` as UNSAT.

    "UNKNOWN" means a limit stopped the solver: any result is partial and no bound was proven.
    """
    if status not in SOLVER_STATUSES:
        raise ValueError(f"[ERROR] Unknown solver status: '{status}'. Expected one of {list(SOLVER_STATUSES)}.")
    config_solver["solver_status"] = status
    if reason is not None:
        config_solver["solver_status_reason"] = reason


def _modules_available(*module_names):
    """Return True if every named module is importable, without importing any of them."""
    return all(find_spec(module_name) is not None for module_name in module_names)


def _external_binary_available(solver):
    """Return True if the registered external solver's binary is on $PATH (registry default name)."""
    return which(EXTERNAL_SAT_SOLVERS[solver]({})[0]) is not None


def _backend_status(available, *, implemented=True, solvers=None):
    """Build a solver-backend capability entry (availability, implemented flag, solver names)."""
    status = {
        "available": bool(available),
        "implemented": bool(implemented),
    }
    if solvers is not None:
        status["solvers"] = list(solvers)
    return status


def solver_capabilities():
    """Return optional solver backend availability without importing backends."""
    return {
        "milp": {
            "GUROBI": _backend_status(
                _modules_available("gurobipy"),
                solvers=("DEFAULT", "GUROBI"),
            ),
            "SCIP": _backend_status(
                _modules_available("pyscipopt"),
                solvers=("SCIP",),
            ),
        },
        "sat": {
            "PySAT": _backend_status(
                _modules_available("pysat"),
                solvers=("DEFAULT", *PYSAT_SOLVERS),
            ),
            "ORTools": _backend_status(
                _modules_available("ortools"),
                solvers=("ORTools",),
            ),
            "External": _backend_status(
                any(_external_binary_available(s) for s in EXTERNAL_SAT_SOLVERS),
                solvers=tuple(EXTERNAL_SAT_SOLVERS),
            ),
        },
        "default": {
            "milp": DEFAULT_MILP_SOLVER,
            "sat": DEFAULT_SAT_BACKEND,
        },
    }


def normalize_milp_solver_name(solver="DEFAULT"):
    """Normalize a MILP solver name to a supported value ('GUROBI'/'SCIP'), raising on unknown."""
    solver_name = str(solver).upper()
    if solver_name == "DEFAULT":
        return DEFAULT_MILP_SOLVER
    if solver_name not in MILP_SOLVERS:
        raise ValueError(f"[ERROR] Unsupported solver: '{solver}'. Supported: 'GUROBI' (DEFAULT), 'SCIP'.")
    return solver_name


def normalize_sat_solver_name(solver="DEFAULT", backend=None):
    """Normalize a SAT solver name to a supported value, raising on unknown.

    ``backend`` selects the solving engine explicitly and takes precedence over
    name-based matching. This disambiguates solver names shared by more than one
    backend (e.g. PySAT's "CryptoMinisat" vs the external "CryptoMiniSat" binary,
    identical once lowercased): set ``backend="pysat"`` or ``backend="external"``
    to pick. With ``backend=None`` the name is matched against the PySAT names
    first (preserving historical behaviour), then the external registry.
    """
    solver_text = str(solver)
    if backend is not None and backend not in ("pysat", "external"):
        raise ValueError(f"[ERROR] Unsupported backend: '{backend}'. Supported: 'pysat', 'external'.")
    if backend == "external":
        if solver_text.lower() in EXTERNAL_SAT_SOLVER_NAME_MAP:
            return EXTERNAL_SAT_SOLVER_NAME_MAP[solver_text.lower()]
        raise ValueError(
            f"[ERROR] Unsupported external solver: '{solver}'. Supported: "
            + ", ".join(EXTERNAL_SAT_SOLVERS)
            + "."
        )
    if solver_text.upper() == "DEFAULT":
        return "DEFAULT"
    if solver_text.lower() == "ortools":
        return "ORTools"
    if solver_text.lower() in PYSAT_SOLVER_NAME_MAP:
        if backend is None and solver_text.lower() in EXTERNAL_SAT_SOLVER_NAME_MAP:
            print(f"[INFO] '{solver}' matches both PySAT and an external solver; using PySAT. "
                  f"Set config_solver['backend'] = 'external' to use the external binary.")
        return PYSAT_SOLVER_NAME_MAP[solver_text.lower()]
    if backend is None and solver_text.lower() in EXTERNAL_SAT_SOLVER_NAME_MAP:
        return EXTERNAL_SAT_SOLVER_NAME_MAP[solver_text.lower()]
    raise ValueError(
        f"[ERROR] Unsupported solver: '{solver}'. Supported: ORTools, DEFAULT, "
        + ", ".join(PYSAT_SOLVERS + tuple(EXTERNAL_SAT_SOLVERS))
        + "."
    )


def is_solver_available(kind, solver="DEFAULT", backend=None):
    """Return whether a configured solver route is both installed and implemented.

    ``backend`` ('pysat'/'external') disambiguates SAT solver names shared by both
    backends, as in :func:`normalize_sat_solver_name`.
    """
    kind = kind.lower()
    capabilities = solver_capabilities()

    if kind == "milp":
        try:
            solver_name = normalize_milp_solver_name(solver)
        except ValueError:
            return False
        backend_entry = capabilities["milp"].get(solver_name)
    elif kind == "sat":
        try:
            solver_name = normalize_sat_solver_name(solver, backend=backend)
        except ValueError:
            return False
        if solver_name == "DEFAULT" or solver_name in PYSAT_SOLVERS:
            backend_entry = capabilities["sat"]["PySAT"]
        elif solver_name == "ORTools":
            backend_entry = capabilities["sat"]["ORTools"]
        elif solver_name in EXTERNAL_SAT_SOLVERS:
            return _external_binary_available(solver_name)
        else:
            backend_entry = None
    else:
        raise ValueError(f"Unsupported solver kind: {kind!r}. Supported: 'milp', 'sat'.")

    return bool(backend_entry and backend_entry["available"] and backend_entry["implemented"])


def _load_gurobi():
    """Import and return the ``gurobipy`` module, or ``None`` if it is unavailable."""
    try:
        return import_module("gurobipy")
    except ImportError:
        return None


def _load_scip_model():
    """Import and return PySCIPOpt's ``Model`` class, or ``None`` if it is unavailable."""
    try:
        return import_module("pyscipopt").Model
    except ImportError:
        return None


def _scip_error_types():
    """Return the exception types SCIP may raise (generic fallbacks if PySCIPOpt is unavailable)."""
    try:
        scip_module = import_module("pyscipopt")
    except ImportError:
        return (OSError, RuntimeError, ValueError)
    scip_error = getattr(scip_module, "SCIPError", None)
    return tuple(t for t in (scip_error, OSError, RuntimeError, ValueError) if t is not None)


def _load_pysat():
    """Import and return PySAT's ``(CNF, Solver)``, or ``(None, None)`` if it is unavailable."""
    try:
        return import_module("pysat.formula").CNF, import_module("pysat.solvers").Solver
    except ImportError:
        return None, None


def _load_ortools_cpsat():
    """Import and return OR-Tools' ``cp_model`` module, or ``None`` if it is unavailable."""
    try:
        return import_module("ortools.sat.python.cp_model")
    except (ImportError, OSError):
        return None


# ------------------------------ Solver Versions ------------------------------
def _module_version(module_name):
    """Return a module's ``__version__`` string ('not installed' / 'unknown' when unavailable)."""
    try:
        module = import_module(module_name)
    except ImportError:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def _milp_solver_version(config_solver):
    """Resolve the MILP solver (Gurobi / SCIP) version string for ``config_solver`` (no printing)."""
    solver = normalize_milp_solver_name(config_solver.get("solver", "DEFAULT"))
    if solver == "GUROBI":
        gurobipy = _load_gurobi()
        if gurobipy is None:
            return "gurobipy not installed"
        try:
            return "Gurobi " + ".".join(str(part) for part in gurobipy.gurobi.version())
        except AttributeError:
            return "gurobipy " + _module_version("gurobipy")
    scip_model = _load_scip_model()
    if scip_model is None:
        return "pyscipopt not installed"
    try:
        return "SCIP " + str(scip_model().version())
    except _scip_error_types():
        return "pyscipopt " + _module_version("pyscipopt")


def _pysat_solver_version(solver):
    """Best-effort version of a PySAT-bundled solver (python-sat release; pycryptosat for CryptoMinisat)."""
    if "crypto" in solver.lower():
        crypto_version = _module_version("pycryptosat")
        if crypto_version not in ("not installed", "unknown"):
            return "pycryptosat " + crypto_version
    return "bundled with python-sat " + _module_version("pysat")


def _external_solver_version(solver, config_solver):
    """Query an external solver binary for its version string.

    Runs the binary with common version flags and returns the first non-empty
    output line. Falls back to a descriptive message if the binary is missing or
    does not report a version.
    """
    # Reuse the registry builder to resolve the binary path (honours "solver_bin").
    binary = EXTERNAL_SAT_SOLVERS[solver](config_solver)[0]
    for flag in ("--version", "-version", "--help"):
        try:
            proc = subprocess.run([binary, flag], capture_output=True, text=True, timeout=10)
        except FileNotFoundError:
            return f"binary '{binary}' not found on PATH"
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode != 0:  # e.g. "invalid option '--version'" also contains "version"
            continue
        text = proc.stdout or proc.stderr
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            if "version" in ln.lower():
                m = re.search(r'version[:\s]+([0-9][\w.\-]*)', ln, re.IGNORECASE)
                return m.group(1) if m else (ln[2:] if ln.startswith("c ") else ln)
        if lines:
            return lines[0]
    return "unknown"


def _sat_solver_version(config_solver):
    """Resolve the SAT solver (PySAT-bundled, OR-Tools CPSAT, or external binary) version string (no printing)."""
    solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"),
                                       backend=config_solver.get("backend"))
    if solver == "ORTools":
        return "OR-Tools " + _module_version("ortools")
    if solver in EXTERNAL_SAT_SOLVERS:
        return _external_solver_version(solver, config_solver)
    return _pysat_solver_version(solver)  # "DEFAULT" and every PySAT solver


def get_solver_version(kind, config_solver=None):
    """Report (print and return) the concrete solver name and version for a 'milp'/'sat' route."""
    config_solver = config_solver or {}
    kind = kind.lower()
    if kind == "milp":
        solver = normalize_milp_solver_name(config_solver.get("solver", "DEFAULT"))
        version = _milp_solver_version(config_solver)
    elif kind == "sat":
        solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"),
                                           backend=config_solver.get("backend"))
        version = _sat_solver_version(config_solver)
    else:
        raise ValueError(f"Unsupported solver kind: {kind!r}. Supported: 'milp', 'sat'.")
    print(f"[INFO] {kind} solver: {solver} | version: {version}")
    return {"kind": kind, "solver": solver, "solver_version": version}


def solve_milp(filename, config_solver=None):
    """
    Solve a MILP model.

    Args:
        filename (str): Path to the MILP model file.
        config_solver (dict):
            - solver: solver name (e.g. "GUROBI", "SCIP").
            - solution_number: The number of solutions to find (default: 1).

    Returns:
        A list of solutions. Each solution is represented as a dictionary mapping variable names to their values.
        An empty list means no solution only under ``config_solver["solver_status"] == "UNSAT"``.
    """

    if config_solver is None:
        config_solver = {}
    if not isinstance(config_solver, dict):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")
    for key in ("solver_status", "solver_status_reason"):
        config_solver.pop(key, None)  # the setdefault below must not meet the previous solve's status
    solver = normalize_milp_solver_name(config_solver.get("solver", "DEFAULT"))
    config_solver["solver"] = solver
    if "solver_version" not in config_solver:  # resolve once (guard: setdefault would recompute every call)
        config_solver["solver_version"] = _milp_solver_version(config_solver)
    print(f"[INFO] Solving MILP model with settings: {_printable_config(config_solver)}")
    monitor = RuntimeResourceMonitor(interval=0.2)
    monitor.start()
    time_start = time.time()
    try:
        if solver == "GUROBI":
            solutions = solve_milp_gurobi(filename, config_solver)
        elif solver == "SCIP":
            solutions = solve_milp_scip(filename, config_solver)
        # Normalize a missing-backend None to [] so callers always get a list.
        solutions = solutions if solutions is not None else []
        config_solver.setdefault("solver_status", "SAT" if solutions else "UNSAT")
        return solutions
    finally:
        config_solver["resource_usage"] = monitor.stop()
        elapsed = round(time.time() - time_start, 2)
        config_solver.setdefault("solving_time", []).append(elapsed)  # per-call trace; total = sum(solving_time)
        print(f"[INFO] solve #{len(config_solver['solving_time'])}: {elapsed} s | cumulative solver time: {round(sum(config_solver['solving_time']), 2)} s")


def solve_milp_gurobi(filename, config_solver):
    """Solve a MILP model using Gurobi; raise if gurobipy is unavailable."""
    gp = _load_gurobi()
    if gp is None:
        raise RuntimeError("[ERROR] gurobipy can't be loaded. Install it: pip install gurobipy")

    try:
        model = gp.read(filename) # Load the model from file.
        # Set Parameters provided by Gurobi. Example: TimeLimit, SolutionLimit, PoolSearchMode, PoolSolutions, MIPFocus, etc.
        for key, val in config_solver.items():
            if hasattr(model.Params, key):
                setattr(model.Params, key, val)
        solution_number = config_solver.get("solution_number", 1)
        if isinstance(solution_number, int) and solution_number > 1:
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = solution_number
        # Solve the model
        model.optimize()
        sol_count = getattr(model, "SolCount", 0)
        status = getattr(model, "Status", None)
    except gp.GurobiError as e:
        raise RuntimeError(f"[ERROR] Gurobi failed: {e}. Check your license, "
                           f"see https://gurobi.com/unrestricted") from e

    # How it ended, before the solutions: no solution is UNSAT only if Gurobi says infeasible,
    # and a solution found at a limit is not a proven optimum.
    if status == gp.GRB.INFEASIBLE:
        _set_solver_status(config_solver, "UNSAT")
    elif status != gp.GRB.OPTIMAL:
        _set_solver_status(config_solver, "UNKNOWN", f"Gurobi status {status}")

    # Return a list of solutions
    # Case 1: No solution found
    if sol_count == 0:
        print(f"[INFO] Found no solution from Gurobi (status {status}).")
        return []

    # Case 2: Single optimal solution found
    elif solution_number == 1 and getattr(model.Params, "PoolSearchMode", 0) == 0:
        sol = {v.VarName: v.X for v in model.getVars()}
        sol["obj_fun_value"] = model.ObjVal
        print("[INFO] Found 1 solution from Gurobi.")
        return [sol]

    # Case 3: Multiple solutions found
    elif solution_number > 1 or getattr(model.Params, "PoolSearchMode", 0) > 0:
        sol_list = []
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i
            sol = {v.VarName: v.Xn for v in model.getVars()}
            sol.update({"obj_fun_value": model.PoolObjVal})
            sol_list.append(sol)
        print(f"[INFO] Found {len(sol_list)} solution(s) from Gurobi.")
        return sol_list


def solve_milp_scip(filename, config_solver):
    """Solve a MILP model using SCIP; raise if PySCIPOpt is unavailable."""
    Model = _load_scip_model()
    if Model is None:
        raise RuntimeError("[ERROR] PySCIPOpt can't be loaded. Install it: pip install pyscipopt")

    try:
        model = Model()
        model.readProblem(filename)
        # Set Parameters provided by SCIP. TO DO MORE
        if "time_limit" in config_solver:
            model.setRealParam("limits/time", config_solver["time_limit"])
        solution_number = config_solver.get("solution_number", 1)
        if isinstance(solution_number, int) and solution_number > 1: # TO DO: support multiple solutions
            print("[WARNING] It currently does not support finding multiple solutions ... returning only one solution")
            model.setIntParam("limits/solutions", solution_number)
        # Solve the model
        model.optimize()
        sol_count = model.getNSols()
        status = model.getStatus()
    except _scip_error_types() as e:
        raise RuntimeError(f"[ERROR] SCIP failed: {e}") from e

    # How it ended, before the solutions (as in solve_milp_gurobi).
    if status == "infeasible":
        _set_solver_status(config_solver, "UNSAT")
    elif status != "optimal":
        _set_solver_status(config_solver, "UNKNOWN", f"SCIP status '{status}'")

    # Return a list of solutions
    if sol_count == 0:
        print(f"[INFO] Found no solution from SCIP (status '{status}').")
        return []

    else:
        sol = model.getBestSol()
        sol_dic = {v.name: model.getSolVal(sol, v) for v in model.getVars()}
        sol_dic["obj_fun_value"] = model.getSolObjVal(sol)
        print("[INFO] Found 1 solution from SCIP.")
        return [sol_dic]


def solve_sat(filename, variable_map, config_solver=None):
    """
    Solve a SAT problem (DIMACS CNF ``filename``).

    Args:
        filename (str): Path to the CNF file.
        variable_map (dict): Mapping of variable names to their DIMACS integer ids.
        config_solver (dict):
            - solver: solver name (e.g. "DEFAULT", "ORTools", "Cadical103",
                or an external binary: "PRS", "CaDiCaL", "CryptoMiniSat", "Kissat").
            - backend (optional): "pysat" or "external"; disambiguates solver names
                shared by both backends (e.g. PySAT "CryptoMinisat" vs the external
                "CryptoMiniSat" binary). See :func:`normalize_sat_solver_name`.
            - solution_number: The number of solutions to find (default: 1).

    Returns:
        list: Up to ``solution_number`` solutions, each a dict mapping variable names to 0/1.
            An empty list means unsatisfiable only under ``config_solver["solver_status"] == "UNSAT"``.
            An unloadable backend, a missing binary or unparsable solver output raises.
    """

    if config_solver is None:
        config_solver = {}
    if not isinstance(config_solver, dict):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")
    for key in ("solver_status", "solver_status_reason"):
        config_solver.pop(key, None)  # the setdefault below must not meet the previous solve's status
    solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"),
                                       backend=config_solver.get("backend"))
    config_solver["solver"] = solver
    if config_solver.get("backend") is None and solver.lower() in EXTERNAL_SAT_SOLVER_NAME_MAP:
        # Record the resolved engine for ambiguous names, so downstream re-normalization
        # (e.g. the version lookup below) does not re-print the disambiguation notice.
        config_solver["backend"] = "external" if solver in EXTERNAL_SAT_SOLVERS else "pysat"
    if "solver_version" not in config_solver:  # resolve once (guard: setdefault would recompute every call)
        config_solver["solver_version"] = _sat_solver_version(config_solver)
    print(f"[INFO] Solving SAT model with settings: {_printable_config(config_solver)}")
    monitor = RuntimeResourceMonitor(interval=0.2)
    monitor.start()
    time_start = time.time()
    try:
        if solver in ("DEFAULT", *PYSAT_SOLVERS):
            solutions = solve_sat_pysat(filename, variable_map, config_solver)
        elif solver == "ORTools":
            solutions = solve_sat_cpsat(filename, variable_map, config_solver)
        elif solver in EXTERNAL_SAT_SOLVERS:
            solutions = solve_sat_external(filename, variable_map, config_solver)
        else:
            solutions = []
        # Normalize a None to [] so callers always get a list (parity with solve_milp).
        solutions = solutions if solutions is not None else []
        config_solver.setdefault("solver_status", "SAT" if solutions else "UNSAT")
        return solutions
    finally:
        config_solver["resource_usage"] = monitor.stop()
        elapsed = round(time.time() - time_start, 2)
        config_solver.setdefault("solving_time", []).append(elapsed)  # per-call trace; total = sum(solving_time)
        print(f"[INFO] solve #{len(config_solver['solving_time'])}: {elapsed} s | cumulative solver time: {round(sum(config_solver['solving_time']), 2)} s")


def solve_sat_pysat(filename, variable_map, config_solver):
    """Solve a SAT model (DIMACS CNF ``filename``) with PySAT.

    Shared arguments as in :func:`solve_sat`. Returns up to ``solution_number``
    ``{variable_name: 0/1}`` solutions; raise if PySAT is unavailable.
    """
    CNF, Solver = _load_pysat()
    if CNF is None or Solver is None:
        raise RuntimeError("[ERROR] pysat can't be loaded. Install it: pip install python-sat")

    solver = config_solver.get("solver", "DEFAULT")
    solution_number = config_solver.get("solution_number", 1)
    trail_var_nums = _resolve_block_projection(variable_map, config_solver)
    cnf = CNF(filename)
    if solver == "DEFAULT":
        pysat_solver = Solver()
    else:
        pysat_solver = Solver(name=solver)

    try:
        pysat_solver.append_formula(cnf.clauses)

        sol_list = []
        while len(sol_list) < solution_number and pysat_solver.solve():
            model = set(pysat_solver.get_model())
            sol_list.append(_solution_from_model(model, variable_map))
            block_clause = _projected_block_clause(model, trail_var_nums)
            if not block_clause:
                break  # nothing to block -> the same solution would be re-found forever
            pysat_solver.add_clause(block_clause)
    finally:
        pysat_solver.delete()
    print(f"[INFO] Found {len(sol_list)} solution(s) from PySAT.")
    return sol_list


def _read_dimacs_clauses(filename):
    """Read a DIMACS CNF file into a list of integer clauses (dropping the trailing 0)."""
    clauses = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line[0] in ("c", "p", "%"):
                continue
            literals = [int(token) for token in line.split()]
            if literals and literals[-1] == 0:
                literals = literals[:-1]
            if literals:
                clauses.append(literals)
    return clauses


def _resolve_block_projection(variable_map, config_solver):
    """Return the DIMACS ids the blocking clause is projected onto
    (``config_solver["trail_var_names"]``, or every named variable by default).

    Without a projection every full model differs, so enumeration would return
    trivially distinct assignments; projecting onto the trail variables
    enumerates distinct trails.
    """
    trail_var_names = config_solver.get("trail_var_names")
    if trail_var_names is None:
        return set(variable_map.values())
    present = [v for v in trail_var_names if v in variable_map]
    dropped = len(trail_var_names) - len(present)
    if dropped:
        print(f"[INFO] {dropped} of {len(trail_var_names)} trail_var_names are not in the model "
              f"(unconstrained variables); dropping them from the block.")
    if not present:
        raise ValueError("None of the supplied trail_var_names exist in the model; "
                         "cannot build a projected block clause.")
    return set(variable_map[v] for v in present)


def _solution_from_model(model, variable_map):
    """Map a set of signed DIMACS literals back to a ``{variable_name: 0/1}`` solution."""
    sol = {}
    for var, value in variable_map.items():
        if value in model:
            sol[var] = 1
        elif -value in model:
            sol[var] = 0
    return sol


def _projected_block_clause(model, trail_var_nums):
    """Negate the projected assignment; an empty clause means nothing to block (caller stops)."""
    return [-l for l in model if abs(l) in trail_var_nums]


def _set_cpsat_status(config_solver, cp_model, status, solutions):
    """Map a CP-SAT status to a solver status; an invalid model is a bug, not an answer."""
    if solutions:
        _set_solver_status(config_solver, "SAT")
    elif status == cp_model.INFEASIBLE:
        _set_solver_status(config_solver, "UNSAT")
    elif status == cp_model.MODEL_INVALID:
        raise RuntimeError("[ERROR] CP-SAT rejected the model as invalid.")
    else:  # UNKNOWN, or a status this version does not name: nothing proven
        _set_solver_status(config_solver, "UNKNOWN", f"CP-SAT status {status}")


def solve_sat_cpsat(filename, variable_map, config_solver):
    """Solve a SAT model (DIMACS CNF ``filename``) with Google OR-Tools CP-SAT.

    Shared arguments as in :func:`solve_sat`. Returns up to ``solution_number``
    ``{variable_name: 0/1}`` solutions; raise if OR-Tools CP-SAT is unavailable.
    Known difference: multi-solution enumeration is callback-based and does not
    project blocking clauses onto ``trail_var_names`` (unlike the PySAT/external backends).
    """
    cp_model = _load_ortools_cpsat()
    if cp_model is None:
        raise RuntimeError("[ERROR] ortools can't be loaded. Install it: pip install ortools")

    # Creates the model
    model = cp_model.CpModel()

    # Creates the variables (the CNF is numeric, so index Boolean variables by DIMACS id).
    clauses = _read_dimacs_clauses(filename)
    boolean_var_map = {}
    for var_id in {abs(lit) for clause in clauses for lit in clause} | set(variable_map.values()):
        boolean_var_map[var_id] = model.new_bool_var(f"x{var_id}")

    # Add constraints
    for clause in clauses:
        model.add_bool_or([boolean_var_map[-lit].Not() if lit < 0 else boolean_var_map[lit] for lit in clause])

    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    if config_solver is None:
        config_solver = {}
    solution_number = config_solver.get("solution_number", 1)
    if not isinstance(solution_number, int) or isinstance(solution_number, bool) or solution_number < 1:
        raise ValueError(f"[ERROR] solution_number = {solution_number!r} must be an integer >= 1.")
    sol_list = []

    if solution_number == 1:
        status = solver.solve(model)
        sol = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for var, var_id in variable_map.items():
                sol[var] = solver.value(boolean_var_map[var_id])
            sol_list.append(sol)
        _set_cpsat_status(config_solver, cp_model, status, sol_list)
        print(f"[INFO] Found {len(sol_list)} solution(s) from OR-Tools CP-SAT.")
        return sol_list

    else:  # guarded above: an int >= 1, so this is solution_number > 1
        class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
            def __init__(self, variable_map, boolean_var_map, solution_number):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.variable_map = variable_map
                self.boolean_var_map = boolean_var_map
                self.solution_number = solution_number
                self.solutions = []
            def on_solution_callback(self):
                if len(self.solutions) >= self.solution_number:
                    return
                sol = {}
                for var, var_id in self.variable_map.items():
                    sol[var] = self.value(self.boolean_var_map[var_id])
                self.solutions.append(sol)

        solution_printer = VarArraySolutionPrinter(variable_map, boolean_var_map, solution_number)
        status = solver.SearchForAllSolutions(model, solution_printer)
        _set_cpsat_status(config_solver, cp_model, status, solution_printer.solutions)
        print(f"[INFO] Found {len(solution_printer.solutions)} solution(s) from OR-Tools CP-SAT.")
        return solution_printer.solutions


def solve_sat_external(filename, variable_map, config_solver):
    """Solve a SAT model (DIMACS CNF ``filename``) with an external binary registered in EXTERNAL_SAT_SOLVERS.

    Shared arguments as in :func:`solve_sat`. Extra ``config_solver`` keys:
        - solver_bin: binary path/name overriding the registry default (optional).
        - solver-specific keys read by the registry builder (e.g. ``threads`` for PRS).

    Returns up to ``solution_number`` solutions, cut short on UNSAT or a timeout -- which
    ``solver_status`` tells apart. An unregistered solver, a missing binary or bad output raises.
    """

    solver = config_solver.get("solver")
    build_args = EXTERNAL_SAT_SOLVERS.get(solver)
    if build_args is None:
        raise ValueError(f"[ERROR] '{solver}' is not a registered external solver. "
                         f"Supported: {', '.join(EXTERNAL_SAT_SOLVERS)}.")

    timeout = config_solver.get("time_limit", None)
    solution_number = config_solver.get("solution_number", 1)
    trail_var_nums = _resolve_block_projection(variable_map, config_solver)
    base_cmd = build_args(config_solver) + list(config_solver.get("extra_args", []))

    # Only enumeration needs a WORKING COPY of the CNF (the original file is untouched):
    # split off the 'p cnf' header so the clause count can be rewritten with the blocks.
    if solution_number > 1:
        with open(filename) as f:
            cnf_lines = f.read().splitlines()
        header_idx = next((i for i, l in enumerate(cnf_lines) if l.lstrip().startswith("p cnf")), None)
        if header_idx is None:
            raise ValueError(f"[ERROR] {filename} has no 'p cnf' header, so the blocking clauses "
                             f"cannot be counted; enumeration needs a well-formed DIMACS file.")
        nv, nc = (int(x) for x in cnf_lines[header_idx].split()[2:4])
        body = cnf_lines[header_idx + 1:]        # the clause lines

    work_path = None
    blocks = []       # accumulated blocking clauses (one per solution found)
    sol_list = []
    try:
        while len(sol_list) < solution_number:
            # First run uses the original file; later runs use the working copy with blocks.
            cmd = base_cmd + [filename if not blocks else work_path]

            # A timeout stops enumeration and already-found solutions are still returned, with
            # the status saying they are partial. A missing binary is a configuration error.
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                print(f"[WARNING] {solver} timed out after {timeout}s ... stopping enumeration.")
                _set_solver_status(config_solver, "UNKNOWN", f"timed out after {timeout}s")
                break
            except FileNotFoundError as e:
                raise RuntimeError(f"[ERROR] {solver} binary not found at '{cmd[0]}'. "
                                   f"Set config_solver['solver_bin'] or put it on PATH.") from e

            # Parse the standard DIMACS competition output (s-line + v model lines).
            status, literals = None, []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("s "):
                    status = line[2:].strip()
                elif line.startswith("v "):
                    literals += [int(tok) for tok in line[2:].split()]

            if status == "UNSATISFIABLE":
                print("[INFO] UNSAT reached.")
                if not sol_list:  # mid-enumeration it only means "no further solution"
                    _set_solver_status(config_solver, "UNSAT")
                break
            if status == "UNKNOWN":  # DIMACS: the solver gave up at its own limit
                print(f"[WARNING] {solver} returned UNKNOWN ... stopping.")
                _set_solver_status(config_solver, "UNKNOWN", f"{solver} returned UNKNOWN")
                break
            if status != "SATISFIABLE":
                unreadable = (f"{solver} returned status '{status}', which is not DIMACS "
                              f"competition output; its result cannot be read")
                if sol_list:  # as for a timeout: the solutions already found are still valid
                    print(f"[WARNING] {unreadable} ... stopping enumeration.")
                    _set_solver_status(config_solver, "UNKNOWN", unreadable)
                    break
                raise RuntimeError(f"[ERROR] {unreadable}.")

            # Set of signed literals (drop the terminating 0). A SATISFIABLE status with no
            # model means the solver did not print its assignment -> guard against a silent
            # empty solution rather than returning garbage.
            model = set(lit for lit in literals if lit != 0)
            if not model:
                raise RuntimeError(f"[ERROR] {solver} reported SATISFIABLE but emitted no model "
                                   f"(v-lines); ensure it is invoked to print the assignment.")

            sol_list.append(_solution_from_model(model, variable_map))

            # Block this (projected) assignment so the next run yields a different trail.
            block_clause = _projected_block_clause(model, trail_var_nums)
            if not block_clause:
                break  # nothing to block -> the same solution would be re-found forever
            blocks.append(block_clause)
            if len(sol_list) < solution_number:  # rebuild the working copy with the blocks
                if work_path is None:
                    fd, work_path = tempfile.mkstemp(prefix="ext_sat_", suffix=".cnf",
                                                     dir=os.path.dirname(os.path.abspath(filename)))
                    os.close(fd)
                with open(work_path, "w") as fh:
                    fh.write(f"p cnf {nv} {nc + len(blocks)}\n")
                    fh.write("\n".join(body) + "\n")
                    for bc in blocks:
                        fh.write(" ".join(str(x) for x in bc) + " 0\n")
    finally:
        if work_path is not None and os.path.exists(work_path):
            os.remove(work_path)

    print(f"[INFO] Found {len(sol_list)} solution(s) from {solver}.")
    return sol_list
