"""
This module provides tools for solving MILP/SAT models. Supports multiple solvers and configurations.
    - MILP solvers: Gurobi, SCIP
    - SAT solvers: PySAT, OR-Tools CPSAT
"""
from importlib import import_module
from importlib.util import find_spec
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
SAT_SOLVERS = ("DEFAULT", *PYSAT_SOLVERS, "ORTools")
PYSAT_SOLVER_NAME_MAP = {name.lower(): name for name in PYSAT_SOLVERS}


def _modules_available(*module_names):
    """Return True if every named module is importable, without importing any of them."""
    return all(find_spec(module_name) is not None for module_name in module_names)


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


def normalize_sat_solver_name(solver="DEFAULT"):
    """Normalize a SAT solver name to a supported value ('DEFAULT'/'ORTools'/a PySAT solver), raising on unknown."""
    solver_text = str(solver)
    if solver_text.upper() == "DEFAULT":
        return "DEFAULT"
    if solver_text.lower() == "ortools":
        return "ORTools"
    if solver_text.lower() in PYSAT_SOLVER_NAME_MAP:
        return PYSAT_SOLVER_NAME_MAP[solver_text.lower()]
    raise ValueError(
        f"[ERROR] Unsupported solver: '{solver}'. Supported: ORTools, DEFAULT, "
        + ", ".join(PYSAT_SOLVERS)
        + "."
    )


def is_solver_available(kind, solver="DEFAULT"):
    """Return whether a configured solver route is both installed and implemented."""
    kind = kind.lower()
    capabilities = solver_capabilities()

    if kind == "milp":
        try:
            solver_name = normalize_milp_solver_name(solver)
        except ValueError:
            return False
        backend = capabilities["milp"].get(solver_name)
    elif kind == "sat":
        try:
            solver_name = normalize_sat_solver_name(solver)
        except ValueError:
            return False
        if solver_name == "DEFAULT" or solver_name in PYSAT_SOLVERS:
            backend = capabilities["sat"]["PySAT"]
        elif solver_name == "ORTools":
            backend = capabilities["sat"]["ORTools"]
        else:
            backend = None
    else:
        raise ValueError(f"Unsupported solver kind: {kind!r}. Supported: 'milp', 'sat'.")

    return bool(backend and backend["available"] and backend["implemented"])


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


def _sat_solver_version(config_solver):
    """Resolve the SAT solver (PySAT-bundled or OR-Tools CPSAT) version string (no printing)."""
    solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"))
    if solver == "ORTools":
        return "OR-Tools " + _module_version("ortools")
    return _pysat_solver_version(solver)  # "DEFAULT" and every PySAT solver


def get_solver_version(kind, config_solver=None):
    """Report (print and return) the concrete solver name and version for a 'milp'/'sat' route."""
    config_solver = config_solver or {}
    kind = kind.lower()
    if kind == "milp":
        solver = normalize_milp_solver_name(config_solver.get("solver", "DEFAULT"))
        version = _milp_solver_version(config_solver)
    elif kind == "sat":
        solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"))
        version = _sat_solver_version(config_solver)
    else:
        raise ValueError(f"Unsupported solver kind: {kind!r}. Supported: 'milp', 'sat'.")
    print(f"[INFO] {kind} solver: {solver} | version: {version}")
    return {"kind": kind, "solver": solver, "solver_version": version}


def solve_milp(filename, config_solver=None):
    """
    Solve a MILP model.

    Parameters:
        filename (str): Path to the MILP model file.
        config_solver (dict):
            - solver: solver name (e.g, "GUROBI", "SCIP").
            - solution_number: The number of solutions to find (default: 1).

    Returns:
            A list of solutions. Each solution is represented as a dictionary mapping variable names to their values.
    """

    if config_solver is None:
        config_solver = {}
    if not isinstance(config_solver, dict):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")
    solver = normalize_milp_solver_name(config_solver.get("solver", "DEFAULT"))
    config_solver["solver"] = solver
    if "solver_version" not in config_solver:  # resolve once (guard: setdefault would recompute every call)
        config_solver["solver_version"] = _milp_solver_version(config_solver)
    print(f"[INFO] Solving MILP model with settings: {config_solver}")
    monitor = RuntimeResourceMonitor(interval=0.2)
    monitor.start()
    time_start = time.time()
    try:
        if solver == DEFAULT_MILP_SOLVER:
            return solve_milp_gurobi(filename, config_solver)
        elif solver == "SCIP":
            return solve_milp_scip(filename, config_solver)
    finally:
        config_solver["resource_usage"] = monitor.stop()
        elapsed = round(time.time() - time_start, 2)
        config_solver.setdefault("solving_time", []).append(elapsed)  # per-call trace; total = sum(solving_time)
        print(f"[INFO] solve #{len(config_solver['solving_time'])}: {elapsed} s | cumulative solver time: {round(sum(config_solver['solving_time']), 2)} s")


def solve_milp_gurobi(filename, config_solver):
    """Solve a MILP model using Gurobi."""
    gp = _load_gurobi()
    if gp is None:
        print("[WARNING] gurobipy module can't be loaded ... skipping test")
        return []

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
    except gp.GurobiError:
        print("[ERROR] Check your Gurobi license, visit https://gurobi.com/unrestricted for more information")
        return []

    # Return a list of solutions
    # Case 1: No solution found
    if sol_count == 0:
        print("[INFO] Found no solution from Gurobi.")
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
    """Solve a MILP model using SCIP."""
    Model = _load_scip_model()
    if Model is None:
        print("[WARNING] PySCIPOpt module can't be loaded ... skipping SCIP test")
        return []

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
    except _scip_error_types() as e:
        print(f"[WARNING] SCIP solver error: {e} ... skipping test")
        return []

    # Return a list of solutions
    if sol_count == 0:
        print("[INFO] Found no solution from SCIP.")
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
            - solver: solver name (e.g. "DEFAULT", "ORTools", "Cadical103").
            - solution_number: The number of solutions to find (default: 1).

    Returns:
        list: Up to ``solution_number`` solutions, each a dict mapping variable names to 0/1;
            an empty list if unsatisfiable or the selected backend can't be loaded.
    """

    if config_solver is None:
        config_solver = {}
    if not isinstance(config_solver, dict):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")
    solver = normalize_sat_solver_name(config_solver.get("solver", "DEFAULT"))
    config_solver["solver"] = solver
    if "solver_version" not in config_solver:  # resolve once (guard: setdefault would recompute every call)
        config_solver["solver_version"] = _sat_solver_version(config_solver)
    print(f"[INFO] Solving SAT model with settings: {config_solver}")
    monitor = RuntimeResourceMonitor(interval=0.2)
    monitor.start()
    time_start = time.time()
    try:
        if solver in ("DEFAULT", *PYSAT_SOLVERS):
            solutions = solve_sat_pysat(filename, variable_map, config_solver)
        elif solver == "ORTools":
            solutions = solve_sat_cpsat(filename, variable_map, config_solver)
        else:
            solutions = []
        # Normalize a missing-backend None to [] so callers always get a list (parity with solve_milp).
        return solutions if solutions is not None else []
    finally:
        config_solver["resource_usage"] = monitor.stop()
        elapsed = round(time.time() - time_start, 2)
        config_solver.setdefault("solving_time", []).append(elapsed)  # per-call trace; total = sum(solving_time)
        print(f"[INFO] solve #{len(config_solver['solving_time'])}: {elapsed} s | cumulative solver time: {round(sum(config_solver['solving_time']), 2)} s")


def solve_sat_pysat(filename, variable_map, config_solver):
    """Solve a SAT model (DIMACS CNF ``filename``) with PySAT.

    Returns up to ``config_solver['solution_number']`` ``{variable_name: 0/1}`` solutions
    over the variables in ``variable_map``, or ``None`` if PySAT is unavailable.
    """
    CNF, Solver = _load_pysat()
    if CNF is None or Solver is None:
        print("[WARNING] pysat module can't be loaded ... skipping test")
        return None

    solver = config_solver.get("solver", "DEFAULT")
    solution_number = config_solver.get("solution_number", 1)
    cnf = CNF(filename)
    if solver == "DEFAULT":
        pysat_solver = Solver()
    else:
        pysat_solver = Solver(name=solver)

    try:
        pysat_solver.append_formula(cnf.clauses)

        sol_count = 0
        sol_list = []
        while sol_count < solution_number and pysat_solver.solve():
            model = pysat_solver.get_model()
            sol = {}
            for var, value in variable_map.items():
                if value in model:
                    sol[var] = 1
                elif -value in model:
                    sol[var] = 0
            sol_list.append(sol)
            block_clause = [-l for l in model] # TO DO: optimaize: if abs(l) in main_vars
            pysat_solver.add_clause(block_clause)
            sol_count += 1
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


def solve_sat_cpsat(filename, variable_map, config_solver):
    """
    Solve a SAT problem using Google OR-Tools CP-SAT solver.

    Args:
        filename (str): Path to the DIMACS CNF file.
        variable_map (dict): Mapping of variable names to their DIMACS integer ids.
        config_solver (dict):
            - solution_number: The number of solutions to find (default: 1).

    Returns:
        - A list of solutions, each a dict mapping variable names to 0/1 assignments.
        - None if OR-Tools CP-SAT can't be loaded.
    """
    cp_model = _load_ortools_cpsat()
    if cp_model is None:
        print("[WARNING] OR-Tools CP-SAT module can't be loaded ... skipping test")
        return None

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
    config_solver = config_solver or {}
    solution_number = config_solver.get("solution_number", 1)
    sol_list = []

    if solution_number == 1:
        status = solver.solve(model)
        sol = {}
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            for var, var_id in variable_map.items():
                sol[var] = solver.value(boolean_var_map[var_id])
            sol_list.append(sol)
        print(f"[INFO] Found {len(sol_list)} solution(s) from OR-Tools CP-SAT.")
        return sol_list

    elif solution_number > 1:
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
        solver.SearchForAllSolutions(model, solution_printer)
        print(f"[INFO] Found {len(solution_printer.solutions)} solution(s) from OR-Tools CP-SAT.")
        return solution_printer.solutions
    return sol_list
