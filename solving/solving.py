"""
This module provides tools for constructing and solving MILP/SAT models.
Core functionalities:
1. Formulate MILP/SAT models from constraints and objective functions.
2. Solve MILP/SAT models using supported solvers:
    - MILP solvers: Gurobi, SCIP, OR-Tools
    - SAT solvers: PySAT, OR-Tools 
3. Solve MILP/SAT models under different parameters and settings.
"""

import os

try: # Solve MILP model using Gurobi solver
    import gurobipy as gp
    gurobipy_import = True
except ImportError:
    print("[WARNING] gurobipy module can't be loaded \n")
    gurobipy_import = False
    pass

try: # Solve MILP model using SCIP solver
    from pyscipopt import Model
    scip_import = True
except ImportError:
    print("[WARNING] PySCIPOpt module can't be loaded \n")
    scip_import = False
    pass

try: # Solve MILP model using Or-tools solver. TO DO
    from ortools.linear_solver import pywraplp
    import ortoolslpparser
    ortools_import = True
except ImportError:
    print("[WARNING] ortools module can't be loaded \n")
    ortools_import = False
    pass

try: # Solve SAT model using a solver from python-sat
    from pysat.solvers import Solver
    from pysat.formula import CNF
    pysat_import = True
except ImportError:
    print("[WARNING] pysat module can't be loaded \n")
    pysat_import = False
    pass

def solve_milp(filename, solving_args=None):
    """
    Solve a MILP model.
    
    Parameters:
        filename (str): Path to the MILP model file.
        solving_args (dict): 
            - solver: solver name (e.g, "GUROBI", "SCIP").
            - solution_number: The number of solutions to find (default: 1).
    
    Returns:
            A list of solutions. Each solution is represented as a dictionary mapping variable names to their values.
    """

    solving_args = solving_args or {}
    solver = solving_args.get("solver", "DEFAULT")
    print(f"[INFO] Solving MILP model with settings: {solving_args}")
    if solver.upper() in ["GUROBI", "DEFAULT"]:
        return solve_milp_gurobi(filename, solving_args)    
    elif solver.upper() == "SCIP":
        return solve_milp_scip(filename, solving_args)
    raise ValueError(f"[ERROR] Unsupported solver: '{solver}'. Supported: 'GUROBI' (DEFAULT), 'SCIP'.")


def solve_milp_gurobi(filename, solving_args): # Solve a MILP model using Gurobi.
    if gurobipy_import == False: 
        print("[WARNING] gurobipy module can't be loaded ... skipping test\n")
        return []
    
    try:
        model = gp.read(filename) # Load the model from file.
        # Set Parameters provided by Gurobi.
        for key, val in solving_args.items():
            if hasattr(model.Params, key):
                setattr(model.Params, key, val)
        solution_number = solving_args.get("solution_number", 1)
        if isinstance(solution_number, int) and solution_number > 1:
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = solution_number
        # Solve the model
        model.optimize()
        sol_count = getattr(model, "SolCount", 0)
    except gp.GurobiError:
        print("[ERROR] Check your Gurobi license, visit https://gurobi.com/unrestricted for more information\n")
        return []

    # Return a list of solutions
    # Case 1: No solution found
    if sol_count == 0:
        print(f"[INFO] Found no solution.")
        return []
    
    # Case 2: Single optimal solution found
    elif solution_number == 1 or getattr(model.Params, "PoolSearchMode", 0) == 0:
        sol = {v.VarName: v.X for v in model.getVars()}
        sol["obj_fun_value"] = model.ObjVal
        print(f"[INFO] Found 1 solution.")
        return [sol]
    
    # Case 3: Multiple solutions found
    elif getattr(model.Params, "PoolSearchMode", 0) > 0:
        sol_list = []
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i  
            sol = {v.VarName: v.Xn for v in model.getVars()}
            sol.update({"obj_fun_value": model.PoolObjVal})
            sol_list.append(sol)
        print(f"[INFO] Found {len(sol_list)} solutions.")
        return sol_list


def solve_milp_scip(filename, solving_args): # Solve a MILP model using SCIP. It supports finding one solution currently. TO DO: finding multiple solutions
    if not scip_import:
        print("[WARNING] PySCIPOpt module can't be loaded ... skipping SCIP test\n")
        return []
    
    try:
        model = Model()
        model.readProblem(filename)
        # Set Parameters provided by SCIP. TO DO MORE 
        if "time_limit" in solving_args: 
            model.setRealParam("limits/time", solving_args["time_limit"])
        solution_number = solving_args.get("solution_number", 1)
        if isinstance(solution_number, int) and solution_number > 1: # TO DO: support multiple solutions
            print("[WARNING] It currently does not support finding multiple solutions ... returning only one solution\n")
            model.setIntParam("limits/solutions", solution_number)
        # Solve the model
        model.optimize()
        sol_count = model.getNSols()
    except Exception as e:
        print(f"[WARNING] SCIP solver error: {e} ... skipping test\n")
        return []
    
    # Return a list of solutions
    if sol_count == 0:
        print(f"[INFO] Found no solution.")
        return []
    
    else:
        sol = model.getBestSol()
        sol_dic = {v.name: model.getSolVal(sol, v) for v in model.getVars()}
        sol_dic["obj_fun_value"] = model.getSolObjVal(sol)
        print(f"[INFO] Found 1 solution.")
        return [sol_dic]

def gen_milp_model(constraints, obj_fun=None, filename=""): # Generate and write the MILP model in standard .lp format, based on the given constraints and objective function.
    if not filename:
        raise ValueError("Please specify an output filename for the MILP model.")
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filename, "w") as f:
        # === Step 1: Define the MILP Model Structure === #
        # If an objective function (obj_fun) is provided, write a symbolic objective name "obj", which will be defined later in the constraint section. Otherwise, write "Minimize 0" to indicate a feasibility-only model.
        if obj_fun:
            f.write("Minimize\n obj\nSubject To\n")
        else:
            f.write("Minimize\n 0\nSubject To\n")

        # === Step 2: Process Constraints === #
        bin_vars, in_vars = set(), set()
        for constraint in constraints:
            if "Binary" in constraint:
                parts = constraint.split('Binary\n')
                if parts[0].strip():
                    f.write(parts[0].strip() + "\n")
                for segment in parts[1:]:
                    seg = segment.strip()
                    if seg:
                        bin_vars.update(seg.split())
            elif "Integer" in constraint:
                parts = constraint.split('Integer\n')
                if parts[0].strip():
                    f.write(parts[0].strip() + "\n")
                for segment in parts[1:]:
                    seg = segment.strip()
                    if seg:
                        in_vars.update(seg.split())
            else:
                f.write(constraint if constraint.endswith('\n') else constraint + '\n')
        
        # === Step 3: Define the Objective Function === #
        if obj_fun:
            if isinstance(obj_fun[0], list):
                obj_terms = [obj for row in obj_fun for obj in row]
            else:
                obj_terms = obj_fun
            f.write(" + ".join(obj_terms) + " - obj = 0" + "\n")              

        # === Step 4: Declare Binary and Integer Variables === #
        if bin_vars:
            f.write("Binary\n" + " ".join(sorted(bin_vars)) + "\n")
        if in_vars:
            f.write("Integer\n" + " ".join(sorted(in_vars)) + "\n")

        f.write("End\n")
    
    return None


def solve_sat(filename, variable_map, solving_args=None):
    """
    Solve a SAT problem
    
    Args:
        filename (str): Path to the CNF file.
        solving_args (dict): 
            - target: The optimization target:
                - "SATISFIABLE": Find a feasible solution.
                - "All": Find all feasible solutions.
            - solver: solver name (e.g, "ORTools", "Cadical103")
    
    Returns: 
        - If target is "SATISFIABLE", returns a dict of variable assignments (a solution).
        - If target is "ALL", returns a list of such dicts (all solutions).
        - None if no feasible solution is found or solver fails.
    """
    
    solving_args = solving_args or {}
    solver = solving_args.get("solver", "DEFAULT")
    print(f"[INFO] Solving SAT model with settings: {solving_args}")
    if solver in ["DEFAULT", "Cadical103", "Cadical153", "Cadical195", "CryptoMinisat", "Gluecard3", "Gluecard4", "Glucose3", "Glucose4", "Lingeling", "MapleChrono", "MapleCM", "Maplesat", "Mergesat3", "Minicard", "Minisat22", "MinisatGH"]:
        return solve_sat_pysat(filename, variable_map, solving_args)    
    elif solver == "ORTools":
        return solve_sat_ortools(filename, variable_map, solving_args)
    raise ValueError(f"[ERROR] Unsupported solver: '{solver}'. Supported: ORTools, DEFAULT, Cadical103, Cadical153, Cadical195, CryptoMinisat, Gluecard3, Gluecard4, Glucose3, Glucose4, Lingeling, MapleChrono, MapleCM, Maplesat, Mergesat3, Minicard, Minisat22, MinisatGH'.")


def solve_sat_pysat(filename, variable_map, solving_args):
    if not pysat_import:
        print("[WARNING] pysat module can't be loaded ... skipping test\n")
        return None
    
    solver = solving_args.get("solver", "DEFAULT")
    solution_number = solving_args.get("solution_number", 1)
    cnf = CNF(filename)
    if solver == "DEFAULT":
        solver = Solver()
    else:
        solver = Solver(name=solver)
    
    solver.append_formula(cnf.clauses)

    sol_count = 0
    sol_list = []
    while sol_count < solution_number and solver.solve():
        model = solver.get_model()
        sol = {}
        for var, value in variable_map.items():
            if value in model:
                sol[var] = 1
            elif -value in model:
                sol[var] = 0
        sol_list.append(sol)
        block_clause = [-l for l in model] # TO DO: optimaize: if abs(l) in main_vars
        solver.add_clause(block_clause)
        sol_count += 1
    solver.delete()
    print(f"[INFO] Found {len(sol_list)} solutions.")
    return sol_list
    

def solve_sat_ortools(filename, variable_map, solving_args): # TO DO
    return None


def create_numerical_cnf(cnf): # Convert a given CNF formula into numerical CNF format. Return (number of variables, mapping of variables to numerical IDs, numerical CNF constraints)
    # Extract unique variables and assign numerical IDs
    family_of_variables = ' '.join(cnf).replace('-', '')
    variables = sorted(set(family_of_variables.split()))
    variable2number = {variable: i + 1 for (i, variable) in enumerate(variables)}
    
    # Convert CNF constraints to numerical format
    numerical_cnf = []
    for clause in cnf:
        literals = clause.split()
        numerical_literals = []
        lits_are_neg = (literal[0] == '-' for literal in literals)
        numerical_literals.extend(tuple(f'{"-" * lit_is_neg}{variable2number[literal[lit_is_neg:]]}'
                                  for lit_is_neg, literal in zip(lits_are_neg, literals)))
        numerical_clause = ' '.join(numerical_literals)
        numerical_cnf.append(numerical_clause)
    return len(variables), variable2number, numerical_cnf


def gen_sat_model(constraints=[], filename=""): # Generate and write the SAT model.
    if not filename:
        raise ValueError("Please specify an output filename for the SAT model.")
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # === Step 1: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints)
    
    # === Step 2: Prepare and write CNF file === #
    num_clause = len(constraints)

    with open(filename, "w") as f:
        f.write(f"p cnf {num_var} {num_clause}\n")
        for constraint in numerical_cnf:
            f.write(f"{constraint} 0\n")

    # === Step 3: Return metadata === #
    return {"variable_map": variable_map}