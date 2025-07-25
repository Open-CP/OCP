"""
This module provides tools for constructing and solving MILP/SAT models.
Core functionalities:
1. Formulate MILP/SAT models from constraints and objective functions.
2. Solve MILP/SAT models using supported solvers:
    - MILP solvers: Gurobi, SCIP, OR-Tools
    - SAT solvers: PySAT, OR-Tools 
3. Solve MILP/SAT models under different targets: 'OPTIMAL', 'SATISFIABLE', 'ALL solutions'
"""

import os

try: # Solve MILP model using Gurobi solver
    import gurobipy as gp
    gurobipy_import = True
except ImportError:
    print("gurobipy module can't be loaded \n")
    gurobipy_import = False
    pass

try: # Solve MILP model using SCIP solver
    from pyscipopt import Model
    scip_import = True
except ImportError:
    print("PySCIPOpt module can't be loaded \n")
    scip_import = False
    pass

try: # Solve MILP model using Or-tools solver. TO DO
    from ortools.linear_solver import pywraplp
    import ortoolslpparser
    ortools_import = True
except ImportError:
    print("ortools module can't be loaded \n")
    ortools_import = False
    pass

try: # Solve SAT model using a solver from python-sat
    from pysat.solvers import Solver
    from pysat.formula import CNF
    pysat_import = True
except ImportError:
    print("pysat module can't be loaded \n")
    pysat_import = False
    pass


def solve_milp(filename, solving_args=None):
    """
    Solve a MILP model.
    
    Args:
        filename (str): Path to the MILP model file.
        solving_args (dict): 
            - target: The optimization target:
                - "OPTIMAL": Find the optimal solution.
                - "SATISFIABLE": Find a feasible solution.
                - "ALL": Find All feasible solutions.
            - solver: solver name (e.g, "Gurobi", "SCIP")
    
    Returns: 
        - If target is "OPTIMAL" or "SATISFIABLE", returns a dict of variable assignments (a solution).
        - If target is "ALL", returns a list of such dicts (all solutions).
        - None if no feasible solution is found or solver fails.
    """

    solving_args = solving_args or {}
    solver = solving_args.get("solver", "DEFAULT")
    print(f"[INFO] Solving MILP model with settings: {solving_args}")
    if solver in ["Gurobi", "DEFAULT"]:
        return solve_milp_gurobi(filename, solving_args)    
    elif solver == "SCIP":
        return solve_milp_scip(filename, solving_args)
    raise ValueError(f"[ERROR] Unsupported solver: '{solver}'. Supported: 'Gurobi' (DEFAULT), 'SCIP'.")


def solve_milp_gurobi(filename, solving_args): # Solve a MILP model using Gurobi.
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return None
    
    # Load the model
    try:
        model = gp.read(filename)
    except gp.GurobiError as e:
        print(f"[ERROR] Model loading from {filename}: {e}")
        return None

    # Set Parameters provided by Gurobi. TO DO MORE 
    if "time_limit" in solving_args: 
        model.Params.timeLimit = solving_args["time_limit"]
    if "OutputFlag" in solving_args:
        model.Params.OutputFlag = solving_args["OutputFlag"]

    # Solve the model
    target = solving_args.get("target", "DEFAULT")
    try:
        if target == "ALL":
            model.Params.PoolSearchMode = 2
            model.Params.PoolSolutions = 1000000
        elif target == "SATISFIABLE":
            model.Params.MIPFocus = 1
            model.Params.SolutionLimit = 1
        model.optimize()
        print("[INFO] Solving status:", model.status)
    except gp.GurobiError:
        print("[ERROR] Check your Gurobi license,  visit https://gurobi.com/unrestricted for more information\n")
        return None
    
    if target in ["OPTIMAL", "DEFAULT", "SATISFIABLE"]:
        if model.status in [gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT, gp.GRB.Status.SUBOPTIMAL, gp.GRB.Status.INTERRUPTED]:
            sol_dic = {v.VarName: int(round(v.Xn)) for v in model.getVars()}
            sol_dic["obj_fun_value"] = model.ObjVal
            sol_dic["status"] = model.status
            return sol_dic
        else:
            return None

    elif target == "ALL":
        print("[INFO] Number of solutions found: ", model.SolCount)
        sol_list = []
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i  
            sol_dic = {v.VarName: int(round(v.Xn)) for v in model.getVars()}
            sol_dic["obj_fun_value"] = model.PoolObjVal
            sol_dic["status"] = model.status
            sol_list.append(sol_dic)
        return sol_list
    
    raise ValueError(f"[ERROR] Unknown target: {target}")


def solve_milp_scip(filename, solving_args): # Solve a MILP model using SCIP.
    if not scip_import:
        print("PySCIPOpt module can't be loaded ... skipping SCIP test\n")
        return None
    
    try:
        model = Model()
        model.readProblem(filename)
    except Exception as e:
        print(f"[ERROR] Failed to read MILP model from '{filename}': {e}")
        return None
    
    # Set Parameters provided by SCIP. TO DO MORE 
    if "time_limit" in solving_args: 
        model.setRealParam("limits/time", solving_args["time_limit"])
    
    # Solve the model
    target = solving_args.get("target", "DEFAULT")
    if target == "SATISFIABLE":
        model.setIntParam("limits/solutions", 1)
    if target in ["OPTIMAL", "DEFAULT", "SATISFIABLE"]:
        model.optimize()
        print("[INFO] Solving status: ",model.getStatus())
        if model.getStatus() in ["optimal", "feasible", "solution limit", "time limit", "userinterrupt"]:
            sol_dic = {v.name: int(round(model.getVal(v))) for v in model.getVars()}
            sol_dic["obj_fun_value"] = model.getObjVal()
            sol_dic["status"] = model.getStatus()
            return sol_dic
        return None
        
    elif target == "ALL":
        sol_list = []
        while model.getStatus() != "infeasible":
            model.optimize()
            print("[INFO] Solving status: ",model.getStatus())
            if model.getStatus() in ["optimal", "feasible", "solution limit", "time limit", "userinterrupt"]:
                sol_dic = {v.name: int(round(model.getVal(v))) for v in model.getVars()}
                sol_dic["obj_fun_value"] = model.getObjVal()
                sol_dic["status"] = model.getStatus()
                sol_list.append(sol_dic)
            else:
                break
        return sol_list
        
    raise ValueError(f"Unknown target: {target}")


def gen_milp_model(constraints, obj_fun=None, filename=""): # Generate anf write the MILP model in standard .lp format, based on the given constraints and objective function.
    # === Step 1: Define the MILP Model Structure === #
    content = "Minimize\nobj\nSubject To\n"

    # === Step 2: Process Constraints === #
    bin_vars, in_vars = [], []
    for constraint in constraints:
        if "Binary" in constraint:
            constraint_split = constraint.split('Binary\n')
            content += constraint_split[0]
            bin_vars += constraint_split[1].strip().split()
        elif "Integer" in constraint:
            constraint_split = constraint.split('Integer\n')
            content += constraint_split[0]
            in_vars += constraint_split[1].strip().split()
        else: content += constraint + '\n'
    
    # === Step 3: Define the Objective Function === #
    if obj_fun:
        if isinstance(obj_fun[0], list):
            obj_fun_flatten = [obj for row in obj_fun for obj in row]
            content += " + ".join(obj_fun_flatten) + ' - obj = 0\n'
        elif isinstance(obj_fun[0], str):
            content += " + ".join(obj_fun) + ' - obj = 0\n'        

    # === Step 4: Declare Binary and Integer Variables === #
    if bin_vars: 
        content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if in_vars: 
        content += "Integer\n" + " ".join(set(in_vars)) + "\n"

    # === Step 5: Write the model into a file === #
    if filename:
        dir_path = os.path.dirname(filename)
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path, exist_ok=True)
        with open(filename, "w") as myfile:
            myfile.write(content + "End\n")    
    
    return {"content": content}  # Return the model content and objective function


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
        print("pysat module can't be loaded ... skipping test\n")
        return None
    
    solver = solving_args.get("solver", "DEFAULT")
    target = solving_args.get("target", "DEFAULT")
    cnf = CNF(filename)
    if solver == "DEFAULT":
        solver = Solver()
    else:
        solver = Solver(name=solver)
    
    solver.append_formula(cnf.clauses)

    if target in ["SATISFIABLE", "DEFAULT"]:
        if solver.solve():
            model = solver.get_model()
            sol_dic = {}
            for var, value in variable_map.items():
                if value in model:
                    sol_dic[var] = 1
                elif -value in model:
                    sol_dic[var] = 0
            return sol_dic
        else:
            print("No solution exists.")
            return None
        
    elif target == "ALL":
        sol_list = []
        while solver.solve():
            model = solver.get_model()
            sol_dic = {}
            for var, value in variable_map.items():
                if value in model:
                    sol_dic[var] = 1
                elif -value in model:
                    sol_dic[var] = 0       
            sol_list.append(sol_dic)
            block_clause = [-l for l in model] # TO DO: optimaize: if abs(l) in main_vars
            solver.add_clause(block_clause)
        solver.delete()
        print("[INFO] Number of solutions found: ", len(sol_list))
        return sol_list
    
    raise ValueError(f"[ERROR] Unknown target: {target}")
    

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
    # === Step 1: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints)
    
    # === Step 2: Generate the CNF Model === #
    num_clause = len(constraints)
    content = f"p cnf {num_var} {num_clause}\n"  
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    
    # === Step 3. Write the model into a file === #
    if filename:
        dir_path = os.path.dirname(filename)
        if not os.path.exists(dir_path): 
            os.makedirs(dir_path, exist_ok=True)
        with open(filename, "w") as myfile:
            myfile.write(content)    

    return {"content": content, "variable_map": variable_map}