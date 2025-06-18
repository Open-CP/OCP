import os
# Solve MILP model using Gurobi solver
try:
    import gurobipy as gp
    gurobipy_import = True
except ImportError:
    print("gurobipy module can't be loaded \n")
    gurobipy_import = False
    pass

# Solve MILP model using SCIP solver
try:
    from pyscipopt import Model
    scip_import = True
except ImportError:
    print("PySCIPOpt module can't be loaded \n")
    scip_import = False
    pass

# Solve MILP model using Or-tools solver
try: # TO DO
    from ortools.linear_solver import pywraplp
    import ortoolslpparser
    ortools_import = True
except ImportError:
    print("ortools module can't be loaded \n")
    ortools_import = False
    pass

# Solve SAT model using a solver from python-sat
try:
    from pysat.solvers import Solver
    from pysat.formula import CNF
    pysat_import = True
except ImportError:
    print("pysat module can't be loaded \n")
    pysat_import = False
    pass


"""
This module provides functions for building and solving MILP, SAT and CP models.

### Features:
1. **MILP-Based Attack**:
   - Formulates the attack as a Mixed Integer Linear Programming (MILP) problem.
   - Uses specified solver (e.g., Gurobi, Scip) to solve MILP models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.

2. **SAT-Based Attack**:
   - Formulates the attack as a Boolean Satisfiability Problem (SAT).
   - Uses specified solver (e.g., Cadical, CryptoMinisat) to solve SAT models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.
"""

def solve_milp(filename, solving_goal="Default", solver="Default", solving_args=None):
    """
    Solve a MILP model.
    
    Args:
        filename (str): Path to the MILP model file.
        solving_goal (str): The optimization goal:
            - "Optimal": Find the optimal solution.
            - "All": Find all feasible solutions.
        solver (str): solver name (e.g, "Gurobi", "SCIP")
    
    Returns: 
        a tuple (sol_list, obj_list) containing solutions and objective value.
    """
    print(f"Solving the MILP model: solving_goal = {solving_goal}, solver = {solver}, solving_args = {solving_args}")
    if solver == "Gurobi" or solver == "Default":
        return solve_milp_gurobi(filename, solving_goal, solving_args)
    
    elif solver == "SCIP":
        return solve_milp_scip(filename, solving_goal, solving_args)
        
    return None, None


def solve_milp_gurobi(filename, solving_goal="Default", solving_args=None): # Solve a MILP model using Gurobi.
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return None, None
    
    solving_args = solving_args or {}
    model = gp.read(filename)
    if "timeLimit" in solving_args: model.Params.timeLimit = solving_args["timeLimit"]

    if solving_goal == "Optimal" or solving_goal == "Default":
        try:            
            model.optimize()
        except gp.GurobiError:
            print("Error: check your gurobi license, visit https://gurobi.com/unrestricted for more information\n")
            return None, None
        if model.status == gp.GRB.Status.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
            sol_dic = {}
            for v in model.getVars():
                sol_dic[str(v.VarName)] = int(round(v.Xn))          
            return sol_dic, model.ObjVal
        else:
            return None, None
            
    elif solving_goal == "All":
        model.Params.PoolSearchMode = 2   # Enable searching for multiple solutions
        model.Params.PoolSolutions = 1000000  # Set the maximum limit for the number of solutions
        try:
            model.optimize()
        except gp.GurobiError:
            print("Error: check your gurobi license. Visit https://gurobi.com/unrestricted for more information\n")
            return None, None
        print("Number of solutions by solving the MILP model: ", model.SolCount)
        sol_list, obj_list = [], []
        for i in range(model.SolCount):
            sol_dic = {}
            model.Params.SolutionNumber = i  
            for v in model.getVars():
                sol_dic[str(v.VarName)] = v.Xn   
            sol_list.append(sol_dic)
            obj_list.append(model.ObjVal)
        return sol_list, obj_list
    
    return None, None


def solve_milp_scip(filename, solving_goal="Default", solving_args=None): # Solve a MILP model using SCIP.
    if not scip_import:
        print("PySCIPOpt module can't be loaded ... skipping SCIP test\n")
        return None, None
    
    solving_args = solving_args or {}
    model = Model()
    model.readProblem(filename)        
    if solving_goal == "Optimal" or solving_goal == "Default":
        model.optimize()
        if model.getStatus() == "optimal":
            sol_dic = {}
            for v in model.getVars():
                sol_dic[str(v.name)] = int(round(model.getVal(v)))
            return sol_dic, model.getObjVal()
        else:
            return None, None
        
    elif solving_goal == "All":
        sol_list, obj_list = [], []
        while model.getStatus() != "infeasible":
            model.optimize()
            if model.getStatus() == "optimal":
                sol_dic = {}
                for v in model.getVars():
                    sol_dic[str(v.name)] = int(round(model.getVal(v)))
                sol_list.append(sol_dic)
                obj_list.append(model.getObjVal())
                # Extend the time limit to find next solution
                model.setSolvingLimit(model.getSolvingTimeLimit() + 1)  # Increasing the time limit for the next solution
            else:
                break
        return sol_list, obj_list
        
    return None, None


def gen_milp_model(constraints, obj_fun=None, filename=""): # Generate anf write the MILP model in standard .lp format, based on the given constraints and objective function.
    # === Step 1: Define the MILP Model Structure === #
    content = ""
    if obj_fun:
        content += "Minimize\nobj\n"
    content += "Subject To\n"


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
    
    return content


def solve_sat(filename, variable_map, solving_goal="Default", solver="Default"):
    """
    Solve a SAT problem
    
    Args:
        filename (str): Path to the CNF file.
        solving_goal (str): The optimization goal:
            - "Feasible": Find a feasible solution.
            - "All": Find all feasible solutions.
        solver (str): solver name.
    
    Returns: 
        a list of all solutions found.
    """
    print(f"Solving the SAT model: solving_goal = {solving_goal}, solver = {solver}")
    if not pysat_import:
        print("pysat module can't be loaded ... skipping test\n")
        return None
    
    cnf = CNF(filename)
    if solver in ["Default", "Cadical103", "Cadical153", "Cadical195", "CryptoMinisat", "Gluecard3", "Gluecard4", "Glucose3", "Glucose4", "Lingeling", "MapleChrono", "MapleCM", "Maplesat", "Mergesat3", "Minicard", "Minisat22", "MinisatGH"]:
        if solver == "Default":
            solver = Solver()
        else:
            solver = Solver(name=solver)
    else: print("No SAT Solver Support!")
    solver.append_formula(cnf.clauses)

    if solving_goal == "Feasible" or solving_goal == "Default":
        if solver.solve():
            model = solver.get_model()
            sol_dic = {}
            for v in model:
                sol_dic[abs(v)] = v  
            for var in variable_map:
                if sol_dic[variable_map[var]] > 0:
                    variable_map[var] = 1
                else:
                    variable_map[var] = 0
            return variable_map
        else:
            print("No solution exists.")
            return None
        
    elif solving_goal == "All":
        sol_list = []
        while solver.solve():
            model = solver.get_model()
            sol_dic = {}
            for v in model:
                sol_dic[abs(v)] = v          
            sol_list.append(sol_dic)
            block_clause = [-l for l in model]
            solver.add_clause(block_clause)
        solver.delete()
        print("Number of solutions by solving the SAT model: ", len(sol_list))
        return sol_list
    
    else:
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

    return content, variable_map

