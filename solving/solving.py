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
   - Uses **Gurobi** to solve MILP models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.

2. **SAT-Based Attack**:
   - Formulates the attack as a Boolean Satisfiability Problem (SAT).
   - Uses **CryptoMiniSat** to solve SAT models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.
"""

def formulate_solutions(cipher, solitions):
    nbr_rounds_table = [cipher.states[s].nbr_rounds for s in cipher.states_display_order]
    nbr_layers_table = [cipher.states[s].nbr_layers for s in cipher.states_display_order]
    vars_table = [cipher.states[s].vars for s in cipher.states_display_order]
    for i in range(len(cipher.states)):
       for r in range(1,nbr_rounds_table[i]+1):
           for l in range(nbr_layers_table[i]+1):
               for w in range(len(vars_table[i][r][l])): 
                    if vars_table[i][r][l][w].ID in solitions:
                       vars_table[i][r][l][w].value = solitions[vars_table[i][r][l][w].ID]
                    elif vars_table[i][r][l][w].ID + "_0" in solitions:
                        vars_value = [solitions[var] for var in [vars_table[i][r][l][w].ID + "_" + str(n) for n in range(vars_table[i][r][l][w].bitsize)]]
                        vars_table[i][r][l][w].value = int(str(vars_value).replace("[","").replace("]","").replace(", ",""), 2) 
                    else:
                        vars_table[i][r][l][w].value = 0

def solve_milp(filename, solving_goal="optimize", solver="Gurobi", solver_params={}):
    """
    Solve a MILP model.
    
    Args:
        filename (str): The path to the MILP model file.
        solving_goal (str): The optimization goal. Possible values:
            - "optimize": Find the optimal solution.
            - "all_solutions": Find all feasible solutions.
        solver (str):
            - "Gurobi": Solve a MILP problem by using Gurobi
    
    Returns: 
        a tuple (sol_list, obj_list) containing solutions and objective value.
    """
    if solver == "Gurobi":
        if gurobipy_import == False: 
            print("gurobipy module can't be loaded ... skipping test\n")
            return None, None
        model = gp.read(filename)
        if solving_goal == "optimize":
            try:
                if "timeLimit" in solver_params:
                    model.Params.timeLimit = solver_params["timeLimit"]
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
            
        elif solving_goal == "all_solutions":
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
    
    elif solver == "SCIP":
        if not scip_import:
            print("PySCIPOpt module can't be loaded ... skipping SCIP test\n")
            return None, None
        
        model = Model()
        model.readProblem(filename)        
        if solving_goal == "optimize":
            model.optimize()
            if model.getStatus() == "optimal":
                sol_dic = {}
                for v in model.getVars():
                    sol_dic[str(v.name)] = int(round(model.getVal(v)))
                return sol_dic, model.getObjVal()
            else:
                return None, None
            
        elif solving_goal == "all_solutions":
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
        
    else:
        return None, None


def gen_milp_model(constraints=[], obj_fun=[], filename=""):
    """
    Generate a MILP model in standard .lp format, based on the given constraints and objective function.

    Args:
        constraints (list[str]): A list of MILP constraints in string format.
        obj_fun (list[str]): A list of terms to be used in the objective function.
    """

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
        with open(filename, "w") as myfile:
            myfile.write(content + "End\n")    
    
    return content


def solve_sat(filename, variable_map, solving_goal="optimize", solver="Default"):
    """
    Solve a SAT problem using CryptoMiniSat.
    
    Args:
        filename (str): Path to the CNF file.
        solving_goal (str): The optimization goal. Possible values:
            - "optimize": Find a feasible solution.
            - "all_solutions": Find all feasible solutions.
    
    Returns: 
        a list of all solutions found.
    """
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

    if solving_goal == "optimize":
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
        
    elif solving_goal == "all_solutions":
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
    

def create_numerical_cnf(cnf):
    """
    Convert a given CNF formula into numerical CNF format.
    
    Args:
        cnf (list[str]): A list of CNF constraints in string format.
    
    Returns:
        tuple: (number of variables, mapping of variables to numerical IDs, numerical CNF constraints)
    """
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


def gen_sat_model(constraints=[], filename=""):       
    """
    Generate and solve a SAT model based on the given constraints.

    Args:
        constraints (list[str]): A list of SAT constraints in string format.
    """

    # === Step 1: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints)
    
    # === Step 2: Generate the CNF Model === #
    num_clause = len(constraints)
    content = f"p cnf {num_var} {num_clause}\n"  
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    
    # === Step 3. Write the model into a file === #
    if filename:
        with open(filename, "w") as myfile:
            myfile.write(content)    

    return content, variable_map

