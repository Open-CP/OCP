import os, os.path


try:
    import gurobipy as gp
    gurobipy_import = True
except ImportError:
    print("gurobipy module can't be loaded \n")
    gurobipy_import = False
    pass

try:
    from pysat.solvers import CryptoMinisat
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


def solve_milp(filename, solving_goal="optimize", solver="Gurobi"):
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
        If solving_goal = "optimize": Returns the optimal objective value.
        If solving_goal = "all_solutions": Returns a tuple (var_list, sol_list, model) containing variable names, solutions, and the Gurobi model object.
    """
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return None
    model = gp.read(filename)
    if solving_goal == "optimize":
        model.optimize()
        if model.status == gp.GRB.Status.OPTIMAL:
            sol_dic = {}
            for v in model.getVars():
                sol_dic[str(v.VarName)] = v.Xn            
            return [sol_dic], [model.ObjVal]
        else:
            print("No optimal solution found.")
            return None, None
        
    elif solving_goal == "all_solutions":
        model.Params.PoolSearchMode = 2   # Enable searching for multiple solutions
        model.Params.PoolSolutions = 1000000  # Set the maximum limit for the number of solutions
        model.optimize()
        print("Number of solutions by solving the MILP model: ", model.SolCount)
        sol_list = []
        obj_list = []
        for i in range(model.SolCount):
            sol_dic = {}
            model.Params.SolutionNumber = i  
            for v in model.getVars():
                sol_dic[str(v.VarName)] = v.Xn   
            sol_list.append(sol_dic)
            obj_list.append(model.ObjVal)
        return sol_list, obj_list

    else:
        return None, None


def gen_milp_model(constraints=[], obj_fun=[], filename="milp_model.lp"):
    """
    Generate a MILP model in standard .lp format, based on the given constraints and objective function.

    Args:
        constraints (list[str]): A list of MILP constraints in string format.
        obj_fun (list[str]): A list of terms to be used in the objective function.
        filename (str): The file path to save the MILP model in .lp format.   
    """

    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):  # Ensure directory exists 
        os.makedirs(dir_path, exist_ok=True)
    
    
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
    
    
    # === Step 3: Declare Binary and Integer Variables === #
    if bin_vars: 
        content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if in_vars: 
        content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    

    # === Step 4: Define the Objective Function === #
    if obj_fun:
        content += " + ".join(obj_fun) + ' - obj = 0\n'


    # === Step 5: Write Model to File === #
    with open(filename, "w") as myfile:
        myfile.write(content)


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


def solve_sat(filename, solving_goal="optimize", solver="CryptoMiniSat"):
    """
    Solve a SAT problem using CryptoMiniSat.
    
    Args:
        filename (str): Path to the CNF file.
        solving_goal (str): The optimization goal. Possible values:
            - "optimize": Find a feasible solution.
            - "all_solutions": Find all feasible solutions.
    
    Returns:
        bool or list: If "optimize", returns True if a solution exists, False otherwise.
                      If "all_solutions", returns a list of all solutions found.
    """
    if not pysat_import:
        print("pysat module can't be loaded ... skipping test\n")
        return ""
    
    cnf = CNF(filename)
    solver = CryptoMinisat()
    solver.append_formula(cnf.clauses)
    
    if solving_goal == "optimize":
        if solver.solve():
            model = solver.get_model()
            sol_dic = {}
            for v in model:
                sol_dic[abs(v)] = v          
            return [sol_dic]
        else:
            print("No solution exists.")
            return []
        
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
    

def gen_sat_model(constraints=[], obj=0, obj_var=[], filename=""):       
    """
    Generate and solve a SAT model based on the given constraints and objective variables.

    Args:
        constraints (list[str]): A list of SAT constraints in string format.
        obj (int): The target value (lower bound) for the objective function.
        obj_var (list[str]): A list of variables representing the objective function.
        filename (str): The file path to save the SAT model.
    """

    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path):  # Ensure directory exists 
        os.makedirs(dir_path, exist_ok=True)

    model_cons = constraints
    
    # === Step 1: Generate The Constraint of "objective Function Value Greater or Equal to the Given obj" Using the Sequential Encoding Method === #
    if obj_var:
        if obj == 0: 
            obj_cons = [f'-{var}' for var in obj_var] 
        else:
            n = len(obj_var)
            dummy_var = [[f'obj_{i}_{j}' for j in range(obj)] for i in range(n - 1)]
            obj_cons = [f'-{obj_var[0]} {dummy_var[0][0]}']
            obj_cons += [f'-{dummy_var[0][j]}' for j in range(1, obj)]
            for i in range(1, n - 1):
                obj_cons += [f'-{obj_var[i]} {dummy_var[i][0]}']
                obj_cons += [f'-{dummy_var[i - 1][0]} {dummy_var[i][0]}']
                obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][j - 1]} {dummy_var[i][j]}' for j in range(1, obj)]
                obj_cons += [f'-{dummy_var[i - 1][j]} {dummy_var[i][j]}' for j in range(1, obj)]
                obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][obj - 1]}']
            obj_cons += [f'-{obj_var[n - 1]} -{dummy_var[n - 2][obj - 1]}']
        model_cons += obj_cons
    
    # === Step 2: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(model_cons)
    
    # === Step 3: Write the CNF Model to a File === #
    num_clause = len(model_cons)
    content = f"p cnf {num_var} {num_clause}\n"  
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    with open(filename, "w") as myfile:
        myfile.write(content)
    
    return num_var, variable_map, numerical_cnf

