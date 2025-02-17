import os, os.path
import operators as op


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
attacks.py

This module provides functions for performing MILP- and SAT-based attacks on cryptographic ciphers.

### Features:
1. **MILP-Based Attack**:
   - Formulates the attack as a Mixed Integer Linear Programming (MILP) problem.
   - Uses **Gurobi** to solve MILP models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.

2. **SAT-Based Attack**:
   - Formulates the attack as a Boolean Satisfiability Problem (SAT).
   - Uses **CryptoMiniSat** to solve SAT models.
   - Supports both optimizing for the best solution and exhaustive search for all possible solutions.

3. **Customization Options**:
   - Automates constraint generation for operations within ciphers.
   - Supports additional constraints (`add_cons`).
   - Allows specifying different model versions (`model_versions`) to control modelling the difference propagation behavior.
   - Enables defining the objective function (`model_weights`).
"""


def solve_milp(filename, solving_goal="optimize"): # Solve a MILP problem using Gurobi.
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return ""
    model = gp.read(filename)
    # model.setParam('OutputFlag', 0)   # no output 
    if solving_goal == "optimize":
        model.optimize()
        if model.status == gp.GRB.Status.OPTIMAL:
            print("Optimal Objective Value:", model.ObjVal)
            # for v in model.getVars(): 
                # print(f"{v.VarName} = {v.Xn}")
            return model.ObjVal
        else:
            print("No optimal solution found.")
    elif solving_goal == "all_solutions":
        model.Params.PoolSearchMode = 2   # Search for all solutions
        model.Params.PoolSolutions = 1000000  # Assuming you want a large number of solutions
        model.optimize()
        print("Number of solutions by solving the MILP model: ", model.SolCount)
        var_list = [v.VarName for v in model.getVars()]
        var_index_map = {v.VarName: idx for idx, v in enumerate(model.getVars())}
        sol_list = []
        # print("Solutions:")
        # print("var_list: ", var_list)
        for i in range(model.SolCount):
            model.Params.SolutionNumber = i  
            solution = [int(round(model.getVars()[var_index_map[var_name]].Xn)) for var_name in var_list]
            # print(solution)
            sol_list.append(solution)
        return var_list, sol_list, model


def create_numerical_cnf(cnf):
    # creating dictionary (variable -> string, numeric_id -> int)
    family_of_variables = ' '.join(cnf).replace('-', '')
    variables = sorted(set(family_of_variables.split()))
    variable2number = {variable: i + 1 for (i, variable) in enumerate(variables)}
    # creating numerical CNF
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


def solve_sat(filename, solving_goal="optimize"): # Solve a SAT problem using CryptoMiniSat.
    if pysat_import == False: 
        print("pysat module can't be loaded ... skipping test\n")
        return ""
    cnf = CNF(filename)
    solver = CryptoMinisat()
    solver.append_formula(cnf.clauses)
    if solving_goal == "optimize":
        if solver.solve():
            print("A solution exists.")
            # solution = solver.get_model()
            # print("solution: \n", solution)
            return True
        else:
            print("No solution exists.")
            return False
    elif solving_goal == "all_solutions":
        sol_list = []
        while solver.solve():
            model = solver.get_model()
            sol_list.append(model)
            block_clause = [-l for l in model]
            solver.add_clause(block_clause)
        solver.delete()
        print("Number of solutions by solving the SAT model: ", len(sol_list))
        # print("Solutions:")
        # for solution in sol_list:
        #     print(solution)
        return sol_list
    

def gen_round_constraints(cipher, model_type = "milp", rounds=None, states=None, layers=None, positions=None, model_versions={}, no_weights={}):
    """
    Generate constraints for a given cipher based on user-specified parameters.

    Args:
        cipher (object): The cipher instance.
        rounds (list[int, str] | None, optional): List of rounds to consider. Options: "inputs" and int (e.g., 1, 2, 3). Defaults to "inputs" and all rounds.
        states (list[str] | None, optional): List of states to consider. Options: "STATE", "KEY_STATE", "SUBKEYS". Defaults to all states.
        layers (dict | None, optional): Dictionary specifying the layers of each state. Options: int (e.g., 0, 1, 2). Defaults to all layers.
        positions (dict | None, optional): Dictionary mapping positions for constraints. Options: int (e.g., 0, 1, 2). Defaults to all positions.
        model_versions (dict | None, optional): Dictionary mapping constraint IDs to model versions.
        no_weights (dict | None, optional): Dictionary mapping constraint IDs to specify which constraints should be excluded from the objective function.

    Returns:
        tuple: 
            - **list[str]**: Generated constraints in string format.
            - **list[str]**: Objective function terms.
    """

    if states is None:
        states = [s for s in cipher.states]
    
    if rounds is None:
        rounds = {"inputs": "inputs"}
        rounds.update({s: list(range(1, cipher.states[s].nbr_rounds + 1)) for s in states})

    if layers is None:
        layers = {s: list(range(cipher.states[s].nbr_layers + 1)) for s in states}

    if positions is None:
        positions = {}
    if "inputs" in rounds:
        positions["inputs"] = list(range(len(cipher.inputs_constraints)))
    for s in states:
        if s not in positions:
            positions[s] = {}
        for r in rounds[s]:
            if r not in positions[s]:
                positions[s][r] = {}
            for l in layers[s]:
                positions[s][r][l] = list(range(len(cipher.states[s].constraints[r][l])))

    constraint, obj = [], []
    if r == "inputs": # constrains for linking the input and the first round 
        for p in positions["inputs"]:
            cons = cipher.inputs_constraints[p]
            model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0" 
            cons_gen = cons.generate_model(model_type=model_type, model_version = model_v, unroll=True)
            constraint += cons_gen
    for s in states:  
        for r in rounds[s]:
            for l in layers[s]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0"  
                    cons_gen = cons.generate_model(model_type=model_type, model_version = model_v, unroll=True)
                    constraint += cons_gen
                    if cons.ID not in no_weights and hasattr(cons, 'weight'): obj += [cons.weight]
    return constraint, obj


def gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", rounds=None, states=None, layers=None, positions=None, value=None, vars=None, bitwise=True): 
    """
    Generate additional constraints to the model based on specified parameters.

    Args:
        cipher (object): The cipher instance.
        model_type (str): The type of model to use. Options: "milp", "sat", "cp".
        cons_type (str): The type of constraint to generate. Options:
            - "EQUAL": Enforces the selected variable equals `value`.
            - "GREATER_EQUAL": Enforces the selected variable is at least `value`.
            - "LESS_EQUAL": Enforces the selected variable does not exceed `value`.
            - "SUM_EQUAL": Enforces the sum of selected variables equals `value`.
            - "SUM_GREATER_EQUAL": Enforces the sum of selected variables is at least `value`.
            - "SUM_LESS_EQUAL": Enforces the sum of selected variables does not exceed `value`.
        rounds (list[int] | None, optional): List of rounds to consider. Options: "inputs" and int (e.g., 1, 2, 3).  Defaults to None.
        states (list[str] | None, optional): List of states to consider. Options: "STATE", "KEY_STATE", "SUBKEYS".  Defaults to None.
        layers (dict | None, optional): Dictionary specifying the layers of each state. Options: int (e.g., 0, 1, 2). Defaults to None.
        positions (dict | None, optional): Dictionary mapping positions for constraints. Options: int (e.g., 0, 1, 2). Defaults to None.
        bitwise (bool, optional): If True, constraints are applied at the bit level. Defaults to True.
        vars (list[str] | None, optional): List of variable names to include in the constraints.
        value (int | None, optional): The target value for the constraint. Options: int(e.g., 0, 1, 2).

    Returns:
        list[str]: A list of generated constraints in string format.
    """

    add_cons, add_vars = [], []
    if (rounds is not None) and (states is not None) and (layers is not None) and (positions is not None):
        for r in rounds:
            for s in states:
                for l in layers[s]:
                    if bitwise: add_vars += [f"{cipher.states[s].vars[r][l][p].ID}_{j}" for p in positions[r][s][l] for j in range(cipher.states[s].vars[r][l][p].bitsize)]
                    else: add_vars += [cipher.states[s].vars[r][l][p].ID for p in positions[r][s][l]]
    if vars: add_vars += vars    
    if cons_type == "EQUAL":
        if model_type == "milp": add_cons += [f"{add_vars[i]} = {value}" for i in range(len(add_vars))]
        elif model_type == "sat" and value == 0: add_cons += [f"-{add_vars[i]}" for i in range(len(add_vars))]
        elif model_type == "sat" and value == 1: add_cons += [f"{add_vars[i]}" for i in range(len(add_vars))]
    elif cons_type == "GREATER_EQUAL":
        if model_type == "milp": add_cons += [f"{add_vars[i]} >= {value}" for i in range(len(add_vars))]
    elif cons_type == "LESS_EQUAL":
        if model_type == "milp": add_cons += [f"{add_vars[i]} <= {value}" for i in range(len(add_vars))]
    elif cons_type == "SUM_EQUAL":
        if model_type == "milp": add_cons += [' + '.join(f"{add_vars[i]}" for i in range(len(add_vars))) + f" = {value}"]
    elif cons_type == "SUM_GREATER_EQUAL":
        if model_type == "milp": add_cons += [' + '.join(f"{add_vars[i]}" for i in range(len(add_vars))) + f" >= {value}"]
        elif model_type == "sat" and value == 1: add_cons += [' '.join(f"{add_vars[i]}" for i in range(len(add_vars)))]
    elif cons_type == "SUM_LESS_EQUAL":
        if model_type == "milp": add_cons += [' + '.join(f"{add_vars[i]}" for i in range(len(add_vars))) + f" <= {value}"]
    return add_cons


def set_model_versions(cipher, version, rounds=None, states=None, layers=None, positions=None, consIDs=None):
    """
    Assigns a specified model_version to constraints in the cipher based on specified parameters.

    Args:
        cipher (object): The cipher instance.
        version (str): The model_version to apply.
        rounds (list[int, str] | None, optional): List of rounds to consider. Options: "inputs" and int (e.g., 1, 2, 3).  Defaults to None.
        states (list[str] | None, optional): List of states to consider. Options: "STATE", "KEY_STATE", "SUBKEYS".  Defaults to None.
        layers (dict | None, optional): Dictionary specifying the layers of each state. Options: int (e.g., 0, 1, 2). Defaults to None.
        positions (dict | None, optional): Dictionary mapping positions for constraints. Options: int (e.g., 0, 1, 2). Defaults to None.
        consIDs (list[str] | None, optional): List of constraint IDs to consider. If provided, only these constraints' model_version will be assigned.

    Returns:
        dict: A dictionary mapping constraint IDs to their assigned model version.
    """
    
    model_versions = {}

    if consIDs: # Assign the model version to a specific constraint ID.
        for consID in consIDs:
            model_versions[consID] = version
        return model_versions

    if rounds: # Handle input constraints when "inputs" or a number is specified for rounds.
        for r in rounds:
            if r == "inputs":
                model_versions["Input_Cons"] = version
                for p in positions["inputs"]:
                    model_versions[cipher.inputs_constraints[p].ID] = version
            elif isinstance(r, int): # Set model versions for constraints in a specific round
                for s in states: # Set model versions for constraints in a specific state
                    for l in layers[s]: # Set model versions for constraints in a specific layer
                        for p in positions[r][s][l]:
                            model_versions[f"{cipher.states[s].constraints[r][l][p].ID}"] = version    
    return model_versions


def attacks_milp_model(constraints=[], obj_fun=[], solving_goal="optimize", filename=""):
    """
    Generate and solve a MILP model based on the given constraints and objective function.

    Args:
        constraints (list[str]): A list of MILP constraints in string format.
        obj_fun (list[str]): A list of terms to be used in the objective function.
        solving_goal (str): The optimization goal. Possible values:
            - "optimize": Minimize the objective function.
            - "all_solutions": Find all feasible solutions.
        filename (str): The file path to save the MILP model.

    Returns:
        The result of the MILP solver, depending on the solving goal.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # === Step 1: Define the MILP Model Structure === #
    content = ""
    if solving_goal == "optimize": 
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
    if solving_goal == "optimize": content += " + ".join(obj_fun) + ' - obj = 0\n'
    
    # === Step 4: Declare Binary and Integer Variables === #
    if bin_vars: content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if in_vars: content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    
    # === Step 5: Write Model to File === #
    with open(filename, "w") as myfile:
        myfile.write(content)
    
    # === Step 6: Solve the MILP Model === #
    return solve_milp(filename, solving_goal)


def attacks_sat_model_obj(constraints=[], obj=0, obj_var=[], solving_goal="optimize", filename=""):       
    """
    Generate and solve a SAT model based on the given constraints and objective variables.

    Args:
        constraints (list[str]): A list of SAT constraints in string format.
        obj (int): The target value (lower bound) for the objective function.
        obj_var (list[str]): A list of variables representing the objective function.
        solving_goal (str): The optimization goal. Possible values:
            - "optimize": Find one feasible solutions.
            - "all_solutions": Find all feasible solutions.
        filename (str): The file path to save the SAT model.

    Returns:
        The result of the SAT solver.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # === Step 1: Generate The Constraint of "Weight Greater or Equal to the Given obj" Using the Sequential Encoding Method === #
    if obj == 0: obj_cons = [f'-{var}' for var in obj_var] 
    else:
        n = len(obj_var)
        dummy_var = [[f'obj_d_{i}_{j}' for j in range(obj)] for i in range(n - 1)]
        obj_cons = [f'-{obj_var[0]} {dummy_var[0][0]}']
        obj_cons += [f'-{dummy_var[0][j]}' for j in range(1, obj)]
        for i in range(1, n - 1):
            obj_cons += [f'-{obj_var[i]} {dummy_var[i][0]}']
            obj_cons += [f'-{dummy_var[i - 1][0]} {dummy_var[i][0]}']
            obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][j - 1]} {dummy_var[i][j]}' for j in range(1, obj)]
            obj_cons += [f'-{dummy_var[i - 1][j]} {dummy_var[i][j]}' for j in range(1, obj)]
            obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][obj - 1]}']
        obj_cons += [f'-{obj_var[n - 1]} -{dummy_var[n - 2][obj - 1]}']
    model_cons = constraints + obj_cons
    
    # === Step 2: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(model_cons)
    
    # === Step 3: Write the CNF Model to a File === #
    num_clause = len(model_cons)
    content = f"p cnf {num_var} {num_clause}\n"  
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    with open(filename, "w") as myfile:
        myfile.write(content)

    # === Step 4: Solve the SAT Model === #
    return solve_sat(filename, solving_goal=solving_goal)


def attacks_sat_model(constraints=[], obj=0, obj_var=[], solving_goal="optimize", filename=""):
    """
    Iteratively solve a SAT model while increasing the objective value until a valid solution is found.

    Args:
        constraints (list[str]): A list of SAT constraints in string format.
        obj (int): The initial objective value to test.
        obj_var (list[str]): A list of variables representing the objective function.
        filename (str): The file path to save the SAT model.

    Returns:
        int: The minimum objective value for which a valid solution exists.
    """
    flag = False
    while not flag:
        print("obj", obj)
        flag = attacks_sat_model_obj(constraints=constraints, obj=obj, obj_var=obj_var, solving_goal=solving_goal, filename=filename)
        obj += 1
    return obj-1


def set_model_noweight(): # TO DO
    """
    Specify constraints IDs that should not contribute to the objective function.

    Returns:
        dict: A list of constraints IDs.
    """
    noweight = []
    return noweight


def diff_attacks(r, cipher, model_versions={}, add_constraints="", model_type="milp", bitwise=True):
    """
    Perform differential attacks using either MILP or SAT models.
    
    Args:
        r (int): The number of rounds to analyze.
        cipher (Cipher): The cipher object.
        model_versions (dict): A dictionary mapping constraint IDs within the cipher to their specific model_version.
        add_constraints (str): Additional constraints to be added to the model.
        model_type (str): The type of model to use for the attack ('milp' or 'sat'). Defaults to 'milp'.
        
    Returns:
        result: The result of the MILP or SAT model.
    """

    # Validate model_type input
    if model_type not in ["milp", "sat"]:
        raise ValueError("Invalid model type specified. Choose 'milp' or 'sat'.")

    # Step 1. Generate constraints for the input, each round, and the objective function
    constraints, obj_fun = gen_round_constraints(cipher=cipher, model_type=model_type, model_versions=model_versions)
    
    # Step 2. Generate constraints ensuring that the input difference of the first round is not zero
    states = {s: cipher.states[s] for s in ["STATE", "KEY_STATE"] if s in cipher.states}
    constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", rounds=[1], states=states, layers = {s: [0] for s in states}, positions = {1: {s: {0: list(range(states[s].nbr_words))} for s in states}}, value=1, bitwise=bitwise)    
    
    # Step 3. Include user-defined additional constraints if provided
    constraints += add_constraints
    
    # Step 4. Execute the attack based on the specified model_type
    if model_type == "milp": 
        result = attacks_milp_model(constraints=constraints, obj_fun=obj_fun, filename=f"files/{r}_round_{cipher.name}_differential_trail_search_milp.lp")
    elif model_type == "sat": 
        result = attacks_sat_model(constraints=constraints, obj_var=list(sum(obj_fun, [])), filename=f"files/{r}_round_{cipher.name}_differential_trail_search_sat.cnf")
    
    return result