import os
import sys
import time
try:
    from pysat.card import CardEnc
    from pysat.formula import IDPool
    vpool = IDPool(start_from=1000)
    pysat_import = True
except ImportError:
    print("pysat module can't be loaded \n")
    pysat_import = False
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.linear_cryptanalysis as lin
import attacks.differential_cryptanalysis as dif
import solving.solving as solving
import visualisations.visualisations as vis 
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files'))


# ********************* ATTACKS ********************* # 
"""
This module consists of four main functional components:
1. Model Behavior Configuration
   - Provides flexible specification and automatic filling of cipher states, rounds, layers, and positions.
   - Enables configuration of model versions to customize and control the modeling process.
2. Model Constraint and Objective Function Generation
   - Automates the generation of constraints and objective functions for each states, round, layer, or operation in the cipher.
   - Supports parameterized.
3. Modeling and Solving for Attacks
   - Offers unified interfaces for building and solving MILP/SAT models for attacks.
   - Supports parameterized.
4. Additional Constraints and Advanced Strategies
   - Supports injection of user-defined constraints and standard predefined constraints (e.g., input non-zero).
   - Implements advanced cryptanalysis techniques such as Matsui’s branch-and-bound strategy to enhance search efficiency.
"""

# =================== Model Behavior Configuration ===================
def fill_states_rounds_layers_positions(cipher, states=None, rounds=None, layers=None, positions=None):
    """
    Fill in states, rounds, layers, and positions to full coverage when the corresponding argument is None; otherwise, keep user-supplied values.

    Args:
        cipher (object): The cipher object.
        states (list[str]): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). If None, use all.
        rounds (dict): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}). If None, use all.
        layers (dict): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}). If None, use all.
        positions (dict): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}). If None, use all.

    Returns:
        tuple: (states, rounds, layers, positions)
    """

    if states is None:
        states = [s for s in cipher.states]
    if rounds is None:
        rounds = {s: list(range(1, cipher.states[s].nbr_rounds + 1)) for s in states}
    if layers is None:
        layers = {s: {r: list(range(cipher.states[s].nbr_layers+1)) for r in rounds[s]} for s in states}
    if positions is None:
        positions = {s: {r: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in layers[s][r]} for r in rounds[s]} for s in states}
    return states, rounds, layers, positions


def set_model_versions(cipher, version, states=None, rounds=None, layers=None, positions=None, operator_name=None): # Assigns a specified model_version to constraints (operators) in the cipher based on specified parameters.
    states, rounds, layers, positions = fill_states_rounds_layers_positions(cipher, states, rounds, layers, positions)
    for s in states:
        for r in rounds[s]:            
            for l in layers[s][r]:
                for cons in cipher.states[s].constraints[r][l]:
                    if operator_name is None: # Assign model_version to all operators in the cipher.
                        cons.model_version = cons.__class__.__name__ + "_" + version
                    elif operator_name is not None and operator_name in cons.__class__.__name__: #  Assign model_version to operators with a specific name.
                        cons.model_version = cons.__class__.__name__ + "_" + version                            


def configure_model_version_from_goal(cipher, goal): # Configure the model version for all operators in the cipher based on the attack goal.
    goal = goal.upper()

    if goal == 'DIFFERENTIAL_SBOXCOUNT':
        set_model_versions(cipher, "XORDIFF") # Set model_version = "XORDIFF" for all operators
        set_model_versions(cipher, "XORDIFF_A", operator_name="Sbox") # Set model_version = "XORDIFF_A" for all Sbox operators

    elif goal == 'DIFFERENTIALPATH_PROB' or  goal == "DIFFERENTIAL_PROB":
        set_model_versions(cipher, "XORDIFF") # Set model_version = "XORDIFF" for all operators
        set_model_versions(cipher, "XORDIFF_PR", operator_name="Sbox") # Set model_version = "XORDIFF_PR" for all Sbox operators

    elif goal == 'LINEAR_SBOXCOUNT':
        set_model_versions(cipher, "LINEAR") # Set model_version = "LINEAR" for all operators
        set_model_versions(cipher, "LINEAR_A", operator_name="Sbox") # Set model_version = "LINEAR_A" for all Sbox operators

    elif goal == 'LINEARPATH_PROB' or goal == "LINEARHULL_PROB":
        set_model_versions(cipher, "LINEAR") # Set model_version = "LINEAR" for all operators
        set_model_versions(cipher, "LINEAR_PR", operator_name="Sbox") # Set model_version = "LINEAR_PR" for all Sbox operators

    elif goal == "TRUNCATEDDIFF_SBOXCOUNT" in goal:   
        set_model_versions(cipher, "TRUNCATEDDIFF") # Set model_version = "TRUNCATEDDIFF" for all operators
        set_model_versions(cipher, "TRUNCATEDDIFF_A", operator_name="Sbox") # Set model_version = "TRUNCATEDDIFF_A" for all Sbox operators
    
    else:
        raise ValueError(f"Invalid goal: {goal}.")


# =================== Model Constraint and Objective Function Generation ===================
def gen_round_model_constraint_obj_fun(cipher, model_type, config_model): # Generate constraints for a given cipher based on user-specified parameters.
    states, rounds, layers, positions = fill_states_rounds_layers_positions(cipher, config_model.get("states"), config_model.get("rounds"), config_model.get("layers"), config_model.get("positions"))
        
    constraint = []
    obj_fun = [[] for _ in range(cipher.states["STATE"].nbr_rounds)]
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for cons in cipher.states[s].constraints[r][l]:
                    cons_class_name = cons.__class__.__name__
                    params = (config_model.get("model_params") or {}).get(cons_class_name, {}) # get operator-specific params if available
                    constraint += cons.generate_model(model_type=model_type, **params)                  
                    if hasattr(cons, 'weight'): 
                        obj_fun[r-1] += cons.weight
    return constraint, obj_fun


# =================== Modeling and Solving for Attacks ===================
def modeling_solving_model(cipher, goal, constraints, objective_target, config_model, config_solver):
    time_start = time.time()

    configure_model_version_from_goal(cipher, goal)
    model_type = config_model.get("model_type")     
    
    if objective_target == "OPTIMAL":
        if model_type == "milp":
            sol = modeling_solving_optimal_milp(cipher, constraints, config_model, config_solver)
        elif model_type == "sat":
            sol = modeling_solving_optimal_sat(cipher, constraints, config_model, config_solver) 
        else:
            ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("OPTIMAL STARTING FROM"):
        try:
            start_value = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'OPTIMAL STARTING FROM X'.")
        if model_type == "milp":
            sol =  modeling_solving_optimal_start_from_milp(cipher, constraints, config_model, config_solver, start_value)
        elif model_type == "sat":
            sol = modeling_solving_optimal_start_from_sat(cipher, constraints, config_model, config_solver, start_value)
        else:
            ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("AT MOST"):
        try:
            max_val = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT MOST X'.")
        if model_type == "milp":
            sol = modeling_solving_at_most_milp(cipher, constraints, config_model, config_solver, max_val)
        elif model_type == "sat":
            sol = modeling_solving_at_most_sat(cipher, constraints, config_model, config_solver, max_val)
        else:
            ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("EXACTLY"):
        try:
            exact_val = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'EXACTLY X'.")
        if model_type == "milp":
            sol = modeling_solving_exact_milp(cipher, constraints, config_model, config_solver, exact_val)
        elif model_type == "sat":
            sol = modeling_solving_exact_sat(cipher, constraints, config_model, config_solver, exact_val)
        else:
            ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    else:
        raise ValueError(f"Invalid objective_target: {objective_target}. Expected one of ['OPTIMAL', 'OPTIMAL STARTING FROM X', 'AT MOST X', 'EXACTLY X']")
    
    config_solver["time"] = round(time.time() - time_start, 2)
    print("=== Modeling and Solving Information ===")
    for key, value in {**config_model, **config_solver}.items():
        print(f"--- {key} ---: {value}")

    return sol

    
def modeling_solving_optimal_milp(cipher, add_constraints, config_model, config_solver): # Generate an MILP model and solve the optimal solution. 
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "milp", config_model) # Generate round constraints and objective function.

    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        cons_type = config_model["matsui_constraint"].get("matsui_milp_cons_type", "ALL")
        
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        
        constraints += gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type)
    
    constraints += (add_constraints or []) # Add additional constraints
    
    filename = config_model.get("filename", f"{cipher.name}_optimal_model.lp") # Get the filename from the model configuration or use a default name.
    model = solving.gen_milp_model(constraints, obj_fun, filename) # Generate an MILP model without the objective function in standard format
    
    sol = solving.solve_milp(filename, config_solver)

    if sol is not None:
        sol["status"] = "OPTIMAL"
        sol["obj_fun"] = obj_fun      

    return sol


def modeling_solving_optimal_start_from_milp(cipher, add_constraints, config_model, config_solver, start_value):      
    constraints = (add_constraints or []) # Add additional constraints
    constraints += gen_predefined_constraints("milp", "GREATER_EQUAL", start_value, ["obj"]) # Generate the constraint for the objective function value to be equal to start_value.      
    return modeling_solving_optimal_milp(cipher, constraints, config_model, config_solver)


def modeling_solving_satisfy_milp(cipher, add_constraints, config_model, config_solver):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "milp", config_model) # Generate round constraints and objective function.

    constraints += (add_constraints or []) # Add additional constraints
    
    filename = config_model.get("filename", f"{cipher.name}_satisfy_model.lp") # Get the filename from the model configuration or use a default name.
    model = solving.gen_milp_model(constraints, obj_fun, filename) # Generate an MILP model without the objective function in standard format
    
    config_solver["target"] = "SATISFIABLE"
    sol = solving.solve_milp(filename, config_solver)
    
    if sol is not None:
        sol["status"] = "SATISFIABLE"
        sol["obj_fun"] = obj_fun

    return sol


def modeling_solving_at_most_milp(cipher, add_constraints, config_model, config_solver, atmost_val):
    constraints = (add_constraints or []) # Add additional constraints

    constraints += gen_predefined_constraints("milp", "LESS_EQUAL", atmost_val, ["obj"]) # Generate the constraint for the objective function value to be equal to start_value.      
    
    return modeling_solving_satisfy_milp(cipher, add_constraints, config_model, config_solver)


def modeling_solving_exact_milp(cipher, add_constraints, config_model, config_solver, exact_val):
    constraints = (add_constraints or []) # Add additional constraints

    constraints += gen_predefined_constraints("milp", "EQUAL", exact_val, ["obj"]) # Generate the constraint for the objective function value to be equal to start_value.      
    
    return modeling_solving_satisfy_milp(cipher, add_constraints, config_model, config_solver)


def modeling_solving_optimal_sat(cipher, constraints, config_model, config_solver):
    return modeling_solving_optimal_start_from_sat(cipher, constraints, config_model, config_solver, 0) # Start from 0 to find the optimal solution.


def modeling_solving_optimal_start_from_sat(cipher, add_constraints, config_model, config_solver, start_value):
    obj_sat = start_value
    sol = {}
    while not sol:
        print("Current SAT objective value: ", obj_sat)
        sol = modeling_solving_at_most_sat(cipher, add_constraints, config_model, config_solver, obj_sat)
        obj_sat += 1
    sol["obj_fun_value"] = obj_sat - 1  # Add the objective value to the solution dictionary
    return sol


def modeling_solving_at_most_sat(cipher, add_constraints, config_model, config_solver, max_val):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model) # Generate round constraints and objective function.

    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        GroupConstraintChoice = config_model["matsui_constraint"].get("GroupConstraintChoice", 1)
        GroupNumForChoice = config_model["matsui_constraint"].get("GroupNumForChoice", 1)
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        constraints += gen_matsui_constraints_sat(Round, best_obj, max_val, obj_fun, GroupConstraintChoice, GroupNumForChoice)
    
    else:
        hw_list = [obj for row in obj_fun for obj in row]
        constraints += gen_sequential_encoding_sat(hw_list, max_val)
    
    constraints += (add_constraints or []) # Add additional constraints
    
    filename = config_model.get("filename", f"{cipher.name}_satisfy_at_most_model.cnf") # Get the filename from the model configuration or use a default name.
    
    model = solving.gen_sat_model(constraints=constraints, filename=filename) # Generate an SAT model in standard format
    
    sol = solving.solve_sat(filename, model["variable_map"], config_solver)

    if sol is not None:
        sol["obj_fun_value"] = max_val
    
    return sol  
     

def gen_sequential_encoding_sat(hw_list, weight, dummy_variables=None, greater_or_equal=False): # reference: https://github.com/Crypto-TII/claasp/blob/main/claasp/cipher_modules/models/sat/sat_model.py#L262
    n = len(hw_list)
    # === Special case: require all variables to be False ===
    if (not greater_or_equal and weight == 0) or (greater_or_equal and weight == n):
        constraints = [f'-{var}' for var in hw_list]
        return constraints
    # === At-least-k is transformed into at-most-(n-k) ===
    if greater_or_equal:
        weight = n - weight
        minus = ''
    else:
        minus = '-'
    if dummy_variables is None:
        dummy_variables = [[f'dummy_hw_{i}_{j}' for j in range(weight)] for i in range(n - 1)]
    constraints = [f'{minus}{hw_list[0]} {dummy_variables[0][0]}']
    constraints.extend([f'-{dummy_variables[0][j]}' for j in range(1, weight)])
    for i in range(1, n - 1):
        constraints.append(f'{minus}{hw_list[i]} {dummy_variables[i][0]}')
        constraints.append(f'-{dummy_variables[i - 1][0]} {dummy_variables[i][0]}')
        constraints.extend([f'{minus}{hw_list[i]} -{dummy_variables[i - 1][j - 1]} {dummy_variables[i][j]}'
                            for j in range(1, weight)])
        constraints.extend([f'-{dummy_variables[i - 1][j]} {dummy_variables[i][j]}'
                            for j in range(1, weight)])
        constraints.append(f'{minus}{hw_list[i]} -{dummy_variables[i - 1][weight - 1]}')
    constraints.append(f'{minus}{hw_list[n - 1]} -{dummy_variables[n - 2][weight - 1]}')
    return constraints


def modeling_solving_exact_sat(cipher, constraints, config_model, config_solver, exact_val): # TO DO
    pass
    

# =================== Additional Constraints and Advanced Strategies ===================
def gen_predefined_constraints(model_type, cons_type, cons_value, cons_vars, bitwise=True): 
    """
    Generate commonly used, predefined model constraints based on type and parameters.

    Args:
        cons_type (str): The constraint type, must be one of the predefined types:
            - "EQUAL": All selected variables equal a target value.
            - "GREATER_EQUAL": All selected variables >= target value.
            - "LESS_EQUAL": All selected variables <= target value.
            - "SUM_EQUAL": Sum of selected variables == target value.
            - "SUM_GREATER_EQUAL": Sum of selected variables >= target value.
            - "SUM_LESS_EQUAL": Sum of selected variables <= target value.
        cons_args (dict): Parameters for constraint generation, including:
            - cons_value (int): Target value for the constraint.
            - cons_vars (list[str]): Additional variable names to include.
            - bitwise (bool): If True, expand variables by bit.

    Returns:
        list[str]: List of generated model constraint strings.

    """
    if cons_type in ["EQUAL", "SUM_EQUAL", "GREATER_EQUAL", "SUM_GREATER_EQUAL", "LESS_EQUAL", "SUM_LESS_EQUAL"]:
        cons_vars_name = []
        for var in cons_vars:
            if isinstance(var, str):
                cons_vars_name.append(var) 
            else:
                if bitwise and var.bitsize > 1:
                    cons_vars_name.extend([f"{var.ID}_{j}" for j in range(var.bitsize)])
                else:
                    cons_vars_name.append(var.ID)
        if cons_type == "EQUAL":
            return gen_constraints_equal(model_type, cons_value, cons_vars_name)
        elif cons_type == "GREATER_EQUAL":
            return gen_constraints_greater_equal(model_type, cons_value, cons_vars_name)
        elif cons_type == "LESS_EQUAL":
            return gen_constraints_less_equal(model_type, cons_value, cons_vars_name)
        elif cons_type == "SUM_EQUAL":
            return gen_constraints_sum_equal(model_type, cons_value, cons_vars_name)
        elif cons_type == "SUM_GREATER_EQUAL":
            return gen_constraints_sum_greater_equal(model_type, cons_value, cons_vars_name)
        elif cons_type == "SUM_LESS_EQUAL":
            return gen_constraints_sum_less_equal(model_type, cons_value, cons_vars_name)
    

def gen_constraints_equal(model_type, cons_value, cons_vars):
    if model_type == "milp": 
        return [f"{cons_vars[i]} = {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0: 
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1: 
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for EQUAL constraint.")


def gen_constraints_sum_equal(model_type, cons_value, cons_vars, encoding=1):
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" = {cons_value}"]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and pysat_import:
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        cnf = CardEnc.equals(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)   
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_EQUAL constraint.")

def gen_constraints_less_equal(model_type, cons_value, cons_vars):
    if model_type == "milp": 
        return [f"{cons_vars[i]} <= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for LESS_EQUAL constraint.")

def gen_constraints_sum_less_equal(model_type, cons_value, cons_vars, encoding=1):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and pysat_import:
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        cnf = CardEnc.atmost(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)   
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_LESS_EQUAL constraint.")   
    
def gen_constraints_greater_equal(model_type, cons_value, cons_vars): 
    if model_type == "milp": 
        return [f"{cons_vars[i]} >= {cons_value}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for GREATER_EQUAL constraint.")

def gen_constraints_sum_greater_equal(model_type, cons_value, cons_vars, encoding=1):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" >= {cons_value}"]
    elif model_type == "sat" and cons_value == 1: 
        return [' '.join(f"{cons_vars[i]}" for i in range(len(cons_vars)))]
    elif model_type == "sat" and pysat_import:
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        cnf = CardEnc.atleast(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)   
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_GREATER_EQUAL constraint.")


# Adding Matsui's branch-and-bound constraints in differential and linear cryptanalysis
def gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type="ALL"): # Generate Matsui's additional constraints for MILP models. Reference: Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy.
    add_cons = []
    for i in range(1, Round):
        if best_obj[i-1] > 0:
            if cons_type == "ALL" or cons_type == "UPPER":
                w_vars = [var for r in range(i + 1, Round + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="LESS_EQUAL", cons_value=-best_obj[i-1], cons_vars=all_vars)  
            if cons_type == "ALL" or cons_type == "LOWER":
                w_vars = [var for r in range(1, Round - i + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="LESS_EQUAL", cons_value=-best_obj[i-1], cons_vars=all_vars)                        
    return add_cons


def gen_matsui_constraints_sat(Round, best_obj, obj_sat, obj_var, GroupConstraintChoice=1, GroupNumForChoice=1): # Generate Matsui's additional constraints for SAT models. Reference: Ling Sun, Wei Wang and Meiqin Wang. Accelerating the Search of Differential and Linear Characteristics with the SAT Method. https://github.com/SunLing134340/Accelerating_Automatic_Search
    if len(best_obj) == Round-1:
        best_obj = [0] + best_obj
    Main_Vars = list([])
    for r in range(Round):
        for i in range(len(obj_var[Round - 1 - r])):
            Main_Vars += [obj_var[Round - 1 - r][i]]
    dummy_var = [[f'dummy_hw_{i}_{j}' for j in range(obj_sat)] for i in range(len(Main_Vars) - 1)]
    constraints = gen_sequential_encoding_sat(hw_list=Main_Vars, weight=obj_sat, dummy_variables=dummy_var) #  # Generate the constraint of "objective Function Value Greater or Equal to the Given obj" Using the Sequential Encoding Method
    
    MatsuiRoundIndex = []
    if GroupConstraintChoice == 1:
        for group in range(GroupNumForChoice):
            for round_offset in range(1, Round - group + 1):
                MatsuiRoundIndex.append([group, group + round_offset])
    
    for matsui_count in range(0, len(MatsuiRoundIndex)):
        StartingRound = MatsuiRoundIndex[matsui_count][0]
        EndingRound = MatsuiRoundIndex[matsui_count][1]
        PartialCardinalityCons = obj_sat - best_obj[StartingRound] - best_obj[Round-EndingRound]
        left = len(obj_var[0]) * StartingRound
        right = len(obj_var[0]) * EndingRound-1
        constraints += gen_matsui_partial_cardinality_sat(Main_Vars, dummy_var, obj_sat, left, right, PartialCardinalityCons)
    return constraints


def gen_matsui_partial_cardinality_sat(obj_var, dummy_var, k, left, right, m): # Generate CNF clauses that constrain the number of active variables in the range [left, right] to be at most `m`, using sequential counter encoded auxiliary variables `dummy_var`.
    n = len(obj_var)
    add_cons = []

    if m > 0:
        if left == 0 and right < n - 1:
            for i in range(1, right + 1):
                add_cons.append(f"-{obj_var[i]} -{dummy_var[i - 1][m - 1]}")

        if left > 0 and right == n - 1:
            for i in range(0, k - m):
                add_cons.append(f"{dummy_var[left - 1][i]} -{dummy_var[right - 1][i + m]}")
            for i in range(0, k - m + 1):
                add_cons.append(f"{dummy_var[left - 1][i]} -{obj_var[right]} -{dummy_var[right - 1][i + m - 1]}")

        if left > 0 and right < n - 1:
            for i in range(0, k - m):
                add_cons.append(f"{dummy_var[left - 1][i]} -{dummy_var[right][i + m]}")

    elif m == 0:
        for i in range(left, right + 1):
            add_cons.append(f"-{obj_var[i]}")

    return add_cons