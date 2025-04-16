import os, os.path
import solving.solving as solving
import visualisations.visualisations as vis 


# ********************* ATTACKS ********************* # 
"""
This module provides functions for performing attacks on ciphers:
1. Automated generation of constraints for round operations of the cipher.
2. Customizable generation of additional constraints.
3. Configuration of model versions to control modeling behavior.
4. Definition of the objective function.
"""


def gen_round_constraints(cipher, model_type = "milp", states=None, rounds=None, layers=None, positions=None, no_weights=[]):
    """
    Generate constraints for a given cipher based on user-specified parameters.

    Args:
        cipher (object): The cipher object.
        states (list[str] | None, optional): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). Defaults to all states.
        rounds (dict | None, optional): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}). Defaults to all rounds.
        layers (dict | None, optional): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}). Defaults to all layers.
        positions (dict | None, optional): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}). Defaults to all positions.
        no_weights (list[str] | [], optional): List of constraint IDs to specify which constraints should be excluded from the objective function.

    Returns:
        tuple: 
            - **list[str]**: Generated constraints in string format.
            - **list[str]**: Objective function terms.
    """

    if states is None:
        states = [s for s in cipher.states]
    if rounds is None:
        rounds = {s: list(range(1, cipher.states[s].nbr_rounds + 1)) for s in states}
    if layers is None:
        layers = {s: {r: list(range(cipher.states[s].nbr_layers+1)) for r in rounds[s]} for s in states}
    if positions is None:
        positions = {s: {r: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in layers[s][r]} for r in rounds[s]} for s in states}

    constraint, obj = [], []
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    cons_gen = cons.generate_model(model_type=model_type)
                    constraint += cons_gen
                    if cons.ID not in no_weights and hasattr(cons, 'weight'): obj += cons.weight
    return constraint, obj


def gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", states=None, rounds=None, layers=None, positions=None, value=None, vars=None, bitwise=True): 
    """
    Generate additional constraints to the model based on specified parameters.

    Args:
        cipher (object): The cipher object.
        model_type (str): The type of model to use (e.g., "milp", "sat", "cp").
        cons_type (str): The type of constraint to generate. Options:
            - "EQUAL": Enforces the selected variable equals `value`.
            - "GREATER_EQUAL": Enforces the selected variable is at least `value`.
            - "LESS_EQUAL": Enforces the selected variable does not exceed `value`.
            - "SUM_EQUAL": Enforces the sum of selected variables equals `value`.
            - "SUM_GREATER_EQUAL": Enforces the sum of selected variables is at least `value`.
            - "SUM_LESS_EQUAL": Enforces the sum of selected variables does not exceed `value`.
        states (list[str] | None, optional): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). 
        rounds (dict | None, optional): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}).
        layers (dict | None, optional): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}).
        positions (dict | None, optional): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}).
        value (int | None, optional): The target value for the constraint (e.g., 0, 1, 2).
        vars (list[str] | None, optional): List of variable names to include in the constraints.
        bitwise (bool, optional): If True, constraints are applied at the bit level. 
        
    Returns:
        list[str]: A list of generated constraints in string format.
    """

    add_cons, add_vars = [], []
    if (rounds is not None) and (states is not None) and (layers is not None) and (positions is not None):
        for s in states:
            for r in rounds[s]:            
                for l in layers[s][r]:
                    if bitwise: add_vars += [f"{cipher.states[s].vars[r][l][p].ID}_{j}" for p in positions[s][r][l] for j in range(cipher.states[s].vars[r][l][p].bitsize)]
                    else: add_vars += [cipher.states[s].vars[r][l][p].ID for p in positions[s][r][l]]
    if vars is not None: add_vars += vars    
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


def set_model_versions(cipher, version, states=None, rounds=None, layers=None, positions=None):
    """
    Assigns a specified model_version to constraints in the cipher based on specified parameters.

    Args:
        cipher (object): The cipher object.
        version (str): The model_version to apply (e.g., "truncated_diff").
        states (list[str] | None, optional): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). 
        rounds (dict | None, optional): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}).
        layers (dict | None, optional): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}).
        positions (dict | None, optional): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}).
    """
    
    if (rounds is not None) and (states is not None) and (layers is not None) and (positions is not None):
        for s in states:
            for r in rounds[s]:            
                for l in layers[s][r]:
                    for p in positions[s][r][l]:
                        cipher.states[s].constraints[r][l][p].model_version = version


def set_model_noweight(): # TO DO
    """
    Specify constraints IDs that should not contribute to the objective function.

    Returns:
        dict: A list of constraints IDs.
    """
    noweight = []
    return noweight


def gen_attacks_model(cipher, model_type="milp", add_constraints=[], obj_sat=0, filename=""):
    """
    Generate MILP or SAT models for attacks.
    
    Args:
        cipher (Cipher): The cipher object.
        model_type (str): The type of model to use for the attack (e.g., 'milp', 'sat'). 
        add_constraints (list[str]): Additional constraints to be added to the model.
        
    Returns:
        result: the MILP or SAT model.
    """

    # Step 1. Generate round constraints and the objective function from the cipher
    constraints, obj_fun = gen_round_constraints(cipher=cipher, model_type=model_type)

    # Step 2. Generate specific constraints
    for cons in add_constraints:
        if cons == "input_not_zero" or cons == "truncated_input_not_zero": 
            states = [s for s in ["STATE", "KEY_STATE"] if s in cipher.states]
            rounds = {s: [1] for s in states}
            layers = {s: {r: [0] for r in rounds[s]} for s in states}
            positions = {s: {r: {l: list(range(cipher.states[s].nbr_words)) for l in layers[s][r]} for r in rounds[s]} for s in states}
            bitwise = True if cons != "truncated_input_not_zero" else False
            constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", states=states, rounds=rounds, layers=layers, positions=positions, value=1, bitwise=bitwise)    
        else:
            constraints += [cons]

    # Step 3. Generate a MILP/SAT/CP model in standard format
    if model_type == "milp":
        return solving.gen_milp_model(constraints=constraints, obj_fun=obj_fun, filename=filename), {}
    elif model_type == "sat":
        return solving.gen_sat_model(constraints=constraints, obj_var=obj_fun, obj=obj_sat, filename=filename)


def diff_attacks(cipher, model_type="milp", goal="search_optimal_trail", add_constraints=None, obj_sat=0, show_mode=0):
    """
    Perform differential attacks on a given cipher using specified model_type.

    Parameters:
        cipher (Cipher): The cipher object.
        model_type (str): Type of model to use (e.g., 'milp', 'sat').
        goal (str): Type of trail to search (e.g., 'search_optimal_trail', 'search_optimal_truncated_trail').
        add_constraints (list): Additional model constraints.
        obj_sat (int): Starting objective value for SAT model (e.g., 0, 1, 2).
        show_mode (int): Mode to display results (e.g., 0, 1, 2).

    Returns:
        tuple: Solution and its objective value, or (None, None) if no solution found.
    """
    if add_constraints is None:
        add_constraints = []

    # Define filename and ensure the directory exists, create if not
    filename = f"files/{cipher.name}_{goal}_{model_type}_model.{'lp' if model_type == 'milp' else 'cnf'}"
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path, exist_ok=True)

    # Adjust constraints based on the goal. For searching for the optimal trail, the input difference should not be zero.
    if goal == "search_optimal_trail":
        add_constraints += ["input_not_zero"]
    elif goal == "search_optimal_truncated_trail":
        add_constraints += ["truncated_input_not_zero"]

    if model_type == "milp":
        model = gen_attacks_model(cipher, model_type=model_type, add_constraints=add_constraints, filename=filename)
        sol, obj = solving.solve_milp(filename)
        if sol == None: return None, None
        else: solving.formulate_solutions(cipher, sol)  

    elif model_type == "sat":
        sol = {}
        while not sol:
            print("Current SAT objective value: ", obj_sat)
            model, variable_map = gen_attacks_model(cipher, model_type=model_type, add_constraints=add_constraints, obj_sat=obj_sat, filename=filename)
            sol = solving.solve_sat(filename, variable_map)
            obj_sat += 1
        if sol == None: return None, None
        else: solving.formulate_solutions(cipher, sol)
        obj = obj_sat - 1 
    
    print(f"******** objective value of the optimal solution: {int(round(obj))} ********")
    vis.print_trails(cipher, mode=show_mode)
    return sol, obj