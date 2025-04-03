import os, os.path
import solving.solving as solving
import visualisations.visualisations as vis 


"""
This module provides functions for performing attacks on ciphers.

### Features:
1.
3. **Customization Options**:
   - Automates constraint generation for operations within ciphers.
   - Supports additional constraints (`add_cons`).
   - Allows specifying different model versions (`model_versions`) to control modelling the difference propagation behavior.
   - Enables defining the objective function (`model_weights`).
"""


def gen_round_constraints(cipher, model_type = "milp", rounds=None, states=None, layers=None, positions=None, no_weights={}):
    """
    Generate constraints for a given cipher based on user-specified parameters.

    Args:
        cipher (object): The cipher instance.
        rounds (list[int, str] | None, optional): List of rounds to consider. Options: "inputs" and int (e.g., 1, 2, 3). Defaults to "inputs" and all rounds.
        states (list[str] | None, optional): List of states to consider. Options: "STATE", "KEY_STATE", "SUBKEYS". Defaults to all states.
        layers (dict | None, optional): Dictionary specifying the layers of each state. Options: int (e.g., 0, 1, 2). Defaults to all layers.
        positions (dict | None, optional): Dictionary mapping positions for constraints. Options: int (e.g., 0, 1, 2). Defaults to all positions.
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
    if "inputs" in rounds: # constrains for linking the input and the first round 
        for p in positions["inputs"]:
            cons = cipher.inputs_constraints[p]
            cons_gen = cons.generate_model(model_type=model_type, unroll=True)
            constraint += cons_gen
    for s in states:  
        for r in rounds[s]:
            for l in layers[s]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    cons_gen = cons.generate_model(model_type=model_type, unroll=True)
                    constraint += cons_gen
                    if cons.ID not in no_weights and hasattr(cons, 'weight'): obj += cons.weight
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


def set_model_versions(cipher, version, states=None, rounds=None, layers=None, positions=None):
    """
    Assigns a specified model_version to constraints in the cipher based on specified parameters.

    Args:
        cipher (object): The cipher object.
        version (str): The model_version to apply.
        states (list[str] | None, optional): List of states. Options: "inputs", "STATE", "KEY_STATE", "SUBKEYS".
        rounds (dict | None, optional): Dictionary specifying the rounds of each state. 
        layers (dict | None, optional): Dictionary specifying the layers of each round of each state. 
        positions (dict | None, optional): Dictionary specifying positions of each layer of each round of each state.
    """
    
    if states is not None: 
        for s in states:
            if s == "inputs": # in the input
                for p in positions["inputs"]:
                    cipher.inputs_constraints[p].model_version = version
            else: # in a specific state
                for r in rounds[s]: # in a specific round
                    for l in layers[s][r]: # in a specific layer
                        for p in positions[s][r][l]: # in a specific position
                            cipher.states[s].constraints[r][l][p].model_version = version    


def set_model_noweight(): # TO DO
    """
    Specify constraints IDs that should not contribute to the objective function.

    Returns:
        dict: A list of constraints IDs.
    """
    noweight = []
    return noweight


def attacks(cipher, add_constraints=[], model_type="milp", obj_sat=0, filename=""):
    """
    Perform attacks by using MILP or SAT models.
    
    Args:
        cipher (Cipher): The cipher object.
        add_constraints (list[str]): Additional constraints to be added to the model.
        model_type (str): The type of model to use for the attack ('milp' or 'sat'). Defaults to 'milp'.
        
    Returns:
        result: the MILP or SAT model.
    """

    # Validate model_type input
    if model_type not in ["milp", "sat"]:
        raise ValueError("Invalid model type specified. Choose 'milp' or 'sat'.")

    # Step 1. Generate constraints and the objective function from the cipher
    constraints, obj_fun = gen_round_constraints(cipher=cipher, model_type=model_type)

    # Step 2. Generate specific constraints
    for cons in add_constraints:
        if cons == "input_not_zero": # the input of the first round is not zero
            states = {s: cipher.states[s] for s in ["STATE", "KEY_STATE"] if s in cipher.states}
            constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", rounds=[1], states=states, layers = {s: [0] for s in states}, positions = {1: {s: {0: list(range(states[s].nbr_words))} for s in states}}, value=1)    
        elif cons == "truncated_input_not_zero": # the truncated input of the first round is not zero
            states = {s: cipher.states[s] for s in ["STATE", "KEY_STATE"] if s in cipher.states}
            constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", rounds=[1], states=states, layers = {s: [0] for s in states}, positions = {1: {s: {0: list(range(states[s].nbr_words))} for s in states}}, value=1, bitwise=False)    
        else:
            constraints += [cons]

    # Step 3. Generate a MILP/SAT/CP model in standard format
    if model_type == "milp":
        return solving.gen_milp_model(constraints=constraints, obj_fun=obj_fun, filename=filename), {}
    elif model_type == "sat":
        return solving.gen_sat_model(constraints=constraints, obj_var=obj_fun, obj=obj_sat, filename=filename)


def diff_attacks(cipher, add_constraints=[], model_type="milp", goal="search_optimal_trail", obj_sat=0, show_mode=0):
    """
    goal:
        - "search_optimal_trail"
        - "search_optimal_truncated_trail"
    """
    if model_type == "milp":
        filename = f"files/{cipher.name}_{goal}_milp_model.lp"
    elif model_type == "sat":
        filename = f"files/{cipher.name}_{goal}_sat_model.cnf"
    dir_path = os.path.dirname(filename)
    if dir_path and not os.path.exists(dir_path): 
        os.makedirs(dir_path, exist_ok=True)

    if goal == "search_optimal_trail":
        add_constraints += ["input_not_zero"]
    elif goal == "search_optimal_truncated_trail":
        add_constraints += ["truncated_input_not_zero"]

    if model_type == "milp":
        model = attacks(cipher, add_constraints=add_constraints, model_type=model_type, filename=filename)
        sol, obj = solving.solve_milp(filename)
        if sol==None: return None, None
        solving.formulate_solutions(cipher, sol)  

    elif model_type == "sat":
        sol = {}
        while not sol:
            model, variable_map = attacks(cipher, add_constraints=add_constraints, model_type=model_type, obj_sat=obj_sat, filename=filename)
            sol = solving.solve_sat(filename, variable_map)
            obj_sat += 1
        if sol==None: return None, None
        solving.formulate_solutions(cipher, sol)
        obj = obj_sat-1 
    print(f"******** objective value of the optimal solution: {int(round(obj))} ********")
    vis.print_trails(cipher, mode=show_mode)
    return sol, obj