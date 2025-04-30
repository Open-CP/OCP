import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.linear_cryptanalysis as lin
import attacks.differential_cryptanalysis as dif
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

def set_model_versions(cipher, version, states=None, rounds=None, layers=None, positions=None, operator_name=None):
    """
    Assigns a specified model_version to constraints (operators) in the cipher based on specified parameters.

    Args:
        cipher (object): The cipher object.
        version (str): The model_version to apply (e.g., "truncated_diff").
        states (list[str]): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). 
        rounds (dict): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}).
        layers (dict): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}).
        positions (dict): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}).
    """
    
    states, rounds, layers, positions = gen_full_states_rounds_layers_positions(cipher, states, rounds, layers, positions)
    for s in states:
        for r in rounds[s]:            
            for l in layers[s][r]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    if (operator_name is None) or (operator_name is not None and cons.__class__.__name__ == operator_name):
                        cons.model_version = version


def set_model_noweight(): # TO DO
    """
    Specify constraints IDs that should not contribute to the objective function.

    Returns:
        dict: A list of constraints IDs.
    """
    noweight = []
    return noweight


def gen_full_states_rounds_layers_positions(cipher, states=None, rounds=None, layers=None, positions=None):
    """
    Generate mappings of states, rounds, layers, and constraint positions for a given cipher.

    Args:
        cipher (object): The cipher object.
        states (list[str]): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). Defaults to all states.
        rounds (dict): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}). Defaults to all rounds.
        layers (dict): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}). Defaults to all layers.
        positions (dict): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}). Defaults to all positions.
        
    Returns:
        states, rounds, layers, positions (dict)
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


def gen_round_constraints(cipher, model_type = "milp", states=None, rounds=None, layers=None, positions=None): # Generate constraints for a given cipher based on user-specified parameters.
    states, rounds, layers, positions = gen_full_states_rounds_layers_positions(cipher, states, rounds, layers, positions)
        
    constraint = []
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    cons_gen = cons.generate_model(model_type=model_type)
                    constraint += cons_gen
    return constraint


def gen_round_obj_fun(cipher, model_type = "milp", states=None, rounds=None, layers=None, positions=None, flatten=True): # Generate objective functions for a given cipher based on user-specified parameters.   
    states, rounds, layers, positions = gen_full_states_rounds_layers_positions(cipher, states, rounds, layers, positions)

    obj_fun = [[] for _ in range(cipher.states["STATE"].nbr_rounds)]
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    cons.generate_model(model_type=model_type)
                    if hasattr(cons, 'weight'): 
                        obj_fun[r-1] += cons.weight
    if flatten:
        return [obj for row in obj_fun for obj in row]
    return obj_fun


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
            - "INPUT_NOT_ZERO": Enforces the sum of input variables is at least 1.
            - "TRUNCATED_INPUT_NOT_ZERO": Enforces the sum of truncated input variables is at least 1.
        states (list[str]): List of states (e.g., ["STATE", "KEY_STATE", "SUBKEYS"]). 
        rounds (dict): Dictionary specifying rounds (e.g., {"STATE": [1, 2, 3]}).
        layers (dict): Dictionary specifying layers (e.g., {"STATE": {1: [0, 1, 2]}}).
        positions (dict): Dictionary specifying positions (e.g., {"STATE": {1: {0: [1, 2, 3]}}}).
        value (int): The target value for the constraint (e.g., 0, 1, 2).
        vars (list[str]): List of variable names to include in the constraints.
        bitwise (bool): If True, constraints are applied at the bit level. 
        
    Returns:
        list[str]: A list of generated constraints.
    """

    add_cons, add_vars = [], []
    if (rounds is not None) and (states is not None) and (layers is not None) and (positions is not None):
        for s in states:
            for r in rounds[s]:            
                for l in layers[s][r]:
                    if bitwise: 
                        add_vars += [f"{cipher.states[s].vars[r][l][p].ID}_{j}" for p in positions[s][r][l] for j in range(cipher.states[s].vars[r][l][p].bitsize)]
                    else: 
                        add_vars += [cipher.states[s].vars[r][l][p].ID for p in positions[s][r][l]]
    if vars is not None: 
        add_vars += vars    
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
    elif cons_type == "INPUT_NOT_ZERO" or "TRUNCATED_INPUT_NOT_ZERO":
        states = [s for s in ["STATE", "KEY_STATE"] if s in cipher.states]
        rounds = {s: [1] for s in states}
        layers = {s: {r: [0] for r in rounds[s]} for s in states}
        positions = {s: {r: {l: list(range(cipher.states[s].nbr_words)) for l in layers[s][r]} for r in rounds[s]} for s in states}
        bitwise = True if cons_type != "TRUNCATED_INPUT_NOT_ZERO" else False
        add_cons += gen_add_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", states=states, rounds=rounds, layers=layers, positions=positions, value=1, bitwise=bitwise)        
    return add_cons


def gen_sequential_encoding_sat(hw_list, weight, dummy_variables=None, greater_or_equal=False): # reference: https://github.com/Crypto-TII/claasp/blob/main/claasp/cipher_modules/models/sat/sat_model.py#L262
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
    n = len(hw_list)
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


def create_numerical_cnf(cnf): # Convert given CNF clauses into numerical CNF format.
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
    constraints = gen_round_constraints(cipher=cipher, model_type=model_type)
    obj_fun = gen_round_obj_fun(cipher, model_type = model_type)

    # Step 2. Add additional constraints
    constraints += add_constraints
    if model_type == "sat" and obj_fun: # Generate The Constraint of "objective Function Value Greater or Equal to the Given obj" Using the Sequential Encoding Method.
        constraints += gen_sequential_encoding_sat(hw_list=obj_fun, weight=obj_sat)  
        
    # Step 3. Generate a MILP/SAT/CP model in standard format
    if model_type == "milp":
        return solving.gen_milp_model(constraints=constraints, obj_fun=obj_fun, filename=filename), {}
    elif model_type == "sat":
        return solving.gen_sat_model(constraints=constraints, filename=filename)


def gen_matsui_constraints_milp(cipher, Round, best_obj, cons_type="all"): # Generate Matsui's additional constraints for MILP models. Reference: Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy.
    states, rounds, layers, positions = gen_full_states_rounds_layers_positions(cipher)
    add_cons = []
    for i in range(1, Round):
        if best_obj[i-1] > 0:
            w_vars = []
            if cons_type == "all" or cons_type == "upper":
                for r in range(i+1, Round+1):
                    for s in states: 
                        if r <= max(rounds[s]): 
                            for l in layers[s][r]:
                                for p in positions[s][r][l]:
                                    cons = cipher.states[s].constraints[r][l][p]
                                    cons.generate_model(model_type='milp')
                                    if hasattr(cons, 'weight'):
                                        w_vars += cons.weight
            elif cons_type == "all" or cons_type == "lower":
                for r in range(1, Round-i+1):
                    for s in states: 
                        if r <= max(rounds[s]): 
                            for l in layers[s][r]:
                                for p in positions[s][r][l]:
                                    cons = cipher.states[s].constraints[r][l][p]
                                    cons.generate_model(model_type='milp')
                                    if hasattr(cons, 'weight'):
                                        w_vars += cons.weight
            vars = [" + ".join(w_vars) + " - obj"]
            add_cons += gen_add_constraints(cipher, model_type="milp", cons_type="SUM_LESS_EQUAL", vars=vars, value=-best_obj[i-1])
      
    return add_cons


def gen_matsui_constraints_sat(obj_var, dummy_var, k, left, right, m): # Generate Matsui's additional constraints for MILP models. Reference: Ling Sun, Wei Wang and Meiqin Wang. Accelerating the Search of Differential and Linear Characteristics with the SAT Method. https://github.com/SunLing134340/Accelerating_Automatic_Search
    """
    Generate CNF clauses that constrain the number of active variables in the range [left, right] to be at most `m`, using sequential counter encoded auxiliary variables `dummy_var`.
    """
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


def diff_attacks(cipher, model_type="milp", goal="search_optimal_trail", add_constraints=None, obj_sat=0, solver="Default", show_mode=0):
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
    
    # Step 1. Define filename and ensure the directory exists, create if not
    filename = f"files/{cipher.name}_{goal}_{model_type}_model.{'lp' if model_type == 'milp' else 'cnf'}"
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path, exist_ok=True)

    # Step 2. Add constraints based on the goal. For searching for the optimal trail, the input difference should not be zero.
    if add_constraints is None:
        add_constraints = []
    if goal == "search_optimal_trail":
        add_constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="INPUT_NOT_ZERO")    
    elif goal == "search_optimal_truncated_trail":
        add_constraints += gen_add_constraints(cipher, model_type=model_type, cons_type="TRUNCATED_INPUT_NOT_ZERO") 

    # Step 3. Generate and Solve the MILP/SAT/CP model for the attack
    if model_type == "milp":
        model = gen_attacks_model(cipher, model_type=model_type, add_constraints=add_constraints, filename=filename)
        sol, obj = solving.solve_milp(filename, solver=solver)
        if sol == None: return None, None
        else: solving.formulate_solutions(cipher, sol)  

    elif model_type == "sat":
        sol = {}
        while not sol:
            print("Current SAT objective value: ", obj_sat)
            model, variable_map = gen_attacks_model(cipher, model_type=model_type, add_constraints=add_constraints, obj_sat=obj_sat, filename=filename)
            sol = solving.solve_sat(filename, variable_map, solver=solver)
            obj_sat += 1
        if sol == None: return None, None
        else: solving.formulate_solutions(cipher, sol)
        obj = obj_sat - 1 
    
    # Step 4. Display results
    print(f"******** objective value of the optimal solution: {int(round(obj))} ********")
    vis.print_trails(cipher, mode=show_mode)
    return sol, obj


def diff_attacks_matsui_milp(cipher, Round, BEST_OBJ, cons_type="all", solver="Default", show_mode=0):
    """
    Perform differential attacks using MILP models with Matsui's strategy.

    Args:
        cipher (object): The cipher object.
        Round (int): Number of rounds to model.
        BEST_OBJ (list[int]): List of known best objective values for each round.
        cons_type (str): Type of Matsui constraint to apply ("all", "upper", or "lower"). Default is "all".
        show_mode (int): Mode for visualizing the solution. Default is 0.

    Returns:
        int: Objective value of the solved MILP model.
    """
    
    # Step 1. Generate Matsui's Additional Constraints
    add_cons = gen_matsui_constraints_milp(cipher=cipher, Round=Round, BEST_OBJ=BEST_OBJ, cons_type=cons_type)

    # Step 2. Generate and Solve MILP model
    sol, obj = diff_attacks(cipher, model_type="milp", add_constraints=add_cons, solver=solver, show_mode=show_mode)

    return obj


def diff_attacks_matsui_sat(cipher, Round, obj_sat, GroupConstraintChoice, GroupNumForChoice, BEST_OBJ, solver="Default", show_mode=0):
    """
    Perform differential attacks using SAT model with Matsui constraints.

    Args:
        cipher (object): The cipher object.
        Round (int): Number of rounds.
        obj_sat (int): Target objective value for SAT model.
        GroupConstraintChoice (int): Choice of grouping method.
        GroupNumForChoice1 (int): Number of groups if GroupConstraintChoice == 1.
        BEST_OBJ (list[int]): List of known best objective values for each round.
        show_mode (int): Mode for visualizing solutions. Default is 0.

    Returns:
        bool: True if a valid solution is found, False otherwise.
    """

    # Step 1: Generate basic round constraints
    constraints = gen_round_constraints(cipher, model_type="sat")
    
    # Step 2: Enforce non-zero input difference
    constraints += gen_add_constraints(cipher, model_type="sat", cons_type="INPUT_NOT_ZERO") 

    # Generate The Constraint of "objective Function Value Greater or Equal to the Given obj" Using the Sequential Encoding Method
    obj_var = gen_round_obj_fun(cipher, model_type = "sat", flatten=False)
    Main_Vars = list([])
    for r in range(Round):
        for i in range(len(obj_var[Round - 1 - r])):
            Main_Vars += [obj_var[Round - 1 - r][i]]
    dummy_var = [[f'dummy_hw_{i}_{j}' for j in range(obj_sat)] for i in range(len(Main_Vars) - 1)]
    constraints += gen_sequential_encoding_sat(hw_list=Main_Vars, weight=obj_sat, dummy_variables=dummy_var)

    # Step 4: Generate Matsui condition and corresponding constraints
    MatsuiRoundIndex = []
    if GroupConstraintChoice == 1:
        for group in range(GroupNumForChoice):
            for round_offset in range(1, Round - group + 1):
                MatsuiRoundIndex.append([group, group + round_offset])
                
    for matsui_count in range(0, len(MatsuiRoundIndex)):
        StartingRound = MatsuiRoundIndex[matsui_count][0]
        EndingRound = MatsuiRoundIndex[matsui_count][1]
        PartialCardinalityCons = obj_sat - BEST_OBJ[StartingRound] - BEST_OBJ[Round - EndingRound]
        left = len(obj_var[0]) * StartingRound
        right = len(obj_var[0]) * EndingRound-1
        constraints += gen_matsui_constraints_sat(Main_Vars, dummy_var, obj_sat, left, right, PartialCardinalityCons)
    
    
    # Step 5: Generate and solve the SAT model
    filename = f"files/{cipher.name}_search_optimal_solution_sat_model_matsui.cnf"
    dir_path = os.path.dirname(filename)
    if not os.path.exists(dir_path): 
        os.makedirs(dir_path, exist_ok=True)
    model, variable_map = solving.gen_sat_model(constraints=constraints, filename=filename)
    sol = solving.solve_sat(filename, variable_map, solver=solver)
    if sol == None: 
        return False
    else: 
        solving.formulate_solutions(cipher, sol)       
        print(f"******** objective value of the optimal solution: {int(round(obj_sat))} ********")
        vis.print_trails(cipher, mode=show_mode)   
    return True