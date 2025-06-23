import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.linear_cryptanalysis as lin
import attacks.differential_cryptanalysis as dif
import solving.solving as solving
import visualisations.visualisations as vis 


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
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    if operator_name is None:
                        cons.model_version = cons.__class__.__name__ + "_" + version
                    elif operator_name is not None and operator_name in cons.__class__.__name__:
                        cons.model_version = cons.__class__.__name__ + "_" + version                            


# =================== Model Constraint and Objective Function Generation ===================
def gen_round_model_constraint_obj_fun(cipher, model_type, states=None, rounds=None, layers=None, positions=None, **sbox_model_params): # Generate constraints for a given cipher based on user-specified parameters.
    states, rounds, layers, positions = fill_states_rounds_layers_positions(cipher, states, rounds, layers, positions)
        
    constraint = []
    obj_fun = [[] for _ in range(cipher.states["STATE"].nbr_rounds)]
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for p in positions[s][r][l]:
                    cons = cipher.states[s].constraints[r][l][p]
                    if "Sbox" in cons.__class__.__name__:
                        cons_gen = cons.generate_model(model_type=model_type, **sbox_model_params)
                    else:
                        cons_gen = cons.generate_model(model_type=model_type)
                    constraint += cons_gen
                    if hasattr(cons, 'weight'): 
                        obj_fun[r-1] += cons.weight
    return constraint, obj_fun


# =================== Modeling and Solving for Attacks ===================
def gen_attacks_model(cipher, model_type, filename="", add_constraints=None, model_args=None):
    """
    Generate an MILP/SAT model for the attack.
    
    Args:
        cipher (Cipher): The cipher object.
        model_type (str): The type of model to use for the attack (e.g., 'milp', 'sat'). 
        add_constraints (list[str]): Additional constraints to be added to the model.
    Returns:
        result: the MILP or SAT model.
    """
    obj_fun_flag = model_args.get("obj_fun_flag", True)
    sbox_model_params = model_args.get("sbox_model_params", {})
    
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, model_type, **sbox_model_params) # Generate round constraints
    if "matsui_constraint" in model_args:
        Round = model_args.get("matsui_constraint").get("Round")
        best_obj = model_args.get("matsui_constraint").get("best_obj")
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        if model_type == "milp":
            cons_type = model_args["matsui_constraint"].get("matsui_milp_cons_type", "all")
            constraints += gen_matsui_constraints_milp(cipher, Round, best_obj, cons_type)
        elif model_type == "sat":
            GroupConstraintChoice = model_args["matsui_constraint"].get("GroupConstraintChoice", 1)
            GroupNumForChoice = model_args["matsui_constraint"].get("GroupNumForChoice", 1)
            obj_sat = model_args.get("obj_sat", 0)
            constraints += gen_matsui_constraints_sat(Round, best_obj, obj_sat, GroupConstraintChoice, GroupNumForChoice, obj_fun)
            obj_fun_flag = False          

    if obj_fun_flag: 
        obj_fun = [obj for row in obj_fun for obj in row]
        if model_type == "sat":  # Generate The Constraint of "objective Function Value Greater or Equal to the Given obj" Using the Sequential Encoding Method.
            constraints += gen_sequential_encoding_sat(hw_list=obj_fun, weight=model_args.get("obj_sat", 0))
    else:
        obj_fun = None
        
    constraints += (add_constraints or []) # Add additional constraints

    if model_type == "milp":
        return solving.gen_milp_model(constraints=constraints, obj_fun=obj_fun, filename=filename) # Generate a MILP model in standard format
    elif model_type == "sat":
        return solving.gen_sat_model(constraints=constraints, filename=filename) # Generate a SAT model in standard format
    else:
        raise Exception(str(cipher.__class__.__name__) + ": unknown model type '" + model_type + "'")
 

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


def modeling_solving_optimal_solution(cipher, model_type, filename, add_constraints=None, model_args=None, solving_args=None):
    time_start = time.time()
    if model_type == "milp":
        model = gen_attacks_model(cipher, "milp", filename, add_constraints=add_constraints, model_args=model_args)
        sol, obj = solving.solve_milp(filename, solving_args)
        
    elif model_type == "sat":
        obj_sat = model_args.get("obj_sat", 0)
        sol = {}
        while not sol:
            print("Current SAT objective value: ", obj_sat)
            model_args["obj_sat"] = obj_sat
            model, variable_map = gen_attacks_model(cipher, "sat", filename, add_constraints=add_constraints, model_args=model_args)
            sol = solving.solve_sat(filename, variable_map, solving_args)
            obj_sat += 1
        obj = obj_sat - 1
    if obj is not None:
        print(f"******** objective value of the optimal solution: {int(round(obj))} ********")
    else:
        print("******** optimal solution not found ********")
    time_end = time.time()
    solving_args["time"] = round(time_end - time_start, 2)
    return sol, obj


# =================== Additional Constraints and Advanced Strategies ===================
def gen_predefined_constraints(cipher, model_type, cons_type, cons_args=None): 
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
            - "INPUT_NOT_ZERO": Input variables sum >= 1 (at least one active).
            - "TRUNCATED_INPUT_NOT_ZERO": Truncated input variables sum >= 1.
        cons_args (dict): Parameters for constraint generation, including:
            - cons_value (int): Target value for the constraint.
            - cons_vars (list[str]): Additional variable names to include.
            - bitwise (bool): If True, expand variables by bit.

    Returns:
        list[str]: List of generated model constraint strings.

    """
    cons_args = cons_args or {}

    if cons_type in ["EQUAL", "SUM_EQUAL", "GREATER_EQUAL", "SUM_GREATER_EQUAL", "LESS_EQUAL", "SUM_LESS_EQUAL"]:
        cons_vars = cons_args.get("cons_vars", [])
        cons_value = cons_args.get("cons_value", 0)
        bitwise = cons_args.get("bitwise", True)
        cons_vars_name = []
        for var in cons_vars:
            if isinstance(var, str):
                cons_vars_name.append(var) 
            else:
                cons_vars_name.extend([f"{var.ID}_{j}" for j in range(var.bitsize)] if bitwise else [f"{var.ID}"])
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

    elif cons_type in ["INPUT_NOT_ZERO", "TRUNCATED_INPUT_NOT_ZERO"]:
        cons_args["cons_vars"] = cipher.states["STATE"].vars[1][0][:cipher.states["STATE"].nbr_words] + (cipher.states["KEY_STATE"].vars[1][0][:cipher.states["KEY_STATE"].nbr_words] if "KEY_STATE" in cipher.states else [])
        cons_args["cons_value"] = 1
        return gen_predefined_constraints(cipher, model_type=model_type, cons_type="SUM_GREATER_EQUAL", cons_args=cons_args)        
    

def gen_constraints_equal(model_type, cons_value, cons_vars):
    if model_type == "milp": 
        return [f"{cons_vars[i]} = {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0: 
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1: 
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]


def gen_constraints_sum_equal(model_type, cons_value, cons_vars):
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" = {cons_value}"]

def gen_constraints_less_equal(model_type, cons_value, cons_vars):
    if model_type == "milp": 
        return [f"{cons_vars[i]} <= {cons_value}" for i in range(len(cons_vars))]

def gen_constraints_sum_less_equal(model_type, cons_value, cons_vars):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]

def gen_constraints_greater_equal(model_type, cons_value, cons_vars): 
    if model_type == "milp": 
        return [f"{cons_vars[i]} >= {cons_value}" for i in range(len(cons_vars))]

def gen_constraints_sum_greater_equal(model_type, cons_value, cons_vars):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" >= {cons_value}"]
    elif model_type == "sat" and cons_value == 1: 
        return [' '.join(f"{cons_vars[i]}" for i in range(len(cons_vars)))]


# Adding Matsui's branch-and-bound constraints in differential and linear cryptanalysis
def gen_matsui_constraints_milp(cipher, Round, best_obj, cons_type="all"): # Generate Matsui's additional constraints for MILP models. Reference: Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy.
    states, rounds, layers, positions = fill_states_rounds_layers_positions(cipher)
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
            all_vars = [" + ".join(w_vars) + " - obj"]
            add_cons += gen_predefined_constraints(cipher, model_type="milp", cons_type="SUM_LESS_EQUAL", cons_args={"cons_vars": all_vars, "cons_value":-best_obj[i-1]})  
    return add_cons


def gen_matsui_constraints_sat(Round, best_obj, obj_sat, GroupConstraintChoice=1, GroupNumForChoice=1, obj_var=[]): # Generate Matsui's additional constraints for SAT models. Reference: Ling Sun, Wei Wang and Meiqin Wang. Accelerating the Search of Differential and Linear Characteristics with the SAT Method. https://github.com/SunLing134340/Accelerating_Automatic_Search
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