import os
import sys
import time
import heapq
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
def modeling_and_solving(cipher, goal, objective_target, constraints, config_model, config_solver): # Main interface for modeling and solving based on the given objective_target
    time_start = time.time()

    configure_model_version_from_goal(cipher, goal)
    model_type = config_model.get("model_type")

    if objective_target == "OPTIMAL":
        if model_type == "milp":
            sol = modeling_solving_optimal_milp(cipher, constraints, config_model, config_solver)
        elif model_type == "sat":
            sol = modeling_solving_optimal_sat(cipher, goal, constraints, config_model, config_solver) 
        else:
            raise ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("OPTIMAL STARTING FROM"):
        try:
            start_value = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'OPTIMAL STARTING FROM X'.")
        if model_type == "milp":
            sol =  modeling_solving_optimal_start_from_milp(cipher, constraints, config_model, config_solver, start_value)
        elif model_type == "sat":
            sol = modeling_solving_optimal_start_from_sat(cipher, goal, constraints, config_model, config_solver, start_value)
        else:
            raise ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("AT MOST"):
        try:
            max_val = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT MOST X'.")
        if model_type == "milp":
            sol = modeling_solving_at_most_milp(cipher, constraints, config_model, config_solver, max_val)
        elif model_type == "sat":
            sol = modeling_solving_at_most_sat(cipher, goal, constraints, config_model, config_solver, max_val)
        else:
            raise ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    elif objective_target.startswith("EXACTLY"):
        try:
            exact_val = int(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'EXACTLY X'.")
        if model_type == "milp":
            sol = modeling_solving_exactly_milp(cipher, constraints, config_model, config_solver, exact_val)
        elif model_type == "sat":
            sol = modeling_solving_exactly_sat(cipher, goal, constraints, config_model, config_solver, exact_val)
        else:
            raise ValueError(f"Invalid objective_target: {objective_target}, model_type = {model_type}.")

    else:
        raise ValueError(f"Invalid objective_target: {objective_target}. Expected one of ['OPTIMAL', 'OPTIMAL STARTING FROM X', 'AT MOST X', 'EXACTLY X']")
    
    config_solver["time"] = round(time.time() - time_start, 2)
    print("=== Modeling and Solving Information ===")
    for key, value in {**config_model, **config_solver}.items():
        print(f"--- {key} ---: {value}")

    return sol


# =================== Modeling and Solving MILP ===================
def modeling_solving_optimal_milp(cipher, add_constraints, config_model, config_solver): # Generate and Solve MILP to find the optimal (minimal) objective function value.
    config_solver["target"] = "OPTIMAL"
    return modeling_solving_milp(cipher, add_constraints, config_model, config_solver)

def modeling_solving_optimal_start_from_milp(cipher, add_constraints, config_model, config_solver, start_value):      
    constraints = (add_constraints or [])
    constraints += gen_predefined_constraints("milp", "AT_LEAST", ["obj"], start_value) # Generate the constraint for the objective function value >= start_value.      
    config_solver["target"] = "OPTIMAL"
    return modeling_solving_milp(cipher, constraints, config_model, config_solver)

def modeling_solving_at_most_milp(cipher, add_constraints, config_model, config_solver, atmost_val):
    constraints = (add_constraints or [])
    constraints += gen_predefined_constraints("milp", "AT_MOST", ["obj"], atmost_val) # Generate the constraint for the objective function value <= atmost_val.
    config_solver["target"] = "SATISFIABLE"
    return modeling_solving_milp(cipher, constraints, config_model, config_solver)

def modeling_solving_exactly_milp(cipher, add_constraints, config_model, config_solver, exact_val):
    constraints = (add_constraints or [])
    constraints += gen_predefined_constraints("milp", "EXACTLY", ["obj"], exact_val) # Generate the constraint for the objective function value = exact_val.      
    config_solver["target"] = "SATISFIABLE"
    return modeling_solving_milp(cipher, constraints, config_model, config_solver)

def modeling_solving_milp(cipher, add_constraints, config_model, config_solver):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "milp", config_model)

    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        cons_type = config_model["matsui_constraint"].get("matsui_milp_cons_type", "ALL")
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        constraints += gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type)
    
    constraints += (add_constraints or [])
    filename = config_model.get("filename", f"{cipher.name}_milp_model.lp")
    model = solving.gen_milp_model(constraints, obj_fun, filename)    
    sol = solving.solve_milp(filename, config_solver)
    if sol is not None:
        sol["obj_fun"] = obj_fun
    return sol


# =================== Modeling and Solving SAT ===================
def modeling_solving_optimal_sat(cipher, goal, constraints, config_model, config_solver): # Find the optimal solution starting from objective value = 0
    return modeling_solving_optimal_start_from_sat(cipher, goal, constraints, config_model, config_solver, 0)

def modeling_solving_optimal_start_from_sat(cipher, goal, add_constraints, config_model, config_solver, start_value): # Incrementally solve SAT from a starting objective value until a feasible solution/maximum limit is found.
    sol = {}
    obj_val = start_value
    max_val = config_model.get("max_obj_sat", 100)
    strategy = config_model.get("optimal_search_strategy_sat", "AT_MOST")
    
    if has_Sbox_with_decimal_weights(cipher, goal):
        while not sol and obj_val <= max_val:
            print("Current SAT objective value: ", obj_val)
            decimal_encoding = config_model.get("decimal_encoding_sat", "INTEGER_DECIMAL")
            obj_list, obj_encodings_list = generate_obj_encodings_sat(cipher, goal, obj_val, obj_val-1, encoding=decimal_encoding)
            for i in range(len(obj_list)):
                if strategy == "AT_MOST":
                    sol = modeling_solving_at_most_decimal_sat(cipher, add_constraints, config_model, config_solver, obj_encodings_list[i])
                elif strategy == "EXACTLY":
                    sol = modeling_solving_exactly_decimal_sat(cipher, add_constraints, config_model, config_solver, obj_encodings_list[i])
                if sol:
                    sol.update({"obj_fun_value": obj_list[i], "status": "OPTIMAL"})
                    return sol
            obj_val += 1

    else:
        while not sol and obj_val <= max_val:
            print("Current SAT objective value: ", obj_val)
            if strategy == "AT_MOST":
                sol = modeling_solving_at_most_integer_sat(cipher, add_constraints, config_model, config_solver, obj_val)
            elif strategy == "EXACTLY":
                sol = modeling_solving_exactly_integer_sat(cipher, add_constraints, config_model, config_solver, obj_val)
            if sol:
                sol.update({"obj_fun_value": obj_val, "status": "OPTIMAL"})
                return sol
            obj_val += 1


def modeling_solving_at_most_decimal_sat(cipher, add_constraints, config_model, config_solver, max_val_encoding):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model) # Generate round constraints and objective function.
    
    decimal_encoding = config_model.get("decimal_encoding_sat", "INTEGER_DECIMAL")
    atmost_encoding = config_model.get("atmost_encoding_sat", "SEQUENTIAL")
    obj_fun_list = gen_obj_fun_encoding_sat(obj_fun, len(max_val_encoding), decimal_encoding)

    for i in range(len(max_val_encoding)):
        constraints += gen_predefined_constraints("sat", "SUM_AT_MOST", obj_fun_list[i], max_val_encoding[i], encoding=atmost_encoding)
    
    constraints += (add_constraints or []) 
    return modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)


def modeling_solving_at_most_integer_sat(cipher, add_constraints, config_model, config_solver, max_val):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model) # Generate round constraints and objective function.
    obj_fun_vars = gen_obj_fun_variables(obj_fun)
    atmost_encoding = config_model.get("atmost_encoding_sat", "SEQUENTIAL")
        
    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        GroupConstraintChoice = config_model["matsui_constraint"].get("GroupConstraintChoice", 1)
        GroupNumForChoice = config_model["matsui_constraint"].get("GroupNumForChoice", 1)
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        constraints += gen_matsui_constraints_sat(Round, best_obj, max_val, obj_fun_vars, GroupConstraintChoice, GroupNumForChoice)
    
    else:
        hw_list = [obj for row in obj_fun_vars for obj in row]
        constraints += gen_predefined_constraints("sat", "SUM_AT_MOST", hw_list, max_val, encoding=atmost_encoding)
            
    constraints += (add_constraints or [])
    return modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)


def modeling_solving_at_most_sat(cipher, goal, add_constraints, config_model, config_solver, max_val):
    if has_Sbox_with_decimal_weights(cipher, goal):
        decimal_encoding = config_model.get("decimal_encoding_sat", "INTEGER_DECIMAL")
        obj_list, obj_encodings_list = generate_obj_encodings_sat(cipher, goal, max_val, -1, encoding=decimal_encoding)
        for i in reversed(range(len(obj_list))):
            sol = modeling_solving_at_most_decimal_sat(cipher, add_constraints, config_model, config_solver, obj_encodings_list[-1])
            if sol:
                sol.update({"obj_fun_value": obj_list[i], "status": "SATISFIABLE"})
                return sol
    else:
        sol = modeling_solving_at_most_integer_sat(cipher, add_constraints, config_model, config_solver, max_val)
        if sol:
            sol.update({"obj_fun_value": max_val, "status": "SATISFIABLE"})
            return sol


def modeling_solving_exactly_decimal_sat(cipher, add_constraints, config_model, config_solver, exact_val_encoding):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model) # Generate round constraints and objective function.
    
    decimal_encoding = config_model.get("decimal_encoding_sat", "INTEGER_DECIMAL")
    exact_encoding = config_model.get("exact_encoding_sat", 1)
    obj_fun_list = gen_obj_fun_encoding_sat(obj_fun, len(exact_val_encoding), decimal_encoding)
    
    for i in range(len(exact_val_encoding)):
        constraints += gen_predefined_constraints("sat", "SUM_EXACTLY", obj_fun_list[i], exact_val_encoding[i], encoding=exact_encoding)

    constraints += (add_constraints or [])
    return modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)

def modeling_solving_exactly_integer_sat(cipher, add_constraints, config_model, config_solver, exact_val):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model)
    obj_fun_vars = gen_obj_fun_variables(obj_fun)    
    hw_list = [obj for row in obj_fun_vars for obj in row]
    exact_encoding = config_model.get("exact_encoding_sat", 1)
    constraints += gen_predefined_constraints("sat", "SUM_EXACTLY", hw_list, exact_val, encoding=exact_encoding)

    constraints += (add_constraints or [])
    return modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)


def modeling_solving_exactly_sat(cipher, goal, add_constraints, config_model, config_solver, exact_val):
    if has_Sbox_with_decimal_weights(cipher, goal):
        decimal_encoding = config_model.get("decimal_encoding_sat", "INTEGER_DECIMAL")
        obj_list, obj_encodings_list = generate_obj_encodings_sat(cipher, goal, exact_val, exact_val-0.1, encoding=decimal_encoding)
        for i in reversed(range(len(obj_list))):
            if obj_list[i] == exact_val:
                sol = modeling_solving_exactly_decimal_sat(cipher, add_constraints, config_model, config_solver, obj_encodings_list[-1])
                if sol:
                    sol.update({"obj_fun_value": obj_list[i], "status": "SATISFIABLE"})
                    return sol    
    else:
        sol = modeling_solving_exactly_integer_sat(cipher, add_constraints, config_model, config_solver, exact_val)
        if sol:
            sol.update({"obj_fun_value": exact_val, "status": "SATISFIABLE"})
            return sol

def modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun):
    filename = config_model.get("filename", f"{cipher.name}_sat_model.cnf")
    model = solving.gen_sat_model(constraints=constraints, filename=filename)
    sol = solving.solve_sat(filename, model["variable_map"], config_solver)
    if sol:
        sol.update({"obj_fun": obj_fun, "status": "SATISFIABLE"})
    return sol

# =================== Utilities for S-box-based ciphers with float weights ===================
def detect_Sbox(cipher): # Detect and return the first Sbox operator in the cipher
    states, rounds, layers, positions = fill_states_rounds_layers_positions(cipher)
    for s in states:  
        for r in rounds[s]:
            for l in layers[s][r]:
                for cons in cipher.states[s].constraints[r][l]:
                    if "Sbox" in cons.__class__.__name__:
                        return cons
    return None


def has_Sbox_with_decimal_weights(cipher, goal):
    Sbox = detect_Sbox(cipher)
    if Sbox and goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
        if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
            table = Sbox.computeDDT()
        weights = Sbox.gen_weights(table)
        return any(not float(w).is_integer() for w in weights)
    return False


def linear_combinations_bounds(weights, upper_bound, lower_bound=-1): # Enumerate all integer linear combinations of weights such that the sum is within (lower_bound, upper_bound].
    n = len(weights)
    seen = set()
    result = []
    # Each state is (sum, coeffs), Start with zero combination
    initial = (0.0, (0,) * n)
    heap = [initial]
    seen.add(initial[1])
    while heap:
        total, coeffs = heapq.heappop(heap)
        if lower_bound < total <= upper_bound:
            result.append((total, coeffs))
        # Try to increment each coefficient
        for i in range(n):
            new_coeffs = list(coeffs)
            new_coeffs[i] += 1
            new_coeffs = tuple(new_coeffs)
            if new_coeffs not in seen:
                new_sum = total + weights[i]
                if new_sum <= upper_bound:
                    heapq.heappush(heap, (new_sum, new_coeffs))
                    seen.add(new_coeffs)
    return result


def generate_obj_encodings_sat(cipher, goal, obj_ub, obj_lb=-1, encoding="INTEGER_DECIMAL"):
    Sbox = detect_Sbox(cipher)
    if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
        table = Sbox.computeDDT()
    weights = Sbox.gen_weights(table)
    integers_weight, floats_weight = Sbox.gen_integer_float_weight(table)
    obj_list, obj_encoding_list = [], []
    combs = linear_combinations_bounds(weights, obj_ub, obj_lb)
    weight_pattern_map = {str(w): Sbox.gen_weight_pattern_sat(integers_weight, floats_weight, w) for w in weights}

    for total, coeffs in combs:
        print(f"Sum: {total:.4f}, Coeffs: {coeffs}")
        obj = [0 for _ in range(max(integers_weight)+len(floats_weight))]
        for i in range(len(coeffs)):
            if coeffs[i] > 0:
                w = weights[i]
                pattern = weight_pattern_map[str(w)]
                for j in range(len(obj)):
                    obj[j] += coeffs[i] * pattern[j]
        if encoding == "BOOLEAN":
            obj_list.append(total)
            obj_encoding_list.append(obj)
        elif encoding == "INTEGER_DECIMAL":
            new_obj = [sum(obj[:max(integers_weight)])] + obj[max(integers_weight):]
            if total not in obj_list and new_obj not in obj_encoding_list:
                obj_list.append(total)
                obj_encoding_list.append(new_obj)
    return obj_list, obj_encoding_list


def gen_obj_fun_variables(obj_fun):
    obj_fun_var = []
    for obj_fun_r in obj_fun:
        obj_fun_var_r = []
        for obj in obj_fun_r:
            obj_fun_var_r.extend([item.strip().split()[1] if len(item.split()) > 1 else item.strip() for item in obj.split('+')])
        obj_fun_var.append(obj_fun_var_r)
    return obj_fun_var


def gen_obj_fun_encoding_sat(obj_fun, dim, encoding="INTEGER_DECIMAL"): # In the case of a decimal-weighted objective function, parse objective function variables and group them into separate components for SAT modeling 
    hw_list = [[] for _ in range(dim)]
    total_dim = len(obj_fun[0][0].split('+'))
    obj_fun_vars = gen_obj_fun_variables(obj_fun)
    for obj_fun_vars_r in obj_fun_vars:
        n_vars = len(obj_fun_vars_r)
        group_size = n_vars // total_dim
        if encoding == "INTEGER_DECIMAL": # Decompose obj_fun into integer and fractional parts.
            for i in range(group_size):
                hw_list[0].extend(obj_fun_vars_r[i*total_dim: i*total_dim+total_dim-dim+1])
                for j in range(1, dim):
                    hw_list[j].append(obj_fun_vars_r[i*total_dim+total_dim-dim+j])

        elif encoding == "BOOLEAN": # Decompose obj_fun into individual Boolean variables
            assert total_dim == dim, f"Error total_dim = {total_dim}, dim = {dim}"
            for i in range(group_size):
                for j in range(dim):
                    hw_list[j].append(obj_fun_vars_r[i * dim + j])
    
    return hw_list

# =================== Additional Constraints and Advanced Strategies ===================
def gen_predefined_constraints(model_type, cons_type, cons_vars, cons_value, bitwise=True, encoding=None): 
    """
    Generate commonly used, predefined model constraints based on type and parameters.

    Args:
        cons_type (str): The constraint type, must be one of the predefined types:
            - "EXACTLY": All selected variables == a target value.
            - "AT_LEAST": All selected variables >= target value.
            - "AT_MOST": All selected variables <= target value.
            - "SUM_EXACTLY": Sum of selected variables == target value.
            - "SUM_AT_LEAST": Sum of selected variables >= target value.
            - "SUM_AT_MOST": Sum of selected variables <= target value.
        cons_args (dict): Parameters for constraint generation, including:
            - cons_value (int): Target value for the constraint.
            - cons_vars (list[str]): Additional variable names to include.
            - bitwise (bool): If True, expand variables by bit.

    Returns:
        list[str]: List of generated model constraint strings.

    """
    if cons_type in ["EXACTLY", "SUM_EXACTLY", "AT_LEAST", "SUM_AT_LEAST", "AT_MOST", "SUM_AT_MOST"]:
        cons_vars_name = []
        for var in cons_vars:
            if isinstance(var, str):
                cons_vars_name.append(var) 
            else:
                if bitwise and var.bitsize > 1:
                    cons_vars_name.extend([f"{var.ID}_{j}" for j in range(var.bitsize)])
                else:
                    cons_vars_name.append(var.ID)
        if cons_type == "EXACTLY":
            return gen_constraints_exactly(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_EXACTLY":
            return gen_constraints_sum_exactly(model_type, cons_vars_name, cons_value, encoding)
        elif cons_type == "AT_MOST":
            return gen_constraints_at_most(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_AT_MOST":
            return gen_constraints_sum_at_most(model_type, cons_vars_name, cons_value, encoding)
        elif cons_type == "AT_LEAST":
            return gen_constraints_at_least(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_AT_LEAST":
            return gen_constraints_sum_at_least(model_type, cons_vars_name, cons_value, encoding)
    raise ValueError(f"Unsupported cons_type '{cons_type}'.")
        
def gen_constraints_exactly(model_type, cons_vars, cons_value):
    if model_type == "milp": 
        return [f"{cons_vars[i]} = {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0: 
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1: 
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for EXACTLY constraint.")

def gen_constraints_sum_exactly(model_type, cons_vars, cons_value, encoding=1):
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" = {cons_value}"]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and pysat_import:
        assert encoding in [0,1,2,3,4,5,6,7,8,9], f"[ERROR] Invalid encoding = {encoding}, refer https://pysathq.github.io/docs/html/api/card.html"
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
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_EXACTLY constraint.")

def gen_constraints_at_most(model_type, cons_vars, cons_value):
    if model_type == "milp": 
        return [f"{cons_vars[i]} <= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for AT_MOST constraint.")

def gen_constraints_sum_at_most(model_type, cons_vars, cons_value, encoding="SEQUENTIAL"):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]
    elif model_type == "sat" and encoding == "SEQUENTIAL":
        return gen_sequential_encoding_sat(cons_vars, cons_value)
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
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_AT_MOST constraint.")   
    
def gen_constraints_at_least(model_type, cons_vars, cons_value): 
    if model_type == "milp": 
        return [f"{cons_vars[i]} >= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1: 
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0: 
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for GREATER_EQUAL constraint.")

def gen_constraints_sum_at_least(model_type, cons_vars, cons_value, encoding="SEQUENTIAL"):     
    if model_type == "milp": 
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" >= {cons_value}"]
    elif model_type == "sat" and cons_value == 1: 
        return [' '.join(f"{cons_vars[i]}" for i in range(len(cons_vars)))]
    elif model_type == "sat" and encoding == "SEQUENTIAL":
        return gen_sequential_encoding_sat(cons_vars, cons_value, greater_or_equal=True)
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

def gen_sequential_encoding_sat(hw_list, weight, dummy_variables=None, greater_or_equal=False): # Generate SAT constraints for a sequential counter encoding of a cardinality constraint. reference: https://github.com/Crypto-TII/claasp/blob/main/claasp/cipher_modules/models/sat/sat_model.py#L262
    if not hasattr(gen_sequential_encoding_sat, "_counter"): # Use function attribute to set global counter
        gen_sequential_encoding_sat._counter = 0
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
        gen_sequential_encoding_sat._counter += 1
        prefix = f'dummy_seq_{gen_sequential_encoding_sat._counter}'
        dummy_variables = [[f'{prefix}_{i}_{j}' for j in range(weight)] for i in range(n - 1)]
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


# Adding Matsui's branch-and-bound constraints in differential and linear cryptanalysis
def gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type="ALL"): # Generate Matsui's additional constraints for MILP models. Reference: Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy.
    add_cons = []
    for i in range(1, Round):
        if best_obj[i-1] > 0:
            if cons_type == "ALL" or cons_type == "UPPER":
                w_vars = [var for r in range(i + 1, Round + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="AT_MOST", cons_value=-best_obj[i-1], cons_vars=all_vars)  
            if cons_type == "ALL" or cons_type == "LOWER":
                w_vars = [var for r in range(1, Round - i + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="AT_MOST", cons_value=-best_obj[i-1], cons_vars=all_vars)                        
    return add_cons


def gen_matsui_constraints_sat(Round, best_obj, obj_sat, obj_var, GroupConstraintChoice=1, GroupNumForChoice=1): # Generate Matsui's additional constraints for SAT models. Reference: Ling Sun, Wei Wang and Meiqin Wang. Accelerating the Search of Differential and Linear Characteristics with the SAT Method. https://github.com/SunLing134340/Accelerating_Automatic_Search
    if not hasattr(gen_matsui_constraints_sat, "_counter"): # Use function attribute to set global counter
        gen_matsui_constraints_sat._counter = 0
    if len(best_obj) == Round-1:
        best_obj = [0] + best_obj
    Main_Vars = list([])
    for r in range(Round):
        for i in range(len(obj_var[Round - 1 - r])):
            Main_Vars += [obj_var[Round - 1 - r][i]]
    gen_matsui_constraints_sat._counter += 1
    dummy_var = [[f'dummy_matsui_{gen_matsui_constraints_sat._counter}_{i}_{j}' for j in range(obj_sat)] for i in range(len(Main_Vars) - 1)]
    constraints = gen_sequential_encoding_sat(hw_list=Main_Vars, weight=obj_sat, dummy_variables=dummy_var) # Generate the constraint of "the objective function value is at most obj" using the sequential encoding method
    
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