import sys
import time
import heapq
import re
import copy
try:
    from pysat.card import CardEnc
    from pysat.formula import IDPool
    vpool = IDPool(start_from=1000)
    pysat_import = True
except ImportError:
    print("[WARNING] pysat module can't be loaded \n")
    pysat_import = False
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] # this file -> attacks -> <ROOT>
sys.path.insert(0, str(ROOT))
import attacks.linear_cryptanalysis as lin
import attacks.differential_cryptanalysis as dif
import solving.solving as solving
import visualisations.visualisations as vis 

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


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
def parse_and_set_configs(cipher, config_model, config_solver): # Parse input parameters and apply default values for model and solver configurations.
    """
    Available options for config_model:
        - model_type: Type of the model (MILP or SAT)
        - model_params: Parameters for the model, Options: {cons_class_name: {parame_name: param_value}}. E.g., {"PRESENT_Sbox": {"tool_type": "polyhedron"}}
        - functions: List of functions to include in the model. By default, use all functions.
        - rounds: Dictionary with "functions" key, containing a list of rounds. By default, use all rounds.
        - layers: Dictionary with "functions" key, "rounds" key, containing a list of layers. By default, use all layers.
        - positions: Dictionary with "functions" key, "rounds" key, "layers" key containing a list of positions. By default, use all positions.
        - model_version: Configuration for model version assignment. Example: {"model_version": "XORDIFF", "operator_name": "Sbox"}
        - matsui_constraint: Arguments for Matsui branch-and-bound constraints. Example: {"Round": 2, "best_obj": [1], "matsui_milp_cons_type": "ALL"}.
        - filename: The filename for saving the model.
        - optimal_search_strategy_sat: Strategy for searching optimal SAT solutions. Options: "INCREASING FROM AT MOST X", "INCREASING FROM EXACTLY X", "DECREASING FROM AT MOST X", "DECREASING FROM EXACTLY X".
        - atmost_encoding_sat, exact_encoding_sat, atleast_encoding_sat: Encoding strategies for SAT constraints.

    Available options for config_solver:
        - solver: The solver to use for solving the model (e.g., "GUROBI", "SCIP").
        - TimeLimit, SolutionLimit, PoolSearchMode, PoolSolutions, MIPFocus, etc. (for GUROBI solvers).
    """

    # ===== Set Default config_model =====
    config_model = config_model or {}
    
    # Set "model_type", the automated model framework
    config_model["model_type"] = config_model.get("model_type", "milp").lower()
    
    # Set "functions", "rounds", "layers", "positions"
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)

    if config_model["model_type"] == "milp":
        # Set the model "filename".
        config_model["filename"] = str(FILES_DIR / f"{cipher.name}_{config_model['model_type']}_model.lp")
    elif config_model["model_type"] == "sat":
        # Set the model "filename".
        config_model["filename"] = str(FILES_DIR / f"{cipher.name}_{config_model['model_type']}_model.cnf")
        
    # ===== Set Default config_solver =====
    config_solver = config_solver or {}
    
    # Set "solver" for solving the model
    config_solver.setdefault("solver", "DEFAULT")    

    return config_model, config_solver

def fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None):
    """
    Fill in functions, rounds, layers, and positions to full coverage when the corresponding argument is None; otherwise, keep user-supplied values.

    Parameters:
        cipher (object): The cipher object.
        functions (list[str]): List of functions. If None, use all functions of the cipher. Example: ["FUNCTION", "KEY_SCHEDULE", "SUBKEYS"].
        rounds (dict): Dictionary specifying rounds. If None, use all. Example: {"FUNCTION": [1, 2, 3]}.
        layers (dict): Dictionary specifying layers. If None, use all. Example: {"FUNCTION": {1: [0, 1], 2: [0, 1], 3: [0, 1]}}.
        positions (dict): Dictionary specifying positions. If None, use all. Example: {"FUNCTION": {1: {0: [0, 1], 1: [0, 1]}, 2: {0: [0, 1], 1: [0, 1]}, 3: {0: [0, 1], 1: [0, 1]}}}.

    Returns:
        tuple: (functions, rounds, layers, positions)
    """

    if functions is None:
        functions = [f for f in cipher.functions]
    if rounds is None:
        rounds = {f: list(range(1, cipher.functions[f].nbr_rounds + 1)) for f in functions}
    if layers is None:
        layers = {f: {r: list(range(cipher.functions[f].nbr_layers+1)) for r in rounds[f]} for f in functions}
    if positions is None:
        positions = {f: {r: {l: list(range(len(cipher.functions[f].constraints[r][l]))) for l in layers[f][r]} for r in rounds[f]} for f in functions}
    return functions, rounds, layers, positions

def set_model_versions(cipher, version, functions=None, rounds=None, layers=None, positions=None, operator_name=None): # Assigns a specified model_version to constraints (operators) in the cipher based on specified parameters.
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher, functions, rounds, layers, positions)
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for cons in cipher.functions[f].constraints[r][l]:
                    if operator_name is None: # Assign model_version to all operators in the cipher.
                        cons.model_version = cons.__class__.__name__ + "_" + version
                    elif operator_name is not None and operator_name in cons.__class__.__name__: #  Assign model_version to operators with a specific name.
                        cons.model_version = cons.__class__.__name__ + "_" + version                            

def configure_model_version(cipher, goal, config_model):
    """
    Configure the model version for all operators in the cipher based on the attack goal and config_model.

    Parameters:
        cipher (object): The cipher object.
        goal (str): The attack goal, which determines the model version assignment.
        config_model (dict): Configuration dictionary that may contain model version settings.
            Example:
                config_model['model_version'] = {
                    'model_version': 'XORDIFF',
                    'operator_name': 'Sbox'  # Optional, can be None
                }
    """
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

    elif goal == "TRUNCATEDDIFF_SBOXCOUNT":
        set_model_versions(cipher, "TRUNCATEDDIFF") # Set model_version = "TRUNCATEDDIFF" for all operators
        set_model_versions(cipher, "TRUNCATEDDIFF_A", operator_name="Sbox") # Set model_version = "TRUNCATEDDIFF_A" for all Sbox operators
    
    else:
        raise ValueError(f"Invalid goal: {goal}.")


# =================== Model Constraint and Objective Function Generation ===================
def gen_round_model_constraint_obj_fun(cipher, model_type, config_model): # Generate constraints for a given cipher based on user-specified parameters.
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher, config_model.get("functions"), config_model.get("rounds"), config_model.get("layers"), config_model.get("positions"))
        
    constraint = []
    obj_fun = [[] for _ in range(cipher.functions["FUNCTION"].nbr_rounds)]
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for cons in cipher.functions[f].constraints[r][l]:
                    cons_class_name = cons.__class__.__name__
                    params = (config_model.get("model_params") or {}).get(cons_class_name, {}) # get operator-specific params if available
                    constraint += cons.generate_model(model_type=model_type, **params)                  
                    if hasattr(cons, 'weight'): 
                        obj_fun[r-1] += cons.weight
    return constraint, obj_fun


# =================== Modeling and Solving for Attacks ===================
def modeling_and_solving(cipher, goal, objective_target, constraints, config_model, config_solver): # Main interface for modeling and solving based on the given objective_target
    time_start = time.time()

    configure_model_version(cipher, goal, config_model)
    model_type = config_model.get("model_type")

    if model_type == "milp":
        sol = modeling_solving_milp(cipher, goal, objective_target, constraints, config_model, config_solver)

    elif model_type == "sat":
        sol = modeling_solving_sat(cipher, goal, objective_target, constraints, config_model, config_solver)

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat']")

    print("=== Modeling and Solving Information ===")
    print(f"--- Found {len(sol)} solution(s) ---")
    print(f"--- Total Time ---: {time.time() - time_start} seconds")
    for key, value in {**config_model, **config_solver}.items():
        print(f"--- {key} ---: {value}")
    return sol


# =================== Modeling and Solving MILP ===================
def gen_constraints_from_objective_target_milp(objective_target):
    if objective_target.startswith("AT MOST"):
        try:
            max_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT MOST X'.")
        constraints = gen_predefined_constraints("milp", "AT_MOST", ["obj"], max_val) # Generate the constraint for the objective function value <= atmost_val
    elif objective_target.startswith("EXACTLY"):
        try:
            exact_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'EXACTLY X'.")
        constraints = gen_predefined_constraints("milp", "EXACTLY", ["obj"], exact_val) # Generate the constraint for the objective function value = exact_val.    
    elif objective_target.startswith("AT LEAST"):
        try:
            atleast_value = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT LEAST X'.")
        constraints = gen_predefined_constraints("milp", "AT_LEAST", ["obj"], atleast_value) # Generate the constraint for the objective function value >= atleast_value.
    else:
        constraints = []
    return constraints

def modeling_solving_milp(cipher, goal, objective_target, constraints, config_model, config_solver):
    round_constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "milp", config_model)
    add_constraints = (constraints or [])
    add_constraints += gen_constraints_from_objective_target_milp(objective_target)
    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        cons_type = config_model["matsui_constraint"].get("matsui_milp_cons_type", "ALL")
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        add_constraints += gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type)
    constraints = round_constraints + add_constraints
    filename = config_model.get("filename", f"{cipher.name}_milp_model.lp")
    model = solving.gen_milp_model(constraints, obj_fun, filename)    
    solutions = solving.solve_milp(filename, config_solver)
    for sol in solutions:
        sol["rounds_obj_fun_values"] = cal_round_obj_fun_values_from_solution(obj_fun, sol)
    return solutions


# =================== Modeling and Solving SAT ===================
def modeling_solving_sat(cipher, goal, objective_target, constraints, config_model, config_solver):
    if objective_target == "OPTIMAL":
        return modeling_solving_optimal_sat(cipher, goal, constraints, config_model, config_solver)
    
    elif objective_target.startswith("AT MOST"):
        try:
            max_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT MOST X'.")
    
        solutions = modeling_solving_at_most_sat(cipher, constraints, config_model, config_solver, int(max_val), atmost_val_decimal=None)
        if has_Sbox_with_decimal_weights(cipher, goal) and isinstance(solutions, list) and len(solutions) > 0:
            solutions_new = []
            for sol in solutions:
                true_obj = sol["obj_fun_value"]
                if true_obj <= max_val:
                    solutions_new.append(sol)
                elif true_obj > max_val:
                    obj_decimal_list = generate_obj_decimal_coms(cipher, goal, -1, max_val)
                    for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list): 
                        sol_decimal_list = modeling_solving_at_most_sat(cipher, constraints, config_model, config_solver, obj_integer, atmost_val_decimal=obj_decimal)
                        if isinstance(sol_decimal_list, list) and len(sol_decimal_list) > 0:
                            solutions_new.extend(sol_decimal_list)
            solutions = solutions_new
        return solutions
    
    elif objective_target.startswith("EXACTLY"):
        try:
            exact_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'EXACTLY X'.")
        if has_Sbox_with_decimal_weights(cipher, goal):
            EPS = 0.001
            obj_decimal_list = generate_obj_decimal_coms(cipher, goal, -1, exact_val)
            for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list): 
                if abs(true_obj - exact_val) < EPS: # Allow a small tolerance for floating-point comparison
                    solutions = modeling_solving_exactly_sat(cipher, constraints, config_model, config_solver, obj_integer, exact_val_decimal=obj_decimal)
                    if isinstance(solutions, list) and len(solutions) > 0:
                        break
        else:
            solutions = modeling_solving_exactly_sat(cipher, constraints, config_model, config_solver, int(exact_val), exact_val_decimal=None)
        return solutions

    elif objective_target.startswith("AT LEAST"):
        try:
            atleast_value = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT LEAST X'.")
        solutions = modeling_solving_at_least_sat(cipher, constraints, config_model, config_solver, int(atleast_value), atleast_val_decimal=None)
        if has_Sbox_with_decimal_weights(cipher, goal):
            solutions_new = []
            at_least_int = int(atleast_value)
            while not solutions_new and at_least_int >= 0:
                for sol in solutions:
                    true_obj = sol["obj_fun_value"]
                    if true_obj >= atleast_value: # Check if the true objective value meets the 'at least' condition
                        solutions_new.append(sol)
                        break
                if not solutions_new: # If no valid solution is found, reduce the threshold and re-solve
                    config_solver_local = copy.deepcopy(config_solver)
                    config_solver_local["solution_number"] = 10000 # Allow 10000 solutions for each iteration
                    solutions = modeling_solving_at_least_sat(cipher, constraints, config_model, config_solver_local, at_least_int, atleast_val_decimal=None)
                    at_least_int -= 1
            solutions = solutions_new        
        return solutions    
    else:
        raise ValueError(f"Invalid objective_target: {objective_target}. Expected one of 'OPTIMAL', 'AT MOST X', 'EXACTLY X', 'AT LEAST X'.")

def modeling_solving_optimal_sat(cipher, goal, constraints, config_model, config_solver): # Find the optimal solution starting from objective value = 0
    
    solutions = modeling_solving_optimal_sat_intobj(cipher, goal, constraints, config_model, config_solver)

    if not has_Sbox_with_decimal_weights(cipher, goal):
        return solutions
    
    # Search for the best solutions with decimal weights in the objective function
    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0")
    if "AT MOST" in optimal_search_strategy_sat:
        strategy = "AT_MOST"
    elif "EXACTLY" in optimal_search_strategy_sat:
        strategy = "EXACTLY"
    max_obj_val = solutions[0]["obj_fun_value"] # The best objective function value found with the minimal integer weight
    int_obj_val = solutions[0]["integer_obj_fun_value"] # Start searching from the integer part of the optimal solution
    while int_obj_val < max_obj_val:
        print("[INFO] Current SAT objective value: ", int_obj_val)
        if solutions:
            obj_decimal_list = generate_obj_decimal_coms(cipher, goal, int_obj_val, max_obj_val)
            for (true_obj, obj_integer, obj_decimal) in obj_decimal_list:
                if true_obj >= max_obj_val:
                    continue
                if strategy == "AT_MOST":
                    decimal_solutions = modeling_solving_at_most_sat(cipher, constraints, config_model, config_solver, int_obj_val, atmost_val_decimal=obj_decimal)
                elif strategy == "EXACTLY":
                    decimal_solutions = modeling_solving_exactly_sat(cipher, constraints, config_model, config_solver, int_obj_val, exact_val_decimal=obj_decimal)
                if isinstance(decimal_solutions, list) and len(decimal_solutions) > 0:
                    for sol in decimal_solutions:
                        max_obj_val = min(max_obj_val, sol["obj_fun_value"])
        int_obj_val += 1
    return solutions


def modeling_solving_optimal_sat_intobj(cipher, goal, constraints, config_model, config_solver): # Find the optimal solution with integer objective function value
    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0")
    obj_val = int(float(optimal_search_strategy_sat.split()[-1]))
    solutions = None
        
    if optimal_search_strategy_sat.startswith("INCREASING FROM AT MOST"):
        strategy = "AT MOST"
        step = 1
        end_obj_value = 100
    elif optimal_search_strategy_sat.startswith("INCREASING FROM EXACTLY"):
        strategy = "EXACTLY"
        step = 1
        end_obj_value = 100
    elif optimal_search_strategy_sat.startswith("DECREASING FROM AT MOST"):
        strategy = "AT MOST"
        step = -1
        end_obj_value = 0
    elif optimal_search_strategy_sat.startswith("DECREASING FROM EXACTLY"):
        strategy = "EXACTLY"
        step = -1
        end_obj_value = 0
    else:
        raise ValueError(f"Invalid optimal_search_strategy_sat: {optimal_search_strategy_sat}.")
    
    while obj_val != end_obj_value:
        print("[INFO] Current SAT objective value: ", obj_val)
        if strategy == "AT MOST":
            current_solutions = modeling_solving_at_most_sat(cipher, constraints, config_model, config_solver, obj_val, atmost_val_decimal=None)
        elif strategy == "EXACTLY":
            current_solutions = modeling_solving_exactly_sat(cipher, constraints, config_model, config_solver, obj_val, exact_val_decimal=None)
        if isinstance(current_solutions, list) and len(current_solutions) > 0:
            for sol in current_solutions:
                sol["integer_obj_fun_value"] = obj_val
        if optimal_search_strategy_sat.startswith("INCREASING FROM") and current_solutions:
            return current_solutions
        elif optimal_search_strategy_sat.startswith("DECREASING FROM") and not current_solutions:
            return solutions
        obj_val += step
        solutions = current_solutions


def modeling_solving_at_most_sat(cipher, add_constraints, config_model, config_solver, atmost_val, atmost_val_decimal=None): # 
    atmost_encoding = config_model.get("atmost_encoding_sat", "SEQUENTIAL")
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model) # Generate round constraints and objective function.

    if atmost_val_decimal is not None:
        obj_fun_vars, obj_fun_vars_decimal = gen_obj_fun_variables(obj_fun, obj_fun_decimal=True)
        for i in range(len(obj_fun_vars_decimal)):
            hw_list = [obj for row in obj_fun_vars_decimal[i] for obj in row]
            constraints += gen_predefined_constraints("sat", "SUM_AT_MOST", hw_list, atmost_val_decimal[i], encoding=atmost_encoding)

    else:
        obj_fun_vars = gen_obj_fun_variables(obj_fun, obj_fun_decimal=False)

    if "matsui_constraint" in config_model:
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        GroupConstraintChoice = config_model["matsui_constraint"].get("GroupConstraintChoice", 1)
        GroupNumForChoice = config_model["matsui_constraint"].get("GroupNumForChoice", 1)
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        constraints += gen_matsui_constraints_sat(Round, best_obj, atmost_val, obj_fun_vars, GroupConstraintChoice, GroupNumForChoice)
    
    else:
        hw_list = [obj for row in obj_fun_vars for obj in row]
        constraints += gen_predefined_constraints("sat", "SUM_AT_MOST", hw_list, atmost_val, encoding=atmost_encoding)
    
    constraints += (add_constraints or [])
    solutions = modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)

    return solutions


def modeling_solving_exactly_sat(cipher, add_constraints, config_model, config_solver, exact_val, exact_val_decimal=None):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model)
    exact_encoding = config_model.get("exact_encoding_sat", 1)

    if exact_val_decimal is not None:
        obj_fun_vars, obj_fun_vars_decimal = gen_obj_fun_variables(obj_fun, obj_fun_decimal=True)
        for i in range(len(obj_fun_vars_decimal)):
            hw_list = [obj for row in obj_fun_vars_decimal[i] for obj in row]
            constraints += gen_predefined_constraints("sat", "SUM_EXACTLY", hw_list, exact_val_decimal[i], encoding=exact_encoding)

    else:
        obj_fun_vars = gen_obj_fun_variables(obj_fun, obj_fun_decimal=False)

    hw_list = [obj for row in obj_fun_vars for obj in row]
    constraints += gen_predefined_constraints("sat", "SUM_EXACTLY", hw_list, exact_val, encoding=exact_encoding)
    
    constraints += (add_constraints or [])
    solutions = modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)

    return solutions

def modeling_solving_at_least_sat(cipher, add_constraints, config_model, config_solver, atleast_val, atleast_val_decimal=None):
    constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, "sat", config_model)
    atleast_encoding = config_model.get("atleast_encoding_sat", 1)

    if atleast_val_decimal is not None:
        obj_fun_vars, obj_fun_vars_decimal = gen_obj_fun_variables(obj_fun, obj_fun_decimal=True)
        for i in range(len(obj_fun_vars_decimal)):
            hw_list = [obj for row in obj_fun_vars_decimal[i] for obj in row]
            constraints += gen_predefined_constraints("sat", "SUM_AT_LEAST", hw_list, atleast_val_decimal[i], encoding=atleast_encoding)

    else:
        obj_fun_vars = gen_obj_fun_variables(obj_fun, obj_fun_decimal=False)

    hw_list = [obj for row in obj_fun_vars for obj in row]
    constraints += gen_predefined_constraints("sat", "SUM_AT_LEAST", hw_list, atleast_val, encoding=atleast_encoding)
    
    constraints += (add_constraints or [])
    solutions = modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun)

    return solutions

def modeling_solving_sat_model(cipher, constraints, config_model, config_solver, obj_fun):
    filename = config_model.get("filename", f"{cipher.name}_sat_model.cnf")
    model = solving.gen_sat_model(constraints=constraints, filename=filename)
    solutions = solving.solve_sat(filename, model["variable_map"], config_solver)
    if isinstance(solutions, list) and len(solutions) > 0:
        for sol in solutions:
            round_values = cal_round_obj_fun_values_from_solution(obj_fun, sol)
            sol["rounds_obj_fun_values"] = round_values
            sol["obj_fun_value"] = sum(round_values)
    return solutions

# =================== Utilities for S-box-based ciphers with decimal weights ===================
def detect_Sbox(cipher): # Detect and return the first Sbox operator in the cipher
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher)
    for f in functions:  
        for r in rounds[f]:
            for l in layers[f][r]:
                for cons in cipher.functions[f].constraints[r][l]:
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
    EPS = 0.001
    while heap:
        total, coeffs = heapq.heappop(heap)
        if lower_bound <= total <= (upper_bound + EPS):
            result.append((total, coeffs))
        # Try to increment each coefficient
        for i in range(n):
            new_coeffs = list(coeffs)
            new_coeffs[i] += 1
            new_coeffs = tuple(new_coeffs)
            if new_coeffs not in seen:
                new_sum = total + weights[i]
                if new_sum <= (upper_bound + EPS):
                    heapq.heappush(heap, (new_sum, new_coeffs))
                    seen.add(new_coeffs)
    return result


def generate_obj_decimal_coms(cipher, goal, obj_integer, obj_max_value): # generate all combinations of decimal weights, with integer weight = obj_val, and the total weight < max_val.
    obj_decimal_coms = []
    Sbox = detect_Sbox(cipher)
    if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
        table = Sbox.computeDDT()
    weights = Sbox.gen_weights(table)
    combs = linear_combinations_bounds(weights, obj_max_value, obj_integer)
    integers_weight, floats_weight = Sbox.gen_integer_float_weight(table)
    weight_pattern_map = {str(w): Sbox.gen_weight_pattern_sat(integers_weight, floats_weight, w) for w in weights}

    for total, coeffs in combs:
        obj = [0 for _ in range(max(integers_weight)+len(floats_weight))]
        for i in range(len(coeffs)):
            if coeffs[i] > 0:
                w = weights[i]
                pattern = weight_pattern_map[str(w)]
                for j in range(len(obj)):
                    obj[j] += coeffs[i] * pattern[j]
        decimal_com = obj[max(integers_weight):]
        int_obj = sum(obj[:max(integers_weight)])
        if  int_obj>= obj_integer and [total,int_obj,decimal_com] not in obj_decimal_coms:
            obj_decimal_coms.append([total,int_obj,decimal_com])
    return obj_decimal_coms


def gen_obj_fun_variables(obj_fun, obj_fun_decimal=False): # In the case of a decimal-weighted objective function, parse objective function variables and group them into separate components for SAT modeling 
    obj_fun_var_int = []
    for obj_fun_r in obj_fun:
        obj_fun_var_r_int = []
        for obj in obj_fun_r:
            terms = [t.strip() for t in obj.split('+')]
            for term in terms:
                parts = term.split()
                if len(parts) == 1:
                    obj_fun_var_r_int.append(parts[0])
                elif len(parts) >= 2:
                    coef, var = parts[0], parts[1]
                    if float(coef).is_integer():
                        obj_fun_var_r_int.append(var)
        obj_fun_var_int.append(obj_fun_var_r_int)
    if not obj_fun_decimal:
        return obj_fun_var_int
    else:
        decimal_vars = []
        terms = [t.strip() for t in obj_fun[0][0].split('+')]
        for term in terms:
            parts = term.split()
            if len(parts) >= 2:
                coef, var = parts[0], parts[1]
                if not float(coef).is_integer():
                    decimal_vars.append(coef)

        obj_fun_var_dec = {k: [] for k in decimal_vars}

        for obj_fun_r in obj_fun:
            obj_fun_var_r_dec = {k: [] for k in decimal_vars}
            for obj in obj_fun_r:
                terms = [t.strip() for t in obj.split('+')]
                for term in terms:
                    parts = term.split()
                    if len(parts) >= 2 and (not float(parts[0]).is_integer()):
                        obj_fun_var_r_dec[coef].append(parts[1]) 
            for k in decimal_vars:
                obj_fun_var_dec[k].append(obj_fun_var_r_dec[k])
        return obj_fun_var_int, [obj_fun_var_dec[k] for k in decimal_vars]
    

# =================== Objective Function Value Calculation ===================
def cal_round_obj_fun_values_from_solution(obj_fun, solution): # Calculate the objective function value for each round from the solution
    round_obj_fun_values = []
    for obj_fun_r in obj_fun:
        w = 0
        for obj_fun_r_i in obj_fun_r:
            terms = [t.strip() for t in obj_fun_r_i.split('+')]
            for term in terms:
                match = re.match(r'(\d*\.?\d*)\s*(\w+)', term.strip())
                if match:
                    coefficient = float(match.group(1)) if match.group(1) != '' else 1  # Default coefficient is 1 if not found
                    variable = match.group(2)
                    if variable in solution:
                        w += coefficient * solution[variable]
                else:
                    print(f"Warning: Unable to parse '{term.strip()}'")
        round_obj_fun_values.append(w)
    return round_obj_fun_values


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
        cons_vars (list[str]): Variable names.
        cons_value (int): Target value.
        bitwise (bool): If True, expand variables by bit.

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
        if not encoding:
            encoding = 1  # Default to 1 if not specified
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
        if not encoding:
            encoding = "SEQUENTIAL"  # Default to "SEQUENTIAL" if not specified
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
        if not encoding:
            encoding = "SEQUENTIAL"  # Default to "SEQUENTIAL" if not specified
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