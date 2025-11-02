import sys
import os
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] # this file -> tools -> <ROOT>
sys.path.insert(0, str(ROOT))

import tools.model_constraints as model_constraints
import tools.model_objective as model_objective
import solving.solving as solving


# **************************************************************************** #
# This module provides a interface for SAT-based modeling and solving for automated cryptanalysis, including:
# 1. Search with optimal / at-most / exactly / at-least strategies.
# 2. Generate standard CNF-format models.
# 3. Call the SAT solver (MiniSat, Glucose, etc.) to solve the model.
# **************************************************************************** #


# ------------------------- Modeling and Solving SAT --------------------------
def parse_objective_target(obj_target):
    if obj_target == "OPTIMAL":
        return "OPTIMAL", None
    for keyword in ["AT MOST", "EXACTLY", "AT LEAST"]:
        if obj_target.startswith(keyword):
            try:
                value = float(obj_target.split()[-1])
                return keyword, value
            except ValueError:
                raise ValueError(f"Invalid format: '{obj_target}'. Expected '{keyword} X'.")
    raise ValueError(f"Unsupported objective_target: {obj_target}")


def modeling_solving_sat(objective_target, constraints, objective_function, config_model, config_solver):
    strategy, value = parse_objective_target(objective_target)

    if strategy == "OPTIMAL":
        return modeling_solving_optimal(constraints, objective_function, config_model, config_solver)
    elif strategy == "AT MOST":
        return modeling_solving_at_most(constraints, objective_function, config_model, config_solver, value)
    elif strategy == "EXACTLY":
        return modeling_solving_exactly(constraints, objective_function, config_model, config_solver, value)
    elif strategy == "AT LEAST":
        return modeling_solving_at_least(constraints, objective_function, config_model, config_solver, value)
    else:
        raise ValueError(f"Invalid objective_target: {objective_target}")


# ------------------------- Optimal Search Strategy --------------------------
def modeling_solving_optimal(constraints, objective_function, config_model, config_solver): # Find the optimal SAT solution.
    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if not decimal_objective_function:
        return modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver)
    return modeling_solving_optimal_decimalobj(constraints, objective_function, config_model, config_solver)


def modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver):
    print(f"[INFO] Search for the optimal solutions with integer objective function value.")

    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0") # Strategy for searching optimal SAT solutions. Options: "INCREASING FROM AT MOST X", "INCREASING FROM EXACTLY X", "DECREASING FROM AT MOST X", "DECREASING FROM EXACTLY X".
    try:
        obj_val = int(optimal_search_strategy_sat.split()[-1])
    except ValueError:
        raise ValueError(f"Invalid format: '{optimal_search_strategy_sat}'. Expected 'INCREASING FROM AT MOST X', 'INCREASING FROM EXACTLY X', 'DECREASING FROM AT MOST X', or 'DECREASING FROM EXACTLY X'.")
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
            current_solutions = modeling_solving("AT MOST", constraints, objective_function, config_model, config_solver, obj_val, obj_val_decimal=None)
        elif strategy == "EXACTLY":
            current_solutions = modeling_solving("EXACTLY", constraints, objective_function, config_model, config_solver, obj_val, obj_val_decimal=None)
        if isinstance(current_solutions, list) and len(current_solutions) > 0:
            for sol in current_solutions:
                sol["integer_obj_fun_value"] = obj_val
        if optimal_search_strategy_sat.startswith("INCREASING FROM") and current_solutions:
            return current_solutions
        elif optimal_search_strategy_sat.startswith("DECREASING FROM") and not current_solutions:
            return solutions
        obj_val += step
        solutions = current_solutions


def modeling_solving_optimal_decimalobj(constraints, objective_function, config_model, config_solver):
    print(f"[INFO] Search for the optimal solutions with decimal objective function value.")

    # Step 1: Find the optimal solution with integer objective function value
    solutions = modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver)

    # Step 2: Refine search for decimal weights
    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0")
    if "AT MOST" in optimal_search_strategy_sat:
        strategy = "AT_MOST"
    elif "EXACTLY" in optimal_search_strategy_sat:
        strategy = "EXACTLY"
    max_obj_val = solutions[0]["obj_fun_value"] # The current objective function value is the upper bound
    int_obj_val = solutions[0]["integer_obj_fun_value"] # Start searching from the minimal integer objective function value
    Sbox = config_model.get("decimal_objective_function", {}).get("Sbox")
    table = config_model.get("decimal_objective_function", {}).get("table")
    if not Sbox or not table:
        raise ValueError("Missing Sbox or table information for decimal objective function search.")
    obj_decimal_list = model_objective.generate_obj_decimal_coms(Sbox, table, int_obj_val, max_obj_val)
    print(f"[INFO] True objective function value = {max_obj_val} with integer value = {int_obj_val}")
    for (true_obj, obj_integer, obj_decimal) in obj_decimal_list:
        if true_obj >= max_obj_val:
            continue
        print("[INFO] Trying decimal combination with true_obj =", true_obj, ", int_obj =", obj_integer, ", obj_decimal =", obj_decimal)
        if strategy == "AT_MOST":
            decimal_solutions = modeling_solving("AT MOST", constraints, objective_function, config_model, config_solver, int_obj_val, obj_val_decimal=obj_decimal)
        elif strategy == "EXACTLY":
            decimal_solutions = modeling_solving("EXACTLY", constraints, objective_function, config_model, config_solver, int_obj_val, obj_val_decimal=obj_decimal)
        if isinstance(decimal_solutions, list) and len(solutions) > 0:
            for sol in decimal_solutions:
                max_obj_val = min(max_obj_val, sol["obj_fun_value"])
            solutions = decimal_solutions
            break
    return solutions


# ------------------------- AT MOST Search Strategy --------------------------
def modeling_solving_at_most(constraints, objective_function, config_model, config_solver, at_most_value):
    print(f"[INFO] Search for solutions with the objective function value <= {at_most_value}.")

    solutions = modeling_solving("AT MOST", constraints, objective_function, config_model, config_solver, int(at_most_value), obj_val_decimal=None)

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if decimal_objective_function and isinstance(solutions, list) and len(solutions) > 0: # For ciphers with S-boxes having decimal weights, further filter and search for one solution with true objective value <= max_val
        for sol in solutions:
            try:
                true_obj = sol.get("obj_fun_value")
            except KeyError:
                print("[WARNING] Solution does not contain 'obj_fun_value'. Skipping.")
            if true_obj <= at_most_value:
                return [sol]
        # If no solution meets the true objective value <= atmost_value, further search
        Sbox = config_model.get("decimal_objective_function", {}).get("Sbox")
        table = config_model.get("decimal_objective_function", {}).get("table")
        if not Sbox or not table:
            raise ValueError("Missing Sbox or table information for decimal objective function search.")
        int_obj_val = int(at_most_value)
        while solutions:
            obj_decimal_list = model_objective.generate_obj_decimal_coms(Sbox, table, int_obj_val, at_most_value)
            for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list):
                if obj_integer > int_obj_val:
                    continue
                print("[INFO] Trying decimal combination with true_obj =", true_obj, ", int_obj =", obj_integer, ", obj_decimal =", obj_decimal)
                decimal_solutions = modeling_solving("AT MOST", constraints, objective_function, config_model, config_solver, int_obj_val, obj_val_decimal=obj_decimal)
                if isinstance(decimal_solutions, list) and len(decimal_solutions) > 0:
                    return decimal_solutions
            int_obj_val -= 1
            solutions = modeling_solving("AT MOST", constraints, objective_function, config_model, config_solver, int_obj_val, obj_val_decimal=None)
    return solutions


# ------------------------- EXACTLY Search Strategy --------------------------
def modeling_solving_exactly(constraints, objective_function, config_model, config_solver, exactly_value):
    print(f"[INFO] Search for solutions with the objective function value = {exactly_value}")

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if not decimal_objective_function:
        return modeling_solving("EXACTLY", constraints, objective_function, config_model, config_solver, int(exactly_value), obj_val_decimal=None)

    EPS = 0.001
    Sbox = config_model.get("decimal_objective_function", {}).get("Sbox")
    table = config_model.get("decimal_objective_function", {}).get("table")
    if not Sbox or not table:
        raise ValueError("Missing Sbox or table information for decimal objective function search.")
    obj_decimal_list = model_objective.generate_obj_decimal_coms(Sbox, table, -1, exactly_value)
    for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list):
        if abs(true_obj - exactly_value) < EPS: # Allow a small tolerance for floating-point comparison
            print("[INFO] Trying decimal combination with true_obj =", true_obj, ", int_obj =", obj_integer, ", obj_decimal =", obj_decimal)
            solutions = modeling_solving("EXACTLY", constraints, objective_function, config_model, config_solver, obj_integer, obj_val_decimal=obj_decimal)
            if isinstance(solutions, list) and len(solutions) > 0:
                return solutions
    return []


# ------------------------- AT LEAST Search Strategy -------------------------
def modeling_solving_at_least(constraints, objective_function, config_model, config_solver, at_least_value):
    print(f"[INFO] Search for solutions with objective function value >= {at_least_value}")
    solutions = modeling_solving("AT LEAST", constraints, objective_function, config_model, config_solver, int(at_least_value), obj_val_decimal=None)

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if decimal_objective_function:
        if solutions:
            for sol in solutions:
                try:
                    true_obj = sol.get("obj_fun_value")
                except Exception as e:
                    print(f"[ERROR] Failed to retrieve objective function value: {e}")
                    continue
                if true_obj >= at_least_value:
                    return [sol]
        print(f"[INFO] No solution found with integer objective function value >= {at_least_value}.") # TO DO: Search for
        return []
    return solutions


# ------------------ Core SAT Model Construction and Solving -----------------
def modeling_solving(objective_target, constraints, objective_function, config_model, config_solver, obj_val, obj_val_decimal=None):
    model_cons = copy.deepcopy(constraints) or []
    if objective_target == "AT MOST":
        cons_type = "SUM_AT_MOST"
        encoding = config_model.get("atmost_encoding_sat", "SEQUENTIAL")

    elif objective_target == "EXACTLY":
        cons_type = "SUM_EXACTLY"
        encoding = config_model.get("exact_encoding_sat", 1)

    elif objective_target == "AT LEAST":
        cons_type = "SUM_AT_LEAST"
        encoding = config_model.get("atleast_encoding_sat", "SEQUENTIAL")

    else:
        raise

    if obj_val_decimal is not None:
        obj_fun_vars, obj_fun_vars_decimal = model_objective.gen_obj_fun_variables(objective_function, obj_fun_decimal=True)
        assert len(obj_val_decimal) == len(obj_fun_vars_decimal), f"Length mismatch between objective function decimal variables and obj_val_decimal."
        for i in range(len(obj_fun_vars_decimal)):
            hw_list = [obj for row in obj_fun_vars_decimal[i] for obj in row]
            model_cons += model_constraints.gen_predefined_constraints("sat", cons_type, hw_list, obj_val_decimal[i], encoding=encoding)

    else:
        obj_fun_vars = model_objective.gen_obj_fun_variables(objective_function, obj_fun_decimal=False)

    if "matsui_constraint" in config_model:
        assert objective_target == "AT MOST", "Matsui constraints only support 'AT MOST' objective target."
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        GroupConstraintChoice = config_model["matsui_constraint"].get("GroupConstraintChoice", 1)
        GroupNumForChoice = config_model["matsui_constraint"].get("GroupNumForChoice", 1)
        if Round is None or best_obj is None:
            raise ValueError("Must provide 'Round' and 'best_obj' for Matsui strategy.")
        model_cons += model_constraints.gen_matsui_constraints_sat(Round, best_obj, obj_val, obj_fun_vars, GroupConstraintChoice, GroupNumForChoice)

    else:
        hw_list = [obj for row in obj_fun_vars for obj in row]
        model_cons += model_constraints.gen_predefined_constraints("sat", cons_type, hw_list, obj_val, encoding=encoding)

    model = write_sat_model(constraints=model_cons, filename=config_model.get("filename"))
    solutions = solving.solve_sat(config_model.get("filename"), model["variable_map"], config_solver)

    if isinstance(solutions, list) and len(solutions) > 0:
        for sol in solutions:
            round_values = model_objective.cal_round_obj_fun_values_from_solution(objective_function, sol)
            sol["rounds_obj_fun_values"] = round_values
            sol["obj_fun_value"] = sum(round_values)
    return solutions


# ------------------- CNF Generation and SAT Model Writing -------------------
def create_numerical_cnf(cnf): # Convert a given CNF formula into numerical CNF format. Return (number of variables, mapping of variables to numerical IDs, numerical CNF constraints)
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
        numerical_literals.extend(tuple(f'{"-" * lit_is_neg}{variable2number[literal[lit_is_neg:]]}' for lit_is_neg, literal in zip(lits_are_neg, literals)))
        numerical_clause = ' '.join(numerical_literals)
        numerical_cnf.append(numerical_clause)
    return len(variables), variable2number, numerical_cnf

def write_sat_model(constraints=[], filename="sat.cnf"): # Generate and write the SAT model.
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # === Step 1: Convert Constraints to Numerical CNF Format === #
    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints)

    # === Step 2: Prepare and write CNF file === #
    num_clause = len(constraints)

    with open(filename, "w") as f:
        f.write(f"p cnf {num_var} {num_clause}\n")
        for constraint in numerical_cnf:
            f.write(f"{constraint} 0\n")

    # === Step 3: Return metadata === #
    return {"variable_map": variable_map}
