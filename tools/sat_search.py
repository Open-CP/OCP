"""The interface for SAT-based modeling and solving for automated cryptanalysis.

Provides:

1. Generate SAT constraints from the objective target.
2. Generate DIMACS CNF models.
3. Call the SAT solver (PySAT, OR-Tools CP-SAT) to solve the model.
"""

import os

import tools.model_objective as model_objective
from tools.model_objective import cal_round_obj_fun_values_from_solution
from tools.model_constraints import gen_predefined_constraints, gen_matsui_constraints_sat
import solving.solving as solving


# Upper bound for the incrementing SAT objective search (INCREASING / ADAPTIVE strategies).
# The search stops once it reaches this objective value without finding a feasible solution.
SAT_OBJ_SEARCH_CAP = 10000


# ----------------------- Objective Target (SAT) ------------------------
def parse_objective_target(objective_target):
    """Parse a SAT objective-target string.

    Args:
        objective_target (str): One of ``"OPTIMAL"``, ``"EXISTENCE"``, or
            ``"AT MOST X"`` / ``"EXACTLY X"`` / ``"AT LEAST X"`` (``X`` a number).

    Returns:
        tuple: ``(strategy, value)`` where ``strategy`` is the target keyword and
        ``value`` is the parsed number (``None`` for ``"OPTIMAL"`` / ``"EXISTENCE"``).
    """
    if objective_target == "OPTIMAL" or objective_target == "EXISTENCE":
        return objective_target, None
    for keyword in ["AT MOST", "EXACTLY", "AT LEAST"]:
        if objective_target.startswith(keyword):
            try:
                value = float(objective_target.split()[-1])
                return keyword, value
            except ValueError:
                raise ValueError(f"Invalid format: '{objective_target}'. Expected '{keyword} X'.")
    raise ValueError(f"Unsupported objective_target: {objective_target}")


def decimal_objective_combinations(config_model, min_int_obj_value, max_obj_value):
    """Return decimal objective combinations for the configured S-box table."""
    Sbox = config_model.get("decimal_objective_function", {}).get("Sbox")
    table = config_model.get("decimal_objective_function", {}).get("table")
    if not Sbox or not table:
        raise ValueError("Missing Sbox or table information for decimal objective function search.")
    return model_objective.generate_obj_decimal_coms(Sbox, table, min_int_obj_value, max_obj_value)


def _sat_objective_encoding(config_model, cons_type):
    if cons_type == "SUM_AT_MOST":
        return config_model.get("atmost_encoding_sat", "SEQUENTIAL")
    if cons_type == "SUM_EXACTLY":
        return config_model.get("exact_encoding_sat", 1)
    if cons_type == "SUM_AT_LEAST":
        return config_model.get("atleast_encoding_sat", 1)
    return None


def gen_sat_constraints_from_objective_target(
    objective_function,
    config_model,
    cons_type,
    obj_val,
    obj_val_decimal=None,
):
    """Generate SAT constraints induced by an objective target."""
    encoding = _sat_objective_encoding(config_model, cons_type)
    if encoding is None:
        return []

    constraints = []
    if obj_val_decimal is not None:
        obj_fun_vars, obj_fun_vars_decimal = model_objective.gen_obj_fun_variables(
            objective_function,
            obj_fun_decimal=True,
        )
        if len(obj_val_decimal) != len(obj_fun_vars_decimal):
            raise ValueError("Length mismatch between objective function decimal variables and obj_val_decimal.")
        for i in range(len(obj_fun_vars_decimal)):
            hw_list = [obj for row in obj_fun_vars_decimal[i] for obj in row]
            constraints += gen_predefined_constraints(
                "sat",
                cons_type,
                hw_list,
                obj_val_decimal[i],
                encoding=encoding,
            )
    else:
        obj_fun_vars = model_objective.gen_obj_fun_variables(
            objective_function,
            obj_fun_decimal=False,
        )

    if "matsui_constraint" in config_model and obj_val > 0:
        print("[INFO] Applying Matsui constraints for SAT modeling.")
        if cons_type != "SUM_AT_MOST":
            raise ValueError("Matsui constraints only support 'AT MOST' objective target.")
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        GroupConstraintChoice = config_model["matsui_constraint"].get("GroupConstraintChoice", 1)
        GroupNumForChoice = config_model["matsui_constraint"].get("GroupNumForChoice", 1)
        if Round is None or best_obj is None:
            raise ValueError("[WARNING] Please provide 'Round' and 'best_obj' for Matsui strategy.")
        if obj_val >= best_obj[-1]:
            constraints += gen_matsui_constraints_sat(
                Round,
                best_obj,
                obj_val,
                obj_fun_vars,
                GroupConstraintChoice,
                GroupNumForChoice,
            )
        else:
            print(
                f"[WARNING] Skipping Matsui constraints since obj_val = {obj_val} < best_obj[-1] = {best_obj[-1]}."
            )
            hw_list = [obj for row in obj_fun_vars for obj in row]
            constraints += gen_predefined_constraints(
                "sat",
                cons_type,
                hw_list,
                obj_val,
                encoding=encoding,
            )
    else:
        hw_list = [obj for row in obj_fun_vars for obj in row]
        constraints += gen_predefined_constraints(
            "sat",
            cons_type,
            hw_list,
            obj_val,
            encoding=encoding,
        )
    return constraints


def _solve_with_objective_target(
    constraints,
    objective_function,
    config_model,
    config_solver,
    cons_type,
    obj_val,
    obj_val_decimal=None,
):
    obj_constraints = gen_sat_constraints_from_objective_target(
        objective_function,
        config_model,
        cons_type,
        obj_val,
        obj_val_decimal=obj_val_decimal,
    )
    return modeling_solving(constraints + obj_constraints, objective_function, config_model, config_solver)


def modeling_solving_sat(objective_target, constraints, objective_function, config_model, config_solver):
    strategy, value = parse_objective_target(objective_target)

    if strategy == "OPTIMAL":
        solutions = modeling_solving_optimal(constraints, objective_function, config_model, config_solver)
    elif strategy == "AT MOST":
        solutions = modeling_solving_at_most(constraints, objective_function, config_model, config_solver, value)
    elif strategy == "EXACTLY":
        solutions = modeling_solving_exactly(constraints, objective_function, config_model, config_solver, value)
    elif strategy == "AT LEAST":
        solutions = modeling_solving_at_least(constraints, objective_function, config_model, config_solver, value)
    elif strategy == "EXISTENCE":
        solutions = modeling_solving(constraints, objective_function, config_model, config_solver)
    else:
        raise ValueError(f"Invalid objective_target: {objective_target}")

    print("====== Modeling and Solving SAT Information ======")
    print(f"--- Found {len(solutions)} solution(s) ---")
    for key, val in {**config_model, **config_solver}.items():
        if key not in ["positions", "decimal_objective_function"]:
            print(f"--- {key} ---: {val}")
    return solutions


# ------------------------- Optimal Search Strategy --------------------------
def modeling_solving_optimal(constraints, objective_function, config_model, config_solver): # Find the optimal SAT solution.
    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if not decimal_objective_function:
        return modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver)
    return modeling_solving_optimal_decimalobj(constraints, objective_function, config_model, config_solver)


def modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver):
    print("[INFO] Search for the optimal solutions.")

    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0")
    try:
        obj_val = int(optimal_search_strategy_sat.split()[-1])
    except ValueError:
        raise ValueError(
            f"Invalid format: '{optimal_search_strategy_sat}'. Expected 'INCREASING FROM AT MOST X', "
            "'INCREASING FROM EXACTLY X', 'DECREASING FROM AT MOST X', or 'ADAPTIVE FROM AT MOST X'."
        )
    if optimal_search_strategy_sat.startswith("INCREASING FROM AT MOST"):
        cons_type, step, end_obj_value, mode = "SUM_AT_MOST", 1, SAT_OBJ_SEARCH_CAP, "INCREASING"
    elif optimal_search_strategy_sat.startswith("INCREASING FROM EXACTLY"):
        cons_type, step, end_obj_value, mode = "SUM_EXACTLY", 1, SAT_OBJ_SEARCH_CAP, "INCREASING"
    elif optimal_search_strategy_sat.startswith("DECREASING FROM AT MOST"):
        cons_type, step, end_obj_value, mode = "SUM_AT_MOST", -1, -1, "DECREASING"
    elif optimal_search_strategy_sat.startswith("ADAPTIVE FROM AT MOST"):
        cons_type, step, end_obj_value, mode = "SUM_AT_MOST", 1, SAT_OBJ_SEARCH_CAP, "ADAPTIVE"
    else:
        raise ValueError(f"Invalid optimal_search_strategy_sat: {optimal_search_strategy_sat}.")

    found_feasible = None
    solutions = None

    while obj_val != end_obj_value:
        print(f"[INFO] Current SAT objective value: {obj_val}")
        current_solutions = _solve_with_objective_target(
            constraints,
            objective_function,
            config_model,
            config_solver,
            cons_type,
            obj_val,
        )
        if isinstance(current_solutions, list) and len(current_solutions) > 0:
            for sol in current_solutions:
                sol["integer_obj_fun_value"] = obj_val
        if mode == "INCREASING" and current_solutions:
            return current_solutions
        elif mode == "DECREASING" and not current_solutions:
            if solutions is None:
                print(f"[INFO] No feasible solution found. Please set the strategy {optimal_search_strategy_sat} with an appropriate starting value.")
                return []
            return solutions
        elif mode == "ADAPTIVE":
            if current_solutions and found_feasible is None:
                found_feasible = True
                step = -1
                end_obj_value = -1
            elif (not current_solutions) and found_feasible is True:
                return solutions
            elif (not current_solutions) and found_feasible is None:
                found_feasible = False
            elif current_solutions and found_feasible is False:
                return current_solutions
        obj_val += step
        solutions = current_solutions
    return solutions


def modeling_solving_optimal_decimalobj(constraints, objective_function, config_model, config_solver):
    print("[INFO] Search for the optimal solutions with decimal objective function value.")

    # Step 1: Find the optimal solution with integer objective function value
    solutions = modeling_solving_optimal_intobj(constraints, objective_function, config_model, config_solver)

    # Step 2: Refine search for decimal weights
    if solutions is None or len(solutions) == 0:
        return []
    optimal_search_strategy_sat = config_model.get("optimal_search_strategy_sat", "INCREASING FROM AT MOST 0")
    cons_type = "SUM_EXACTLY" if "EXACTLY" in optimal_search_strategy_sat else "SUM_AT_MOST"
    max_obj_val = solutions[0]["obj_fun_value"] # The current objective function value is the upper bound
    int_obj_val = solutions[0]["integer_obj_fun_value"] # Start searching from the minimal integer objective function value
    print(f"[INFO] True objective function value = {max_obj_val} with integer value = {int_obj_val}")

    if max_obj_val == int_obj_val:
        return solutions

    obj_decimal_list = decimal_objective_combinations(config_model, int_obj_val, max_obj_val)
    for (true_obj, obj_integer, obj_decimal) in obj_decimal_list:
        if true_obj >= max_obj_val:
            continue
        print(f"[INFO] Trying decimal combination with true_obj = {true_obj}, int_obj = {obj_integer}, obj_decimal = {obj_decimal}")
        decimal_solutions = _solve_with_objective_target(
            constraints,
            objective_function,
            config_model,
            config_solver,
            cons_type,
            obj_integer,
            obj_val_decimal=obj_decimal,
        )
        if isinstance(decimal_solutions, list) and len(decimal_solutions) > 0:
            for sol in decimal_solutions:
                max_obj_val = min(max_obj_val, sol["obj_fun_value"])
                sol["integer_obj_fun_value"] = int_obj_val
            solutions = decimal_solutions
            break
    return solutions


# ------------------------- AT MOST Search Strategy --------------------------
def modeling_solving_at_most(constraints, objective_function, config_model, config_solver, at_most_value):
    print(f"[INFO] Search for solutions with the objective function value <= {at_most_value}.")

    # Search for solutions with integer objective function values <= int(at_most_value)
    solutions = _solve_with_objective_target(
        constraints,
        objective_function,
        config_model,
        config_solver,
        "SUM_AT_MOST",
        int(at_most_value),
    )

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if decimal_objective_function and isinstance(solutions, list) and len(solutions) > 0: # For ciphers with S-boxes having decimal weights, further filter and search for one solution with true objective value <= max_val
        for sol in solutions:
            true_obj = sol.get("obj_fun_value")
            if true_obj is None:
                print("[WARNING] Solution does not contain 'obj_fun_value'. Skipping.")
                continue
            if true_obj <= at_most_value:
                return [sol]
        # If no solution meets the true objective value <= atmost_value, further search
        int_obj_val = int(at_most_value)
        while solutions:
            obj_decimal_list = decimal_objective_combinations(config_model, int_obj_val, at_most_value)
            for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list):
                if obj_integer > int_obj_val:
                    continue
                print(f"[INFO] Trying decimal combination with true_obj = {true_obj}, int_obj = {obj_integer}, obj_decimal = {obj_decimal}")
                decimal_solutions = _solve_with_objective_target(
                    constraints,
                    objective_function,
                    config_model,
                    config_solver,
                    "SUM_AT_MOST",
                    int_obj_val,
                    obj_val_decimal=obj_decimal,
                )
                if isinstance(decimal_solutions, list) and len(decimal_solutions) > 0: # Support searching only a subset of solutions in the multiple-solution setting. TO DO.
                    return decimal_solutions
            int_obj_val -= 1
            solutions = _solve_with_objective_target(
                constraints,
                objective_function,
                config_model,
                config_solver,
                "SUM_AT_MOST",
                int_obj_val,
            )
    return solutions


# ------------------------- EXACTLY Search Strategy --------------------------
def modeling_solving_exactly(constraints, objective_function, config_model, config_solver, exactly_value):
    print(f"[INFO] Search for solutions with the objective function value = {exactly_value}")

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if not decimal_objective_function:
        return _solve_with_objective_target(
            constraints,
            objective_function,
            config_model,
            config_solver,
            "SUM_EXACTLY",
            int(exactly_value),
        )

    EPS = 0.001  # Tolerance for floating-point comparison
    obj_decimal_list = decimal_objective_combinations(config_model, -1, exactly_value)
    for (true_obj, obj_integer, obj_decimal) in reversed(obj_decimal_list):
        if abs(true_obj - exactly_value) < EPS: # Allow a small tolerance for floating-point comparison
            print(f"[INFO] Trying decimal combination with true_obj = {true_obj}, int_obj = {obj_integer}, obj_decimal = {obj_decimal}")
            solutions = _solve_with_objective_target(
                constraints,
                objective_function,
                config_model,
                config_solver,
                "SUM_EXACTLY",
                obj_integer,
                obj_val_decimal=obj_decimal,
            )
            if isinstance(solutions, list) and len(solutions) > 0:
                return solutions
    return []


# ------------------------- AT LEAST Search Strategy -------------------------
def modeling_solving_at_least(constraints, objective_function, config_model, config_solver, at_least_value):
    print(f"[INFO] Search for solutions with objective function value >= {at_least_value}")

    # Search for solutions with integer objective function values >= int(at_least_value)
    solutions = _solve_with_objective_target(
        constraints,
        objective_function,
        config_model,
        config_solver,
        "SUM_AT_LEAST",
        int(at_least_value),
    )

    decimal_objective_function = config_model.get("decimal_objective_function", False)
    if decimal_objective_function:
        if solutions:
            for sol in solutions:
                true_obj = sol.get("obj_fun_value")
                if true_obj is None:
                    print("[WARNING] Solution does not contain 'obj_fun_value'. Skipping.")
                    continue
                if true_obj >= at_least_value:
                    return [sol]
        print("[INFO] No solution found. Need to search further.") # TO DO
        return []
    return solutions


# Core function for modeling and solving SAT.
def modeling_solving(constraints, objective_function, config_model, config_solver):
    print("[INFO] Modeling and solving SAT.")
    model = write_sat_model(constraints=constraints, filename=config_model.get("filename"))
    # Normalize to a list: solve_sat returns None when the solver backend is unavailable, and
    # every caller/strategy treats "no solution" as a (possibly empty) list. This keeps None from
    # leaking through the search strategies into len(solutions).
    solutions = solving.solve_sat(config_model.get("filename"), model["variable_map"], config_solver) or []

    if len(solutions) > 0:
        for sol in solutions:
            round_values = cal_round_obj_fun_values_from_solution(objective_function, sol)
            sol["rounds_obj_fun_values"] = round_values
            sol["obj_fun_value"] = sum(round_values)
    return solutions


def create_numerical_cnf(cnf):
    """Convert symbolic CNF clauses to DIMACS-style integer clauses."""
    variables = sorted({literal.lstrip("-") for clause in cnf for literal in clause.split()})
    variable2number = {variable: i + 1 for (i, variable) in enumerate(variables)}

    numerical_cnf = []
    for clause in cnf:
        literals = clause.split()
        numerical_literals = (
            f'{"-" if literal.startswith("-") else ""}{variable2number[literal.lstrip("-")]}'
            for literal in literals
        )
        numerical_cnf.append(' '.join(numerical_literals))
    return len(variables), variable2number, numerical_cnf


def write_sat_model(constraints=None, filename="sat.cnf"):
    """Generate and write a DIMACS CNF SAT model."""
    constraints = constraints or []
    filename = filename or "sat.cnf"

    empty_clause_idx = [i for i, clause in enumerate(constraints) if not clause.split()]
    if empty_clause_idx:
        print(
            f"[WARNING] write_sat_model: skipping {len(empty_clause_idx)} empty clause(s) at index "
            f"{empty_clause_idx[:10]} (a malformed/degenerate constraint reached the model writer; "
            "keeping it would make the SAT model trivially UNSAT)."
        )
        constraints = [clause for clause in constraints if clause.split()]

    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints)
    num_clause = len(constraints)

    with open(filename, "w") as f:
        f.write(f"p cnf {num_var} {num_clause}\n")
        for constraint in numerical_cnf:
            f.write(f"{constraint} 0\n")

    return {"variable_map": variable_map}
