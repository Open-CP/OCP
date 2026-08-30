"""The interface for MILP-based modeling and solving for automated cryptanalysis.

Provides:

1. Generate MILP constraints from the objective target.
2. Generate standard LP-format models.
3. Call the MILP solver (Gurobi, SCIP, etc.) to solve the model.
"""

import os

import tools.model_objective as model_objective
from tools.model_constraints import gen_matsui_constraints_milp, gen_predefined_constraints
import solving.solving as solving


# ----------------------- Objective Target (MILP) ------------------------
def parse_objective_target(objective_target):
    """Parse a MILP objective-target string.

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


def gen_milp_constraints_from_objective_target(objective_target):
    """Generate MILP constraints from an objective target string.

    In MILP models the variable ``obj`` represents the objective function.
    """
    strategy, value = parse_objective_target(objective_target)
    if strategy == "AT MOST":
        return gen_predefined_constraints("milp", "AT_MOST", ["obj"], value)
    if strategy == "EXACTLY":
        return gen_predefined_constraints("milp", "EXACTLY", ["obj"], value)
    if strategy == "AT LEAST":
        return gen_predefined_constraints("milp", "AT_LEAST", ["obj"], value)
    return []


# -------------------------- MILP Model Writing ---------------------------
# ------------------------ Modeling and Solving Interface ----------------------
def _milp_objective_function(objective_target, objective_function):
    if objective_target == "EXISTENCE":
        return None
    return objective_function[:]


def _build_milp_model_constraints(objective_target, constraints, objective_function, config_model):
    model_cons = list(constraints or [])
    model_cons.extend(gen_milp_constraints_from_objective_target(objective_target))

    matsui_config = config_model.get("matsui_constraint")
    if matsui_config is None:
        return model_cons

    Round = matsui_config.get("Round")
    best_obj = matsui_config.get("best_obj")
    cons_type = matsui_config.get("matsui_milp_cons_type", "ALL")
    if Round is None or best_obj is None or len(best_obj) != (Round - 1):
        raise ValueError("Must provide correct 'Round' and 'best_obj' for Matsui strategy.")
    model_cons.extend(gen_matsui_constraints_milp(Round, best_obj, objective_function, cons_type))
    return model_cons


def _attach_milp_solution_objectives(solutions, objective_function):
    for sol in solutions:
        round_values = model_objective.cal_round_obj_fun_values_from_solution(objective_function, sol)
        sol["rounds_obj_fun_values"] = round_values
        if "obj_fun_value" not in sol or sol["obj_fun_value"] == 0:
            sol["obj_fun_value"] = sum(round_values)


def write_milp_model(constraints, obj_fun=None, filename="milp.lp"):
    """Generate and write a standard LP-format MILP model."""
    filename = filename or "milp.lp"
    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(filename, "w") as f:
        # === Step 1: Define the MILP Model Structure === #
        # If an objective function (obj_fun) is provided, write a symbolic objective name "obj", which will be defined later in the constraint section. Otherwise, write "Minimize 0" to indicate a feasibility-only model.
        if obj_fun:
            f.write("Minimize\n obj\nSubject To\n")
        else:
            f.write("Minimize\n 0\nSubject To\n")

        # === Step 2: Process Constraints === #
        bin_vars, in_vars = set(), set()
        for constraint in constraints:
            if "Binary" in constraint:
                parts = constraint.split('Binary\n')
                if parts[0].strip():
                    f.write(parts[0].strip() + "\n")
                for segment in parts[1:]:
                    seg = segment.strip()
                    if seg:
                        bin_vars.update(seg.split())
            elif "Integer" in constraint:
                parts = constraint.split('Integer\n')
                if parts[0].strip():
                    f.write(parts[0].strip() + "\n")
                for segment in parts[1:]:
                    seg = segment.strip()
                    if seg:
                        in_vars.update(seg.split())
            else:
                f.write(constraint if constraint.endswith('\n') else constraint + '\n')

        # === Step 3: Define the Objective Function === #
        if obj_fun:
            if isinstance(obj_fun[0], list):
                obj_terms = [obj for row in obj_fun for obj in row]
            else:
                obj_terms = obj_fun
            f.write(" + ".join(obj_terms) + " - obj = 0" + "\n")

        # === Step 4: Declare Binary and Integer Variables === #
        if bin_vars:
            f.write("Binary\n" + " ".join(sorted(bin_vars)) + "\n")
        if in_vars:
            f.write("Integer\n" + " ".join(sorted(in_vars)) + "\n")

        f.write("End\n")
    return None


def modeling_solving_milp(objective_target, constraints, objective_function, config_model, config_solver): # Construct and solve the MILP model.
    # Step 1. Generate model constraints.
    model_cons = _build_milp_model_constraints(objective_target, constraints, objective_function, config_model)

    # Step 2. Generate the standard MILP model.
    obj_fun = _milp_objective_function(objective_target, objective_function)
    write_milp_model(model_cons, obj_fun, config_model.get("filename"))

    # Step 3. Solve the MILP model.
    solutions = solving.solve_milp(config_model.get("filename"), config_solver)
    _attach_milp_solution_objectives(solutions, objective_function)

    # Step 4. Print modeling and solving information.
    print("====== Modeling and Solving MILP Information ======")
    print(f"--- Found {len(solutions)} solution(s) ---")
    for key, value in {**config_model, **config_solver}.items():
        if key not in ["positions"]:
            print(f"--- {key} ---: {value}")
    return solutions
