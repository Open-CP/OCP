import sys
import os
import copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] # milp_search.py -> tools -> <ROOT>
sys.path.insert(0, str(ROOT))

import tools.model_constraints as model_constraints
import tools.model_objective as model_objective
import solving.solving as solving


# **************************************************************************** #
# This module provides a interface for MILP-based modeling and solving for automated cryptanalysis, including:
# 1. Generate MILP constraints from the objective target.
# 2. Generate standard LP-format models.
# 3. Call the MILP solver (Gurobi, SCIP, etc.) to solve the model.
# **************************************************************************** #


# ----------------- Constraint Generation from Objective Target ----------------
def gen_milp_constraints_from_objective_target(objective_target): # Generate constraints based on the objective target. In MILP models, the variable 'obj' is used to represent the objective function.
    if objective_target.startswith("AT MOST"):
        try:
            max_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT MOST X'.")
        constraints = model_constraints.gen_predefined_constraints("milp", "AT_MOST", ["obj"], max_val) # Generate the constraint for the objective function value <= atmost_val
    elif objective_target.startswith("EXACTLY"):
        try:
            exact_val = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'EXACTLY X'.")
        constraints = model_constraints.gen_predefined_constraints("milp", "EXACTLY", ["obj"], exact_val) # Generate the constraint for the objective function value = exact_val.
    elif objective_target.startswith("AT LEAST"):
        try:
            atleast_value = float(objective_target.split()[-1])
        except ValueError:
            raise ValueError(f"Invalid format: '{objective_target}'. Expected 'AT LEAST X'.")
        constraints = model_constraints.gen_predefined_constraints("milp", "AT_LEAST", ["obj"], atleast_value) # Generate the constraint for the objective function value >= atleast_value.
    else:
        constraints = []
    return constraints


# -------------------------- MILP Model Writing ---------------------------
def write_milp_model(constraints, obj_fun=None, filename="milp.lp"): # Generate and write the MILP model in standard .lp format, based on the given constraints and objective function.
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


# ------------------------ Modeling and Solving Interface ----------------------
def modeling_solving_milp(objective_target, constraints, objective_function, config_model, config_solver): # Construct and solve the MILP model.
    # Step 1. Generate model constraints
    model_cons = copy.deepcopy(constraints) or []
    model_cons += gen_milp_constraints_from_objective_target(objective_target)

    # Step 2: Add Matsui acceleration constraints ---
    if "matsui_constraint" in config_model:  # Arguments for Matsui branch-and-bound constraints. Example: config_model["matsui_constraint"] = {"Round": 2, "best_obj": [1], "matsui_milp_cons_type": "ALL"}.
        Round = config_model.get("matsui_constraint").get("Round")
        best_obj = config_model.get("matsui_constraint").get("best_obj")
        cons_type = config_model["matsui_constraint"].get("matsui_milp_cons_type", "ALL")
        if Round is None or best_obj is None or len(best_obj) != (Round-1):
            raise ValueError("Must provide correct 'Round' and 'best_obj' for Matsui strategy.")
        model_cons += model_constraints.gen_matsui_constraints_milp(Round, best_obj, objective_function, cons_type)

    # Step 3. Generate the standard MILP model.
    write_milp_model(model_cons, objective_function, config_model.get("filename"))

    # Step 4. Solve the MILP model.
    solutions = solving.solve_milp(config_model.get("filename"), config_solver)
    for sol in solutions:
        sol["rounds_obj_fun_values"] = model_objective.cal_round_obj_fun_values_from_solution(objective_function, sol)
        if "obj_fun_value" not in sol: sol["obj_fun_value"] = sum(sol["rounds_obj_fun_values"])

    # Step 5. Print modeling and solving information.
    print("====== Modeling and Solving MILP Information ======")
    print(f"--- Found {len(solutions)} solution(s) ---")
    for key, value in {**config_model, **config_solver}.items():
        print(f"--- {key} ---: {value}")
    return solutions
