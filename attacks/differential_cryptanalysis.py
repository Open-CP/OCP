"""The interface for differential attacks.

Provides:

1. search for differential trails
"""

import time
from math import log2
from attacks.attack_trace import DifferentialTrail, extract_and_format_trails
from tools.model_constraints import gen_input_non_zero_constraints, gen_required_fixed_boundary_constraints
from tools.model_configuration import parse_and_set_configs, gen_round_model_constraint_obj_fun
from tools.model_objective import has_Sbox_with_decimal_weights, detect_Sbox
from tools.milp_search import modeling_solving_milp
from tools.sat_search import modeling_solving_sat


# ---------------------- Model and Solver Configuration ----------------------
def _validate_request(goal, constraints, objective_target, show_mode, config_model, config_solver):
    """Validate the public ``search_diff_trail`` arguments, raising ``ValueError`` on any invalid one."""
    allowed_goals = ["DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB", "TRUNCATEDDIFF_SBOXCOUNT"]
    allowed_objective_targets = ["OPTIMAL", "AT MOST", "EXACTLY", "AT LEAST", "EXISTENCE"]
    allowed_show = [0, 1, 2, 3]
    if not any(goal.startswith(prefix) for prefix in allowed_goals):
        raise ValueError(f"Invalid goal: {goal}. Expected one of {allowed_goals}.")
    if not isinstance(constraints, list) or any(not isinstance(c, str) for c in constraints):
        raise ValueError(f"Invalid constraints: {constraints}. Expected a list of strings.")
    if not any(objective_target.startswith(prefix) for prefix in allowed_objective_targets):
        raise ValueError(f"Invalid objective_target: {objective_target}. Expected one of {allowed_objective_targets}.")
    if show_mode not in allowed_show:
        raise ValueError(f"Invalid show_mode: {show_mode}. Expected one of {allowed_show}.")
    if not (isinstance(config_model, dict) or config_model is None):
        raise ValueError(f"Invalid config_model: {config_model}. Expected a dictionary or None.")
    if not (isinstance(config_solver, dict) or config_solver is None):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")


def _parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver):
    """Apply default model/solver configuration and return ``(config_model, config_solver)``."""
    config_model, config_solver = parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver)
    if goal == "DIFFERENTIAL_PROB": # Enumerate a maximum of 1000000 trails by default when searching for a differential's probability.
        config_solver.setdefault("solution_number", 1000000)
    return config_model, config_solver


# ------------------------ Differential Trail Search -------------------------
def search_diff_trail(
    cipher,
    goal="DIFFERENTIALPATH_PROB",
    constraints=None,
    objective_target="OPTIMAL",
    show_mode=0,
    config_model=None,
    config_solver=None,
):
    """Search for differential trails of the specified cipher.

    Args:
        cipher: The cipher object to analyze.
        goal (str): Cryptanalysis goal, one of ``"DIFFERENTIAL_SBOXCOUNT"``,
            ``"DIFFERENTIALPATH_PROB"``, ``"DIFFERENTIAL_PROB"``,
            ``"TRUNCATEDDIFF_SBOXCOUNT"``.
        constraints (list, optional): Extra model constraints. ``["INPUT_NOT_ZERO"]``
            (the default) auto-adds an input-non-zero constraint; entries may also be
            explicit variable constraints (e.g. ``"v_1_0_0 = 1"`` for MILP,
            ``"v_1_0_0"`` / ``"-v_2_1_0"`` for SAT) or any user-defined constraint.
        objective_target (str): The target for the objective function, which can be:
            - 'OPTIMAL': Find the optimal solution.
            - 'AT MOST X': Find a solution with an objective value at most X.
            - 'EXACTLY X': Find a solution with an objective value exactly X.
            - 'AT LEAST X': Find a solution with an objective value at least X.
            - 'EXISTENCE': Find any feasible solution.
        show_mode (int): Result-printing detail level (0-3).
        config_model (dict, optional): Advanced modeling options; see
            ``_parse_and_set_configs``.
        config_solver (dict, optional): Advanced solver options; see
            ``_parse_and_set_configs``.

    Returns:
        list: The differential trail objects found.
    """

    t_start = time.time()  # Start of the whole search (for total_time).
    if constraints is None:
        constraints = ["INPUT_NOT_ZERO"]
    _validate_request(goal, constraints, objective_target, show_mode, config_model, config_solver)
    if goal == "DIFFERENTIAL_PROB" and objective_target != "EXISTENCE":
        print(f"[WARNING] goal='DIFFERENTIAL_PROB' supports only objective_target='EXISTENCE'; "
              f"overriding '{objective_target}' -> 'EXISTENCE'.")
        objective_target = "EXISTENCE"

    # Step 1. Parse and set model and solver configurations.
    config_model, config_solver = _parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver)
    model_type = config_model["model_type"]
    config_solver["solving_time"] = []  # Per-call solve times; pure-solver total = sum(solving_time).

    # Step 2. Generate round constraints and objective function for the cipher.
    round_constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, goal, config_model)

    # Step 3. Process additional constraints.
    bitwise = "TRUNCATEDDIFF" not in goal
    model_cons = []
    for cons in constraints:
        if cons == "INPUT_NOT_ZERO":  # Expand the symbolic input-non-zero marker.
            model_cons += gen_input_non_zero_constraints(cipher, config_model, bitwise)
        else:
            model_cons += [cons]
    model_cons.extend(round_constraints)

    # For the goal of searching for a differential, fix the input and output differences.
    if goal == "DIFFERENTIAL_PROB":
        model_cons.extend(
            gen_required_fixed_boundary_constraints(
                cipher,
                config_model.get("input_diff"),
                config_model.get("output_diff"),
                model_type,
            )
        )

    # Time 1: model generation (entry through Step 3: validation + config parsing + model building).
    t_step4 = time.time()
    config_solver["generation_time"] = round(t_step4 - t_start, 2)

    # Step 4: Modeling and Solving.
    if model_type == "milp":
        solutions = modeling_solving_milp(objective_target, model_cons, obj_fun, config_model, config_solver)

    elif model_type == "sat":
        if goal in ["DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB"] and has_Sbox_with_decimal_weights(cipher, goal):
            config_model["decimal_objective_function"] = {}
            Sbox = detect_Sbox(cipher)
            config_model["decimal_objective_function"]["Sbox"] = Sbox
            config_model["decimal_objective_function"]["table"] = Sbox.computeDDT()

        solutions = modeling_solving_sat(objective_target, model_cons, obj_fun, config_model, config_solver)

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")

    # Time 2: modeling + solving (Step 4: writing the model file and solving it, end-to-end).
    config_solver["modeling_solving_time"] = round(time.time() - t_step4, 2)

    # Step 5: Extract and Visualize Trails from Solutions.
    if isinstance(solutions, list):
        trails = _extract_and_format_diff_trails(cipher, goal, config_model, config_solver, show_mode, solutions)
        # Total time of the whole search (Step 1-5, including post-processing).
        config_solver["total_time"] = round(time.time() - t_start, 2)
        print(f"--- [TIME] generation_time ---: {config_solver['generation_time']} s  (config setup + constraint/objective generation)")
        print(f"--- [TIME] modeling_solving_time ---: {config_solver['modeling_solving_time']} s  (write model + solve, of which pure solving = {round(sum(config_solver['solving_time']), 2)} s)")
        print(f"--- [TIME] total_time ---: {config_solver['total_time']} s  (whole search, incl. post-processing)")
        return trails
    raise ValueError(f"Solving did not return a list of solutions (got {type(solutions).__name__}); check the solver configuration.")


# -------------------- Trail Extraction and Visualization --------------------
def _extract_and_format_diff_trails(cipher, goal, config_model, config_solver, show_mode, solutions):
    trails = extract_and_format_trails(cipher, goal, config_model, config_solver, show_mode, solutions, DifferentialTrail, "TRUNCATEDDIFF", "diff_weight", "rounds_diff_weight")
    # Aggregate probability of a differential = sum of trail probabilities = sum 2^(-w).
    if trails and goal == "DIFFERENTIAL_PROB":
        pr = sum(2 ** (-t.data["diff_weight"]) for t in trails if t.data["diff_weight"] is not None)
        print(f"[INFO] Total probability of all {len(trails)} found trails: "
            f"2^{log2(pr) if pr > 0 else 'undefined'}")
    return trails
