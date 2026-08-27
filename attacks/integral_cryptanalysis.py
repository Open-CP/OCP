"""The interface for integral attacks.

Provides:

1. search for bit-based two-subset division-property distinguishers
"""

from pathlib import Path

from attacks.attack_trace import IntegralDistinguisher
from tools.model_configuration import fill_functions_rounds_layers_positions, gen_round_model_constraint_obj_fun
from tools.model_constraints import gen_predefined_constraints
from tools.milp_search import modeling_solving_milp
from tools.paths import get_files_dir


# ---------------------- Model and Solver Configuration ----------------------
def _parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver):
    """Fill in default model/solver configuration and return ``(config_model, config_solver)``."""
    # ===== Set Default config_model and config_solver =====
    config_model = config_model or {}
    config_solver = config_solver or {}

    # Set "model_type", currently only MILP is supported for two-subset integral search.
    config_model["model_type"] = config_model.get("model_type", "milp").lower()
    if config_model["model_type"] != "milp":
        raise ValueError(f"Invalid model_type: {config_model['model_type']}. INTEGRAL_TWOSUBSET currently supports only 'milp'.")

    # Set "functions", "rounds", "layers", "positions" for modeling.
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)

    # Set the model "filename".
    config_model.setdefault("filename", str(get_files_dir() / f"{cipher.nbr_rounds}round_{cipher.name}_{goal}_{objective_target}_milp_model.lp"))

    # Set "solver" and "solution_number" for solving the model.
    config_solver.setdefault("solver", "DEFAULT")
    config_solver.setdefault("solution_number", 1)
    return config_model, config_solver


# -------------------- Predefined Additional Constraints --------------------
def _expand_var_ids(var):
    """Return the variable's ID, expanded to per-bit IDs when its width > 1."""
    if var.bitsize > 1:
        return [f"{var.ID}_{i}" for i in range(var.bitsize)]
    return [var.ID]


def _get_initial_state_var_ids(cipher, function="PERMUTATION"):
    """Return the per-bit variable IDs of the first-round input state of ``function``."""
    func = cipher.functions[function]
    return [var_id for var in func.vars[1][0][:func.nbr_words] for var_id in _expand_var_ids(var)]


def _get_final_state_var_ids(cipher, function="PERMUTATION"):
    """Return the per-bit variable IDs of the last-round output state of ``function``."""
    func = cipher.functions[function]
    return [var_id for var in func.vars[func.nbr_rounds][func.nbr_layers][:func.nbr_words] for var_id in _expand_var_ids(var)]


def _normalize_bit_positions(bit_positions, bit_size):
    """Return the sorted, de-duplicated bit positions, validating each is in ``[0, bit_size)``."""
    if bit_positions is None:
        return []
    normalized = sorted(set(bit_positions))
    for bit in normalized:
        if bit < 0 or bit >= bit_size:
            raise ValueError(f"Invalid bit position {bit}. Expected 0 <= bit < {bit_size}.")
    return normalized


def _gen_initial_two_subset_constraints(cipher, constant_bits=None, active_bits=None, function="PERMUTATION"):
    """Return the two-subset initial-state constraints (constant bits fixed to 0, active bits to 1)."""
    initial_var_ids = _get_initial_state_var_ids(cipher, function=function)
    bit_size = len(initial_var_ids)
    if constant_bits is None:
        raise ValueError("constant_bits must be explicitly provided for TWO_SUBSET_INIT.")
    constant_bits = _normalize_bit_positions(constant_bits, bit_size)
    if active_bits is None:
        active_bits = [bit for bit in range(bit_size) if bit not in constant_bits]
    active_bits = _normalize_bit_positions(active_bits, bit_size)

    constant_var_ids = [var_id for bit, var_id in enumerate(initial_var_ids) if bit in constant_bits]
    active_var_ids = [var_id for bit, var_id in enumerate(initial_var_ids) if bit in active_bits]

    constraints = []
    constraints += gen_predefined_constraints("milp", "EXACTLY", constant_var_ids, 0, bitwise=False)
    constraints += gen_predefined_constraints("milp", "EXACTLY", active_var_ids, 1, bitwise=False)
    constraints.append("Binary\n" + " ".join(initial_var_ids))
    return constraints


# ---------------- Two-Subset Balanced Bit Search Utilities -----------------
def _build_final_objective(final_var_ids):
    """Return the objective maximizing the sum of the final-state variables."""
    return [[" + ".join(final_var_ids)]]


def _extract_unit_final_var(solution, final_var_ids):
    """Return the first final-state variable set to 1 in ``solution``, or None."""
    for var_id in final_var_ids:
        if int(round(solution.get(var_id, 0))) == 1:
            return var_id
    return None


def _gen_ban_final_var_constraint(var_id):
    """Return a constraint forcing ``var_id`` to 0."""
    return f"{var_id} = 0"


def _final_var_id_to_bit_position(var_id):
    """Return the trailing bit index of a final-state variable ID (e.g. ``v_3_0_5`` -> 5)."""
    bit_position = str(var_id).rsplit("_", 1)[-1]
    if not bit_position.isdigit():
        raise ValueError(f"Invalid final state variable ID: {var_id}.")
    return int(bit_position)


def _final_var_ids_to_bit_positions(var_ids):
    """Map a list of final-state variable IDs to their bit positions."""
    return [_final_var_id_to_bit_position(var_id) for var_id in var_ids]


def _search_balanced_bits(base_constraints, final_var_ids, config_model, config_solver):
    """Iteratively identify the balanced output bits via the two-subset MILP search.

    Returns:
        dict: ``{"status", "banned_bits", "balanced_bits"}``.
    """
    banned_var_ids = []
    status = "unknown"
    objective = _build_final_objective(final_var_ids)

    while len(banned_var_ids) < len(final_var_ids):
        constraints = list(base_constraints) + [_gen_ban_final_var_constraint(var_id) for var_id in banned_var_ids]
        solutions = modeling_solving_milp("OPTIMAL", constraints, objective, config_model, config_solver)
        if not solutions:
            status = "found"
            break

        solution = solutions[0]
        obj_value = solution.get("obj_fun_value")
        if obj_value is None:
            obj_value = sum(solution.get(var_id, 0) for var_id in final_var_ids)
        if obj_value > 1:
            status = "found"
            break

        var_id = _extract_unit_final_var(solution, final_var_ids)
        if var_id is None:
            status = "found"
            break
        if var_id not in banned_var_ids:
            banned_var_ids.append(var_id)

    if status == "unknown":
        status = "not_found"
    return {
        "status": status,
        "banned_bits": [i for i, var_id in enumerate(final_var_ids) if var_id in banned_var_ids],
        "balanced_bits": [i for i, var_id in enumerate(final_var_ids) if var_id not in banned_var_ids],
    }


# ------------------------ Integral Distinguisher Search ---------------------
def search_integral_distinguisher(cipher, goal="INTEGRAL_TWOSUBSET", constraints=None, objective_target="EXISTENCE", show_mode=0, config_model=None, config_solver=None):
    """Search for integral distinguishers of the given cipher (bit-based two-subset division property).

    Args:
        cipher: The cipher object to analyze.
        goal (str): Cryptanalysis goal; currently only ``"INTEGRAL_TWOSUBSET"``.
        constraints (list, optional): Extra model constraints. ``"TWO_SUBSET_INIT"``
            auto-adds the two-subset initial-state constraints; entries may also be
            explicit MILP variable constraints (e.g. ``"v_1_0_0 = 1"``) or any
            user-defined constraint. Defaults to an empty list.
        objective_target (str): Currently only ``"EXISTENCE"``.
        show_mode (int): Result-printing detail level (0-3).
        config_model (dict, optional): Advanced modeling options; see
            ``_parse_and_set_configs``.
        config_solver (dict, optional): Advanced solver options; see
            ``_parse_and_set_configs``.

    Returns:
        list: The integral distinguisher objects found.
    """
    constraints = constraints or []

    if goal != "INTEGRAL_TWOSUBSET":
        raise ValueError(f"Invalid goal: {goal}. Expected 'INTEGRAL_TWOSUBSET'.")
    if not isinstance(constraints, list):
        raise ValueError(f"Invalid constraints: {constraints}. Expected a list of strings.")
    if objective_target != "EXISTENCE":
        raise ValueError("INTEGRAL_TWOSUBSET currently supports objective_target='EXISTENCE'.")
    if show_mode not in (0, 1, 2, 3):
        raise ValueError(f"Invalid show_mode: {show_mode}. Expected one of [0, 1, 2, 3].")
    if not (isinstance(config_model, dict) or config_model is None):
        raise ValueError(f"Invalid config_model: {config_model}. Expected a dictionary or None.")
    if not (isinstance(config_solver, dict) or config_solver is None):
        raise ValueError(f"Invalid config_solver: {config_solver}. Expected a dictionary or None.")

    # Generate a new cipher instance with added copy layer after each operator.
    cipher.add_copy_operators()

    # Step 1. Parse and set model and solver configurations.
    config_model, config_solver = _parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver)
    model_type = config_model["model_type"]

    # Step 2. Generate round constraints and objective function for the cipher.
    round_constraints, obj_fun = gen_round_model_constraint_obj_fun(cipher, goal, config_model)

    # Step 3. Process additional constraints.
    model_cons = []
    for cons in constraints:
        if cons == "TWO_SUBSET_INIT": # Deal with specific additional constraints.
            model_cons += _gen_initial_two_subset_constraints(
                cipher,
                constant_bits=config_model.get("constant_bits"),
                active_bits=config_model.get("active_bits"),
            )
        else:
            model_cons += [cons]
    model_cons += round_constraints

    # Step 4. Search balanced bits or solve the model directly.
    if "TWO_SUBSET_INIT" in constraints:
        final_var_ids = _get_final_state_var_ids(cipher, function=config_model.get("state_function", "PERMUTATION"))
        search_result = _search_balanced_bits(model_cons, final_var_ids, config_model, config_solver)
        if search_result["status"] != "found":
            return []
        return _extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, [search_result])

    solutions = modeling_solving_milp(objective_target, model_cons, obj_fun, config_model, config_solver)
    if isinstance(solutions, list):
        return _extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, solutions)

    raise ValueError(f"Solving did not return a list of solutions (got {type(solutions).__name__}); check the solver configuration.")


# ---------------- Distinguisher Extraction and Visualization ----------------
def _add_index(path, i):
    """Insert ``_i`` before the file extension: ``a/b.json`` -> ``a/b_i.json``."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}_{i}{p.suffix}"))


def _extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, solutions):
    """Build, print, and save an :class:`IntegralDistinguisher` for each solution.

    Returns:
        list: The integral distinguisher objects.
    """
    distinguishers = []
    for i, sol in enumerate(solutions):
        data = {"cipher": f"{cipher.nbr_rounds}_round_{cipher.name}",
                "rounds": config_model["rounds"],
                "goal": goal,
                "status": sol.get("status", "found"),
                "balanced_bits": sol.get("balanced_bits", []),
                "banned_bits": sol.get("banned_bits", []),
                "config_model": config_model,
                "config_solver": config_solver}
        distinguisher = IntegralDistinguisher(data, solution_trace=sol)
        if i > 0:
            print(f"[INFO] Saving distinguisher #{i+1}.")
            distinguisher.json_filename = _add_index(distinguisher.json_filename, i)
            distinguisher.txt_filename = _add_index(distinguisher.txt_filename, i)
        distinguisher.print_distinguisher()
        distinguisher.save_json()
        distinguisher.save_txt()
        distinguishers.append(distinguisher)
    return distinguishers
