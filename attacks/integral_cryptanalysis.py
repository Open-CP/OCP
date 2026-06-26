from pathlib import Path

from attacks.attack_trace import IntegralDistinguisher
import tools.model_constraints as model_constraints
import tools.milp_search as milp_search

ROOT = Path(__file__).resolve().parents[1] # integral_cryptanalysis.py -> attacks -> <ROOT>
FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


# **************************************************************************** #
# This module is the interface for integral attacks, including:
# 1. search bit-based two-subset division property distinguishers
# **************************************************************************** #


# ---------------------- Model and Solver Configuration ----------------------
def parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver): # Parse input parameters and apply default values for model and solver configurations.
    # ===== Set Default config_model and config_solver =====
    config_model = config_model or {}
    config_solver = config_solver or {}

    # Set "model_type", currently only MILP is supported for two-subset integral search.
    config_model["model_type"] = config_model.get("model_type", "milp").lower()
    if config_model["model_type"] != "milp":
        raise ValueError(f"Invalid model_type: {config_model['model_type']}. INTEGRAL_TWOSUBSET currently supports only 'milp'.")

    # Set "functions", "rounds", "layers", "positions" for modeling.
    functions, rounds, layers, positions = model_constraints.fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)

    # Set the model "filename".
    config_model.setdefault("filename", str(FILES_DIR / f"{cipher.nbr_rounds}round_{cipher.name}_{goal}_{objective_target}_milp_model.lp"))

    # Set "solver" and "solution_number" for solving the model.
    config_solver.setdefault("solver", "DEFAULT")
    config_solver.setdefault("solution_number", 1)
    return config_model, config_solver


# -------------------- Predefined Additional Constraints --------------------
def expand_var_ids(var): # Expand variable IDs by bits if necessary.
    if var.bitsize > 1:
        return [f"{var.ID}_{i}" for i in range(var.bitsize)]
    return [var.ID]


def get_initial_state_var_ids(cipher, function="PERMUTATION"):
    func = cipher.functions[function]
    return [var_id for var in func.vars[1][0][:func.nbr_words] for var_id in expand_var_ids(var)]


def get_final_state_var_ids(cipher, function="PERMUTATION"):
    func = cipher.functions[function]
    return [var_id for var in func.vars[func.nbr_rounds][func.nbr_layers][:func.nbr_words] for var_id in expand_var_ids(var)]


def normalize_bit_positions(bit_positions, bit_size):
    if bit_positions is None:
        return []
    normalized = sorted(set(bit_positions))
    for bit in normalized:
        if bit < 0 or bit >= bit_size:
            raise ValueError(f"Invalid bit position {bit}. Expected 0 <= bit < {bit_size}.")
    return normalized


def gen_initial_two_subset_constraints(cipher, constant_bits=None, active_bits=None, function="PERMUTATION"):
    initial_var_ids = get_initial_state_var_ids(cipher, function=function)
    bit_size = len(initial_var_ids)
    if constant_bits is None:
        raise ValueError("constant_bits must be explicitly provided for TWO_SUBSET_INIT.")
    constant_bits = normalize_bit_positions(constant_bits, bit_size)
    if active_bits is None:
        active_bits = [bit for bit in range(bit_size) if bit not in constant_bits]
    active_bits = normalize_bit_positions(active_bits, bit_size)

    constant_var_ids = [var_id for bit, var_id in enumerate(initial_var_ids) if bit in constant_bits]
    active_var_ids = [var_id for bit, var_id in enumerate(initial_var_ids) if bit in active_bits]

    constraints = []
    constraints += model_constraints.gen_predefined_constraints("milp", "EXACTLY", constant_var_ids, 0, bitwise=False)
    constraints += model_constraints.gen_predefined_constraints("milp", "EXACTLY", active_var_ids, 1, bitwise=False)
    constraints.append("Binary\n" + " ".join(initial_var_ids))
    return constraints


# ---------------- Two-Subset Balanced Bit Search Utilities -----------------
def build_final_objective(final_var_ids):
    return [[" + ".join(final_var_ids)]]


def extract_unit_final_var(solution, final_var_ids):
    for var_id in final_var_ids:
        if int(round(solution.get(var_id, 0))) == 1:
            return var_id
    return None


def gen_ban_final_var_constraint(var_id):
    return f"{var_id} = 0"


def final_var_id_to_bit_position(var_id):
    bit_position = str(var_id).rsplit("_", 1)[-1]
    if not bit_position.isdigit():
        raise ValueError(f"Invalid final state variable ID: {var_id}.")
    return int(bit_position)


def final_var_ids_to_bit_positions(var_ids):
    return [final_var_id_to_bit_position(var_id) for var_id in var_ids]


def search_balanced_bits(base_constraints, final_var_ids, config_model, config_solver):
    banned_var_ids = []
    status = "unknown"
    objective = build_final_objective(final_var_ids)

    while len(banned_var_ids) < len(final_var_ids):
        constraints = list(base_constraints) + [gen_ban_final_var_constraint(var_id) for var_id in banned_var_ids]
        solutions = milp_search.modeling_solving_milp("OPTIMAL", constraints, objective, config_model, config_solver)
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

        var_id = extract_unit_final_var(solution, final_var_ids)
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
    """
    Perform integral attacks on a given cipher using bit-based two-subset division property.

    Parameters:
        cipher (Cipher): The cipher object to analyze.
        goal (str): The specific cryptanalysis goal, currently only INTEGRAL_TWOSUBSET is supported.
        constraints (list of string): User-specified constraints to be added to the model.
            - ["TWO_SUBSET_INIT"]: Automatically add two-subset initial state constraints.
            - Specific variables constraints, e.g., ["v_1_0_0 = 1", "v_2_1_0 = 0"] for MILP.
            - Any other user-defined constraints.
        objective_target (str): Currently only "EXISTENCE" is supported.
        show_mode (int): The level of solution/result visualization: 0, 1, 2, 3.
        config_model (dict): Optional advanced arguments for modeling, see attacks.parse_and_set_configs() for details.
        config_solver (dict): Optional advanced arguments for solving, see attacks.parse_and_set_configs() for details.

    Returns: A list of integral distinguisher objects.
    """

    constraints = constraints or []
    assert goal == "INTEGRAL_TWOSUBSET", f"Invalid goal: {goal}. Expected 'INTEGRAL_TWOSUBSET'."
    assert isinstance(constraints, list), f"Invalid constraints: {constraints}. Expected a list of strings."
    assert objective_target == "EXISTENCE", "INTEGRAL_TWOSUBSET currently supports objective_target='EXISTENCE'."
    assert show_mode in [0, 1, 2, 3], f"Invalid show_mode: {show_mode}. Expected one of [0, 1, 2, 3]."
    assert isinstance(config_model, dict) or config_model is None, f"Invalid config_model: {config_model}. Expected a dictionary or None."
    assert isinstance(config_solver, dict) or config_solver is None, f"Invalid config_solver: {config_solver}. Expected a dictionary or None."

    # Generate a new cipher instance with added copy layer after each operator.
    cipher.add_copy_operators()

    # Step 1. Parse and set model and solver configurations.
    config_model, config_solver = parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver)
    model_type = config_model.get("model_type", "milp")

    # Step 2. Generate round constraints and objective function for the cipher.
    round_constraints, obj_fun = model_constraints.gen_round_model_constraint_obj_fun(cipher, goal, model_type, config_model)

    # Step 3. Process additional constraints.
    model_cons = []
    for cons in constraints:
        if cons == "TWO_SUBSET_INIT": # Deal with specific additional constraints.
            model_cons += gen_initial_two_subset_constraints(
                cipher,
                constant_bits=config_model.get("constant_bits"),
                active_bits=config_model.get("active_bits"),
            )
        else:
            model_cons += [cons]
    model_cons += round_constraints

    # Step 4. Search balanced bits or solve the model directly.
    if "TWO_SUBSET_INIT" in constraints:
        final_var_ids = get_final_state_var_ids(cipher, function=config_model.get("state_function", "PERMUTATION"))
        search_result = search_balanced_bits(model_cons, final_var_ids, config_model, config_solver)
        if search_result["status"] != "found":
            return []
        return extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, [search_result])

    solutions = milp_search.modeling_solving_milp(objective_target, model_cons, obj_fun, config_model, config_solver)
    if isinstance(solutions, list):
        return extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, solutions)

    raise ValueError("[WARNING] No valid solutions found.")


# ---------------- Distinguisher Extraction and Visualization ----------------
def extract_and_format_integral_distinguishers(cipher, goal, config_model, config_solver, solutions):
    distinguishers = []
    for i, sol in enumerate(solutions):
        data = {"cipher": f"{cipher.functions['PERMUTATION'].nbr_rounds}_round_{cipher.name}",
                "rounds": config_model["rounds"],
                "goal": goal,
                "status": sol.get("status", "found"),
                "balanced_bits": sol.get("balanced_bits", []),
                "banned_bits": sol.get("banned_bits", []),
                "config_model": config_model,
                "config_solver": config_solver}
        distinguisher = IntegralDistinguisher(data, solution_trace=sol)
        if i > 0:
            print(f"[INFO] Saving the {i+1}-th Integral Distinguisher.")
            distinguisher.json_filename = distinguisher.json_filename.replace(".json", f"_{i}.json") if distinguisher.json_filename else str(FILES_DIR / f"{distinguisher.data['cipher']}_integral_{i}.json")
            distinguisher.txt_filename = distinguisher.txt_filename.replace(".txt", f"_{i}.txt") if distinguisher.txt_filename else str(FILES_DIR / f"{distinguisher.data['cipher']}_integral_{i}.txt")
        distinguisher.save_json()
        distinguisher.save_txt()
        distinguishers.append(distinguisher)
    return distinguishers
