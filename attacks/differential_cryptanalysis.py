import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] # differential_cryptanalysis.py -> attacks -> <ROOT>
sys.path.insert(0, str(ROOT))

from attacks.trail import DifferentialTrail
import tools.model_constraints as model_constraints
import tools.model_objective as model_objective
import tools.milp_search as milp_search
import tools.sat_search as sat_search
import visualisations.visualisations as vis

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


# **************************************************************************** #
# This module is the interface for differential attacks, including:
# 1. search differential trails
# **************************************************************************** #


# ---------------------- Model and Solver Configuration ----------------------
def parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver): # Parse input parameters and apply default values for model and solver configurations.
    # ===== Set Default config_model and config_solver =====
    config_model = config_model or {}
    config_solver = config_solver or {}

    # Set "model_type", the automated model framework, 'milp' or 'sat'
    config_model["model_type"] = config_model.get("model_type", "milp").lower()

    # Set "functions", "rounds", "layers", "positions" for modeling
    functions, rounds, layers, positions = model_constraints.fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)

    # Set "solver" for solving the model
    config_solver.setdefault("solver", "DEFAULT")

    if config_model["model_type"] == "milp":
        # Set the model "filename".
        config_model["filename"] = str(FILES_DIR / f"{cipher.name}_{goal}_{objective_target}_{config_solver['solver']}_model.lp")

    elif config_model["model_type"] == "sat":
        # Set the model "filename".
        config_model["filename"] = str(FILES_DIR / f"{cipher.name}_{goal}_{objective_target}_{config_solver['solver']}_model.cnf")

    return config_model, config_solver


# -------------------- Predefined Additional Constraints --------------------
def expand_var_ids(var, bitwise=False): # Expand variable IDs by bits if necessary.
    if bitwise and var.bitsize > 1:
        return [f"{var.ID}_{i}" for i in range(var.bitsize)]
    return [var.ID]

def gen_input_non_zero_constraints(cipher, goal, config_model): # Generate input non-zero constraints for the cipher based on the goal and model type.
    cons_vars = [var for cons in cipher.inputs_constraints for var in cons.input_vars]
    model_type = config_model.get("model_type", "milp").lower()
    encoding = config_model.get("atleast_encoding_sat", "SEQUENTIAL") if model_type == "sat" else None
    bitwise = "TRUNCATEDDIFF" not in goal
    constraints = model_constraints.gen_predefined_constraints(
        model_type=model_type,
        cons_type="SUM_AT_LEAST",
        cons_vars=cons_vars,
        cons_value=1,
        bitwise=bitwise,
        encoding=encoding,
    )
    # MILP-specific: declare decision variables as binary
    if model_type == "milp":
        binary_vars = []
        for var in cons_vars:
            binary_vars += (expand_var_ids(var, bitwise=bitwise))
        if binary_vars:
            constraints.append("Binary\n" + " ".join(binary_vars))
    return constraints


# ------------------------ Differential Trail Search -------------------------
def search_diff_trail(cipher, goal="DIFFERENTIALPATH_PROB", constraints=["INPUT_NOT_ZERO"], objective_target="OPTIMAL", show_mode=0, config_model=None, config_solver=None):
    """
    Perform differential attacks on a given cipher using the specified model_type.

    Parameters:
        cipher (Cipher): The cipher object to analyze.
        goal (str): The specific cryptanalysis goal: GOAL or GOAL_OPERATOR_NUMBER
            - DIFFERENTIAL_SBOXCOUNT
            - DIFFERENTIALPATH_PROB
            - DIFFERENTIAL_PROB
            - TRUNCATEDDIFF_SBOXCOUNT
        constraints (list of string): User-specified constraints to be added to the model.
            - ['INPUT_NOT_ZERO']: Automatically add input non-zero constraints as required by the goal.
            - Specific variables constraints, e.g., ['v_1_0_0 = 1', 'v_2_1_0 = 0'] for MILP, ['v_1_0_0', '-v_2_1_0'] for SAT.
            - Any other user-defined constraints.
        objective_target (str): The target for the objective function, which can be:
            - 'OPTIMAL': Find the optimal solution.
            - 'AT MOST X': Find a solution with an objective value at most X.
            - 'EXACTLY X': Find a solution with an objective value exactly X.
            - 'AT LEAST X': Find a solution with an objective value at least X.
            - 'EXISTENCE': Find any feasible solution.
        show_mode (int): The level of solution/result visualization: 0, 1, 2.
        config_model (dict): Optional advanced arguments for modeling, see attacks.parse_and_set_configs() for details.
        config_solver (dict): Optional advanced arguments for solving, see attacks.parse_and_set_configs() for details.

    Returns: A list of differential trail objects.
    """

    assert any(goal.startswith(prefix) for prefix in ["DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB", "TRUNCATEDDIFF_SBOXCOUNT"]), f"Invalid goal: {goal}. Expected one of ['DIFFERENTIAL_SBOXCOUNT', 'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB', 'TRUNCATEDDIFF_SBOXCOUNT']"
    assert isinstance(constraints, list), f"Invalid constraints: {constraints}. Expected a list of strings."
    assert any(objective_target.startswith(prefix) for prefix in ['OPTIMAL', 'AT MOST', 'EXACTLY', 'AT LEAST', 'EXISTENCE']), f"Invalid objective_target: {objective_target}. Expected one of ['OPTIMAL', 'AT MOST X', 'EXACTLY X', 'AT LEAST X']"
    assert show_mode in [0, 1, 2, 3], f"Invalid show_mode: {show_mode}. Expected one of [0, 1, 2]"
    assert isinstance(config_model, dict) or config_model is None, f"Invalid config_model: {config_model}. Expected a dictionary or None."
    assert isinstance(config_solver, dict) or config_solver is None, f"Invalid config_solver: {config_solver}. Expected a dictionary or None."

    # Step 1. Parse and set model and solver configurations.
    config_model, config_solver = parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver)
    model_type = config_model.get("model_type", "milp")

    # Step 2. Generate round constraints and objective function for the cipher.
    round_constraints, obj_fun = model_constraints.gen_round_model_constraint_obj_fun(cipher, goal, model_type, config_model)

    # Step 3. Process additional constraints.
    model_cons = []
    for cons in constraints:
        if cons == "INPUT_NOT_ZERO":  # Deal with specific additional constraints.
            model_cons += gen_input_non_zero_constraints(cipher, goal, config_model)
        else:
            model_cons += [cons]
    model_cons += round_constraints

    # Step 4: Modeling and Solving.
    if model_type == "milp":
        solutions = milp_search.modeling_solving_milp(objective_target, model_cons, obj_fun, config_model, config_solver)

    elif model_type == "sat":
        if goal in ["DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB"] and model_objective.has_Sbox_with_decimal_weights(cipher, goal):
            config_model["decimal_objective_function"] = {}
            Sbox = model_objective.detect_Sbox(cipher)
            config_model["decimal_objective_function"]["Sbox"] = Sbox
            if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
                config_model["decimal_objective_function"]["table"] = Sbox.computeDDT()

        solutions = sat_search.modeling_solving_sat(objective_target, model_cons, obj_fun, config_model, config_solver)

    else:
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")

    # Step 5: Extract and Visualize Trails from Solutions.
    if isinstance(solutions, list):
        return extract_and_format_diff_trails(cipher, goal, config_model, show_mode, solutions)

    raise ValueError("[WARNING] No valid solutions found.")


# -------------------- Trail Extraction and Visualization --------------------
def extract_and_format_diff_trails(cipher, goal, config_model, show_mode, solutions):
    trails = []
    for i, sol in enumerate(solutions):
        trail_struct = extract_trail_structures(cipher, goal, sol)
        data = {"cipher": f"{cipher.functions['PERMUTATION'].nbr_rounds}_round_{cipher.name}", "functions": config_model["functions"], "rounds": config_model["rounds"], "trail_struct": trail_struct, "diff_weight": sol.get("obj_fun_value"), "rounds_diff_weight": sol.get("rounds_obj_fun_values")}
        trail = DifferentialTrail(data, solution_trace=sol)
        if i > 0:
            print(f"[INFO] Saving the {i+1}-th Trail.")
            trail.json_filename = trail.json_filename.replace(".json", f"_{i}.json") if trail.json_filename else str(FILES_DIR / f"{trail.data['cipher']}_trail_{i}.json")
            trail.txt_filename = trail.txt_filename.replace(".txt", f"_{i}.txt") if trail.txt_filename else str(FILES_DIR / f"{trail.data['cipher']}_trail_{i}.txt")
        trail.save_json()
        trail.save_trail_txt(show_mode=show_mode)  # Print the trail in a human-readable format and save it to a file.
        trails.append(trail)
    return trails

def extract_trail_structures(cipher, goal, solution):
    """
    Extract a structured differential trail (trail_struct) from a solver assignment.

    Returned structure (example):
    """
    bitwise = "TRUNCATEDDIFF" not in goal

    def _get_solution_bit(var_id): # Map a variable id to '0'/'1'/'-'.
        v = solution.get(var_id, None)
        if v is None:
            return "-"
        try: # robust handling for bool/int/float
            return "1" if int(round(v)) == 1 else "0"
        except Exception:
            return "-"

    def node(var):
        """Build a per-variable node."""
        ids = expand_var_ids(var, bitwise=bitwise)
        bits = "".join(_get_solution_bit(v_id) for v_id in ids)
        return {
            "var_ID": getattr(var, "ID", str(var)), # ID of var
            "variables": ids, # List of extended word/bit variables from the given var
            "bin_values": bits, # Binary string value
            }

    # ------------------------------ Build trail_struct ------------------------------
    trail_struct = {
        "bitwise": bitwise,
        "inputs": {},
        "outputs": {},
        "functions": {}
    }

    # ------------------------------ Inputs / Outputs ------------------------------
    # Prefer cipher.inputs/cipher.outputs if present; otherwise fall back to constraints.
    if hasattr(cipher, "inputs") and isinstance(cipher.inputs, dict):
        for name, var_list in cipher.inputs.items():
            trail_struct["inputs"][name] = [node(v) for v in var_list]
    if hasattr(cipher, "outputs") and isinstance(cipher.outputs, dict):
        for name, var_list in cipher.outputs.items():
            trail_struct["outputs"][name] = [node(v) for v in var_list]

    # ------------------------------ Functions / Rounds / Layers ------------------------------
    for fun in cipher.functions:
        fun_store = {
        "nbr_words": cipher.functions[fun].nbr_words if hasattr(cipher.functions[fun], "nbr_words") else None,
        "nbr_temp_words": cipher.functions[fun].nbr_temp_words if hasattr(cipher.functions[fun], "nbr_temp_words") else None
        }
        for r in range(1, cipher.functions[fun].nbr_rounds + 1):
            round_store = {}
            for l in range(cipher.functions[fun].nbr_layers + 1):
                layer_nodes = [node(v) for v in cipher.functions[fun].vars[r][l]]
                round_store[l] = layer_nodes
            fun_store[r] = round_store
        trail_struct["functions"][fun] = fun_store
    return trail_struct
