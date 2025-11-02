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
            - 'INPUT_NOT_ZERO' (str): Automatically add input non-zero constraints as required by the goal.
            - Any other user-defined constraints.
        objective_target (str): The target for the objective function, which can be:
            - 'OPTIMAL' (str): Find the optimal solution.
            - 'AT MOST X' (str): Find a solution with an objective value at most X.
            - 'EXACTLY X' (str): Find a solution with an objective value exactly X.
            - 'AT LEAST X' (str): Find a solution with an objective value at least X.
        show_mode (int): The level of solution/result visualization: 0, 1, 2.
        config_model (dict): Optional advanced arguments for modeling, see attacks.parse_and_set_configs() for details.
        config_solver (dict): Optional advanced arguments for solving, see attacks.parse_and_set_configs() for details.

    Returns: A list of differential trail objects.
    """

    assert any(goal.startswith(prefix) for prefix in ["DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB", "TRUNCATEDDIFF_SBOXCOUNT"]), f"Invalid goal: {goal}. Expected one of ['DIFFERENTIAL_SBOXCOUNT', 'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB', 'TRUNCATEDDIFF_SBOXCOUNT']"
    assert isinstance(constraints, list), f"Invalid constraints: {constraints}. Expected a list of strings."
    assert any(objective_target.startswith(prefix) for prefix in ['OPTIMAL', 'AT MOST', 'EXACTLY', 'AT LEAST']), f"Invalid objective_target: {objective_target}. Expected one of ['OPTIMAL', 'AT MOST X', 'EXACTLY X', 'AT LEAST X']"
    assert show_mode in [0, 1, 2], f"Invalid show_mode: {show_mode}. Expected one of [0, 1, 2]"
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
            model_cons += model_constraints.gen_input_non_zero_constraints(cipher, goal, config_model)
        else:
            model_cons += [cons]
    model_cons += round_constraints

    # Step 4: Modeling and Solving.
    if model_type == "milp":
        solutions = milp_search.modeling_solving_milp(objective_target, model_cons, obj_fun, config_model, config_solver)

    elif model_type == "sat":
        if model_objective.has_Sbox_with_decimal_weights(cipher, goal):
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
        trail_values, trail_vars = extract_trail_values_and_vars(cipher, goal, sol)
        data = {"cipher": f"{cipher.functions['FUNCTION'].nbr_rounds}_round_{cipher.name}", "functions": config_model["functions"], "rounds": config_model["rounds"], "trail_vars": trail_vars, "trail_values": trail_values, "diff_weight": sol.get("obj_fun_value"), "rounds_diff_weight": sol.get("rounds_obj_fun_values")}
        trail = DifferentialTrail(data, solution_trace=sol)
        if i > 0:
            trail.json_filename = trail.json_filename.replace(".json", f"_{i}.json") if trail.json_filename else str(FILES_DIR / f"{trail.data['cipher']}_trail_{i}.json")
            trail.txt_filename = trail.txt_filename.replace(".txt", f"_{i}.txt") if trail.txt_filename else str(FILES_DIR / f"{trail.data['cipher']}_trail_{i}.txt")
        trail.save_json()
        trail.save_trail_txt(show_mode=show_mode)  # Print the trail in a human-readable format and save it to a file.
        trails.append(trail)
    return trails

def extract_trail_values_and_vars(cipher, goal, solution, hex_format=True):
    bitwise = "TRUNCATEDDIFF" not in goal
    trail_values, trail_vars = {}, {}

    def expand_var_ids(var): # Expand variable IDs by bits if necessary.
        if bitwise and var.bitsize > 1:
            return [f"{var.ID}_{i}" for i in range(var.bitsize)]
        return [var.ID]

    def get_value_str(var): # Get binary string value for one variable (expanded)
        bits = ""
        for v in expand_var_ids(var):
            val = solution.get(v)
            bits += str(int(round(val))) if val is not None else "-"
        return bits

    def format_bits(bits, hex_format=True): # Format a binary string into hex or binary representation, or return '-' if unknown.
        if "-" in bits:
            return "-" * (len(bits) // 4 if hex_format else len(bits))
        if hex_format:
            return "0x" + hex(int(bits, 2))[2:].zfill(len(bits) // 4)
        return "0b" + bits

    for fun in cipher.functions:
        fun_vals, fun_vars = [], []
        for r in range(1, cipher.functions[fun].nbr_rounds + 1):
            round_vals, round_vars = [], []
            for l in range(cipher.functions[fun].nbr_layers + 1):
                vars_layer, value_bits = [], ""
                for var in cipher.functions[fun].vars[r][l]:
                    bits = get_value_str(var)
                    ids = expand_var_ids(var)
                    vars_layer.extend(ids)
                    value_bits += bits
                formatted = format_bits(value_bits, hex_format)
                if formatted:
                    round_vals.append(formatted)
                if vars_layer:
                    round_vars.append(vars_layer)
            if round_vals:
                fun_vals.append(round_vals)
            if round_vars:
                fun_vars.append(round_vars)
        if fun_vals:
            trail_values[fun] = fun_vals
        if fun_vars:
            trail_vars[fun] = fun_vars
    return trail_values, trail_vars
