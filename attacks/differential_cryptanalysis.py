import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1] # this file -> attacks -> <ROOT>
sys.path.insert(0, str(ROOT))
import attacks.attacks as attacks
from attacks.trail import DifferentialTrail

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Constraint Generation ----------
def gen_input_non_zero_constraints(cipher, goal, config_model): # Generate a standard input non-zero constraint list according to the attack goal.
    functions, rounds, layers, positions = config_model["functions"], config_model["rounds"], config_model["layers"], config_model["positions"]
    model_type = config_model.get("model_type")
    atleast_encoding = config_model.get("atleast_encoding_sat")
    cons_vars = []
    for fun in functions:
        if fun in ["FUNCTION", "KEY_SCHEDULE"]:
            assert fun in cipher.functions, f"Function {fun} not found in cipher."
            start_round = rounds[fun][0]
            start_layer = layers[fun][start_round][0]
            start_words, end_words = positions[fun][start_round][start_layer][0], positions[fun][start_round][start_layer][-1]
            cons_vars += cipher.functions[fun].vars[start_round][start_layer][start_words:end_words+1]        
    bitwise = False if "TRUNCATEDDIFF" in goal else True
    return attacks.gen_predefined_constraints(model_type=model_type, cons_type="SUM_AT_LEAST", cons_vars=cons_vars, cons_value=1, bitwise=bitwise, encoding=atleast_encoding)


# ---------- Trail Representation ----------
def extract_trail_from_solution(cipher, goal, solution, hex_format=True):
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
    print(trail_values, trail_vars)
    return trail_values, trail_vars

# ---------- Differential Attack Interface ----------
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
    
        
    # Step 1: Configures
    config_model, config_solver = attacks.parse_and_set_configs(cipher, config_model, config_solver)

    # Step 2: Add additional constraints.
    additional_constraints = []
    for cons in constraints:
        if cons == "INPUT_NOT_ZERO":
            additional_constraints += gen_input_non_zero_constraints(cipher, goal, config_model)
        else:
            additional_constraints += [cons]

    # Step 3: Generate the model and solve the optimal solution
    config_model["filename"] = config_model["filename"].replace(f"model", f"{goal}_{objective_target}_{config_solver['solver']}_model")  # Update the filename to include goal and objective target.
    solutions = attacks.modeling_and_solving(cipher, goal, objective_target, additional_constraints, config_model, config_solver) # Call the core modeling and solving function.
    
    # Step 4: Generate and visualize the trail.
    if isinstance(solutions, list):
        trails = []
        for i, sol in enumerate(solutions):
            trail_values, trail_vars = extract_trail_from_solution(cipher, goal, sol)
            data = {"cipher": f"{cipher.functions['FUNCTION'].nbr_rounds}_round_{cipher.name}", "functions": config_model["functions"], "rounds": config_model["rounds"], "trail_vars": trail_vars, "trail_values": trail_values, "diff_weight": sol.get("obj_fun_value"), "rounds_diff_weight": sol.get("rounds_obj_fun_values")}
            trail = DifferentialTrail(data, solution_trace=sol)
            if i > 0:
                trail.save_json(filename = str(FILES_DIR / f"{trail.data['cipher']}_trail_{i}.json"))
            else:
                trail.save_json()
            trail.save_trail_txt(show_mode=show_mode)  # Print the trail in a human-readable format and save it to a file.
            trails.append(trail)
        return trails
    
    print("No trail found!")
    return []