import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks
from attacks.trail import DifferentialTrail
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files'))


# ---------- Configuration Functions ----------
def set_default_configs(config_model, config_solver):
    config_model = config_model or {}
    config_solver = config_solver or {}
    config_model.setdefault("model_type", "milp") # Default to 'milp' if not specified
    config_solver.setdefault("solver", "DEFAULT") # Default to 'DEFAULT' if not specified
    return config_model, config_solver
    

def configure_start_end_round(cipher, config_model):
    start_round, end_round = {}, {}
    for state in cipher.states:
        rounds_cfg = config_model.get("rounds", {}).get(state)
        if rounds_cfg:
            assert isinstance(rounds_cfg, list) and len(rounds_cfg) > 0, f"Invalid rounds configuration for state {state}: {rounds_cfg}"
            start_round[state], end_round[state] = rounds_cfg[0], rounds_cfg[-1]
        else:
            start_round[state], end_round[state] = 1, cipher.states[state].nbr_rounds
    return start_round, end_round


def gen_input_non_zero_constraints(cipher, model_type, goal, start_round, atleast_encoding): # Generate a standard input non-zero constraint list according to the attack goal.
    cons_vars = cipher.states["STATE"].vars[start_round["STATE"]][0][:cipher.states["STATE"].nbr_words]
    if "KEY_STATE" in cipher.states:
        cons_vars += cipher.states["KEY_STATE"].vars[start_round["KEY_STATE"]][0][:cipher.states["KEY_STATE"].nbr_words]
    bitwise = False if "TRUNCATEDDIFF" in goal else True
    return attacks.gen_predefined_constraints(model_type=model_type, cons_type="SUM_AT_LEAST", cons_vars=cons_vars, cons_value=1, bitwise=bitwise, encoding=atleast_encoding) 


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
            - 'OPTIMAL STARTING FROM X' (str): Find the optimal solution starting from a specific value X.
            - 'AT MOST X' (str): Find a solution with an objective value at most X.
            - 'EXACTLY X' (str): Find a solution with an objective value exactly X.
        show_mode (int): The level of solution/result visualization: 0, 1, or 2.
        config_model (dict): Optional advanced arguments for modeling:
            - 'model_type' (str): The automated model framework (e.g., 'milp', 'sat').
            - 'rounds' (dict): Dictionary with "STATE" key containing a list of rounds to analyze.
            - 'matsui_constraints' (dict): Arguments for Matsui branch-and-bound constraints (e.g., {"Round": 2, "best_obj": [1]}.)
            - Any other model-specific options for building models.
        config_solver (dict): Optional advanced arguments for solving:
            - 'solver' (str): The solver for solving the model (e.g., "gurobi", "scip").
            - Any other solver-specific options for solving models.
        
    Returns: The differential trail object containing the results of the attack.
    """

    assert any(goal.startswith(prefix) for prefix in ["DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB", "TRUNCATEDDIFF_SBOXCOUNT"]), f"Invalid goal: {goal}. "
    assert isinstance(constraints, list), f"Invalid constraints: {constraints}. Expected a list of strings."
    assert any(objective_target.startswith(prefix) for prefix in ['OPTIMAL', 'OPTIMAL STARTING FROM', 'AT MOST', 'EXACTLY']), f"Invalid objective_target: {objective_target}."
    assert show_mode in [0, 1, 2], f"Invalid show_mode: {show_mode}. Expected one of [0, 1, 2]"
    assert isinstance(config_model, dict) or config_model is None, f"Invalid config_model: {config_model}. Expected a dictionary or None."
    assert isinstance(config_solver, dict) or config_solver is None, f"Invalid config_solver: {config_solver}. Expected a dictionary or None."
    
        
    # Step 1: Configures
    config_model, config_solver = set_default_configs(config_model, config_solver)
    start_round, end_round = configure_start_end_round(cipher, config_model)
    model_type = config_model.get("model_type")
    solver = config_solver.get('solver')    

    # Step 2: Add additional constraints.
    additional_constraints = copy.deepcopy(constraints)
    if "INPUT_NOT_ZERO" in additional_constraints: # Add input non-zero constraints if not disabled by user
        atleast_encoding = config_model.get("atleast_encoding_sat", "SEQUENTIAL")
        non_zero_cons = gen_input_non_zero_constraints(cipher, model_type, goal, start_round, atleast_encoding)
        idx = additional_constraints.index("INPUT_NOT_ZERO")
        additional_constraints = (additional_constraints[:idx] + non_zero_cons + additional_constraints[idx+1:]) # Replace 'INPUT_NOT_ZERO' with the generated non-zero constraints.
    
    # Step 3: Generate the model and solve the optimal solution
    filename = os.path.join(base_path, f"{cipher.name}_{goal}_{objective_target}_{model_type}_{solver}_model.{'lp' if model_type == 'milp' else 'cnf'}")
    config_model["filename"] = filename  # Set the filename in the model configuration.
    sol = attacks.modeling_and_solving(cipher, goal, objective_target, additional_constraints, config_model, config_solver) # Call the core modeling and solving function.
    
    # Step 4: Generate and visualize the trail. TO DO.
    if sol is not None:
        trail = DifferentialTrail(cipher, goal, start_round, end_round, sol) 
        trail.print_trail(show_mode, hex_format=True)  # Print the trail in a human-readable format.
        return trail
    
    print("No trail found!")
    return None