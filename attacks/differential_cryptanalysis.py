import sys
import os
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files'))


# ---------- Utility for Model Version and Constraints ----------
def configure_model_version(cipher, goal): # Configure the model version for all operators in the cipher based on the attack goal.
    if goal == "calculate_min_diff_active_sbox": # Set model_version of all operators in the cipher to 'XORDIFF'. 
        attacks.set_model_versions(cipher, "XORDIFF") 
    elif goal == "search_best_diff_trail": # Set model_version of all operators in the cipher to 'XORDIFF', then set model_version to 'XORDIFF_PR' for modeling probablities of Sbox operators 
        attacks.set_model_versions(cipher, "XORDIFF")
        attacks.set_model_versions(cipher, "XORDIFF_PR", operator_name="Sbox")    
    elif goal == "search_best_truncated_diff_trail": # Set model_version of all operators in the cipher to 'XORDIFF_TRUNCATED'. 
        attacks.set_model_versions(cipher, "XORDIFF_TRUNCATED")
    else: raise Exception(str(cipher.__class__.__name__) + ": unknown goal '" + goal + "'")


def gen_input_non_zero_constraints(cipher, model_type, goal): # Generate a standard input non-zero constraint list according to the attack goal.
    if goal == "calculate_min_diff_active_sbox" or goal == "search_best_diff_trail":
        cons_args={"bitwise": True}
    elif goal == "search_best_truncated_diff_trail":
        cons_args={"bitwise": False}
    return attacks.gen_predefined_constraints(cipher, model_type=model_type, cons_type="INPUT_NOT_ZERO", cons_args=cons_args)


# ---------- Differential Attack Interface ----------
def search_diff_trail(cipher, model_type, goal="search_best_diff_trail", add_constraints=["INPUT_NOT_ZERO"], model_args=None, solving_args=None):
    """
    Perform differential attacks on a given cipher using the specified model_type.

    Parameters:
        cipher (Cipher): The cipher object to analyze.
        model_type (str): The automated model framework (e.g., 'milp', 'sat').
        goal (str): The specific cryptanalysis goal (e.g., 'calculate_min_diff_active_sbox', 'search_best_diff_trail', 'search_best_truncated_diff_trail').
        add_constraints (list of string): User-specified additional constraints to be added to the model.
            - 'INPUT_NOT_ZERO' (str): Automatically add input non-zero constraints as required by the goal.
        model_args (dict): Optional advanced arguments for modeling:
            - 'model_version' (str): Custom model version identifier. If not provided or set to 'DEFAULT', uses the default version based on goal.
            - 'obj_sat' (int): Starting objective value for SAT model. Defaults to 0 if not provided.
            - 'matsui_constraint' (dict): Arguments for Matsui branch-and-bound constraints (e.g., {"Round": 1, "best_obj": [...]}.)
            - Any other model-specific options for building models.
        solving_args (dict): Optional advanced arguments for solving:
            - 'solver' (str): The solver for solving the model (e.g., "gurobi", "scip").
            - 'show_mode' (int): Level or mode for solution/result visualization. 
            - Any other solver-specific options for solving models.
    Returns: (solution, obj)
    """

    model_args = model_args or {}
    solving_args = solving_args or {}

    # Step 1: Configure model version if not specified by the user. If "model_version" is not provided or set to "DEFAULT", use the default setting based on the analysis goal.
    if model_args.get("model_version", "DEFAULT") == "DEFAULT": 
        configure_model_version(cipher, goal=goal)

    # Step 2: Add additional constraints.
    additional_constraints = copy.deepcopy(add_constraints)
    if "INPUT_NOT_ZERO" in additional_constraints: # Add input non-zero constraints if not disabled by user
        additional_constraints += gen_input_non_zero_constraints(cipher, model_type, goal)
        additional_constraints.remove("INPUT_NOT_ZERO")
    
    # Step 3: Generate the model and solve the optimal solution
    filename = os.path.join(base_path, f"{cipher.name}_{model_type}_{goal}_{solving_args.get('solver', '')}_model.{'lp' if model_type == 'milp' else 'cnf'}")
    sol, obj = attacks.modeling_solving_optimal_solution(cipher, model_type, filename, additional_constraints, model_args=model_args, solving_args=solving_args) # Call the core modeling and solving function.
    
    # Step 4: Generate and visualize the trail. TO DO.

    return sol, obj