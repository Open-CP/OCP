import time

import attacks.differential_cryptanalysis as diff
import attacks.linear_cryptanalysis as linear

# **************************************************************************** #
# This module provides a high-level attack interfaces, including:
# 1. differential attacks
# 2. linear attacks
# 3. other types of attacks (to be contributed in the future)
# **************************************************************************** #


# =================== Differential Attacks ===================
def diff_attacks(cipher, goal="DIFFERENTIALPATH_PROB", constraints=["INPUT_NOT_ZERO"], objective_target="OPTIMAL", show_mode=0, config_model=None, config_solver=None):
    time_start = time.time()

    if goal in ["DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB", "TRUNCATEDDIFF_SBOXCOUNT"]:
        trails = diff.search_diff_trail(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)
    else:
        raise ValueError(f"[WARNING] Invalid goal: {goal}.")

    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return trails


# =================== Linear Attacks ===================
def linear_attacks(cipher, goal="LINEARPATH_CORRE", constraints=["INPUT_NOT_ZERO"], objective_target="OPTIMAL", show_mode=0, config_model=None, config_solver=None):
    time_start = time.time()

    if goal in ["LINEAR_SBOXCOUNT", "LINEARPATH_CORRE", "LINEAR_CORRE", "TRUNCATEDLINEAR_SBOXCOUNT"]:
        trails = linear.search_linear_trail(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)
    else:
        raise ValueError(f"[WARNING] Invalid goal: {goal}.")

    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return trails
