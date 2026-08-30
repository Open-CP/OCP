"""High-level attack interfaces.

Provides:

1. differential attacks
2. linear attacks
3. two-stage differential/linear trail search (truncated- then bit-level)
4. integral attacks
5. impossible-differential distinguisher search
6. zero-correlation linear distinguisher search

Other types of attacks to be contributed in the future.
"""
import time

import attacks.differential_cryptanalysis as diff
import attacks.linear_cryptanalysis as linear
import attacks.integral_cryptanalysis as integral
import attacks.impossible_differential as impossible_diff
import attacks.zero_corr_linear as zero_corr
from tools.model_constraints import gen_predefined_constraints


# =================== Differential Attacks ===================
def diff_attacks(
    cipher,
    goal="DIFFERENTIALPATH_PROB",
    constraints=None,
    objective_target="OPTIMAL",
    show_mode=0,
    config_model=None,
    config_solver=None,
):
    """Search for differential trails of the given cipher.

    See :func:`~attacks.differential_cryptanalysis.search_diff_trail` for the
    accepted parameters.

    Returns:
        list: The differential trail objects found.
    """
    time_start = time.time()
    trails = diff.search_diff_trail(
        cipher,
        goal=goal,
        constraints=constraints,
        objective_target=objective_target,
        show_mode=show_mode,
        config_model=config_model,
        config_solver=config_solver,
    )
    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return trails


# =================== Linear Attacks ===================
def linear_attacks(
    cipher,
    goal="LINEARPATH_CORR",
    constraints=None,
    objective_target="OPTIMAL",
    show_mode=0,
    config_model=None,
    config_solver=None,
):
    """Search for linear trails of the given cipher.

    See :func:`~attacks.linear_cryptanalysis.search_linear_trail` for the
    accepted parameters.

    Returns:
        list: The linear trail objects found.
    """
    time_start = time.time()
    trails = linear.search_linear_trail(
        cipher,
        goal=goal,
        constraints=constraints,
        objective_target=objective_target,
        show_mode=show_mode,
        config_model=config_model,
        config_solver=config_solver,
    )
    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return trails


# =================== Two-Stage Trail Search ===================
def two_stage_trail_search(
    cipher_factory,
    r,
    goal="DIFFERENTIALPATH_PROB",
    stage1_config_model=None,
    stage2_config_model=None,
    stage1_config_solver=None,
    stage2_config_solver=None,
):
    """Two-stage (truncated- then bit-level) trail search for word-oriented ciphers.

    Stage 1 searches a minimum-active-S-box truncated pattern; stage 2 fixes that
    pattern and searches the best bit-level trail on it. Only word-oriented cipher
    representations are supported (e.g. the AES or SKINNY permutation).

    A factory (rather than a cipher instance) is required because each stage runs a
    full attack that mutates the cipher (config filling, and the non-idempotent
    ``add_copy_operators`` for linear), so every stage needs a fresh cipher.

    Args:
        cipher_factory (callable): ``cipher_factory(r)`` returns a fresh cipher of ``r`` rounds.
        r (int): Number of rounds.
        goal (str): ``"DIFFERENTIALPATH_PROB"`` or ``"LINEARPATH_CORR"``.
        stage1_config_model (dict, optional): Stage-1 model configuration.
        stage2_config_model (dict, optional): Stage-2 model configuration.
        stage1_config_solver (dict, optional): Stage-1 solver configuration.
        stage2_config_solver (dict, optional): Stage-2 solver configuration.

    Returns:
        tuple or None: ``(min_active, best_weight)``, i.e. the stage-1 minimum
        active S-box count and the stage-2 best trail weight, or None if no
        truncated trail is found.
    """
    if goal == "DIFFERENTIALPATH_PROB":
        stage1_goal, weight_key, attack = "TRUNCATEDDIFF_SBOXCOUNT", "diff_weight", diff_attacks
    elif goal == "LINEARPATH_CORR":
        stage1_goal, weight_key, attack = "TRUNCATEDLINEAR_SBOXCOUNT", "linear_weight", linear_attacks
    else:
        raise ValueError(f"Unsupported goal '{goal}' in two-stage search. Expected 'DIFFERENTIALPATH_PROB' or 'LINEARPATH_CORR'.")

    # Stage 1: search the truncated trail with minimum active S-boxes.
    cipher1 = cipher_factory(r)
    trails1 = attack(
        cipher1,
        goal=stage1_goal,
        constraints=["INPUT_NOT_ZERO"],
        objective_target="OPTIMAL",
        config_model=stage1_config_model,
        config_solver=stage1_config_solver,
        show_mode=2,
    ) or []
    if not trails1:
        print("[INFO] no truncated trail found.")
        return None
    min_active = trails1[0].data[weight_key]
    print(f"[Stage 1] minimum active S-boxes = {min_active}")

    # Stage 2: fix the stage-1 activity pattern and search the best (min-weight) bit-level trail on it.
    def _fix_activity(model_type, cipher, trail):
        sol = trail.solution_trace
        cons = ["INPUT_NOT_ZERO"]
        for S in cipher.functions.values():
            for rr in range(1, S.nbr_rounds + 1):
                for l in range(S.nbr_layers + 1):
                    for v in S.vars[rr][l]:
                        active = round(float(sol.get(v.ID, 0))) >= 1
                        kind, val = ("SUM_AT_LEAST", 1) if active else ("EXACTLY", 0)
                        cons += gen_predefined_constraints(model_type, kind, [v], val, bitwise=True)
        return cons

    stage2_model_type = (stage2_config_model or {}).get("model_type", "milp")
    cipher2 = cipher_factory(r)
    stage2_cons = _fix_activity(stage2_model_type, cipher2, trails1[0])
    trails2 = attack(
        cipher2,
        goal=goal,
        constraints=stage2_cons,
        objective_target="OPTIMAL",
        config_model=stage2_config_model,
        config_solver=stage2_config_solver,
        show_mode=2,
    ) or []
    best_weight = trails2[0].data[weight_key] if trails2 else None
    print(f"[Stage 2] best trail weight = {best_weight}")
    return min_active, best_weight


# =================== Integral Attacks ===================
def integral_attacks(
    cipher,
    goal="INTEGRAL_TWOSUBSET",
    constraints=None,
    objective_target="EXISTENCE",
    show_mode=0,
    config_model=None,
    config_solver=None,
):
    """Search for integral distinguishers of the given cipher.

    See :func:`~attacks.integral_cryptanalysis.search_integral_distinguisher` for
    the accepted parameters.

    Returns:
        list: The integral distinguisher objects found.
    """
    time_start = time.time()
    distinguishers = integral.search_integral_distinguisher(
        cipher,
        goal=goal,
        constraints=constraints,
        objective_target=objective_target,
        show_mode=show_mode,
        config_model=config_model,
        config_solver=config_solver,
    )
    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return distinguishers


# =================== Impossible-Differential Attacks ===================
def impossible_differential_attacks(
    cipher,
    goal="IMPOSSIBLETRUNCATEDDIFF",
    config_model=None,
    config_solver=None,
    show_mode=0,
):
    """Search for impossible-differential distinguishers of the given cipher.

    See :func:`~attacks.impossible_differential.search_id_distinguisher_enumeration`
    for the accepted parameters.

    Returns:
        list: The impossible ``(in_positions, out_positions)`` pairs found.
    """
    time_start = time.time()
    distinguishers = impossible_diff.search_id_distinguisher_enumeration(
        cipher,
        goal=goal,
        config_model=config_model,
        config_solver=config_solver,
        show_mode=show_mode,
    )
    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return distinguishers


# =================== Zero-Correlation Linear Attacks ===================
def zero_correlation_attacks(
    cipher,
    goal="ZEROCORRELATIONTRUNCATEDLINEAR",
    config_model=None,
    config_solver=None,
    show_mode=0,
):
    """Search for zero-correlation linear distinguishers of the given cipher.

    See :func:`~attacks.zero_corr_linear.search_zc_distinguisher_enumeration` for
    the accepted parameters.

    Returns:
        list: The zero-correlation ``(in_positions, out_positions)`` pairs found.
    """
    time_start = time.time()
    distinguishers = zero_corr.search_zc_distinguisher_enumeration(
        cipher,
        goal=goal,
        config_model=config_model,
        config_solver=config_solver,
        show_mode=show_mode,
    )
    print(f"--- Total Time ---: {time.time() - time_start:.2f} seconds")
    return distinguishers
