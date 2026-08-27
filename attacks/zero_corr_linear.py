"""The interface for zero-correlation linear attacks.

Provides:

1. search for zero-correlation distinguishers (by feasibility enumeration)
"""

from pathlib import Path
from itertools import combinations
import time

from tools.model_configuration import fill_functions_rounds_layers_positions, gen_round_model_constraint_obj_fun
from tools.model_constraints import gen_predefined_constraints
import tools.milp_search as milp_search
import tools.sat_search as sat_search

ROOT = Path(__file__).resolve().parents[1]
FILES_DIR = ROOT / "files"

# Map the user-facing goal to the concrete distinguisher model goal and its granularity.
#   ZEROCORRELATIONTRUNCATEDLINEAR -> word-level truncated;  ZEROCORRELATIONLINEAR -> bit-level.
_GOAL_MIDDLE = {"ZEROCORRELATIONTRUNCATEDLINEAR": "TRUNCATEDLINEAR_SBOXCOUNT",
                "ZEROCORRELATIONLINEAR": "LINEAR_SBOXCOUNT"}


def _parse_and_set_configs(cipher, goal, config_model, config_solver):
    """Fill in default model/solver configuration and return ``(config_model, config_solver)``."""
    # ===== Set Default config_model and config_solver =====
    config_model = config_model or {}
    config_solver = config_solver or {}

    # Set "model_type", the automated model framework, 'milp' or 'sat'
    config_model["model_type"] = config_model.get("model_type", "milp").lower()

    # Set "functions", "rounds", "layers", "positions" for modeling
    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)

    # Set "solver" for solving the model
    config_solver.setdefault("solver", "DEFAULT")

    FILES_DIR.mkdir(parents=True, exist_ok=True)  # ensure the output directory exists (lazy)
    # Set the model "filename".
    if config_model["model_type"] == "milp":
        config_model["filename"] = str(FILES_DIR / f"{cipher.nbr_rounds}round_{cipher.name}_{goal}_milp_model.lp")
    elif config_model["model_type"] == "sat":
        config_model["filename"] = str(FILES_DIR / f"{cipher.nbr_rounds}round_{cipher.name}_{goal}_sat_model.cnf")

    return config_model, config_solver


def _gen_fixed_activity_constraints(vars_, active_positions, model_type="milp", bitwise=False):
    """Fix a mask pattern over the boundary vars: active positions -> 1, the rest -> 0.
    bitwise=False: word-level (one unit per var); bitwise=True: bit-level (one unit per bit)."""
    ids = [f"{v.ID}_{j}" for v in vars_ for j in range(v.bitsize)] if bitwise else [v.ID for v in vars_]
    active = set(active_positions)
    cons = []
    for i, vid in enumerate(ids):
        cons += gen_predefined_constraints(model_type, "EXACTLY", [vid], 1 if i in active else 0, bitwise=False)
    return cons


def _verify_zero_correlation(base_cons, obj, fix_constraints, model_type, config_model, config_solver):
    """Solve the base model with the fixed (Gamma_X, Gamma_Y) added and test feasibility.
    Returns True if NO trail exists -> confirmed zero-correlation; False if a trail exists."""
    model_cons = base_cons + list(fix_constraints)
    if model_type == "milp":
        solutions = milp_search.modeling_solving_milp("EXISTENCE", model_cons, obj, config_model, config_solver)
    elif model_type == "sat":
        solutions = sat_search.modeling_solving_sat("EXISTENCE", model_cons, obj, config_model, config_solver)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'milp' or 'sat'.")
    return not solutions   # no trail -> confirmed zero-correlation


# ---------------------------- distinguisher search ----------------------------
def search_zc_distinguisher_enumeration(cipher, goal="ZEROCORRELATIONTRUNCATEDLINEAR",
                                        config_model=None, config_solver=None, show_mode=0):
    """Search the zero-correlation DISTINGUISHER by enumerating (Gamma_X, Gamma_Y): fix each pair and
    keep the infeasible ones (no trail). This mirrors the impossible-differential distinguisher
    search (``attacks/impossible_differential.py``); see [1] and [2].

    goal: "ZEROCORRELATIONTRUNCATEDLINEAR" (word-level truncated) or "ZEROCORRELATIONLINEAR" (bit-level).
    Returns the list of zero-correlation (in_positions, out_positions) pairs.

    References:
        1. A. Bogdanov, V. Rijmen, "Linear hulls with correlation zero and linear cryptanalysis of block ciphers", DCC 2014.
        2. S. Sun, D. Gerault, P. Lafourcade, Q. Yang, Y. Todo, K. Qiao, L. Hu, "Analysis of AES, SKINNY, and others with Constraint Programming", ToSC 2017(1), 281-306.
    """
    if goal not in _GOAL_MIDDLE:
        raise ValueError(f"Invalid goal: {goal}. Expected one of {list(_GOAL_MIDDLE)}.")
    goal_middle = _GOAL_MIDDLE[goal]
    config_model, config_solver = _parse_and_set_configs(cipher, goal, config_model, config_solver)
    model_type, bitwise = config_model["model_type"], ("TRUNCATEDLINEAR" not in goal_middle)

    # Boundary mask variables: Gamma_X / Gamma_Y at the segment's start / end (round, layer).
    functions = config_model["functions"]
    if len(functions) != 1:
        raise ValueError("zero-correlation search supports single-function primitives only.")
    fname = functions[0]
    perm = cipher.functions[fname]
    nw, L = perm.nbr_words, perm.nbr_layers
    rs, ls = config_model["rounds"][fname], config_model["layers"][fname]
    gx_vars = perm.vars[rs[0]][min(ls[rs[0]])][:nw]                     # Gamma_X: first-round input mask
    gy_vars = perm.vars[rs[-1]][min(max(ls[rs[-1]]) + 1, L)][:nw]       # Gamma_Y: last-round output mask

    enum = config_model.get("enumeration", {})
    max_total = enum.get("max_active_total")
    n_units = sum(v.bitsize for v in gx_vars) if bitwise else len(gx_vars)   # Gamma_X, Gamma_Y have the same size
    cand_in  = list(enum.get("positions_in",  range(n_units)))
    cand_out = list(enum.get("positions_out", range(n_units)))

    # Build the base model once (configure operator versions, then generate round constraints);
    # only the fixed (Gamma_X, Gamma_Y) constraints change per iteration.
    base_cons, obj = gen_round_model_constraint_obj_fun(cipher, goal_middle, config_model)

    # Enumerate (Gamma_X, Gamma_Y): active counts wi/wo in the configured ranges, then their positions.
    zero_corr, tested, start = [], 0, time.time()
    for wi in range(enum.get("min_active_in", 1), enum.get("max_active_in", 1) + 1):
        for in_pos in combinations(cand_in, wi):
            fix_in = _gen_fixed_activity_constraints(gx_vars, in_pos, model_type, bitwise)   # built once per Gamma_X
            for wo in range(enum.get("min_active_out", 1), enum.get("max_active_out", 1) + 1):
                if max_total is not None and wi + wo > max_total:
                    continue
                for out_pos in combinations(cand_out, wo):
                    tested += 1
                    fix = fix_in + _gen_fixed_activity_constraints(gy_vars, out_pos, model_type, bitwise)
                    if _verify_zero_correlation(base_cons, obj, fix, model_type, config_model, config_solver):
                        zero_corr.append((in_pos, out_pos))
    if show_mode:
        print(f"[INFO] tested={tested}, zero_correlation={len(zero_corr)}, time={time.time() - start:.1f}s")
    return zero_corr
