"""Model-level SAT/MILP constraints for cryptanalysis.

This module provides:
1. Model-level constraint encodings
   - Predefined value/cardinality constraints (EXACTLY, AT_LEAST, SUM_AT_MOST, ...)
   - SAT sequential-counter cardinality encoding
   - Matsui's branch-and-bound constraints for MILP/SAT trail search
2. Attack boundary/input constraints shared by the differential and linear cryptanalysis
   - Nonzero input constraint builder
   - Fixed input/output difference or mask constraints for hull/probability goals

"""

from importlib import import_module
from importlib.util import find_spec


# ------------------------- PySAT cardinality backend ------------------------- #
# PySAT is an optional dependency: at import we only check that it is available
# (``find_spec``, no import), and load it on first use, so importing this module never
# pulls in PySAT. ``CardEnc``/``vpool`` stay ``None`` until the first SAT cardinality call.
CardEnc = None
vpool = None
pysat_import = find_spec("pysat") is not None


def _load_pysat_cardinality_backend():
    """Import PySAT's CardEnc/IDPool on first use; return ``(CardEnc, vpool)`` or ``(None, None)``."""
    global CardEnc, pysat_import, vpool

    if CardEnc is not None and vpool is not None:
        return CardEnc, vpool
    try:
        CardEnc = import_module("pysat.card").CardEnc
        IDPool = import_module("pysat.formula").IDPool
    except ImportError:
        pysat_import = False
        return None, None
    vpool = IDPool(start_from=1000)
    pysat_import = True
    return CardEnc, vpool


def _require_pysat_cardenc():
    """Return the PySAT ``CardEnc`` backend, raising ``ValueError`` if PySAT is unavailable."""
    card_enc, _ = _load_pysat_cardinality_backend()
    if card_enc is None:
        raise ValueError("PySAT is required for SAT cardinality constraints.")
    return card_enc


def _pysat_cardinality_error_types():
    """Return the exception types a PySAT ``CardEnc`` call may raise for an unsupported encoding/bound."""
    try:
        card_module = import_module("pysat.card")
    except ImportError:
        return (ValueError, RuntimeError)
    return tuple(
        error_type
        for error_type in (
            getattr(card_module, "NoSuchEncodingError", None),
            getattr(card_module, "UnsupportedBound", None),
            ValueError,
            RuntimeError,
        )
        if error_type is not None
    )


def _readable_cardinality_clauses(cnf, reverse_map):
    """Render PySAT numeric CNF clauses back into named (or ``dummy_*``) model constraint strings."""
    readable_clauses = []
    for clause in cnf.clauses:
        readable = " ".join(
            f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}"
            if lit < 0
            else reverse_map.get(abs(lit), f"dummy_{abs(lit)}")
            for lit in clause
        )
        readable_clauses.append(readable)
    return readable_clauses


def _pysat_cardinality_constraints(cons_vars, cons_value, encoding, encoder, encoder_name):
    """Build SAT cardinality clauses via a PySAT ``CardEnc`` encoder, returning ``[]`` if the encoding is unsupported."""
    if not encoding:
        encoding = 1
    _, card_vpool = _load_pysat_cardinality_backend()
    if card_vpool is None:
        raise ValueError("PySAT is required for SAT cardinality constraints.")
    variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
    reverse_map = {v: k for k, v in variable_map.items()}
    lits = [variable_map[name] for name in cons_vars]
    try:
        cnf = encoder(lits=lits, bound=cons_value, vpool=card_vpool, encoding=encoding)
    except _pysat_cardinality_error_types():
        print(f"[WARNING] CardEnc.{encoder_name} does not support encoding {encoding}; no constraints generated.")
        return []
    return _readable_cardinality_clauses(cnf, reverse_map)


# -------------------- Predefined Constraint Generation -------------------- #
def expand_var_ids(var, bitwise=False):
    """Expand a single variable into its per-bit (or whole-word) model variable IDs."""
    if bitwise and var.bitsize > 1:
        return [f"{var.ID}_{i}" for i in range(var.bitsize)]
    return [var.ID]


def expand_constraint_vars(cons_vars, bitwise=True):
    """Expand a list of variable names / Variable objects into model variable IDs."""
    cons_vars_name = []
    for var in cons_vars:
        if isinstance(var, str):
            cons_vars_name.append(var)
        else:
            cons_vars_name.extend(expand_var_ids(var, bitwise=bitwise))
    return cons_vars_name


def gen_predefined_constraints(model_type, cons_type, cons_vars, cons_value, bitwise=True, encoding=None):
    """
    Generate commonly used, predefined model constraints based on type and parameters.

    Args:
        model_type (str): The model backend, ``"milp"`` or ``"sat"``.
        cons_type (str): The constraint type, must be one of the predefined types:
            - "EXACTLY": All selected variables == a target value.
            - "AT_LEAST": All selected variables >= target value.
            - "AT_MOST": All selected variables <= target value.
            - "SUM_EXACTLY": Sum of selected variables == target value.
            - "SUM_AT_LEAST": Sum of selected variables >= target value.
            - "SUM_AT_MOST": Sum of selected variables <= target value.
        cons_vars (list): Variable names or Variables objects with ID and bitsize attributes.
        cons_value (int): Target value.
        bitwise (bool): If True, expand variables by bit.
        encoding: SAT cardinality encoding passed to the backend (ignored for MILP;
            ``None`` selects the default).

    Returns:
        list[str]: List of generated model constraint strings.
    """
    builders = {
        "EXACTLY": gen_constraints_exactly,
        "SUM_EXACTLY": gen_constraints_sum_exactly,
        "AT_MOST": gen_constraints_at_most,
        "SUM_AT_MOST": gen_constraints_sum_at_most,
        "AT_LEAST": gen_constraints_at_least,
        "SUM_AT_LEAST": gen_constraints_sum_at_least,
    }
    if cons_type not in builders:
        raise ValueError(f"Unsupported cons_type '{cons_type}'.")

    cons_vars_name = expand_constraint_vars(cons_vars, bitwise=bitwise)
    if not cons_vars_name:  # No variables in scope: nothing to constrain (keeps milp/sat consistent).
        return []
    return builders[cons_type](model_type, cons_vars_name, cons_value, encoding=encoding)


def gen_no_good_cut(var_ids, bits, model_type="milp"):
    """No-good cut forbidding one 0/1 assignment ``bits`` over ``var_ids``, forcing any later
    solution to differ in at least one of these variables (e.g. to enumerate distinct solutions).

        MILP:  sum_{b=0} v + sum_{b=1} (1 - v) >= 1
        SAT :  one blocking clause = OR of the negated literals of the assignment.

    Returns a list with the single constraint string (empty list if ``var_ids`` is empty).
    """
    if len(var_ids) != len(bits):
        raise ValueError(f"gen_no_good_cut: var_ids and bits differ in length ({len(var_ids)} vs {len(bits)}).")
    if not var_ids:
        return []
    if model_type == "milp":
        # violated only at the given assignment
        ones = sum(1 for b in bits if int(b) != 0)
        parts = []
        for vid, b in zip(var_ids, bits):
            sign = "+" if int(b) == 0 else "-"
            if not parts:  # first term: drop a leading '+', keep a leading '-'
                parts.append(vid if sign == "+" else f"-{vid}")
            else:
                parts.append(f"{sign} {vid}")
        lhs = " ".join(parts)
        return [f"{lhs} >= {1 - ones}"]
    elif model_type == "sat":
        return [" ".join((f"-{vid}" if int(b) else f"{vid}") for vid, b in zip(var_ids, bits))]
    else:
        raise ValueError(f"gen_no_good_cut: unknown model_type '{model_type}'.")


def gen_constraints_exactly(model_type, cons_vars, cons_value, encoding=None):
    """Constrain each selected variable to equal ``cons_value`` (MILP, or SAT for value 0/1)."""
    if model_type == "milp":
        return [f"{cons_vars[i]} = {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for EXACTLY constraint.")


def gen_constraints_sum_exactly(model_type, cons_vars, cons_value, encoding=1):
    """Constrain the sum of the selected variables to equal ``cons_value``."""
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" = {cons_value}"]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_vars and cons_value > len(cons_vars):
        return _sat_false(cons_vars)
    elif model_type == "sat" and pysat_import:
        _check_sat_encoding(encoding, "SUM_EXACTLY")
        card_enc = _require_pysat_cardenc()
        return _pysat_cardinality_constraints(cons_vars, cons_value, encoding, card_enc.equals, "equals")
    elif model_type == "sat":
        raise RuntimeError("SUM_EXACTLY over SAT requires PySAT (pip install python-sat).")
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_EXACTLY constraint.")


def gen_constraints_at_most(model_type, cons_vars, cons_value, encoding=None):
    """Constrain each selected variable to be at most ``cons_value`` (MILP, or SAT for value 0/1)."""
    if model_type == "milp":
        return [f"{cons_vars[i]} <= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for AT_MOST constraint.")


def _check_sat_encoding(encoding, cons_type):
    """Refuse an encoding the SAT cardinality builders cannot take."""
    if encoding is None:
        return
    if isinstance(encoding, int) and not isinstance(encoding, bool) and 0 <= encoding <= 9:
        return
    if cons_type == "SUM_AT_MOST" and encoding == "SEQUENTIAL":
        return
    allowed = "0-9" + (" or 'SEQUENTIAL'" if cons_type == "SUM_AT_MOST" else "")
    raise ValueError(f"[ERROR] Invalid encoding {encoding!r} for {cons_type}: expected {allowed}. "
                     f"Refer to https://pysathq.github.io/docs/html/api/card.html")


def _sat_false(cons_vars):
    """Two clauses that cannot both hold: a SAT constraint which is false over these variables."""
    return [f"{cons_vars[0]}", f"-{cons_vars[0]}"]


def gen_constraints_sum_at_most(model_type, cons_vars, cons_value, encoding="SEQUENTIAL"):
    """Constrain the sum of the selected variables to be at most ``cons_value``."""
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]
    elif model_type == "sat" and cons_vars and cons_value >= len(cons_vars):
        return []   # at most n of n binary variables: trivially true, so no clause is needed
    elif model_type == "sat" and (encoding == "SEQUENTIAL" or encoding is None):
        return gen_sequential_encoding_sat(cons_vars, cons_value)
    elif model_type == "sat" and pysat_import:
        _check_sat_encoding(encoding, "SUM_AT_MOST")
        card_enc = _require_pysat_cardenc()
        return _pysat_cardinality_constraints(cons_vars, cons_value, encoding, card_enc.atmost, "atmost")
    elif model_type == "sat":
        raise RuntimeError(
            "SUM_AT_MOST with a numeric PySAT encoding requires PySAT (pip install python-sat); "
            "use the default SEQUENTIAL encoding otherwise."
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_AT_MOST constraint.")


def gen_constraints_at_least(model_type, cons_vars, cons_value, encoding=None):
    """Constrain each selected variable to be at least ``cons_value`` (MILP, or SAT for value 0/1)."""
    if model_type == "milp":
        return [f"{cons_vars[i]} >= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for AT_LEAST constraint.")


def gen_constraints_sum_at_least(model_type, cons_vars, cons_value, encoding=1):
    """Constrain the sum of the selected variables to be at least ``cons_value``."""
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" >= {cons_value}"]
    elif model_type == "sat" and cons_value == 1:
        return [' '.join(f"{cons_vars[i]}" for i in range(len(cons_vars)))]
    elif model_type == "sat" and cons_vars and cons_value > len(cons_vars):
        return _sat_false(cons_vars)   # as in gen_constraints_sum_exactly
    elif model_type == "sat" and pysat_import:
        _check_sat_encoding(encoding, "SUM_AT_LEAST")
        card_enc = _require_pysat_cardenc()
        return _pysat_cardinality_constraints(cons_vars, cons_value, encoding, card_enc.atleast, "atleast")
    elif model_type == "sat":
        raise RuntimeError("SUM_AT_LEAST over SAT requires PySAT (pip install python-sat).")
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_AT_LEAST constraint.")


# ----------- SAT sequential-counter cardinality encoding ------------- #
def gen_sequential_encoding_sat(hw_list, weight, dummy_variables=None):
    """Generate SAT clauses for a sequential counter cardinality encoding."""
    if not hasattr(gen_sequential_encoding_sat, "_counter"):
        gen_sequential_encoding_sat._counter = 0
    n = len(hw_list)
    if not isinstance(weight, int) or weight < 0:
        raise ValueError(f"[ERROR] weight should be a non-negative integer, got {weight!r}.")
    if weight == 0:
        return [f'-{var}' for var in hw_list]
    if weight >= n:
        return []   # at most n (or more) of n: trivially true, so no clause is needed
    if dummy_variables is None:
        gen_sequential_encoding_sat._counter += 1
        prefix = f'dummy_seq_{gen_sequential_encoding_sat._counter}'
        dummy_variables = [[f'{prefix}_{i}_{j}' for j in range(weight)] for i in range(n - 1)]
    elif (not isinstance(dummy_variables, list) or len(dummy_variables) != n - 1
          or any(not isinstance(row, list) or len(row) < weight for row in dummy_variables)):
        raise ValueError(f"[ERROR] dummy_variables must hold {n - 1} rows of at least {weight} "
                         f"counter names: one row per position, one column per weight level.")
    constraints = [f'-{hw_list[0]} {dummy_variables[0][0]}']
    constraints.extend([f'-{dummy_variables[0][j]}' for j in range(1, weight)])
    for i in range(1, n - 1):
        constraints.append(f'-{hw_list[i]} {dummy_variables[i][0]}')
        constraints.append(f'-{dummy_variables[i - 1][0]} {dummy_variables[i][0]}')
        constraints.extend([f'-{hw_list[i]} -{dummy_variables[i - 1][j - 1]} {dummy_variables[i][j]}'
                            for j in range(1, weight)])
        constraints.extend([f'-{dummy_variables[i - 1][j]} {dummy_variables[i][j]}'
                            for j in range(1, weight)])
        constraints.append(f'-{hw_list[i]} -{dummy_variables[i - 1][weight - 1]}')
    constraints.append(f'-{hw_list[n - 1]} -{dummy_variables[n - 2][weight - 1]}')
    return constraints


# ----------- Matsui's branch-and-bound constraints Generation ------------- #
def gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type="ALL"):
    """Matsui's branch-and-bound constraints for a MILP model of a ``Round``-round search.

    Any contiguous ``i`` rounds of a valid trail are a valid ``i``-round trail, so they weigh at
    least ``best_obj[i-1]`` and the complementary block weighs at most ``obj - best_obj[i-1]``.
    That is emitted for each ``i`` in ``1..Round-1`` with a positive bound:

        UPPER:  sum of rounds i+1..Round   <=  obj - best_obj[i-1]
        LOWER:  sum of rounds 1..Round-i   <=  obj - best_obj[i-1]

    Args:
        Round: how many rounds are searched, which need not be ``cipher.nbr_rounds``.
        best_obj: ``[B_1, ..., B_{Round-1}]``, ``B_i`` the best weight known for ``i`` rounds --
            true weights, not the SAT side's integer objective.
        obj_fun: round ``r``'s weight variables in ``obj_fun[r-1]``; ``Round`` rows, or
            ``Round + 1`` when the extra one is empty.
        cons_type: ``"UPPER"``, ``"LOWER"`` or ``"ALL"`` for both.

    One empty extra row is tolerated; every round that remains must carry weight. That row is the
    one added for a leftover key addition (see ``primitives/primitives.py``), so dropping it costs
    nothing. A weightless round elsewhere is refused: ``best_obj`` bounds rounds that transform
    the state, so a complement holding one is held to weight it cannot carry.
    """
    if Round < 2:
        raise ValueError(f"[ERROR] Round = {Round} must be at least 2: one round cannot be split in two.")
    if len(best_obj) != Round - 1:
        raise ValueError(f"[ERROR] best_obj holds {len(best_obj)} bound(s), must hold Round-1 = {Round - 1}.")
    if any(not isinstance(bound, (int, float)) or isinstance(bound, bool) for bound in best_obj):
        raise ValueError(f"[ERROR] best_obj must hold numbers, got "
                         f"{[type(bound).__name__ for bound in best_obj]}.")
    if obj_fun is None:
        raise ValueError("[ERROR] obj_fun must hold one list of weight variables per round, got None.")
    obj_fun = list(obj_fun)  # accept any sequence of rows
    if not all(isinstance(obj, list) for obj in obj_fun):
        raise ValueError("[ERROR] obj_fun must hold one list of weight variables per round.")
    if len(obj_fun) == Round + 1 and (obj_fun[0] == []) != (obj_fun[-1] == []): # Exactly one end is empty: drop it
        obj_fun = obj_fun[1:] if obj_fun[0] == [] else obj_fun[:-1]
    if len(obj_fun) != Round:
        raise ValueError(f"[ERROR] obj_fun holds {len(obj_fun)} round(s) but Round = {Round}.")
    if not all(obj_fun):
        raise ValueError("[ERROR] obj_fun holds an empty round, so Matsui's bounds do not apply to "
                         "it. Every round searched must carry weight; check the goal and the scope.")
    if cons_type not in ["ALL", "UPPER", "LOWER"]:
        raise ValueError(f"[ERROR] cons_type = {cons_type} must be one of ['ALL', 'UPPER', 'LOWER'].")

    add_cons = []
    for i in range(1, Round):
        if best_obj[i-1] > 0:
            if cons_type == "ALL" or cons_type == "UPPER":
                w_vars = [var for r in range(i + 1, Round + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints("milp", "AT_MOST", all_vars, -best_obj[i-1])
            if cons_type == "ALL" or cons_type == "LOWER":
                w_vars = [var for r in range(1, Round - i + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints("milp", "AT_MOST", all_vars, -best_obj[i-1])
    return add_cons


def gen_matsui_constraints_sat(
    Round,
    best_obj,
    obj_sat,
    obj_var,
    GroupConstraintChoice=1,
    GroupNumForChoice=1,
):
    """Matsui's branch-and-bound constraints for a SAT model of a ``Round``-round search.

    The same bounding condition as :func:`gen_matsui_constraints_milp`, imposed on the sequential
    encoding's counter over all weight variables: the window spanning a group of rounds may hold at
    most ``obj_sat - best_obj[start] - best_obj[Round-end]`` active variables. With the default
    ``GroupNumForChoice = 1`` every window starts at round 0, so only the suffix family is emitted:
    MILP's UPPER, not its LOWER. That is deliberate: it matches the reference implementation.

    Args:
        Round: how many rounds are searched, which need not be ``cipher.nbr_rounds``.
        best_obj: ``[B_1, ..., B_{Round-1}]``, whole numbers; they count weight variables, so pass
            a round's ``"integer_obj_fun_value"`` where its true weight is fractional.
        obj_sat: the weight being tested, a positive integer.
        obj_var: round ``r``'s weight variables in ``obj_var[r-1]``; ``Round`` rows, or
            ``Round + 1`` when the extra one is empty.
        GroupConstraintChoice: only ``1`` is implemented.
        GroupNumForChoice: how many groups of windows to emit.

    One empty extra row is dropped and every round that remains must carry weight, exactly as in
    :func:`gen_matsui_constraints_milp`.
    """
    if Round < 2:
        raise ValueError(f"[ERROR] Round = {Round} must be at least 2: one round cannot be split in two.")
    if len(best_obj) != Round - 1:
        raise ValueError(f"[ERROR] best_obj holds {len(best_obj)} bound(s), must hold Round-1 = {Round - 1}.")
    if not isinstance(obj_sat, int) or obj_sat <= 0:
        raise ValueError(f"[ERROR] obj_sat = {obj_sat} must be a positive integer.")
    if any(not isinstance(bound, (int, float)) or isinstance(bound, bool) for bound in best_obj):
        raise ValueError(f"[ERROR] best_obj must hold numbers, got "
                         f"{[type(bound).__name__ for bound in best_obj]}.")
    if any(not float(bound).is_integer() for bound in best_obj):  # these count weight variables
        raise ValueError(f"[ERROR] best_obj = {best_obj} must hold whole numbers; where a weight is "
                         f"fractional, pass that round's \"integer_obj_fun_value\" instead.")
    best_obj = [int(bound) for bound in best_obj]
    if obj_var is None:
        raise ValueError("[ERROR] obj_var must hold one list of weight variables per round, got None.")
    obj_var = list(obj_var)  # accept any sequence of rows
    if not all(isinstance(row, list) for row in obj_var):
        raise ValueError("[ERROR] obj_var must hold one list of weight variables per round.")
    if len(obj_var) == Round + 1 and (obj_var[0] == []) != (obj_var[-1] == []):
        obj_var = obj_var[1:] if obj_var[0] == [] else obj_var[:-1]
    if len(obj_var) != Round:
        raise ValueError(f"[ERROR] obj_var holds {len(obj_var)} round(s) but Round = {Round}.")
    if not all(obj_var):
        raise ValueError("[ERROR] obj_var holds an empty round, so Matsui's bounds do not apply to "
                         "it. Every round searched must carry weight; check the goal and the scope.")
    if GroupConstraintChoice != 1:
        raise ValueError(f"Currently only support GroupConstraintChoice = 1, but got {GroupConstraintChoice}.")
    if GroupNumForChoice < 1:
        raise ValueError(f"GroupNumForChoice = {GroupNumForChoice} must be at least 1.")

    if not hasattr(gen_matsui_constraints_sat, "_counter"):  # Use function attribute to set global counter
        gen_matsui_constraints_sat._counter = 0
    if len(best_obj) == Round-1:
        best_obj = [0] + best_obj
    Main_Vars = list([])
    for r in range(Round):
        for i in range(len(obj_var[Round - 1 - r])):
            Main_Vars += [obj_var[Round - 1 - r][i]]
    if obj_sat >= len(Main_Vars):
        print(f"[INFO] obj_sat = {obj_sat} >= {len(Main_Vars)} weight variables, trivially satisfied.")
        return []
    gen_matsui_constraints_sat._counter += 1
    dummy_var = [
        [
            f'dummy_matsui_{gen_matsui_constraints_sat._counter}_{i}_{j}'
            for j in range(obj_sat)
        ]
        for i in range(len(Main_Vars) - 1)
    ]
    constraints = gen_sequential_encoding_sat(hw_list=Main_Vars, weight=obj_sat, dummy_variables=dummy_var)

    MatsuiRoundIndex = []
    if GroupConstraintChoice == 1:
        for group in range(GroupNumForChoice):
            for round_offset in range(1, Round - group + 1):
                MatsuiRoundIndex.append([group, group + round_offset])

    for matsui_count in range(0, len(MatsuiRoundIndex)):
        StartingRound = MatsuiRoundIndex[matsui_count][0]
        EndingRound = MatsuiRoundIndex[matsui_count][1]
        PartialCardinalityCons = obj_sat - best_obj[StartingRound] - best_obj[Round-EndingRound]
        left = sum(len(obj_var[i]) for i in range(Round - StartingRound, Round))
        right = sum(len(obj_var[i]) for i in range(Round - EndingRound, Round)) - 1
        constraints += gen_matsui_partial_cardinality_sat(
            Main_Vars,
            dummy_var,
            obj_sat,
            left,
            right,
            PartialCardinalityCons,
        )
    return constraints


def gen_matsui_partial_cardinality_sat(obj_var, dummy_var, k, left, right, m):
    """Generate partial cardinality SAT clauses for Matsui constraints."""
    if not isinstance(obj_var, list) or len(obj_var) == 0:
        raise ValueError("obj_var must be a non-empty list.")
    if not isinstance(dummy_var, list) or len(dummy_var) != len(obj_var) - 1:
        raise ValueError("dummy_var must be a list with length equal to len(obj_var) - 1.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if not isinstance(left, int) or left < 0:
        raise ValueError(f"[ERROR] left index = {left} must be a non-negative integer.")
    if not isinstance(right, int) or right < -1 or right >= len(obj_var):
        raise ValueError(f"[ERROR] right index = {right} must lie in [-1, {len(obj_var) - 1}].")
    if left > right + 1:  # the caller's own left <= right + 1 never breaks, so this is an index bug
        raise ValueError(f"[ERROR] window [left, right] = [{left}, {right}] is inverted; an empty "
                         f"window is written right = left - 1.")
    if not isinstance(m, int) or m < 0:
        raise ValueError(f"[ERROR] m = {m} must be a non-negative integer: obj_sat is below the "
                         f"weight best_obj already forces.")
    if right < left:  # empty window: nothing in it to count, so nothing to constrain
        return []

    n = len(obj_var)
    add_cons = []

    if m > 0:
        if left == 0 and right < n - 1:
            for i in range(1, right + 1):
                add_cons.append(f"-{obj_var[i]} -{dummy_var[i - 1][m - 1]}")

        if left > 0 and right == n - 1:
            for i in range(0, k - m):
                add_cons.append(f"{dummy_var[left - 1][i]} -{dummy_var[right - 1][i + m]}")
            for i in range(0, k - m + 1):
                add_cons.append(f"{dummy_var[left - 1][i]} -{obj_var[right]} -{dummy_var[right - 1][i + m - 1]}")

        if left > 0 and right < n - 1:
            for i in range(0, k - m):
                add_cons.append(f"{dummy_var[left - 1][i]} -{dummy_var[right][i + m]}")

    elif m == 0:
        for i in range(left, right + 1):
            add_cons.append(f"-{obj_var[i]}")

    return add_cons


# ----------- Attack boundary and input constraints ------------- #
def gen_input_non_zero_constraints(cipher, config_model, bitwise):
    """Generate the standard nonzero input constraint.

    Args:
        cipher: The cipher whose input variables must be non-zero.
        config_model (dict): Model configuration (``model_type``, SAT encoding, ...).
        bitwise (bool): If True, apply the constraint per bit; otherwise per word
            (word-level for truncated goals).

    Returns:
        list[str]: The generated model constraint strings.
    """
    cons_vars = [var for cons in cipher.inputs_constraints for var in cons.input_vars]
    model_type = config_model.get("model_type", "milp").lower()
    encoding = config_model.get("atleast_encoding_sat", 1) if model_type == "sat" else None
    constraints = gen_predefined_constraints(
        model_type=model_type,
        cons_type="SUM_AT_LEAST",
        cons_vars=cons_vars,
        cons_value=1,
        bitwise=bitwise,
        encoding=encoding,
    )
    if model_type == "milp":
        binary_vars = [var_id for var in cons_vars for var_id in expand_var_ids(var, bitwise=bitwise)]
        if binary_vars:
            constraints.append("Binary\n" + " ".join(binary_vars))
    return constraints


def _cipher_boundary_vars(in_out, cipher):
    """Return the cipher's boundary variables for ``in_out`` (``"input"`` or ``"output"``)."""
    cons_vars = []
    if in_out == "input":
        if not hasattr(cipher, "inputs") or not isinstance(cipher.inputs, dict):
            raise ValueError("[ERROR] Cipher 'inputs' attribute invalid.")
        for input_name in cipher.inputs:
            cons_vars += cipher.inputs[input_name]
    elif in_out == "output":
        if not hasattr(cipher, "outputs") or not isinstance(cipher.outputs, dict):
            raise ValueError("[ERROR] Cipher 'outputs' attribute invalid.")
        for output_name in cipher.outputs:
            cons_vars += cipher.outputs[output_name]
    else:
        raise ValueError(f"[ERROR] Invalid in_out: {in_out}. Expected 'input' or 'output'.")

    if not cons_vars:
        raise ValueError(f"[ERROR] Cipher has no {in_out} variables.")
    return cons_vars


def normalize_fixed_value_bits(fixed_value, bit_count, value_name):
    """Normalize a binary or hexadecimal fixed mask/difference string to bits.

    A value wider than ``bit_count`` is rejected, never truncated: truncation would silently pin
    the wrong bits.
    """

    text = fixed_value.strip().lower()
    if text.startswith("0b"):
        digits = text[2:]
        if any(c not in "01" for c in digits):  # 0x validates via int(); 0b must too
            raise ValueError(
                f"[ERROR] Invalid {value_name} format: {fixed_value}. "
                "Expected binary (0b...) or hexadecimal (0x...) string."
            )
        bits = digits.zfill(bit_count)
    elif text.startswith("0x"):
        try:
            bits = bin(int(text, 16))[2:].zfill(bit_count)
        except ValueError as exc:
            raise ValueError(
                f"[ERROR] Invalid {value_name} format: {fixed_value}. "
                "Expected binary (0b...) or hexadecimal (0x...) string."
            ) from exc
    else:
        raise ValueError(
            f"[ERROR] Invalid {value_name} format: {fixed_value}. "
            "Expected binary (0b...) or hexadecimal (0x...) string."
        )

    if len(bits) > bit_count:
        raise ValueError(
            f"[ERROR] {value_name} has {len(bits)} bits but the {bit_count}-bit boundary was expected."
        )
    return bits


def state_bits(cons_vars, fixed_value, value_name="value"):
    """``[(var_id, bit)]`` for ``cons_vars`` pinned to ``fixed_value``, MSB first.

    The word count and the word width come from the variables themselves, so a caller cannot
    disagree with the primitive about how a state splits. ``fixed_value`` is the whole state as
    ``"0x..."`` / ``"0b..."`` and must match the target's width.
    """
    ids = state_var_ids(cons_vars)
    bits = normalize_fixed_value_bits(fixed_value, len(ids), value_name)
    return [(var_id, int(bit)) for var_id, bit in zip(ids, bits)]


def state_var_ids(cons_vars):
    """The model variable IDs of a state, MSB first.

    Split out so reading a value off a solution uses the same order as pinning one into a model;
    the two directions disagreeing would make a claim written here unreadable there.
    """
    return [var_id for var in cons_vars for var_id in expand_var_ids(var, bitwise=True)]


def _fix_vars_to_bits(cons_vars, fixed_value, model_type, value_name="value"):
    """Pin ``cons_vars`` to ``fixed_value`` as model constraints; see ``state_bits``."""
    return [constraint
            for var_id, bit in state_bits(cons_vars, fixed_value, value_name)
            for constraint in gen_predefined_constraints(model_type, "EXACTLY", [var_id], bit,
                                                         bitwise=False)]


def gen_fixed_input_output_constraints(in_out, fixed_value, cipher, model_type, value_name="value"):
    """Generate constraints that fix input/output differences or masks."""

    if model_type not in ("milp", "sat"):
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")

    cons_vars = _cipher_boundary_vars(in_out, cipher)
    return _fix_vars_to_bits(cons_vars, fixed_value, model_type, value_name)


def gen_fixed_activity_constraints(cons_vars, active_positions, model_type="milp", bitwise=False):
    """Fix an activity pattern over ``cons_vars``: positions in ``active_positions`` are 1, the
    rest 0.

    ``bitwise=False`` counts one position per variable (word-level, for a truncated model);
    ``bitwise=True`` counts one per bit. Positions index the expanded list, so with ``bitwise``
    the i-th position is the i-th BIT across all the variables.
    """
    ids = [vid for var in cons_vars for vid in expand_var_ids(var, bitwise=bitwise)]
    active = set(active_positions)
    constraints = []
    for i, var_id in enumerate(ids):
        constraints += gen_predefined_constraints(
            model_type, "EXACTLY", [var_id], 1 if i in active else 0, bitwise=False
        )
    return constraints


def gen_fixed_state_constraints(cipher, function, rnd, layer, fixed_value, model_type,
                                nwords=None, value_name="value"):
    """Fix the state of ``function`` at round ``rnd``, layer ``layer`` to ``fixed_value``.

    Addresses ``cipher.functions[function].vars[rnd][layer]``, so the word count and the word
    width are read from the cipher. ``fixed_value`` is the whole state as a ``"0x..."``/``"0b..."`` 
    string, MSB first, and must match the target's width. Pick only the first ``nwords`` words.
    """
    if model_type not in ("milp", "sat"):
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")
    if function not in cipher.functions:
        raise ValueError(f"Unknown function: {function!r}. Expected one of {list(cipher.functions)}.")

    fn = cipher.functions[function]
    if not 0 <= rnd < len(fn.vars):
        raise ValueError(f"Round {rnd} out of range for {function!r} (0..{len(fn.vars) - 1}).")
    if not 0 <= layer < len(fn.vars[rnd]):
        raise ValueError(f"Layer {layer} out of range for {function!r} round {rnd} "
                         f"(0..{len(fn.vars[rnd]) - 1}).")
    if nwords is not None and not 0 < nwords <= fn.nbr_words:
        raise ValueError(f"nwords {nwords} out of range for {function!r} (1..{fn.nbr_words}).")

    cons_vars = fn.vars[rnd][layer][: nwords if nwords is not None else fn.nbr_words]
    return _fix_vars_to_bits(cons_vars, fixed_value, model_type, value_name)


def gen_required_fixed_boundary_constraints(cipher, input_value, output_value, model_type):
    """Generate constraints fixing the cipher's input and/or output boundary values.

    The caller decides when this applies (e.g. a specific goal); at least one of
    ``input_value`` / ``output_value`` must be provided.

    Args:
        cipher: The cipher whose input/output boundary is fixed.
        input_value: The fixed input value, or None to leave the input free.
        output_value: The fixed output value, or None to leave the output free.
        model_type (str): ``"milp"`` or ``"sat"``.

    Returns:
        list[str]: The generated model constraint strings.
    """
    if input_value is None and output_value is None:
        raise ValueError("Either an input or output value must be specified to fix the boundary.")

    constraints = []
    if input_value is not None:
        constraints.extend(gen_fixed_input_output_constraints("input", input_value, cipher, model_type))
    if output_value is not None:
        constraints.extend(gen_fixed_input_output_constraints("output", output_value, cipher, model_type))
    return constraints
