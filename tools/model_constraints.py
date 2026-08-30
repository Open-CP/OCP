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
    elif model_type == "sat" and pysat_import:
        # ``None`` is defaulted downstream in ``_pysat_cardinality_constraints``; here we
        # only reject an explicitly out-of-range encoding.
        if encoding is not None and encoding not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            raise ValueError(
                f"[ERROR] Invalid encoding = {encoding}, refer "
                "https://pysathq.github.io/docs/html/api/card.html"
            )
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


def gen_constraints_sum_at_most(model_type, cons_vars, cons_value, encoding="SEQUENTIAL"):
    """Constrain the sum of the selected variables to be at most ``cons_value``."""
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]
    elif model_type == "sat" and (encoding == "SEQUENTIAL" or encoding is None):
        return gen_sequential_encoding_sat(cons_vars, cons_value)
    elif model_type == "sat" and pysat_import:
        if not isinstance(encoding, int):
            encoding = 1  # Default to 1 if the encoding is not specified as an integer
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
    elif model_type == "sat" and pysat_import:
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
    if not isinstance(weight, int) or weight < 0 or weight > n:
        raise ValueError(f"weight should be an integer: 0 <= weight <= n (n={n}), got {weight}")
    if weight == 0:
        return [f'-{var}' for var in hw_list]
    if weight >= n:
        return []
    if dummy_variables is None:
        gen_sequential_encoding_sat._counter += 1
        prefix = f'dummy_seq_{gen_sequential_encoding_sat._counter}'
        dummy_variables = [[f'{prefix}_{i}_{j}' for j in range(weight)] for i in range(n - 1)]
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
    """Generate Matsui branch-and-bound constraints for MILP models."""
    if Round < 2:
        raise ValueError(f"Round = {Round} must be at least 2.")
    if len(best_obj) != Round - 1:
        raise ValueError(f"best_obj = {best_obj} length must be Round-1 = {Round - 1}.")
    if obj_fun is not None:  # copy so the caller's list is not mutated
        obj_fun = list(obj_fun)
    while obj_fun and obj_fun[-1] == []:  # drop trailing empty rounds
        obj_fun.pop()
    if obj_fun is None or len(obj_fun) != Round or not all(isinstance(obj, list) for obj in obj_fun):
        raise ValueError(f"obj_fun = {obj_fun} must be a list of lists, and with length equal to Round = {Round}.")
    if cons_type not in ["ALL", "UPPER", "LOWER"]:
        raise ValueError(f"cons_type = {cons_type} must be one of ['ALL', 'UPPER', 'LOWER'].")

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
    """Generate Matsui branch-and-bound constraints for SAT models."""
    if Round < 2:
        raise ValueError(f"Round = {Round} must be at least 2.")
    if len(best_obj) != Round - 1:
        raise ValueError(f"best_obj length = {len(best_obj)} must be (Round-1) = {Round - 1}.")
    if not isinstance(obj_sat, int) or obj_sat <= 0:
        raise ValueError(f"obj_sat = {obj_sat} must be a positive integer.")
    if obj_var is not None:  # copy so the caller's list is not mutated
        obj_var = list(obj_var)
    while obj_var and obj_var[-1] == []:  # drop trailing empty rounds
        obj_var.pop()
    if obj_var is None or len(obj_var) != Round or not all(isinstance(row, list) for row in obj_var):
        obj_var_len = "None" if obj_var is None else len(obj_var)
        raise ValueError(f"obj_var must be a list of lists, and with length = {obj_var_len} equal to Round = {Round}.")
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
        left = 0
        for i in range(StartingRound):
            left += len(obj_var[i])
        right = 0
        for i in range(EndingRound):
            right += len(obj_var[i])
        right -= 1
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
        raise ValueError("left index must be a non-negative integer.")
    if not isinstance(right, int) or right >= len(obj_var):
        raise ValueError(f"right index = {right} out of range of obj_var = {len(obj_var)}.")
    if not isinstance(m, int) or m < 0:
        raise ValueError(f"m={m} must be a non-negative integer.")

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
    encoding = config_model.get("atleast_encoding_sat", "SEQUENTIAL") if model_type == "sat" else None
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
            raise ValueError("[WARNING] Cipher 'inputs' attribute invalid.")
        for input_name in cipher.inputs:
            cons_vars += cipher.inputs[input_name]
    elif in_out == "output":
        if not hasattr(cipher, "outputs") or not isinstance(cipher.outputs, dict):
            raise ValueError("[WARNING] Cipher 'outputs' attribute invalid.")
        for output_name in cipher.outputs:
            cons_vars += cipher.outputs[output_name]
    else:
        raise ValueError(f"[WARNING] Invalid in_out: {in_out}. Expected 'input' or 'output'.")

    if not cons_vars:
        raise ValueError(f"[WARNING] Cipher has no {in_out} variables.")
    return cons_vars


def normalize_fixed_value_bits(fixed_value, bit_count, value_name):
    """Normalize a binary or hexadecimal fixed mask/difference string to bits."""

    text = fixed_value.strip().lower()
    if text.startswith("0b"):
        digits = text[2:]
        if any(c not in "01" for c in digits):  # 0x validates via int(); 0b must too
            raise ValueError(
                f"[WARNING] Invalid {value_name} format: {fixed_value}. "
                "Expected binary (0b...) or hexadecimal (0x...) string."
            )
        bits = digits.zfill(bit_count)
    elif text.startswith("0x"):
        try:
            bits = bin(int(text, 16))[2:].zfill(bit_count)
        except ValueError as exc:
            raise ValueError(
                f"[WARNING] Invalid {value_name} format: {fixed_value}. "
                "Expected binary (0b...) or hexadecimal (0x...) string."
            ) from exc
    else:
        raise ValueError(
            f"[WARNING] Invalid {value_name} format: {fixed_value}. "
            "Expected binary (0b...) or hexadecimal (0x...) string."
        )

    if len(bits) > bit_count:
        raise ValueError(
            f"[WARNING] {value_name} has {len(bits)} bits but the {bit_count}-bit boundary was expected."
        )
    return bits


def gen_fixed_input_output_constraints(in_out, fixed_value, cipher, model_type, value_name="value"):
    """Generate constraints that fix input/output differences or masks."""

    if model_type not in ("milp", "sat"):
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")

    cons_vars = _cipher_boundary_vars(in_out, cipher)
    bit_count = sum(var.bitsize for var in cons_vars)
    bits = normalize_fixed_value_bits(fixed_value, bit_count, value_name)

    constraints = []
    offset = 0
    for var in cons_vars:
        for bit_index in range(var.bitsize):
            bit = bits[offset + bit_index]
            var_id = var.ID if var.bitsize == 1 else f"{var.ID}_{bit_index}"
            if model_type == "sat":
                constraints.append(var_id if bit == "1" else f"-{var_id}")
            else:  # milp
                constraints.append(f"{var_id} = {bit}")
                constraints.append("Binary\n" + var_id)
        offset += var.bitsize
    return constraints


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
