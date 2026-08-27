"""SAT/MILP constraint of the operators.

These are the building blocks each operator uses to generate its model: XOR / n-XOR /
matrix constraints (bit and word level), the MILP ``Binary`` declaration.
"""

from itertools import combinations


def _require_string_variables(*variables):
    if not all(isinstance(variable, str) for variable in variables):
        raise TypeError(f"variables {variables} must be strings")


def _require_string_list(variables):
    if not isinstance(variables, list) or not all(isinstance(v, str) for v in variables):
        raise TypeError(f"variables {variables} must be provided as a list of strings")


def _iter_group_vars(var_groups):
    """Yield every variable name from the groups, rejecting bare strings and non-strings.

    Each group must be a list/tuple of variable-name strings. A bare string (which
    would otherwise iterate character by character) or a non-string variable is an error.
    """
    for group in var_groups:
        if isinstance(group, str):
            raise TypeError(f"each variable group must be a list of names, not a bare string {group!r}")
        for v in group:
            if not isinstance(v, str):
                raise TypeError(f"variable {v!r} must be a string")
            yield v


def binary_declaration(*var_groups):
    """Return the MILP ``Binary`` declaration for the given variable groups."""
    return 'Binary\n' + ' '.join(_iter_group_vars(var_groups))


def integer_declaration(*var_groups):
    """Return the MILP ``Integer`` declaration for the given variable groups."""
    return 'Integer\n' + ' '.join(_iter_group_vars(var_groups))


def gen_equivalence_constraints(var_in, var_out, model_type):
    """Return the constraints enforcing ``var_in[i] == var_out[i]`` bitwise.

    SAT returns the clause pair ``-a b`` / ``a -b`` per bit; MILP returns ``a - b = 0`` per bit.

    Valid propagations (in -> out): 0 -> 0, 1 -> 1.
    """
    if len(var_in) != len(var_out):
        raise ValueError(f"gen_equivalence_constraints: var_in and var_out must have equal length, got {len(var_in)} vs {len(var_out)}")
    if model_type == 'sat':
        return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
    elif model_type == 'milp':
        return [f"{vin} - {vout} = 0" for vin, vout in zip(var_in, var_out)]
    raise ValueError(f"gen_equivalence_constraints: unknown model type '{model_type}'")


def gen_or_constraints(vin1, vin2, vout, model_type):
    """SAT/MILP constraints for the boolean OR ``vin1 | vin2 = vout``.

    Valid propagations ((vin1, vin2) -> vout):
    (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 1.
    """
    _require_string_variables(vin1, vin2, vout)
    if model_type == 'sat':
        return [f'-{vin1} {vout}', f'-{vin2} {vout}', f'-{vout} {vin1} {vin2}']
    elif model_type == 'milp':
        return [f'{vout} - {vin1} >= 0', f'{vout} - {vin2} >= 0', f'{vin1} + {vin2} - {vout} >= 0']
    raise ValueError(f"gen_or_constraints: unknown model type '{model_type}'")


def gen_implication_constraints(vin, vout, model_type):
    """SAT/MILP constraints for the implication ``vin -> vout`` (i.e. ``vin <= vout``).

    Valid propagations (vin -> vout): 0 -> 0, 0 -> 1, 1 -> 1  (1 -> 0 is forbidden).
    """
    _require_string_variables(vin, vout)
    if model_type == 'sat':
        return [f'-{vin} {vout}']
    elif model_type == 'milp':
        return [f'{vout} - {vin} >= 0']
    raise ValueError(f"gen_implication_constraints: unknown model type '{model_type}'")


def gen_xor_constraints(vin1, vin2, vout, model_type, v_dummy=None, version=0):
    """SAT/MILP constraints for the bitwise XOR ``vin1 ^ vin2 = vout``.

    Valid propagations ((vin1, vin2) -> vout):
    (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0.
    Different versions encode the same relation.
    """
    _require_string_variables(vin1, vin2, vout)
    if model_type == "sat":
        if version == 0:
            return [
                f'{vin1} {vin2} -{vout}',
                f'{vin1} -{vin2} {vout}',
                f'-{vin1} {vin2} {vout}',
                f'-{vin1} -{vin2} -{vout}',
            ]
        else:
            raise ValueError(f"Unknown version {version} for XOR in SAT.")
    elif model_type == 'milp':
        if version == 0:
            return [f'{vin1} + {vin2} - {vout} >= 0',
                    f'{vin2} + {vout} - {vin1} >= 0',
                    f'{vin1} + {vout} - {vin2} >= 0',
                    f'{vin1} + {vin2} + {vout} <= 2']
        elif version == 1:
            _require_string_variables(v_dummy)
            return [f'{vin1} + {vin2} + {vout} - 2 {v_dummy} >= 0',
                    f'{vin1} + {vin2} + {vout} <= 2',
                    f'{v_dummy} - {vin1} >= 0',
                    f'{v_dummy} - {vin2} >= 0',
                    f'{v_dummy} - {vout} >= 0']
        elif version == 2:
            _require_string_variables(v_dummy)
            return [f'{vin1} + {vin2} + {vout} - 2 {v_dummy} = 0']
        else:
            raise ValueError(f"Unknown version {version} for XOR in MILP.")
    else:
        raise ValueError(f"gen_xor_constraints: unknown model type '{model_type}'")


def gen_word_xor_constraints(vin1, vin2, vout, model_type, v_dummy=None, version=0):
    """SAT/MILP constraints for the word-level (truncated) XOR of the activity bits vin1, vin2 -> vout.

    Valid propagations on activity ((vin1, vin2) -> vout), 0 = inactive / 1 = active:
    (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0 or 1  (both active: output activity undetermined).
    """
    _require_string_variables(vin1, vin2, vout)
    if model_type == "sat":
        if version == 0:
            return [f'{vin1} {vin2} -{vout}',
                    f'{vin1} -{vin2} {vout}',
                    f'-{vin1} {vin2} {vout}']
        else:
            raise ValueError(f"Unknown version {version} for Word-wise XOR in SAT.")
    elif model_type == 'milp':
        if version == 0:
            return [f'{vin1} + {vin2} - {vout} >= 0',
                    f'{vin2} + {vout} - {vin1} >= 0',
                    f'{vin1} + {vout} - {vin2} >= 0']
        elif version == 1:
            _require_string_variables(v_dummy)
            return [f'{vin1} + {vin2} + {vout} - 2 {v_dummy} >= 0',
                    f'{v_dummy} - {vin1} >= 0',
                    f'{v_dummy} - {vin2} >= 0',
                    f'{v_dummy} - {vout} >= 0']
        else:
            raise ValueError(f"Unknown version {version} for Word-wise XOR in MILP.")
    else:
        raise ValueError(f"gen_word_xor_constraints: unknown model type '{model_type}'")


def gen_nxor_constraints(vin, vout, model_type, v_dummy=None, version=0):
    """SAT/MILP constraints for the n-ary bitwise XOR ``vin[0] ^ ... ^ vin[n-1] = vout``.

    Valid propagations (vin -> vout): vout = vin[0] ^ ... ^ vin[n-1],
    iff ``sum(vin) + vout`` is even. E.g. n=3: (0,0,0) -> 0, (1,1,0) -> 0, (1,0,0) -> 1, (1,1,1) -> 1.
    """
    _require_string_list(vin)
    _require_string_variables(vout)
    constraints = []
    if model_type == "sat":
        for k in range(0, len(vin) + 1):  # All subsets (0 to n elements)
            for comb in combinations(vin, k):
                is_odd_parity = (len(comb) % 2 == 1)
                clause = [f"{vout}" if is_odd_parity else f"-{vout}"]
                clause += [f"-{v}" if v in comb else f"{v}" for v in vin]
                constraints.append(" ".join(clause))
        return constraints
    elif model_type == "milp":
        if version == 0:
            _require_string_variables(v_dummy)
            constraints += [" + ".join(vin) + " + " + vout + f" - 2 {v_dummy} = 0"]
            constraints += [f"{v_dummy} >= 0"]
            constraints += [f"{v_dummy} <= {int((len(vin)+1)/2)}"]
            return constraints
        elif version == 1:  # Reference: MILP-aided cryptanalysis of the future block cipher.
            _require_string_list(v_dummy)
            m = len(v_dummy)
            lhs = " + ".join(vin + [vout])
            lhs += "".join(f" - {2 * (m - j)} {v_dummy[j]}" for j in range(m))
            return [f"{lhs} = 0"]
        else:
            raise ValueError(f"Unknown version {version} for n-XOR in MILP.")
    else:
        raise ValueError(f"gen_nxor_constraints: unknown model type '{model_type}'")


def gen_word_nxor_constraints(vin, vout, model_type, v_dummy=None, version=0):
    """SAT/MILP constraints for the word-level (truncated) n-ary XOR of ``vin`` into ``vout``.

    Valid propagations (activity of inputs -> output): the number of active variables among
    ``vin`` and ``vout`` together is never exactly 1 -- it is 0 (all inactive) or >= 2. E.g. n=3:
    (0,0,0) -> 0; (1,1,0) -> 0 or 1; (1,0,0) -> 1  (a lone active input with vout=0 is forbidden).
    """
    _require_string_list(vin)
    _require_string_variables(vout)
    constraints = []
    if model_type == "milp":  # Reference: Related-Key Differential Analysis of the AES.
        constraints += [f"{' + '.join(vin)} - {vout} >= 0"]
        for k, ik in enumerate(vin):
            others = [x for j, x in enumerate(vin) if j != k]
            constraints.append(f"{' + '.join(others)} + {vout} - {ik} >= 0")
        return constraints
    elif model_type == "sat":
        constraints.append(" ".join([f"-{vout}"] + list(vin)))
        for k, ik in enumerate(vin):
            others = [x for j, x in enumerate(vin) if j != k]
            constraints.append(f"{' '.join(others)} {vout} -{ik}")
        return constraints
    else:
        raise ValueError(f"gen_word_nxor_constraints: unknown model type '{model_type}'")


def gen_matrix_row_constraints(vin, vout, model_type, v_dummy=None):
    """SAT/MILP constraints for one output bit = XOR of the input bits ``vin`` (a binary-matrix row).

    Dispatches by number of inputs: 1 -> equivalence, 2 -> XOR, >=3 -> n-XOR.
    Valid propagations (vin -> vout): vout = XOR of all vin bits (valid iff ``sum(vin) + vout`` is even).
    """
    _require_string_list(vin)
    _require_string_variables(vout)
    if len(vin) == 1:
        return gen_equivalence_constraints(vin, [vout], model_type)
    elif len(vin) == 2:
        return gen_xor_constraints(vin[0], vin[1], vout, model_type)
    elif len(vin) >= 3:
        if model_type == 'milp':
            _require_string_variables(v_dummy)
        return gen_nxor_constraints(vin, vout, model_type, v_dummy=v_dummy)
    else:
        raise ValueError("gen_matrix_row_constraints: at least one input variable is required.")


def gen_word_matrix_row_constraints(vin, vout, model_type, v_dummy=None):
    """Word-level (truncated) version of :func:`gen_matrix_row_constraints`, on activity variables.

    Dispatches by number of inputs: 1 -> equivalence, 2 -> word-XOR, >=3 -> word n-XOR.
    Valid propagations (activity): 1 input -> vout equals it; >=2 inputs -> the number of active
    variables among ``vin`` and ``vout`` is 0 or >= 2 (never exactly 1).
    """
    _require_string_list(vin)
    _require_string_variables(vout)
    if len(vin) == 1:
        return gen_equivalence_constraints(vin, [vout], model_type)
    elif len(vin) == 2:
        return gen_word_xor_constraints(vin[0], vin[1], vout, model_type)
    elif len(vin) >= 3:
        return gen_word_nxor_constraints(vin, vout, model_type)
    else:
        raise ValueError("gen_word_matrix_row_constraints: at least one input variable is required.")


def gen_matrix_constraints(bin_matrix, source_bits, target_bits, model_type, dummy_prefix=None):
    """SAT/MILP constraints for a whole binary matrix: ``target_bits[i]`` = XOR of the ``source_bits[j]``
    with ``bin_matrix[i][j] == 1`` (one row per target bit, dispatched by :func:`gen_matrix_row_constraints`).

    Pure relation, returns CONSTRAINTS ONLY. For MILP, pair with :func:`gen_matrix_declarations` to
    obtain the Binary/Integer declaration lines. ``dummy_prefix`` names the per-row n-XOR dummy
    (row ``i`` -> ``f"{dummy_prefix}_{i}"``) and is required for MILP when any row has >=3 active inputs.
    """
    model_list = []
    for i, row in enumerate(bin_matrix):
        var_in = [source_bits[j] for j, bit in enumerate(row) if bit == 1]
        v_dummy = f"{dummy_prefix}_{i}" if dummy_prefix is not None else None
        model_list.extend(gen_matrix_row_constraints(var_in, target_bits[i], model_type, v_dummy=v_dummy))
    return model_list


def gen_matrix_declarations(bin_matrix, source_bits, target_bits, dummy_prefix):
    """MILP Binary/Integer declarations matching :func:`gen_matrix_constraints`.

    Every bit appearing in a constraint is Binary; each row with >=3 active inputs contributes one
    Integer n-XOR dummy (``f"{dummy_prefix}_{i}"``). Returns the declaration line(s) to append.
    """
    binary_vars = {}       # order-preserving set of every bit that appears in a constraint
    integer_dummies = []   # rows with >=3 active inputs use an integer n-XOR dummy
    for i, row in enumerate(bin_matrix):
        var_in = [source_bits[j] for j, bit in enumerate(row) if bit == 1]
        for v in var_in + [target_bits[i]]:
            binary_vars[v] = None
        if len(var_in) >= 3:
            integer_dummies.append(f"{dummy_prefix}_{i}")
    declarations = [binary_declaration(list(binary_vars))]
    if integer_dummies:
        declarations.append(integer_declaration(integer_dummies))
    return declarations
