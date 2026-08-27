"""Objective-function processing for MILP/SAT-based automated cryptanalysis.

Provides:

1. Detecting S-box operators and checking for decimal weights.
2. Generating valid weight combinations.
3. Parsing and grouping objective-function variables.
4. Computing per-round objective values from solutions.
"""

import heapq
import re


# -------------------- S-box Detection and Property Checks --------------------
def detect_Sbox(cipher):
    """Detect and return the first Sbox operator in the cipher (or ``None``)."""
    for f in cipher.functions:
        for r in range(1, cipher.functions[f].nbr_rounds + 1):
            for l in range(cipher.functions[f].nbr_layers + 1):
                for cons in cipher.functions[f].constraints[r][l]:
                    if "Sbox" in cons.__class__.__name__:
                        return cons
    return None

def has_Sbox_with_decimal_weights(cipher, goal):
    """Return True if the cipher's S-box has non-integer (decimal) weights for ``goal``."""
    Sbox = detect_Sbox(cipher)
    if Sbox and goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB', 'LINEARPATH_CORR', 'LINEARHULL_CORR'}:
        if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
            table = Sbox.computeDDT()
        else:
            table = Sbox.computeLAT()
        weights = Sbox.gen_weights(table)
        return any(not float(w).is_integer() for w in weights)
    return False

def linear_combinations_bounds(weights, upper_bound, lower_bound=-1):
    """Enumerate all integer linear combinations of ``weights`` whose sum is within ``(lower_bound, upper_bound]``.

    Pruning relies on each increment strictly raising the sum toward ``upper_bound``, so
    ``weights`` must be strictly positive; a zero or negative weight makes the enumeration
    unbounded and raises ``ValueError``.
    """
    if any(w <= 0 for w in weights):
        raise ValueError(f"linear_combinations_bounds requires strictly positive weights, got {weights}.")
    n = len(weights)
    seen = set()
    result = []
    # Each state is (sum, coeffs), Start with zero combination
    initial = (0.0, (0,) * n)
    heap = [initial]
    seen.add(initial[1])
    EPS = 0.001
    while heap:
        total, coeffs = heapq.heappop(heap)
        if lower_bound <= total <= (upper_bound + EPS):
            result.append((total, coeffs))
        # Try to increment each coefficient
        for i in range(n):
            new_coeffs = list(coeffs)
            new_coeffs[i] += 1
            new_coeffs = tuple(new_coeffs)
            if new_coeffs not in seen:
                new_sum = total + weights[i]
                if new_sum <= (upper_bound + EPS):
                    heapq.heappush(heap, (new_sum, new_coeffs))
                    seen.add(new_coeffs)
    return result

def generate_obj_decimal_coms(Sbox, table, min_int_obj_value, max_obj_value):
    """Generate decimal objective-value combinations with integer value >= ``min_int_obj_value``, and total < ``max_obj_value``."""
    obj_decimal_coms = []
    weights = Sbox.gen_weights(table)
    combs = linear_combinations_bounds(weights, max_obj_value, min_int_obj_value)
    integers_weight, floats_weight = Sbox.gen_integer_float_weight(table)
    weight_pattern_map = {str(w): Sbox.gen_weight_pattern_sat(integers_weight, floats_weight, w) for w in weights}

    for total, coeffs in combs:
        obj = [0 for _ in range(max(integers_weight)+len(floats_weight))]
        for i in range(len(coeffs)):
            if coeffs[i] > 0:
                w = weights[i]
                pattern = weight_pattern_map[str(w)]
                for j in range(len(obj)):
                    obj[j] += coeffs[i] * pattern[j]
        decimal_com = obj[max(integers_weight):]
        int_obj = sum(obj[:max(integers_weight)])
        if int_obj >= min_int_obj_value and [total, int_obj, decimal_com] not in obj_decimal_coms:
            obj_decimal_coms.append([total, int_obj, decimal_com])
    return obj_decimal_coms


# ------------------ Objective Function Variable Processing -------------------
def parse_objective_term(term):
    """Parse an objective term into ``(coefficient, variable)``.

    Supports a single ``[coefficient] variable`` term with a NON-NEGATIVE coefficient
    (examples: ``x``, ``2 x``, ``0.5000 p0``, ``2x``). Signs (``-``) and operators
    (``*``) are not supported: objective expressions are assumed to be ``+``-separated
    non-negative terms. Returns ``None`` for empty or unsupported terms.
    """

    term = term.strip()
    if not term:
        return None
    match = re.match(r"^(\d*\.?\d*)\s*([A-Za-z_]\w*)$", term)
    if not match:
        return None
    coeff_str = match.group(1)
    try:
        coefficient = float(coeff_str) if coeff_str else 1.0
    except ValueError:
        return None
    return coefficient, match.group(2)


def gen_obj_fun_variables(obj_fun, obj_fun_decimal=False):
    """Parse objective-function variables and group them into components for SAT modeling.

    For a decimal-weighted objective function, integer- and decimal-coefficient variables
    are grouped separately.
    """
    obj_fun_var_int = []
    for obj_fun_r in obj_fun:
        obj_fun_var_r_int = []
        for obj in obj_fun_r:
            terms = [t.strip() for t in obj.split('+')]
            for term in terms:
                parsed = parse_objective_term(term)
                if parsed is None:
                    continue
                coefficient, variable = parsed
                if coefficient.is_integer():
                    obj_fun_var_r_int.append(variable)
        obj_fun_var_int.append(obj_fun_var_r_int)
    if not obj_fun_decimal:
        return obj_fun_var_int
    else:
        decimal_vars = []
        for obj_fun_r in obj_fun:
            for obj in obj_fun_r:
                for term in (t.strip() for t in obj.split('+')):
                    parsed = parse_objective_term(term)
                    if parsed is None:
                        continue
                    coefficient, _ = parsed
                    key = str(coefficient)
                    if not coefficient.is_integer() and key not in decimal_vars:
                        decimal_vars.append(key)

        obj_fun_var_dec = {k: [] for k in decimal_vars}

        for obj_fun_r in obj_fun:
            obj_fun_var_r_dec = {k: [] for k in decimal_vars}
            for obj in obj_fun_r:
                terms = [t.strip() for t in obj.split('+')]
                for term in terms:
                    parsed = parse_objective_term(term)
                    if parsed is None:
                        continue
                    coefficient, variable = parsed
                    key = str(coefficient)
                    if key in obj_fun_var_r_dec:
                        obj_fun_var_r_dec[key].append(variable)
            for k in decimal_vars:
                obj_fun_var_dec[k].append(obj_fun_var_r_dec[k])
        return obj_fun_var_int, [obj_fun_var_dec[k] for k in decimal_vars]


# -------------------- Objective Function Value Calculation -------------------
def cal_round_obj_fun_values_from_solution(obj_fun, solution):
    """Calculate the objective-function value for each round from the solution."""
    round_obj_fun_values = []
    for obj_fun_r in obj_fun:
        w = 0
        for obj_fun_r_i in obj_fun_r:
            terms = [t.strip() for t in obj_fun_r_i.split('+')]
            for term in terms:
                if not term:  # empty term (e.g. a trailing '+'); nothing to parse
                    continue
                parsed = parse_objective_term(term)
                if parsed is not None:
                    coefficient, variable = parsed
                    if variable in solution:
                        w += coefficient * solution[variable]
                else:
                    print(f"[WARNING] Unable to parse objective term '{term.strip()}'.")
        round_obj_fun_values.append(w)
    return round_obj_fun_values
