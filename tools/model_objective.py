import heapq
import re


# **************************************************************************** #
#  This module provides functions for processing objective functions in MILP/SAT-based automated cryptanalysis, including:
# 1. Detecting S-box operators and checking for decimal weights
# 2. Generating valid weight combinations
# 3. Parsing and grouping objective function variables
# 4. Computing per-round objective values from solutions
# **************************************************************************** #


# -------------------- S-box Detection and Property Checks --------------------
def detect_Sbox(cipher): # Detect and return the first Sbox operator in the cipher
    for f in cipher.functions:
        for r in range(1, cipher.functions[f].nbr_rounds + 1):
            for l in range(cipher.functions[f].nbr_layers + 1):
                for cons in cipher.functions[f].constraints[r][l]:
                    if "Sbox" in cons.__class__.__name__:
                        return cons
    return None

def has_Sbox_with_decimal_weights(cipher, goal):
    Sbox = detect_Sbox(cipher)
    if Sbox and goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
        if goal in {'DIFFERENTIALPATH_PROB', 'DIFFERENTIAL_PROB'}:
            table = Sbox.computeDDT()
        weights = Sbox.gen_weights(table)
        return any(not float(w).is_integer() for w in weights)
    return False

def linear_combinations_bounds(weights, upper_bound, lower_bound=-1): # Enumerate all integer linear combinations of weights such that the sum is within (lower_bound, upper_bound].
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

def generate_obj_decimal_coms(Sbox, table, min_int_obj_value, max_obj_value): # generate all combinations of decimal objective function value, with integer value >= min_int_obj_value, and the total value < max_val.
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
def gen_obj_fun_variables(obj_fun, obj_fun_decimal=False): # In the case of a decimal-weighted objective function, parse objective function variables and group them into separate components for SAT modeling
    obj_fun_var_int = []
    for obj_fun_r in obj_fun:
        obj_fun_var_r_int = []
        for obj in obj_fun_r:
            terms = [t.strip() for t in obj.split('+')]
            for term in terms:
                parts = term.split()
                if len(parts) == 1:
                    obj_fun_var_r_int.append(parts[0])
                elif len(parts) >= 2:
                    coef, var = parts[0], parts[1]
                    if float(coef).is_integer():
                        obj_fun_var_r_int.append(var)
        obj_fun_var_int.append(obj_fun_var_r_int)
    if not obj_fun_decimal:
        return obj_fun_var_int
    else:
        decimal_vars = []
        for obj_fun_r in obj_fun:
            if obj_fun_r:
                terms = [t.strip() for t in obj_fun_r[0].split('+')]
                break
        for term in terms:
            parts = term.split()
            if len(parts) >= 2:
                coef, var = parts[0], parts[1]
                if not float(coef).is_integer():
                    decimal_vars.append(coef)

        obj_fun_var_dec = {k: [] for k in decimal_vars}

        for obj_fun_r in obj_fun:
            obj_fun_var_r_dec = {k: [] for k in decimal_vars}
            for obj in obj_fun_r:
                terms = [t.strip() for t in obj.split('+')]
                for term in terms:
                    parts = term.split()
                    if len(parts) >= 2 and (not float(parts[0]).is_integer()):
                        obj_fun_var_r_dec[coef].append(parts[1])
            for k in decimal_vars:
                obj_fun_var_dec[k].append(obj_fun_var_r_dec[k])
        return obj_fun_var_int, [obj_fun_var_dec[k] for k in decimal_vars]


# -------------------- Objective Function Value Calculation -------------------
def cal_round_obj_fun_values_from_solution(obj_fun, solution): # Calculate the objective function value for each round from the solution
    round_obj_fun_values = []
    for obj_fun_r in obj_fun:
        w = 0
        for obj_fun_r_i in obj_fun_r:
            terms = [t.strip() for t in obj_fun_r_i.split('+')]
            for term in terms:
                match = re.match(r'(\d*\.?\d*)\s*(\w+)', term.strip())
                if match:
                    coefficient = float(match.group(1)) if match.group(1) != '' else 1  # Default coefficient is 1 if not found
                    variable = match.group(2)
                    if variable in solution:
                        w += coefficient * solution[variable]
                else:
                    print(f"Warning: Unable to parse '{term.strip()}'")
        round_obj_fun_values.append(w)
    return round_obj_fun_values
