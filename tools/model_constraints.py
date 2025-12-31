try:
    from pysat.card import CardEnc
    from pysat.formula import IDPool
    vpool = IDPool(start_from=1000)
    pysat_import = True
except ImportError:
    print("[WARNING] pysat module can't be loaded")
    pysat_import = False


# **************************************************************************** #
# This module provides the unified interface for generating MILP/SAT model constraints for cryptanalysis, including:
# 1. Cipher Model Configuration
#    - Assign model versions based on attack goals
#    - Generate constraints and objective functions
# 2. Constraint Generation Utilities
#    - Predefined constraints (EXACTLY, AT_LEAST, SUM_AT_MOST, etc.)
#    - SAT sequential encoding for cardinality constraints
# 3. Advanced Search Strategies
#    - Matsui’s branch-and-bound acceleration techniques for MILP and SAT-based differential and linear trail searches
# **************************************************************************** #


# --------------------------- Model Configuration ---------------------------
def fill_functions_rounds_layers_positions(cipher, functions=None, rounds=None, layers=None, positions=None):
    """
    Fill in functions, rounds, layers, and positions to full coverage when the corresponding argument is None; otherwise, keep user-supplied values.

    Parameters:
        cipher (object): The cipher object.
        functions (list[str]): List of functions. If None, use all functions of the cipher. Example: ["PERMUTATION", "KEY_SCHEDULE", "SUBKEYS"].
        rounds (dict): Dictionary specifying rounds. If None, use all. Example: {"PERMUTATION": [1, 2, 3]}.
        layers (dict): Dictionary specifying layers. If None, use all. Example: {"PERMUTATION": {1: [0, 1], 2: [0, 1], 3: [0, 1]}}.
        positions (dict): Dictionary specifying positions. If None, use all. Example: {"PERMUTATION": {1: {0: [0, 1], 1: [0, 1]}, 2: {0: [0, 1], 1: [0, 1]}, 3: {0: [0, 1], 1: [0, 1]}}}.

    Returns:
        tuple: (functions, rounds, layers, positions)
    """
    if functions is None:
        functions = [f for f in cipher.functions]
    if rounds is None:
        rounds = {f: list(range(1, cipher.functions[f].nbr_rounds + 1)) for f in functions}
    if layers is None:
        layers = {f: {r: list(range(cipher.functions[f].nbr_layers+1)) for r in rounds[f]} for f in functions}
    if positions is None:
        positions = {f: {r: {l: list(range(len(cipher.functions[f].constraints[r][l]))) for l in layers[f][r]} for r in rounds[f]} for f in functions}
    return functions, rounds, layers, positions


def configure_model_version(cipher, goal, config_model): # Configure the model version for all operators in the cipher based on the attack goal and config_model.
    functions, rounds, layers, positions = config_model.get("functions"), config_model.get("rounds"), config_model.get("layers"), config_model.get("positions")

    if goal == 'DIFFERENTIAL_SBOXCOUNT':
        set_model_versions(cipher, "XORDIFF", functions, rounds, layers, positions) # Set model_version = "XORDIFF" for all operators
        set_model_versions(cipher, "XORDIFF_A", functions, rounds, layers, positions, operator_name="Sbox") # Set model_version = "XORDIFF_A" for all Sbox operators

    elif goal == 'DIFFERENTIALPATH_PROB' or  goal == "DIFFERENTIAL_PROB":
        set_model_versions(cipher, "XORDIFF", functions, rounds, layers, positions) # Set model_version = "XORDIFF" for all operators
        set_model_versions(cipher, "XORDIFF_PR", functions, rounds, layers, positions, operator_name="Sbox") # Set model_version = "XORDIFF_PR" for all Sbox operators

    elif goal == 'LINEAR_SBOXCOUNT':
        set_model_versions(cipher, "LINEAR", functions, rounds, layers, positions) # Set model_version = "LINEAR" for all operators
        set_model_versions(cipher, "LINEAR_A", functions, rounds, layers, positions, operator_name="Sbox") # Set model_version = "LINEAR_A" for all Sbox operators

    elif goal == 'LINEARPATH_PROB' or goal == "LINEARHULL_PROB":
        set_model_versions(cipher, "LINEAR", functions, rounds, layers, positions) # Set model_version = "LINEAR" for all operators
        set_model_versions(cipher, "LINEAR_PR", functions, rounds, layers, positions, operator_name="Sbox") # Set model_version = "LINEAR_PR" for all Sbox operators

    elif goal == "TRUNCATEDDIFF_SBOXCOUNT":
        set_model_versions(cipher, "TRUNCATEDDIFF", functions, rounds, layers, positions) # Set model_version = "TRUNCATEDDIFF" for all operators
        set_model_versions(cipher, "TRUNCATEDDIFF_A", functions, rounds, layers, positions, operator_name="Sbox") # Set model_version = "TRUNCATEDDIFF_A" for all Sbox operators

    else:
        raise ValueError(f"Invalid goal: {goal}.")

    if "model_version" in config_model: # Set a specific model version for an operator. Example: config_model['model_version'] = {'model_version': 'XOR_XORDIFF_1', 'operator_name': 'XOR'}.
        version = config_model.get("model_version").get("model_version")
        operator_name = config_model.get("model_version").get("operator_name", None)
        set_model_versions(cipher, version, functions, rounds, layers, positions, operator_name=operator_name)


def set_model_versions(cipher, version, functions, rounds, layers, positions, operator_name=None): # Assigns a specified model_version to constraints (operators) in the cipher based on specified parameters.
    def _assgn_version(cons):
        if operator_name is None: # Assign model_version to all operators in the cipher.
            cons.model_version = cons.__class__.__name__ + "_" + version
        elif operator_name is not None and (operator_name == cons.__class__.__name__ or (operator_name=="Sbox" and cons.__class__.__name__.endswith("Sbox"))): # Assign model_version to operators with a specific name.
            cons.model_version = cons.__class__.__name__ + "_" + version

    # Assign model_version to input/output constraints
    for cons in cipher.inputs_constraints:
        _assgn_version(cons)
    for cons in cipher.outputs_constraints:
        _assgn_version(cons)

    # Assign model_version to function
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for cons in cipher.functions[f].constraints[r][l]: # Only support all constraints in a layer for now.
                    _assgn_version(cons)


def gen_round_model_constraint_obj_fun(cipher, goal, model_type, config_model): # Generate constraints for a given cipher based on user-specified parameters.
    configure_model_version(cipher, goal, config_model)
    constraint = []
    obj_fun = [[] for _ in range(cipher.functions["PERMUTATION"].nbr_rounds)]

    # Generate constraints linking input and output
    for cons in cipher.inputs_constraints:
        constraint += cons.generate_model(model_type=model_type)
    for cons in cipher.outputs_constraints:
        constraint += cons.generate_model(model_type=model_type)

    # Generate constraints and objective function for each round/layer/operator
    functions, rounds, layers, positions = config_model.get("functions"), config_model.get("rounds"), config_model.get("layers"), config_model.get("positions")
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for i in positions[f][r][l]:
                    cons = cipher.functions[f].constraints[r][l][i]
                    cons_class_name = cons.__class__.__name__
                    params = (config_model.get("model_params") or {}).get(cons_class_name, {}) # get operator-specific params if available. Options: {cons_class_name: {parame_name: param_value}}. Example: config_model["model_params"] = {"PRESENT_Sbox": {"tool_type": "polyhedron"}}
                    constraint += cons.generate_model(model_type=model_type, **params)
                    if hasattr(cons, 'weight'):
                        obj_fun[r-1] += cons.weight
    return constraint, obj_fun


# -------------------- Predefined Constraint Generation --------------------
def gen_predefined_constraints(model_type, cons_type, cons_vars, cons_value, bitwise=True, encoding=None):
    """
    Generate commonly used, predefined model constraints based on type and parameters.

    Args:
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

    Returns:
        list[str]: List of generated model constraint strings.

    """
    if cons_type in ["EXACTLY", "SUM_EXACTLY", "AT_LEAST", "SUM_AT_LEAST", "AT_MOST", "SUM_AT_MOST"]:
        cons_vars_name = []
        for var in cons_vars:
            if isinstance(var, str):
                cons_vars_name.append(var)
            else:
                if bitwise and var.bitsize > 1:
                    cons_vars_name.extend([f"{var.ID}_{j}" for j in range(var.bitsize)])
                else:
                    cons_vars_name.append(var.ID)
        if cons_type == "EXACTLY":
            return gen_constraints_exactly(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_EXACTLY":
            return gen_constraints_sum_exactly(model_type, cons_vars_name, cons_value, encoding)
        elif cons_type == "AT_MOST":
            return gen_constraints_at_most(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_AT_MOST":
            return gen_constraints_sum_at_most(model_type, cons_vars_name, cons_value, encoding)
        elif cons_type == "AT_LEAST":
            return gen_constraints_at_least(model_type, cons_vars_name, cons_value)
        elif cons_type == "SUM_AT_LEAST":
            return gen_constraints_sum_at_least(model_type, cons_vars_name, cons_value, encoding)
    raise ValueError(f"Unsupported cons_type '{cons_type}'.")

def gen_constraints_exactly(model_type, cons_vars, cons_value):
    if model_type == "milp":
        return [f"{cons_vars[i]} = {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for EXACTLY constraint.")

def gen_constraints_sum_exactly(model_type, cons_vars, cons_value, encoding=1):
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" = {cons_value}"]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and pysat_import:
        if not encoding:
            encoding = 1  # Default to 1 if not specified
        assert encoding in [0,1,2,3,4,5,6,7,8,9], f"[ERROR] Invalid encoding = {encoding}, refer https://pysathq.github.io/docs/html/api/card.html"
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        try:
            cnf = CardEnc.equals(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        except Exception as e:
            print(f"[WARNING] Don't support encoding {encoding} in CardEnc.equals. Passing...")
            return []
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_EXACTLY constraint.")

def gen_constraints_at_most(model_type, cons_vars, cons_value):
    if model_type == "milp":
        return [f"{cons_vars[i]} <= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return [f"-{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for AT_MOST constraint.")

def gen_constraints_sum_at_most(model_type, cons_vars, cons_value, encoding="SEQUENTIAL"):
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" <= {cons_value}"]
    elif model_type == "sat" and (encoding == "SEQUENTIAL" or encoding is None):
        return gen_sequential_encoding_sat(cons_vars, cons_value)
    elif model_type == "sat" and pysat_import:
        if not isinstance(encoding, int):
            encoding = 1  # Default to 1 if the encoding is not specified as an integer
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        try:
            cnf = CardEnc.atmost(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        except Exception as e:
            print(f"[WARNING] Don't support encoding {encoding} in CardEnc.atmost. Passing...")
            return []
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_AT_MOST constraint.")

def gen_constraints_at_least(model_type, cons_vars, cons_value):
    if model_type == "milp":
        return [f"{cons_vars[i]} >= {cons_value}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 1:
        return [f"{cons_vars[i]}" for i in range(len(cons_vars))]
    elif model_type == "sat" and cons_value == 0:
        return []
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for GREATER_EQUAL constraint.")

def gen_constraints_sum_at_least(model_type, cons_vars, cons_value, encoding=1):
    if model_type == "milp":
        return [' + '.join(f"{cons_vars[i]}" for i in range(len(cons_vars))) + f" >= {cons_value}"]
    elif model_type == "sat" and cons_value == 1:
        return [' '.join(f"{cons_vars[i]}" for i in range(len(cons_vars)))]
    elif model_type == "sat" and pysat_import:
        if not encoding:
            encoding = 1  # Default to 1 if not specified
        variable_map = {name: idx + 1 for idx, name in enumerate(cons_vars)}
        reverse_map = {v: k for k, v in variable_map.items()}
        lits = [variable_map[name] for name in cons_vars]
        try:
            cnf = CardEnc.atleast(lits=lits, bound=cons_value, vpool=vpool, encoding=encoding)
        except Exception as e:
            print(f"[WARNING] Don't support encoding {encoding} in CardEnc.atleast. Passing...")
            return []
        readable_clauses = []
        for clause in cnf.clauses:
            readable = " ".join(f"-{reverse_map.get(abs(lit), f'dummy_{abs(lit)}')}" if lit < 0 else reverse_map.get(abs(lit), f'dummy_{abs(lit)}') for lit in clause)
            readable_clauses.append(readable)
        return readable_clauses
    else:
        raise ValueError(f"Unsupported model_type '{model_type}' for SUM_GREATER_EQUAL constraint.")

def gen_sequential_encoding_sat(hw_list, weight, dummy_variables=None): # Generate SAT constraints for a sequential counter encoding of a cardinality constraint. Reference: Conjunctive normal form. Technical report, Encyclopedia of Mathematics, http://encyclopediaofmath.org/index.php?title= Conjunctive_normal_form&oldid=35078. Refer to the code from: https://github.com/Crypto-TII/claasp/blob/main/claasp/cipher_modules/models/sat/sat_model.py#L262
    if not hasattr(gen_sequential_encoding_sat, "_counter"): # Use function attribute to set global counter
        gen_sequential_encoding_sat._counter = 0
    n = len(hw_list)
    if not isinstance(weight, int) or weight < 0 or weight > n:
        raise ValueError(f"weight should be an integer: 0 <= weight <= n (n={n}), got {weight}")
    if weight == 0:
        return [f'-{var}' for var in hw_list]
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
    return  constraints

# ----------- Matsui's branch-and-bound constraints Generation -------------
def gen_matsui_constraints_milp(Round, best_obj, obj_fun, cons_type="ALL"): # Generate Matsui's additional constraints for MILP models. Reference: Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy.
    assert Round >= 2, f"Round = {Round} must be at least 2."
    assert len(best_obj) == Round-1, f"best_obj = {best_obj} length must be Round-1 = {Round-1}."
    while obj_fun and obj_fun[-1] == []: # Remove empty lists at the end of obj_fun
        obj_fun.pop()
    assert obj_fun is not None and len(obj_fun) == Round and all(isinstance(obj, list) for obj in obj_fun), f"obj_fun = {obj_fun} must be a list of lists, and with length equal to Round = {Round}."
    assert cons_type in ["ALL", "UPPER", "LOWER"], f"cons_type = {cons_type} must be one of ['ALL', 'UPPER', 'LOWER']."

    add_cons = []
    for i in range(1, Round):
        if best_obj[i-1] > 0:
            if cons_type == "ALL" or cons_type == "UPPER":
                w_vars = [var for r in range(i + 1, Round + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="AT_MOST", cons_vars=all_vars, cons_value=-best_obj[i-1])
            if cons_type == "ALL" or cons_type == "LOWER":
                w_vars = [var for r in range(1, Round - i + 1) for var in obj_fun[r - 1]]
                all_vars = [" + ".join(w_vars) + " - obj"]
                add_cons += gen_predefined_constraints(model_type="milp", cons_type="AT_MOST", cons_vars=all_vars, cons_value=-best_obj[i-1])
    return add_cons


def gen_matsui_constraints_sat(Round, best_obj, obj_sat, obj_var, GroupConstraintChoice=1, GroupNumForChoice=1): # Generate Matsui's additional constraints for SAT models. Reference: Ling Sun, Wei Wang and Meiqin Wang. Accelerating the Search of Differential and Linear Characteristics with the SAT Method. https://github.com/SunLing134340/Accelerating_Automatic_Search
    assert Round >= 2, f"Round = {Round} must be at least 2."
    assert len(best_obj) == Round-1, f"best_obj length = {len(best_obj)} must be (Round-1) = {Round-1}."
    assert isinstance(obj_sat, int) and obj_sat > 0, f"obj_sat = {obj_sat} must be a positive integer."
    while obj_var and obj_var[-1] == []: # Remove empty lists at the end of obj_var
        obj_var.pop()
    assert obj_var is not None and len(obj_var) == Round and all(isinstance(row, list) for row in obj_var), f"obj_var must be a list of lists, and with length = {len(obj_var)} equal to Round = {Round}."
    assert GroupConstraintChoice == 1, f"Currently only support GroupConstraintChoice = 1, but got {GroupConstraintChoice}."
    assert GroupNumForChoice >= 1, f"GroupNumForChoice = {GroupNumForChoice} must be at least 1."

    if not hasattr(gen_matsui_constraints_sat, "_counter"): # Use function attribute to set global counter
        gen_matsui_constraints_sat._counter = 0
    if len(best_obj) == Round-1:
        best_obj = [0] + best_obj
    Main_Vars = list([])
    for r in range(Round):
        for i in range(len(obj_var[Round - 1 - r])):
            Main_Vars += [obj_var[Round - 1 - r][i]]
    gen_matsui_constraints_sat._counter += 1
    dummy_var = [[f'dummy_matsui_{gen_matsui_constraints_sat._counter}_{i}_{j}' for j in range(obj_sat)] for i in range(len(Main_Vars) - 1)]
    constraints = gen_sequential_encoding_sat(hw_list=Main_Vars, weight=obj_sat, dummy_variables=dummy_var) # Generate the constraint of "the objective function value is at most obj" using the sequential encoding method

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
        constraints += gen_matsui_partial_cardinality_sat(Main_Vars, dummy_var, obj_sat, left, right, PartialCardinalityCons)
    return constraints


def gen_matsui_partial_cardinality_sat(obj_var, dummy_var, k, left, right, m): # Generate CNF clauses that constrain the number of active variables in the range [left, right] to be at most `m`, using sequential counter encoded auxiliary variables `dummy_var`.
    assert isinstance(obj_var, list) and len(obj_var) > 0, "obj_var must be a non-empty list."
    assert isinstance(dummy_var, list) and len(dummy_var) == len(obj_var) - 1, "dummy_var must be a list with length equal to len(obj_var) - 1."
    assert isinstance(k, int) and k > 0, "k must be a positive integer."
    assert isinstance(left, int) and left >= 0, "left index must be a non-negative integer."
    assert isinstance(right, int) and right < len(obj_var), f"right index = {right} out of range of obj_var = {len(obj_var)}."
    assert isinstance(m, int) and m >= 0, f"m={m} must be a non-negative integer."

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
