"""Model-building configuration for cipher analysis.
"""

from tools.paths import get_files_dir


# Each goal maps to a tuple of (version, operator_name) rules applied in order. A ``None``
# operator_name targets every operator; "Sbox" matches any *Sbox class; "AESround" carries
# the S-box-level version onto the composite AES round (its inner S-boxes then inherit it).
GOAL_MODEL_VERSION_RULES = {
    "DIFFERENTIAL_SBOXCOUNT": (("XORDIFF", None), ("XORDIFF_A", "Sbox"), ("XORDIFF_A", "AESround")),
    "DIFFERENTIALPATH_PROB": (("XORDIFF", None), ("XORDIFF_PR", "Sbox"), ("XORDIFF_PR", "AESround")),
    "DIFFERENTIAL_PROB": (("XORDIFF", None), ("XORDIFF_PR", "Sbox"), ("XORDIFF_PR", "AESround")),
    "LINEAR_SBOXCOUNT": (("LINEAR", None), ("LINEAR_A", "Sbox"), ("LINEAR_A", "AESround")),
    "LINEARPATH_CORR": (("LINEAR", None), ("LINEAR_PR", "Sbox"), ("LINEAR_PR", "AESround")),
    "LINEARHULL_CORR": (("LINEAR", None), ("LINEAR_PR", "Sbox"), ("LINEAR_PR", "AESround")),
    "TRUNCATEDDIFF_SBOXCOUNT": (
        ("TRUNCATEDDIFF", None),
        ("TRUNCATEDDIFF_A", "Sbox"),
        ("TRUNCATEDDIFF_A", "AESround"),
    ),
    "TRUNCATEDLINEAR_SBOXCOUNT": (
        ("TRUNCATEDLINEAR", None),
        ("TRUNCATEDLINEAR_A", "Sbox"),
        ("TRUNCATEDLINEAR_A", "AESround"),
    ),
    "INTEGRAL_TWOSUBSET": (("INTEGRAL_TWOSUBSET", None),),
}


def fill_functions_rounds_layers_positions(
    cipher,
    functions=None,
    rounds=None,
    layers=None,
    positions=None,
):
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
        layers = {
            f: {r: list(range(cipher.functions[f].nbr_layers+1)) for r in rounds[f]}
            for f in functions
        }
    if positions is None:
        positions = {
            f: {
                r: {
                    l: list(range(len(cipher.functions[f].constraints[r][l])))
                    for l in layers[f][r]
                }
                for r in rounds[f]
            }
            for f in functions
        }
    return functions, rounds, layers, positions


def configure_model_version(cipher, goal, config_model):
    """Assign each in-scope constraint the model version required by ``goal``.

    Applies the ``(version, operator_name)`` rules from ``GOAL_MODEL_VERSION_RULES``,
    then any explicit overrides in ``config_model["model_version"]``. The override is a
    ``{operator_name: version}`` dict, so different operators can take different versions;
    a ``None`` key targets every operator and is applied first, so named-operator entries
    win on overlap (e.g. ``{None: "XORDIFF", "Sbox": "XORDIFF_PR"}``).
    """
    functions = config_model.get("functions")
    rounds = config_model.get("rounds")
    layers = config_model.get("layers")
    positions = config_model.get("positions")
    missing = [key for key in ("functions", "rounds", "layers", "positions") if config_model.get(key) is None]
    if missing:
        raise ValueError(f"config_model is missing required scope keys: {missing}.")

    if goal not in GOAL_MODEL_VERSION_RULES:
        raise ValueError(f"Invalid goal: {goal}.")

    for version, operator_name in GOAL_MODEL_VERSION_RULES[goal]:
        set_model_versions(
            cipher,
            version,
            functions,
            rounds,
            layers,
            positions,
            operator_name=operator_name,
        )

    overrides = config_model.get("model_version") or {}
    for operator_name in sorted(overrides, key=lambda name: name is not None):
        set_model_versions(
            cipher,
            overrides[operator_name],
            functions,
            rounds,
            layers,
            positions,
            operator_name=operator_name,
        )


def set_model_versions(cipher, version, functions, rounds, layers, positions, operator_name=None):
    """Assign a model version to matching constraints in the selected cipher scope."""
    def assign_version(cons):
        if operator_name is None: # Assign model_version to all operators in the cipher.
            cons.model_version = cons.__class__.__name__ + "_" + version
        elif operator_name == cons.__class__.__name__ or (
            operator_name == "Sbox"
            and cons.__class__.__name__.endswith("Sbox")
        ):
            cons.model_version = cons.__class__.__name__ + "_" + version

    # Assign model_version to input/output constraints.
    for cons in cipher.inputs_constraints:
        assign_version(cons)
    for cons in cipher.outputs_constraints:
        assign_version(cons)

    # Assign model_version to each function.
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for cons in cipher.functions[f].constraints[r][l]:
                    assign_version(cons)


def gen_round_model_constraint_obj_fun(cipher, goal, config_model):
    """Generate constraints and objective-function variables for selected rounds."""
    model_type = config_model["model_type"]
    configure_model_version(cipher, goal, config_model)
    constraint = []
    obj_fun = [[] for _ in range(cipher.nbr_rounds)]

    # Generate constraints linking input and output.
    if config_model.get("gen_input_model", True):
        for cons in cipher.inputs_constraints:
            constraint.extend(cons.generate_model(model_type=model_type))
    if config_model.get("gen_output_model", True):
        for cons in cipher.outputs_constraints:
            constraint.extend(cons.generate_model(model_type=model_type))

    # Generate constraints and objective function for each round/layer/operator.
    functions = config_model.get("functions")
    rounds = config_model.get("rounds")
    layers = config_model.get("layers")
    positions = config_model.get("positions")
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                for i in positions[f][r][l]:
                    cons = cipher.functions[f].constraints[r][l][i]
                    cons_class_name = cons.__class__.__name__
                    # Operator-specific params, if any: {cons_class_name: {param_name: param_value}}.
                    # Example: config_model["model_params"] = {"PRESENT_Sbox": {"tool_type": "polyhedron"}}
                    params = (config_model.get("model_params") or {}).get(cons_class_name, {})
                    constraint.extend(cons.generate_model(model_type=model_type, **params))
                    if hasattr(cons, 'weight'):
                        obj_fun[r-1].extend(cons.weight)
    return constraint, obj_fun


# ------------------- Attack-search configuration ------------------- #
def normalize_model_type(model_type):
    """Lower-case a ``model_type`` and validate it is ``"milp"`` or ``"sat"``, returning it."""

    normalized = str(model_type).lower()
    if normalized not in ("milp", "sat"):
        raise ValueError(f"Invalid model_type: {model_type}. Expected one of ['milp', 'sat'].")
    return normalized


def default_model_filename(cipher, goal, objective_target, model_type):
    """Return the default runtime model filename for a configured attack."""

    suffix = "milp_model.lp" if model_type == "milp" else "sat_model.cnf"
    return str(
        get_files_dir()
        / f"{cipher.nbr_rounds}round_{cipher.name}_{goal}_{objective_target}_{suffix}"
    )


def normalize_solution_number(value):
    """Validate that a solver ``solution_number`` is a positive integer, returning it."""

    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid solution_number: {value}. Expected a positive integer.")
    return value


def parse_and_set_configs(cipher, goal, objective_target, config_model, config_solver):
    """Apply common model and solver defaults and return ``(config_model, config_solver)``.

    Goal-specific defaults (e.g. a large ``solution_number`` for many-trail searches) are
    applied by each attack frontend after calling this.
    """

    config_model = dict(config_model or {})
    config_solver = dict(config_solver or {})
    config_model["model_type"] = normalize_model_type(config_model.get("model_type", "milp"))

    functions, rounds, layers, positions = fill_functions_rounds_layers_positions(cipher)
    config_model.setdefault("functions", functions)
    config_model.setdefault("rounds", rounds)
    config_model.setdefault("layers", layers)
    config_model.setdefault("positions", positions)
    config_solver.setdefault("solver", "DEFAULT")

    config_model.setdefault(
        "filename",
        default_model_filename(cipher, goal, objective_target, config_model["model_type"]),
    )

    if "solution_number" in config_solver:
        config_solver["solution_number"] = normalize_solution_number(config_solver["solution_number"])

    return config_model, config_solver
