"""Model-building configuration for cipher analysis.
"""

from operators.Sbox import Sbox
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

# Goal vocabularies, partitioned by the attack that accepts them.
DIFFERENTIAL_GOALS = ("DIFFERENTIAL_SBOXCOUNT", "DIFFERENTIALPATH_PROB", "DIFFERENTIAL_PROB",
                      "TRUNCATEDDIFF_SBOXCOUNT")
LINEAR_GOALS = ("LINEAR_SBOXCOUNT", "LINEARPATH_CORR", "LINEARHULL_CORR",
                "TRUNCATEDLINEAR_SBOXCOUNT")
INTEGRAL_GOALS = ("INTEGRAL_TWOSUBSET",)

# These two searches are strategies over a middle-round goal rather than models of their own, so
# each goal names the goal it is modelled as.
IMPOSSIBLE_GOAL_MIDDLE = {"IMPOSSIBLETRUNCATEDDIFF": "TRUNCATEDDIFF_SBOXCOUNT",
                          "IMPOSSIBLEDIFF": "DIFFERENTIAL_SBOXCOUNT"}
ZEROCORR_GOAL_MIDDLE = {"ZEROCORRELATIONTRUNCATEDLINEAR": "TRUNCATEDLINEAR_SBOXCOUNT",
                        "ZEROCORRELATIONLINEAR": "LINEAR_SBOXCOUNT"}

# objective_target comes in two forms: a bare keyword, or a keyword plus a number ("AT MOST 12").
OBJECTIVE_TARGETS_BARE = ("OPTIMAL", "EXISTENCE")
OBJECTIVE_TARGETS_WITH_VALUE = ("AT MOST", "EXACTLY", "AT LEAST")
OBJECTIVE_TARGETS = OBJECTIVE_TARGETS_BARE + OBJECTIVE_TARGETS_WITH_VALUE

# Symbolic markers that the `constraints` list expands.
ANALYSIS_CONSTRAINT_MARKERS = {
    "INPUT_NOT_ZERO": "add an input difference/mask non-zero constraint (differential, linear)",
    "TWO_SUBSET_INIT": "set the two-subset division-property initial state (integral)",
}

SHOW_MODES = (0, 1, 2, 3)

# SAT only: how modeling_solving_optimal_intobj walks the objective bound.
SAT_OPTIMAL_STRATEGIES = ("INCREASING FROM AT MOST", "INCREASING FROM EXACTLY",
                          "DECREASING FROM AT MOST", "ADAPTIVE FROM AT MOST")


def fill_functions_rounds_layers_positions(
    cipher,
    functions=None,
    rounds=None,
    layers=None,
    positions=None,
):
    """
    Fill in functions, rounds, layers, and positions to full coverage when the corresponding argument is None; otherwise, keep user-supplied values.

    Args:
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


def operator_in_scope(operator_class, operator_name):
    """Whether ``operator_class`` is covered by the ``operator_name`` scope of a version rule.

    ``None`` covers every operator. ``"Sbox"`` covers every S-box.
    """
    if operator_name is None:
        return True
    if operator_name == "Sbox":
        return issubclass(operator_class, Sbox)
    return operator_name == operator_class.__name__


def scoped_constraints(cipher, functions, rounds, layers):
    """Every operator of ``cipher`` inside the given scope, inputs and outputs included."""
    constraints = list(cipher.inputs_constraints) + list(cipher.outputs_constraints)
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                constraints.extend(cipher.functions[f].constraints[r][l])
    return constraints


def set_model_versions(cipher, version, functions, rounds, layers, positions, operator_name=None):
    """Assign a model version to matching constraints in the selected cipher scope."""
    for cons in scoped_constraints(cipher, functions, rounds, layers):
        if operator_in_scope(type(cons), operator_name):
            cons.model_version = cons.__class__.__name__ + "_" + version


def model_scope(cipher, config_model):
    """``[(round, cons)]`` for exactly the operators a model will be generated for.

    ``round`` is ``None`` for the boundary constraints. Shared with the check below so the two
    cannot disagree about scope: ``set_model_versions`` assigns more broadly, and checking that
    wider set would reject requests that generate fine.
    """
    scope = []
    for switch, boundary in (("gen_input_model", cipher.inputs_constraints),
                             ("gen_output_model", cipher.outputs_constraints)):
        if config_model.get(switch, True):
            scope.extend((None, cons) for cons in boundary)
    functions = config_model.get("functions")
    rounds = config_model.get("rounds")
    layers = config_model.get("layers")
    positions = config_model.get("positions")
    for f in functions:
        for r in rounds[f]:
            for l in layers[f][r]:
                scope.extend((r, cipher.functions[f].constraints[r][l][i])
                             for i in positions[f][r][l])
    return scope


def check_assigned_model_versions(cipher, config_model):
    """Raise once, naming **every** operator left with a version it does not declare.

    Reads the versions actually assigned, not the goal's rules, so a ``model_version`` override is
    accepted when it works. Exemptions match ``Operator.check_supported_model_version``: nothing
    new is refused, only refused before any model is built.
    """
    model_type = config_model["model_type"]
    blocked = {}
    for _, cons in model_scope(cipher, config_model):
        declared = getattr(type(cons), "SUPPORTED_MODEL_VERSIONS", {}).get(model_type, ())
        if not declared or cons.model_version is None:
            continue
        allowed = {f"{type(cons).__name__}_{suffix}" for suffix in declared}
        if cons.model_version not in allowed:
            blocked[type(cons).__name__] = cons.model_version
    if blocked:
        listed = ", ".join(f"{name} (assigned {version})" for name, version in sorted(blocked.items()))
        raise ValueError(
            f"[ERROR] {len(blocked)} operator class(es) do not model the version this request needs "
            f"under model_type='{model_type}': {listed}. Nothing was generated."
        )


def gen_round_model_constraint_obj_fun(cipher, goal, config_model):
    """Generate constraints and objective-function variables for selected rounds."""
    model_type = config_model["model_type"]
    configure_model_version(cipher, goal, config_model)
    check_assigned_model_versions(cipher, config_model)
    constraint = []
    obj_fun = [[] for _ in range(cipher.nbr_rounds)]

    for r, cons in model_scope(cipher, config_model):
        # Operator-specific params, if any: {cons_class_name: {param_name: param_value}}.
        # Example: config_model["model_params"] = {"PRESENT_Sbox": {"tool_type": "polyhedron"}}
        params = (config_model.get("model_params") or {}).get(cons.__class__.__name__, {})
        constraint.extend(cons.generate_model(model_type=model_type, **params))
        if r is not None and hasattr(cons, "weight"):
            obj_fun[r - 1].extend(cons.weight)
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

    # config_model = dict(config_model or {})
    # config_solver = dict(config_solver or {})
    config_model = config_model if config_model is not None else {}
    config_solver = config_solver if config_solver is not None else {}
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
