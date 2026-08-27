"""Constraint template generation, caching, and instantiation helpers."""

import ast
import os
import platform
import re
import sys
import time

from tools.minimize_logic import ttb_to_ineq_logic
from tools.polyhedron import ttb_to_ineq_convex_hull

# Tool types that produce inequalities from a truth table, per model type.
_SUPPORTED_TOOLS = {
    "milp": ("minimize_logic", "minimize_logic_espresso", "polyhedron"),
    "sat": ("minimize_logic", "minimize_logic_espresso"),
}


def generate_and_save_constraints(
    model_type,
    tool_type,
    mode,
    ttable,
    input_variables,
    output_variables,
    weight_variables=None,
    objective_fun=None,
    model_filename=None,
):
    """
    Generate template constraints/objective function and save them to model_filename.

    For a MILP model the returned constraints list carries a trailing ``"Binary\\n<vars>"``
    declaration (not a real constraint).

    Returns:
        tuple[list[str], str]: (constraints, objective_fun)
    """
    if not input_variables or not output_variables:
        raise ValueError("generate_and_save_constraints requires non-empty input_variables and output_variables.")
    variables = (
        input_variables + output_variables + weight_variables
        if weight_variables
        else input_variables + output_variables
    )
    time_start = time.time()
    if model_type not in _SUPPORTED_TOOLS:
        raise ValueError(f"generate_and_save_constraints: unknown model type '{model_type}'")
    if tool_type not in _SUPPORTED_TOOLS[model_type]:
        raise ValueError(f"generate_and_save_constraints: unsupported tool type '{tool_type}' for {model_type} model")

    if tool_type == "minimize_logic" or tool_type == "minimize_logic_espresso":
        inequalities, information = ttb_to_ineq_logic(
            ttable,
            variables,
            mode=mode,
            tool_type=tool_type,
        )

    elif tool_type == "polyhedron": # Generate inequalities from the truth table using Convex Hull
        inequalities, information = ttb_to_ineq_convex_hull(ttable, variables)

    # Drop any all-zero inequality (no coefficients): espresso/cddlib do not emit these,
    # but an empty SAT clause / LHS-less MILP row would be malformed if one appeared.
    inequalities = [ineq for ineq in inequalities if any(ineq[:-1])]

    if model_type == 'milp': # Generate MILP constraints from inequalities
        constraints = [inequality_to_constraint_milp(ineq, variables) for ineq in inequalities]
        num_cons = len(constraints)  # real inequalities only (before the Binary line)
        # The MILP constraints list carries a trailing "Binary\n<vars>" declaration (NOT a real
        # constraint); "Number of constraints" stays num_cons, so the list length is num_cons + 1.
        # Consumers must handle the trailing Binary line.
        constraints.append('Binary\n' + ' '.join(variables))
    elif model_type == 'sat':  # Generate SAT constraints from inequalities
        constraints = [inequality_to_constraint_sat(ineq, variables) for ineq in inequalities]
        num_cons = len(constraints)

    time_used = time.time() - time_start
    if model_filename is not None:
        with open(model_filename, "w", encoding="utf-8") as file:
            file.write(f"Input: {'||'.join(input_variables)}; msb: {input_variables[0]}\n")
            file.write(f"Output: {'||'.join(output_variables)}; msb: {output_variables[0]}\n")
            file.write(f"Time used to simplify the constraints: {time_used:.4f} s\n")
            file.write(f"Number of constraints: {num_cons}\n")
            file.write(f"Constraints: {constraints}\n")
            if objective_fun:
                file.write(f"Weight: {objective_fun}\n")
            file.write(f"\n\nInformation\n")
            for key, value in information.items():
                file.write(f"{key}: {value}\n")
            file.write(f"Model type: {model_type}\n")
            file.write(f"Tool type: {tool_type}\n")
            file.write(f"Python version: {sys.version.split()[0]}\n")
            file.write(f"Platform: {platform.platform()}\n")
    return constraints, objective_fun


def load_constraints_template(filename):
    """
    Load template constraints/objective function from file.

    Returns:
        tuple[list[str] | None, str | None]: (constraints, objective_fun)
    """
    constraints, objective_fun = None, None
    if not os.path.exists(filename):
        return None, None
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Constraints:"):
                constraints_str = line.split(":", 1)[1].strip()
                try:
                    constraints = ast.literal_eval(constraints_str)
                except (SyntaxError, ValueError) as e:
                    raise ValueError(f"Failed to parse constraints from {filename}: {constraints_str}") from e
            elif line.startswith("Weight:"):
                objective_fun = line.split(":", 1)[1].strip()
    return constraints, objective_fun


def build_template_replacer(*template_var_groups):
    replacements = {}
    for prefix, variables in template_var_groups:
        if variables is None:
            continue
        replacements.update(
            (f"{prefix}{index}", str(variable))
            for index, variable in enumerate(variables)
        )

    if not replacements:
        return lambda expr: expr

    token_pattern = "|".join(
        re.escape(token)
        for token in sorted(replacements, key=len, reverse=True)
    )
    pattern = re.compile(rf"\b(?:{token_pattern})\b")

    def replace(expr):
        if expr is None:
            return None
        return pattern.sub(lambda match: replacements[match.group(0)], expr)

    return replace


def instantiate_constraints_template(constraints, objective_fun, var_in, var_out, var_p=None):
    """
    Instantiate template constraints/objective function by replacing template variables:
        a0, a1, ... -> var_in[i]
        b0, b1, ... -> var_out[i]
        p0, p1, ... -> var_p[i] (optional)

    Returns:
        tuple[list[str], str]: (mapped_constraints, mapped_objective_fun)
    """
    replace_constraint_vars = build_template_replacer(
        ("a", var_in),
        ("b", var_out),
        ("p", var_p),
    )
    replace_objective_vars = build_template_replacer(("p", var_p))

    return (
        [replace_constraint_vars(con) for con in constraints],
        replace_objective_vars(objective_fun),
    )


def gen_constraints_obj_func_from_template(filename, var_in, var_out, var_p=None):
    """
    Load template constraints/objective function from file, then instantiate them.

    Returns:
        tuple[list[str], str]: (mapped_constraints, mapped_objective_fun)
    """
    constraints, objective_fun = load_constraints_template(filename)

    if constraints is None:
        raise ValueError(f"Failed to load constraints or objective function from {filename}.")

    return instantiate_constraints_template(constraints, objective_fun, var_in, var_out, var_p)


def inequality_to_constraint_sat(inequality, variables):
    """
    Convert coefficients plus RHS into SAT clause format.
    
    Example:
        inequality = [1, -1, 0, -1, -1], variables = ['x1', 'x2', 'x3', 'x4']
        Return: 'x1 -x2 -x4'
    """
    terms = []
    for coeff, var in zip(inequality[:-1], variables):
        if coeff == 1:
            terms.append(f"{var}")
        elif coeff == -1:
            terms.append(f"-{var}")
        # coeff == 0 -> variable not used
    if not terms:
        raise ValueError(f"Degenerate all-zero inequality has no literals: {inequality}.")
    return " ".join(terms).strip()


def inequality_to_constraint_milp(inequality, variables):
    """
    Convert coefficients plus RHS into MILP inequality format.
    
    Example:
        ineq = [1, -1, 0, -1, -1], variables = ['x1', 'x2', 'x3', 'x4']
        Return: 'x1 - x2 - x4 >= -1'
    """
    terms = []
    rhs = inequality[-1]
    for coeff, var in zip(inequality[:-1], variables):
        sign = '+' if coeff > 0 else '-'
        abs_coeff = abs(coeff)
        if abs_coeff == 1:
            terms.append(f"{sign} {var}")
        elif abs_coeff > 0:
            terms.append(f"{sign} {abs_coeff} {var}")
        # coeff == 0 -> variable not used
    if not terms:
        raise ValueError(f"Degenerate all-zero inequality has no left-hand side: {inequality}.")
    return " ".join(terms).lstrip('+ ').strip() + f" >= {rhs}"
