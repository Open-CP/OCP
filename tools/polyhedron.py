"""Truth-table to MILP/SAT inequalities via the convex-hull (pycddlib) method.

``ttb_to_ineq_convex_hull`` builds the convex hull of the truth table's valid points with
pycddlib, extracts the facet inequalities (equalities are split into inequality pairs),
normalizes them to integer coefficients, and greedily selects a minimal subset that cuts
off every invalid point.
"""

import copy
from fractions import Fraction
from math import gcd
from functools import reduce
try:
    import cdd
    backend_version = getattr(cdd, "__version__", "unknown")
except (ImportError, OSError):
    cdd = None
    backend_version = "unknown"


def cdd_ineq_to_coeff_rhs(ineq):
    """Convert a cddlib inequality ``[b, a1, ..., an]`` to ``[a1, ..., an, -b]`` (i.e. ``a1*x1 + ... + an*xn >= -b``)."""
    return ineq[1:] + [-ineq[0]]


def cdd_eq_to_coeff_rhs(eq):
    """
    Convert a cddlib-style equality [b, a1, ..., an] (meaning b + a x = 0) into 2 inequalities matching is_sat() format (a * x >= b_form):
      1)  a * x >= -b
      2) (-a) * x >= b
    Return: a list of two inequalities, each as [a1, ..., an, rhs]
    """
    b = eq[0]
    a = eq[1:]
    ineq1 = a + [-b] # a * x >= -b
    ineq2 = [-ai for ai in a] + [b] # (-a) * x >= b
    return [ineq1, ineq2]


def normalize_inequality(ineq):
    """Scale an inequality to integer coefficients of minimal magnitude (clear denominators, divide by the gcd)."""
    ineq = [Fraction(x) for x in ineq]
    lcm_den = reduce(lambda a, b: a * b // gcd(a, b), [x.denominator for x in ineq], 1)
    scaled = [int(x * lcm_den) for x in ineq]
    g = reduce(gcd, scaled)
    if g == 0:  # all-zero inequality: nothing to scale (avoid division by zero)
        return scaled
    scaled = [x // g for x in scaled]
    return scaled


def is_sat(point, ineq):
    """Return True if ``point`` satisfies ``ineq`` (``a1*x1 + ... + an*xn >= b`` with ``b = ineq[-1]``)."""
    return sum(x * a for x, a in zip(point, ineq[:-1])) >= ineq[-1]


def collect_cutoffs(points, ineq):
    """Return the points that do NOT satisfy ``ineq`` (the points it cuts off)."""
    return [p for p in points if not is_sat(p, ineq)]


def minimize_constraints_greedy(inequalities, variables, ttable):
    """Greedily select a minimal subset of inequalities that cuts off every impossible point.

    Repeatedly picks the inequality removing the most still-uncovered impossible points. Warns
    and stops if the remaining points cannot be cut off (the inequalities do not exactly
    describe the truth table).
    """
    num_vars = len(variables)
    all_points = [list(map(int, bin(i)[2:].zfill(num_vars))) for i in range(2 ** num_vars)]
    impossible_points = [pt for i, pt in enumerate(all_points) if ttable[i] == '0']
    ine = copy.deepcopy(inequalities)
    point = copy.deepcopy(impossible_points)
    select_ine = []
    while point != []:
        cutoff = []
        count_of_cutoff = []
        for l in ine:
            cutoff_of_ine = collect_cutoffs(point, l)
            cutoff.append(cutoff_of_ine)
            count_of_cutoff.append(len(cutoff_of_ine))
        if not count_of_cutoff or max(count_of_cutoff) == 0:
            print(f"[WARNING] No inequality can further remove the remaining ({len(point)}) invalid points.") # In this case, the selected inequalities cannot exactly describe the truth table, some invalid points may remain.
            break
        max_count_index = count_of_cutoff.index(max(count_of_cutoff))
        select_ine.append(ine[max_count_index])
        ine.remove(ine[max_count_index])
        for p in cutoff[max_count_index]:
            point.remove(p)
    return select_ine


def extract_equalities_indices(poly):
    """Parse the polyhedron's H-representation for the ``linearity`` line, returning the 1-based indices of equality rows."""
    lines = str(poly).splitlines()
    for line in lines:
        if line.strip().startswith("linearity"):
            parts = line.strip().split() # e.g. "linearity 1 1 5 8" → [1,5,8]
            return [int(x) for x in parts[2:]]
    return []


def ttb_to_ineq_convex_hull(ttable, variables):
    """Convert a truth table to a minimal set of MILP/SAT inequalities via the convex-hull method (pycddlib).

    Returns ``(inequalities, information)``; raises ``ImportError`` if pycddlib is unavailable.

    Requires the pycddlib 2.x API (``cdd.Matrix`` / ``cdd.RepType.GENERATOR`` /
    ``cdd.Polyhedron`` / ``get_inequalities``). pycddlib >= 3 renamed these to
    ``matrix_from_array`` / ``polyhedron_from_matrix`` / ... and would raise
    ``AttributeError`` here, so pin ``pycddlib<3`` (in requirements-solvers.txt / pyproject.toml).
    """
    if cdd is None:
        raise ImportError(
            "pycddlib is required for convex-hull constraint generation. "
            "Install it with: pip install pycddlib"
        )
    num_vars = len(variables)
    all_points = [list(map(int, bin(i)[2:].zfill(num_vars))) for i in range(2 ** num_vars)]
    possible_points = [pt for i, pt in enumerate(all_points) if ttable[i] == '1']
    gen_matrix = cdd.Matrix([[1] + pt for pt in possible_points], number_type='fraction')
    gen_matrix.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(gen_matrix)
    inequalities = poly.get_inequalities()
    all_rows = [list(row) for row in inequalities]
    equalities_index = extract_equalities_indices(inequalities)
    raw_ineqs = [cdd_ineq_to_coeff_rhs(list(ineq)) for ineq in all_rows]
    for i in equalities_index:
        eq = all_rows[i - 1]
        raw_ineqs.extend(cdd_eq_to_coeff_rhs(list(eq)))
    processed_ineqs = [normalize_inequality(ineq) for ineq in raw_ineqs]
    minmized_ineqs = minimize_constraints_greedy(processed_ineqs, variables, ttable)

    information = {"Backend": "convex_hull_cdd", "Backend version": backend_version}
    return minmized_ineqs, information
