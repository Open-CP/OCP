import os
import copy
from fractions import Fraction
from math import gcd
from functools import reduce
try:
    import cdd
except ImportError:
    print("[WARNING] pycddlib is not installed, installing it by 'pip install pycddlib', https://pypi.org/project/pycddlib/")


def cdd_ineq_to_coeff_rhs(ineq): # Convert a cddlib-style inequality of the form: [b, a1, a2, ..., an] to the coefficients [a1, a2, ..., an, -b], which represents 'a1*x1 + ... + an*xn >= -b'
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


def normalize_inequality(ineq):  # Ensure integer coefficients with minimal scaling
    ineq = [Fraction(x) for x in ineq]
    lcm_den = reduce(lambda a, b: a * b // gcd(a, b), [x.denominator for x in ineq], 1)
    scaled = [int(x * lcm_den) for x in ineq]
    g = reduce(gcd, scaled)
    scaled = [x // g for x in scaled]
    return scaled


def is_sat(point, ineq): # Check whether a given point [x1, x2, ..., xn] satisfies the inequality [a1, a2, ..., an, b], which represents: a1*x1 + a2*x2 + ... + an*xn >= b.
    return sum(x * a for x, a in zip(point, ineq[:-1])) >= ineq[-1]


def collect_cutoffs(points, ineq): # Collect all points that do not satisfy the given inequality.
    return [p for p in points if not is_sat(p, ineq)]


def	minimize_constraints_greedy(inequalities, variables, ttable): # Select a minimal subset of inequalities to eliminate all impossible points by using the Greedy Algorithm.
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
            print(f"[INFO]: No inequality can further remove the remaining ({len(point)}) invalid points.") # In this case, the selected inequalities cannot exactly describe the truth table, some invalid points may remain.
            break
        max_count_index = count_of_cutoff.index(max(count_of_cutoff))
        select_ine.append(ine[max_count_index])
        ine.remove(ine[max_count_index])
        for p in cutoff[max_count_index]:
            point.remove(p)
    return select_ine


def extract_equalities_indices(poly): # Parse the H-representation text of the polyhedron to extract the 'linearity' line, which indicates the indices (1-based) of equality constraints.
    lines = str(poly).splitlines()
    for line in lines:
        if line.strip().startswith("linearity"):
            parts = line.strip().split() # e.g. "linearity 1 1 5 8" → [1,5,8]
            return [int(x) for x in parts[2:]]
    return []


def ttb_to_ineq_convex_hull(ttable, variables): # Convert a truth table to CNF or MILP constraints using the convex hull method via pycddlib.
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
    return minmized_ineqs
