import os
import copy
from fractions import Fraction
from math import gcd
from functools import reduce
try:
    import cdd
except ImportError:
    print("pycddlib is not installed, installing it by 'pip install pycddlib', https://pypi.org/project/pycddlib/")

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files/sbox_modeling/'))
if not os.path.exists(base_path): 
    os.makedirs(base_path, exist_ok=True)


def cdd_ineq_to_coeff_rhs(ineq): # Convert a cddlib-style inequality of the form: [b, a1, a2, ..., an] to the coefficients [a1, a2, ..., an, -b], which represents 'a1*x1 + ... + an*xn >= -b'
    return ineq[1:] + [-ineq[0]]


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
        max_count_index = count_of_cutoff.index(max(count_of_cutoff))
        select_ine.append(ine[max_count_index])
        ine.remove(ine[max_count_index])
        for p in cutoff[max_count_index]:
            point.remove(p)
    return select_ine


def ttb_to_ineq_convex_hull(ttable, variables): # Convert a truth table to CNF or MILP constraints using the convex hull method via pycddlib.
    num_vars = len(variables)
    all_points = [list(map(int, bin(i)[2:].zfill(num_vars))) for i in range(2 ** num_vars)]
    possible_points = [pt for i, pt in enumerate(all_points) if ttable[i] == '1']
    gen_matrix = cdd.Matrix([[1] + pt for pt in possible_points], number_type='fraction')
    gen_matrix.rep_type = cdd.RepType.GENERATOR
    poly = cdd.Polyhedron(gen_matrix)
    inequalities = poly.get_inequalities()
    raw_ineqs = [cdd_ineq_to_coeff_rhs(list(ineq)) for ineq in inequalities]
    processed_ineqs = [normalize_inequality(ineq) for ineq in raw_ineqs]
    minmized_ineqs = minimize_constraints_greedy(processed_ineqs, variables, ttable)
    return minmized_ineqs