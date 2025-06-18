import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from tools.inequality import minimize_constraints_greedy, normalize_inequality
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files'))

try:
    import cdd
except ImportError:
    print("pycddlib is not installed, installing it by 'pip install pycddlib', https://pypi.org/project/pycddlib/")


def cdd_ineq_to_coeff_rhs(ineq): # Convert a cddlib-style inequality of the form: [b, a1, a2, ..., an] to the coefficients [a1, a2, ..., an, -b], which represents 'a1*x1 + ... + an*xn >= -b'
    return ineq[1:] + [-ineq[0]]


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