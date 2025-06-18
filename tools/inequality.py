import copy
from fractions import Fraction
from math import gcd
from functools import reduce


def inequality_to_constraint_sat(inequality, variables): # Convert an inequality (coefficients + RHS) into the constraint into SAT format.
    """
    Example:
        inequality = [1, -1, 0, -1, -1], variables = ['x1', 'x2', 'x3', 'x4']
        Return: 'x1 -x2 -x4'
    """
    terms = []
    rhs = inequality[-1]
    for coeff, var in zip(inequality[:-1], variables):
        if coeff == 1:
            terms.append(f"{var}")
        elif coeff == -1:
            terms.append(f"-{var}")
        # coeff == 0 → variable not used
    return " ".join(terms).strip()

    
def inequality_to_constraint_milp(inequality, variables): #  Convert an inequality (coefficients + RHS) into the constraint into MILP format.
    """
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
        # coeff == 0 → variable not used
    return " ".join(terms).lstrip('+ ').strip() + f" >= {rhs}"    

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


def normalize_inequality(ineq):  # Ensure integer coefficients with minimal scaling
    ineq = [Fraction(x) for x in ineq]
    lcm_den = reduce(lambda a, b: a * b // gcd(a, b), [x.denominator for x in ineq], 1)
    scaled = [int(x * lcm_den) for x in ineq]
    g = reduce(gcd, scaled)
    scaled = [x // g for x in scaled]
    return scaled