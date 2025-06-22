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