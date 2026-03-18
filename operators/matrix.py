import numpy as np
import os
import copy
from pathlib import Path
from operators.operators import Operator, UnaryOperator, RaiseExceptionVersionNotExisting
from tools.model_constraints import gen_matrix_constraints, gen_constraints_obj_func_from_template, generate_and_save_constraints, gen_word_matrix_constraints, gen_word_nxor_constraints
from itertools import product

ROOT = Path(__file__).resolve().parents[1]  # this file -> operators -> <ROOT>
BASE_PATH = ROOT / "files/matrix_modeling"
BASE_PATH.mkdir(parents=True, exist_ok=True)


def find_primitive_element_gf2m(mod_poly, degree): # Find a primitive root for GF(2^m)
    for candidate in range(2, 1 << degree):
        num_elements = (1 << degree) - 1
        generated = set()
        current_value = 1
        for _ in range(num_elements):
            generated.add(current_value)
            current_value = gf2_multiply(current_value, candidate, mod_poly, degree)
        if len(generated) == num_elements:
            return candidate
    raise ValueError("No primitive root found.")


def gf2_multiply(a, b, mod_poly, degree): #  Multiply two elements in GF(2^m) under a given modulus polynomial
    result = 0
    while b > 0:
        if b & 1:
            result ^= a
        a <<= 1
        if a & (1 << degree):  # If `a` exceeds m bits, reduce modulo `mod_poly`.
            a ^= mod_poly
        b >>= 1
    return result & ((1 << degree) - 1)


def _normalize_mod_poly(mod_poly, degree):
    """
    Normalize the irreducible polynomial.

    The polynomial can be provided as an int or as a string
    (e.g. "0x11b", "0b100011011"). This function ensures that the
    highest term x^degree is present.
    """
    if isinstance(mod_poly, str):
        mod_poly = int(mod_poly, 0)

    sig_degree = (1 << degree)

    # Ensure the polynomial contains the term x^degree
    if mod_poly < sig_degree:
        mod_poly += sig_degree

    return mod_poly


def gf2_pow(a, e, mod_poly, degree):
    """
    Compute a^e in GF(2^m) using binary exponentiation.
    """
    result = 1
    base = a

    while e > 0:
        if e & 1:
            result = gf2_multiply(result, base, mod_poly, degree)

        base = gf2_multiply(base, base, mod_poly, degree)
        e >>= 1

    return result


def gf2_inv(a, mod_poly, degree):
    """
    Compute the multiplicative inverse of a in GF(2^m).

    Using the identity:
        a^{-1} = a^(2^m - 2)
    """
    if a == 0:
        raise ZeroDivisionError("Inverse of 0 does not exist in GF(2^m).")

    return gf2_pow(a, (1 << degree) - 2, mod_poly, degree)


def generate_gf2_elements_and_exponents(pri, mod_poly, degree): # Generate all elements of GF(2^m) and map them to their corresponding exponents (α^k).
    num_elements = (1 << degree)
    elements_to_exponents = {}
    exponents_to_elements = {}
    current_value = 1
    for k in range(num_elements - 1):
        elements_to_exponents[current_value] = k
        exponents_to_elements[k] = current_value
        current_value = gf2_multiply(current_value, pri, mod_poly, degree)
    return elements_to_exponents, exponents_to_elements


def generate_binary_matrix_1(degree):
    return [[1 if i == j else 0 for j in range(degree)] for i in range(degree)]


def generate_binary_matrix_2(mod_poly, degree): # Construct the binary matrix for GF(2^m) based on its modulus polynomial.
    matrix = [[0 for _ in range(degree)] for _ in range(degree)]
    coefficients = [(mod_poly >> i) & 1 for i in range(degree)]
    for i in range(degree):
        matrix[i][0] = coefficients[degree-i-1]
    for i in range(1, degree):
        matrix[i - 1][i] = 1
    return matrix


def generate_binary_matrix_3(mod_poly, degree): # Generate the binary matrix representation for the element 3 (x + 1) in GF(2^m).
    matrix1 = generate_binary_matrix_1(degree)
    matrix2 = generate_binary_matrix_2(mod_poly, degree)
    matrix = [[(matrix1[i][j] + matrix2[i][j]) % 2 for j in range(len(matrix1[0]))] for i in range(len(matrix1))]
    return matrix


def matrix_multiply_mod2(A, B): # Multiply two matrices in GF(2) (mod 2).
    size = len(A)
    result = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(size)) % 2
    return result


def matrix_power_mod2(matrix, power): # Compute the power of a matrix (mod 2).
    size = len(matrix)
    result = [[1 if i == j else 0 for j in range(size)] for i in range(size)]  # Identity matrix.
    base = matrix
    while power:
        if power % 2 == 1:
            result = matrix_multiply_mod2(result, base)
        base = matrix_multiply_mod2(base, base)
        power //= 2
    return result


def generate_pmr_for_mds(mds, mod_poly, degree): # Generate the Primitive Matrix Representation (PMR) for a given MDS matrix.
    sig_degree = (1 << degree)
    if isinstance(mod_poly, str):
        mod_poly = int(mod_poly, 0)
    if mod_poly < sig_degree: mod_poly += sig_degree
    matrix2 = generate_binary_matrix_2(mod_poly, degree)
    matrix3 = generate_binary_matrix_3(mod_poly, degree)
    pri = find_primitive_element_gf2m(mod_poly, degree)
    elements_to_exponents, exponents_to_elements = generate_gf2_elements_and_exponents(pri, mod_poly, degree)
    if pri == 2: companion_matrix = matrix2
    elif pri == 3: companion_matrix = matrix3
    matrix_representation = {exp: matrix_power_mod2(companion_matrix, exp) for exp in range((1 << degree) - 1)}
    size = len(mds)
    pmr = [[matrix_representation[elements_to_exponents[mds[i][j]]]for j in range(size)] for i in range(size)]
    pmr_new = [[0 for _ in range(size * degree)] for _ in range(size * degree)]
    # print("\nPMR Binary Matrix Representation:\n", pmr)
    for i in range(size):
        for row_offset in range(degree):
            base_index = i * degree + row_offset
            for j in range(size):
                start_index = j * degree
                end_index = start_index + degree
                pmr_new[base_index][start_index:end_index] = pmr[i][j][row_offset]
    return pmr_new


def generate_bin_matrix(mat, bitsize):
    bin_matrix = []
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[i])):
            if mat[i][j] == 1:
                row.append(np.eye(bitsize, dtype=int))
            elif mat[i][j] == 0:
                row.append(np.zeros((bitsize, bitsize), dtype=int))
        bin_matrix.append(row)
    bin_matrix = np.block(bin_matrix)
    return bin_matrix


class Matrix(Operator):   # Operator of the Matrix multiplication: appplies the matrix "mat" (stored as a list of lists) to the input vector of variables, towards the output vector of variables
                          # The optional "polynomial" allors to define the polynomial reduction (not implemted yet)
    def __init__(self, name, input_vars, output_vars, mat, polynomial = None, ID = None):
        r, c = len(mat), len(mat[0])
        for i in mat:
            if len(i)!=c: raise Exception(str(self.__class__.__name__) + ": matrix size not consistent")
        if len(input_vars)!=c: raise Exception(str(self.__class__.__name__) + ": input vector does not match matrix size")
        if len(output_vars)!=r: raise Exception(str(self.__class__.__name__) + ": output vector does not match matrix size")
        super().__init__(input_vars, output_vars, ID = ID)
        self.name = name
        self.mat = mat
        self.polynomial = polynomial

    def inverse_over_gf2m(self):
        """
        Compute the inverse of the matrix over GF(2^m) using
        Gauss–Jordan elimination.

        The field arithmetic uses the irreducible polynomial
        stored in self.polynomial.
        """

        r = len(self.mat)
        c = len(self.mat[0])

        if r != c:
            raise ValueError("Matrix must be square to be invertible.")

        if not self.polynomial:
            raise ValueError("self.polynomial is required to invert over GF(2^m).")

        degree = self.input_vars[0].bitsize
        mod_poly = _normalize_mod_poly(self.polynomial, degree)

        # Copy of the matrix
        A = [row[:] for row in self.mat]

        # Identity matrix
        I = [[0] * r for _ in range(r)]
        for i in range(r):
            I[i][i] = 1

        # Gauss–Jordan elimination
        for col in range(r):

            # Search for a non-zero pivot
            pivot = None
            for row in range(col, r):
                if A[row][col] != 0:
                    pivot = row
                    break

            if pivot is None:
                raise ValueError("Matrix is not invertible (singular) over GF(2^m).")

            # Swap rows if necessary
            if pivot != col:
                A[col], A[pivot] = A[pivot], A[col]
                I[col], I[pivot] = I[pivot], I[col]

            # Normalize pivot row so that pivot becomes 1
            piv_val = A[col][col]
            inv_piv = gf2_inv(piv_val, mod_poly, degree)

            for j in range(r):
                A[col][j] = gf2_multiply(A[col][j], inv_piv, mod_poly, degree)
                I[col][j] = gf2_multiply(I[col][j], inv_piv, mod_poly, degree)

            # Eliminate the pivot column in all other rows
            for row in range(r):

                if row == col:
                    continue

                factor = A[row][col]

                if factor == 0:
                    continue

                # In characteristic 2, subtraction = addition (XOR)
                for j in range(r):
                    A[row][j] ^= gf2_multiply(factor, A[col][j], mod_poly, degree)
                    I[row][j] ^= gf2_multiply(factor, I[col][j], mod_poly, degree)

        return I

    def differential_branch_number(self): # Return differential branch number of the Matrix. TO DO
        pass

    def linear_branch_number(self): # Return linear branch number of the Matrix. TO DO
        pass

    def zero_star_io_patterns(self):
        """
        Enumerate all input patterns (x1..xn) avec xi in {0, '*'}
        et deduce the output pattern (y1..ym) with the rules:

            y_i = 0  iff  for all j such that mat[i][j] != 0, we have x_j == 0
            y_i = '*' otherwise

        Returns:
            list[tuple] : list of tuples (x1..xn, y1..ym) with values 0 or '*'
        """
        n = len(self.input_vars)   # nb columns
        m = len(self.output_vars)  # nb rows

        patterns = []
        for x in product([0, '*'], repeat=n):
            y = []
            for i in range(m):
                forced_zero = True
                for j in range(n):
                    if self.mat[i][j] != 0 and x[j] == '*':
                        forced_zero = False
                        break
                y.append(0 if forced_zero else '*')
            patterns.append(tuple(x + tuple(y)))
        return patterns

    def zero_star_patterns_from_output_via_inverse(self):
        """
        Enumerate all output patterns y = (y1..yn) with yi in {0, '*'}
        and deduce the corresponding input pattern x = (x1..xn) induced by x = M^{-1} y.

        Rule (support-based):
            x_j is forced to 0 iff for every i such that (M^{-1})[j][i] != 0, we have y_i == 0.
            Otherwise x_j is '*'.

        Returns:
            list[tuple]: list of tuples (x1..xn, y1..yn) with entries 0 or '*'.
        """
        inv = self.inverse_over_gf2m()

        n_rows = len(inv)
        n_cols = len(inv[0])

        # For x = M^{-1} y, inv must be n x n (square)
        if n_rows != n_cols:
            raise ValueError("M^{-1} must be square for x = M^{-1} y with same dimension.")

        n = n_rows

        patterns = []
        for y in product([0, '*'], repeat=n):
            x = []
            for j in range(n):
                forced_zero = True
                for i in range(n):
                    if inv[j][i] != 0 and y[i] == '*':
                        forced_zero = False
                        break
                x.append(0 if forced_zero else '*')

            patterns.append(tuple(tuple(x) + tuple(y)))

        return patterns

    def patterns_where_a_star_is_forced_zero(self):
        """
        Enumerate all (x_pattern, y_pattern) in {0,'*'}^n x {0,'*'}^m.
        Keep only those for which at least one '*' coordinate is provably forced to 0
        by the linear constraint Mx + y = 0, i.e. (M|I) (x||y) = 0.

        Method for a given pattern:
          - Build A = (M | I)
          - Remove columns fixed to 0 by the pattern -> A_z
          - Compute RREF(A_z) over GF(2^m) (or GF(2) if no polynomial)
          - If some unit vector e_i is in the row space (equivalently, RREF has a row equal to e_i),
            then the corresponding variable z_i must be 0 in every solution.
          - We keep the pattern iff at least one such forced variable corresponds to a '*' in the pattern.

        Returns:
            list[tuple[tuple, tuple]]: list of (x_pattern, y_pattern) pairs.
        """

        # Dimensions
        n = len(self.mat[0])  # x size
        m = len(self.mat)     # y size

        # ---- Field parameters ----
        degree = self.input_vars[0].bitsize if self.input_vars else None
        use_gf2m = self.polynomial is not None

        if use_gf2m:
            mod_poly = _normalize_mod_poly(self.polynomial, degree)

            def f_add(a, b):
                return a ^ b

            def f_mul(a, b):
                return gf2_multiply(a, b, mod_poly, degree)

            def f_inv(a):
                return gf2_inv(a, mod_poly, degree)
        else:
            # GF(2) fallback
            def f_add(a, b):
                return a ^ b

            def f_mul(a, b):
                return a & b  # assuming 0/1 coefficients

            def f_inv(a):
                if a == 0:
                    raise ZeroDivisionError("Inverse of 0 does not exist in GF(2).")
                return 1

        # ---- Build augmented matrix A = (M | I), size m x (n+m) ----
        A = []
        for i in range(m):
            row = list(self.mat[i]) + [0] * m
            row[n + i] = 1
            A.append(row)

        total_cols = n + m

        # Cache: key = tuple(kept_cols) -> set of kept-column indices (0..k-1) that are forced to zero
        # (i.e., those i for which e_i is in row space of A_z)
        cache_forced_indices = {}

        def rref_forced_unit_positions(Az):
            """
            Given Az (rows x cols), compute which column positions i (0..cols-1)
            satisfy e_i in row space, i.e. RREF has a row exactly equal to e_i.
            Return a set of such i.
            """
            R = [r[:] for r in Az]
            rows = len(R)
            cols = len(R[0]) if rows > 0 else 0
            forced = set()

            if cols == 0:
                return forced

            pivot_row = 0
            pivot_col_for_row = [-1] * rows

            for col in range(cols):
                # Find pivot
                sel = None
                for r in range(pivot_row, rows):
                    if R[r][col] != 0:
                        sel = r
                        break
                if sel is None:
                    continue

                # Swap
                if sel != pivot_row:
                    R[pivot_row], R[sel] = R[sel], R[pivot_row]

                # Normalize pivot to 1
                pv = R[pivot_row][col]
                inv_pv = f_inv(pv)
                for j in range(cols):
                    R[pivot_row][j] = f_mul(R[pivot_row][j], inv_pv)

                # Eliminate in other rows
                for r in range(rows):
                    if r == pivot_row:
                        continue
                    factor = R[r][col]
                    if factor == 0:
                        continue
                    for j in range(cols):
                        R[r][j] = f_add(R[r][j], f_mul(factor, R[pivot_row][j]))

                pivot_col_for_row[pivot_row] = col
                pivot_row += 1
                if pivot_row == rows:
                    break

            # Detect unit rows
            for r in range(rows):
                pc = pivot_col_for_row[r]
                if pc == -1:
                    continue
                is_unit = True
                for j in range(cols):
                    if j == pc:
                        if R[r][j] != 1:
                            is_unit = False
                            break
                    else:
                        if R[r][j] != 0:
                            is_unit = False
                            break
                if is_unit:
                    forced.add(pc)

            return forced

        results = []

        # Enumerate all patterns for x and y
        for x_pattern in product([0, '*'], repeat=n):
            for y_pattern in product([0, '*'], repeat=m):

                # Build kept columns list (those with '*')
                kept_cols = []
                kept_meta = []  # map kept position -> ('x'/'y', original_index)

                for j in range(n):
                    if x_pattern[j] == '*':
                        kept_cols.append(j)
                        kept_meta.append(('x', j))
                for i in range(m):
                    if y_pattern[i] == '*':
                        kept_cols.append(n + i)
                        kept_meta.append(('y', i))

                # If no remaining variables, nothing can be forced among '*'
                if not kept_cols:
                    continue

                key = tuple(kept_cols)

                if key in cache_forced_indices:
                    forced_positions = cache_forced_indices[key]
                else:
                    # Build Az by selecting kept columns
                    Az = [[row[c] for c in kept_cols] for row in A]
                    forced_positions = rref_forced_unit_positions(Az)
                    cache_forced_indices[key] = forced_positions

                # Keep the pattern iff at least one forced position corresponds to a '*'
                # (by construction, all kept positions are '*' already)
                if forced_positions:
                    # sanity: forced_positions are indices in 0..len(kept_cols)-1
                    # so they always correspond to '*'
                    results.append((tuple(x_pattern), tuple(y_pattern), '*'))
                else:
                    results.append((tuple(x_pattern), tuple(y_pattern), '0'))

        return results

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return ['(' + ''.join([self.get_var_ID('out', i, unroll) + ", " for i in range(len(self.output_vars))])[:-2] + ") = " + self.name + "(" + ''.join([self.get_var_ID('in', i, unroll) + ", " for i in range(len(self.input_vars))])[:-2] + ")"]
        elif implementation_type == 'c':
            return [self.name + "(" + ''.join([self.get_var_ID('in', i, unroll) + ", " for i in range(len(self.input_vars))])[:-2] + ", " + ''.join([self.get_var_ID('out', i, unroll) + ", " for i in range(len(self.output_vars))])[:-2] + ");"]
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")

    def get_header_ID(self):
        return [self.__class__.__name__, self.model_version, self.input_vars[0].bitsize, self.mat, self.polynomial]

    def generate_implementation_header_unique(self, implementation_type='python'):
        if implementation_type == 'python':
            model_list = ["#Galois Field Multiplication Macro", "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
        elif implementation_type == 'c':
            model_list = ["//Galois Field Multiplication Macro", "#define GMUL(a, b, p, d) ({ \\", "\tunsigned int result = 0; \\", "\tunsigned int temp_a = a; \\", "\tunsigned int temp_b = b; \\", "\twhile (temp_b > 0) { \\", "\t\tif (temp_b & 1) \\", "\t\t\tresult ^= temp_a; \\", "\t\ttemp_a <<= 1; \\", "\t\tif (temp_a & (1 << d)) \\", "\t\t\ttemp_a ^= p; \\", "\t\ttemp_b >>= 1; \\", "\t} \\", "\tresult & ((1 << d) - 1); \\","})"];
        return model_list

    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python':
            model_list= ["#Matrix Macro "]
            model_list.append("def " + self.name + "(" + ''.join(["x" + str(i) + ", " for i in range (len(self.mat[0]))])[:-2]  + "):")
            for i, out_v in enumerate(self.output_vars):
                model = '\t' + 'y' + str(i) + ' = '
                first = True
                for j, in_v in enumerate(self.input_vars):
                    if self.mat[i][j] == 1:
                        if first:
                            model = model + "x" + str(j)
                            first = False
                        else: model = model + " ^ " + "x" + str(j)
                    elif self.mat[i][j] != 0:
                        if first:
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")"
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")"
                model_list.append(model)
            model_list.append("\treturn (" + ''.join(["y" + str(i) + ", " for i in range (len(self.mat))])[:-2]  + ")")
            return model_list
        elif implementation_type == 'c':
            model_list = ["//Matrix Macro "]
            model_list.append("#define " + self.name + "(" + ''.join(["x" + str(i) + ", " for i in range (len(self.mat[0]))])[:-2] + ", "  + ''.join(["y" + str(i) + ", " for i in range (len(self.mat))])[:-2] + ")  { \\")
            for i, out_v in enumerate(self.output_vars):
                model = '\t' + 'y' + str(i) + ' = '
                first = True
                for j, in_v in enumerate(self.input_vars):
                    if self.mat[i][j] == 1:
                        if first:
                            model = model + "x" + str(j)
                            first = False
                        else: model = model + " ^ " + "x" + str(j)
                    elif self.mat[i][j] != 0:
                        if first:
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")"
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")"
                model_list.append(model + "; \\")
            model_list.append("} ")
            return model_list

    def generate_model(self, model_type='sat', branch_num=None, tool_type="minimize_logic", filename_load=True):
        # Modeling for differential / linear cryptanalysis
        if model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
            # Convert the (word-level) matrix into a binary matrix representation:
            # 1. polynomial is specified: expand the matrix over GF(2^m) into its binary PMR form.
            if self.polynomial: # Example: AES MDS matrix
                bin_matrix = generate_pmr_for_mds(self.mat, self.polynomial, self.input_vars[0].bitsize)
            # 2. word-level matrix: expand each word coefficient into a binary submatrix
            elif len(self.input_vars) == len(self.mat): # Example: SKINNY 4*4 matrix
                bin_matrix = generate_bin_matrix(self.mat, self.input_vars[0].bitsize)
            # 3. matrix is already given in binary form: use it directly
            elif self.input_vars[0].bitsize * len(self.input_vars) == len(self.mat): # Example: SKINNY 64*64 binary matrix
                bin_matrix = self.mat
            else:
                raise ValueError(f"Matrix {self.mat} not supported.")
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                return self._generate_model_diff(model_type, bin_matrix)
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                return self._generate_model_linear(model_type, bin_matrix)

        # Modeling for truncated differential / linear cryptanalysis
        elif model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1", self.__class__.__name__ + "_TRUNCATEDDIFF_2", self.__class__.__name__ + "_TRUNCATEDLINEAR_2"]:
            if branch_num is not None:
                return self._generate_model_truncated_diff_linear_branch_num(model_type, branch_num)
            else: # TO DO: Branch number computation is not implemented yet.
                # if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1"]:
                    # branch_num =self.differential_branch_number()
                # elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                    # branch_num =self.linear_branch_number()
                print("[WARNING] Please provide branch number as its calculation is not implemented yet.")
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1"]:
                self.model_version = self.__class__.__name__ + "_TRUNCATEDDIFF_2"
            elif  self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                self.model_version = self.__class__.__name__ + "_TRUNCATEDLINEAR_2"
            print("[WARNING] Turn to model_version " + self.model_version)
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_2", self.__class__.__name__ + "_TRUNCATEDLINEAR_2"]:
                self.model_filename = str(BASE_PATH / f"constraints_{model_type}_{self.name}_{self.model_version}_{tool_type}.txt")
                self.filename_load = filename_load
                return self._generate_model_truncated_diff_linear_valid_patterns(model_type, tool_type)

        # Modeling for truncated differential / linear cryptanalysis
        elif model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_3", self.__class__.__name__ + "_TRUNCATEDDIFF_4", self.__class__.__name__ + "_TRUNCATEDLINEAR_3", self.__class__.__name__ + "_TRUNCATEDLINEAR_4"]:
            self.model_filename = str(BASE_PATH / f"constraints_{model_type}_{self.name}_{self.model_version}_{tool_type}.txt")
            self.filename_load = filename_load
            return self._generate_model_truncated_diff_linear_zero_star_patterns(model_type, tool_type)

        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'" + self.model_version)

    def _generate_model_diff(self, model_type, bin_matrix): # Modeling for bit-wise differential cryptanalysis
        model_list = []
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        bits_per_input = self.input_vars[0].bitsize
        bits_per_output = self.output_vars[0].bitsize
        if self.model_version in [self.__class__.__name__ + "_XORDIFF"]:
            for i in range(output_words):  # Loop over the ith output word
                for j in range(bits_per_output):  # Loop over the jth bit in the ith word
                    var_in = []
                    for k in range(input_words): # Loop over the kth input word
                        for l in range(bits_per_input): # Loop over the lth bit in the kth word
                            if bin_matrix[bits_per_output*i+j][bits_per_input*k+l] == 1:
                                if bits_per_output > 1:
                                    var_in.append(self.input_vars[k].ID + '_' + str(l))
                                else:
                                    var_in.append(self.input_vars[k].ID)
                    if bits_per_output > 1:
                        var_out = self.output_vars[i].ID + '_' + str(j)
                    else:
                        var_out = self.output_vars[i].ID
                    if model_type == 'milp':
                        d = self.ID + '_d_' + str(i) + '_' + str(j)
                    else:
                        d = None
                    model_list.extend(gen_matrix_constraints(var_in, var_out, model_type, v_dummy=d))
            return model_list
        else:
            Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

    def _generate_model_linear(self, model_type, bin_matrix): # Modeling for bit-wise linear cryptanalysis
        model_list = []
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        bits_per_input = self.input_vars[0].bitsize
        bits_per_output = self.output_vars[0].bitsize
        # Modeling for linear cryptanalysis
        if self.model_version in [self.__class__.__name__ + "_LINEAR"]:
            bin_matrix = np.transpose(bin_matrix)
            for i in range(input_words):  # Loop over the ith input word
                for j in range(bits_per_input):  # Loop over the jth bit in the ith word
                    var_in = []
                    for k in range(output_words): # Loop over the kth output word
                        for l in range(bits_per_output): # Loop over the lth bit in the kth word
                            if bin_matrix[bits_per_input*i+j][bits_per_output*k+l] == 1:
                                if bits_per_output > 1:
                                    var_in.append(self.output_vars[k].ID + '_' + str(l))
                                else:
                                    var_in.append(self.output_vars[k].ID)
                    if bits_per_output > 1:
                        var_out = self.input_vars[i].ID + '_' + str(j)
                    else:
                        var_out = self.input_vars[i].ID
                    if model_type == 'milp':
                        d = self.ID + '_d_' + str(i) + '_' + str(j)
                    else:
                        d = None
                    model_list.extend(gen_matrix_constraints(var_in, var_out, model_type, v_dummy=d))
            return model_list
        else:
            Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

    def _generate_model_truncated_diff_linear_branch_num(self, model_type, branch_num):
        # Generate the MILP model for truncated differential or truncated linear propagation using the branch number of the matrix.
        # The branch number enforces a lower bound on the total number of active input/output words when the propagation is nonzero.
        var_in = []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model('in', i, bitwise=False)
        var_out = []
        for i in range(len(self.output_vars)):
            var_out += self.get_var_model('out', i, bitwise=False)
        var_d = [f"{self.ID}_d"]
        if model_type == 'milp':
            # The first type of modeling. Reference: Nicky Mouha, Qingju Wang, Dawu Gu, and Bart Preneel. Differential and linear cryptanalysis using mixed-integer linear programming.
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            # The second type of modeling. Reference: [1] Christina Boura, Patrick Derbez and Margot Funk. Related-Key Differential Analysis of the AES. [2] Patrick Derbez, Marie Euler, Pierre-Alain Fouque, Phuong Hoa Nguyen. Revisiting Related-Key Boomerang attacks on AES using computer-aided tool.
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [" + ".join(var_in + var_out) + f" - {len(var_in+var_out)} {var_d[0]} <= 0"]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

    def _generate_model_truncated_diff_linear_valid_patterns(self, model_type, tool_type):
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        var_in = []
        for i in range(input_words):
            var_in += self.get_var_model('in', i, bitwise=False, dim=1)
        var_out = []
        for i in range(output_words):
            var_out += self.get_var_model('out', i, bitwise=False, dim=1)

        if self.filename_load and os.path.exists(self.model_filename):
            model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
            return model_list

        if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_2"]:
            all_patterns = self.patterns_where_a_star_is_forced_zero()
            patterns = [(xp, yp) for xp, yp, tag in all_patterns if tag == '0']
            patterns.append(((0, 0, 0, 0), (0, 0, 0, 0)))

        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR_2"]:
            mat = copy.deepcopy(self.mat)
            mat_trans = np.transpose(self.mat)
            self.mat = mat_trans
            all_patterns = self.patterns_where_a_star_is_forced_zero()
            patterns = [(yp, xp) for xp, yp, tag in all_patterns if tag == '0']
            patterns.append(((0, 0, 0, 0), (0, 0, 0, 0)))
            self.mat = mat
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

        ttable = ""
        for i in range(2**input_words):
            x = tuple('*' if b == '1' else 0 for b in format(i, f"0{input_words}b"))
            for j in range(2**output_words):
                y = tuple('*' if b == '1' else 0 for b in format(j, f"0{output_words}b"))
                pattern = (x, y)
                if pattern in patterns:
                    ttable += "1"
                else: ttable += "0"

        input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
        generate_and_save_constraints(model_type, tool_type, 0, ttable, input_variables, output_variables, model_filename=self.model_filename)
        model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
        return model_list

    def _generate_model_truncated_diff_linear_zero_star_patterns(self, model_type, tool_type):
        print("_generate_model_truncated_diff_linear_zero_star_patterns", self.model_version)
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        var_in = []
        for i in range(input_words):
            var_in += self.get_var_model('in', i, bitwise=False, dim=1)
        var_out = []
        for i in range(output_words):
            var_out += self.get_var_model('out', i, bitwise=False, dim=1)

        if self.filename_load and os.path.exists(self.model_filename):
            model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
            return model_list

        if self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_3":
            patterns = self.zero_star_io_patterns()
        elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_4":
            patterns = self.zero_star_patterns_from_output_via_inverse()
        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR_3", self.__class__.__name__ + "_TRUNCATEDLINEAR_4"]:
            mat = copy.deepcopy(self.mat)
            mat_trans = np.transpose(self.mat)
            self.mat = mat_trans
            if self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR_3":
                patterns = self.zero_star_io_patterns()
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR_4":
                patterns = self.zero_star_patterns_from_output_via_inverse()
            self.mat = mat
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        print("patterns", patterns)

        ttable = ""
        for i in range(2**input_words):
            x = tuple('*' if b == '1' else 0 for b in format(i, f"0{input_words}b"))
            for j in range(2**output_words):
                y = tuple('*' if b == '1' else 0 for b in format(j, f"0{output_words}b"))
                pattern = x + y
                if pattern in patterns:
                    ttable += "1"
                else: ttable += "0"

        input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
        generate_and_save_constraints(model_type, tool_type, 0, ttable, input_variables, output_variables, model_filename=self.model_filename)
        model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
        return model_list


class GF2Linear_Trans(UnaryOperator):  # Operator for the linear transformation in GF(2^n) defined by a binary matrix: y = M*x
    def __init__(self, input_vars, output_vars, mat, ID = None, constants=None):
        super().__init__(input_vars, output_vars, ID = ID)
        assert len(mat) == len(mat[0]), "The matrix should be a square matrix."
        self.mat = mat
        self.constants = constants


    def generate_implementation(self, implementation_type='python', unroll=False):
        var_in = self.get_var_ID('in', 0, unroll)
        var_out = self.get_var_ID('out', 0, unroll)
        if implementation_type == 'python':
            n = len(self.mat)
            s = var_out + ' = '
            for i in range(n):
                s += "(("
                first = True
                for j in range(n):
                    if self.mat[i][j] == 1:
                        if first is False:
                            s += " ^ "
                        s += f"(({var_in} >> {n-j-1}) & 1)"
                        first = False
                if self.constants is not None and self.constants[i] is not None and self.constants[i] != 0:
                    s += f" ^ {self.constants[i]}) << {n-i-1}) | "
                else:
                    s += f") << {n-i-1}) | "
            s = s.rstrip(' | ')
            return [s]
        elif implementation_type == 'c':
            n = len(self.mat)
            s = var_out + ' = '
            for i in range(n):
                s += "("
                first = True
                for j in range(n):
                    if self.mat[i][j] == 1:
                        if first is False:
                            s += " ^ "
                        s += f"(({var_in} >> {n-j-1}) & 1)"
                        first = False
                if self.constants is not None and self.constants[i] is not None and self.constants[i] != 0:
                    s += f" ^ {self.constants[i]}) << {n-i-1} | "
                else:
                    s += f") << {n-i-1} | "
            s = s.rstrip(' | ') + ';'
            return [s]
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type in ['sat', 'milp'] and (self.model_version in [self.__class__.__name__ + "_XORDIFF"]):
            for i in range(self.output_vars[0].bitsize):
                var_in = []
                for j in range(self.input_vars[0].bitsize):
                    if self.mat[i][j] == 1:
                        var_in.append(self.input_vars[0].ID + '_' + str(j))
                var_out = self.output_vars[0].ID + '_' + str(i)
                model_list.extend(gen_matrix_constraints(var_in, var_out, model_type))
            return model_list
        elif model_type in ['sat', 'milp'] and (self.model_version in [self.__class__.__name__ + "_LINEAR"]):
            mat = np.transpose(self.mat)
            for i in range(self.output_vars[0].bitsize):
                var_in = []
                for j in range(self.input_vars[0].bitsize):
                    if mat[i][j] == 1:
                        var_in.append(self.output_vars[0].ID + '_' + str(j))
                var_out = self.input_vars[0].ID + '_' + str(i)
                model_list.extend(gen_matrix_constraints(var_in, var_out, model_type))
            return model_list
        elif model_type == 'sat':
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                unit_vectors = set()
                for row in self.mat:
                    if row.count(1) == 1 and all(x in (0, 1) for x in row):
                        unit_vectors.add(tuple(row))
                if len(unit_vectors) >= len(self.mat) - 1:
                    model_list = [f'{var_in[0]} -{var_out[0]}', f'-{var_in[0]} {var_out[0]}']
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

        elif model_type == 'milp':
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                unit_vectors = set()
                for row in self.mat:
                    if row.count(1) == 1 and all(x in (0, 1) for x in row):
                        unit_vectors.add(tuple(row))
                if len(unit_vectors) >= len(self.mat) - 1:
                    model_list = [f'{var_in[0]} - {var_out[0]} = 0']
                    model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
