"""Linear-layer operators over GF(2): ``Matrix`` (binary / MDS / polynomial matrices) and
``GF2Linear_Trans``.

Provides binary-matrix generation (including MDS polynomial-matrix expansion to a bit matrix),
python/C/Verilog code generation, and SAT/MILP models built row by row as XOR constraints over
the active input bits.
"""

import numpy as np
import os
import copy
from operators.operators import Operator, UnaryOperator, RaiseExceptionVersionNotExisting, raise_unknown_implementation_type, raise_unknown_model_type
from tools.operator_constraints import gen_matrix_constraints, gen_matrix_declarations, binary_declaration, integer_declaration, gen_equivalence_constraints
from tools.model_templates import generate_and_save_constraints, gen_constraints_obj_func_from_template, instantiate_constraints_template
from tools.paths import get_matrix_constraints_files_dir
from itertools import product


def gf2_multiply(a, b, mod_poly, degree):
    """Multiply two elements of GF(2^m) modulo ``mod_poly``."""
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


def _is_irreducible_gf2(poly, degree):
    """Test whether ``poly`` (with its x^degree term) is irreducible over GF(2)."""
    def poly_mod(a, b):
        db = b.bit_length()
        while a.bit_length() >= db:
            a ^= b << (a.bit_length() - db)
        return a
    for d in range(2, 1 << (degree // 2 + 1)):
        if poly_mod(poly, d) == 0:
            return False
    return True


def generate_binary_matrix_1(degree):
    """Identity matrix -- the bit-matrix of the GF(2^m) element 1."""
    return [[1 if i == j else 0 for j in range(degree)] for i in range(degree)]


def generate_binary_matrix_2(mod_poly, degree):
    """Companion "multiply-by-x" bit-matrix for GF(2^m), derived from the modulus polynomial."""
    mod_poly = _normalize_mod_poly(mod_poly, degree)
    matrix = [[0 for _ in range(degree)] for _ in range(degree)]
    coefficients = [(mod_poly >> i) & 1 for i in range(degree)]
    for i in range(degree):
        matrix[i][0] = coefficients[degree-i-1]
    for i in range(1, degree):
        matrix[i - 1][i] = 1
    return matrix


def matrix_multiply_mod2(A, B):
    """Multiply two matrices over GF(2) (entrywise mod 2); supports non-square shapes."""
    columns = list(zip(*B))
    return [
        [sum(a & b for a, b in zip(row, col)) & 1 for col in columns]
        for row in A
    ]


def generate_pmr_for_mds(mds, mod_poly, degree):
    """Expand a word-level GF(2^m) matrix into its bit-level binary (PMR) form.

    multiply-by-c is linear over GF(2): its bit-matrix is the XOR of the companion "multiply-by-x"
    powers selected by the bits of c. This handles c == 0 (zero block) and needs no discrete logs.
    """
    mod_poly = _normalize_mod_poly(mod_poly, degree)
    assert _is_irreducible_gf2(mod_poly, degree), f"mod_poly {hex(mod_poly)} is not irreducible over GF(2), so GF(2^{degree}) is not a field (check that degree matches the word bitsize)"
    field_size = 1 << degree
    companion_x = generate_binary_matrix_2(mod_poly, degree)        # binary matrix of multiply-by-x
    powers = [generate_binary_matrix_1(degree)]                     # powers[k] = companion_x^k, built incrementally
    for _ in range(1, degree):
        powers.append(matrix_multiply_mod2(powers[-1], companion_x))
    cache = {}
    def mult_by(c):                                                 # binary matrix of multiply-by-c
        if not (0 <= c < field_size):                              # a coefficient must be a valid GF(2^degree) element
            raise ValueError(f"generate_pmr_for_mds: coefficient {c} is out of range for GF(2^{degree}); "
                             f"expected 0 <= c < {field_size}.")
        if c not in cache:
            acc = [[0] * degree for _ in range(degree)]
            for k in range(degree):
                if (c >> k) & 1:
                    acc = [[acc[i][j] ^ powers[k][i][j] for j in range(degree)] for i in range(degree)]
            cache[c] = acc
        return cache[c]
    size = len(mds)
    pmr = [[mult_by(mds[i][j]) for j in range(size)] for i in range(size)]
    pmr_new = [[0 for _ in range(size * degree)] for _ in range(size * degree)]
    for i in range(size):
        for row_offset in range(degree):
            base_index = i * degree + row_offset
            for j in range(size):
                start_index = j * degree
                pmr_new[base_index][start_index:start_index + degree] = pmr[i][j][row_offset]
    return pmr_new


def generate_bin_matrix(mat, bitsize):
    """Expand a GF(2) word-level matrix into block-binary form (1 -> identity block, 0 -> zero block)."""
    bin_matrix = []
    for i in range(len(mat)):
        row = []
        for j in range(len(mat[i])):
            if mat[i][j] == 1:
                row.append(np.eye(bitsize, dtype=int))
            elif mat[i][j] == 0:
                row.append(np.zeros((bitsize, bitsize), dtype=int))
            else:
                raise ValueError(f"generate_bin_matrix expects a binary (0/1) matrix, but got {mat[i][j]} at ({i},{j}).")
        bin_matrix.append(row)
    return np.block(bin_matrix)


class Matrix(Operator):
    """Matrix multiplication operator: apply the matrix ``mat`` (stored as a list of lists) to the input
    vector of variables, towards the output vector of variables.

    The optional ``polynomial`` defines the GF(2^m) reduction polynomial (e.g. 0x1b for AES); when
    omitted, ``mat`` is interpreted over GF(2). See :meth:`__init__` for the three supported matrix layouts.
    """

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDDIFF_1", "TRUNCATEDDIFF_2", "TRUNCATEDLINEAR", "TRUNCATEDLINEAR_1", "TRUNCATEDLINEAR_2"),
        "milp": ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDDIFF_1", "TRUNCATEDDIFF_2", "TRUNCATEDLINEAR", "TRUNCATEDLINEAR_1", "TRUNCATEDLINEAR_2"),
    }
    SUPPORTED_IMPLEMENTATIONS = ("python", "c")

                          # The optional "polynomial" allors to define the polynomial reduction (not implemted yet)
    def __init__(self, name, input_vars, output_vars, mat, polynomial = None, ID = None):
        """Matrix operator applying ``mat`` to the input word vector.

        Three matrix definitions are currently supported:
          1. GF(2^m) word-level matrix with a reduction polynomial (not None): mat is n*n over GF(2^m),
             entries are field elements (e.g. AES MixColumns [[2,3,1,1],...] with polynomial=0x1b, bitsize=8).
          2. GF(2) word-level matrix: mat is n*n with entries in {0,1} and len(input_vars) == len(mat)
             (e.g. the SKINNY MixColumns 4*4 binary matrix [[1,0,1,1],...], applied per column).
          3. Bit-level binary matrix: mat is already the full (n*bitsize)*(n*bitsize) {0,1} matrix, detected
             by bitsize*len(input_vars) == len(mat) (e.g. SKINNY-64 MixColumns on one column, expanded to 16*16).
        """
        r, c = len(mat), len(mat[0])
        for i in mat:
            if len(i) != c:
                raise ValueError(f"{self.__class__.__name__}: matrix size not consistent")
        if len(input_vars) != c:
            raise ValueError(f"{self.__class__.__name__}: input vector does not match matrix size")
        if len(output_vars) != r:
            raise ValueError(f"{self.__class__.__name__}: output vector does not match matrix size")
        super().__init__(input_vars, output_vars, ID = ID)
        self.name = name
        self.mat = mat
        self.polynomial = polynomial # For AES, polynomial = 0x1b, degree = 8. For SKINNY, polynomial = None, degree = None (i.e. binary matrix over GF(2))

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

    def differential_branch_number(self):
        """Differential branch number of the matrix (not implemented)."""
        raise NotImplementedError("Matrix differential branch number computation is not implemented.")

    def linear_branch_number(self):
        """Linear branch number of the matrix (not implemented)."""
        raise NotImplementedError("Matrix linear branch number computation is not implemented.")

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

    def _word_model_vars(self, dim=1):
        """Return (var_in, var_out) as word-level (activity) model variables for every input/output word."""
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model('in', i, bitwise=False, dim=dim)
        for i in range(len(self.output_vars)):
            var_out += self.get_var_model('out', i, bitwise=False, dim=dim)
        return var_in, var_out

    def _binary_matrix_representation(self):
        """Return the bit-level GF(2) matrix, dispatching on the three ``mat`` definitions (see __init__)."""
        if self.polynomial:  # case 1: GF(2^m) word matrix -> expand to its primitive matrix representation
            return generate_pmr_for_mds(self.mat, self.polynomial, self.input_vars[0].bitsize)
        if len(self.input_vars) == len(self.mat):  # case 2: GF(2) word matrix -> block expansion (1 -> I, 0 -> 0)
            return generate_bin_matrix(self.mat, self.input_vars[0].bitsize)
        if self.input_vars[0].bitsize * len(self.input_vars) == len(self.mat):  # case 3: already a bit-level matrix
            return self.mat
        raise ValueError(f"Matrix {self.mat} not supported.")

    def _generate_bit_matrix_constraints(self, model_type, bin_matrix, source_vars, target_vars, dummy_prefix=None):
        """Per output bit, constrain it to the XOR of the input bits selected by ``bin_matrix``; for MILP
        append the Binary declaration (and Integer dummies for rows with >=3 active inputs).

        Flattens the word/bit variables into plain bit-name lists and delegates the row-by-row
        modeling to the matrix-level helpers in operator_constraints.
        """
        bits_per_source = source_vars[0].bitsize
        bits_per_target = target_vars[0].bitsize
        source_bits = [sv.ID + (f"_{l}" if bits_per_source > 1 else "")
                       for sv in source_vars for l in range(bits_per_source)]
        target_bits = [tv.ID + (f"_{j}" if bits_per_target > 1 else "")
                       for tv in target_vars for j in range(bits_per_target)]
        model_list = gen_matrix_constraints(bin_matrix, source_bits, target_bits, model_type, dummy_prefix)
        if model_type == 'milp':
            model_list += gen_matrix_declarations(bin_matrix, source_bits, target_bits, dummy_prefix)
        return model_list

    def generate_implementation(self, implementation_type='python', unroll=False):
        """Emit a call to the matrix macro: ``(out...) = name(in...)`` (python) or ``name(in..., out...);`` (c)."""
        self.check_supported_implementation(implementation_type)
        input_args = ", ".join(self.get_var_ID('in', i, unroll) for i in range(len(self.input_vars)))
        output_args = ", ".join(self.get_var_ID('out', i, unroll) for i in range(len(self.output_vars)))
        if implementation_type == 'python':
            return [f"({output_args}) = {self.name}({input_args})"]
        elif implementation_type == 'c':
            return [f"{self.name}({input_args}, {output_args});"]
        else:
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def get_header_ID(self):
        """Identity used to de-duplicate headers: class, version, bitsize, matrix and polynomial."""
        return [self.__class__.__name__, self.model_version, self.input_vars[0].bitsize, self.mat, self.polynomial]

    def generate_implementation_header_unique(self, implementation_type='python'):
        """Emit the shared GF(2^m) multiplication macro (GMUL), once per distinct header."""
        if implementation_type == 'python':
            model_list = ["#Galois Field Multiplication Macro", "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
        elif implementation_type == 'c':
            model_list = ["//Galois Field Multiplication Macro", "#define GMUL(a, b, p, d) ({ \\", "\tunsigned int result = 0; \\", "\tunsigned int temp_a = a; \\", "\tunsigned int temp_b = b; \\", "\twhile (temp_b > 0) { \\", "\t\tif (temp_b & 1) \\", "\t\t\tresult ^= temp_a; \\", "\t\ttemp_a <<= 1; \\", "\t\tif (temp_a & (1 << d)) \\", "\t\t\ttemp_a ^= p; \\", "\t\ttemp_b >>= 1; \\", "\t} \\", "\tresult & ((1 << d) - 1); \\","})"];
        return model_list

    def generate_implementation_header(self, implementation_type='python'):
        """Emit the matrix's macro definition (each output word = XOR / GMUL of the input words)."""
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
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + str(self.polynomial) + "," + str(self.input_vars[0].bitsize) + ")"
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + str(self.polynomial) + "," + str(self.input_vars[0].bitsize) + ")"
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
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + str(self.polynomial) + "," + str(self.input_vars[0].bitsize) + ")"
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + str(self.polynomial) + "," + str(self.input_vars[0].bitsize) + ")"
                model_list.append(model + "; \\")
            model_list.append("} ")
            return model_list

    def generate_model(self, model_type='sat', branch_num=None, tool_type="minimize_logic", filename_load=True):
        """Generate the SAT/MILP model. XORDIFF/LINEAR use the bit-matrix; the TRUNCATED* versions use
        either the branch-number inequalities (when ``branch_num`` is given) or the valid-pattern template."""
        self.check_supported_model_version(model_type)
        # Modeling for differential / linear cryptanalysis
        if model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
            bin_matrix = self._binary_matrix_representation()
            dummy_prefix = self.ID + '_d' if model_type == 'milp' else None
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                # XORDIFF: each output bit = XOR of the input bits (rows of the bit-matrix).
                return self._generate_bit_matrix_constraints(model_type, bin_matrix, self.input_vars, self.output_vars, dummy_prefix)
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                # LINEAR: masks propagate through the transpose (input mask = XOR of output masks).
                return self._generate_bit_matrix_constraints(model_type, np.transpose(bin_matrix), self.output_vars, self.input_vars, dummy_prefix)

        # Modeling for truncated differential / linear cryptanalysis
        elif model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1", self.__class__.__name__ + "_TRUNCATEDDIFF_2", self.__class__.__name__ + "_TRUNCATEDLINEAR_2"]:
            # If a branch number is provided, use the MILP branch-number model.
            # TODO: auto-compute branch_num via differential_branch_number()/linear_branch_number() (not implemented).
            if model_type == 'milp' and branch_num is not None and self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                return self._generate_model_truncated_diff_linear_branch_num(model_type, branch_num)
            # Otherwise fall back to the exact valid-pattern model (the _2 variant).
            effective_version = self.model_version
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1"]:
                effective_version = self.__class__.__name__ + "_TRUNCATEDDIFF_2"
                print(f"[WARNING] The {model_type} model for differential branch number = {branch_num} is not implemented. Turn to model_version {effective_version}")
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                effective_version = self.__class__.__name__ + "_TRUNCATEDLINEAR_2"
                print(f"[WARNING] The {model_type} model for linear branch number = {branch_num} is not implemented. Turn to model_version {effective_version}")
            # Generate the model describing all valid input/output patterns.
            self.model_filename = str(get_matrix_constraints_files_dir() / f"constraints_{model_type}_{self.name}_{effective_version}_{tool_type}.txt")
            self.filename_load = filename_load
            return self._generate_model_truncated_diff_linear_valid_patterns(model_type, tool_type, effective_version)

        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else:
            raise_unknown_model_type(self.__class__.__name__, model_type, context=self.model_version)

    def _generate_model_truncated_diff_linear_branch_num(self, model_type, branch_num):
        """MILP truncated model from the matrix branch number: it lower-bounds the total number of active
        input/output words whenever the propagation is nonzero (word-level activity variables)."""
        var_in, var_out = self._word_model_vars()
        var_d = [f"{self.ID}_d"]
        if model_type == 'milp':
            # The first type of modeling. Reference: Nicky Mouha, Qingju Wang, Dawu Gu, and Bart Preneel. Differential and linear cryptanalysis using mixed-integer linear programming.
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                model_list.append(binary_declaration(var_in, var_out, var_d))
                return model_list
            # The second type of modeling. Reference: [1] Christina Boura, Patrick Derbez and Margot Funk. Related-Key Differential Analysis of the AES. [2] Patrick Derbez, Marie Euler, Pierre-Alain Fouque, Phuong Hoa Nguyen. Revisiting Related-Key Boomerang attacks on AES using computer-aided tool.
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDLINEAR_1"]:
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [" + ".join(var_in + var_out) + f" - {len(var_in+var_out)} {var_d[0]} <= 0"]
                model_list.append(binary_declaration(var_in, var_out, var_d))
                return model_list
        else:
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)

    def _generate_model_truncated_diff_linear_valid_patterns(self, model_type, tool_type, effective_version):
        """MILP/SAT truncated model from the exact set of valid word-activity patterns: build the truth table
        of feasible (x,y) patterns, then compile it to constraints (cached to a template file).
        """
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        var_in, var_out = self._word_model_vars(dim=1)

        if self.filename_load and os.path.exists(self.model_filename):
            model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
            return model_list

        if effective_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_2"]:
            all_patterns = self.patterns_where_a_star_is_forced_zero()
            patterns = [(xp, yp) for xp, yp, tag in all_patterns if tag == '0']
            patterns.append(((0,) * input_words, (0,) * output_words))

        elif effective_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR_2"]:
            mat = copy.deepcopy(self.mat)
            mat_trans = np.transpose(self.mat)
            self.mat = mat_trans
            all_patterns = self.patterns_where_a_star_is_forced_zero()
            patterns = [(yp, xp) for xp, yp, tag in all_patterns if tag == '0']
            patterns.append(((0,) * input_words, (0,) * output_words))
            self.mat = mat
        else:
            RaiseExceptionVersionNotExisting(self.__class__.__name__, effective_version, model_type)

        pattern_set = set(patterns)
        truth_bits = []
        for i in range(2**input_words):
            x = tuple('*' if b == '1' else 0 for b in format(i, f"0{input_words}b"))
            for j in range(2**output_words):
                y = tuple('*' if b == '1' else 0 for b in format(j, f"0{output_words}b"))
                pattern = (x, y)
                truth_bits.append("1" if pattern in pattern_set else "0")
        ttable = "".join(truth_bits)

        input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
        constraints, template_obj_fun = generate_and_save_constraints(
            model_type,
            tool_type,
            0,
            ttable,
            input_variables,
            output_variables,
            model_filename=self.model_filename,
        )
        model_list, _ = instantiate_constraints_template(
            constraints,
            template_obj_fun,
            var_in,
            var_out,
        )
        return model_list


class GF2Linear_Trans(UnaryOperator):
    """Linear transformation over GF(2^n) defined by a binary matrix (y = M*x)."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
    }
    SUPPORTED_IMPLEMENTATIONS = ("python", "c")

    def __init__(self, input_vars, output_vars, mat, ID = None, constants=None):
        """Build a GF(2)-linear transform y = M*x defined by the square binary matrix ``mat`` (optional constants)."""
        super().__init__(input_vars, output_vars, ID = ID)
        if len(mat) != len(mat[0]):
            raise ValueError("GF2Linear_Trans: the matrix should be square.")
        self.mat = mat
        self.constants = constants


    def generate_implementation(self, implementation_type='python', unroll=False):
        """Emit y = M*x expanded bit-by-bit (each output bit = XOR of the selected input bits, plus any constant)."""
        self.check_supported_implementation(implementation_type)
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
        else:
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def _bit_diff_linear_constraints(self, matrix, source_id, target_id, model_type):
        """Per target bit i, constrain it to the XOR of the source bits j with ``matrix[i][j] == 1``.

        Builds the source/target bit-name lists for this single-word transform and delegates to the
        matrix-level helpers in operator_constraints.
        """
        source_bits = [f"{source_id}_{j}" for j in range(len(matrix[0]))]
        target_bits = [f"{target_id}_{i}" for i in range(len(matrix))]
        dummy_prefix = self.ID + '_d' if model_type == 'milp' else None
        model_list = gen_matrix_constraints(matrix, source_bits, target_bits, model_type, dummy_prefix)
        if model_type == 'milp':
            model_list += gen_matrix_declarations(matrix, source_bits, target_bits, dummy_prefix)
        return model_list

    def generate_model(self, model_type='sat'):
        """SAT/MILP model: XORDIFF/LINEAR expand each bit as an XOR over the (transposed for LINEAR) matrix row;
        the TRUNCATED* versions collapse to a single word-activity equality when the matrix is (near-)a permutation."""
        self.check_supported_model_version(model_type)
        if model_type in ['sat', 'milp']:
            # Differential: each output bit = XOR of the input bits selected by its matrix row.
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                return self._bit_diff_linear_constraints(self.mat, self.input_vars[0].ID, self.output_vars[0].ID, model_type)
            # Linear: masks propagate through the transposed matrix (input mask = XOR of output masks).
            if self.model_version == self.__class__.__name__ + "_LINEAR":
                return self._bit_diff_linear_constraints(np.transpose(self.mat), self.output_vars[0].ID, self.input_vars[0].ID, model_type)
            # Truncated: if the matrix is (near-)a permutation of unit rows, model as a single word-activity equality.
            if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                unit_vectors = set()
                for row in self.mat:
                    if row.count(1) == 1 and all(x in (0, 1) for x in row):
                        unit_vectors.add(tuple(row))
                if len(unit_vectors) >= len(self.mat) - 1:
                    model_list = gen_equivalence_constraints(var_in, var_out, model_type)
                    if model_type == 'milp':
                        model_list.append(binary_declaration(var_in, var_out))
                    return model_list
                RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else:
            raise_unknown_model_type(self.__class__.__name__, model_type)
