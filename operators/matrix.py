import numpy as np
from operators.operators import Operator, UnaryOperator, RaiseExceptionVersionNotExisting
from operators.boolean_operators import xor_constraints, word_xor_constraints, nxor_constraints, word_nxor_constraints


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


def matrix_constraints(vin, vout, model_type, v_dummy=None):
    assert isinstance(vin, list), "Input variables should be provided as a list in matrix_constraints."
    assert isinstance(vout, str), "Output variable should be provided as a string in matrix_constraints."
    if len(vin) == 1:
        if model_type == 'milp':
            return [f"{vout} - {vin[0]} = 0", "Binary\n" + vin[0] + " " + vout]
        elif model_type == 'sat':
            return [f"{vin[0]} -{vout}", f"-{vin[0]} {vout}"]
    elif len(vin) == 2:
        return xor_constraints(vin[0], vin[1], vout, model_type)
    elif len(vin) >= 3:
        if model_type == 'milp':
            assert isinstance(v_dummy, str), "Dummy variables must be provided for MILP model with more than 2 inputs."
        return nxor_constraints(vin, vout, model_type, v_dummy=v_dummy)
    else:
        raise ValueError(f"[WARNING] Unknown model type {model_type} for Matrix.")

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

    def differential_branch_number(self): # Return differential branch number of the Matrix. TO DO
        pass

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

    def generate_model(self, model_type='sat', branch_num=None):
        model_list = []
        input_words = len(self.input_vars)
        output_words = len(self.output_vars)
        bits_per_input = self.input_vars[0].bitsize
        bits_per_output = self.output_vars[0].bitsize
        if model_type in ['sat', 'milp'] and self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
            if self.polynomial: # Example: AES MDS matrix
                bin_matrix = generate_pmr_for_mds(self.mat, self.polynomial, self.input_vars[0].bitsize)
            elif self.input_vars[0].bitsize * len(self.input_vars) > len(self.mat): # Example: SKINNY 4*4 matrix
                bin_matrix = generate_bin_matrix(self.mat, self.input_vars[0].bitsize)
            elif self.input_vars[0].bitsize * len(self.input_vars) == len(self.mat): # Example: SKINNY 64*64 binary matrix
                bin_matrix = self.mat
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
                        model_list.extend(matrix_constraints(var_in, var_out, model_type, v_dummy=d))
                return model_list
            elif self.model_version in [self.__class__.__name__ + "_LINEAR"]:
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
                        model_list.extend(matrix_constraints(var_in, var_out, model_type, v_dummy=d))
                return model_list
        elif model_type == 'milp' and self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDDIFF_2"]:
            var_in = []
            for i in range(len(self.input_vars)):
                var_in += self.get_var_model('in', i, bitwise=False)
            var_out = []
            for i in range(len(self.output_vars)):
                var_out += self.get_var_model('out', i, bitwise=False)
            var_d = [f"{self.ID}_d"]
            if self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":
                if branch_num == None: 
                    # branch_num =self.differential_branch_number()
                    raise ValueError("[WARNING] Please provide branch number as its calculation is not implemented yet.")
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_1": # Refer:
                if branch_num == None: 
                    # branch_num =self.differential_branch_number()
                    raise ValueError("[WARNING] Please provide branch number as its calculation is not implemented yet.")
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [" + ".join(var_in + var_out) + f" - {len(var_in+var_out)} {var_d[0]} <= 0"]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_2": # The matrix is represented as truncated binary
                assert output_words == len(self.mat) and input_words == len(self.mat[0]), "Matrix size does not match input and output variable sizes."
                for i in range(output_words):  # Loop over the ith output word
                    vin = []
                    for k in range(input_words): # Loop over the kth input word
                        if self.mat[i][k] == 1:
                            vin.append(self.input_vars[k].ID)
                    vout = self.output_vars[i].ID
                    if len(vin) == 1:
                        if model_type == 'milp':
                                model_list += [f"{vout} - {vin[0]} = 0"]
                        elif model_type == 'sat':
                            model_list += [f"{vin[0]}, -{vout}", f"-{vin[0]} {vout}"]
                    elif len(vin) == 2:
                        model_list.extend(word_xor_constraints(vin[0], vin[1], vout, model_type))
                    elif len(vin) >= 3:
                        model_list.extend(word_nxor_constraints(vin, vout, model_type))
                return model_list
            else:  RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


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
                model_list.extend(matrix_constraints(var_in, var_out, model_type))
            return model_list
        elif model_type in ['sat', 'milp'] and (self.model_version in [self.__class__.__name__ + "_LINEAR"]):
            mat = np.transpose(self.mat)
            for i in range(self.output_vars[0].bitsize):
                var_in = []
                for j in range(self.input_vars[0].bitsize):
                    if mat[i][j] == 1:
                        var_in.append(self.output_vars[0].ID + '_' + str(j))
                var_out = self.input_vars[0].ID + '_' + str(i)
                model_list.extend(matrix_constraints(var_in, var_out, model_type))
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
