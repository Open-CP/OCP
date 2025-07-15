import numpy as np
import copy
from operators.operators import Operator, RaiseExceptionVersionNotExisting
from operators.boolean_operators import N_XOR


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
        return 5 # the branch number of matrix for aes is 5, to do for other ciphers
        
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
        if model_type == 'milp' or model_type == 'sat': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1"]:
                model_list = []
                if self.polynomial: bin_matrix = generate_pmr_for_mds(self.mat, self.polynomial, self.input_vars[0].bitsize)
                elif self.input_vars[0].bitsize * len(self.input_vars) > len(self.mat):
                    bin_matrix = generate_bin_matrix(self.mat, self.input_vars[0].bitsize)
                elif self.input_vars[0].bitsize * len(self.input_vars) == len(self.mat):
                    bin_matrix = self.mat
                for i in range(len(self.mat)):
                    for j in range(self.input_vars[0].bitsize):
                        var_in = []
                        var_out = []
                        for k in range(len(self.mat)):
                            for l in range(self.input_vars[0].bitsize):
                                if bin_matrix[self.input_vars[0].bitsize*i+j][self.input_vars[0].bitsize*k+l] == 1:
                                    vi = copy.deepcopy(self.input_vars[i])
                                    vi.bitsize = 1
                                    vi.ID = self.input_vars[k].ID + '_' + str(l)
                                    var_in.append(vi)
                        vo = copy.deepcopy(self.output_vars[i])
                        vo.bitsize = 1
                        vo.ID = self.output_vars[i].ID + '_' + str(j)
                        var_out.append(vo)
                        n_xor = N_XOR(var_in, var_out, ID=self.ID+"_"+str(self.input_vars[0].bitsize*i+j))
                        n_xor.model_version = self.model_version.replace(self.__class__.__name__, n_xor.__class__.__name__)
                        cons = n_xor.generate_model(model_type)
                        model_list += cons
                return model_list
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":
                var_in, var_out = [self.get_var_ID('in', i, unroll=True) for i in range(len(self.input_vars))], [self.get_var_ID('out', i, unroll=True)for i in range (len(self.output_vars))]
                var_d = [f"{self.ID}_d"] 
                if branch_num == None: branch_num =self.differential_branch_number() 
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_1":
                var_in, var_out = [self.get_var_ID('in', i, unroll=True) for i in range(len(self.input_vars))], [self.get_var_ID('out', i, unroll=True)for i in range (len(self.output_vars))]
                var_d = [f"{self.ID}_d"] 
                if branch_num == None: branch_num =self.differential_branch_number() 
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [" + ".join(var_in + var_out) + f" - {len(var_in+var_out)} {var_d[0]} <= 0"]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            else:  RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
