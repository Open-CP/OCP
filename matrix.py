import numpy as np

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
    for row in bin_matrix:
        print(row)


if __name__ == '__main__':
    # Example for GF(2^4)
    mod_poly_4 = 0b10011  # Modulus polynomial f(x) = x^4 + x + 1
    degree_4 = 4
    mds_future = [
        [1, 2, 3, 8],
        [9, 1, 2, 3],
        [8, 9, 1, 2],
        [3, 8, 9, 1]
    ]
    mds_led = [
        [4, 1, 2, 2],
        [8, 6, 5, 6],
        [0xB, 0xE, 0xA, 9],
        [2, 2, 0xF, 0xB]
    ]
    print("\nGF(2^4):")
    print("mds:\n", mds_led)
    prm = generate_pmr_for_mds(mds_led, mod_poly_4, degree_4)
    print("\nPrimitive Matrix Representation:\n", prm)
    

    # Example for GF(2^8)
    mod_poly_8 = 0b100011011  # Modulus polynomial f(x) = x^8 + x^4 + x^3 + x + 1
    degree_8 = 8
    mds_aes = [
        [2, 3, 1, 1],
        [1, 2, 3, 1],
        [1, 1, 2, 3],
        [3, 1, 1, 2]
    ]
    print("\nGF(2^8):")
    print("mds:\n", mds_aes)
    prm = generate_pmr_for_mds(mds_aes, mod_poly_8, degree_8)
    print("\nPrimitive Matrix Representation:\n", prm)


    # Example for skinny matrix
    mat_skinny = [
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 1, 1, 0],
    [1, 0, 1, 0]
    ]
    expanded_list = generate_bin_matrix(mat_skinny, 4)
    print(expanded_list)

