# TODOs:
# possibility to have several dimensions on the state

import operators as op
import primitives as prim
import variables as var
import attacks 


    
def test_operator_MILP(operator, model_v="diff_0", mode=0):
    """
    This function generates MILP constraints for the given operator and writes them to a .lp file.
    Args:
    operator (object): The operator object for which the MILP model will be generated.
    model_v (str): Version of the model to be used (default is 'diff_0').
    mode (int): Mode for the operator S-box (default is 0).
    """
    
    # generate milp constraints
    if "Sbox" in str(type(operator).__name__): milp_constraints = operator.generate_model(model_type='milp', model_version = model_v, mode= mode, unroll=True)
    else: milp_constraints = operator.generate_model(model_type='milp', model_version = model_v, unroll=True)
    print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
    
    # generate and solve the milp model 
    filename = f'files/milp_{type(operator).__name__}_{model_v}.lp'
    attacks.attacks_milp_model(constraints=milp_constraints, solving_goal="all_solutions", filename=filename)


def test_operator_SAT(operator, model_v="diff_0", mode=0):
    """
    This function generates SAT constraints for the given operator and writes them to a .cnf file.
    Args:
    operator (object): The operator object for which the SAT model will be generated.
    model_v (str): Version of the model to be used (default is 'diff_0').
    mode (int): Mode for the operator S-box (default is 0).
    """

    # generate sat constraints
    if "Sbox" in str(type(operator).__name__): sat_constraints = operator.generate_model(model_type='sat', model_version=model_v, mode= mode, unroll=True)    
    else: sat_constraints = operator.generate_model(model_type='sat', model_version=model_v, unroll=True)        
    print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
    
    # generate and solve the sat model 
    filename = f'files/sat_{type(operator).__name__}_{model_v}.cnf'
    attacks.attacks_sat_model(constraints=sat_constraints, solving_goal="all_solutions", filename=filename)


# ********************* TEST OF OPERATORS MODELING IN MILP and SAT********************* #   
def TEST_Equal_MILP_SAT():  
    print("\n********************* operation: Equal ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("Input:")
    my_input[0].display()
    print("Output:")
    my_output[0].display()
    equal = op.Equal(my_input, my_output, ID='Equal')
    python_code = equal.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = equal.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(equal)
    test_operator_MILP(equal, "truncated_diff")
    test_operator_SAT(equal)
    test_operator_SAT(equal, "truncated_diff")



def TEST_Rot_MILP_SAT(): 
    print("\n********************* operation: Rot ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    # rot = op.Rot(my_input, my_output, direction= 'l', amount=2, ID='Rot')
    rot = op.Rot(my_input, my_output, direction= 'r', amount=2, ID='Rot')
    python_code = rot.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = rot.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))       
    test_operator_MILP(rot)
    test_operator_SAT(rot)



def TEST_Shift_MILP_SAT(): 
    print("\n********************* operation: Shift ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    shift = op.Shift(my_input, my_output, direction='l', amount=1, ID='Shift')
    # shift = op.Shift(my_input, my_output, direction='r', amount=1, ID='Shift')
    python_code = shift.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = shift.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(shift)
    test_operator_SAT(shift)
    
    

def TEST_Modadd_MILP_SAT(): 
    print("\n********************* operation: ModAdd ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    mod_add = op.ModAdd(my_input, my_output, ID = 'ModAdd')
    python_code = mod_add.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = mod_add.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))   
    test_operator_MILP(mod_add)
    test_operator_SAT(mod_add)
    
    

def TEST_bitwiseAND_MILP_SAT(): 
    print("\n********************* operation: bitwiseAND ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    bit_and = op.bitwiseAND(my_input, my_output, ID = 'AND')
    python_code = bit_and.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = bit_and.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))       
    test_operator_MILP(bit_and)
    test_operator_SAT(bit_and)
    # regard bit-wise AND as an S-box and compute its ddt 
    and_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="and_sbox")
    and_sbox.table = [0,0,0,1]
    ddt = and_sbox.computeDDT()
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    

def TEST_bitwiseOR_MILP_SAT():   
    print("\n********************* operation: bitwiseOR ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    bit_or = op.bitwiseOR(my_input, my_output, ID = 'OR')
    python_code = bit_or.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = bit_or.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(bit_or)
    test_operator_SAT(bit_or)
    or_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="or_sbox")
    or_sbox.table = [0,1,1,1]
    ddt = or_sbox.computeDDT()    
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
    
def TEST_bitwiseXOR_MILP_SAT():  
    print("\n********************* operation: bitwiseXOR ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    bit_xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR')
    python_code = bit_xor.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = bit_xor.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))      
    for v in range(3): 
        test_operator_MILP(bit_xor, model_v = "diff_" + str(v))
    test_operator_MILP(bit_xor, model_v = "truncated_diff")
    test_operator_MILP(bit_xor, model_v = "truncated_diff_1")
    test_operator_SAT(bit_xor)
      
   
     
def TEST_bitwiseNOT_MILP_SAT(): 
    print("\n********************* operation: bitwiseNOT ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    bit_not = op.bitwiseNOT(my_input, my_output, ID = 'NOT')
    python_code = bit_not.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = bit_not.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(bit_not)
    test_operator_SAT(bit_not)
    not_sbox = op.Sbox([var.Variable(1,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=1, output_bitsize=1, ID="not_sbox")
    not_sbox.table = [1,0]
    ddt = not_sbox.computeDDT()   
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
    
def TEST_Sbox_MILP_SAT(): 
    print("\n********************* operation: Sbox ********************* ")
    ascon_sbox = op.ASCON_Sbox([var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of ascon_sbox: ", ascon_sbox.differential_branch_number())
    python_code = ascon_sbox.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = ascon_sbox.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code)) 

    skinny4_sbox = op.Skinny_4bit_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of skinny4_sbox: ", skinny4_sbox.differential_branch_number())

    skinny8_sbox = op.Skinny_8bit_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of skinny8_sbox: ", skinny8_sbox.differential_branch_number())

    gift_sbox = op.GIFT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of gift_sbox: ", gift_sbox.differential_branch_number())

    aes_sbox = op.AES_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of aes_sbox: ", aes_sbox.differential_branch_number())

    twine_sbox = op.TWINE_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of twine_sbox: ", twine_sbox.differential_branch_number())

    present_sbox = op.PRESENT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of present_sbox: ", present_sbox.differential_branch_number())

    knot_sbox = op.KNOT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="sbox")
    print("differential branch number of knot_sbox: ", knot_sbox.differential_branch_number())


    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox]:
        for model_v in ["diff_0", "diff_1", "truncated_diff"]: 
            test_operator_MILP(sbox, model_v, mode=0)
            if str(sbox.__class__.__name__) != "GIFT_Sbox": test_operator_SAT(sbox, model_v, mode=0)


    for sbox in [skinny8_sbox]:
        for model_v in ["diff_0", "diff_p", "truncated_diff"]: 
            test_operator_MILP(sbox, model_v, mode=0)
        test_operator_SAT(sbox, "diff_0", mode=0)


    for sbox in [aes_sbox]:
        for model_v in ["diff_0", "diff_1", "truncated_diff"]: 
            test_operator_MILP(sbox, model_v, mode=0)
        test_operator_MILP(sbox, "diff_p", mode=1)
        test_operator_SAT(sbox, "diff_0", mode=0)    


    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox, skinny8_sbox, aes_sbox]:
        model_v = "truncated_diff_1"
        test_operator_MILP(sbox, model_v)


def TEST_N_XOR_MILP_SAT(): 
    print("\n********************* operation: N_XOR ********************* ")
    n = 2
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(n+1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)]
    print("input:")
    for i in range(n):
        my_input[i].display()
    print("output:")
    my_output[0].display()
    n_xor = op.N_XOR(my_input, my_output, ID = 'N_XOR')
    test_operator_MILP(n_xor)
    test_operator_MILP(n_xor, model_v="diff_1")
    test_operator_SAT(n_xor)
    
   

def TEST_Matrix_MILP_SAT(): 
    print("\n********************* operation: Matrix ********************* ")
    # test aes's matrix
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(4)], [var.Variable(8,ID="out"+str(i)) for i in range(4)]
    print("input:")
    for i in range(len(my_input)): my_input[i].display()
    print("output:")
    for i in range(len(my_output)): my_output[i].display()
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    matrix = op.Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial=0x1b, ID = 'Matrix_AES')
    python_code = matrix.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = matrix.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(matrix)
    test_operator_MILP(matrix, model_v="diff_1")
    test_operator_MILP(matrix, model_v="truncated_diff")
    test_operator_MILP(matrix, model_v="truncated_diff_1")
    test_operator_SAT(matrix)

    # test skinny64's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    print("input:")
    for i in range(len(my_input)): my_input[i].display()
    print("output:")
    for i in range(len(my_output)): my_output[i].display()
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = op.Matrix("mat_skinny", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    python_code = matrix_skinny64.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = matrix_skinny64.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(matrix_skinny64)
    test_operator_MILP(matrix_skinny64, model_v="diff_1")
    test_operator_SAT(matrix_skinny64)
    
    
    # test future's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    print("input:")
    for i in range(len(my_input)):
        my_input[i].display()
    print("output:")
    my_output[0].display()
    mat_future = [[8,9,1,8], [3,2,9,9], [2,3,8,9], [9,9,8,1]]
    matrix = op.Matrix("mat_future", my_input, my_output, mat = mat_future, polynomial=0x3, ID = 'Matrix')
    test_operator_MILP(matrix)
    test_operator_MILP(matrix, model_v="diff_1")
    test_operator_SAT(matrix)
    

def TEST_ConstantAdd_MILP_SAT(): 
    print("\n********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    cons_add = op.ConstantAdd(my_input, my_output, 2, "xor", ID = 'ConstantAddXor')
    python_code = cons_add.generate_model(model_type='python', unroll=True)
    print("Python code: \n", python_code)  
    c_code = cons_add.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(cons_add)
    test_operator_MILP(cons_add, model_v="truncated_diff")
    test_operator_SAT(cons_add)
    test_operator_SAT(cons_add, model_v="truncated_diff")


def TEST_OPERATORS_MILP_SAT():  
    TEST_Equal_MILP_SAT()
    TEST_Rot_MILP_SAT()
    TEST_Shift_MILP_SAT()
    TEST_Modadd_MILP_SAT()
    TEST_bitwiseAND_MILP_SAT()
    TEST_bitwiseOR_MILP_SAT()
    TEST_bitwiseXOR_MILP_SAT()
    TEST_bitwiseNOT_MILP_SAT()
    TEST_Sbox_MILP_SAT()
    TEST_N_XOR_MILP_SAT()
    TEST_Matrix_MILP_SAT()
    TEST_ConstantAdd_MILP_SAT()
    


# ********************* TEST OF CIPHERS CODING IN PYTHON AND C********************* #  
def generate_codes(cipher):
    cipher.generate_code("files/" + cipher.name + ".py", "python")
    cipher.generate_code("files/" + cipher.name + "_unrolled.py", "python", True)
    cipher.generate_code("files/" + cipher.name + ".c", "c")
    cipher.generate_code("files/" + cipher.name + "_unrolled.c", "c", True)
    cipher.generate_figure("files/" + cipher.name + ".pdf")
    

# ********************* TEST OF CIPHERS MODELING IN MILP and SAT********************* #   
def TEST_SPECK_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation(f"SPECK{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher
   

def TEST_SIMON_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_permutation(f"SIMON{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_ASCON_PERMUTATION(r=None):
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(320)], [var.Variable(1,ID="out"+str(i)) for i in range(320)]
    my_cipher = prim.ASCON_permutation("ASCON_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY_PERMUTATION(r=None, version=64):
    my_input, my_output = [var.Variable(int(version/16),ID="in"+str(i)) for i in range(16)], [var.Variable(int(version/16),ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_permutation("SKINNY_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_AES_PERMUTATION(r=None):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_GIFT_PERMUTATION(r=None, version=64): 
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(version)], [var.Variable(1,ID="out"+str(i)) for i in range(version)]
    my_cipher = prim.GIFT_permutation(f"GIFT{version}_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_ROCCA_AD(r=None):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(128+32*r)], [var.Variable(8,ID="out"+str(i)) for i in range(128+32*r)]
    my_cipher = prim.Rocca_AD_permutation(f"ROCCA_AD", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SPECK_BLOCKCIPHER(r=None, version = [32, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher(f"SPECK{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY_BLOCKCIPHER(r=None, version=[64, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/16), int(version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(16)], [var.Variable(word_size,ID="k"+str(i)) for i in range(16*m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_block_cipher(f"SKINNY{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_AES_BLOCKCIPHER(r=None, version = [128, 128]): 
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1]/version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_block_cipher(f"AES{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_SIMON_BLOCKCIPHER(r=None, version=[32,64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_block_cipher(f"SIMON{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_GIFT_BLOCKCIPHER(r=None, version = [64, 128]): 
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = prim.GIFT_block_cipher(f"GIFT{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_DIFF_ATTACK():
    """
    Perform differential cryptanalysis on several ciphers using the "attacks.diff_attacks().
    
    Procedure:
    === Step 1: Set the number of rounds and instantiate the Cipher object ===
    Initialize the cipher with the desired number of rounds. This involves setting up the cipher's structure and defining the number of rounds.

    === Step 2: Configure Model Versions ===
    Configures the differential behavior of the cipher's operations based on the parameter model_version.
    - model_versions = {} by default, which applies the model_version = "diff_0" to all operations.
    - For ARX ciphers, the default setting aims to search for the differential trails with the highest probability.
    - For S-box-based ciphers like ASCON and GIFT, the default setting aims to search for the differential trails with the minimal number of active S-boxes.
    - For S-box-based ciphers, setting model_version = "diff_1" models difference propagation with probabilities, aimed at searching for the highest probability.
    
    === Step 3: Generate Additional Constraints ===
    Allows additional constraints to the model:
    - add_constraints = "" by default, indicating no additional constraints are added to the model.
    - generate additional constraints using "attacks.gen_add_constraints()".

    === Step 4: Execute Differential Attacks ===
    Calls attacks.diff_attacks() to construct and solve the MILP or SAT model.
    - model_type = "milp" by default.
    """
    

    # TEST 1: Search for the best differential trail of r-round SPECK
    r = 3
    cipher = TEST_SPECK_PERMUTATION(r, version = 32) 
    result = attacks.diff_attacks(r, cipher, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_type="sat") 

    # TEST 2: Search for the best differential trail of r-round SIMON
    r = 3
    cipher = TEST_SIMON_PERMUTATION(r, version = 32)
    result = attacks.diff_attacks(r, cipher, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_type="sat") 

    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round ASCON
    r = 2
    cipher = TEST_ASCON_PERMUTATION(r) 
    result = attacks.diff_attacks(r, cipher, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_type="sat") 

    # TEST 4:Search for the minimal number of active differentially S-boxes ofr-round GIFT
    r = 4
    cipher = TEST_GIFT_PERMUTATION(r, version = 64)
    result = attacks.diff_attacks(r, cipher, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_type="sat") 

    # TEST 5: Search for the best differential trail of r-round ASCON
    # set `model_version = "diff_1"` for each S-box.
    r = 1
    cipher = TEST_ASCON_PERMUTATION(r) 
    model_versions = attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[1]}, positions = {r: {"STATE": {1: list(range(64))}} for r in range(1, cipher.nbr_rounds + 1)})
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, model_type="sat") 

    # TEST 6: Search for the best differential trail of r-round GIFT
    # set `model_version = "diff_1"` for each S-box.
    r = 3
    cipher = TEST_GIFT_PERMUTATION(r, version = 64)
    model_versions = attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[0]}, positions = {r: {"STATE": {0: list(range(len(cipher.states["STATE"].constraints[r][0])))}} for r in range(1, cipher.nbr_rounds + 1)})
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, model_type="milp") 

    # TEST 7: Search for the best related-key differential trail of r-round SPECK
    r = 6
    cipher = TEST_SPECK_BLOCKCIPHER(r, version=[32,64]) 
    result = attacks.diff_attacks(r, cipher,  model_type="milp") 
    result = attacks.diff_attacks(r, cipher,  model_type="sat") 

    # TEST 8: Search for the best related-key differential trail of r-round SIMON
    r = 7
    cipher = TEST_SIMON_BLOCKCIPHER(r, version = [32, 64])
    result = attacks.diff_attacks(r, cipher, model_type="milp") 
    result = attacks.diff_attacks(r, cipher, model_type="sat")

    # TEST 9: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT
    r = 6
    cipher = TEST_GIFT_BLOCKCIPHER(r, version = [64, 128])
    result = attacks.diff_attacks(r, cipher, model_type="milp") 

    # TEST 10: Search for the best related-key differential trail of r-round GIFT
    # set `model_version = "diff_1"` for each S-box.
    r = 7
    cipher = TEST_GIFT_BLOCKCIPHER(r, version = [64, 128])
    model_versions = attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[0]}, positions = {r: {"STATE": {0: list(range(16))}} for r in range(1, cipher.nbr_rounds + 1)})
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, model_type="milp") 


    # TEST 11: Search for the best truncated differential trail of r-round AES
    # set model_version = "truncated_diff" for each operation within the cipher
    r = 6
    cipher = TEST_AES_PERMUTATION(r)
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    model_versions = attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    result = attacks.diff_attacks(r, cipher,  model_versions=model_versions, model_type="milp", bitwise=False) 

    # TEST 12: Search for the best truncated related-key differential trail of r-round AES
    # set model_version = "truncated_diff" for each operation within the cipher
    r = 5
    cipher = TEST_AES_BLOCKCIPHER(r, version = [128, 128])
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    model_versions = attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, model_type="milp", bitwise=False) 

    # TEST 13: Search for the best truncated differential trail that are used in Forgery attacks of r-round ROCCA_AD
    # set model_version = "truncated_diff" for each operation within the cipher
    # generate the following constraints to search for the differential trails that are used in Forgery attacks:
    # (1) input difference of the state is 0;
    # (2) difference of the data block is not 0;
    # (3) output difference of the state is 0
    r = 7
    cipher = TEST_ROCCA_AD(r)
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    model_versions = attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    add_cons = attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", rounds=[1], states=["STATE"], layers={"STATE":[0]}, positions={1:{"STATE":{0:[i for i in range(128)]}}}, bitwise=False, value=0)
    add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", rounds=[cipher.nbr_rounds], states=["STATE"], layers={"STATE":[4]}, positions={cipher.nbr_rounds:{"STATE":{4:[i for i in range(128)]}}}, bitwise=False, value=0)
    add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="SUM_GREATER_EQUAL", rounds=[1], states=["STATE"], layers={"STATE":[0]}, positions={1:{"STATE":{0:[i for i in range(128, 128+32*r)]}}}, bitwise=False, value=1)
    result = attacks.diff_attacks(r, cipher, model_versions=model_versions, add_constraints=add_cons, model_type="milp", bitwise=False) 
    
    return result    


if __name__ == '__main__':
    TEST_OPERATORS_MILP_SAT()
    r = 2
    cipher = TEST_SPECK_PERMUTATION(r, version = 32) # version = 32, 48, 64, 96, 128
    cipher = TEST_SIMON_PERMUTATION(r, version = 32) # version = 32, 48, 64, 96, 128
    cipher = TEST_AES_PERMUTATION(r)
    cipher = TEST_ASCON_PERMUTATION(r) 
    cipher = TEST_SKINNY_PERMUTATION(r, version = 64) # version = 64, 128
    cipher = TEST_GIFT_PERMUTATION(r, version = 64) # version = 64, 128
    cipher = TEST_ROCCA_AD(r)

    cipher = TEST_SPECK_BLOCKCIPHER(r, version=[32,64]) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    cipher = TEST_SIMON_BLOCKCIPHER(r, version = [32, 64]) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    cipher = TEST_AES_BLOCKCIPHER(r, version = [128, 128]) # version = [128, 128], [128, 192], [128, 256] 
    cipher = TEST_SKINNY_BLOCKCIPHER(r, version = [64, 64]) # version = [64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]  
    cipher = TEST_GIFT_BLOCKCIPHER(r, version = [64, 128]) # version = [64, 128],  [128, 128]
    generate_codes(cipher)

    TEST_DIFF_ATTACK()







