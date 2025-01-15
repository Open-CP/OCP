# TODOs:
# possibility to have several dimensions on the state

import operators as op
import primitives as prim
import variables as var
import matplotlib.pyplot as plt



try:
    import gurobipy as gp
    gurobipy_import = True
except ImportError:
    print("gurobipy module can't be loaded \n")
    gurobipy_import = False
    pass

try:
    from pysat.solvers import CryptoMinisat
    from pysat.formula import CNF
    pysat_import = True
except ImportError:
    print("pysat module can't be loaded \n")
    pysat_import = False
    pass



    
def test_operator_MILP(operator, model_v="diff_0", mode=0):
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return ""
    if "Sbox" in str(type(operator).__name__): milp_constraints = operator.generate_model(model_type='milp', model_version = model_v, mode= mode, unroll=True)
    else: milp_constraints = operator.generate_model(model_type='milp', model_version = model_v, unroll=True)
    print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
    content = "Minimize\n obj\nSubject To\n"
    if hasattr(operator, 'weight'): content += operator.weight + ' - obj = 0\n'
    bin_vars = []
    in_vars = []
    for constraint in milp_constraints:
        if "Binary" in constraint:
            constraint_split = constraint.split('Binary\n')
            content += constraint_split[0]
            bin_vars += constraint_split[1].strip().split()
        elif "Integer" in constraint:
            constraint_split = constraint.split('Integer\n')
            content += constraint_split[0]
            in_vars += constraint_split[1].strip().split()
        else: content += constraint + '\n'
    if len(bin_vars) > 0: content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if len(in_vars) > 0: content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    filename = f'files/milp_{type(operator).__name__}_{model_v}.lp'
    with open(filename, "w") as file:
        file.write(content)
    model = gp.read(filename)
    model.setParam('OutputFlag', 0)   # no output 
    model.Params.PoolSearchMode = 2   # Search for all solutions
    model.Params.PoolSolutions = 1000000  # Assuming you want a large number of solutions
    model.optimize()
    print("Number of solutions by solving the MILP model: ", model.SolCount)
    var_list = [v.VarName for v in model.getVars()]
    var_index_map = {v.VarName: idx for idx, v in enumerate(model.getVars())}
    sol_list = []
    # print("Solutions:")
    # print("var_list: ", var_list)
    for i in range(model.SolCount):
        model.Params.SolutionNumber = i  
        solution = [int(round(model.getVars()[var_index_map[var_name]].Xn)) for var_name in var_list]
        # print(solution)
        sol_list.append(solution)
    return var_list, sol_list, model

    

def create_numerical_cnf(cnf):
    # creating dictionary (variable -> string, numeric_id -> int)
    family_of_variables = ' '.join(cnf).replace('-', '')
    variables = sorted(set(family_of_variables.split()))
    variable2number = {variable: i + 1 for (i, variable) in enumerate(variables)}
    # creating numerical CNF
    numerical_cnf = []
    for clause in cnf:
        literals = clause.split()
        numerical_literals = []
        lits_are_neg = (literal[0] == '-' for literal in literals)
        numerical_literals.extend(tuple(f'{"-" * lit_is_neg}{variable2number[literal[lit_is_neg:]]}'
                                  for lit_is_neg, literal in zip(lits_are_neg, literals)))
        numerical_clause = ' '.join(numerical_literals)
        numerical_cnf.append(numerical_clause)
    return len(variables), variable2number, numerical_cnf



def test_operator_SAT(operator, model_v="diff_0", mode=0):
    if pysat_import == False: 
        print("pysat module can't be loaded ... skipping test\n")
        return ""
    if "Sbox" in str(type(operator).__name__): sat_constraints = operator.generate_model(model_type='sat', model_version=model_v, mode= mode, unroll=True)    
    else: sat_constraints = operator.generate_model(model_type='sat', model_version=model_v, unroll=True)        
    print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
    num_clause = len(sat_constraints)
    num_var, variable_map, numerical_cnf = create_numerical_cnf(sat_constraints)
    content = f"p cnf {num_var} {num_clause}\n"   
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    filename = f'files/sat_{type(operator).__name__}_{model_v}.cnf'
    with open(filename, "w") as file:
        file.write(content)
    cnf = CNF(filename)
    solver = CryptoMinisat()
    solver.append_formula(cnf.clauses)
    sol_list = []
    while solver.solve():
        model = solver.get_model()
        sol_list.append(model)
        block_clause = [-l for l in model]
        solver.add_clause(block_clause)
    solver.delete()
    print("Number of solutions by solving the SAT model: ", len(sol_list))
    # print("Solutions:")
    # print("variable_map: ", variable_map)
    # for solution in sol_list:
    #     print(solution)
    return variable_map, sol_list



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
    for i in range(len(my_input)):
        my_input[i].display()
    print("output:")
    my_output[0].display()
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    matrix = op.Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial=0x1b, ID = 'Matrix')
    python_code = matrix.generate_model(model_type='python', unroll=True)
    print("Python code: \n", "\n".join(python_code))    
    c_code = matrix.generate_model(model_type='c', unroll=True)
    print("C code: \n", "\n".join(c_code))    
    test_operator_MILP(matrix)
    test_operator_MILP(matrix, model_v="diff_1")
    test_operator_MILP(matrix, model_v="truncated_diff")
    test_operator_MILP(matrix, model_v="truncated_diff_1")
    test_operator_SAT(matrix)
    
    
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
def TEST_SPECK32_PERMUTATION(r):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation("SPECK32_PERM", 32, my_input, my_output, nbr_rounds=r)
    return my_cipher
   

def TEST_SIMON32_PERMUTATION(r):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_permutation("SIMON32_PERM", 32, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_ASCON_PERMUTATION(r):
    my_input, my_output = [var.Variable(5,ID="in"+str(i)) for i in range(64)], [var.Variable(5,ID="out"+str(i)) for i in range(64)]
    my_cipher = prim.ASCON_permutation("ASCON_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY_PERMUTATION(r):
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_permutation("SKINNY_PERM", 64, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_AES_PERMUTATION(r):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_GIFT64_permutation(r):
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.GIFT_permutation("GIFT64_PERM", 64, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SPECK32_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="k"+str(i)) for i in range(4)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher("SPECK32", [32, 64], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY64_192_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="k"+str(i)) for i in range(48)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_block_cipher("SKINNY64_192", [64, 192], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_AES_BLOCKCIPHER(r): 
    version = [128, 128] # block_size = version[0] = 128, key_size = version[1] = 128, 192, 256
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1] / version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_block_cipher(f"AES_{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_Rocca_AD(r):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(128+32*r)], [var.Variable(8,ID="out"+str(i)) for i in range(128+32*r)]
    my_cipher = prim.Rocca_AD(f"ROCCA_AD", my_input, my_output, nbr_rounds=r)
    return my_cipher


if __name__ == '__main__':
    TEST_OPERATORS_MILP_SAT()
    r = 2
    cipher = TEST_SPECK32_PERMUTATION(r)
    # cipher = TEST_SIMON32_PERMUTATION(r)
    # cipher = TEST_AES_PERMUTATION(r)
    # cipher = TEST_ASCON_PERMUTATION(r) # TO DO
    # cipher = TEST_SKINNY_PERMUTATION(r) # TO DO
    # cipher = TEST_GIFT64_permutation(r) # TO DO
    # cipher = TEST_Rocca_AD(r)

    # cipher = TEST_SPECK32_BLOCKCIPHER(r)
    # cipher = TEST_AES_BLOCKCIPHER(r)
    # cipher = TEST_SKINNY64_192_BLOCKCIPHER(r) # TO DO
    generate_codes(cipher)







