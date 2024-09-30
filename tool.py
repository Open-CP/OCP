# TODOs:
# possibility to have several dimensions on the state

import operators as op
import primitives as prim
import variables as var
import time 
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

def computeDDT(table, input_bitsize, output_bitsize):  # method computing the DDT of the Sbox
    DDT = [[0]*(2**output_bitsize) for _ in range(2**input_bitsize)] 
    for in_diff in range(2**input_bitsize):
        for j in range(2**input_bitsize):
            out_diff = table[j] ^ table[j^in_diff]
            DDT[in_diff][out_diff] += 1 
    return DDT

def test_operator_MILP(constraints):
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return ""
    content = "Minimize\n obj\nSubject To\n"
    if "Weight" in constraints[-1]:
        content += constraints[-1]["Weight"] + ' - obj = 0\n'
    for i, constraint in enumerate(constraints[0:-1]):
        content += constraint + '\n'
    content += "Binary\n" + constraints[-1]["Binary"] + "\nEnd\n"
    filename = 'files/milp.lp'
    with open(filename, "w") as file:
        file.write(content)
    model = gp.read(filename)
    model.setParam('OutputFlag', 0)
    model.Params.PoolSearchMode = 2   # Search for all solutions
    model.Params.PoolSolutions = 1000000  # Assuming you want a large number of solutions
    model.optimize()
    print("Number of total solutions using MILP: ", model.SolCount)
    var_list = [v.VarName for v in model.getVars()]
    var_index_map = {v.VarName: idx for idx, v in enumerate(model.getVars())}
    print(var_list)
    for i in range(model.SolCount):
        model.Params.SolutionNumber = i  
        solution = [model.getVars()[var_index_map[var_name]].Xn for var_name in var_list]
        print(f"Solution #{i + 1}:")
        print(solution)
    print("\n")

    


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


def test_operator_SAT(constraints):
    if pysat_import == False: 
        print("pysat module can't be loaded ... skipping test\n")
        return ""
    num_clause = len(constraints[0:-1])
    num_var, variable_map, numerical_cnf = create_numerical_cnf(constraints[0:-1])
    print("variable_map", variable_map)
    content = f"p cnf {num_var} {num_clause}\n"   
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    filename = 'files/sat.cnf'
    with open(filename, "w") as file:
        file.write(content)
    cnf = CNF(filename)
    solver = CryptoMinisat()
    solver.append_formula(cnf.clauses)
    solutions = []
    while solver.solve():
        model = solver.get_model()
        solutions.append(model)
        block_clause = [-l for l in model]
        solver.add_clause(block_clause)
    solver.delete()
    print("Number of total solutions using SAT: ", len(solutions))
    for solution in solutions:
        print(solution)
    return solutions


# ********************* TEST OF OPERATORS MODELING IM MILP and SAT********************* #   
def TEST_OPERATORS_MILP_SAT():  
    # test Equal
    print("********************* operation: Equal ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("Input variables:")
    my_input[0].display()
    print("Output variables:")
    my_output[0].display()
    equal = op.Equal(my_input, my_output, ID='Equal')
    python_code = equal.generate_model(model_type='python', unroll=True)
    print("Python code: \n", python_code)
    milp_constraints = equal.generate_model(model_type='milp', unroll=True)
    print("MILP constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = equal.generate_model(model_type='sat', unroll=True)    
    print("SAT constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")



    # test rot
    print("********************* operation: Rot ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    rot = op.Rot(my_input, my_output, direction= 'l', amount=2, ID='Rot')
    # rot = op.Rot(my_input, my_output, direction= 'r', amount=2, ID='Rot')
    python_code = rot.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = rot.generate_model(model_type='milp', unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = rot.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")
    
    

    # test Shift
    print("********************* operation: Shift ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    shift = op.Shift(my_input, my_output, direction='l', amount=1, ID='Shift')
    # shift = op.Shift(my_input, my_output, direction='r', amount=1, ID='Shift')
    print("python code:", python_code)
    python_code = shift.generate_model(model_type='python', unroll=True)
    milp_constraints = shift.generate_model(model_type='milp', unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = shift.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")
    

    
    # test ConstantAdd
    print("********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    cons_add = op.ConstantAdd(my_input, my_output, 2, "xor", ID = 'ConstantAddXor')
    python_code = cons_add.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = cons_add.generate_model("milp", unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = cons_add.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")    
    
    
    # test Modadd
    print("********************* operation: ModAdd ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    mod_add = op.ModAdd(my_input, my_output, ID = 'ModAdd')
    python_code = mod_add.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = mod_add.generate_model("milp", unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = mod_add.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")
    

    # test bitwiseAND
    print("********************* operation: bitwiseAND ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    bit_and = op.bitwiseAND(my_input, my_output, ID = 'AND')
    python_code = bit_and.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = bit_and.generate_model("milp", unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = bit_and.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    ddt = computeDDT([0,0,0,1],2,1)
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    print("\n")
    
        
    # test bitwiseOR
    print("********************* operation: bitwiseOR ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    bit_or = op.bitwiseOR(my_input, my_output, ID = 'OR')
    python_code = bit_or.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = bit_or.generate_model("milp", unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = bit_or.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    ddt = computeDDT([0,1,1,1],2,1)
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    print("\n")  
    
   
    # test bitwiseXOR
    print("********************* operation: bitwiseXOR ********************* ")
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    my_input[1].display()
    print("output:")
    my_output[0].display()
    xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR')
    python_code = xor.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints0 = xor.generate_model("milp", unroll=True)
    milp_constraints1 = xor.generate_model("milp", model_version =1, unroll=True)
    milp_constraints2 = xor.generate_model("milp", model_version =2, unroll=True)
    milp_constraints3 = xor.generate_model("milp", model_version =3, unroll=True)
    print("milp constraints 0: \n", milp_constraints0)
    test_operator_MILP(milp_constraints0)
    print("milp constraints 1: \n", milp_constraints1)
    test_operator_MILP(milp_constraints1)
    print("milp constraints 2: \n", milp_constraints2)
    test_operator_MILP(milp_constraints2)
    print("milp constraints 3: \n", milp_constraints3)
    test_operator_MILP(milp_constraints3)
    sat_constraints = xor.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    print("\n")   
   
     
    
    # test bitwiseNOT
    print("********************* operation: bitwiseNOT ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    bit_not = op.bitwiseNOT(my_input, my_output, ID = 'NOT')
    python_code = bit_not.generate_model(model_type='python', unroll=True)
    print("python code:", python_code)
    milp_constraints = bit_not.generate_model("milp", unroll=True)
    print("milp constraints: \n", milp_constraints)
    test_operator_MILP(milp_constraints)
    sat_constraints = bit_not.generate_model(model_type='sat', unroll=True)    
    print("sat constraints: \n", sat_constraints)
    test_operator_SAT(sat_constraints)
    ddt = computeDDT([1,0],1,1)
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    print("\n")  
    
    
    
    # test sbox
    print("********************* operation: ASCON_Sbox ********************* ")
    my_input, my_output = [var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)]
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    sbox = op.ASCON_Sbox(my_input, my_output, ID="sbox")
    python_code = sbox.generate_model(model_type='python', unroll=True)
    print("python code:", python_code) 
    milp_constraints0 = sbox.generate_model("milp", unroll=True)
    milp_constraints1 = sbox.generate_model("milp", model_version =1, unroll=True)
    milp_constraints2 = sbox.generate_model("milp", model_version =2, unroll=True)
    print("milp constraints 0: \n", milp_constraints0)
    test_operator_MILP(milp_constraints0)
    print("milp constraints 1: \n", milp_constraints1)
    test_operator_MILP(milp_constraints1)
    print("milp constraints 2: \n", milp_constraints2)
    test_operator_MILP(milp_constraints2)
    sat_constraints0 = sbox.generate_model(model_type='sat', unroll=True)   
    sat_constraints1 = sbox.generate_model(model_type='sat', model_version =1, unroll=True) 
    sat_constraints2 = sbox.generate_model(model_type='sat', model_version =2, unroll=True)  
    print("sat constraints 0: \n", sat_constraints0)
    test_operator_SAT(sat_constraints0)
    print("sat constraints 1: \n", sat_constraints1)
    test_operator_SAT(sat_constraints1)
    print("sat constraints 2: \n", sat_constraints2)
    test_operator_SAT(sat_constraints2)
    ddt = sbox.computeDDT()
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    print("\n")  
    
 


def solve_milp(filename):
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return ""
    model = gp.read(filename)
    model.optimize()
    if model.status == gp.GRB.Status.OPTIMAL:
        print("Optimal Objective Value:", model.ObjVal)
        # for v in model.getVars(): 
            # print(f"{v.VarName} = {v.Xn}")
        return model.ObjVal
    else:
        print("No optimal solution found.")


def solve_SAT(filename):
    if pysat_import == False: 
        print("pysat module can't be loaded ... skipping test\n")
        return ""
    cnf = CNF(filename)
    solver = CryptoMinisat()
    solver.append_formula(cnf.clauses)
    if solver.solve():
        print("A solution exists.")
        # solution = solver.get_model()
        # print("solution: \n", solution)
        return True
    else:
        print("No solution exists.")
        return False
    


def generate_codes(ciphername, cipher, obj=0):
    cipher.generate_code("files/" + ciphername + ".py", "python")
    cipher.generate_code("files/" + ciphername + "_unrolled.py", "python", True)
    cipher.generate_code("files/" + ciphername + ".c", "c")
    cipher.generate_code("files/" + ciphername + "_unrolled.c", "c", True)
    cipher.generate_code("files/" + ciphername + ".lp", "milp", True)
    cipher.generate_code("files/" + ciphername + ".cnf", "sat", True, obj)
    cipher.generate_figure("files/" + ciphername + ".pdf")
    

def TEST_SPECK32_PERMUTATION(r,obj=0):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation("SPECK32_PERM", 32, my_input, my_output, nbr_rounds=r)
    generate_codes("SPECK32_PERM", my_cipher, obj)

def TEST_SIMON32_PERMUTATION(r, obj=0):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_permutation("SIMON32_PERM", 32, my_input, my_output, nbr_rounds=r)
    generate_codes("SIMON32_PERM", my_cipher, obj)

def TEST_ASCON_PERMUTATION(r):
    my_input, my_output = [var.Variable(5,ID="in"+str(i)) for i in range(64)], [var.Variable(5,ID="out"+str(i)) for i in range(64)]
    my_cipher = prim.ASCON_permutation("ASCON_PERM", my_input, my_output, nbr_rounds=r)
    generate_codes("ASCON_PERM", my_cipher)

def TEST_SKINNY_PERMUTATION(r):
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_permutation("SKINNY_PERM", 64, my_input, my_output, nbr_rounds=r)
    generate_codes("SKINNY_PERM", my_cipher)

def TEST_AES_PERMUTATION(r):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    generate_codes("AES_PERM", my_cipher)
    
def TEST_SPECK32_BLOCKCIPHER(r, obj=0):
    my_plaintext, my_key, my_ciphertext = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="k"+str(i)) for i in range(4)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher("SPECK32", [32, 64], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    generate_codes("SPECK32", my_cipher, obj)

def TEST_SKINNY64_192_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="in"+str(i)) for i in range(48)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_block_cipher("SKINNY64_192", [64, 192], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    generate_codes("SKINNY64_192", my_cipher)


def TEST_CIPHERS_MILP():
    data = []
    for r in range(1,5):
        TEST_SPECK32_PERMUTATION(r)
        ciphername = "SPECK32_PERM"
        # TEST_SIMON32_PERMUTATION(r)
        # ciphername = "SIMON32_PERM"
        # TEST_SPECK32_BLOCKCIPHER(r)
        # ciphername = "SPECK32"
        file_name = "files/" + ciphername + ".lp"
        strat_time = time.time()
        result = solve_milp(file_name)
        end_time = time.time()
        data.append([r, result, end_time - strat_time])
        with open(file_name.replace(".lp", "_DC_MILP.txt"), 'w') as file:
            file.write(f"{'Rounds':<10}{'Result':<10}{'Time (s)':<10}\n")
            file.write('-' * 30 + '\n')
            for row in data:
                file.write(f"{row[0]:<10}{row[1]:<10}{row[2]:<10.2f}\n")


def TEST_CIPHERS_SAT():
    data = []
    for r in range(1,5):
        obj = 0
        flag = False
        strat_time = time.time()
        while not flag:
            print("obj", obj)
            TEST_SPECK32_PERMUTATION(r, obj)
            ciphername = "SPECK32_PERM"
            # TEST_SIMON32_PERMUTATION(r, obj)
            # ciphername = "SIMON32_PERM"
            # TEST_SPECK32_BLOCKCIPHER(r, obj)
            # ciphername = "SPECK32"
            file_name = "files/" + ciphername + ".cnf"
            flag = solve_SAT(file_name)
            obj += 1
        end_time = time.time()
        data.append([r, obj-1, end_time - strat_time])
        with open(file_name.replace(".cnf", "_DC_SAT.txt"), 'w') as file:
            file.write(f"{'Rounds':<10}{'Result':<10}{'Time (s)':<10}\n")
            file.write('-' * 30 + '\n')
            for row in data:
                file.write(f"{row[0]:<10}{row[1]:<10}{row[2]:<10.2f}\n")



TEST_OPERATORS_MILP_SAT()
TEST_CIPHERS_MILP()
TEST_CIPHERS_SAT()

# TEST_SPECK32_PERMUTATION(2)
