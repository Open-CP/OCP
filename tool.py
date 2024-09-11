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
    # init
    content = "Minimize\n obj\nSubject To\n"
    if "Weight" in  constraints[-1]:
        content += constraints[-1]["Weight"] + ' - obj = 0\n'
    for i, constraint in enumerate(constraints[0:-1]):
        content += constraint + '\n'
    content += "Binary\n" + constraints[-1]["Binary"] + "\nEnd\n"
    filename = 'milp.lp'
    with open(filename, "w") as file:
        file.write(content)
    model = gp.read(filename)
    model.Params.PoolSearchMode = 2   # Search for all solutions
    model.Params.PoolSolutions = 1000  # Assuming you want a large number of solutions
    model.optimize()
    print(model.SolCount)
    for i in range(model.SolCount):
        model.Params.SolutionNumber = i
        print(f"Solution #{i+1}:")
        for v in model.getVars():
            print(f"{v.VarName} = {v.Xn}")


# ********************* TEST OF OPERATORS MODELING IM MILP ********************* #   
def TEST_OPERATORS_MILP():    
    # test equal
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    operator = op.Equal(my_input, my_output, ID='Equal')
    print("********************* operation: Equal ********************* ")
    print(operator.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    constraints = operator.generate_model(model_type='milp', unroll=True)
    print("Equal constraints: \n", constraints)
    print("\n")
    
    # test rot
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)]
    rot = op.Rot(my_input, my_output, direction='l',amount=3)
    print("********************* operation: Rot ********************* ")
    print(rot.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    print("Rot constraints: \n", rot.generate_model("milp", unroll=True))
    print("\n")    
    
    # test shift
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)]
    shift = op.Shift(my_input, my_output, direction='l',amount=3)
    print("********************* operation: Shift ********************* ")
    print("python:", shift.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    print("Shift constraints: \n", shift.generate_model("milp", unroll=True))
    print("\n")    
    
    # test ConstantAdd
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    cons_add = op.ConstantAdd(my_input, my_output, 2, "xor")
    print("********************* operation: ConstantAdd ********************* ")
    print("python:", cons_add.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    print("ConstantAdd constraints: \n", cons_add.generate_model("milp", unroll=True))
    print("\n")    
    
    # test Modadd
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(2)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    mod_add = op.ModAdd(my_input, my_output, ID = 'ModAdd')
    print("********************* operation: ModAdd ********************* ")
    print("python:", mod_add.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    mod_add_constraints = mod_add.generate_model("milp", unroll=True)
    print("ModAdd constraints: \n", mod_add_constraints)
    print("\n")

    # test bitwiseAND
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    and_ = op.bitwiseAND(my_input, my_output, ID = 'AND')
    print("********************* operation: bitwiseAND ********************* ")
    print("python:", and_.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    and_constraints = and_.generate_model("milp", unroll=True)
    print("ConstantAND constraints: \n", and_constraints)
    print("ddt of and_table", computeDDT([0,0,0,1],2,1))
    test_operator_MILP(and_constraints)
    print("\n")
        
    # test bitwiseOR
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(2)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    or_ = op.bitwiseOR(my_input, my_output, ID = 'OR')
    print("********************* operation: bitwiseOR ********************* ")
    print("python:", or_.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    or_constraints = or_.generate_model("milp", unroll=True)
    print("ConstantOR constraints: \n", or_constraints)
    print("ddt of or_table", computeDDT([0,1,1,1],2,1))
    test_operator_MILP(or_constraints)
    print("\n")    
    
    # test bitwiseXOR
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR')
    print("********************* operation: bitwiseXOR ********************* ")
    print("python:", xor.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    xor_constraints0 = xor.generate_model("milp", unroll=True)
    xor_constraints1 = xor.generate_model("milp", model_version =1, unroll=True)
    xor_constraints2 = xor.generate_model("milp", model_version =2, unroll=True)
    xor_constraints3 = xor.generate_model("milp", model_version =3, unroll=True)
    print("bitwiseXOR constraints 0: \n", xor_constraints0)
    print("bitwiseXOR constraints 1: \n", xor_constraints1)
    print("bitwiseXOR constraints 2: \n", xor_constraints2)
    print("bitwiseXOR constraints 3: \n", xor_constraints3)
    test_operator_MILP(xor_constraints0)
    print("\n")    
    
    # test bitwiseNOT
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    not_ = op.bitwiseNOT(my_input, my_output)
    print("********************* operation: bitwiseNOT ********************* ")
    print("python:", not_.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    not_constraints = not_.generate_model("milp", unroll=True)
    print("bitwiseNOT constraints: \n", not_constraints)
    print("ddt of not_table", computeDDT([1,0],1,1))
    test_operator_MILP(not_constraints)
    print("\n")    
    
    # test sbox
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    operator = op.GIFT_Sbox(my_input, my_output, ID="sb")
    print("********************* operation: bitwiseNOT ********************* ")
    print("python:", operator.generate_model(model_type='python', unroll=True))
    print("input:")
    my_input[0].display()
    print("output:")
    my_output[0].display()
    constraints0 = operator.generate_model(model_type='milp', model_version=0, unroll=True)
    constraints1 = operator.generate_model(model_type='milp', model_version=1, unroll=True)
    constraints2 = operator.generate_model(model_type='milp', model_version=2, unroll=True)
    print("GIFT_Sbox constraints 0: \n", constraints0)
    print("GIFT_Sbox constraints 1: \n", constraints1)
    print("GIFT_Sbox constraints 2: \n", constraints2)
    print("\n")
    test_operator_MILP(constraints0)
    # GIFT_ddt = operator.computeDDT()



def solve_milp(filename):
    if gurobipy_import == False: 
        print("gurobipy module can't be loaded ... skipping test\n")
        return ""
    model = gp.read(filename)
    # model.Params.PoolSearchMode = 2   # Search for all solutions
    # model.Params.PoolSolutions = 100000  # Assuming you want a large number of solutions
    model.optimize()
    if model.status == gp.GRB.Status.OPTIMAL:
        print("Optimal Objective Value:", model.ObjVal)
        # for v in model.getVars():
        #     print(f"{v.VarName} = {v.Xn}")
        return model.ObjVal
    else:
        print("No optimal solution found.")

def generate_codes(ciphername, cipher):
    cipher.generate_code("files/" + ciphername + ".py", "python")
    cipher.generate_code("files/" + ciphername + "_unrolled.py", "python", True)
    cipher.generate_code("files/" + ciphername + ".c", "c")
    cipher.generate_code("files/" + ciphername + "_unrolled.c", "c", True)
    cipher.generate_code("files/" + ciphername + ".lp", "milp", True)
    cipher.generate_figure("files/" + ciphername + ".pdf")
    

def TEST_SPECK32_PERMUTATION(r):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation("SPECK32_PERM", 32, my_input, my_output, nbr_rounds=r)
    generate_codes("SPECK32_PERM", my_cipher)

def TEST_SIMON32_PERMUTATION(r):
    my_input, my_output = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_permutation("SIMON32_PERM", 32, my_input, my_output, nbr_rounds=r)
    generate_codes("SIMON32_PERM", my_cipher)

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
    
def TEST_SPECK32_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(16,ID="in"+str(i)) for i in range(2)], [var.Variable(16,ID="k"+str(i)) for i in range(4)], [var.Variable(16,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher("SPECK32", [32, 64], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    generate_codes("SPECK32", my_cipher)

def TEST_SKINNY64_192_BLOCKCIPHER(r):
    my_plaintext, my_key, my_ciphertext = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="in"+str(i)) for i in range(48)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_block_cipher("SKINNY64_192", [64, 192], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    generate_codes("SKINNY64_192", my_cipher)


def TEST_CIPHERS_MILP():
    data = []
    for r in range(2,8):
        # TEST_SPECK32_PERMUTATION(r)
        # ciphername = "SPECK32_PERM"
        TEST_SIMON32_PERMUTATION(r)
        ciphername = "SIMON32_PERM"
        # TEST_SPECK32_BLOCKCIPHER(r)
        # ciphername = "SPECK32"
        file_name = "files/" + ciphername + ".lp"
        strat_time = time.time()
        result = solve_milp(file_name)
        end_time = time.time()
        t = end_time - strat_time
        data.append([r, result, t])
        with open(file_name.replace(".lp", "_DAS.txt"), 'w') as file:
            file.write(f"{'Rounds':<10}{'Result':<10}{'Time (s)':<10}\n")
            file.write('-' * 30 + '\n')
            for row in data:
                file.write(f"{row[0]:<10}{row[1]:<10}{row[2]:<10.2f}\n")


# TEST_OPERATORS_MILP()
# TEST_CIPHERS_MILP()

TEST_SPECK32_PERMUTATION(10)
