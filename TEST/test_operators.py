import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import variables.variables as var
import solving.solving as solving
import attacks.attacks as attacks
from operators.operators import Equal, Rot, Shift, ConstantAdd
from operators.matrix import Matrix, generate_pmr_for_mds, generate_bin_matrix
from operators.boolean_operators import XOR, N_XOR, AND, OR, NOT, ANDXOR
from operators.Sbox import Sbox, ASCON_Sbox, Skinny_4bit_Sbox, Skinny_8bit_Sbox, GIFT_Sbox, AES_Sbox, TWINE_Sbox, PRESENT_Sbox, KNOT_Sbox
from operators.modular_operators import ModAdd
from operators.AESround import AESround
script_dir = os.path.dirname(os.path.abspath(__file__)) 



def test_operator_implementation(operator, implementation_type):
    # Generate and print implementation code
    code = operator.generate_implementation(implementation_type=implementation_type, unroll=True)
    print(f"{implementation_type} code with unroll=True: \n", "\n".join(code)) 
    

def test_operator_model(operator, model_type, model_v_list=["diff_0"], mode=0):
    # Generate and solve MILP models
    base_path = os.path.join(script_dir, '..', 'files')
    base_path = os.path.abspath(base_path)
    if model_type == "milp":
        # Support MILP Solvers: "Gurobi", "SCIP"
        for model_v in model_v_list:
            filename = os.path.join(base_path, f'milp_{operator.ID}_{model_v}.lp')
            # Set model_version of the operator
            operator.model_version = model_v

            # Generate milp constraints
            if "Sbox" in operator.__class__.__name__: 
                milp_constraints = operator.generate_model(model_type='milp', mode= mode)
            else:
                milp_constraints = operator.generate_model(model_type='milp')
            print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
            
            # Define objective function if weight exists
            if hasattr(operator, "weight"):
                obj_fun = operator.weight

                # Generate MILP model
                model = solving.gen_milp_model(constraints=milp_constraints, obj_fun=obj_fun, filename=filename) 
                
                # Solve MILP model for the optimal solution
                sol_list, obj_list = solving.solve_milp(filename, solving_goal="optimize", solver="Gurobi")  
                # print(f"Optimal solutions:\n{sol_list}\nObjective function:\n{obj_list}\n")
            
            # Generate MILP model without objective function
            model = solving.gen_milp_model(constraints=milp_constraints, filename=filename) 
            # Solve MILP model for all solutions
            sol_list, obj_list = solving.solve_milp(filename, solving_goal="all_solutions")
            # print(f"All solutions:\n{sol_list}\nObjective function:\n{obj_list}\nnumber of solutions: {len(sol_list)}")

    elif model_type == "sat":
        # Support SAT Solvers: "Default", "Cadical103", "Cadical153", "Cadical195", "CryptoMinisat", "Gluecard3", "Gluecard4", "Glucose3", "Glucose4", "Lingeling", "MapleChrono", "MapleCM", "Maplesat", "Mergesat3", "Minicard", "Minisat22"     
        # Generate and solve SAT models
        for model_v in model_v_list:
            filename = os.path.join(base_path, f'sat_{operator.ID}_{model_v}.cnf')
            # Set model_version of the operator
            operator.model_version = model_v

            # Generate sat constraints
            if "sbox" in operator.ID: 
                sat_constraints = operator.generate_model(model_type='sat', mode= mode, unroll=True)
            else:
                sat_constraints = operator.generate_model(model_type='sat', unroll=True)
            print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
            
            # Define objective function if weight exists
            if hasattr(operator, "weight"):
                # set the value of the objective function equal to 0.
                obj_var = operator.weight  
                obj_constraints = attacks.gen_sequential_encoding_sat(hw_list=obj_var, weight=0)

                # Generate SAT model.
                model, variable_map = solving.gen_sat_model(constraints=sat_constraints+obj_constraints, filename=filename) 
                # print("variable_map in sat:\n", variable_map)

                # Solve SAT model for a feasible solution.
                sol_list = solving.solve_sat(filename, variable_map, solving_goal="optimize", solver="Default")  
                # print(f"Optimal solutions:\n{sol_list}")
            
            # Generate SAT model without the objective function.
            model, variable_map = solving.gen_sat_model(constraints=sat_constraints, filename=filename) 
            # print("variable_map in sat:\n", variable_map)
            # Solve SAT model for all solutions
            sol_list = solving.solve_sat(filename, variable_map, solving_goal="all_solutions")
            # print(f"All solutions:\n{sol_list}")
        


# ********************* TEST OF OPERATORS MODELING IN MILP and SAT********************* #   
def TEST_Equal():  
    print("\n********************* operation: Equal ********************* ")
    # Create input and output variables for the Equal operation
    my_input, my_output = [var.Variable(2, ID=f"in{i}") for i in range(1)], [var.Variable(2, ID=f"out{i}") for i in range(1)]
    # Instantiate the Equal operation
    equal = Equal(my_input, my_output, ID='Equal')
    equal.display()
    test_operator_implementation(equal, "python")
    test_operator_implementation(equal, "c")
    test_operator_model(equal, "milp", model_v_list=[equal.__class__.__name__+"_DIFF", equal.__class__.__name__+"_DIFF_TRUNCATED"])
    test_operator_model(equal, "sat", model_v_list=[equal.__class__.__name__+"_DIFF", equal.__class__.__name__+"_DIFF_TRUNCATED"])
    

def TEST_Rot(): 
    print("\n********************* operation: Rot ********************* ")
    # Create input and output variables, instantiate the Rot operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    # rot = Rot(my_input, my_output, direction= 'l', amount=2, ID='Rot') # rotation by left
    rot = Rot(my_input, my_output, direction= 'r', amount=2, ID='Rot') # rotation by right
    rot.display()
    test_operator_implementation(rot, "python")
    test_operator_implementation(rot, "c")
    test_operator_model(rot, "milp", [rot.__class__.__name__+"_DIFF"])
    test_operator_model(rot, "sat", [rot.__class__.__name__+"_DIFF"])


def TEST_Shift(): 
    print("\n********************* operation: Shift ********************* ")
    # Create input and output variables, instantiate the Shift operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    shift = Shift(my_input, my_output, direction='l', amount=1, ID='Shift') # shift by left
    # shift = Shift(my_input, my_output, direction='r', amount=1, ID='Shift') # shift by right
    shift.display()
    test_operator_implementation(shift, "python")
    test_operator_implementation(shift, "c")
    test_operator_model(shift, "milp", [shift.__class__.__name__+"_DIFF"])
    test_operator_model(shift, "sat", [shift.__class__.__name__+"_DIFF"])
    

def TEST_Modadd(): 
    print("\n********************* operation: ModAdd ********************* ")
    # Create input and output variables, instantiate the ModAdd operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    mod_add = ModAdd(my_input, my_output, ID = 'ModAdd')
    mod_add.display()
    test_operator_implementation(mod_add, "python")
    test_operator_implementation(mod_add, "c")
    test_operator_model(mod_add, "milp", [mod_add.__class__.__name__+"_DIFF"])
    test_operator_model(mod_add, "sat", [mod_add.__class__.__name__+"_DIFF"])
    
    
def TEST_AND(): 
    print("\n********************* operation: bitwiseAND ********************* ")
    # Create input and output variables, instantiate the bitwiseAND operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_and = AND(my_input, my_output, ID = 'AND')
    bit_and.display()
    test_operator_implementation(bit_and, "python")
    test_operator_implementation(bit_and, "c")
    test_operator_model(bit_and, "milp", [bit_and.__class__.__name__+"_DIFF"])
    test_operator_model(bit_and, "sat", [bit_and.__class__.__name__+"_DIFF"])
    # regard bit-wiseAND as an S-box and compute its ddt 
    and_sbox = Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="and_sbox")
    and_sbox.table = [0,0,0,1]
    ddt = and_sbox.computeDDT()
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_OR():   
    print("\n********************* operation: bitwiseOR ********************* ")
    # Create input and output variables, instantiate the bitwiseOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_or = OR(my_input, my_output, ID = 'OR')
    bit_or.display()
    test_operator_implementation(bit_or, "python")
    test_operator_implementation(bit_or, "c")
    test_operator_model(bit_or, "milp", [bit_or.__class__.__name__+"_DIFF"])
    test_operator_model(bit_or, "sat", [bit_or.__class__.__name__+"_DIFF"])
    # regard bit-wiseOR as an S-box and compute its ddt 
    or_sbox = Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="or_sbox")
    or_sbox.table = [0,1,1,1]
    ddt = or_sbox.computeDDT()    
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_XOR():  
    print("\n********************* operation: bitwiseXOR ********************* ")
    # Create input and output variables, instantiate the bitwiseXOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_xor = XOR(my_input, my_output, ID = 'XOR')
    bit_xor.display()    
    test_operator_implementation(bit_xor, "python")
    test_operator_implementation(bit_xor, "c")
    test_operator_model(bit_xor, "milp", [bit_xor.__class__.__name__+"_DIFF", bit_xor.__class__.__name__+"_DIFF_1", bit_xor.__class__.__name__+"_DIFF_2", bit_xor.__class__.__name__+"_DIFF_TRUNCATED", bit_xor.__class__.__name__+"_DIFF_TRUNCATED_1"])
    test_operator_model(bit_xor, "sat", [bit_xor.__class__.__name__+"_DIFF", bit_xor.__class__.__name__+"_DIFF_TRUNCATED"])
    
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(2)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    bit_xor = XOR(my_input, my_output, ID = 'XOR_m', mat=[[1,None],[2,None],[3,None],[0,1]])
    bit_xor.display()    
    test_operator_implementation(bit_xor, "python")
    test_operator_implementation(bit_xor, "c")
    test_operator_model(bit_xor, "milp", [bit_xor.__class__.__name__+"_DIFF", bit_xor.__class__.__name__+"_DIFF_1", bit_xor.__class__.__name__+"_DIFF_2", bit_xor.__class__.__name__+"_DIFF_TRUNCATED", bit_xor.__class__.__name__+"_DIFF_TRUNCATED_1"])
    test_operator_model(bit_xor, "sat", [bit_xor.__class__.__name__+"_DIFF", bit_xor.__class__.__name__+"_DIFF_TRUNCATED"])
    
     
def TEST_NOT(): 
    print("\n********************* operation: bitwiseNOT ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    bit_not = NOT(my_input, my_output, ID = 'NOT')
    bit_not.display()
    test_operator_implementation(bit_not, "python")
    test_operator_implementation(bit_not, "c")
    test_operator_model(bit_not, "milp", [bit_not.__class__.__name__+"_DIFF"])
    test_operator_model(bit_not, "sat", [bit_not.__class__.__name__+"_DIFF"])
    # regard bitwiseNOT as an S-box and compute its ddt 
    not_sbox = Sbox([var.Variable(1,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=1, output_bitsize=1, ID="not_sbox")
    not_sbox.table = [1,0]
    ddt = not_sbox.computeDDT()   
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_Sbox(): 
    ascon_sbox = ASCON_Sbox([var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)], ID="ascon_sbox")

    
    skinny4_sbox = Skinny_4bit_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="skinny4_sbox")
    
    
    skinny8_sbox = Skinny_8bit_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="skinny8_sbox")
    
    
    gift_sbox = GIFT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="gift_sbox")
    
    
    aes_sbox = AES_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="aes_sbox")
    
    
    twine_sbox = TWINE_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="twine_sbox")
    
    
    present_sbox = PRESENT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="present_sbox")
    
    
    knot_sbox = KNOT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="knot_sbox")
    

    for sbox in [knot_sbox]: # [ascon_sbox, gift_sbox, skinny4_sbox, skinny8_sbox, twine_sbox, present_sbox, knot_sbox, aes_sbox]:
        print(f"\n********************* operation: {sbox.ID} Sbox ********************* ")
        sbox.display()
        test_operator_implementation(sbox, "python")
        test_operator_implementation(sbox, "c")
        print("differential branch number: ", sbox.differential_branch_number())

        if sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox]:
            test_operator_model(sbox, "milp", model_v_list=[sbox.__class__.__name__+"_DIFF_PR", sbox.__class__.__name__+"_DIFF", sbox.__class__.__name__+"_DIFF_TRUNCATED"], mode=0)
            test_operator_model(sbox, "sat", model_v_list=[sbox.__class__.__name__+"_DIFF", sbox.__class__.__name__+"_DIFF_TRUNCATED"], mode=0)
            if sbox.ID != "gift_sbox": 
                test_operator_model(sbox, "sat", model_v_list=[sbox.__class__.__name__+"_DIFF_PR"], mode=0)
        
        if sbox in [skinny8_sbox]:
            test_operator_model(sbox, "milp", model_v_list=[sbox.__class__.__name__+"_DIFF_PR", sbox.__class__.__name__+"_DIFF_P", sbox.__class__.__name__+"_DIFF_TRUNCATED"], mode=0)
            test_operator_model(sbox, "sat", model_v_list=[sbox.__class__.__name__+"_DIFF_PR"], mode=0)

        if sbox in [aes_sbox]:
            test_operator_model(sbox, "milp", model_v_list=[sbox.__class__.__name__+"_DIFF_PR", sbox.__class__.__name__+"_DIFF", sbox.__class__.__name__+"_DIFF_TRUNCATED"], mode=0)
            test_operator_model(sbox, "milp", model_v_list=[sbox.__class__.__name__+"_DIFF_P"], mode=1)
            test_operator_model(sbox, "sat", model_v_list=[sbox.__class__.__name__+"_DIFF_PR"], mode=0)

        if sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox, skinny8_sbox, aes_sbox]:
            test_operator_model(sbox, "milp", model_v_list=[sbox.__class__.__name__+"_DIFF_TRUNCATED_BN"])
        

def TEST_N_XOR(): 
    print("\n********************* operation: N_XOR ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    n = 4
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(n+1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)]
    n_xor = N_XOR(my_input, my_output, ID = 'N_XOR')    
    n_xor.display()
    test_operator_implementation(n_xor, "python")
    test_operator_implementation(n_xor, "c")
    test_operator_model(n_xor, "milp", [n_xor.__class__.__name__+"_DIFF", n_xor.__class__.__name__+"_DIFF_1"])
    test_operator_model(n_xor, "sat", [n_xor.__class__.__name__+"_DIFF"])
    
   
def TEST_Matrix(): 
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

    print("\n********************* operation: Matrix ********************* ")
    # aes's matrix
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(4)], [var.Variable(8,ID="out"+str(i)) for i in range(4)]
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    matrix_aes = Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial=0x1b, ID = 'Matrix_AES')
    matrix_aes.display()
    test_operator_implementation(matrix_aes, "python")
    test_operator_implementation(matrix_aes, "c")
    test_operator_model(matrix_aes, "milp", [matrix_aes.__class__.__name__+"_DIFF", matrix_aes.__class__.__name__+"_DIFF_1", matrix_aes.__class__.__name__+"_DIFF_TRUNCATED", matrix_aes.__class__.__name__+"_DIFF_TRUNCATED_1"])
    test_operator_model(matrix_aes, "sat", [matrix_aes.__class__.__name__+"_DIFF"])

    # skinny64's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = Matrix("mat_skinny", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    matrix_skinny64.display()
    test_operator_implementation(matrix_skinny64, "python")
    test_operator_implementation(matrix_skinny64, "c")
    test_operator_model(matrix_skinny64, "milp", [matrix_aes.__class__.__name__+"_DIFF", matrix_aes.__class__.__name__+"_DIFF_1"])
    test_operator_model(matrix_skinny64, "sat", [matrix_aes.__class__.__name__+"_DIFF"])
    
    
    # future's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_future = [[8,9,1,8], [3,2,9,9], [2,3,8,9], [9,9,8,1]]
    matrix_future = Matrix("mat_future", my_input, my_output, mat = mat_future, polynomial=0x3, ID = 'Matrix')
    matrix_future.display()
    test_operator_implementation(matrix_future, "python")
    test_operator_implementation(matrix_future, "c")
    test_operator_model(matrix_future, "milp", [matrix_aes.__class__.__name__+"_DIFF", matrix_aes.__class__.__name__+"_DIFF_1"])
    test_operator_model(matrix_future, "sat", [matrix_aes.__class__.__name__+"_DIFF"])
    

def TEST_ConstantAdd(): 
    print("\n********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    cons_add = ConstantAdd(my_input, my_output, "xor", [[2]], ID = 'ConstantAddXor')
    cons_add.display()
    test_operator_implementation(cons_add, "python")
    test_operator_implementation(cons_add, "c")
    test_operator_model(cons_add, "milp", [cons_add.__class__.__name__+"_DIFF", cons_add.__class__.__name__+"_DIFF_TRUNCATED"])
    test_operator_model(cons_add, "sat", [cons_add.__class__.__name__+"_DIFF", cons_add.__class__.__name__+"_DIFF_TRUNCATED"])


def TEST_AESround(): 
    print("\n********************* operation: AESround ********************* ")
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    aesround = AESround(my_input, my_output, ID="AESround")
    aesround.display()
    test_operator_implementation(aesround, "python")
    test_operator_implementation(aesround, "c")
    

def TEST_ANDXOR(): 
    print("\n********************* operation: bitwiseANDXOR ********************* ")
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(3)], [var.Variable(1,ID="out"+str(i)) for i in range(1)]
    bit_andxor = ANDXOR(my_input, my_output, ID = 'ANDXOR')
    bit_andxor.display()
    test_operator_implementation(bit_andxor, "python")
    test_operator_implementation(bit_andxor, "c")
    test_operator_model(bit_andxor, "milp", [bit_andxor.__class__.__name__+"_DIFF", bit_andxor.__class__.__name__+"_DIFF_1", bit_andxor.__class__.__name__+"_DIFF_2", bit_andxor.__class__.__name__+"_DIFF_3"])


if __name__ == '__main__':
    TEST_Equal()
    TEST_Rot()
    TEST_Shift()
    TEST_Modadd()
    TEST_AND()
    TEST_OR()
    TEST_XOR()
    TEST_NOT()
    TEST_Sbox()
    TEST_N_XOR()
    TEST_Matrix()
    TEST_ConstantAdd()
    TEST_AESround()
    TEST_ANDXOR()