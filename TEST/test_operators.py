import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import operators.operators as op
import variables.variables as var
import solving.solving as solving
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
        for model_v in model_v_list:
            filename = os.path.join(base_path, f'milp_{operator.ID}_{model_v}.lp')
            # Set model_version of the operator
            operator.model_version = model_v

            # Generate milp constraints
            if "sbox" in operator.ID: 
                milp_constraints = operator.generate_model(model_type='milp', mode= mode)
            else:
                milp_constraints = operator.generate_model(model_type='milp')
            print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
            
            # Define objective function if weight exists
            obj_fun = operator.weight if hasattr(operator, "weight") else []
            
            # Generate MILP model
            model = solving.gen_milp_model(constraints=milp_constraints, obj_fun=obj_fun, filename=filename) 
            
            # Solve MILP model for the optimal solution
            sol_list, obj_list = solving.solve_milp(filename, solving_goal="optimize", solver="SCIP")  
            print(f"Optimal solutions:\n{sol_list}\nObjective function:\n{obj_list}\n")
            
            # Solve MILP model for all solutions
            sol_list, obj_list = solving.solve_milp(filename, solving_goal="all_solutions")
            print(f"All solutions:\n{sol_list}\nObjective function:\n{obj_list}\nnumber of solutions: {len(sol_list)}")

    elif model_type == "sat":
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
            obj_var = operator.weight if hasattr(operator, "weight") else []
            
            # Generate SAT model
            model, variable_map = solving.gen_sat_model(constraints=sat_constraints, obj_var=obj_var, filename=filename) 
            print("variable_map in sat:\n", variable_map)
            
            # Solve SAT model for the optimal solution
            # support solvers: "Default", "Cadical103", "Cadical153", "Cadical195", "CryptoMinisat", "Gluecard3", "Gluecard4", "Glucose3", "Glucose4", "Lingeling", "MapleChrono", "MapleCM", "Maplesat", "Mergesat3", "Minicard", "Minisat22"
            sol_list = solving.solve_sat(filename, variable_map, solving_goal="optimize", solver="Default")  
            print(f"Optimal solutions:\n{sol_list}")
            
            # Solve SAT model for all solutions
            sol_list = solving.solve_sat(filename, variable_map, solving_goal="all_solutions")
            print(f"All solutions:\n{sol_list}")
        


# ********************* TEST OF OPERATORS MODELING IN MILP and SAT********************* #   
def TEST_Equal():  
    print("\n********************* operation: Equal ********************* ")
    # Create input and output variables for the Equal operation
    my_input, my_output = [var.Variable(2, ID=f"in{i}") for i in range(1)], [var.Variable(2, ID=f"out{i}") for i in range(1)]
    # Instantiate the Equal operation
    equal = op.Equal(my_input, my_output, ID='Equal')
    equal.display()
    test_operator_implementation(equal, "python")
    test_operator_implementation(equal, "c")
    test_operator_model(equal, "milp", model_v_list=["diff_0", "truncated_diff"])
    test_operator_model(equal, "sat", model_v_list=["diff_0", "truncated_diff"])
    

def TEST_Rot(): 
    print("\n********************* operation: Rot ********************* ")
    # Create input and output variables, instantiate the Rot operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    # rot = op.Rot(my_input, my_output, direction= 'l', amount=2, ID='Rot') # rotation by left
    rot = op.Rot(my_input, my_output, direction= 'r', amount=2, ID='Rot') # rotation by right
    rot.display()
    test_operator_implementation(rot, "python")
    test_operator_implementation(rot, "c")
    test_operator_model(rot, "milp", ["diff_0"])
    test_operator_model(rot, "sat", ["diff_0"])


def TEST_Shift(): 
    print("\n********************* operation: Shift ********************* ")
    # Create input and output variables, instantiate the Shift operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    shift = op.Shift(my_input, my_output, direction='l', amount=1, ID='Shift') # shift by left
    # shift = op.Shift(my_input, my_output, direction='r', amount=1, ID='Shift') # shift by right
    shift.display()
    test_operator_implementation(shift, "python")
    test_operator_implementation(shift, "c")
    test_operator_model(shift, "milp", ["diff_0"])
    test_operator_model(shift, "sat", ["diff_0"])
    

def TEST_Modadd(): 
    print("\n********************* operation: ModAdd ********************* ")
    # Create input and output variables, instantiate the ModAdd operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    mod_add = op.ModAdd(my_input, my_output, ID = 'ModAdd')
    mod_add.display()
    test_operator_implementation(mod_add, "python")
    test_operator_implementation(mod_add, "c")
    test_operator_model(mod_add, "milp", ["diff_0"])
    test_operator_model(mod_add, "sat", ["diff_0"])
    
    
def TEST_bitwiseAND(): 
    print("\n********************* operation: bitwiseAND ********************* ")
    # Create input and output variables, instantiate the bitwiseAND operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_and = op.bitwiseAND(my_input, my_output, ID = 'AND')
    bit_and.display()
    test_operator_implementation(bit_and, "python")
    test_operator_implementation(bit_and, "c")
    test_operator_model(bit_and, "milp", ["diff_0"])
    test_operator_model(bit_and, "sat", ["diff_0"])
    # regard bit-wiseAND as an S-box and compute its ddt 
    and_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="and_sbox")
    and_sbox.table = [0,0,0,1]
    ddt = and_sbox.computeDDT()
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_bitwiseOR():   
    print("\n********************* operation: bitwiseOR ********************* ")
    # Create input and output variables, instantiate the bitwiseOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_or = op.bitwiseOR(my_input, my_output, ID = 'OR')
    bit_or.display()
    test_operator_implementation(bit_or, "python")
    test_operator_implementation(bit_or, "c")
    test_operator_model(bit_or, "milp", ["diff_0"])
    test_operator_model(bit_or, "sat", ["diff_0"])
    # regard bit-wiseOR as an S-box and compute its ddt 
    or_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="or_sbox")
    or_sbox.table = [0,1,1,1]
    ddt = or_sbox.computeDDT()    
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_bitwiseXOR():  
    print("\n********************* operation: bitwiseXOR ********************* ")
    # Create input and output variables, instantiate the bitwiseXOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR')
    bit_xor.display()    
    test_operator_implementation(bit_xor, "python")
    test_operator_implementation(bit_xor, "c")
    test_operator_model(bit_xor, "milp", ["diff_0", "diff_1", "diff_2", "truncated_diff", "truncated_diff_1"])
    test_operator_model(bit_xor, "sat", ["diff_0", "truncated_diff"])
    
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(2)], [var.Variable(4,ID="out"+str(i)) for i in range(1)]
    bit_xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR', mat=[[1,None],[2,None],[3,None],[0,1]])
    bit_xor.display()    
    test_operator_implementation(bit_xor, "python")
    test_operator_implementation(bit_xor, "c")
    test_operator_model(bit_xor, "milp", ["diff_0", "diff_1", "diff_2", "truncated_diff", "truncated_diff_1"])
    test_operator_model(bit_xor, "sat", ["diff_0", "truncated_diff"])
    
     
def TEST_bitwiseNOT(): 
    print("\n********************* operation: bitwiseNOT ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    bit_not = op.bitwiseNOT(my_input, my_output, ID = 'NOT')
    bit_not.display()
    test_operator_implementation(bit_not, "python")
    test_operator_implementation(bit_not, "c")
    test_operator_model(bit_not, "milp", ["diff_0"])
    test_operator_model(bit_not, "sat", ["diff_0"])
    # regard bitwiseNOT as an S-box and compute its ddt 
    not_sbox = op.Sbox([var.Variable(1,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=1, output_bitsize=1, ID="not_sbox")
    not_sbox.table = [1,0]
    ddt = not_sbox.computeDDT()   
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_Sbox(): 
    ascon_sbox = op.ASCON_Sbox([var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)], ID="ascon_sbox")

    
    skinny4_sbox = op.Skinny_4bit_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="skinny4_sbox")
    
    
    skinny8_sbox = op.Skinny_8bit_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="skinny8_sbox")
    
    
    gift_sbox = op.GIFT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="gift_sbox")
    
    
    aes_sbox = op.AES_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="aes_sbox")
    
    
    twine_sbox = op.TWINE_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="twine_sbox")
    
    
    present_sbox = op.PRESENT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="present_sbox")
    
    
    knot_sbox = op.KNOT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="knot_sbox")
    

    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, skinny8_sbox, twine_sbox, present_sbox, knot_sbox, aes_sbox]:
        print(f"\n********************* operation: {sbox.ID} Sbox ********************* ")
        sbox.display()
        test_operator_implementation(sbox, "python")
        test_operator_implementation(sbox, "c")
        print("differential branch number: ", sbox.differential_branch_number())

        if sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox]:
            test_operator_model(sbox, "milp", model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)
            if sbox.ID != "gift_sbox": 
                test_operator_model(sbox, "sat", model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)

        if sbox in [skinny8_sbox]:
            test_operator_model(sbox, "milp", model_v_list=["diff_0", "diff_p", "truncated_diff"], mode=0)
            test_operator_model(sbox, "sat", model_v_list=["diff_0"], mode=0)

        if sbox in [aes_sbox]:
            test_operator_model(sbox, "milp", model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)
            test_operator_model(sbox, "milp", model_v_list=["diff_P"], mode=1)
            test_operator_model(sbox, "sat", model_v_list=["diff_0"], mode=0)

        if sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox, skinny8_sbox, aes_sbox]:
            test_operator_model(sbox, "milp", model_v_list=["truncated_diff_1"])
        

def TEST_N_XOR(): 
    print("\n********************* operation: N_XOR ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    n = 4
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(n+1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)]
    n_xor = op.N_XOR(my_input, my_output, ID = 'N_XOR')    
    n_xor.display()
    test_operator_implementation(n_xor, "python")
    test_operator_implementation(n_xor, "c")
    test_operator_model(n_xor, "milp", ["diff_0", "diff_1"])
    test_operator_model(n_xor, "sat", ["diff_0"])
    
   
def TEST_Matrix(): 
    print("\n********************* operation: Matrix ********************* ")
    # aes's matrix
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(4)], [var.Variable(8,ID="out"+str(i)) for i in range(4)]
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    matrix_aes = op.Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial=0x1b, ID = 'Matrix_AES')
    matrix_aes.display()
    test_operator_implementation(matrix_aes, "python")
    test_operator_implementation(matrix_aes, "c")
    test_operator_model(matrix_aes, "milp", ["diff_0", "diff_1", "truncated_diff", "truncated_diff_1"])
    test_operator_model(matrix_aes, "sat", ["diff_0"])

    # skinny64's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = op.Matrix("mat_skinny", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    matrix_skinny64.display()
    test_operator_implementation(matrix_skinny64, "python")
    test_operator_implementation(matrix_skinny64, "c")
    test_operator_model(matrix_skinny64, "milp", ["diff_0", "diff_1"])
    test_operator_model(matrix_skinny64, "sat", ["diff_0"])
    
    
    # future's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_future = [[8,9,1,8], [3,2,9,9], [2,3,8,9], [9,9,8,1]]
    matrix_future = op.Matrix("mat_future", my_input, my_output, mat = mat_future, polynomial=0x3, ID = 'Matrix')
    matrix_future.display()
    test_operator_implementation(matrix_future, "python")
    test_operator_implementation(matrix_future, "c")
    test_operator_model(matrix_future, "milp", ["diff_0", "diff_1"])
    test_operator_model(matrix_future, "sat", ["diff_0"])
    

def TEST_ConstantAdd(): 
    print("\n********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    cons_add = op.ConstantAdd(my_input, my_output, "xor", [[2]], ID = 'ConstantAddXor')
    cons_add.display()
    test_operator_implementation(cons_add, "python")
    test_operator_implementation(cons_add, "c")
    test_operator_model(cons_add, "milp", ["diff_0", "truncated_diff"])
    test_operator_model(cons_add, "sat", ["diff_0", "truncated_diff"])


if __name__ == '__main__':
    TEST_Equal()
    TEST_Rot()
    TEST_Shift()
    TEST_Modadd()
    TEST_bitwiseAND()
    TEST_bitwiseOR()
    TEST_bitwiseXOR()
    TEST_bitwiseNOT()
    TEST_Sbox()
    TEST_N_XOR()
    TEST_Matrix()
    TEST_ConstantAdd()