import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import operators.operators as op
import variables.variables as var
import solving.solving as solving


def test_operator_model(model_type, operator, model_v_list=["diff_0"], mode=0):
    if model_type == "python": 
        # Generate and print Python model code
        python_code = operator.generate_model(model_type='python', unroll=True)
        print("Python code: \n", "\n".join(python_code))    
        
    elif model_type == "c":
        # Generate and print C model code
        c_code = operator.generate_model(model_type='c', unroll=True)
        print("C code: \n", "\n".join(c_code))  
    
    # Generate and solve MILP models
    elif model_type == "milp":
        for model_v in model_v_list:
            filename = f'files/milp_{operator.ID}_{model_v}.lp' 
            # Set model_version of the operator
            operator.model_version = model_v

            # Generate milp constraints
            if "sbox" in operator.ID: 
                milp_constraints = operator.generate_model(model_type='milp', mode= mode, unroll=True)
            else:
                milp_constraints = operator.generate_model(model_type='milp', unroll=True)
            print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
            
            # Define objective function if weight exists
            obj_fun = operator.weight if hasattr(operator, "weight") else []
            
            # Generate MILP model
            model = solving.gen_milp_model(constraints=milp_constraints, obj_fun=obj_fun, filename=filename) 
            
            # Solve MILP model for the optimal solution
            sol_list, obj_list = solving.solve_milp(filename, solving_goal="optimize")  
            print(f"Optimal solutions:\n{sol_list}\nObjective function:\n{obj_list}\nnumber of solutions: {len(sol_list)}")
            
            # Solve MILP model for all solutions
            sol_list, obj_list = solving.solve_milp(filename, solving_goal="all_solutions")
            print(f"All solutions:\n{sol_list}\nObjective function:\n{obj_list}\nnumber of solutions: {len(sol_list)}")

    elif model_type == "sat":
        # Generate and solve SAT models
        for model_v in model_v_list:
            filename = f'files/sat_{operator.ID}_{model_v}.cnf'
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
            sol_list = solving.solve_sat(filename, solving_goal="optimize")  
            print(f"Optimal solutions:\n{sol_list}")
            
            # Solve SAT model for all solutions
            sol_list = solving.solve_sat(filename, solving_goal="all_solutions")
            print(f"All solutions:\n{sol_list}")
        


# ********************* TEST OF OPERATORS MODELING IN MILP and SAT********************* #   
def TEST_Equal_MILP_SAT():  
    print("\n********************* operation: Equal ********************* ")
    # Create input and output variables for the Equal operation
    my_input, my_output = [var.Variable(2, ID=f"in{i}") for i in range(1)], [var.Variable(2, ID=f"out{i}") for i in range(1)]
    # Instantiate the Equal operation
    equal = op.Equal(my_input, my_output, ID='Equal')
    equal.display()
    test_operator_model("python", equal)
    test_operator_model("c", equal)
    test_operator_model("milp", equal, model_v_list=["diff_0", "truncated_diff"])
    test_operator_model("sat", equal, model_v_list=["diff_0", "truncated_diff"])
    

def TEST_Rot_MILP_SAT(): 
    print("\n********************* operation: Rot ********************* ")
    # Create input and output variables, instantiate the Rot operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    # rot = op.Rot(my_input, my_output, direction= 'l', amount=2, ID='Rot') # rotation by left
    rot = op.Rot(my_input, my_output, direction= 'r', amount=2, ID='Rot') # rotation by right
    rot.display()
    test_operator_model("python", rot)
    test_operator_model("c", rot)
    test_operator_model("milp", rot)
    test_operator_model("sat", rot)


def TEST_Shift_MILP_SAT(): 
    print("\n********************* operation: Shift ********************* ")
    # Create input and output variables, instantiate the Shift operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    shift = op.Shift(my_input, my_output, direction='l', amount=1, ID='Shift') # shift by left
    # shift = op.Shift(my_input, my_output, direction='r', amount=1, ID='Shift') # shift by right
    shift.display()
    test_operator_model("python", shift)
    test_operator_model("c", shift)
    test_operator_model("milp", shift)
    test_operator_model("sat", shift)
    

def TEST_Modadd_MILP_SAT(): 
    print("\n********************* operation: ModAdd ********************* ")
    # Create input and output variables, instantiate the ModAdd operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    mod_add = op.ModAdd(my_input, my_output, ID = 'ModAdd')
    mod_add.display()
    test_operator_model("python", mod_add)
    test_operator_model("c", mod_add)
    test_operator_model("milp", mod_add)
    test_operator_model("sat", mod_add)
    
    
def TEST_bitwiseAND_MILP_SAT(): 
    print("\n********************* operation: bitwiseAND ********************* ")
    # Create input and output variables, instantiate the bitwiseAND operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_and = op.bitwiseAND(my_input, my_output, ID = 'AND')
    bit_and.display()
    test_operator_model("python", bit_and)
    test_operator_model("c", bit_and)
    test_operator_model("milp", bit_and)
    test_operator_model("sat", bit_and)
    # regard bit-wiseAND as an S-box and compute its ddt 
    and_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="and_sbox")
    and_sbox.table = [0,0,0,1]
    ddt = and_sbox.computeDDT()
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_bitwiseOR_MILP_SAT():   
    print("\n********************* operation: bitwiseOR ********************* ")
    # Create input and output variables, instantiate the bitwiseOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_or = op.bitwiseOR(my_input, my_output, ID = 'OR')
    bit_or.display()
    test_operator_model("python", bit_or)
    test_operator_model("c", bit_or)
    test_operator_model("milp", bit_or)
    test_operator_model("sat", bit_or)
    # regard bit-wiseOR as an S-box and compute its ddt 
    or_sbox = op.Sbox([var.Variable(2,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=2, output_bitsize=1, ID="or_sbox")
    or_sbox.table = [0,1,1,1]
    ddt = or_sbox.computeDDT()    
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
def TEST_bitwiseXOR_MILP_SAT():  
    print("\n********************* operation: bitwiseXOR ********************* ")
    # Create input and output variables, instantiate the bitwiseXOR operation
    my_input, my_output = [var.Variable(2,ID="in"+str(i)) for i in range(2)], [var.Variable(2,ID="out"+str(i)) for i in range(1)]
    bit_xor = op.bitwiseXOR(my_input, my_output, ID = 'XOR')
    bit_xor.display()
    test_operator_model("python", bit_xor)
    test_operator_model("c", bit_xor)
    test_operator_model("milp", bit_xor, model_v_list=["diff_0", "diff_1", "diff_2", "truncated_diff", "truncated_diff_1"])
    test_operator_model("sat", bit_xor)      
   
     
def TEST_bitwiseNOT_MILP_SAT(): 
    print("\n********************* operation: bitwiseNOT ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    bit_not = op.bitwiseNOT(my_input, my_output, ID = 'NOT')
    bit_not.display()
    test_operator_model("python", bit_not)
    test_operator_model("c", bit_not)
    test_operator_model("milp", bit_not)
    test_operator_model("sat", bit_not) 
    # regard bitwiseNOT as an S-box and compute its ddt 
    not_sbox = op.Sbox([var.Variable(1,ID="in"+str(i)) for i in range(1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)], input_bitsize=1, output_bitsize=1, ID="not_sbox")
    not_sbox.table = [1,0]
    ddt = not_sbox.computeDDT()   
    print("ddt and number of non-zeros", ddt, len([ddt[i][j] for i in range(len(ddt)) for j in range(len(ddt[i])) if ddt[i][j] != 0]))
    
    
    
def TEST_Sbox_MILP_SAT(): 
    print("\n********************* operation: Sbox ********************* ")
    ascon_sbox = op.ASCON_Sbox([var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)], ID="ascon_sbox")

    
    skinny4_sbox = op.Skinny_4bit_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="skinny4_sbox")
    
    
    skinny8_sbox = op.Skinny_8bit_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="skinny8_sbox")
    
    
    gift_sbox = op.GIFT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="gift_sbox")
    
    
    aes_sbox = op.AES_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="aes_sbox")
    
    
    twine_sbox = op.TWINE_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="twine_sbox")
    
    
    present_sbox = op.PRESENT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="present_sbox")
    
    
    knot_sbox = op.KNOT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="knot_sbox")
    

    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, skinny8_sbox, twine_sbox, present_sbox, knot_sbox, aes_sbox]:
        sbox.display()
        test_operator_model("python", sbox)
        test_operator_model("c", sbox)
        print("differential branch number of knot_sbox: ", sbox.differential_branch_number())
    

    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox]:
        test_operator_model("milp", sbox, model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)
        if sbox.ID != "gift_sbox": 
            test_operator_model("sat", sbox, model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)


    for sbox in [skinny8_sbox]:
        test_operator_model("milp", sbox, model_v_list=["diff_0", "diff_p", "truncated_diff"], mode=0)
        test_operator_model("sat", sbox, model_v_list=["diff_0"], mode=0)


    for sbox in [aes_sbox]:
        test_operator_model("milp", sbox, model_v_list=["diff_0", "diff_1", "truncated_diff"], mode=0)
        test_operator_model("milp", sbox, model_v_list=["diff_P"], mode=1)
        test_operator_model("sat", sbox, model_v_list=["diff_0"], mode=0)


    for sbox in [ascon_sbox, gift_sbox, skinny4_sbox, twine_sbox, present_sbox, knot_sbox, skinny8_sbox, aes_sbox]:
        test_operator_model("milp", sbox, model_v_list=["truncated_diff_1"])
        


def TEST_N_XOR_MILP_SAT(): 
    print("\n********************* operation: N_XOR ********************* ")
    # Create input and output variables, instantiate the bitwiseNOT operation
    n = 4
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(n+1)], [var.Variable(1,ID="out"+str(i)) for i in range(1)]
    n_xor = op.N_XOR(my_input, my_output, ID = 'N_XOR')    
    n_xor.display()
    test_operator_model("python", n_xor)
    test_operator_model("c", n_xor)
    test_operator_model("milp", n_xor, model_v_list=["diff_0", "diff_1"])
    test_operator_model("sat", n_xor) 
    
   

def TEST_Matrix_MILP_SAT(): 
    print("\n********************* operation: Matrix ********************* ")
    # aes's matrix
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(4)], [var.Variable(8,ID="out"+str(i)) for i in range(4)]
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    matrix_aes = op.Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial=0x1b, ID = 'Matrix_AES')
    matrix_aes.display()
    test_operator_model("python", matrix_aes)
    test_operator_model("c", matrix_aes)
    test_operator_model("milp", matrix_aes, model_v_list=["diff_0", "diff_1", "truncated_diff", "truncated_diff_1"])
    test_operator_model("sat", matrix_aes) 

    # skinny64's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = op.Matrix("mat_skinny", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    matrix_skinny64.display()
    test_operator_model("python", matrix_skinny64)
    test_operator_model("c", matrix_skinny64)
    test_operator_model("milp", matrix_skinny64, model_v_list=["diff_0", "diff_1"])
    test_operator_model("sat", matrix_skinny64) 
    
    
    # future's matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_future = [[8,9,1,8], [3,2,9,9], [2,3,8,9], [9,9,8,1]]
    matrix_future = op.Matrix("mat_future", my_input, my_output, mat = mat_future, polynomial=0x3, ID = 'Matrix')
    matrix_future.display()
    test_operator_model("python", matrix_future)
    test_operator_model("c", matrix_future)
    test_operator_model("milp", matrix_future, model_v_list=["diff_0", "diff_1"])
    test_operator_model("sat", matrix_future) 
    

def TEST_ConstantAdd_MILP_SAT(): 
    print("\n********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)]
    cons_add = op.ConstantAdd(my_input, my_output, 2, "xor", ID = 'ConstantAddXor')
    cons_add.display()
    test_operator_model("python", cons_add)
    test_operator_model("c", cons_add)
    test_operator_model("milp", cons_add, model_v_list=["diff_0", "truncated_diff"])
    test_operator_model("sat", cons_add, model_v_list=["diff_0", "truncated_diff"]) 


if __name__ == '__main__':
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