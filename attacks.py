import os, os.path
import tool


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
    

def singlekey_differential_path_search_milp(primitive, nbr_rounds, model_type, model_versions={}, add_cons=[]):
    filename = f"files/{nbr_rounds}_round_{primitive.name}_singlekey_differential_path_search_{model_type}.lp"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    bin_vars = []
    in_vars = []
    obj = ""
    content = "Minimize\nobj\nSubject To\n"
    # constraint that the input is non-zero
    if "IN" in primitive.inputs: # for permutation
        content += ' + '.join(f"{primitive.inputs['IN'][i].ID}_{j}" for i in range(len(primitive.inputs['IN'])) for j in range(primitive.inputs['IN'][i].bitsize)) + ' >= 1\n'
    elif "plaintext" in primitive.inputs: # for block cipher
        content += ' + '.join(f"{primitive.inputs['plaintext'][i].ID}_{j}" for i in range(len(primitive.inputs['plaintext'])) for j in range(primitive.inputs['plaintext'][i].bitsize)) + ' >= 1\n'
    # constrains for linking the input and the first round 
    for cons in primitive.inputs_constraints:
        if "_K_" not in cons.ID:
            cons_gen = cons.generate_model("milp", unroll=True)
            for constraint in cons_gen:
                if "Binary" in constraint:
                    constraint_split = constraint.split('Binary\n')
                    content += constraint_split[0]
                    bin_vars += constraint_split[1].strip().split()
                elif "Integer" in constraint:
                    constraint_split = constraint.split('Integer\n')
                    content += constraint_split[0]
                    in_vars += constraint_split[1].strip().split()
                else: content += constraint + '\n'
    # constrains for each operation 
    for r in range(1,nbr_rounds+1):
        for l in range(primitive.states["STATE"].nbr_layers+1):                        
            for cons in primitive.states["STATE"].constraints[r][l]: 
                if cons.ID[0:3] == 'ARK':
                    var_in, var_out = [cons.get_var_ID('in', 0, unroll=True) + '_' + str(i) for i in range(cons.input_vars[0].bitsize)], [cons.get_var_ID('out', 0, unroll=True) + '_' + str(i) for i in range(cons.input_vars[0].bitsize)]
                    for vin, vout in zip(var_in, var_out):
                        content += f"{vin} - {vout} = 0\n"
                    bin_vars += var_in + var_out
                else:
                    if cons.ID in model_versions: cons_gen = cons.generate_model("milp", model_version = model_versions[cons.ID], unroll=True)
                    else: cons_gen = cons.generate_model("milp", unroll=True)
                    for constraint in cons_gen:
                        if "Binary" in constraint:
                            constraint_split = constraint.split('Binary\n')
                            content += constraint_split[0]
                            bin_vars += constraint_split[1].strip().split()
                        elif "Integer" in constraint:
                            constraint_split = constraint.split('Integer\n')
                            content += constraint_split[0]
                            in_vars += constraint_split[1].strip().split()
                        else: content += constraint + '\n'
                    if hasattr(cons, 'weight'): obj += ' + ' + cons.weight
    content += "\n".join(cons for cons in add_cons) + "\n" # add addtional constrains
    content += obj + ' - obj = 0\n'
    if len(bin_vars) > 0: content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if len(in_vars) > 0: content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    with open(filename, "w") as myfile:
        myfile.write(content)
    solve_milp(filename)



def singlekey_differential_path_search_sat(primitive, nbr_rounds, model_type, model_versions={}, obj=0, add_cons=[]):       
    filename = f"files/{nbr_rounds}_round_{primitive.name}_singlekey_differential_path_search_{model_type}.cnf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    obj_var = []
    # constraint that the input is non-zero
    if "IN" in primitive.inputs: # for permutation
        model_cons = [" ".join(f"{primitive.inputs['IN'][i].ID}_{j}" for i in range(len(primitive.inputs['IN'])) for j in range(primitive.inputs['IN'][i].bitsize))]
    elif "plaintext" in primitive.inputs: # for block cipher
        model_cons = [" ".join(f"{primitive.inputs['plaintext'][i].ID}_{j}" for i in range(len(primitive.inputs['plaintext'])) for j in range(primitive.inputs['plaintext'][i].bitsize))]
    # constrains for linking the input and the first round 
    for cons in primitive.inputs_constraints:
        if "_K_" not in cons.ID:
            cons_gen = cons.generate_model("sat", model_version = "diff_0", unroll=True)
            model_cons += cons_gen  
    # constrains for each operation 
    for r in range(1,nbr_rounds+1):
        for s in ["STATE"]:
            for l in range(primitive.states[s].nbr_layers+1):                        
                for cons in primitive.states[s].constraints[r][l]: 
                    if cons.ID[0:3] == 'ARK':
                        var_in, var_out = [cons.get_var_ID('in', 0, unroll=True) + '_' + str(i) for i in range(cons.input_vars[0].bitsize)], [cons.get_var_ID('out', 0, unroll=True) + '_' + str(i) for i in range(cons.input_vars[0].bitsize)]
                        for vin, vout in zip(var_in, var_out):
                            model_cons += [f"-{vin} {vout}", f"{vin} -{vout}"]
                    else:
                        if cons.ID in model_versions: cons_gen = cons.generate_model("sat", model_version = model_versions[cons.ID], unroll=True)
                        else: cons_gen = cons.generate_model("sat", unroll=True)
                        model_cons += cons_gen  
                        if hasattr(cons, 'weight'): obj_var += cons.weight
    model_cons += add_cons # add addtional constrains
    # modeling the constraint "weight greater or equal to the given obj using sequential encoding method 
    if obj == 0: obj_cons = [f'-{var}' for var in obj_var] 
    else:
        n = len(obj_var)
        dummy_var = [[f'obj_d_{i}_{j}' for j in range(obj)] for i in range(n - 1)]
        obj_cons = [f'-{obj_var[0]} {dummy_var[0][0]}']
        obj_cons += [f'-{dummy_var[0][j]}' for j in range(1, obj)]
        for i in range(1, n - 1):
            obj_cons += [f'-{obj_var[i]} {dummy_var[i][0]}']
            obj_cons += [f'-{dummy_var[i - 1][0]} {dummy_var[i][0]}']
            obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][j - 1]} {dummy_var[i][j]}' for j in range(1, obj)]
            obj_cons += [f'-{dummy_var[i - 1][j]} {dummy_var[i][j]}' for j in range(1, obj)]
            obj_cons += [f'-{obj_var[i]} -{dummy_var[i - 1][obj - 1]}']
        obj_cons += [f'-{obj_var[n - 1]} -{dummy_var[n - 2][obj - 1]}']
    model_cons += obj_cons
    # creating numerical CNF
    num_var, variable_map, numerical_cnf = tool.create_numerical_cnf(model_cons)
    num_clause = len(model_cons)
    content = f"p cnf {num_var} {num_clause}\n"  
    for constraint in numerical_cnf:
        content += constraint + ' 0\n'
    with open(filename, "w") as myfile:
        myfile.write(content)
    return solve_SAT(filename)



# def relatedkey_differential_path_search_milp(primitive, num_rounds, model_type): # to do


# def relatedkey_differential_path_search_sat(primitive, num_rounds, model_type, obj=0): # to do



if __name__ == '__main__':
    r = 4
    cipher = tool.TEST_SPECK32_BLOCKCIPHER(r)
    # cipher = TEST_SPECK32_PERMUTATION(r)
    # cipher = TEST_SIMON32_PERMUTATION(r)
    # cipher = TEST_ASCON_PERMUTATION(r) # TO DO
    # cipher = TEST_SKINNY_PERMUTATION(r) # TO DO
    # cipher = TEST_AES_PERMUTATION(r) # TO DO
    # cipher = TEST_SKINNY64_192_BLOCKCIPHER(r) # TO DO
    # cipher = TEST_GIFT64_permutation(r) # TO DO
    

    # ****************************** TEST OF MILP MODELING ****************************** #
    # (1) The user can modify the modeling versions for specific operations
    # Example: XOR_1_4_1 is modeled using "diff_1"
    model_versions_milp = {"XOR_1_4_1": "diff_1"}

    # (2) The user can specify additional constraints
    # Example: Force the first input (in0_0) to be equal to 0
    add_cons_milp = ["in0_0 = 0"]

    # (3) The user can specify the weight of operations # TO DO
    

    # Call the single-key differential path search function for MILP
    singlekey_differential_path_search_milp(cipher, r, "milp", model_versions=model_versions_milp, add_cons=add_cons_milp)
    
    # ****************************** TEST OF SAT MODELING ****************************** #
    # (1) The user can modify the modeling versions for specific operations
    model_versions_sat = {}

    # (2) The user can specify additional constraints
    # Example: Force the first input (in0_0) to be non-zero
    add_cons_sat = ["-in0_0"]

    # (3) The user can specify the weight of operations # TO DO

    # (4) The user can specify the value of the objective function starting from obj
    # Example: obj = 0
    obj = 0
    flag = False
    while not flag:
        print("obj", obj)
        flag = singlekey_differential_path_search_sat(cipher, r, "milp", model_versions=model_versions_sat, obj=obj, add_cons=add_cons_sat)
        obj += 1
    result = obj-1