import os, os.path
import tool
import operators as op


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
    

def singlekey_differential_path_search_milp(cipher, nbr_rounds, model_versions={}, add_cons=[]):
    filename = f"files/{nbr_rounds}_round_{cipher.name}_singlekey_differential_path_search_milp.lp"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    bin_vars = []
    in_vars = []
    obj = ""
    content = "Minimize\nobj\nSubject To\n"
    # add constraint that the input difference of the cipher is non-zero
    if "permutation" in cipher.__class__.__name__: # for permutation
        if "Input_Cons" in model_versions:
            if model_versions["Input_Cons"] == "truncated_diff": content += ' + '.join(f"{cipher.inputs['IN'][i].ID}" for i in range(len(cipher.inputs['IN']))) + ' >= 1\n'
        else: content += ' + '.join(f"{cipher.inputs['IN'][i].ID}_{j}" for i in range(len(cipher.inputs['IN'])) for j in range(cipher.inputs['IN'][i].bitsize)) + ' >= 1\n'
    elif "block_cipher" in cipher.__class__.__name__: # for block cipher
        if "Input_Cons" in model_versions:
            if model_versions["Input_Cons"] == "truncated_diff": content += ' + '.join(f"{cipher.inputs['plaintext'][i].ID}" for i in range(len(cipher.inputs['plaintext']))) + ' >= 1\n'
        else: content += ' + '.join(f"{cipher.inputs['plaintext'][i].ID}_{j}" for i in range(len(cipher.inputs['plaintext'])) for j in range(cipher.inputs['plaintext'][i].bitsize)) + ' >= 1\n'
    # constrains for linking the input and the first round 
    for cons in cipher.inputs_constraints:
        if "_K_" not in cons.ID: # ignoring the key of block ciphers
            model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0" 
            cons_gen = cons.generate_model("milp", model_version = model_v, unroll=True)
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
    for r in range(1,cipher.nbr_rounds+1):
        for l in range(cipher.states["STATE"].nbr_layers+1):                        
            for cons in cipher.states["STATE"].constraints[r][l]: 
                model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0"  
                if cons.ID[0:3] == 'ARK' and cons.__class__.__name__ == "bitwiseXOR": # regarding XOR as Equal
                    equal = op.Equal([cons.input_vars[0]], cons.output_vars, ID=cons.ID)
                    cons_gen = equal.generate_model("milp", model_version = model_v, unroll=True)
                else:
                    cons_gen = cons.generate_model("milp", model_version = model_v, unroll=True)
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
    for constraint in add_cons: # add addtional constrains
        if "Binary" in constraint:
            constraint_split = constraint.split('Binary\n')
            content += constraint_split[0]
            bin_vars += constraint_split[1].strip().split()
        elif "Integer" in constraint:
            constraint_split = constraint.split('Integer\n')
            content += constraint_split[0]
            in_vars += constraint_split[1].strip().split()
        else: content += constraint + '\n'
    content += obj + ' - obj = 0\n'
    if len(bin_vars) > 0: content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if len(in_vars) > 0: content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    with open(filename, "w") as myfile:
        myfile.write(content)
    obj = solve_milp(filename)
    return obj



def singlekey_differential_path_search_sat_obj(cipher, nbr_rounds, model_versions={}, add_cons=[], obj=0):       
    filename = f"files/{nbr_rounds}_round_{cipher.name}_singlekey_differential_path_search_sat.cnf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    obj_var = []
    # constraints of non-zero input
    if "permutation" in cipher.__class__.__name__: # for permutation
        if "Input_Cons" in model_versions:
            if model_versions["Input_Cons"] == "truncated_diff":
                model_cons = [" ".join(f"{cipher.inputs['IN'][i].ID}" for i in range(len(cipher.inputs['IN'])))]
        else: model_cons = [" ".join(f"{cipher.inputs['IN'][i].ID}_{j}" for i in range(len(cipher.inputs['IN'])) for j in range(cipher.inputs['IN'][i].bitsize))]
    elif "block_cipher" in cipher.__class__.__name__: # for block cipher
        if "Input_Cons" in model_versions:
            if model_versions["Input_Cons"] == "truncated_diff": 
                model_cons = [" ".join(f"{cipher.inputs['plaintext'][i].ID}" for i in range(len(cipher.inputs['plaintext'])))]
        else: model_cons = [" ".join(f"{cipher.inputs['plaintext'][i].ID}_{j}" for i in range(len(cipher.inputs['plaintext'])) for j in range(cipher.inputs['plaintext'][i].bitsize))]
    # constraints for linking the input and the first round 
    for cons in cipher.inputs_constraints:
        if "_K_" not in cons.ID: # ignoring the key of block ciphers
            model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0" 
            cons_gen = cons.generate_model("sat", model_version = model_v, unroll=True)
            model_cons += cons_gen  
    # constrains for each operation 
    for r in range(1,cipher.nbr_rounds+1):
        for s in ["STATE"]:
            for l in range(cipher.states[s].nbr_layers+1):                        
                for cons in cipher.states[s].constraints[r][l]: 
                    model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0"  
                    if cons.ID[0:3] == 'ARK' and cons.__class__.__name__ == "bitwiseXOR": # regarding XOR as Equal
                        equal = op.Equal([cons.input_vars[0]], cons.output_vars, ID=cons.ID)
                        cons_gen = equal.generate_model("sat", model_version = model_v, unroll=True)
                    else: cons_gen = cons.generate_model("sat", model_version = model_v, unroll=True)    
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


def singlekey_differential_path_search_sat(cipher, nbr_rounds, model_versions={}, add_cons=[], obj=0):
    print(model_versions)
    flag = False
    while not flag:
        print("obj", obj)
        flag = singlekey_differential_path_search_sat_obj(cipher, nbr_rounds, model_versions=model_versions, add_cons=add_cons, obj=obj)
        obj += 1
    return obj-1



def relatedkey_differential_path_search_milp(cipher, nbr_rounds, model_versions={}, add_cons=[]):
    if "block_cipher" not in cipher.__class__.__name__: raise Exception("only support relatedkey differential cryptanalysis for block_ciphers")
    filename = f"files/{nbr_rounds}_round_{cipher.name}_relatedkey_differential_path_search_milp.lp"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    bin_vars = []
    in_vars = []
    obj = ""
    content = "Minimize\nobj\nSubject To\n"
    # add constraint that the input difference of the cipher is non-zero
    if "block_cipher" in cipher.__class__.__name__: # for block cipher
        if "Input_Cons" in model_versions: 
            if model_versions["Input_Cons"] == "truncated_diff": content += ' + '.join(f"{cipher.inputs['plaintext'][i].ID}" for i in range(len(cipher.inputs['plaintext']))) + ' + ' + ' + '.join(f"{cipher.inputs['key'][i].ID}" for i in range(len(cipher.inputs['key']))) + ' >= 1\n'
        else: content += ' + '.join(f"{cipher.inputs['plaintext'][i].ID}_{j}" for i in range(len(cipher.inputs['plaintext'])) for j in range(cipher.inputs['plaintext'][i].bitsize)) + ' + ' + ' + '.join(f"{cipher.inputs['key'][i].ID}_{j}" for i in range(len(cipher.inputs['key'])) for j in range(cipher.inputs['key'][i].bitsize)) + ' >= 1\n'
    # constrains for linking the input and the first round 
    for cons in cipher.inputs_constraints:
        model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0" 
        cons_gen = cons.generate_model("milp", model_version = model_v, unroll=True)
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
    for s in cipher.states: 
        for r in range(1,cipher.nbr_rounds+1):
            for l in range(cipher.states[s].nbr_layers+1):                        
                for cons in cipher.states[s].constraints[r][l]: 
                    model_v = model_versions[cons.ID] if cons.ID in model_versions else "diff_0"  
                    cons_gen = cons.generate_model("milp", model_version = model_v, unroll=True)
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
    for constraint in add_cons: # add addtional constrains
        if "Binary" in constraint:
                constraint_split = constraint.split('Binary\n')
                content += constraint_split[0]
                bin_vars += constraint_split[1].strip().split()
        elif "Integer" in constraint:
            constraint_split = constraint.split('Integer\n')
            content += constraint_split[0]
            in_vars += constraint_split[1].strip().split()
        else: content += constraint + '\n'
    content += obj + ' - obj = 0\n'
    if len(bin_vars) > 0: content += "Binary\n" + " ".join(set(bin_vars)) + "\n"
    if len(in_vars) > 0: content += "Integer\n" + " ".join(set(in_vars)) + "\nEnd\n"
    with open(filename, "w") as myfile:
        myfile.write(content)
    obj = solve_milp(filename)
    return obj


# def relatedkey_differential_path_search_sat(primitive, num_rounds, model_type, obj=0): # TO DO

