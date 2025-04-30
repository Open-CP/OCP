import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks
import OCP
script_dir = os.path.dirname(os.path.abspath(__file__)) 
base_path = os.path.join(script_dir, '..', 'files')
base_path = os.path.abspath(base_path)


def traditional_method(model_type="milp", R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [],'TIME': []}
    for r in result["Rounds"]:
        cipher = OCP.SPECK_PERMUTATION(r, version=32)
        time_start = time.time()
        sol, obj = attacks.diff_attacks(cipher, model_type=model_type)    
        result["w"].append(int(obj))
        result["TIME"].append(round(time.time() - time_start, 2))
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_traditional_{model_type}_result.txt'))
    

def nonlinear_modadd_method_milp(R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [],'TIME_ModAdd_DIFF_1': [], 'TIME_ModAdd_DIFF_2': [], 'TIME_ModAdd_DIFF_3': []}
    for r in result["Rounds"]:
        for model_version in ["ModAdd_DIFF_1", "ModAdd_DIFF_2", "ModAdd_DIFF_3"]:
            cipher = OCP.SPECK_PERMUTATION(r, version=32)
            attacks.set_model_versions(cipher, model_version, operator_name="ModAdd")
            time_start = time.time()
            sol, obj = attacks.diff_attacks(cipher, model_type="milp")
            result["TIME_"+model_version].append(round(time.time() - time_start, 2))
        result["w"].append(int(obj))
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_nonlinear_modadd_milp_result.txt'))


def matsui_method_milp(R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [],'TIME': []}
    for r in result["Rounds"]:
        cipher = OCP.SPECK_PERMUTATION(r, version=32)
        time_start = time.time()
        add_cons = attacks.gen_matsui_constraints_milp(cipher, r, result["w"], cons_type="all")
        sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_cons)  
        result["w"].append(int(obj))
        result["TIME"].append(round(time.time() - time_start, 2))
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_matsui_milp_result.txt'))


def matsui_method_sat(R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [0],'TIME': []}
    GroupConstraintChoice = 1
    GroupNumForChoice1 = 1
    obj_sat = 0
    for r in result["Rounds"]:
        cipher = OCP.SPECK_PERMUTATION(r, version=32)
        flag = False
        time_start = time.time()
        while (flag == False):
            print("current obj_sat", obj_sat)
            flag = attacks.diff_attacks_matsui_sat(cipher, r, obj_sat, GroupConstraintChoice, GroupNumForChoice1, result["w"],solver="Cadical103")
            obj_sat += 1
        result["w"].append(obj_sat-1)
        result["TIME"].append(round(time.time() - time_start, 2))  
        obj_sat -= 1
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_matsui_sat_result.txt'))


def add_window_weight_cons(cipher, na, method=1, type=1):
    add_cons = []
    for i in range(1, cipher.states["STATE"].nbr_rounds+1):
        var_p, var_in1, var_in2, var_out = [], [], [], []
        for cons in cipher.states["STATE"].constraints[i][1]:
            if "ModAdd" in cons.__class__.__name__:
                cons.generate_model("milp")
                var_p += [v.strip() for v in cons.weight[0].split('+')]
                var_in1 += [cons.get_var_ID('in', 0, unroll=True) + '_' + str(j) for j in range(1, cons.input_vars[0].bitsize)]
                var_in2 += [cons.get_var_ID('in', 1, unroll=True) + '_' + str(j) for j in range(1, cons.input_vars[0].bitsize)]
                var_out += [cons.get_var_ID('out', 0, unroll=True) + '_' + str(j) for j in range(1, cons.input_vars[0].bitsize)]
        
        if method == 1:
            for j in range(len(var_p)-na+1):
                var_d = f"ADD_dummy_{na}_{i}_{j}"
                add_cons += [" + ".join([f"{var_p[l]}" for l in range(j, j+na)]) + f" - {var_d} >= 0"]
                add_cons += [f"{var_d} - {var_p[l]} >= 0" for l in range(j, j+na)]
                add_cons += [f"{var_in1[j]} - {var_in1[j+l]} - 100 {var_d} <= 0" for l in range(1, na)]
                add_cons += [f"{var_in1[j+l]} - {var_in1[j]} - 100 {var_d} <= 0" for l in range(1, na)]
                add_cons += [f"{var_in1[j]} - {var_in2[j+l]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_in2[j+l]} - {var_in1[j]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_in1[j]} - {var_out[j+l]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_out[j+l]} - {var_in1[j]} - 100 {var_d} <= 0" for l in range(na)]
            if type == 2:       
                add_cons += [" + ".join([f"ADD_dummy_{na}_{3}_{j}" for j in range(len(var_p)-na+1)]) + " = 1"]
        
        elif method == 2:
            for j in range(na-1,len(var_p),na):
                print("i, j", i, j)
                var_d = f"ADD_dummy_{na}_{i}_{j}"
                add_cons += [" + ".join([f"{var_p[l]}" for l in range(j-na+1, j+1)]) + f" - {var_d} >= 0"]
                add_cons += [f"{var_d} - {var_p[l]} >= 0" for l in range(j-na+1, j+1)]
                add_cons += [f"{var_in1[j-l]} - {var_in1[j]} - 100 {var_d} <= 0" for l in range(1, na)]
                add_cons += [f"{var_in1[j]} - {var_in1[j-l]} - 100 {var_d} <= 0" for l in range(1, na)]
                add_cons += [f"{var_in1[j-l]} - {var_in2[j]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_in2[j]} - {var_in1[j-l]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_in1[j-l]} - {var_out[j]} - 100 {var_d} <= 0" for l in range(na)]
                add_cons += [f"{var_out[j]} - {var_in1[j-l]} - 100 {var_d} <= 0" for l in range(na)]
            if type == 2:       
                add_cons += [" + ".join([f"ADD_dummy_{na}_{3}_{j}" for j in range(len(var_p)-na+1)]) + " = 1"]
        
    return add_cons


def window_weight_method(method=1):
    R_start, R_end = 1, 7
    result = {'Rounds': [i for i in range(R_start,R_end+1)]}
    version = 32
    if method == 1:
        for na in range(2,int(version/2)):
            result[f"TIME_na_{na}"] = []
            result[f"w_{na}"] = []
            for r in result["Rounds"]:        
                cipher = OCP.SPECK_PERMUTATION(r, version=version)
                add_constraints = add_window_weight_cons(cipher, na, method=1, type=1) 
                time_start = time.time()
                sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_constraints)
                result[f"w_{na}"].append(int(obj))
                result[f"TIME_na_{na}"].append(round(time.time() - time_start, 2))            
                print("method1: ", result)

    elif method == 2:
        for na in range(2,int(version/2)):
            result[f"TIME_na_{na}"] = []
            result[f"w_{na}"] = []
            for r in result["Rounds"]:        
                cipher = OCP.SPECK_PERMUTATION(r, version=version)
                add_constraints = add_window_weight_cons(cipher, na, method=2, type=1) 
                time_start = time.time()
                sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_constraints)
                result[f"w_{na}"].append(int(obj))
                result[f"TIME_na_{na}"].append(round(time.time() - time_start, 2))  
                print("method2: ", result)          
        
    elif method == 3:
        for r in result["Rounds"]:  
            result[f"TIME"] = []
            result[f"w"] = [] 
            add_constraints = []
            cipher = OCP.SPECK_PERMUTATION(r, version=version)
            for na in range(2,int(version/2)):
                add_constraints += add_window_weight_cons(cipher, na, method=2, type=1) 
            time_start = time.time()
            sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_constraints)
            result["w"].append(int(obj))
            result["TIME"].append(round(time.time() - time_start, 2))            
            print("method3: ", result)    
    
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_window_method{method}_milp_result.txt'))


def print_in_latex(result, filename=""):
    columns = list(result.keys())
    num_rows = len(next(iter(result.values())))
    with open(filename, "w") as fw:
        fw.write(str(result)) 
    for i in range(num_rows):
        row = [str(result[col][i]) for col in columns]
        print(" & ".join(row) + r" \\")
        with open(filename, "w") as fw:
            fw.write(" & ".join(row) + r" \\") 
    

        
if __name__ == '__main__':
    traditional_method("milp", 1, 6)
    traditional_method("sat", 1, 6)
    nonlinear_modadd_method_milp(1, 3)
    matsui_method_milp(1,6)
    matsui_method_sat(1,8)
    window_weight_method(method=3) # method=1, 2, 3


    # results for speck_32 on dragon2: 
    # (1) milp: traditional, nonlinear_modadd method, matsui
    # result = {'Rounds': [1, 2, 3, 4, 5, 6, 7, 8], 'w': [0, 1, 3, 5, 9, 13, 18, 24], 'TIME_t': [0.01, 0.07, 0.25, 1.42, 4.32, 26.02, 788.9, >10000], 'TIME_diff_1': [0.02, 0.07, 0.46, 1.56, 3.84, 41.69, 762.0], 'TIME_diff_2': [0.01, 0.04, 0.41, 0.75, 4.64, 65.37, 3763.0], 'TIME_diff_3': [0.01, 0.04, 0.5, 1.57, 3.91, 80.4, 3343.15], 'TIME_matsui': [0.02, 0.05, 0.62, 1.24, 23.24, 84.43, 2320.37]}
    # (2) milp: window_weight_method1:
    # result_type1 = {'Rounds': [2, 3, 4, 5, 6, 7, 8], 'w': [1, 3, 5, 9, 13, 18, 24], 'TIME_na_2': [0.07, 0.3, 1.23, 6.34, 23.65, 217.55, 1867.24], 'TIME_na_3': [0.08, 0.49, 2.23, 10.65, 30.12, 226.79, 10535.3], 'TIME_na_4': [0.09, 0.55, 2.9, 9.57, 40.85, 236.48, 3097.43], 'TIME_na_5': [0.11, 1.01, 1.85, 12.5, 18.67, 747.93, 2490.35], 'TIME_na_6': [0.1, 0.91, 1.97, 5.52, 40.33, 223.03, 41210.81], 'TIME_na_7': [0.09, 0.59, 1.85, 13.88, 54.02, 732.06, 3698.17], 'TIME_na_8': [0.09, 0.55, 3.12, 6.39, 48.56, 372.28, 3702.24], 'TIME_na_9': [0.1, 1.44, 1.97, 8.48, 31.02, 215.31, 2719.31], 'TIME_na_10': [0.09, 1.13, 2.69, 4.84, 37.44, 394.95, 37464.6], 'TIME_na_11': [0.11, 0.96, 2.67, 9.78, 55.54, 319.16, 49712.81], 'TIME_na_12': [0.11, 0.71, 1.46, 8.07, 31.12, 464.23, 3836.57], 'TIME_na_13': [0.08, 0.88, 4.53, 7.89, 32.08, 297.62, 20297.33], 'TIME_na_14': [0.09, 0.38, 1.92, 4.98, 23.16, 360.8, 28476.51], 'TIME_na_15': [0.06, 0.69, 0.98, 3.54, 24.36, 336.48, 2457.71]}
    # (3) milp: window_weight_method2:
    # result_type1 = {'Rounds': [2, 3, 4, 5, 6, 7, 8], 'w': [1, 3, 5, 9, 13, 18, 24], 'TIME_na_2': [0.07, 0.42, 2.28, 5.43, 24.29, 191.47, 1524.78], 'TIME_na_3': [0.06, 0.29, 2.1, 5.22, 24.43, 216.17, 6236.72], 'TIME_na_4': [0.06, 0.54, 0.98, 4.64, 20.24, 146.17, 18761.27], 'TIME_na_5': [0.06, 0.34, 2.0, 7.35, 26.78, 316.8, 18674.56], 'TIME_na_6': [0.06, 0.57, 1.52, 3.52, 19.46, 173.18, 4021.87], 'TIME_na_7': [0.06, 0.27, 1.76, 3.8, 21.88, 218.1, 2413.02], 'TIME_na_8': [0.07, 0.34, 1.13, 6.19, 17.87, 165.72, 2669.42], 'TIME_na_9': [0.06, 0.36, 2.11, 3.46, 28.6, 111.76, 3172.47], 'TIME_na_10': [0.06, 0.4, 1.01, 4.62, 24.68, 109.86, 1970.64], 'TIME_na_11': [0.05, 0.5, 1.52, 4.0, 19.27, 318.88, 20933.11], 'TIME_na_12': [0.08, 0.58, 1.51, 8.75, 22.08, 148.71, 16676.38], 'TIME_na_13': [0.05, 0.39, 1.72, 4.99, 25.38, 168.4, 19859.23], 'TIME_na_14': [0.06, 0.6, 1.1, 5.53, 35.88, 278.33, 13870.66], 'TIME_na_15': [0.05, 0.42, 1.25, 7.65, 22.06, 740.05, 1867.56]}
    # (4) milp: window_weight_method3:
    # result_type1 = {'Rounds': [2, 3, 4, 5, 6, 7, 8], 'w': [1, 3, 5, 9, 13, 18, 24], 'TIME\_type1': [0.24, 0.73, 4.29, 10.39, 65.93, 490.28, 4081.14]}

    # print_in_latex(result_type1)
    

    






