import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks 
import OCP
script_dir = os.path.dirname(os.path.abspath(__file__)) 
base_path = os.path.join(script_dir, '..', 'files')
base_path = os.path.abspath(base_path)


def traditional_method(attacks_type="best_trail", model_type="milp", R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [],'TIME': []}
    for r in result["Rounds"]:
        cipher = OCP.PRESENT_PERMUTATION(r)
        if attacks_type == "min_as":
            attacks.set_model_versions(cipher, "PRESENT_Sbox_DIFF", operator_name="PRESENT_Sbox")
        time_start = time.time()
        sol, obj = attacks.diff_attacks(cipher, model_type=model_type)    
        result["w"].append(int(obj))
        result["TIME"].append(round(time.time() - time_start, 2))
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_traditional_{model_type}_result.txt'))
 

def diff_attacks_matsui_milp(attacks_type="best_trail",R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [],'TIME': []}
    for r in result["Rounds"]:
        cipher = OCP.PRESENT_PERMUTATION(r)
        if attacks_type == "min_as":
            attacks.set_model_versions(cipher, "PRESENT_Sbox_DIFF", operator_name="PRESENT_Sbox")
        time_start = time.time()
        add_cons = attacks.gen_matsui_constraints_milp(cipher, r, result["w"], cons_type="all")
        sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_cons,)  
        result["w"].append(int(obj))
        result["TIME"].append(round(time.time() - time_start, 2))
        print(result)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_matsui_milp_result.txt'))


def diff_attacks_matsui_sat(attacks_type="best_trail",R_start=1, R_end=1):
    result = {'Rounds': [i for i in range(R_start,R_end+1)], 'w': [0],'TIME': []}
    GroupConstraintChoice = 1
    GroupNumForChoice1 = 1
    obj_sat = 0
    for r in result["Rounds"]:
        cipher = OCP.PRESENT_PERMUTATION(r)
        if attacks_type == "min_as":
            attacks.set_model_versions(cipher, "PRESENT_Sbox_DIFF", operator_name="PRESENT_Sbox")
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
    result["w"].pop(0)
    print_in_latex(result, os.path.join(base_path, f'{cipher.name}_diff_matsui_sat_result.txt'))


def print_in_latex(result, filename=""):
    print(filename)
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
    traditional_method(attacks_type="min_as", model_type="milp", R_start=1, R_end=8)
    traditional_method(model_type="milp", R_start=1, R_end=8)
    traditional_method(attacks_type="min_as", model_type="sat", R_start=1, R_end=8)
    traditional_method(model_type="sat", R_start=1, R_end=8)
    diff_attacks_matsui_milp(attacks_type="min_as", R_start=1, R_end=9)
    diff_attacks_matsui_milp(R_start=1, R_end=9)
    diff_attacks_matsui_sat(attacks_type="min_as", R_start=1, R_end=31)
    diff_attacks_matsui_sat(R_start=1, R_end=31)