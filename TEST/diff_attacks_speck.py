import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks
import OCP
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files'))


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
    result["w"].pop(0)
    print_in_latex(result, os.path.join(base_path, f'speck_diff_matsui_sat_result.txt'))


def print_in_latex(result, filename=""):
    if filename:
        print(f"Generating LaTeX table from: {filename}")
    columns = list(result.keys())
    num_rows = max(len(result[col]) for col in columns)
    # Print header
    print(" & ".join(columns) + r" \\")
    print(r"\hline")
    # Print rows
    for i in range(num_rows):
        row = []
        for col in columns:
            if i < len(result[col]):
                row.append(str(result[col][i]))
            else:
                row.append(" ")
        print(" & ".join(row) + r" \\")
    

        
if __name__ == '__main__':
    traditional_method("milp", 1, 8)
    traditional_method("sat", 1, 11)
    matsui_method_milp(1,8)
    matsui_method_sat(1,22)
    






