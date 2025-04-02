# Implement the method in paper "Speeding up MILP Aided Differential Characteristic Search with Matsui’s Strategy"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import attacks.attacks as attacks 
import OCP
script_dir = os.path.dirname(os.path.abspath(__file__)) 


def gen_cutoff_constraints_upper(cipher, r, best_obj=[]):
    add_cons = []
    for i in range(1, len(best_obj)+1):
        if best_obj[i-1] > 0:
            vars = []
            for j in range(i+1, r+1):
                for cons in cipher.states["STATE"].constraints[j][1]:
                    cons.generate_model(model_type='milp')
                    if hasattr(cons, 'weight'):
                        vars += cons.weight
            vars = [" + ".join(vars) + " - obj"]
            add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="SUM_LESS_EQUAL", vars=vars, value=-best_obj[i-1])
    return add_cons


def gen_cutoff_constraints_lower(cipher, r, best_obj):
    add_cons = []
    for i in range(1, len(best_obj)+1):
        if best_obj[i-1] > 0:
            vars = []
            for j in range(1, r-i+1):
                for cons in cipher.states["STATE"].constraints[j][1]:
                    cons.generate_model(model_type='milp')
                    if hasattr(cons, 'weight'):
                        vars += cons.weight
            vars = [" + ".join(vars) + " - obj"]
            add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="SUM_LESS_EQUAL", vars=vars, value=-best_obj[i-1])
    return add_cons


if __name__ == '__main__':
    R = 9
    BEST_OBJ = [0,1,3,5,9,13,18,24] # for speck-32 permutation
    for r in range(1, R+1):
        print("r", r)
        add_cons = []
        cipher = OCP.TEST_SPECK_PERMUTATION(r, version = 32) 
        add_cons += gen_cutoff_constraints_upper(cipher, r, BEST_OBJ)
        add_cons += gen_cutoff_constraints_lower(cipher, r, BEST_OBJ)
        sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_cons, show_mode=0)
        BEST_OBJ.append(int(obj))
        print(BEST_OBJ)
