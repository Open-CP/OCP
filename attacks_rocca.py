import primitives as prim
import variables as var
import attacks as at
import time
import pandas as pd

    

# ********************* TEST OF Rocca_AD ********************* #   
def TEST_Rocca_AD(r):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(128+32*r)], [var.Variable(8,ID="out"+str(i)) for i in range(128+32*r)]
    my_cipher = prim.Rocca_AD_permutation(f"ROCCA_AD", my_input, my_output, nbr_rounds=r)
    return my_cipher


# to search for truncated differential for Rocca_AD, set model_version = "truncated_diff" for all operations 
def set_model_versions_truncated_diff(cipher):
    model_versions = {}
    # print("*************constrains in input*************")
    model_versions["Input_Cons"] = "truncated_diff"
    for cons in cipher.inputs_constraints:
        # print(cons.ID, cons.__class__.__name__)
        model_versions[f"{cons.ID}"] = "truncated_diff"
    # print("*************constrains in each round*************")
    for i in range(1,cipher.nbr_rounds+1):
        for s in cipher.states: # cipher.states = ["STATE", "KEY_STATE", "SUBKEYS"]
            for l in range(cipher.states[s].nbr_layers+1):                 
                for cons in cipher.states[s].constraints[i][l]: 
                    # print(cons.ID, cons.__class__.__name__)
                    model_versions[f"{cons.ID}"] = "truncated_diff"
    return model_versions



def addForgeryConstr(r):
    """
    constraits:
    (1) input difference of the state is 0;
    (2) difference of the first data block is not 0;
    (3) output difference is not 0
    """
    add_cons = []
    for i in range(128):
        add_cons += [f"in{i} = 0"]
    add_cons += [" + ".join([f"in{i}" for i in range(128, 128+32)]) + " >= 1"]
    add_cons += [" + ".join([f"v_{r}_4_{i}" for i in range(128)]) + " >= 1"]
    return add_cons



def diff_Rocca_AD():
    data = {'Rounds': [], 'Result':[], 'Time(s)': []}
    for r in range(2, 20):
        cipher = TEST_Rocca_AD(r)
        strat_time = time.time()
        model_versions = set_model_versions_truncated_diff(cipher)
        add_cons = addForgeryConstr(r)
        obj = at.singlekey_differential_path_search_milp(cipher, r, model_versions=model_versions, add_cons=add_cons)
        end_time = time.time()
        data['Rounds'].append(r)
        data['Result'].append(obj)
        data['Time(s)'].append("{:.2f}".format(end_time - strat_time))
        df = pd.DataFrame(data)
        latex_code = df.to_latex(index=False, header=True, caption=f'Experimental results for {cipher.name} by solving MILP models')
        print(latex_code)


if __name__ == '__main__':
    diff_Rocca_AD()






