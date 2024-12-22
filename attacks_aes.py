import primitives as prim
import variables as var
import attacks as at
import time
import pandas as pd
import tool
import operators as op

    

# ********************* TEST OF AES ********************* #   
def TEST_AES_PERMUTATION(r):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_AES_BLOCKCIPHER(r): 
    version = [128, 256] # block_size = version[0] = 128, key_size = version[1] = 128, 192, 256
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1] / version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_block_cipher(f"AES_{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


# The user can specify modeling versions by setting model_version. By defult, model_version = "diff_0".
def set_model_versions_truncated_diff(cipher):
    # Example 1: model truncated differential for AES, that is, model_version = "truncated_diff" for all operations of AES. 
    model_versions = {}
    print("*************constrains in input*************")
    model_versions["Input_Cons"] = "truncated_diff"
    for cons in cipher.inputs_constraints:
        print(cons.ID, cons.__class__.__name__)
        model_versions[f"{cons.ID}"] = "truncated_diff"
    print("*************constrains in each round*************")
    for i in range(1,cipher.nbr_rounds+1):
        for s in cipher.states: # cipher.states = ["STATE", "KEY_STATE", "SUBKEYS"]
            for l in range(cipher.states[s].nbr_layers+1):                 
                for cons in cipher.states[s].constraints[i][l]: 
                    print(cons.ID, cons.__class__.__name__)
                    model_versions[f"{cons.ID}"] = "truncated_diff"
    return model_versions



def single_key_diff_AES():
    data = {'Rounds': [], 'Result':[], 'Time(s)': []}
    for r in range(1, 15):
        # cipher = TEST_AES_PERMUTATION(r)
        cipher = TEST_AES_BLOCKCIPHER(r)
        strat_time = time.time()
        model_versions = set_model_versions_truncated_diff(cipher)
        obj = at.singlekey_differential_path_search_milp(cipher, r, model_versions=model_versions)
        end_time = time.time()
        data['Rounds'].append(r)
        data['Result'].append(obj)
        data['Time(s)'].append("{:.2f}".format(end_time - strat_time))
        df = pd.DataFrame(data)
        latex_code = df.to_latex(index=False, header=True, caption=f'Experimental results for {cipher.name} by solving MILP models')
        print(latex_code)


def addkeyScheduleExtraConstr(cipher):
    """
    Example of AES-128: 
    constraint: w6 = w5 ^ w2, w9 = w8 ^ w5, w10 = w9 ^ w6, ---> w10 = w8 ^ w2
    Constraints represented in OCP: vk_3_0_8 = vk_1_0_8 ^ vk_3_0_0
    """
    add_cons = []
    s = "KEY_STATE"
    nk = int(cipher.k_bitsize/8)
    for i in range(1, cipher.k_nbr_rounds):               
        for j in range(nk-8):
            if nk == 32 and (cipher.nbr_rounds%2) == 1 and i == (cipher.k_nbr_rounds-1) and j >= 8:
                pass
            else:
                a = cipher.states[s].vars[i][0][j+8]
                b = cipher.states[s].vars[i+2][0][j]
                c = cipher.states[s].vars[i+2][0][j+8]
                XOR = op.bitwiseXOR([a, b], [c], ID=f"addkeyScheduleExtraConstr_{i}_{j}")
                add_cons += XOR.generate_model("milp", model_version = "truncated_diff_1", unroll=True)
    return add_cons


def addMixColumnsExtraConstr(cipher):
    """
    constrains of key_schedule
    (vk_2_0_4, vk_2_0_5, vk_2_0_6, vk_2_0_7) ^ (vk_3_0_0, vk_3_0_1, vk_3_0_2, vk_3_0_3) = (vk_3_0_4, vk_3_0_5, vk_3_0_6, vk_3_0_7) 
    constrains of mixcolumn:
    mc(vs_2_2_4, vs_2_2_5, vs_2_2_6, vs_2_2_7) = vs_2_3_4, vs_2_3_5, vs_2_3_6, vs_2_3_7
        (vs_2_3_4, vs_2_3_5, vs_2_3_6, vs_2_3_7) ^ (vk_2_0_4, vk_2_0_5, vk_2_0_6, vk_2_0_7) =  (vs_2_4_4, vs_2_4_5, vs_2_4_6, vs_2_4_7)
    mc(vs_3_2_0, vs_3_2_1, vs_3_2_2, vs_3_2_3) = vs_3_3_0, vs_3_3_1, vs_3_3_2, vs_3_3_3
        (vs_3_3_0, vs_3_3_1, vs_3_3_2, vs_3_3_3) ^ (vk_3_3_0, vk_3_3_1, vk_3_3_2, vk_3_3_3) =  (vs_3_4_0, vs_3_4_1, vs_3_4_2, vs_3_4_3)
    mc(vs_3_2_4, vs_3_2_5, vs_3_2_6, vs_3_2_7) = vs_3_3_4, vs_3_3_5, vs_3_3_6, vs_3_3_7
        (vs_3_3_4, vs_3_3_5, vs_3_3_6, vs_3_3_7) ^ (vk_3_0_4, vk_3_0_5, vk_3_0_6, vk_3_0_7) = (vs_3_4_4, vs_3_4_5, vs_3_4_6, vs_3_4_7)
    ->
    (1) mc(vs_3_2_0^vs_2_2_4, vs_3_2_1^vs_2_2_5, vs_3_2_2^vs_2_2_6, vs_3_2_3^vs_2_2_7) = vs_3_3_0^vs_2_3_4, vs_3_3_1^vs_2_3_5, vs_3_3_2^vs_2_3_6, vs_3_3_3^vs_2_3_7
    =vk_2_5_4^vs_3_4_0^vs_2_4_4
    (2) mc(vs_3_2_0^vs_3_2_4, vs_3_2_1^vs_3_2_5, vs_3_2_2^vs_3_2_6, vs_3_2_3^vs_3_2_7) = vs_3_3_0^vs_3_3_4, vs_3_3_1^vs_3_3_5, vs_3_3_2^vs_3_3_6, vs_3_3_3^vs_3_3_7
    =vk_2_4_4^vs_3_4_0^vs_3_4_4
    (3) mc(vs_2_2_4^vs_3_2_4, vs_2_2_5^vs_3_2_5, vs_2_2_6^vs_3_2_6, vs_2_2_7^vs_3_2_7) = vs_2_3_4^vs_3_3_4, vs_2_3_5^vs_3_3_5, vs_2_3_6^vs_3_3_6, vs_2_3_7^vs_3_3_7
    =vk_2_4_0^vs_2_4_4^vs_3_4_4
    """
    add_cons = []
    nk = int(cipher.k_bitsize/32)
    for i in range(1, cipher.k_nbr_rounds+1):
        for j in range(1, nk):
            k0, k1, k2 = nk*i+j,  nk*(i+1)+(j-1),  nk*(i+1)+j 
            r0, r1, r2 = int(k0/4)+1, int(k1/4)+1, int(k2/4)+1
            c0, c1, c2 = k0%4, k1%4, k2%4
            if r0 >= 2 and r1 >= 2 and r2 >= 2 and r0 < cipher.nbr_rounds and r1 < cipher.nbr_rounds and r2 < cipher.nbr_rounds:
                x0 = [cipher.states["STATE"].vars[r0][2][4*c0+k] for k in range(4)]
                x1 = [cipher.states["STATE"].vars[r1][2][4*c1+k] for k in range(4)]
                x2 = [cipher.states["STATE"].vars[r2][2][4*c2+k] for k in range(4)]
                y0 = [cipher.states["STATE"].vars[r0][4][4*c0+k] for k in range(4)]
                y1 = [cipher.states["STATE"].vars[r1][4][4*c1+k] for k in range(4)]
                y2 = [cipher.states["STATE"].vars[r2][4][4*c2+k] for k in range(4)]
                k0 = [cipher.states["KEY_STATE"].vars[i+1][0][4*j+k] for k in range(4)]
                k1 = [cipher.states["KEY_STATE"].vars[i+2][0][4*j+k-4] for k in range(4)]
                k2 = [cipher.states["KEY_STATE"].vars[i+2][0][4*j+k] for k in range(4)]
                u0 = [var.Variable(8,ID=f"u0_{i}_{j}_" +str(k)) for k in range(4)]
                u1 = [var.Variable(8,ID=f"u1_{i}_{j}_" +str(k)) for k in range(4)]
                u2 = [var.Variable(8,ID=f"u2_{i}_{j}_" +str(k)) for k in range(4)]
                v0 = [var.Variable(8,ID=f"v0_{i}_{j}_" +str(k)) for k in range(4)]
                v1 = [var.Variable(8,ID=f"v1_{i}_{j}_" +str(k)) for k in range(4)]
                v2 = [var.Variable(8,ID=f"v2_{i}_{j}_" +str(k)) for k in range(4)]
                for k in range(4):
                    xor = op.bitwiseXOR([x0[k], x1[k]], [u0[k]], ID=f"addMixColumnsExtraConstr_xor0_{i}_{j}_{k}")
                    add_cons += xor.generate_model("milp", model_version = "truncated_diff_1", unroll=True)
                    nxor0 = op.N_XOR([y0[k], y1[k], k2[k]], [v0[k]], ID=f"addMixColumnsExtraConstr_nxor0_{i}_{j}_{k}")
                    add_cons += nxor0.generate_model("milp", model_version = "truncated_diff", unroll=True)
                matrix = op.Matrix("MC", u0, v0, mat=[[0 for k in range(4)] for l in range(4)], ID=F"addMixColumnsExtraConstr_matrix0_{i}_{j}")
                add_cons += matrix.generate_model("milp", model_version = "truncated_diff", unroll=True, branch_num=5)
                
                for k in range(4):
                    xor = op.bitwiseXOR([x0[k], x2[k]], [u1[k]], ID=f"addMixColumnsExtraConstr_xor1_{i}_{j}_{k}")
                    add_cons += xor.generate_model("milp", model_version = "truncated_diff_1", unroll=True)
                    nxor0 = op.N_XOR([y0[k], y2[k], k1[k]], [v1[k]], ID=f"addMixColumnsExtraConstr_nxor1_{i}_{j}_{k}")
                    add_cons += nxor0.generate_model("milp", model_version = "truncated_diff", unroll=True)
                matrix = op.Matrix("MC", u1, v1, mat=[[0 for k in range(4)] for l in range(4)], ID=F"addMixColumnsExtraConstr_matrix1_{i}_{j}")
                add_cons += matrix.generate_model("milp", model_version = "truncated_diff", unroll=True, branch_num=5)
                
                for k in range(4):
                    xor = op.bitwiseXOR([x1[k], x2[k]], [u2[k]], ID=f"addMixColumnsExtraConstr_xor2_{i}_{j}_{k}")
                    add_cons += xor.generate_model("milp", model_version = "truncated_diff_1", unroll=True)
                    nxor0 = op.N_XOR([y1[k], y2[k], k0[k]], [v2[k]], ID=f"addMixColumnsExtraConstr_nxor2_{i}_{j}_{k}")
                    add_cons += nxor0.generate_model("milp", model_version = "truncated_diff", unroll=True)
                matrix = op.Matrix("MC", u2, v2, mat=[[0 for k in range(4)] for l in range(4)], ID=F"addMixColumnsExtraConstr_matrix2_{i}_{j}")
                add_cons += matrix.generate_model("milp", model_version = "truncated_diff", unroll=True, branch_num=5)
    return add_cons


# implement Boura's MILP models for searching related-key differetial charactristics of AES, cited: "Related-Key Differential Analysis of the AES". 
def related_key_diff_AES():
    data = {'Rounds': [], 'Num':[], 'Time(s)': []}
    for r in range(3, 6):
        strat_time = time.time()
        cipher = TEST_AES_BLOCKCIPHER(r)
        model_versions = set_model_versions_truncated_diff(cipher)
        add_cons = []
        add_cons += addkeyScheduleExtraConstr(cipher)
        add_cons += addMixColumnsExtraConstr(cipher)
        obj = at.relatedkey_differential_path_search_milp(cipher, r, model_versions=model_versions, add_cons=add_cons)
        end_time = time.time()
        data['Rounds'].append(r)
        data['Num'].append(int(obj))
        data['Time(s)'].append("{:.2f}".format(end_time - strat_time))
    df = pd.DataFrame(data)
    latex_code = df.to_latex(index=False, header=True, caption=f'Experimental results for related-key differential cryptanalysis of {cipher.name} by solving MILP models')
    print(data)
    print(latex_code)


if __name__ == '__main__':

    single_key_diff_AES()
    related_key_diff_AES()






