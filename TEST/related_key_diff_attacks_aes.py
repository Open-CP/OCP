import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import primitives.aes as aes
import variables.variables as var
import operators.operators as op
import attacks.attacks as attacks

# Implement the MILP model for related-key differential characteristics of AES.
# Reference: Christina Boura, Patrick Derbez, and Margot Funk. Related-Key Differential Analysis of the AES.

def TEST_AES_BLOCKCIPHER(r, version = [128, 128]): 
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1] / version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = aes.AES_block_cipher(f"AES{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def addkeyScheduleExtraConstr(cipher):
    """
    Generate additional constraints derived from the key schedule. Take AES-128 as an example, consider the relationships between the key schedule words:
    - w6 = w5 ^ w2, w9 = w8 ^ w5, w10 = w9 ^ w6, ---> w10 = w8 ^ w2
    - The constraint is encoded in OCP as: vk_3_0_8 = vk_1_0_8 ^ vk_3_0_0
    """
    add_cons = []
    for i in range(1, cipher.states["KEY_STATE"].nbr_rounds-1):               
        for j in range(cipher.states["KEY_STATE"].nbr_words-8):
            if cipher.states["KEY_STATE"].nbr_words == 24 and i == cipher.states["KEY_STATE"].nbr_rounds-2 and (j >= 16-8*(cipher.states["STATE"].nbr_rounds%3)):
                pass
            elif cipher.states["KEY_STATE"].nbr_words == 32 and ((j >= 8 and j <= 15) or (cipher.states["STATE"].nbr_rounds%2 == 1 and i == cipher.states["KEY_STATE"].nbr_rounds-2 and j >= 16)):
                pass
            else:
                a = cipher.states["KEY_STATE"].vars[i][0][j+8]
                b = cipher.states["KEY_STATE"].vars[i+2][0][j]
                c = cipher.states["KEY_STATE"].vars[i+2][0][j+8]
                XOR = op.bitwiseXOR([a, b], [c], ID=f"addkeyScheduleExtraConstr_{i}_{j}")
                XOR.model_version = "truncated_diff_1"
                add_cons += XOR.generate_model("milp")
    return add_cons


def addMixColumnsExtraConstr(cipher):
    """
    Generate additional constraints derived from the key schedule and encrption. Take AES-128 as an example, 
    - consider of key schedule: 
    (vk_2_0_4, vk_2_0_5, vk_2_0_6, vk_2_0_7) ^ (vk_3_0_0, vk_3_0_1, vk_3_0_2, vk_3_0_3) = (vk_3_0_4, vk_3_0_5, vk_3_0_6, vk_3_0_7) 
    - consider of mixcolumn and addkey:
    mc(vs_2_2_4, vs_2_2_5, vs_2_2_6, vs_2_2_7) ^ (vk_2_0_4, vk_2_0_5, vk_2_0_6, vk_2_0_7) = (vs_2_4_4, vs_2_4_5, vs_2_4_6, vs_2_4_7)
    mc(vs_3_2_0, vs_3_2_1, vs_3_2_2, vs_3_2_3) ^ (vk_3_0_0, vk_3_0_1, vk_3_0_2, vk_3_0_3) = (vs_3_4_0, vs_3_4_1, vs_3_4_2, vs_3_4_3)
    mc(vs_3_2_4, vs_3_2_5, vs_3_2_6, vs_3_2_7) ^ (vk_3_0_4, vk_3_0_5, vk_3_0_6, vk_3_0_7) = (vs_3_4_4, vs_3_4_5, vs_3_4_6, vs_3_4_7)
    --->
    (1) mc(vs_3_2_0^vs_2_2_4, vs_3_2_1^vs_2_2_5, vs_3_2_2^vs_2_2_6, vs_3_2_3^vs_2_2_7) = vk_3_0_4^vs_3_4_0^vs_2_4_4
    (2) mc(vs_3_2_0^vs_3_2_4, vs_3_2_1^vs_3_2_5, vs_3_2_2^vs_3_2_6, vs_3_2_3^vs_3_2_7) = vk_2_0_4^vs_3_4_0^vs_3_4_4
    (3) mc(vs_2_2_4^vs_3_2_4, vs_2_2_5^vs_3_2_5, vs_2_2_6^vs_3_2_6, vs_2_2_7^vs_3_2_7) = vk_3_0_0^vs_2_4_4^vs_3_4_4
    """
    add_cons = []
    nk = int(cipher.states["KEY_STATE"].nbr_words/4)
    for i in range(1, cipher.states["KEY_STATE"].nbr_rounds):
        for j in range(1, nk):
            nk0, nk1, nk2 = nk*(i-1)+j,  nk*i+(j-1),  nk*i+j # subkey words
            r0, r1, r2 = int(nk0/4)+1, int(nk1/4)+1, int(nk2/4)+1 # state round
            c0, c1, c2 = nk0%4, nk1%4, nk2%4
            if cipher.states["KEY_STATE"].nbr_words == 16 and (r0 >= 11 or r1 >= 11 or r2 >= 11):
                pass
            elif cipher.states["KEY_STATE"].nbr_words == 24 and (r0 >= 13 or r1 >= 13 or r2 >= 13):
                pass
            elif cipher.states["KEY_STATE"].nbr_words == 32 and (r0 >= 15 or r1 >= 15 or r2 >= 15 or j%4 == 0):
                pass
            elif r0 >= 2 and r1 >= 2 and r2 >= 2 and r0 <= cipher.states["STATE"].nbr_rounds and r1 <= cipher.states["STATE"].nbr_rounds and r2 <= cipher.states["STATE"].nbr_rounds:
                x0 = [cipher.states["STATE"].vars[r0][2][4*c0+k] for k in range(4)]
                x1 = [cipher.states["STATE"].vars[r1][2][4*c1+k] for k in range(4)]
                x2 = [cipher.states["STATE"].vars[r2][2][4*c2+k] for k in range(4)]
                y0 = [cipher.states["STATE"].vars[r0][4][4*c0+k] for k in range(4)]
                y1 = [cipher.states["STATE"].vars[r1][4][4*c1+k] for k in range(4)]
                y2 = [cipher.states["STATE"].vars[r2][4][4*c2+k] for k in range(4)]
                k0 = [cipher.states["KEY_STATE"].vars[i][0][4*j+k] for k in range(4)]
                k1 = [cipher.states["KEY_STATE"].vars[i+1][0][4*j+k-4] for k in range(4)]
                k2 = [cipher.states["KEY_STATE"].vars[i+1][0][4*j+k] for k in range(4)]
                u0 = [var.Variable(8,ID=f"u0_{i}_{j}_" +str(k)) for k in range(4)]
                u1 = [var.Variable(8,ID=f"u1_{i}_{j}_" +str(k)) for k in range(4)]
                u2 = [var.Variable(8,ID=f"u2_{i}_{j}_" +str(k)) for k in range(4)]
                v0 = [var.Variable(8,ID=f"v0_{i}_{j}_" +str(k)) for k in range(4)]
                v1 = [var.Variable(8,ID=f"v1_{i}_{j}_" +str(k)) for k in range(4)]
                v2 = [var.Variable(8,ID=f"v2_{i}_{j}_" +str(k)) for k in range(4)]
                # for k in range(4):
                #     print(x0[k].ID, x1[k].ID, x2[k].ID)
                #     print(y0[k].ID, y1[k].ID, y2[k].ID)
                #     print(k0[k].ID, k1[k].ID, k2[k].ID)
                #     print(u0[k].ID, u1[k].ID, u2[k].ID)
                #     print(v0[k].ID, v1[k].ID, v2[k].ID)
                for k in range(4):
                    xor = op.bitwiseXOR([x0[k], x1[k]], [u0[k]], ID=f"addMixColumnsExtraConstr_xor0_{i}_{j}_{k}")
                    xor.model_version = "truncated_diff_1"
                    add_cons += xor.generate_model("milp")
                    nxor0 = op.N_XOR([y0[k], y1[k], k2[k]], [v0[k]], ID=f"addMixColumnsExtraConstr_nxor0_{i}_{j}_{k}")
                    nxor0.model_version = "truncated_diff"
                    add_cons += nxor0.generate_model("milp")
                matrix = op.Matrix("MC", u0, v0, mat=[[0 for _ in range(4)] for _ in range(4)], ID=F"addMixColumnsExtraConstr_matrix0_{i}_{j}")
                matrix.model_version = "truncated_diff"
                add_cons += matrix.generate_model("milp", branch_num=5)
                
                for k in range(4):
                    xor = op.bitwiseXOR([x0[k], x2[k]], [u1[k]], ID=f"addMixColumnsExtraConstr_xor1_{i}_{j}_{k}")
                    xor.model_version = "truncated_diff_1"
                    add_cons += xor.generate_model("milp")
                    nxor0 = op.N_XOR([y0[k], y2[k], k1[k]], [v1[k]], ID=f"addMixColumnsExtraConstr_nxor1_{i}_{j}_{k}")
                    nxor0.model_version = "truncated_diff"
                    add_cons += nxor0.generate_model("milp")
                matrix = op.Matrix("MC", u1, v1, mat=[[0 for k in range(4)] for l in range(4)], ID=F"addMixColumnsExtraConstr_matrix1_{i}_{j}")
                matrix.model_version = "truncated_diff"
                add_cons += matrix.generate_model("milp", branch_num=5)
                
                for k in range(4):
                    xor = op.bitwiseXOR([x1[k], x2[k]], [u2[k]], ID=f"addMixColumnsExtraConstr_xor2_{i}_{j}_{k}")
                    xor.model_version = "truncated_diff_1"
                    add_cons += xor.generate_model("milp")
                    nxor0 = op.N_XOR([y1[k], y2[k], k0[k]], [v2[k]], ID=f"addMixColumnsExtraConstr_nxor2_{i}_{j}_{k}")
                    nxor0.model_version = "truncated_diff"
                    add_cons += nxor0.generate_model("milp")
                matrix = op.Matrix("MC", u2, v2, mat=[[0 for k in range(4)] for l in range(4)], ID=F"addMixColumnsExtraConstr_matrix2_{i}_{j}")
                matrix.model_version = "truncated_diff"
                add_cons += matrix.generate_model("milp", branch_num=5)
    return add_cons


def related_key_diff_AES(cipher, r):
    states = [s for s in cipher.states]
    rounds = {s: list(range(1, cipher.states[s].nbr_rounds + 1)) for s in states}
    layers = {s: {r: list(range(cipher.states[s].nbr_layers+1)) for r in rounds[s]} for s in states}
    positions = {s: {r: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in layers[s][r]} for r in rounds[s]} for s in states}
    attacks.set_model_versions(cipher, "truncated_diff", states=states, rounds=rounds, layers=layers, positions=positions) # set model_version = "truncated_diff" for each operation of the cipher
    add_constraints = []
    add_constraints += addkeyScheduleExtraConstr(cipher) # generate the first type of additional constraints 
    add_constraints += addMixColumnsExtraConstr(cipher) # generate the second type of additional constraints 
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_constraints, goal="search_optimal_truncated_trail") # apply milp-based differential attacks
    return obj


if __name__ == '__main__':
    # result = {'Rounds': [i for i in range(3,6)], 'NA0': [3, 9, 11], 'NA1': [3, 9, 12], 'NA2': [], 'NA2_': [5, 12, 17],'TIME': []}
    # result = {'Rounds': [i for i in range(3,10)], 'NA0': [1, 3, 4, 5, 11, 13, 16], 'NA1': [1, 3, 4, 5, 12, 13, 16], 'NA2': [], 'NA2_': [1, 4, 5, 10, 14, 18, 24],'TIME': []}
    result = {'Rounds': [i for i in range(3,15)], 'NA0': [1, 3, 3, 5, 5, 10, 14, 16, 18, 20, 22, 24], 'NA1': [1, 3, 3, 5, 5, 10, 14, 16, 18, 20, 22, 24], 'NA2': [], 'NA2_': [1, 3, 3, 5, 5, 10, 15, 16, 20, 20, 24, 24],'TIME': []}

    for r in result["Rounds"]:
        cipher = TEST_AES_BLOCKCIPHER(r, version=[128, 256])
        time_start = time.time()
        obj = related_key_diff_AES(cipher, r)
        result["NA2"].append(int(obj))
        result["TIME"].append(round(time.time() - time_start, 2))
    print(result)
    
    # print in the latex format
    for i in range(len(result['Rounds'])):
        print(f"{result['Rounds'][i]} & {result['NA0'][i]} & {result['NA1'][i]} & {result['NA2'][i]} & {result['NA2_'][i]} & {result['TIME'][i]} \\\\")







