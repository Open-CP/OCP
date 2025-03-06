# TODOs:
# possibility to have several dimensions on the state
import primitives.primitives as prim
import variables.variables as var
import attacks.attacks as attacks 


# ********************* TEST OF CIPHERS CODING IN PYTHON AND C********************* #  
def generate_codes(cipher):
    cipher.generate_code("files/" + cipher.name + ".py", "python")
    cipher.generate_code("files/" + cipher.name + "_unrolled.py", "python", True)
    cipher.generate_code("files/" + cipher.name + ".c", "c")
    cipher.generate_code("files/" + cipher.name + "_unrolled.c", "c", True)
    cipher.generate_figure("files/" + cipher.name + ".pdf")
    

# ********************* TEST OF CIPHERS MODELING IN MILP and SAT********************* #   
def TEST_SPECK_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_permutation(f"SPECK{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher
   

def TEST_SIMON_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_permutation(f"SIMON{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_ASCON_PERMUTATION(r=None):
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(320)], [var.Variable(1,ID="out"+str(i)) for i in range(320)]
    my_cipher = prim.ASCON_permutation("ASCON_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY_PERMUTATION(r=None, version=64):
    my_input, my_output = [var.Variable(int(version/16),ID="in"+str(i)) for i in range(16)], [var.Variable(int(version/16),ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_permutation("SKINNY_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_AES_PERMUTATION(r=None):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_GIFT_PERMUTATION(r=None, version=64): 
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(version)], [var.Variable(1,ID="out"+str(i)) for i in range(version)]
    my_cipher = prim.GIFT_permutation(f"GIFT{version}_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_ROCCA_AD(r=None):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(128+32*r)], [var.Variable(8,ID="out"+str(i)) for i in range(128+32*r)]
    my_cipher = prim.Rocca_AD_permutation(f"ROCCA_AD", my_input, my_output, nbr_rounds=r)
    return my_cipher


def TEST_SPECK_BLOCKCIPHER(r=None, version = [32, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Speck_block_cipher(f"SPECK{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_SKINNY_BLOCKCIPHER(r=None, version=[64, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/16), int(version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(16)], [var.Variable(word_size,ID="k"+str(i)) for i in range(16*m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.Skinny_block_cipher(f"SKINNY{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_AES_BLOCKCIPHER(r=None, version = [128, 128]): 
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1]/version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = prim.AES_block_cipher(f"AES{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_SIMON_BLOCKCIPHER(r=None, version=[32,64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = prim.Simon_block_cipher(f"SIMON{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def TEST_GIFT_BLOCKCIPHER(r=None, version = [64, 128]): 
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = prim.GIFT_block_cipher(f"GIFT{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


# ********************* Perform differential cryptanalysis on several ciphers ********************* #  
"""
Procedure:
=== Step 1: Initialization of the cipher for a specified number of rounds ===
=== Step 2: Configuration of model versions for differential analysis ===
    Configures the differential behavior of the cipher's operations based on the parameter model_version.
    - model_versions = {} by default, which applies model_version = "diff_0" to all operations, aiming to search for the differential trails with the highest probability.
    - model_version = "diff_1" for S-boxes, which models difference propagation without probabilities, aiming to search for the differential trails with the minimal number of active S-boxes.
=== Step 3: Generate additional constraints ===
    - add_constraints = [] by default, indicating no additional constraints are added to the model.
    - users can generate additional constraints using "attacks.gen_add_constraints()".
=== Step 4: Execute differential attacks ===
    Execute attacks using the "attacks.diff_attacks()" function to construct a MILP or SAT model.
    - model_type = "milp" by default, aiming to construct the MILP model.
    - model_type = "sat",  aiming to construct the SAT model.
=== Step 5: Solve the Model ===
    - Call "solving.solve_milp()" for solving MILP models.
    - Call "solving.solve_sat()" for solving SAT models.
"""
    

def TEST_DIFF_ATTACK_SPECK():
    # TEST 1: Search for the best differential trail of r-round SPECK by solving MILP models
    r = 6
    cipher = TEST_SPECK_PERMUTATION(r, version = 32) 
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")

    
    # TEST 2: Search for the best differential trail of r-round SPECK by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")
    

    # TEST 3: Search for the best related-key differential trail of r-round SPECK by solving MILP models
    cipher = TEST_SPECK_BLOCKCIPHER(r, version=[32,64]) 
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 4: Search for the best related-key differential trail of r-round SPECK by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_SIMON():
    # TEST 1: Search for the best differential trail of r-round SIMON by solving MILP models
    r = 6
    cipher = TEST_SIMON_PERMUTATION(r, version = 32)
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the best differential trail of r-round SIMON by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")


    # TEST 3: Search for the best related-key differential trail of r-round SIMON by solving MILP models
    cipher = TEST_SIMON_BLOCKCIPHER(r, version=[32,64]) 
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 4: Search for the best related-key differential trail of r-round SIMON by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_ASCON():
    # TEST 1: Search for the best differential trail of r-round ASCON by solving MILP models
    r = 2
    cipher = TEST_ASCON_PERMUTATION(r) 
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the best differential trail of r-round ASCON by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")

    
    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving MILP models
    # set `model_version = "diff_1"` for each S-box.
    model_versions = attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[1]}, positions = {r: {"STATE": {1: list(range(64))}} for r in range(1, cipher.nbr_rounds + 1)})
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")
    

    # TEST 4: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_GIFT():
    # TEST 1: Search for the best differential trail of r-round GIFT by solving MILP models
    r = 5
    cipher = TEST_GIFT_PERMUTATION(r, version = 64)
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving MILP models
    # set `model_version = "diff_1"` for each S-box.
    attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[0]}, positions = {r: {"STATE": {0: list(range(len(cipher.states["STATE"].constraints[r][0])))}} for r in range(1, cipher.nbr_rounds + 1)})
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")
    

    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving SAT models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")

    
    # TEST 4: Search for the best related-key differential trail of r-round GIFT by solving MILP models
    cipher = TEST_GIFT_BLOCKCIPHER(r, version = [64, 128])
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")

    # TEST 5: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving MILP models
    # set model_version = "diff_1" for each S-box.
    attacks.set_model_versions(cipher, "diff_1", rounds = [i for i in range(1, cipher.nbr_rounds + 1)], states=["STATE"], layers={"STATE":[0]}, positions = {r: {"STATE": {0: list(range(len(cipher.states["STATE"].constraints[r][0])))}} for r in range(1, cipher.nbr_rounds + 1)})
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp")

    # TEST 6: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving MILP models
    sol_list, obj_list, variable_map = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_AES():
    # TEST 1: Search for the best truncated differential trail of r-round AES by solving MILP models
    # set model_version = "truncated_diff" for each operation within the cipher
    r = 6
    cipher = TEST_AES_PERMUTATION(r)
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp", goal="search_optimal_truncated_trail")
    

    # TEST 2: Search for the best truncated related-key differential trail of r-round AES
    # set model_version = "truncated_diff" for each operation within the cipher
    r = 5
    cipher = TEST_AES_BLOCKCIPHER(r, version = [128, 128])
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    sol_list, obj_list = attacks.diff_attacks(cipher, model_type="milp", goal="search_optimal_truncated_trail")


def TEST_DIFF_ATTACK_ROCCA_AD():
    # TEST 1: Search for the best truncated differential trail that are used in Forgery attacks of r-round ROCCA_AD
    # set model_version = "truncated_diff" for each operation within the cipher
    # generate the following constraints to search for the differential trails that are used in Forgery attacks:
    # (1) input difference of the state is 0;
    # (2) difference of the data block is not 0;
    # (3) output difference of the state is 0
    r = 7
    cipher = TEST_ROCCA_AD(r)
    states = cipher.states
    layers = {s: [i for i in range(cipher.states[s].nbr_layers+1)] for s in states}
    positions = {"inputs": list(range(len(cipher.inputs_constraints))), **{r: {s: {l: list(range(len(cipher.states[s].constraints[r][l]))) for l in range(states[s].nbr_layers+1)} for s in states} for r in range(1, cipher.nbr_rounds + 1)}}
    attacks.set_model_versions(cipher, "truncated_diff", rounds = ["inputs"] + [i for i in range(1, cipher.nbr_rounds + 1)], states=states, layers=layers, positions=positions)
    add_cons = attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", rounds=[1], states=["STATE"], layers={"STATE":[0]}, positions={1:{"STATE":{0:[i for i in range(128)]}}}, bitwise=False, value=0)
    add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", rounds=[cipher.nbr_rounds], states=["STATE"], layers={"STATE":[4]}, positions={cipher.nbr_rounds:{"STATE":{4:[i for i in range(128)]}}}, bitwise=False, value=0)
    add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="SUM_GREATER_EQUAL", rounds=[1], states=["STATE"], layers={"STATE":[0]}, positions={1:{"STATE":{0:[i for i in range(128, 128+32*r)]}}}, bitwise=False, value=1)
    sol_list, obj_list = attacks.diff_attacks(cipher, add_constraints=add_cons, model_type="milp", goal="search_optimal_truncated_trail")


if __name__ == '__main__':
    r = 2
    cipher = TEST_SPECK_PERMUTATION(r, version = 32) # version = 32, 48, 64, 96, 128
    cipher = TEST_SIMON_PERMUTATION(r, version = 32) # version = 32, 48, 64, 96, 128
    cipher = TEST_AES_PERMUTATION(r)
    cipher = TEST_ASCON_PERMUTATION(r) 
    cipher = TEST_SKINNY_PERMUTATION(r, version = 64) # version = 64, 128
    cipher = TEST_GIFT_PERMUTATION(r, version = 64) # version = 64, 128
    cipher = TEST_ROCCA_AD(r)

    cipher = TEST_SPECK_BLOCKCIPHER(r, version=[32,64]) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    cipher = TEST_SIMON_BLOCKCIPHER(r, version = [32, 64]) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    cipher = TEST_AES_BLOCKCIPHER(r, version = [128, 128]) # version = [128, 128], [128, 192], [128, 256] 
    cipher = TEST_SKINNY_BLOCKCIPHER(r, version = [64, 64]) # version = [64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]  
    cipher = TEST_GIFT_BLOCKCIPHER(r, version = [64, 128]) # version = [64, 128],  [128, 128]
    generate_codes(cipher)

    TEST_DIFF_ATTACK_SPECK()
    TEST_DIFF_ATTACK_SIMON()
    TEST_DIFF_ATTACK_ASCON()
    TEST_DIFF_ATTACK_GIFT()
    TEST_DIFF_ATTACK_AES()
    TEST_DIFF_ATTACK_ROCCA_AD()







