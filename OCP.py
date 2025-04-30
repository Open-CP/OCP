import primitives.primitives as prim
import primitives.skinny as skinny
import primitives.simon as simon
import primitives.speck as speck
import primitives.rocca as rocca
import primitives.ascon as ascon
import primitives.gift as gift
import primitives.aes as aes
import primitives.siphash as siphash
import primitives.present as present
import variables.variables as var
import attacks.attacks as attacks 
import implementations.implementations as imp 
import visualisations.visualisations as vis 
import os

if not os.path.exists('files'):
    os.makedirs('files')

# ********************* PYTHON, C IMPLEMENTATIONS AND FIGURES ********************* #  
def generate_codes(cipher):
    imp.generate_implementation(cipher,"files/" + cipher.name + ".py", "python")
    imp.generate_implementation(cipher,"files/" + cipher.name + "_unrolled.py", "python", True)
    imp.generate_implementation(cipher,"files/" + cipher.name + ".c", "c")
    imp.generate_implementation(cipher,"files/" + cipher.name + "_unrolled.c", "c", True)
    vis.generate_figure(cipher,"files/" + cipher.name + ".pdf")
    

# ********************* CIPHERS ********************* #   
def SPECK_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = speck.Speck_permutation(f"SPECK{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher
   

def SIMON_PERMUTATION(r=None, version=32):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = simon.Simon_permutation(f"SIMON{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r)
    return my_cipher


def ASCON_PERMUTATION(r=None, represent_mode=0):
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(320)], [var.Variable(1,ID="out"+str(i)) for i in range(320)]
    my_cipher = ascon.ASCON_permutation("ASCON_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    return my_cipher


def SKINNY_PERMUTATION(r=None, version=64):
    my_input, my_output = [var.Variable(int(version/16),ID="in"+str(i)) for i in range(16)], [var.Variable(int(version/16),ID="out"+str(i)) for i in range(16)]
    my_cipher = skinny.Skinny_permutation("SKINNY_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def AES_PERMUTATION(r=None):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = aes.AES_permutation("AES_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def GIFT_PERMUTATION(r=None, version=64): 
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(version)], [var.Variable(1,ID="out"+str(i)) for i in range(version)]
    my_cipher = gift.GIFT_permutation(f"GIFT{version}_PERM", version, my_input, my_output, nbr_rounds=r)
    return my_cipher


def SIPHASH_PERMUTATION(r=None): 
    my_input, my_output = [var.Variable(64,ID="in"+str(i)) for i in range(4)], [var.Variable(64,ID="out"+str(i)) for i in range(4)]
    my_cipher = siphash.SipHash_permutation("SipHash_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher


def PRESENT_PERMUTATION(r=None): 
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(64)], [var.Variable(1,ID="out"+str(i)) for i in range(64)]
    my_cipher = present.PRESENT_permutation(f"PRESENT_PERM", 64, my_input, my_output, nbr_rounds=r)
    return my_cipher


def ROCCA_AD(r=5, represent_mode=0):
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(128+32*r)], [var.Variable(8,ID="out"+str(i)) for i in range(128+32*r)]
    my_cipher = rocca.Rocca_AD_permutation("ROCCA_AD", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    return my_cipher


def SPECK_BLOCKCIPHER(r=None, version = [32, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = speck.Speck_block_cipher(f"SPECK{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def SKINNY_BLOCKCIPHER(r=None, version=[64, 64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/16), int(version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(16)], [var.Variable(word_size,ID="k"+str(i)) for i in range(16*m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(16)]
    my_cipher = skinny.Skinny_block_cipher(f"SKINNY{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def AES_BLOCKCIPHER(r=None, version = [128, 128]): 
    my_plaintext, my_key, my_ciphertext = [var.Variable(8,ID="in"+str(i)) for i in range(16)], [var.Variable(8,ID="k"+str(i)) for i in range(int(16*version[1]/version[0]))], [var.Variable(8,ID="out"+str(i)) for i in range(16)]
    my_cipher = aes.AES_block_cipher(f"AES{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def SIMON_BLOCKCIPHER(r=None, version=[32,64]):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = simon.Simon_block_cipher(f"SIMON{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def GIFT_BLOCKCIPHER(r=None, version = [64, 128]): 
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = gift.GIFT_block_cipher(f"GIFT{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


def PRESENT_BLOCKCIPHER(r=None, version = [64, 80]): 
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = present.PRESENT_block_cipher(f"PRESENT{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r)
    return my_cipher


# ********************* Perform differential cryptanalysis on several ciphers ********************* #  
"""
Procedure:
=== Step 1: Initialization of the cipher for a specified number of rounds ===
=== Step 2: Configuration of model versions for differential analysis ===
    Configures the differential behavior of the cipher's operations based on the parameter model_version.
    - model_versions = "DEFAULT" by default, which applies to all operations
    - For S-boxes with model_versions = "DEFAULT" or "DIFF_PR", it aims to search for the differential trails with the highest probability.
    - For S-boxes with model_version = "DIFF",it aims to search for the differential trails with the minimal number of active S-boxes.
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
    r = 2
    cipher = SPECK_PERMUTATION(r, version = 32) 
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", show_mode=0)

    
    # TEST 2: Search for the best differential trail of r-round SPECK by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat", show_mode=1)
    

    # TEST 3: Search for the best related-key differential trail of r-round SPECK by solving MILP models
    cipher = SPECK_BLOCKCIPHER(r, version=[32,64]) 
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", show_mode=1)


    # TEST 4: Search for the best related-key differential trail of r-round SPECK by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat", show_mode=2)


def TEST_DIFF_ATTACK_SIMON():
    # TEST 1: Search for the best differential trail of r-round SIMON by solving MILP models
    r = 6
    cipher = SIMON_PERMUTATION(r, version = 32)
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the best differential trail of r-round SIMON by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")


    # TEST 3: Search for the best related-key differential trail of r-round SIMON by solving MILP models
    cipher = SIMON_BLOCKCIPHER(r, version=[32,64]) 
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 4: Search for the best related-key differential trail of r-round SIMON by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_ASCON():
    # TEST 1: Search for the best differential trail of r-round ASCON by solving MILP models
    r = 2
    cipher = ASCON_PERMUTATION(r) 
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the best differential trail of r-round ASCON by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")

    
    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving MILP models
    # set `model_version = "ASCON_Sbox_DIFF"` for each S-box.
    attacks.set_model_versions(cipher, "ASCON_Sbox_DIFF", operator_name="ASCON_Sbox")
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")
    

    # TEST 4: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_GIFT():
    # TEST 1: Search for the best differential trail of r-round GIFT by solving MILP models
    r = 5
    cipher = GIFT_PERMUTATION(r, version = 64)
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")


    # TEST 2: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving MILP models
    # set `model_version = "GIFT_Sbox_DIFF"` for each S-box.
    attacks.set_model_versions(cipher, "GIFT_Sbox_DIFF", operator_name="GIFT_Sbox")
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")
    

    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving SAT models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")

    
    # TEST 4: Search for the best related-key differential trail of r-round GIFT by solving MILP models
    cipher = GIFT_BLOCKCIPHER(r, version = [64, 128])
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")

    # TEST 5: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving MILP models
    # set `model_version = "GIFT_Sbox_DIFF"` for each S-box.
    attacks.set_model_versions(cipher, "GIFT_Sbox_DIFF", operator_name="GIFT_Sbox")
    sol, obj = attacks.diff_attacks(cipher, model_type="milp")

    # TEST 6: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving MILP models
    sol, obj = attacks.diff_attacks(cipher, model_type="sat")


def TEST_DIFF_ATTACK_AES():
    # TEST 1: Search for the best truncated differential trail of r-round AES by solving MILP models
    # set model_version = "DIFF_TRUNCATED" for each operation within the cipher
    r = 6
    cipher = AES_PERMUTATION(r)
    attacks.set_model_versions(cipher, "DIFF_TRUNCATED")
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", goal="search_optimal_truncated_trail")
    

    # TEST 2: Search for the best truncated related-key differential trail of r-round AES
    # set model_version = "DIFF_TRUNCATED" for each operation within the cipher
    r = 5
    cipher = AES_BLOCKCIPHER(r, version = [128, 128])
    attacks.set_model_versions(cipher, "DIFF_TRUNCATED")
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", goal="search_optimal_truncated_trail")


def TEST_DIFF_ATTACK_ROCCA_AD():
    # TEST 1: Search for the best truncated differential trail that are used in Forgery attacks of r-round ROCCA_AD
    # set model_version = "DIFF_TRUNCATED" for each operation within the cipher
    # generate the following constraints to search for the differential trails that are used in Forgery attacks:
    # (1) input difference of the state is 0;
    # (2) output difference of the state is 0
    # (3) difference of the data block is not 0 (default in diff_attacks);
    r = 7
    cipher = ROCCA_AD(r)
    attacks.set_model_versions(cipher, "DIFF_TRUNCATED")
    add_cons = attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", states=["STATE"], rounds={"STATE":[1]}, layers={"STATE":{1:[0]}}, positions={"STATE":{1:{0:list(range(128))}}}, bitwise=False, value=0)
    add_cons += attacks.gen_add_constraints(cipher, model_type="milp", cons_type="EQUAL", states=["STATE"], rounds={"STATE":[cipher.nbr_rounds]}, layers={"STATE":{cipher.nbr_rounds:[4]}}, positions={"STATE":{cipher.nbr_rounds:{4:list(range(128))}}}, bitwise=False, value=0)
    sol, obj = attacks.diff_attacks(cipher, model_type="milp", add_constraints=add_cons, goal="search_optimal_truncated_trail")


if __name__ == '__main__':
    r = 2
    #generate_codes(SPECK_PERMUTATION(r, version = 32)) # version = 32, 48, 64, 96, 128
    #generate_codes(SIMON_PERMUTATION(r, version = 32)) # version = 32, 48, 64, 96, 128
    #generate_codes(AES_PERMUTATION(r))
    #generate_codes(ASCON_PERMUTATION(r))
    #generate_codes(SKINNY_PERMUTATION(r, version = 64)) # version = 64, 128
    #generate_codes(GIFT_PERMUTATION(r, version = 64)) # version = 64, 128
    generate_codes(SIPHASH_PERMUTATION(r))
    #generate_codes(ROCCA_AD(r))
    
    #generate_codes(SPECK_BLOCKCIPHER(r, version=[32,64])) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    #generate_codes(SIMON_BLOCKCIPHER(r, version = [32, 64])) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    #generate_codes(AES_BLOCKCIPHER(r, version = [128, 128])) # version = [128, 128], [128, 192], [128, 256] 
    #generate_codes(SKINNY_BLOCKCIPHER(r, version = [64, 64])) # version = [64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]  
    #generate_codes(GIFT_BLOCKCIPHER(r, version = [64, 128])) # version = [64, 128],  [128, 128]

    TEST_DIFF_ATTACK_SPECK()
    #TEST_DIFF_ATTACK_SIMON()
    #TEST_DIFF_ATTACK_ASCON()
    #TEST_DIFF_ATTACK_GIFT()
    #TEST_DIFF_ATTACK_AES()
    #TEST_DIFF_ATTACK_SIPHASH()
    #TEST_DIFF_ATTACK_ROCCA_AD()
    







