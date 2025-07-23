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
import attacks.differential_cryptanalysis as dif 
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
    # vis.generate_figure(cipher,"files/" + cipher.name + ".pdf")
    

def test_vectors(cipher):
    if cipher.test_vectors == []:
        print("warning: no test vector defined!")
        return False
    
    # test Python implementation
    # imp.test_implementation_python(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1])
    imp.test_implementation_python(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1])

    # test C implementation
    # imp.test_implementation_c(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1])
    imp.test_implementation_c(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1])


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
=== Step 2: Generate additional constraints ===
    - add_constraints = ["INPUT_NOT_ZERO"] by default.
    - users can generate additional constraints using "attacks.gen_predefined_constraints()".
=== Step 3: Set required parameters and execute differential attacks ===
    Execute attacks using the "dif.diff_attacks()" function to construct and solve the MILP/SAT model.
"""
    

def TEST_DIFF_ATTACK_SPECK():
    # TEST 1: Search for the best differential trail of r-round SPECK by solving MILP models
    cipher = SPECK_PERMUTATION(r=5, version = 32) 
    
    trail = dif.search_diff_trail(cipher) # Default parameters setting: goal="DIFFERENTIALPATH_PROB", constraints=["INPUT_NOT_ZERO"], objective_target="OPTIMAL", show_mode=0, config_model={"model_type": "milp"}, config_solver={"solver": "DEFAULT"}

    # Adjust 'objective_target'.
    trail = dif.search_diff_trail(cipher, objective_target="OPTIMAL STARTING FROM 8")
    trail = dif.search_diff_trail(cipher, objective_target="AT MOST 8")
    trail = dif.search_diff_trail(cipher, objective_target="EXACTLY 10")

    # Adjust ‘config_solver’, 'show_mode'
    trail = dif.search_diff_trail(cipher, show_mode=2, config_model={"model_type": "milp"}, config_solver={"solver": "Gurobi", "time_limit": 1, "OutputFlag": False})
    trail = dif.search_diff_trail(cipher, show_mode=2, config_solver={"solver": "SCIP"})
    trail = dif.search_diff_trail(cipher, objective_target="OPTIMAL STARTING FROM 8", show_mode=1, config_solver={"solver": "SCIP", "time_limit": 2})
    trail = dif.search_diff_trail(cipher, objective_target="AT MOST 8", show_mode=1, config_solver={"solver": "SCIP", "time_limit": 2})
    trail = dif.search_diff_trail(cipher, objective_target="EXACTLY 10", show_mode=1, config_solver={"solver": "SCIP", "time_limit": 2})
    
    # Adjust 'config_model': add matsui's constraints
    trail = dif.search_diff_trail(cipher, config_model={"matsui_constraint": {"Round": 5, "best_obj": [0,1,3,5]}})
    trail = dif.search_diff_trail(cipher, config_model={"matsui_constraint": {"Round": 5, "best_obj": [0,1,3,5], "matsui_milp_cons_type": "UPPER"}})
    trail = dif.search_diff_trail(cipher, config_model={"matsui_constraint": {"Round": 5, "best_obj": [0,1,3,5], "matsui_milp_cons_type": "LOWER"}})
    
    # Adjust 'config_model': select search rounds  
    trail = dif.search_diff_trail(cipher, config_model={"rounds": {s: [3,4,5] for s in cipher.states}})


    # TEST 2: Search for the best differential trail of r-round SPECK by solving SAT models
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})
    
    # Adjust 'objective_target'.
    trail = dif.search_diff_trail(cipher, objective_target="OPTIMAL STARTING FROM 8", config_model={"model_type": "sat"})
    trail = dif.search_diff_trail(cipher, objective_target="AT MOST 9", config_model={"model_type": "sat"})
    trail = dif.search_diff_trail(cipher, objective_target="EXACTLY 10", config_model={"model_type": "sat"})
    
    # Adjust config_model - "optimal_search_strategy_sat": "AT_MOST" (Default), "EXACTLY".
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "AT_MOST"})
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "EXACTLY"})

    # Adjust config_model - "atmost_encoding_sat": "SEQUENTIAL" (Default), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "AT_MOST", "atmost_encoding_sat": 1})
    
    # Adjust config_model - "exact_encoding_sat": 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "EXACTLY", "exact_encoding_sat": 1})

    # Adjust config_model - "atleast_encoding_sat": "SEQUENTIAL" (Default), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "atleast_encoding_sat": 2})
    
    # Adjust 'config_model': add matsui's constraints
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "matsui_constraint": {"Round": 5, "best_obj": [0,1,3,5]}})
    
    # Adjust ‘config_solver’
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"}, config_solver={"solver":"Cadical103"})

    # TEST 3: Search for the best related-key differential trail of r-round SPECK by solving MILP models
    cipher = SPECK_BLOCKCIPHER(r=5, version=[32,64])
    trail = dif.search_diff_trail(cipher, show_mode=2)

    # TEST 4: Search for the best related-key differential trail of r-round SPECK by solving SAT models
    trail = dif.search_diff_trail(cipher, show_mode=2, config_model={"model_type": "sat"})


def TEST_DIFF_ATTACK_SIMON():
    # TEST 1: Search for the best differential trail of r-round SIMON by solving MILP models
    cipher = SIMON_PERMUTATION(r=5, version = 32)
    trail = dif.search_diff_trail(cipher)


    # TEST 2: Search for the best differential trail of r-round SIMON by solving SAT models
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})


    # TEST 3: Search for the best related-key differential trail of r-round SIMON by solving MILP models
    cipher = SIMON_BLOCKCIPHER(r=7, version=[32,64]) 
    trail = dif.search_diff_trail(cipher)


    # TEST 4: Search for the best related-key differential trail of r-round SIMON by solving SAT models
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})


def TEST_DIFF_ATTACK_ASCON():
    # TEST 1: Search for the best differential trail of r-round ASCON by solving MILP models
    cipher = ASCON_PERMUTATION(r=1) 
    trail = dif.search_diff_trail(cipher)

    # Adjust 'config_model' - 'model_params' for Sbox
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "milp", "model_params": {"ASCON_Sbox": {"tool_type": "polyhedron"}}})
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "milp", "model_params": {"ASCON_Sbox": {"tool_type": "minimize_logic", "mode": 1}}})


    # TEST 2: Search for the best differential trail of r-round ASCON by solving SAT models
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})
    
    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving MILP models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT")
    

    # TEST 4: Search for the minimal number of active differentially S-boxes of r-round ASCON by solving SAT models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT", config_model={"model_type": "sat"})


def TEST_DIFF_ATTACK_GIFT():
    # TEST 1: Search for the best differential trail of r-round GIFT by solving MILP/SAT models
    cipher = GIFT_PERMUTATION(r=4, version = 64)
    trail = dif.search_diff_trail(cipher)

    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})
    
    # Adjust 'objective_target'.
    trail = dif.search_diff_trail(cipher, objective_target="OPTIMAL STARTING FROM 11", config_model={"model_type": "sat"})
    trail = dif.search_diff_trail(cipher, objective_target="AT MOST 12", config_model={"model_type": "sat"})
    trail = dif.search_diff_trail(cipher, objective_target="EXACTLY 14", config_model={"model_type": "sat"})
    
    # Adjust config_model - "optimal_search_strategy_sat": "AT_MOST" (Default), "EXACTLY".
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "AT_MOST"})
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "EXACTLY"})

    # Adjust config_model - "atmost_encoding_sat": "SEQUENTIAL" (Default), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "AT_MOST", "atmost_encoding_sat": 1})
    
    # Adjust config_model - "exact_encoding_sat": 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "optimal_search_strategy_sat": "EXACTLY", "exact_encoding_sat": 1})

    # Adjust config_model - "atleast_encoding_sat": "SEQUENTIAL" (Default), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9.
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat", "atleast_encoding_sat": 2})
    
    # Adjust config_model - "decimal_encoding_sat": "INTEGER_DECIMAL", "BOOLEAN".
    trail = dif.search_diff_trail(cipher, objective_target="OPTIMAL STARTING FROM 11", config_model={"model_type": "sat", "decimal_encoding_sat":  "BOOLEAN"})
    
    # Adjust ‘config_solver’
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"}, config_solver={"solver":"Cadical103"})


    # TEST 2: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving MILP models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT")
    

    # TEST 3: Search for the minimal number of active differentially S-boxes of r-round GIFT by solving SAT models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT", config_model={"model_type": "sat"})

    
    # TEST 4: Search for the best related-key differential trail of r-round GIFT by solving MILP/SAT models
    cipher = GIFT_BLOCKCIPHER(r=6, version = [64, 128])
    trail = dif.search_diff_trail(cipher)
    trail = dif.search_diff_trail(cipher, config_model={"model_type": "sat"})

    # TEST 5: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving MILP models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT")

    # TEST 6: Search for the minimal number of active related-key differentially S-boxes of r-round GIFT by solving SAT models
    trail = dif.search_diff_trail(cipher, goal = "DIFFERENTIAL_SBOXCOUNT", config_model={"model_type": "sat"})


def TEST_DIFF_ATTACK_AES():
    # TEST 1: Search for the best truncated differential trail of r-round AES by solving MILP models
    cipher = AES_PERMUTATION(r=5)
    trail = dif.search_diff_trail(cipher, goal="TRUNCATEDDIFF_SBOXCOUNT")
    

    # TEST 2: Search for the best truncated related-key differential trail of r-round AES
    cipher = AES_BLOCKCIPHER(r=5, version = [128, 128])
    trail = dif.search_diff_trail(cipher, goal="TRUNCATEDDIFF_SBOXCOUNT")


def TEST_DIFF_ATTACK_ROCCA_AD():
    # TEST 1: Search for the best truncated differential trail that are used in Forgery attacks of r-round ROCCA_AD
    # generate the following constraints to search for the differential trails that are used in Forgery attacks:
    # (1) input difference of the state is 0;
    # (2) output difference of the state is 0
    # (3) difference of the data block is not 0 (default in diff_attacks);
    cipher = ROCCA_AD(r=7)
    add_constraints = ["INPUT_NOT_ZERO"]
    add_constraints += attacks.gen_predefined_constraints(model_type="milp", cons_type="EXACTLY", cons_vars=cipher.states["STATE"].vars[1][0][:128], cons_value=0, bitwise=False)
    add_constraints += attacks.gen_predefined_constraints(model_type="milp", cons_type="EXACTLY", cons_vars=cipher.states["STATE"].vars[cipher.nbr_rounds][4][:128], cons_value=0, bitwise=False)
    trail = dif.search_diff_trail(cipher, constraints=add_constraints, goal="TRUNCATEDDIFF_SBOXCOUNT")


if __name__ == '__main__':
    # generate_codes(SPECK_PERMUTATION(r=3, version = 32)) # version = 32, 48, 64, 96, 128
    # generate_codes(SIMON_PERMUTATION(r=3, version = 32)) # version = 32, 48, 64, 96, 128
    # generate_codes(AES_PERMUTATION(r=3))
    # generate_codes(ASCON_PERMUTATION(r=3, represent_mode=0)) # r=3, 12; mode=0,1
    # test_vectors(ASCON_PERMUTATION(r=3, represent_mode=0))
    # generate_codes(SKINNY_PERMUTATION(r=None, version = 64)) # version = 64, 128
    # generate_codes(GIFT_PERMUTATION(r=None, version = 64)) # version = 64, 128
    # generate_codes(SIPHASH_PERMUTATION(r=2))
    # test_vectors(SIPHASH_PERMUTATION(r=2))
    # generate_codes(ROCCA_AD(r=5, represent_mode=0)) # represent_mode = 0, 1
    # test_vectors(ROCCA_AD(r=5, represent_mode=0))
    
    # generate_codes(SPECK_BLOCKCIPHER(r=None, version=[32,64])) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    # test_vectors(SPECK_BLOCKCIPHER(r=None, version=[32,64]))
    # generate_codes(SIMON_BLOCKCIPHER(r=None, version=[32, 64])) # version = [32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]
    # test_vectors(SIMON_BLOCKCIPHER(r=None, version= [32, 64]))
    # generate_codes(AES_BLOCKCIPHER(r=None, version=[128, 256])) # version = [128, 128], [128, 192], [128, 256] 
    # test_vectors(AES_BLOCKCIPHER(r=None, version=[128, 256]))
    # generate_codes(SKINNY_BLOCKCIPHER(r=None, version=[64, 64])) # version = [64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]  
    # test_vectors(SKINNY_BLOCKCIPHER(r=None, version=[64, 64]))
    # generate_codes(GIFT_BLOCKCIPHER(r=None, version=[64, 128])) # version = [64, 128],  [128, 128]
    # test_vectors(GIFT_BLOCKCIPHER(r=None, version=[64, 128]))
    # generate_codes(PRESENT_BLOCKCIPHER(r=None, version=[64, 80])) # [64, 80], [64, 128]
    # test_vectors(PRESENT_BLOCKCIPHER(r=None, version=[64, 80]))

    TEST_DIFF_ATTACK_SPECK()
    # TEST_DIFF_ATTACK_SIMON()
    # TEST_DIFF_ATTACK_ASCON()
    # TEST_DIFF_ATTACK_GIFT()
    # TEST_DIFF_ATTACK_AES()
    # TEST_DIFF_ATTACK_SIPHASH()
    # TEST_DIFF_ATTACK_ROCCA_AD()