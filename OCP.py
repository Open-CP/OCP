"""
This module provides usage examples for the SPECK primitive and SPECK block cipher, including:

1. Generating software implementations and visualizations
2. Conducting differential cryptanalysis using MILP and SAT methods

Note:
For examples of other ciphers, refer to the following folders:
- test/implementation
- test/differential_cryptanalysis
"""

from pathlib import Path
from primitives.speck import SPECK_PERMUTATION, SPECK_BLOCKCIPHER
import implementations.implementations as imp
import visualisations.visualisations as vis
import attacks.differential_cryptanalysis as dif

FILES_DIR = Path("files")
FILES_DIR.mkdir(parents=True, exist_ok=True)


# ********************* IMPLEMENTATIONS AND FIGURES ********************* #
def test_python_imp(cipher): # Generate Python implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}.py", "python")
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    imp.test_implementation_python(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1])

def test_python_unrolled_imp(cipher): # Generate unrolled Python implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.py", "python", True)
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    imp.test_implementation_python(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1])

def test_c_imp(cipher): # Generate C implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}.c", "c")
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    imp.test_implementation_c(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1])

def test_c_unrolled_imp(cipher): # Generate unrolled C implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.c", "c", True)
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    imp.test_implementation_c(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1])

def test_verilog_imp(cipher): # Generate Verilog implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}.sv", "verilog")
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    # imp.test_implementation_verilog(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1]) # TO DO

def test_verilog_unrolled_imp(cipher): # Generate unrolled Verilog implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.sv", "verilog", True)
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    # imp.test_implementation_verilog(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1]) # TO DO

def test_visualisation(cipher): # Generate visualisation figure
    vis.generate_figure(cipher, FILES_DIR / f"{cipher.name}.pdf")

def test_imp_speck_permutation():
    # ********************* DEFINE CIPHER ********************* #   
    cipher = SPECK_PERMUTATION(r=None, version=32)

    test_python_imp(cipher)

    test_python_unrolled_imp(cipher)

    test_c_imp(cipher)

    test_c_unrolled_imp(cipher)

    test_verilog_imp(cipher)

    test_verilog_unrolled_imp(cipher)

    test_visualisation(cipher)
    
def test_imp_speck_blockcipher():
    # ********************* DEFINE CIPHER ********************* #   
    cipher = SPECK_BLOCKCIPHER(r=None, version=[32, 64])

    test_python_imp(cipher)

    test_python_unrolled_imp(cipher)

    test_c_imp(cipher)

    test_c_unrolled_imp(cipher)

    # test_verilog_imp(cipher) # TO DO

    # test_verilog_unrolled_imp(cipher) # TO DO

    test_visualisation(cipher)


# ********************* Differential Cryptanalysis ********************* #
def test_diff_attack_speck_milp():
    # Step 1. Define the cipher (permutation or block cipher)    
    cipher = SPECK_PERMUTATION(r=5, version = 32)
    # cipher = SPECK_BLOCKCIPHER(r=5, version=[32,64])

    # Step 2. Set parameters. 
    # Example: default parameters. Refer to test/differential_cryptanalysis/test_diff_speck.py for more available parameters.
    goal="DIFFERENTIALPATH_PROB"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model=None
    config_solver=None
    
    # Step 3. Search for the differential trail
    trail = dif.search_diff_trail(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)

def test_diff_attack_speck_sat():
    # Step 1. Define the cipher (permutation or block cipher)    
    cipher = SPECK_PERMUTATION(r=5, version = 32)
    # cipher = SPECK_BLOCKCIPHER(r=5, version=[32,64])

    # Step 2. Set parameters.
    # Example: default parameters. Refer to test/differential_cryptanalysis/test_diff_speck.py for more available parameters.
    goal="DIFFERENTIALPATH_PROB"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model={"model_type": "sat"}
    config_solver=None
    
    # Step 3. Search for the differential trail
    trail = dif.search_diff_trail(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)


if __name__ == "__main__":

    test_imp_speck_permutation()

    test_imp_speck_blockcipher()

    test_diff_attack_speck_milp()

    test_diff_attack_speck_sat()