"""
This module provides usage examples for the SPECK primitive and SPECK block cipher, including:

1. Generating software implementations and visualizations
2. Conducting differential cryptanalysis using MILP and SAT methods

Note:
For examples of other ciphers, refer to the following folders:
- test/implementation
- test/differential_cryptanalysis
"""

import copy
from pathlib import Path

from kiwisolver import Variable
import implementations.implementations as imp
import visualisations.visualisations as vis
import attacks.attacks as attacks

FILES_DIR = Path("files")
FILES_DIR.mkdir(parents=True, exist_ok=True)


# ********************* IMPLEMENTATIONS ********************* #
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
    imp.test_implementation_verilog(cipher, cipher.name, cipher.test_vectors[0], cipher.test_vectors[1])

def test_verilog_unrolled_imp(cipher): # Generate unrolled Verilog implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.sv", "verilog", True)
    if not cipher.test_vectors:
        print("warning: no test vector defined!")
        return False
    imp.test_implementation_verilog(cipher, cipher.name + "_unrolled", cipher.test_vectors[0], cipher.test_vectors[1])

def test_all_implementations(cipher): # Generate all implementations
    test_python_imp(cipher)
    test_python_unrolled_imp(cipher)
    test_c_imp(cipher)
    test_c_unrolled_imp(cipher)
    #test_verilog_imp(cipher)
    #test_verilog_unrolled_imp(cipher)


# ********************* VISUALIZATIONS ********************* #
def test_visualisation(cipher): # Generate visualisation figure
    vis.generate_figure(cipher, FILES_DIR / f"{cipher.name}.pdf")

# ********************* Differential Cryptanalysis ********************* #
def test_diff_attack_milp(cipher):
    # Example: default parameters. Refer to test/differential_cryptanalysis/ for more available parameters.
    goal="DIFFERENTIALPATH_PROB"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model=None
    config_solver=None

    # Search for the differential trail
    trails = attacks.diff_attacks(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)

def test_diff_attack_sat(cipher):
    # Example: default parameters. Refer to test/differential_cryptanalysis/ for more available parameters.
    goal="DIFFERENTIALPATH_PROB"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model={"model_type": "sat"}
    config_solver=None

    # Search for the differential trail
    trails = attacks.diff_attacks(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)


if __name__ == "__main__":
    import primitives.aes as aes
    #cipher = primitives.speck.SPECK_PERMUTATION(version=32)
    cipher = aes.AES_BLOCKCIPHER(version=[128,128])
    imp.generate_implementation(cipher,"files/" + cipher.name + "_unrolled.py", "python", True)
    cipher = aes.AES_BLOCKCIPHER(version=[128,192])
    imp.generate_implementation(cipher,"files/" + cipher.name + "_unrolled.py", "python", True)
    cipher = aes.AES_BLOCKCIPHER(version=[128,256])
    imp.generate_implementation(cipher,"files/" + cipher.name + "_unrolled.py", "python", True)
    import primitives.simon as simon
    #cipher = primitives.speck.SPECK_PERMUTATION(version=32)
    #cipher = simon.SIMON_BLOCKCIPHER(r=8, version=[32,64])

    test_all_implementations(cipher)
    cipher.add_copy_operators()
    test_visualisation(cipher)
    #test_diff_attack_milp(cipher)
    #test_diff_attack_sat(cipher)
