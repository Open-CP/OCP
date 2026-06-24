"""
This module provides usage examples for the SPECK primitive and SPECK block cipher, including:

1. Generating software implementations and visualizations
2. Conducting differential cryptanalysis using MILP and SAT methods
3. Conducting integral cryptanalysis using MILP methods

Note:
For examples of other ciphers, refer to the following folders:
- test/implementation
- test/differential_cryptanalysis
"""

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
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_python(cipher, cipher.name, tv[0], tv[1])

def test_python_unrolled_imp(cipher): # Generate unrolled Python implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.py", "python", True)
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_python(cipher, cipher.name + "_unrolled", tv[0], tv[1])

def test_c_imp(cipher): # Generate C implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}.c", "c")
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_c(cipher, cipher.name, tv[0], tv[1])

def test_c_unrolled_imp(cipher): # Generate unrolled C implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.c", "c", True)
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_c(cipher, cipher.name + "_unrolled", tv[0], tv[1])

def test_verilog_imp(cipher): # Generate Verilog implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}.sv", "verilog")
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_verilog(cipher, cipher.name, tv[0], tv[1])

def test_verilog_unrolled_imp(cipher): # Generate unrolled Verilog implementation and test it with the test vectors
    imp.generate_implementation(cipher, FILES_DIR / f"{cipher.name}_unrolled.sv", "verilog", True)
    if cipher.test_vectors==[]:
        print("warning: no test vector defined!")
        return False
    for tv in cipher.test_vectors: imp.test_implementation_verilog(cipher, cipher.name + "_unrolled", tv[0], tv[1])

def test_all_implementations(cipher): # Generate all implementations
    #test_python_imp(cipher)
    test_python_unrolled_imp(cipher)
    #test_c_imp(cipher)
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


# ********************* Linear Cryptanalysis ********************* #
def test_linear_attack_milp(cipher):
    # Example: default parameters. Refer to test/linear_cryptanalysis/ for more available parameters.
    goal="LINEARPATH_CORR"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model=None
    config_solver=None

    # Search for the linear trail
    trails = attacks.linear_attacks(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)

def test_linear_attack_sat(cipher):
    # Example: default parameters. Refer to test/linear_cryptanalysis/ for more available parameters.
    goal="LINEARPATH_CORR"
    constraints=["INPUT_NOT_ZERO"]
    objective_target="OPTIMAL"
    show_mode=0
    config_model={"model_type": "sat"}
    config_solver=None

    # Search for the linear trail
    trails = attacks.linear_attacks(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)


# ********************* Integral Cryptanalysis ********************* #
def test_integral_attack_milp(cipher, constant_bits, config_model=None, config_solver=None):
    # Example: two-subset integral distinguisher search using MILP.
    goal="INTEGRAL_TWOSUBSET"
    constraints=["TWO_SUBSET_INIT"]
    objective_target="EXISTENCE"
    show_mode=2
    if config_model is None:
        config_model={}
    else:
        config_model=dict(config_model)
    config_model.setdefault("constant_bits", constant_bits)
    config_model.setdefault("filename", str(FILES_DIR / f"{cipher.name}_INTEGRAL_TWOSUBSET_milp_model.lp"))
    if config_solver is None:
        config_solver={"solver": "DEFAULT", "solution_number": 1, "OutputFlag": 0}

    # Search for the integral distinguisher
    distinguishers = attacks.integral_attacks(cipher, goal=goal, constraints=constraints, objective_target=objective_target, show_mode=show_mode, config_model=config_model, config_solver=config_solver)
    for distinguisher in distinguishers:
        print(f"[TEST] Balanced bits: {distinguisher.data['balanced_bits']}")
    return distinguishers


if __name__ == "__main__":
    # import primitives.aes as aes
    # cipher = aes.AES_BLOCKCIPHER(version=[128,256])

    # import primitives.speck as speck
    # cipher = speck.SPECK_PERMUTATION(version=32)
    # cipher = speck.SPECK_BLOCKCIPHER(version=[32,64])

    # import primitives.simon as simon
    # cipher = simon.SIMON_BLOCKCIPHER(version=[32,64])

    # test_all_implementations(cipher)
    # test_visualisation(cipher)
    # test_diff_attack_milp(cipher)
    # test_diff_attack_sat(cipher)
    # test_linear_attack_milp(cipher)
    # test_linear_attack_sat(cipher)

    # PRESENT integral attack example. This MILP example is slower than the default demos.
    import primitives.present as present
    present_cipher = present.PRESENT_PERMUTATION(r=9)
    present_config_model={
        "model_params": {"PRESENT_Sbox": {"tool_type": "polyhedron"}},
        "filename": str(FILES_DIR / "9round_PRESENT_PERM_INTEGRAL_TWOSUBSET_milp_model.lp"),
    }
    test_integral_attack_milp(present_cipher, [60, 61, 62, 63], config_model=present_config_model)
