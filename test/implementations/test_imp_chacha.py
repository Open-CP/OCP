import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.chacha import CHACHA_PERMUTATION, CHACHA_KEYPERMUTATION
import implementations.implementations as imp
import visualisations.visualisations as vis

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


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


def test_imp_chacha_permutation():
    cipher = CHACHA_PERMUTATION(r=None)

    test_python_imp(cipher)

    test_python_unrolled_imp(cipher)

    test_c_imp(cipher)

    test_c_unrolled_imp(cipher)

    # test_verilog_imp(cipher) # TO DO

    # test_verilog_unrolled_imp(cipher) # TO DO

    test_visualisation(cipher)


def test_imp_chacha_keypermutation():
    cipher = CHACHA_KEYPERMUTATION(r=None)

    test_python_imp(cipher)

    test_python_unrolled_imp(cipher)

    test_c_imp(cipher)

    test_c_unrolled_imp(cipher)
    
    # test_verilog_imp(cipher) # TO DO

    # test_verilog_unrolled_imp(cipher) # TO DO

    test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")
        
    test_imp_chacha_permutation()

    test_imp_chacha_keypermutation()

    print("All implementation tests completed!")