import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.simeck import SIMECK_PERMUTATION, SIMECK_BLOCKCIPHER
from OCP import test_python_unrolled_imp, test_visualisation


def test_imp_simeck_permutation():
    cipher = SIMECK_PERMUTATION(r=None, version=32)

    test_python_unrolled_imp(cipher)

    test_visualisation(cipher)


def test_imp_simeck_blockcipher():
    cipher = SIMECK_BLOCKCIPHER(r=None, version=[32,64])

    test_python_unrolled_imp(cipher)

    test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_simeck_permutation()
    test_imp_simeck_blockcipher()

    print("All implementation tests completed!")
