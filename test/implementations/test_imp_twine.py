import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.twine import TWINE_PERMUTATION, TWINE_BLOCKCIPHER
from OCP import test_python_unrolled_imp


def test_imp_twine_permutation():
    cipher = TWINE_PERMUTATION(r=None)

    test_python_unrolled_imp(cipher)


def test_imp_twine_blockcipher():
    TWINE_BLOCKCIPHER_VERSIONS = [[64, 80], [64, 128]]

    for version in TWINE_BLOCKCIPHER_VERSIONS:
        cipher = TWINE_BLOCKCIPHER(r=None, version=version)

        test_python_unrolled_imp(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_twine_permutation()
    test_imp_twine_blockcipher()

    print("All implementation tests completed!")
