import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.rectangle import RECTANGLE_PERMUTATION, RECTANGLE_BLOCKCIPHER
from OCP import test_python_unrolled_imp


def test_imp_rectangle_permutation():
    cipher = RECTANGLE_PERMUTATION(r=None)

    test_python_unrolled_imp(cipher)


def test_imp_rectangle_blockcipher():
    RECTANGLE_BLOCKCIPHER_VERSIONS = [[64, 80], [64, 128]]

    for version in RECTANGLE_BLOCKCIPHER_VERSIONS:
        cipher = RECTANGLE_BLOCKCIPHER(r=None, version=version)

        test_python_unrolled_imp(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_rectangle_permutation()
    test_imp_rectangle_blockcipher()

    print("All implementation tests completed!")
