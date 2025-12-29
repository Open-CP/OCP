import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.present import PRESENT_PERMUTATION, PRESENT_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_present_permutation():
    cipher = PRESENT_PERMUTATION(r=None)

    test_all_implementations(cipher)

    test_visualisation(cipher)


def test_imp_present_blockcipher():
    PRESENT_BLOCKCIPHER_VERSIONS = [[64, 80], [64, 128]]

    for version in PRESENT_BLOCKCIPHER_VERSIONS:
        cipher = PRESENT_BLOCKCIPHER(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    # test_imp_present_permutation()
    test_imp_present_blockcipher()

    print("All implementation tests completed!")
