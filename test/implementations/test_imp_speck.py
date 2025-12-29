import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.speck import SPECK_PERMUTATION, SPECK_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_speck_permutation():
    SPECK_PERMUTATION_VERSIONS = [32, 48, 64, 96, 128]

    for version in SPECK_PERMUTATION_VERSIONS:

        cipher = SPECK_PERMUTATION(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


def test_imp_speck_blockcipher():
    SPECK_BLOCKCIPHER_VERSIONS = [[32, 64], [48, 72], [48, 96], [64, 96], [64, 128], [96, 96], [96, 144], [128, 128], [128, 192], [128, 256]]

    for version in SPECK_BLOCKCIPHER_VERSIONS:
        cipher = SPECK_BLOCKCIPHER(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_speck_permutation()
    test_imp_speck_blockcipher()

    print("All implementation tests completed!")
