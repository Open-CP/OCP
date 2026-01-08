import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.skinny import SKINNY_PERMUTATION, SKINNY_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_skinny_permutation():
    SKINNY_PERMUTATION_VERSIONS = [64, 128]

    for version in SKINNY_PERMUTATION_VERSIONS:

        cipher = SKINNY_PERMUTATION(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


def test_imp_skinny_blockcipher():
    SKINNY_BLOCKCIPHER_VERSIONS = [[64, 64], [64, 128], [64, 192], [128, 128], [128, 256], [128, 384]]
    from OCP import test_python_unrolled_imp
    for version in SKINNY_BLOCKCIPHER_VERSIONS:
        cipher = SKINNY_BLOCKCIPHER(r=None, version=version)

        test_python_unrolled_imp(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    #test_imp_skinny_permutation()
    test_imp_skinny_blockcipher()

    print("All implementation tests completed!")
