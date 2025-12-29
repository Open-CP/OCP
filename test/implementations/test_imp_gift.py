import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.gift import GIFT_PERMUTATION, GIFT_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_gift_permutation():
    GIFT_PERMUTATION_VERSIONS = [64, 128]

    for version in GIFT_PERMUTATION_VERSIONS:

        cipher = GIFT_PERMUTATION(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


def test_imp_gift_blockcipher():
    GIFT_BLOCKCIPHER_VERSIONS = [[64, 128], [128, 128]]

    for version in GIFT_BLOCKCIPHER_VERSIONS:
        cipher = GIFT_BLOCKCIPHER(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_gift_permutation()
    test_imp_gift_blockcipher()

    print("All implementation tests completed!")
