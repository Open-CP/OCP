import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.aes import AES_PERMUTATION, AES_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_aes_permutation():
    cipher = AES_PERMUTATION(r=None)

    test_all_implementations(cipher)

    test_visualisation(cipher)


def test_imp_aes_blockcipher():
    AES_BLOCKCIPHER_VERSIONS = [[128, 128], [128, 192], [128, 256]]

    for version in AES_BLOCKCIPHER_VERSIONS:
        cipher = AES_BLOCKCIPHER(r=None, version=version)
        from OCP import test_python_unrolled_imp
        test_python_unrolled_imp(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    #test_imp_aes_permutation()
    test_imp_aes_blockcipher()

    print("All implementation tests completed!")
