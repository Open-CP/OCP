import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.aes import AES_PERMUTATION, AES_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation

# Note: For AES, only the *unrolled* implementation is currently supported (and verified against the NIST test vectors). 
# The rolled implementation is not yet correct for AES because the rolled code generator emits a single round body inside a loop, which assumes every round is structurally identical and indexes round-dependent data by the loop counter. 
# Hence test_all_implementations is expected to pass only on the unrolled output.

def test_imp_aes_permutation():
    cipher = AES_PERMUTATION(r=None)

    test_all_implementations(cipher)

    test_visualisation(cipher)


def test_imp_aes_blockcipher():
    AES_BLOCKCIPHER_VERSIONS = [[128, 128], [128, 192], [128, 256]]

    for version in AES_BLOCKCIPHER_VERSIONS:
        cipher = AES_BLOCKCIPHER(r=None, version=version)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_aes_permutation()
    test_imp_aes_blockcipher()

    print("All implementation tests completed!")
