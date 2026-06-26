import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.lblock import LBLOCK_PERMUTATION, LBLOCK_BLOCKCIPHER
from OCP import test_python_unrolled_imp


def test_imp_lblock_permutation():
    cipher = LBLOCK_PERMUTATION(r=None)

    test_python_unrolled_imp(cipher)


def test_imp_lblock_blockcipher():
    cipher = LBLOCK_BLOCKCIPHER(r=None, version=[64,80])

    test_python_unrolled_imp(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_lblock_permutation()
    test_imp_lblock_blockcipher()

    print("All implementation tests completed!")
