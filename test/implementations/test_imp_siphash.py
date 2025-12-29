import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.siphash import SIPHASH_PERMUTATION
from OCP import test_all_implementations, test_visualisation


def test_imp_siphash_permutation():
    cipher = SIPHASH_PERMUTATION(r=2)

    test_all_implementations(cipher)

    test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_siphash_permutation()

    print("All implementation tests completed!")
