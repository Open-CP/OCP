import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.salsa import SALSA_PERMUTATION, SALSA_KEYPERMUTATION
from OCP import test_all_implementations, test_visualisation


def test_imp_salsa_permutation():
    cipher = SALSA_PERMUTATION(r=None)

    test_all_implementations(cipher)

    test_visualisation(cipher)


def test_imp_salsa_keypermutation():
    cipher = SALSA_KEYPERMUTATION(r=None)

    test_all_implementations(cipher)

    test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_salsa_permutation()
    test_imp_salsa_keypermutation()

    print("All implementation tests completed!")
