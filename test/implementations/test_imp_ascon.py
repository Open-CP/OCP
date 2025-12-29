import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.ascon import ASCON_PERMUTATION
from OCP import test_all_implementations, test_visualisation


def test_imp_ascon_permutation():
    rounds = [3,12]

    for r in rounds:
        print(f"\nTesting ASCON permutation with {r} rounds:")

        for represent_mode in [0, 1]:
            print(f"  represent_mode = {represent_mode}")

            cipher = ASCON_PERMUTATION(r=r, represent_mode=represent_mode)

            test_all_implementations(cipher)

            test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_ascon_permutation()

    print("All implementation tests completed!")
