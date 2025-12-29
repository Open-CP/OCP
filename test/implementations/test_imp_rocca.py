import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.rocca import ROCCA_AD_PERMUTATION
from OCP import test_all_implementations, test_visualisation


def test_imp_rocca_permutation():
    represent_mode = [0, 1]

    for rm in represent_mode:
        print("represent_mode:", rm)
        cipher = ROCCA_AD_PERMUTATION(r=5, represent_mode=rm)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_rocca_permutation()

    print("All implementation tests completed!")
