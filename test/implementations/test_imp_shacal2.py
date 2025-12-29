import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # this file -> implementation -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from primitives.shacal2 import SHACAL2_BLOCKCIPHER
from OCP import test_all_implementations, test_visualisation


def test_imp_shacal2():

    versions = [[256, 512]]

    for version in versions:

        cipher = SHACAL2_BLOCKCIPHER(r=64, version=version, represent_mode=0)

        test_all_implementations(cipher)

        test_visualisation(cipher)


if __name__ == "__main__":
    print(f"=== Implementation Test Log ===")

    test_imp_shacal2()

    print("All implementation tests completed!")
