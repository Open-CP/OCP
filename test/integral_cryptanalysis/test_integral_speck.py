import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.speck import SPECK_PERMUTATION
import attacks.attacks as attacks


# The test case follows the SPECK support code in
# knowledge/support code/Sun et al - MILP-Aided Bit-Based Division Property for ARX-Based Block Cipher.
SPECK_INTEGRAL_TEST_CASES = [
    # {
    #     "name": "Speck32-6r-c26",
    #     "version": 32,
    #     "rounds": 6,
    #     "constant_bits": {26},
    #     "expected_balanced_bits": [15],
    # },
    # {
    #     "name": "Speck48-6r-c40-41-42",
    #     "version": 48,
    #     "rounds": 6,
    #     "constant_bits": {40, 41, 42},
    #     "expected_balanced_bits": [23],
    # },
    # {
    #     "name": "Speck64-6r-c56-57-58",
    #     "version": 64,
    #     "rounds": 6,
    #     "constant_bits": {56, 57, 58},
    #     "expected_balanced_bits": [31],
    # },
    # {
    #     "name": "Speck96-6r-c88-89-90",
    #     "version": 96,
    #     "rounds": 6,
    #     "constant_bits": {88, 89, 90},
    #     "expected_balanced_bits": [47],
    # },
    {
        "name": "Speck128-6r-c120-121-122",
        "version": 128,
        "rounds": 6,
        "constant_bits": {120, 121, 122},
        "expected_balanced_bits": [63],
    },
]


def run_speck_twosubset_integral(test_case):
    version = test_case["version"]
    rounds = test_case["rounds"]
    constant_bits = test_case["constant_bits"]
    expected_balanced_bits = test_case["expected_balanced_bits"]
    cipher = SPECK_PERMUTATION(r=rounds, version=version)
    config_model = {
        "constant_bits": sorted(constant_bits),
        "filename": str(ROOT / "files" / f"{rounds}round_SPECK{version}_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
    }

    distinguishers = attacks.integral_attacks(
        cipher,
        goal="INTEGRAL_TWOSUBSET",
        constraints=["TWO_SUBSET_INIT"],
        objective_target="EXISTENCE",
        show_mode=2,
        config_model=config_model,
        config_solver={"solver": "DEFAULT", "solution_number": 1, "OutputFlag": 0},
    )

    assert len(distinguishers) == 1
    assert distinguishers[0].data["status"] == "found"
    assert distinguishers[0].data["balanced_bits"] == expected_balanced_bits
    print(f"[TEST] Successfully found the {rounds}-round SPECK{version} integral distinguisher using MILP.")


def test_speck_twosubset_integral():
    for test_case in SPECK_INTEGRAL_TEST_CASES:
        run_speck_twosubset_integral(test_case)


if __name__ == "__main__":
    test_speck_twosubset_integral()
