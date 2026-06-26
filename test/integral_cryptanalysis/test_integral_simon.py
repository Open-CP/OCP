import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.simon import SIMON_PERMUTATION
import attacks.attacks as attacks


# The rounds and balanced output positions follow Appendix E.1-E.5 of Xiang et al.
SIMON_INTEGRAL_TEST_CASES = [
    {
        "name": "Simon32-13r-c0",
        "version": 32,
        "rounds": 13,
        "constant_bits": {0},
        "expected_balanced_bits": list(range(16, 32)),
    },
    # {
    #     "name": "Simon48-15r-c0",
    #     "version": 48,
    #     "rounds": 15,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": list(range(24, 48)),
    # },
    # {
    #     "name": "Simon64-17r-c0",
    #     "version": 64,
    #     "rounds": 17,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": [
    #         32, 33, 34, 35, 36, 37, 38, 39,
    #         40, 41, 42, 43, 44, 50, 56, 57,
    #         58, 59, 60, 61, 62, 63,
    #     ],
    # },
    # {
    #     "name": "Simon96-21r-c0",
    #     "version": 96,
    #     "rounds": 21,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": [48, 50, 55, 89, 94],
    # },
    # {
    #     "name": "Simon128-25r-c0",
    #     "version": 128,
    #     "rounds": 25,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": [64, 66, 126],
    # },
]


def run_simon_twosubset_integral(test_case):
    version = test_case["version"]
    rounds = test_case["rounds"]
    constant_bits = test_case["constant_bits"]
    expected_balanced_bits = test_case["expected_balanced_bits"]
    cipher = SIMON_PERMUTATION(r=rounds, version=version)
    config_model = {
        "constant_bits": sorted(constant_bits),
        "filename": str(ROOT / "files" / f"{rounds}round_SIMON{version}_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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
    print(f"[TEST] Successfully found the {rounds}-round SIMON{version} integral distinguisher using MILP.")


def test_simon_twosubset_integral():
    for test_case in SIMON_INTEGRAL_TEST_CASES:
        run_simon_twosubset_integral(test_case)


if __name__ == "__main__":
    test_simon_twosubset_integral()
