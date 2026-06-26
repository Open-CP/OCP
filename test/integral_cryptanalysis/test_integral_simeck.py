import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.simeck import SIMECK_PERMUTATION
import attacks.attacks as attacks


# The rounds and balanced output positions follow Appendix E.6-E.8 of Xiang et al.
SIMECK_INTEGRAL_TEST_CASES = [
    # {
    #     "name": "Simeck32-14r-c0",
    #     "version": 32,
    #     "rounds": 14,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": [16, 17, 21, 22, 26, 27, 31],
    # },
    # {
    #     "name": "Simeck48-17r-c0",
    #     "version": 48,
    #     "rounds": 17,
    #     "constant_bits": {0},
    #     "expected_balanced_bits": [24, 28, 29, 43, 44],
    # },
    {
        "name": "Simeck64-20r-c0",
        "version": 64,
        "rounds": 20,
        "constant_bits": {0},
        "expected_balanced_bits": [32, 33, 37, 59, 63],
    },
]


def run_simeck_twosubset_integral(test_case):
    version = test_case["version"]
    rounds = test_case["rounds"]
    constant_bits = test_case["constant_bits"]
    expected_balanced_bits = test_case["expected_balanced_bits"]
    cipher = SIMECK_PERMUTATION(r=rounds, version=version)
    config_model = {
        "constant_bits": sorted(constant_bits),
        "filename": str(ROOT / "files" / f"{rounds}round_SIMECK{version}_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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
    print(f"[TEST] Successfully found the {rounds}-round SIMECK{version} integral distinguisher using MILP.")


def test_simeck_twosubset_integral():
    for test_case in SIMECK_INTEGRAL_TEST_CASES:
        run_simeck_twosubset_integral(test_case)


if __name__ == "__main__":
    test_simeck_twosubset_integral()
