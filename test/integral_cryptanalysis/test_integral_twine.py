import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.twine import TWINE_PERMUTATION
import attacks.attacks as attacks


def test_16_round_twine_twosubset_integral():
    r = 16
    cipher = TWINE_PERMUTATION(r=r)
    config_model = {
        "constant_bits": [0],
        "model_params": {"TWINE_Sbox": {"tool_type": "polyhedron"}},
        "filename": str(ROOT / "files" / "16round_TWINE_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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

    expected_balanced_bits = [
        4, 5, 6, 7, 12, 13, 14, 15,
        20, 21, 22, 23, 28, 29, 30, 31,
        36, 37, 38, 39, 44, 45, 46, 47,
        52, 53, 54, 55, 60, 61, 62, 63,
    ]
    assert len(distinguishers) == 1
    assert distinguishers[0].data["status"] == "found"
    assert distinguishers[0].data["balanced_bits"] == expected_balanced_bits
    print(f"[TEST] Successfully found the {r}-round TWINE integral distinguisher using MILP.")


if __name__ == "__main__":
    test_16_round_twine_twosubset_integral()
