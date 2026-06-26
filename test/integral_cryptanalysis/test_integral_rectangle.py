import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.rectangle import RECTANGLE_PERMUTATION
import attacks.attacks as attacks


def test_9_round_rectangle_twosubset_integral():
    r = 9
    cipher = RECTANGLE_PERMUTATION(r=r)
    config_model = {
        "constant_bits": [0, 16, 32, 48],
        "model_params": {"RECTANGLE_Sbox": {"tool_type": "polyhedron"}},
        "filename": str(ROOT / "files" / "9round_RECTANGLE_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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
    assert distinguishers[0].data["balanced_bits"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 17, 19]
    print(f"[TEST] Successfully found the {r}-round RECTANGLE integral distinguisher using MILP.")


if __name__ == "__main__":
    test_9_round_rectangle_twosubset_integral()
