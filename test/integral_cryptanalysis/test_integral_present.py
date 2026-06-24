import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.present import PRESENT_PERMUTATION
import attacks.attacks as attacks


# Example: Test the known 9-round two-subset integral distinguisher for PRESENT.
# OCP uses its own bit order for PRESENT variables; the constant bits and
# balanced bit below are written in that order.
def test_9_round_present_twosubset_integral():
    r = 9
    cipher = PRESENT_PERMUTATION(r=r)
    config_model = {
        "constant_bits": [60, 61, 62, 63],
        "model_params": {"PRESENT_Sbox": {"tool_type": "polyhedron"}},
        "filename": str(ROOT / "files" / "9round_PRESENT_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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
    assert distinguishers[0].data["balanced_bits"] == [63]
    print(f"[TEST] Successfully found the {r}-round PRESENT two-subset integral distinguisher using MILP.")


if __name__ == "__main__":
    test_9_round_present_twosubset_integral()
