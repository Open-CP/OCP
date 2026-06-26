import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from primitives.lblock import LBLOCK_PERMUTATION
import attacks.attacks as attacks


def test_16_round_lblock_twosubset_integral():
    r = 16
    cipher = LBLOCK_PERMUTATION(r=r)
    config_model = {
        "constant_bits": [0],
        "model_params": {"LBlock_Sbox" + str(i): {"tool_type": "polyhedron"} for i in range(8)},
        "filename": str(ROOT / "files" / "16round_LBLOCK_PERM_INTEGRAL_TWOSUBSET_E2E_milp_model.lp"),
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
    assert distinguishers[0].data["balanced_bits"] == list(range(32, 64))
    print(f"[TEST] Successfully found the {r}-round LBlock integral distinguisher using MILP.")


if __name__ == "__main__":
    test_16_round_lblock_twosubset_integral()
