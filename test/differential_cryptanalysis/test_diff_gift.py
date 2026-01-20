import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> differential cryptanalysis -> test -> <ROOT>
sys.path.insert(0, str(ROOT))
from primitives.gift import GIFT_PERMUTATION, GIFT_BLOCKCIPHER
import attacks.attacks as attacks
from tools.model_constraints import gen_predefined_constraints


# Example: Test known 12-round differential trail for GIFT64.
# Reference: Huaifeng Chen, Rui Zong, Xiaoyang Dong. Improved Differential Attacks on GIFT-64 (Table 3). https://link.springer.com/chapter/10.1007/978-3-030-41579-2_26
def test_12_round_gift64_trail():
    r = 12
    weight = 58
    cipher = GIFT_PERMUTATION(r=r, version = 64)

    def gen_trail_constraints(model_type):
        test_trail=[0x0000000600000006, 0x0000000002020000, 0x0000005000000050, 0x0000000000000202, 0x0000000500000005, 0x0000000002020000, 0x0000005000000050, 0x0000000000000202, 0x0000000500000005, 0x0000000002020000, 0x0000005000000050, 0x0000000000000202, 0x0000000500000005]
        constraints = []
        for i in range(r):
            X = bin(test_trail[i])[2:].zfill(64)

            for j in range(64):
                constraints.extend(
                    gen_predefined_constraints(
                        model_type,
                        "EXACTLY",
                        [f"v_{i+1}_0_{j}"],
                        int(X[j]),
                        bitwise=False
                    )
                )
        X = bin(test_trail[r])[2:].zfill(64)
        for j in range(64):
            constraints += gen_predefined_constraints(model_type, "EXACTLY", [f"v_{r}_3_{j}"], int(X[j]), bitwise=False)
        return constraints

    milp_trail_constraints = gen_trail_constraints("milp")
    trails = attacks.diff_attacks(cipher, objective_target="EXISTENCE", constraints=milp_trail_constraints, show_mode=2)
    if len(trails) > 0 and trails[0].data['diff_weight'] == weight:
        print(f"[TEST] Successfully found the specified trail for {r}-round GIFT-64 using MILP.")
    else:
        print(f"[TEST] Found {len(trails)} Trail, with the weight = {trails[0].data['diff_weight']}")


    sat_trail_constraints = gen_trail_constraints("sat")
    trails = attacks.diff_attacks(cipher, objective_target="EXISTENCE", constraints=sat_trail_constraints, show_mode=2, config_model={"model_type": "sat"})
    if len(trails) > 0 and trails[0].data['diff_weight'] == weight:
        print(f"[TEST] Successfully found the specified trail for {r}-round GIFT-64 using SAT.")
    else:
        print(f"[TEST] Found {len(trails)} Trail, with the weight = {trails[0].data['diff_weight']}")


def test_12_round_gift64_differential():
    r = 2
    cipher = GIFT_PERMUTATION(r=r, version = 64)
    input_diff = "0x0000000000000006"
    output_diff = "0x0060000000000000"
    # Given the input and output differences, there are totally 10 differential trails with probability >= 2^{−62}. 
    # 1 with probability 2^{−58},
    # 6 with probability 2^{−60}, 
    # 3 with probability 2^{−62}. 
    # The total probability of this differential is 2^{−56.5737}.

    # Search for differentials based on MILP
    trails = attacks.diff_attacks(cipher, goal = "DIFFERENTIAL_PROB", objective_target="AT MOST 5", config_model={"input_diff": input_diff, "output_diff": output_diff}, show_mode=2)


if __name__ == '__main__':

    test_12_round_gift64_trail()

    test_12_round_gift64_differential()