import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> differential cryptanalysis -> test -> <ROOT>
sys.path.insert(0, str(ROOT))
from primitives.speck import SPECK_PERMUTATION, SPECK_BLOCKCIPHER
import attacks.attacks as attacks
from tools.model_constraints import gen_predefined_constraints


# Example: Test known differential trail for 9-round SPECK-32.
# Reference: Differential Analysis of Block Ciphers SIMON and SPECK (Table 6). https://eprint.iacr.org/2014/922.pdf
def test_9_round_speck_32_trail():
    r = 9
    weight = 30
    cipher = SPECK_PERMUTATION(r=r, version = 32)

    def gen_trail_constraints(model_type):
        test_trail = [[0x8054, 0xA900], [0, 0xA402], [0xA402, 0x3408], [0x50C0, 0x80E0], [0x0181, 0x0203], [0x000c, 0x0800], [0x2000, 0], [0x40, 0x40], [0x8040, 0x8140], [0x0040, 0x0542]]
        constraints = []
        for i in range(r):
            XL = bin(test_trail[i][0])[2:].zfill(16)
            XR = bin(test_trail[i][1])[2:].zfill(16)
            for j in range(16):
                constraints += gen_predefined_constraints(model_type, "EXACTLY", [f"v_{i+1}_0_0_{j}"], int(XL[j]), bitwise=False)
                constraints += gen_predefined_constraints(model_type, "EXACTLY", [f"v_{i+1}_0_1_{j}"], int(XR[j]), bitwise=False)
        XL = bin(test_trail[r][0])[2:].zfill(16)
        XR = bin(test_trail[r][1])[2:].zfill(16)
        for j in range(16):
            constraints += gen_predefined_constraints(model_type, "EXACTLY", [f"v_{r}_4_0_{j}"], int(XL[j]), bitwise=False)
            constraints += gen_predefined_constraints(model_type, "EXACTLY", [f"v_{r}_4_1_{j}"], int(XR[j]), bitwise=False)
        return constraints

    milp_trail_constraints = gen_trail_constraints("milp")
    trails = attacks.diff_attacks(cipher, objective_target="EXISTENCE", constraints=milp_trail_constraints, show_mode=2)
    if len(trails) > 0 and trails[0].data['diff_weight'] == weight:
        print(f"[TEST] Successfully found the specified trail for {r}-round SPECK-32 using MILP.")
    else:
        print(f"[TEST] Found {len(trails)} Trail, with the weight = {trails[0].data['diff_weight']}")


    sat_trail_constraints = gen_trail_constraints("sat")
    trails = attacks.diff_attacks(cipher, objective_target="EXISTENCE", constraints=sat_trail_constraints, show_mode=2, config_model={"model_type": "sat"})
    if len(trails) > 0 and trails[0].data['diff_weight'] == weight:
        print(f"[TEST] Successfully found the specified trail for {r}-round SPECK-32 using SAT.")
    else:
        print(f"[TEST] Found {len(trails)} Trail, with the weight = {trails[0].data['diff_weight']}")


if __name__ == '__main__':

    test_9_round_speck_32_trail()
