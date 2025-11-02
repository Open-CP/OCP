import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.Sbox import Sbox, ASCON_Sbox, Skinny_4bit_Sbox, Skinny_8bit_Sbox, GIFT_Sbox, AES_Sbox, TWINE_Sbox, PRESENT_Sbox, KNOT_Sbox
import tools.milp_search as milp_search
import tools.sat_search as sat_search
import solving.solving as solving

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


# Define the S-box, each has two types of representations
present_sbox = PRESENT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="present_sbox")
present_sbox2 = PRESENT_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(4)]], [[var.Variable(1,ID="out"+str(i)) for i in range(4)]], ID="present_sbox2")

knot_sbox = KNOT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="knot_sbox")
knot_sbox2 = KNOT_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(4)]], [[var.Variable(1,ID="out"+str(i)) for i in range(4)]], ID="knot_sbox2")

twine_sbox = TWINE_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="twine_sbox")
twine_sbox2 = TWINE_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(4)]], [[var.Variable(1,ID="out"+str(i)) for i in range(4)]], ID="twine_sbox2")

ascon_sbox = ASCON_Sbox([var.Variable(5,ID="in"+str(i)) for i in range(1)], [var.Variable(5,ID="out"+str(i)) for i in range(1)], ID="ascon_sbox")
ascon_sbox2 = ASCON_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(5)]], [[var.Variable(1,ID="out"+str(i)) for i in range(5)]], ID="ascon_sbox2")

skinny4_sbox = Skinny_4bit_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="skinny4_sbox")
skinny4_sbox2 = Skinny_4bit_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(4)]], [[var.Variable(1,ID="out"+str(i)) for i in range(4)]], ID="skinny4_sbox2")

skinny8_sbox = Skinny_8bit_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="skinny8_sbox")
skinny8_sbox2 = Skinny_8bit_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(8)]], [[var.Variable(1,ID="out"+str(i)) for i in range(8)]], ID="skinny8_sbox2")

gift_sbox = GIFT_Sbox([var.Variable(4,ID="in"+str(i)) for i in range(1)], [var.Variable(4,ID="out"+str(i)) for i in range(1)], ID="gift_sbox")
gift_sbox2 = GIFT_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(4)]], [[var.Variable(1,ID="out"+str(i)) for i in range(4)]], ID="gift_sbox2")

aes_sbox = AES_Sbox([var.Variable(8,ID="in"+str(i)) for i in range(1)], [var.Variable(8,ID="out"+str(i)) for i in range(1)], ID="aes_sbox")
aes_sbox2 = AES_Sbox([[var.Variable(1,ID="in"+str(i)) for i in range(8)]], [[var.Variable(1,ID="out"+str(i)) for i in range(8)]], ID="aes_sbox2")


test_Sbox = Sbox([var.Variable(3,ID="in"+str(i)) for i in range(1)], [var.Variable(3,ID="out"+str(i)) for i in range(1)], 3,3,ID="test_sbox")
test_Sbox.table = [7,6,0,4,2,5,1,3]


def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


def test_milp_model(op, model_version, mode=0, tool_type="minimize_logic"):
    op.model_version = model_version
    milp_constraints = op.generate_model(model_type='milp', mode=mode, tool_type=tool_type)
    print(f"MILP constraints with model_version={model_version}: \n", "\n".join(milp_constraints))
    filename = str(FILES_DIR / f"milp_{op.ID}_{model_version}.lp")
    if hasattr(op, 'weight'):
        obj_fun=op.weight
    else:
        obj_fun=None
    model = milp_search.write_milp_model(constraints=milp_constraints, obj_fun=obj_fun, filename=filename)
    sol_list = solving.solve_milp(filename, {"solution_number": 100000})
    # print(f"All solutions:\n{sol_list}\n")
    return sol_list


def test_sat_model(op, model_version, mode=0, tool_type="minimize_logic"):
    op.model_version = model_version
    sat_constraints = op.generate_model(model_type='sat', mode=mode, tool_type=tool_type)
    print(f"SAT constraints with model_version={model_version}: \n", "\n".join(sat_constraints))
    filename = str(FILES_DIR / f"sat_{op.ID}_{model_version}.cnf")
    model = sat_search.write_sat_model(constraints=sat_constraints, filename=filename)
    print("variable_map in sat:\n", model["variable_map"])
    sol_list = solving.solve_sat(filename, model["variable_map"], {"solution_number": 100000})
    # print(f"All solutions:\n{sol_list}")
    return sol_list


def test_Sbox_solutions_milp(Sbox, solutions):
    if "DIFF" in Sbox.model_version:
        table = Sbox.computeDDT()
    elif "LINEAR" in Sbox.model_version:
        table = Sbox.computeLAT()
    nonzero_count = sum(1 for row in table for val in row if val != 0)
    if nonzero_count != len(solutions):
        raise ValueError(f"Mismatch: non-zero count in DDT/LAT = {nonzero_count}, but model.SolCount = {len(solutions)}")
    input_variables = [Sbox.get_var_ID('in', 0, True) + '_' + str(i) for i in range(Sbox.input_vars[0].bitsize)]
    output_variables = [Sbox.get_var_ID('out', 0, True) + '_' + str(i) for i in range(Sbox.input_vars[0].bitsize)]
    for solution in solutions:
        row, column = '', ''
        for i in range(Sbox.input_bitsize):
            row += str(int(round(solution[input_variables[i]])))
        for i in range(Sbox.output_bitsize):
            column += str(int(round(solution[output_variables[i]])))
        row = int(''.join(str(bit) for bit in row), 2)
        column = int(''.join(str(bit) for bit in column), 2)
        if Sbox.model_version in [Sbox.__class__.__name__ + "_XORDIFF", Sbox.__class__.__name__+"_XORDIFF_P", Sbox.__class__.__name__ + "_LINEAR", Sbox.__class__.__name__+"_LINEAR_P"]:
            if table[row][column] == 0:
                raise ValueError(f"*****************solution is wrong: row={row}, column={column}, pr={pr} != {table[row][column]}")
        elif Sbox.model_version in [Sbox.__class__.__name__ + "_XORDIFF_PR", Sbox.__class__.__name__ + "_LINEAR_PR"]:
            diff_weights = Sbox.gen_weights(table)
            pr_vars, obj_fun = Sbox._gen_model_pr_variables_objective_fun_milp()
            pr_variables = [Sbox.ID + "_" + p for p in pr_vars]
            pr = 0
            for i in range(len(pr_variables)):
                pr += diff_weights[i] * int(round(solution[pr_variables[i]]))
            if abs(float(math.log(abs(table[row][column])/(2**Sbox.input_bitsize), 2))) != pr:
                raise ValueError(f"*****************solution is wrong: row={row}, column={column}, pr={pr}, table={table[row][column]}")
    print(f"All solutions returned from MILP have been checked and are consistent with the DDT/LAT of S-box")


def test_Sbox_solutions_sat(Sbox, solutions):
    if "DIFF" in Sbox.model_version:
        table = Sbox.computeDDT()
    elif "LINEAR" in Sbox.model_version:
        table = Sbox.computeLAT()
    nonzero_count = sum(1 for row in table for val in row if val != 0)
    if nonzero_count != len(solutions):
        raise ValueError(f"Mismatch: non-zero count in DDT = {nonzero_count}, but model.SolCount = {len(solutions)}")
    input_variables = [Sbox.get_var_ID('in', 0, True) + '_' + str(i) for i in range(Sbox.input_vars[0].bitsize)]
    output_variables = [Sbox.get_var_ID('out', 0, True) + '_' + str(i) for i in range(Sbox.input_vars[0].bitsize)]
    for solution in solutions:
        row, column = '', ''
        for i in range(Sbox.input_bitsize):
            row += str(int(round(solution[input_variables[i]])))
        for i in range(Sbox.output_bitsize):
            column += str(int(round(solution[output_variables[i]])))
        row = int(''.join(str(bit) for bit in row), 2)
        column = int(''.join(str(bit) for bit in column), 2)
        if Sbox.model_version in [Sbox.__class__.__name__ + "_XORDIFF", Sbox.__class__.__name__+"_XORDIFF_P", Sbox.__class__.__name__ + "_LINEAR", Sbox.__class__.__name__+"_LINEAR_P"]:
            if table[row][column] == 0:
                raise ValueError(f"*****************solution is wrong: row={row}, column={column}, pr={pr} != {table[row][column]}")
        elif Sbox.model_version in [Sbox.__class__.__name__ + "_XORDIFF_PR", Sbox.__class__.__name__ + "_LINEAR_PR"]:
            pr_vars, obj_fun = Sbox._gen_model_pr_variables_objective_fun_sat()
            pr_variables = [Sbox.ID + "_" + p for p in pr_vars]
            integers, floats = Sbox.gen_integer_float_weight(table)
            pr = 0
            for i in range(max(integers)):
                pr += solution[pr_variables[i]]
            for i in range(max(integers), max(integers)+len(floats)):
                if solution[pr_variables[i]] == 1:
                    pr += floats[i-max(integers)]
            if abs(float(math.log(abs(table[row][column])/(2**Sbox.input_bitsize), 2))) != pr:
                raise ValueError(f"*****************solution is wrong: row={row}, column={column}, pr={pr}, ddt={table[row][column]}")
    print(f"All solutions returned from SAT have been checked and are consistent with the DDT of S-box")


def test_sbox(sbox):
    print(f"\n********************* operation: {sbox.ID} Sbox ********************* ")
    sbox.display()

    test_implementation(sbox)

    # 1. MILP models and tests for differential cryptanalysis
    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF_PR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF_PR", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF_A", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_XORDIFF_A", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)


    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF_1")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF_A")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF_A_1")


    # 2. MILP models and tests for linear cryptanalysis
    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR_PR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR_PR", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR_A", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_milp(sbox, sol_list)
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_LINEAR_A", tool_type="polyhedron")
    test_Sbox_solutions_milp(sbox, sol_list)

    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR_1")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR_A")
    sol_list = test_milp_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR_A_1")


    # 3. SAT models and tests for differential cryptanalysis
    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_XORDIFF", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_XORDIFF_PR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_XORDIFF_A", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)


    sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF")
    sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_TRUNCATEDDIFF_A")

    # 4. SAT models and tests for linear cryptanalysis
    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_LINEAR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_LINEAR_PR", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)

    for mode in [0,1,2]:
        sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_LINEAR_A", mode=mode, tool_type="minimize_logic")
        test_Sbox_solutions_sat(sbox, sol_list)

    sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR")
    sol_list = test_sat_model(sbox, sbox.__class__.__name__+"_TRUNCATEDLINEAR_A")


if __name__ == '__main__':

    print(f"=== Implementation Test Log ===")

    test_sbox(present_sbox)

    print("All implementation tests completed!")
