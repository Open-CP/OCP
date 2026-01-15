import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.matrix import Matrix
import tools.milp_search as milp_search
import tools.sat_search as sat_search
import solving.solving as solving

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


def gen_aes_matrix_operator():
    # representation 1: AES's polynomial matrix
    my_input, my_output = [var.Variable(8,ID="in"+str(i)) for i in range(4)], [var.Variable(8,ID="out"+str(i)) for i in range(4)]
    mat_aes = [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]]
    print(my_input)
    matrix_aes = Matrix("mat_aes", my_input, my_output, mat = mat_aes, polynomial="0x1b", ID = 'Matrix_AES')
    return matrix_aes


def gen_skinny_matrix_operator():
    # representation 1: SKINNY's 4x4 matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = Matrix("mat_skinny", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    matrix_skinny64.display()
    return matrix_skinny64


def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


def test_milp_model(op, model_versions):
    for model_v in model_versions:
        op.model_version = model_v
        milp_constraints = op.generate_model(model_type='milp')
        print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
        filename = str(FILES_DIR / f"milp_{op.ID}_{model_v}.lp")
        model = milp_search.write_milp_model(constraints=milp_constraints, filename=filename)
        sol_list = solving.solve_milp(filename, {"solution_number": 100000})
        print(f"Number of solutions: {len(sol_list)}")


def test_sat_model(op, model_versions):
    for model_v in model_versions:
        op.model_version = model_v
        sat_constraints = op.generate_model(model_type='sat')
        print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
        filename = str(FILES_DIR / f"sat_{op.ID}_{model_v}.cnf")
        model = sat_search.write_sat_model(constraints=sat_constraints, filename=filename)
        print("variable_map in sat:\n", model["variable_map"])
        sol_list = solving.solve_sat(filename, model["variable_map"], {"solution_number": 100000})
        print(f"Number of solutions: {len(sol_list)}")



def test_aes_matrix():

    op = gen_aes_matrix_operator()

    test_implementation(op)

    test_milp_model(op, [op.__class__.__name__+"_TRUNCATEDDIFF", op.__class__.__name__+"_TRUNCATEDDIFF_1", op.__class__.__name__+"_XORDIFF"])

    test_sat_model(op, [op.__class__.__name__+"_XORDIFF", op.__class__.__name__+"_LINEAR"])


def test_skinny_matrix():

    op = gen_skinny_matrix_operator()

    test_implementation(op)

    test_milp_model(op, [op.__class__.__name__+"_TRUNCATEDDIFF_2", op.__class__.__name__+"_XORDIFF"])

    test_sat_model(op, [op.__class__.__name__+"_XORDIFF", op.__class__.__name__+"_LINEAR"])


if __name__ == '__main__':
    print(f"=== Implementation Test Log ===")

    test_aes_matrix()

    test_skinny_matrix()

    print("All implementation tests completed!")
