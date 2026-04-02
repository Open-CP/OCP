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
    matrix_aes = Matrix("aes_matrix", my_input, my_output, mat = mat_aes, polynomial="0x1b", ID = 'Matrix_AES')
    return matrix_aes


def gen_skinny_matrix_operator():
    # representation 1: SKINNY's 4x4 matrix
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(4)], [var.Variable(4,ID="out"+str(i)) for i in range(4)]
    mat_skinny64 = [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]]
    matrix_skinny64 = Matrix("skinny_matrix", my_input, my_output, mat = mat_skinny64, ID = 'Matrix_SKINNY64')
    return matrix_skinny64


def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


def test_milp_model(op, model_version, tool_type="minimize_logic", branch_num=None):
    op.model_version = model_version
    milp_constraints = op.generate_model(model_type='milp', tool_type=tool_type, branch_num=branch_num)
    print(f"MILP constraints with model_version={model_version}: \n", "\n".join(milp_constraints))
    filename = str(FILES_DIR / f"milp_{op.ID}_{model_version}.lp")
    model = milp_search.write_milp_model(constraints=milp_constraints,filename=filename)
    sol_list = solving.solve_milp(filename, {"solution_number": 100000})
    print(f"Number of solutions: {len(sol_list)}")
    for sol in sol_list:
        print(sol)
    return sol_list

def test_sat_model(op, model_version, tool_type="minimize_logic"):
    op.model_version = model_version
    sat_constraints = op.generate_model(model_type='sat', tool_type=tool_type)
    print(f"SAT constraints with model_version={model_version}: \n", "\n".join(sat_constraints))
    filename = str(FILES_DIR / f"sat_{op.ID}_{model_version}.cnf")
    model = sat_search.write_sat_model(constraints=sat_constraints, filename=filename)
    print("variable_map in sat:\n", model["variable_map"])
    sol_list = solving.solve_sat(filename, model["variable_map"], {"solution_number": 100000})
    print(f"Number of solutions: {len(sol_list)}")
    for sol in sol_list:
        print(sol)


def test_matrix(op):
    op.display()
    test_implementation(op)

    model_versions = [op.__class__.__name__+"_XORDIFF",
                      op.__class__.__name__+"_LINEAR"]
    for model_version in model_versions:
        test_milp_model(op, model_version)
        test_sat_model(op, model_version)

    model_versions = [op.__class__.__name__+"_TRUNCATEDDIFF",
                      op.__class__.__name__+"_TRUNCATEDDIFF_1",
                      op.__class__.__name__+"_TRUNCATEDDIFF_2",
                      op.__class__.__name__+"_TRUNCATEDLINEAR",
                      op.__class__.__name__+"_TRUNCATEDLINEAR_1",
                      op.__class__.__name__+"_TRUNCATEDLINEAR_2"
                      ]
    for model_version in model_versions:
        for tool_type in ["minimize_logic", "minimize_logic_espresso", "polyhedron"]:
            test_milp_model(op, model_version, tool_type=tool_type)
            if tool_type == "polyhedron":
                continue  # skip SAT for polyhedron since it's not supported
            test_sat_model(op, model_version, tool_type=tool_type)


if __name__ == '__main__':
    print(f"=== Implementation Test Log ===")

    aes_matrix = gen_aes_matrix_operator()

    skinny_matrix = gen_skinny_matrix_operator()

    test_matrix(aes_matrix)

    test_matrix(skinny_matrix)

    print("All implementation tests completed!")
