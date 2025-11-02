import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.modular_operators import ModAdd
from operators.Sbox import Sbox
import tools.milp_search as milp_search
import tools.sat_search as sat_search
import solving.solving as solving

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


def gen_operator(bitsize=2):
    print("\n********************* operation: ModAdd ********************* ")
    my_input, my_output = [var.Variable(bitsize,ID="in"+str(i)) for i in range(2)], [var.Variable(bitsize,ID="out"+str(i)) for i in range(1)]
    op = ModAdd(my_input, my_output, ID = 'ModAdd')
    op.display()
    return op

def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


def test_milp_model(op):
    model_versions = [op.__class__.__name__+"_XORDIFF", op.__class__.__name__ + "_XORDIFF_1", op.__class__.__name__ + "_XORDIFF_2", op.__class__.__name__ + "_XORDIFF_3", op.__class__.__name__+"_LINEAR"]
    for model_v in model_versions:
        op.model_version = model_v
        milp_constraints = op.generate_model(model_type='milp')
        print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
        filename = str(FILES_DIR / f"milp_{op.ID}_{model_v}.lp")
        model = milp_search.write_milp_model(constraints=milp_constraints, obj_fun=op.weight, filename=filename)
        sol_list = solving.solve_milp(filename, {"solution_number": 100000})
        print(f"All solutions:\n{sol_list}\n")


def test_sat_model(op):
    model_versions = [op.__class__.__name__+"_XORDIFF", op.__class__.__name__+"_LINEAR"]
    for model_v in model_versions:
        op.model_version = model_v
        sat_constraints = op.generate_model(model_type='sat')
        print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
        filename = str(FILES_DIR / f"sat_{op.ID}_{model_v}.cnf")
        model = sat_search.write_sat_model(constraints=sat_constraints, filename=filename)
        print("variable_map in sat:\n", model["variable_map"])
        sol_list = solving.solve_sat(filename, model["variable_map"], {"solution_number": 100000})
        print(f"All solutions:\n{sol_list}")


def test_modadd(bitsize):

    op = gen_operator(bitsize=bitsize)

    test_implementation(op)

    test_milp_model(op)

    test_sat_model(op)


if __name__ == '__main__':
    print(f"=== Implementation Test Log ===")

    test_modadd(bitsize=1)

    test_modadd(bitsize=2)

    print("All implementation tests completed!")
