import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.operators import CopyOperator
import tools.milp_search as milp_search
import solving.solving as solving

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


def gen_operator(bitsize=2, output_count=2):
    print("\n********************* operation: CopyOperator ********************* ")
    my_input = [var.Variable(bitsize, ID="in")]
    my_output = [var.Variable(bitsize, ID="out" + str(i)) for i in range(output_count)]
    op = CopyOperator(my_input, my_output, ID="Copy")
    op.display()
    return op


def test_milp_model(op):
    model_versions = [op.__class__.__name__+"_INTEGRAL_TWOSUBSET"]
    for model_v in model_versions:
        op.model_version = model_v
        milp_constraints = op.generate_model(model_type='milp')
        print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
        filename = str(FILES_DIR / f"milp_{op.ID}_{model_v}.lp")
        model = milp_search.write_milp_model(constraints=milp_constraints, filename=filename)
        sol_list = solving.solve_milp(filename, {"solution_number": 100000})
        print(f"All solutions:\n{sol_list}\n number of solutions: {len(sol_list)}")


def test_copy(bitsize, output_count):
    op = gen_operator(bitsize=bitsize, output_count=output_count)

    test_milp_model(op)


if __name__ == "__main__":
    print("=== Implementation Test Log ===")

    test_copy(bitsize=1, output_count=2)

    # test_copy(bitsize=2, output_count=3)

    print("All implementation tests completed!")
