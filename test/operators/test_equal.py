import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.operators import Equal
import tools.milp_search as milp_search
import tools.sat_search as sat_search
import solving.solving as solving

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


def gen_operator(bitsize=2):
    print("\n********************* operation: Equal ********************* ")
    my_input, my_output = [var.Variable(bitsize, ID="in")], [var.Variable(bitsize, ID="out")]
    op = Equal(my_input, my_output, ID='Equal')
    op.display()
    return op


def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


def test_milp_model(op):
    model_versions = [op.__class__.__name__+"_XORDIFF", op.__class__.__name__+"_TRUNCATEDDIFF", op.__class__.__name__+"_LINEAR", op.__class__.__name__+"_TRUNCATEDLINEAR"]
    for model_v in model_versions:
        op.model_version = model_v
        milp_constraints = op.generate_model(model_type='milp')
        print(f"MILP constraints with model_version={model_v}: \n", "\n".join(milp_constraints))
        filename = str(FILES_DIR / f"milp_{op.ID}_{model_v}.lp")
        model = milp_search.write_milp_model(constraints=milp_constraints, filename=filename)
        sol_list = solving.solve_milp(filename, {"solution_number": 100000})
        print(f"All solutions:\n{sol_list}\n number of solutions: {len(sol_list)}")


def test_sat_model(op):
    model_versions = [op.__class__.__name__+"_XORDIFF", op.__class__.__name__+"_TRUNCATEDDIFF", op.__class__.__name__+"_LINEAR", op.__class__.__name__+"_TRUNCATEDLINEAR"]
    for model_v in model_versions:
        op.model_version = model_v
        sat_constraints = op.generate_model(model_type='sat')
        print(f"SAT constraints with model_version={model_v}: \n", "\n".join(sat_constraints))
        filename = str(FILES_DIR / f"sat_{op.ID}_{model_v}.cnf")
        model = sat_search.write_sat_model(constraints=sat_constraints, filename=filename)
        print("variable_map in sat:\n", model["variable_map"])
        sol_list = solving.solve_sat(filename, model["variable_map"], {"solution_number": 100000})
        print(f"All solutions:\n{sol_list}")

def test_equal_twosubset(op):
    op.model_version = op.__class__.__name__ + "_INTEGRAL_TWOSUBSET"
    milp_constraints = op.generate_model(model_type='milp')
    var_in, var_out = op.get_var_model("in", 0), op.get_var_model("out", 0)
    expected_constraints = [f"{vin} - {vout} = 0" for vin, vout in zip(var_in, var_out)]
    assert milp_constraints[:-1] == expected_constraints
    assert milp_constraints[-1] == 'Binary\n' + ' '.join(v for v in var_in + var_out)

    try:
        op.generate_model(model_type='sat')
    except Exception as exc:
        assert "not existing for sat" in str(exc)
    else:
        raise AssertionError("Equal_INTEGRAL_TWOSUBSET must reject non-MILP model_type")

    print(f"MILP constraints with model_version={op.model_version}: \n", "\n".join(milp_constraints))



def test_equal(bitsize):

    op = gen_operator(bitsize=bitsize)

    test_implementation(op)

    test_equal_twosubset(op)

    test_milp_model(op)

    test_sat_model(op)


if __name__ == '__main__':
    print(f"=== Implementation Test Log ===")

    test_equal(bitsize=1)

    test_equal(bitsize=2)

    print("All implementation tests completed!")
