import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # this file -> test -> <ROOT>
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import variables.variables as var
from operators.modular_operators import ConstantAdd


def gen_operator(bitsize, table, round, index, modulo = None):
    print("\n********************* operation: ConstantAdd ********************* ")
    my_input, my_output = [var.Variable(bitsize,ID="in")], [var.Variable(bitsize,ID="out")]
    op = ConstantAdd(my_input, my_output, table, round=round, index=index, modulo=modulo, ID = 'ConstantAdd')
    op.display()
    return op


def test_implementation(op):
    code = op.generate_implementation(implementation_type="python", unroll=True)
    print(f"python code with unroll=True: \n", "\n".join(code))

    code = op.generate_implementation(implementation_type="c", unroll=True)
    print(f"c code with unroll=True: \n", "\n".join(code))


if __name__ == '__main__':
    print(f"=== Implementation Test Log ===")

    op = gen_operator(bitsize=1, table=[[1, 2], [3, 4]], round = 1, index = 0, modulo = None) # modular addition with 1 and modulo 2^n-1=1
    test_implementation(op)

    op = gen_operator(bitsize=2, table=[[1, 2], [3, 4]], round = 2, index = 0, modulo = None) # modular addition with 3 and modulo 2^n-1=3
    test_implementation(op)

    op = gen_operator(bitsize=3, table=[[1, 2], [3, 4]], round = 2, index = 1, modulo = 5) # modular addition with 4 and modulo 5
    test_implementation(op)

    print("All implementation tests completed!")
