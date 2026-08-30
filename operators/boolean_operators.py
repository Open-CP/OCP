"""Bitwise boolean operators: AND, OR, XOR, N_XOR (n-ary XOR), NOT, ConstantXOR, and ANDXOR.

Each provides python/C/Verilog code generation and SAT/MILP model generation across the
differential / linear / truncated / integral model versions.
"""

from operators.operators import Operator, BinaryOperator, UnaryOperator, RaiseExceptionVersionNotExisting, raise_unknown_implementation_type, raise_unknown_model_type
from tools.operator_constraints import gen_xor_constraints, gen_word_xor_constraints, gen_nxor_constraints, gen_word_nxor_constraints, binary_declaration, integer_declaration, gen_equivalence_constraints, gen_or_constraints, gen_implication_constraints


class AND(BinaryOperator):
    """Bitwise AND of the two input variables into the output."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "LINEAR", "INTEGRAL_TWOSUBSET"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        lhs, left, right = self.get_var_ID('out', 0, unroll), self.get_var_ID('in', 0, unroll), self.get_var_ID('in', 1, unroll)
        if implementation_type == 'python':
            return [f"{lhs} = {left} & {right}"]
        if implementation_type == 'c':
            return [f"{lhs} = {left} & {right};"]
        if implementation_type == 'verilog':
            return [f"assign {lhs} = {left} & {right};"]
        raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        var_in1, var_in2, var_out = self.get_var_model("in", 0), self.get_var_model("in", 1), self.get_var_model("out", 0)
        var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
        if model_type in ('sat', 'milp'):
            # Differential: the weight bit is active iff either input difference is (p = i1 OR i2),
            # and the output difference can only be active where p is (o -> p).
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                for i1, i2, o, p in zip(var_in1, var_in2, var_out, var_p):
                    model_list.extend(gen_or_constraints(i1, i2, p, model_type))
                    model_list.extend(gen_implication_constraints(o, p, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, var_p))
                self.weight = [" + ".join(var_p)] if model_type == 'milp' else var_p
                return model_list
            # Linear: each input mask implies the weight bit, which equals the output mask (p = o).
            if self.model_version == self.__class__.__name__ + "_LINEAR":
                for i1, i2, o, p in zip(var_in1, var_in2, var_out, var_p):
                    model_list.extend(gen_implication_constraints(i1, p, model_type))
                    model_list.extend(gen_implication_constraints(i2, p, model_type))
                    model_list.extend(gen_equivalence_constraints([p], [o], model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, var_p))
                self.weight = [" + ".join(var_p)] if model_type == 'milp' else var_p
                return model_list
            # Two-subset integral: the output division-property bit is i1 OR i2.
            if model_type == 'milp' and self.model_version == self.__class__.__name__ + "_INTEGRAL_TWOSUBSET":
                for i1, i2, o in zip(var_in1, var_in2, var_out):
                    model_list.extend(gen_or_constraints(i1, i2, o, model_type))
                model_list.append(binary_declaration(var_in1, var_in2, var_out))
                return model_list
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class OR(BinaryOperator):
    """Bitwise OR of the two input variables into the output."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "LINEAR"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        lhs, left, right = self.get_var_ID('out', 0, unroll), self.get_var_ID('in', 0, unroll), self.get_var_ID('in', 1, unroll)
        if implementation_type == 'python':
            return [f"{lhs} = {left} | {right}"]
        if implementation_type == 'c':
            return [f"{lhs} = {left} | {right};"]
        if implementation_type == 'verilog':
            return [f"assign {lhs} = {left} | {right};"]
        raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        var_in1, var_in2, var_out = self.get_var_model("in", 0), self.get_var_model("in", 1), self.get_var_model("out", 0)
        var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
        if model_type in ('sat', 'milp'):
            # OR shares AND's activity-propagation model (support behaves the same under XOR / masks).
            # Differential: p = i1 OR i2, and o -> p.
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                for i1, i2, o, p in zip(var_in1, var_in2, var_out, var_p):
                    model_list.extend(gen_or_constraints(i1, i2, p, model_type))
                    model_list.extend(gen_implication_constraints(o, p, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, var_p))
                self.weight = [" + ".join(var_p)] if model_type == 'milp' else var_p
                return model_list
            # Linear: each input mask implies the weight bit, which equals the output mask (p = o).
            if self.model_version == self.__class__.__name__ + "_LINEAR":
                for i1, i2, o, p in zip(var_in1, var_in2, var_out, var_p):
                    model_list.extend(gen_implication_constraints(i1, p, model_type))
                    model_list.extend(gen_implication_constraints(i2, p, model_type))
                    model_list.extend(gen_equivalence_constraints([p], [o], model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, var_p))
                self.weight = [" + ".join(var_p)] if model_type == 'milp' else var_p
                return model_list
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class XOR(BinaryOperator):
    """Bitwise XOR of the two input variables into the output."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "XORDIFF_1", "XORDIFF_2", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDDIFF_1", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "XORDIFF_1", "XORDIFF_2", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDDIFF_1", "TRUNCATEDLINEAR", "INTEGRAL_TWOSUBSET"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        lhs, left, right = self.get_var_ID('out', 0, unroll), self.get_var_ID('in', 0, unroll), self.get_var_ID('in', 1, unroll)
        if implementation_type == 'python':
            return [f"{lhs} = {left} ^ {right}"]
        if implementation_type == 'c':
            return [f"{lhs} = {left} ^ {right};"]
        if implementation_type == 'verilog':
            return [f"assign {lhs} = {left} ^ {right};"]
        raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        if model_type in ['sat', 'milp']:
            # Modeling for differential cryptanalysis
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1", self.__class__.__name__ + "_XORDIFF_2"]:
                var_in1, var_in2, var_out = self.get_var_model("in", 0), self.get_var_model("in", 1), self.get_var_model("out", 0)
                version = int(t) if (t := self.model_version.rsplit('_', 1)[-1]).isdigit() else 0
                dummies = []
                for i in range(len(var_in1)):
                    if model_type == 'milp' and version in [1, 2]:
                        d = self.ID + '_d_' + str(i)
                        dummies.append(d)
                    else:
                        d = None
                    model_list.extend(gen_xor_constraints(var_in1[i], var_in2[i], var_out[i], model_type, v_dummy=d, version=version))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, dummies))
                return model_list
            # Modeling for word truncated differential cryptanalysis
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_1"]:
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                version = int(t) if (t := self.model_version.rsplit('_', 1)[-1]).isdigit() else 0
                if model_type == 'milp' and version in [1]:
                    d = self.ID + '_d'
                else:
                    d = None
                model_list.extend(gen_word_xor_constraints(var_in1[0], var_in2[0], var_out[0], model_type, v_dummy=d, version=version))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out, [d] if d else []))
                return model_list
            # Modeling for linear cryptanalysis: both input masks equal the output mask.
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = self.get_var_model("in", 0), self.get_var_model("in", 1), self.get_var_model("out", 0)
                model_list.extend(gen_equivalence_constraints(var_in1, var_out, model_type))
                model_list.extend(gen_equivalence_constraints(var_in2, var_out, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out))
                return model_list
            # Modeling for two-subset integral cryptanalysis (MILP only)
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_INTEGRAL_TWOSUBSET":
                var_in1, var_in2, var_out = self.get_var_model("in", 0), self.get_var_model("in", 1), self.get_var_model("out", 0)
                for i1, i2, o in zip(var_in1, var_in2, var_out):
                    model_list += [f'{i1} + {i2} - {o} = 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_out))
                return model_list
            # Modeling for word truncated linear cryptanalysis: both input activities equal the output.
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list.extend(gen_equivalence_constraints(var_in1, var_out, model_type))
                model_list.extend(gen_equivalence_constraints(var_in2, var_out, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in1, var_in2, var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class N_XOR(Operator):
    """N-ary XOR of all input variables into the output (a_0 xor a_1 xor ... xor a_n = b)."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "XORDIFF_1", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "XORDIFF_1", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        expression = " ^ ".join(self.get_var_ID('in', i, unroll) for i in range(len(self.input_vars)))
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + expression]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + expression + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + expression + ';']
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        if model_type in ['sat', 'milp']:
            # Modeling for differential cryptanalysis
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1"]:
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                all_in = sum(var_in, [])  # every input bit, for the Binary declaration
                var_in = [list(group) for group in zip(*var_in)]
                version = int(t) if (t := self.model_version.rsplit('_', 1)[-1]).isdigit() else 0
                integer_dummies, binary_dummies = [], []
                for i in range(self.input_vars[0].bitsize):
                    if model_type == 'milp' and version in [0]:
                        d = self.ID + '_d_' + str(i)
                        integer_dummies.append(d)  # version 0 uses a single integer dummy per bit
                    elif model_type == 'milp' and version in [1]:
                        d = [f"{self.ID}_d_{i}_{j}" for j in range(int((len(self.input_vars)+1)/2))]
                        binary_dummies.extend(d)  # version 1 uses binary dummies
                    else:
                        d = None
                    model_list.extend(gen_nxor_constraints(var_in[i], var_out[i], model_type, v_dummy=d, version=version))
                if model_type == 'milp':
                    model_list.append(binary_declaration(all_in, var_out, binary_dummies))
                    if integer_dummies:
                        model_list.append(integer_declaration(integer_dummies))
                return model_list
            # Modeling for word truncated differential cryptanalysis
            elif len(self.input_vars) >= 2 and self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":  # Reference: Related-Key Differential Analysis of the AES.
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                inputs = [iv[0] for iv in var_in]
                output = var_out[0]
                model_list.extend(gen_word_nxor_constraints(inputs, output, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(inputs, [output]))
                return model_list
            # Modeling for linear cryptanalysis: every input mask equals the output mask.
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                for input_bits in var_in:
                    model_list.extend(gen_equivalence_constraints(var_out, input_bits, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(sum(var_in, []), var_out))
                return model_list
            # Modeling for word truncated linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                for input_word in var_in:
                    model_list.extend(gen_equivalence_constraints(var_out, input_word, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(sum(var_in, []), var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class NOT(UnaryOperator):
    """Bitwise NOT of the input variable into the output."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "LINEAR"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ~' + self.get_var_ID('in', 0, unroll) + ';']
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        if model_type in ('sat', 'milp'):
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = gen_equivalence_constraints(var_in, var_out, model_type)
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class ConstantXOR(UnaryOperator):
    """XOR a round constant into the input variable (constant addition via XOR)."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
    }

    def __init__(self, input_vars, output_vars, constant_table, round = 0, index = 0, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.table = constant_table
        self.table_r, self.table_i = round, index

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        if unroll==True: my_constant=hex(self.table[self.table_r-1][self.table_i])
        else: my_constant=f"RC[i][{self.table_i}]"
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant.replace("//", "/") + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant + ';']
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python':
            return [f"#Constraints List\nRC={self.table}"]
        elif implementation_type == 'c':
            bit_size = max(max(row) for row in self.table).bit_length()
            var_def_c = 'uint8_t' if bit_size <= 8 else "uint32_t" if bit_size <= 32 else "uint64_t" if bit_size <= 64 else "uint128_t"
            return [f"// Constraints List\n{var_def_c} RC[][{len(self.table[0])}] = {{\n    " + ", ".join("{ " + ", ".join(map(str, row)) + " }" for row in self.table) + "\n};"]
        elif implementation_type == 'verilog':
            bit_size = max(max(row) for row in self.table).bit_length()
            return [f"// Constraints List\nreg [{bit_size-1}:0] RC [0:{len(self.table)-1}][0:{len(self.table[0])-1}];", "initial begin"] + [f"    RC[{i}][{j}] = {bit_size}'h{self.table[i][j]:X};" for i in range(len(self.table)) for j in range(len(self.table[0]))] + ["end"]
        else: return None

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        if model_type in ('sat', 'milp'):
            # Each branch only selects the variable granularity; the equivalence and the single
            # Binary declaration are emitted once below.
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = self.get_var_model("in", 0), self.get_var_model("out", 0)
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False)
            else:
                RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
            model_list = gen_equivalence_constraints(var_in, var_out, model_type)
            if model_type == 'milp':
                model_list.append(binary_declaration(var_in, var_out))
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class ANDXOR(Operator):
    """Bitwise AND-XOR: out = (in0 & in1) ^ in2."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "XORDIFF_1", "XORDIFF_2", "XORDIFF_3", "LINEAR", "LINEAR_1"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll) + ';']
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        var_in1, var_in2, var_in3, var_out = (
            self.get_var_model("in", 0),
            self.get_var_model("in", 1),
            self.get_var_model("in", 2),
            self.get_var_model("out", 0),
        )
        var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
        if model_type == 'sat':
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [
                        f'{i1} {i2} -{p}',
                        f'{i1} {i2} -{i3} {o}',
                        f'-{i1} {p}',
                        f'{i1} {i2} {i3} -{o}',
                        f'-{i2} {p}',
                    ]
                self.weight = var_p
                return model_list
            if self.model_version == self.__class__.__name__ + "_LINEAR":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'-{i3} {p}', f'-{i1} {p}', f'-{i2} {p}', f'{i3} -{o}', f'{o} -{p}']
                self.weight = var_p
                return model_list
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{i1} + {i2} - {p} >= 0', f'{i1} + {i2} + {i3} - {o} >= 0', f'{i1} + {i2} - {i3} + {o} >= 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_1":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{i1} + {i2} - {p} >= 0', f'{o} - {i3} + {p} >= 0', f'{i3} - {o} + {p} >= 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_2":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{p} = 0 -> {i1} = 0', f'{p} = 0 -> {i2} = 0', f'{p} = 0 -> {i3} - {o} = 0', f'{p} = 1 -> {i1} + {i2} >= 1']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_3":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{p} = 0 -> {i1} = 0', f'{p} = 0 -> {i2} = 0', f'{p} = 0 -> {i3} - {o} = 0', f'{p} - {i1} - {i2} <= 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{p} - {i3} >= 0', f'{i3} - {o} >= 0', f'{o} - {p} >= 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR_1":
                for i1, i2, i3, o, p in zip(var_in1, var_in2, var_in3, var_out, var_p):
                    model_list += [f'{i3} - {i2} >= 0', f'{o} - {i3} >= 0', f'{i3} - {i1} >= 0', f'{i3} - {o} >= 0', f'{p} - {i3} >= 0', f'{i3} - {p} >= 0']
                model_list.append(binary_declaration(var_in1, var_in2, var_in3, var_out, var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)

    @staticmethod
    def _eval_bit(x):  # x = in0 || in1 || in2
        in0 = (x >> 2) & 1
        in1 = (x >> 1) & 1
        in2 = x & 1
        return (in0 & in1) ^ in2

    @staticmethod
    def bit_andxor_ddt():
        ddt = [[0 for _ in range(2)] for _ in range(8)]
        for dx in range(8):
            for x in range(8):
                dy = ANDXOR._eval_bit(x) ^ ANDXOR._eval_bit(x ^ dx)
                ddt[dx][dy] += 1
        return ddt

    @staticmethod
    def bit_andxor_lat():
        def parity(x):
            return bin(x).count("1") & 1

        lat = [[0 for _ in range(2)] for _ in range(8)]
        for a in range(8):
            for b in range(2):
                total = 0
                for x in range(8):
                    exponent = parity(a & x) ^ parity(b & ANDXOR._eval_bit(x))
                    total += 1 if exponent == 0 else -1
                lat[a][b] = total
        return lat

    @staticmethod
    def bit_andxor_diff_truth_table():
        ddt = ANDXOR.bit_andxor_ddt()
        ttable = ''
        for n in range(1 << 5):
            dx = n >> 2
            dy = (n >> 1) & 1
            wbit = n & 1
            count = ddt[dx][dy]
            ttable += '1' if (count == 8 and wbit == 0) or (count == 4 and wbit == 1) else '0'
        return ttable

    @staticmethod
    def bit_andxor_linear_truth_table():
        lat = ANDXOR.bit_andxor_lat()
        ttable = ''
        for n in range(1 << 5):
            mx = n >> 2
            my = (n >> 1) & 1
            wbit = n & 1
            count = abs(lat[mx][my])
            ttable += '1' if (count == 8 and wbit == 0) or (count == 4 and wbit == 1) else '0'
        return ttable
