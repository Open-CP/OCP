from operators.operators import Operator, BinaryOperator, UnaryOperator, RaiseExceptionVersionNotExisting
from tools.model_constraints import gen_xor_constraints, gen_word_xor_constraints, gen_nxor_constraints, gen_word_nxor_constraints

class AND(BinaryOperator):  # Operator for the bitwise AND operation: compute the bitwise AND on the two input variables towards the output variable
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} {i2} -{p}', f'-{i1} {p}', f'-{i2} {p}']
                self.weight = var_p
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{p} -{i1}', f'{p} -{i2}', f'{p} -{o}', f'-{p} {o}']
                self.weight = var_p
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} + {i2} - {o} >= 0', f'{i1} + {i2} - {p} >= 0', f'- {i1} + {p} >= 0', f'- {i2} + {p} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{p} - {o} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class OR(BinaryOperator):  # Operator for the bitwise OR operation: compute the bitwise OR on the two input variables towards the output variable
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' | ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c':
           return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' | ' + self.get_var_ID('in', 1, unroll) + ';']
        elif implementation_type == 'verilog':
           return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' | ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} {i2} -{p}', f'-{i1} {p}', f'-{i2} {p}']
                self.weight = var_p
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{p} -{i1}', f'{p} -{i2}', f'{p} -{o}', f'-{p} {o}']
                self.weight = var_p
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version == self.__class__.__name__+"_XORDIFF":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} + {i2} - {o} >= 0', f'{i1} + {i2} - {p} >= 0',  f'- {i1} + {p} >= 0', f'- {i2} + {p} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{p} - {o} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class XOR(BinaryOperator):  # Operator for the bitwise XOR operation: compute the bitwise XOR on the two input variables towards the output variable
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type in ['sat', 'milp']:
            # Modeling for differential cryptanalysis
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1", self.__class__.__name__ + "_XORDIFF_2"]:
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                version = int(t) if (t := self.model_version.rsplit('_', 1)[-1]).isdigit() else 0
                for i in range(len(var_in1)):
                    if model_type == 'milp' and version in [1, 2]:
                        d = self.ID + '_d_' + str(i)
                    else:
                        d = None
                    model_list.extend(gen_xor_constraints(var_in1[i], var_in2[i], var_out[i], model_type, v_dummy=d, version=version))
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
                return model_list
            # Modeling for linear cryptanalysis
            elif model_type == 'sat' and self.model_version in [self.__class__.__name__ + "_LINEAR"]:
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):
                    i1, i2, o = var_in1[i],var_in2[i],var_out[i]
                    model_list += [f'{i1} -{o}', f'-{i1} {o}', f'{i2} -{o}', f'-{i2} {o}']
                return model_list
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):
                    i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                    model_list += [f'{i1} - {o} = 0', f'{i2} - {o} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            # Modeling for word truncated linear cryptanalysis
            elif model_type == 'sat' and self.model_version in [self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f'{var_in1[0]} -{var_out[0]}', f'-{var_in1[0]} {var_out[0]}', f'{var_in2[0]} -{var_out[0]}', f'-{var_in2[0]} {var_out[0]}']
                return model_list
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list += [f'{var_in1[0]} - {var_out[0]} = 0', f'{var_in2[0]} - {var_out[0]} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class N_XOR(Operator): # Operator of the n-xor: a_0 xor a_1 xor ... xor a_n = b
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            expression = ' ^ '.join(self.get_var_ID('in', i, unroll) for i in range(len(self.input_vars)))
            return [self.get_var_ID('out', 0, unroll) + ' = ' + expression]
        elif implementation_type == 'c':
            expression_parts = []
            for i in range(len(self.input_vars)):
                expression_parts.append(self.get_var_ID('in', i, unroll))
            expression = ' ^ '.join(expression_parts)
            return [self.get_var_ID('out', 0, unroll) + ' = ' + expression + ';']
        elif implementation_type == 'verilog':
            expression_parts = []
            for i in range(len(self.input_vars)):
                expression_parts.append(self.get_var_ID('in', i, unroll))
            expression = ' ^ '.join(expression_parts)
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + expression + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type in ['sat', 'milp']:
            # Modeling for differential cryptanalysis
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1"]:
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                var_in = [list(group) for group in zip(*var_in)]
                version = int(t) if (t := self.model_version.rsplit('_', 1)[-1]).isdigit() else 0
                for i in range(self.input_vars[0].bitsize):
                    if model_type == 'milp' and version in [0]:
                        d = self.ID + '_d_' + str(i)
                    elif model_type == 'milp' and version in [1]:
                        d = [f"{self.ID}_d_{i}_{j}" for j in range(int((len(self.input_vars)+1)/2))]                
                    else:
                        d = None
                    model_list.extend(gen_nxor_constraints(var_in[i], var_out[i], model_type, v_dummy=d, version=version))
                return model_list
            # Modeling for word truncated differential cryptanalysis
            elif model_type == "milp" and len(self.input_vars) >= 2 and self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":  # Reference: Related-Key Differential Analysis of the AES.
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                inputs = [iv[0] for iv in var_in]
                output = var_out[0]
                model_list.extend(gen_word_nxor_constraints(inputs, output, model_type))
                return model_list
            # Modeling for linear cryptanalysis
            elif model_type == "sat" and self.model_version in [self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                for i in range(self.input_vars[0].bitsize):
                    for j in range(len(var_in)):
                        model_list += [f"{var_out[i]} -{var_in[j][i]}", f"-{var_out[i]} {var_in[j][i]}"]
                return model_list
            elif model_type == "milp" and self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                for i in range(self.input_vars[0].bitsize):
                    for j in range(len(var_in)):
                        model_list += [f"{var_out[i]} - {var_in[j][i]} = 0"]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + var_out))
                return model_list
            # Modeling for word truncated linear cryptanalysis
            elif model_type == "sat" and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                for j in range(len(var_in)):
                    model_list += [f"{var_out[0]} -{var_in[j][0]}", f"-{var_out[0]} {var_in[j][0]}"]
                return model_list
            elif model_type == "milp" and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                for j in range(len(var_in)):
                    model_list += [f"{var_out[0]} - {var_in[j][0]} = 0"]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class NOT(UnaryOperator): # Operator for the bitwise NOT operation: compute the bitwise NOT operation on the input variable towards the output variable
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ~' + self.get_var_ID('in', 0, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        if model_type == 'sat':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class ConstantXOR(UnaryOperator): # Operator for the constant addition using xor, to incorporate the constant with value "constant" to the input variable and result is stored in the output variable
    def __init__(self, input_vars, output_vars, constant_table, round = 0, index = 0, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.table = constant_table
        self.table_r, self.table_i = round, index

    def generate_implementation(self, implementation_type='python', unroll=False):
        if unroll==True: my_constant=hex(self.table[self.table_r-1][self.table_i])
        else: my_constant=f"RC[i][{self.table_i}]"
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant.replace("//", "/") + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

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
        if model_type == 'sat':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f'{var_in[0]} - {var_out[0]} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class ANDXOR(Operator):  # Operator for the bitwise AND-XOR operation: compute the bitwise AND then XOR on the three input variables towards the output variable
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ') ^ ' + self.get_var_ID('in', 2, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1", self.__class__.__name__ + "_XORDIFF_2", self.__class__.__name__ + "_XORDIFF_3"]:
                var_in1, var_in2, var_in3, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("in", 2), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, i3, o, p = var_in1[i], var_in2[i], var_in3[i], var_out[i], var_p[i]
                    if self.model_version in [self.__class__.__name__ + "_XORDIFF"]:
                        model_list += [f'{p} - {i1} >= 0', f'{p} - {i2} >= 0', f'{p} - {i1} - {i2} <= 0', f'{i1} + {i2} + {i3} - {o} >= 0', f'{i1} + {i2} - {i3} + {o} >= 0']
                    elif self.model_version == self.__class__.__name__ + "_XORDIFF_1":
                        model_list += [f'{p} = 0 -> {i1} = 0', f'{p} = 0 -> {i2} = 0', f'{p} = 1 -> {i1} + {i2} >= 1', f'{i1} + {i2} + {i3} - {o} >= 0', f'{i1} + {i2} - {i3} + {o} >= 0']
                    elif self.model_version == self.__class__.__name__ + "_XORDIFF_2":
                        model_list += [f'{p} = 0 -> {i1} = 0', f'{p} = 0 -> {i2} = 0', f'{p} = 0 -> {i3} - {o} = 0', f'{p} = 1 -> {i1} + {i2} >= 1']
                    elif self.model_version == self.__class__.__name__ + "_XORDIFF_3":
                        model_list += [f'{p} = 0 -> {i1} = 0', f'{p} = 0 -> {i2} = 0', f'{p} = 0 -> {i3} - {o} = 0', f'{p} - {i1} - {i2} <= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_in3 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
