import math
from operators.operators import BinaryOperator, UnaryOperator, RaiseExceptionVersionNotExisting


class ModAdd(BinaryOperator): # Operator for the modular addition: add the two input variables together towards the output variable
                              # (optional "modulo" defines the modular value in case of a modular addition, by default it uses 2^bitsize as modular value)
    def __init__(self, input_vars, output_vars, modulo = None, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.modulo = modulo

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            if self.modulo == None:
                return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ') % ' + str(self.modulo)]
        elif implementation_type == 'c':
            if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ") & " + hex(2**self.input_vars[0].bitsize - 1) + ';']
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ') % ' + str(self.modulo) + ';']
        elif implementation_type == 'verilog':
            if self.modulo == None: return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' + ' + self.get_var_ID('in', 1, unroll) + ";"]
            else:
                raise Exception(str(self.__class__.__name__) + ": addition modulo not a power a 2 is not yet handled for verilog '")
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF"]: # Reference: Ling Sun, et al. Accelerating the Search of Differential and Linear Characteristics with the SAT Method
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                n = self.input_vars[0].bitsize
                var_p = [self.ID + '_p_' + str(i) for i in range(n - 1)]
                # Difference propagations constraints
                for i in range(n - 1):
                    a, b, c = var_in1[i], var_in2[i], var_out[i]
                    a1, b1, c1 = var_in1[i + 1], var_in2[i + 1], var_out[i + 1]
                    model_list += [f"{a} {b} -{c} {a1} {b1} {c1}",
                                   f"{a} -{b} {c} {a1} {b1} {c1}",
                                   f"-{a} {b} {c} {a1} {b1} {c1}",
                                   f"-{a} -{b} -{c} {a1} {b1} {c1}",
                                   f"{a} {b} {c} -{a1} -{b1} -{c1}",
                                   f"{a} -{b} -{c} -{a1} -{b1} -{c1}",
                                   f"-{a} {b} -{c} -{a1} -{b1} -{c1}",
                                   f"-{a} -{b} {c} -{a1} -{b1} -{c1}"]
                # Last bit constraints
                a, b, c = var_in1[-1], var_in2[-1], var_out[-1]
                model_list += [f"{a} {b} -{c}", f"{a} -{b} {c}", f"-{a} {b} {c}", f"-{a} -{b} -{c}"]
                # Weight constraints
                for i in range(n - 1):
                    a, b, c, w = var_in1[i + 1], var_in2[i + 1], var_out[i + 1], var_p[i]
                    model_list += [f"-{a} {c} {w}", f"{b} -{c} {w}", f"{a} -{b} {w}", f"{a} {b} {c} -{w}", f"-{a} -{b} -{c} -{w}"]
                self.weight = var_p
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_LINEAR": # Reference: Yunwen Liu, Qingju Wang, and Vincent Rijmen. Automatic Search of Linear Trails in ARX with Applications to SPECK and Chaskey.
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                n = self.input_vars[0].bitsize
                var_p = [self.ID + '_p_' + str(i) for i in range(n)]
                model_list = [f'-{var_p[0]}']
                if n > 1:
                    a, b, c, d = var_in1[0], var_in2[0], var_out[0], var_p[1]
                    model_list += [f'{a} {b} {c} -{d}',
                                f'{a} {b} -{c} {d}',
                                f'{a} -{b} {c} {d}',
                                f'-{a} {b} {c} {d}',
                                f'-{a} -{b} -{c} {d}',
                                f'-{a} {b} -{c} -{d}',
                                f'-{a} -{b} {c} -{d}',
                                f'{a} -{b} -{c} -{d}']
                for i in range(n-2):
                    a, b, c, d, e = var_in1[i+1], var_in2[i+1], var_out[i+1], var_p[i+1], var_p[i+2]
                    model_list += [f'-{a} {b} {c} {d} {e}',
                                   f'{a} -{b} {c} {d} {e}',
                                   f'{a} {b} -{c} {d} {e}',
                                   f'{a} {b} {c} -{d} {e}',
                                   f'{a} {b} {c} {d} -{e}',
                                   f'-{a} -{b} -{c} {d} {e}',
                                   f'-{a} -{b} {c} -{d} {e}',
                                   f'-{a} -{b} {c} {d} -{e}',
                                   f'-{a} {b} -{c} -{d} {e}',
                                   f'-{a} {b} -{c} {d} -{e}',
                                   f'-{a} {b} {c} -{d} -{e}',
                                   f'{a} -{b} -{c} -{d} {e}',
                                   f'{a} -{b} -{c} {d} -{e}',
                                   f'{a} -{b} {c} -{d} -{e}',
                                   f'{a} {b} -{c} -{d} -{e}',
                                   f'-{a} -{b} -{c} -{d} -{e}']
                for i in range(self.input_vars[0].bitsize):
                    a, b, c, d = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{a} -{c} {d}', f'-{a} {c} {d}', f'{b} -{c} {d}', f'-{b} {c} {d}']
                self.weight = var_p
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF"]: # Reference: Kai Fu, Meiqin Wang, Yinghua Guo, Siwei Sun, Lei Hu. MILP-Based Automatic Search Algorithms for Differential and Linear Trails for Speck
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p, var_d = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize-1)], [self.ID + '_d']
                for i in range(self.input_vars[0].bitsize-1) :
                    b = [var_in1[i],var_in2[i],var_out[i]]
                    a = [var_in1[i+1],var_in2[i+1],var_out[i+1]]
                    model_list += [a[1]+' - '+a[2]+' + '+var_p[i]+' >= 0']
                    model_list += [a[0]+' - '+a[1]+' + '+var_p[i]+' >= 0']
                    model_list += [a[2]+' - '+a[0]+' + '+var_p[i]+' >= 0']
                    model_list += [a[0]+' + '+a[1]+' + '+a[2]+' + '+var_p[i]+' <= 3 ']
                    model_list += [a[0]+' + '+a[1]+' + '+a[2]+' - '+var_p[i]+' >= 0 ']
                    model_list += [b[0]+' + '+b[1]+' + '+b[2]+' + '+var_p[i]+' - '+a[1]+' >= 0 ']
                    model_list += [a[1]+' + '+b[0]+' - '+b[1]+' + '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[1]+' - '+b[0]+' + '+b[1]+' + '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[0]+' + '+b[0]+' + '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[2]+' - '+b[0]+' - '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[0]+' - '+a[1]+' - '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[1]+' - '+a[1]+' - '+b[0]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[2]+' - '+a[1]+' - '+b[0]+' - '+b[1]+' + '+var_p[i]+' >= -2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' <= 2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' - 2 ' + var_d[0] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in1[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in2[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_out[-1] + ' >= 0 ']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p + var_d))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_1": # Type 1 constraint using the Indicator constraint provided by Gurobi
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p, var_d = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize-1)], [self.ID + '_d']
                for i in range(self.input_vars[0].bitsize-1) :
                    b = [var_in1[i],var_in2[i],var_out[i]]
                    a = [var_in1[i+1],var_in2[i+1],var_out[i+1]]
                    model_list += [f"{var_p[i]} = 0 -> {a[0]} - {a[1]} = 0", f"{var_p[i]} = 0 -> {a[0]} - {a[2]} = 0"]
                    model_list += [f"{var_p[i]} = 1 -> {a[0]} + {a[1]} + {a[2]} >= 1", f"{var_p[i]} = 1 -> {a[0]} + {a[1]} + {a[2]} <= 2"]
                    model_list += [b[0]+' + '+b[1]+' + '+b[2]+' + '+var_p[i]+' - '+a[1]+' >= 0 ']
                    model_list += [a[1]+' + '+b[0]+' - '+b[1]+' + '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[1]+' - '+b[0]+' + '+b[1]+' + '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[0]+' + '+b[0]+' + '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= 0 ']
                    model_list += [a[2]+' - '+b[0]+' - '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[0]+' - '+a[1]+' - '+b[1]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[1]+' - '+a[1]+' - '+b[0]+' - '+b[2]+' + '+var_p[i]+' >= -2 ']
                    model_list += [b[2]+' - '+a[1]+' - '+b[0]+' - '+b[1]+' + '+var_p[i]+' >= -2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' <= 2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' - 2 ' + var_d[0] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in1[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in2[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_out[-1] + ' >= 0 ']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p + var_d))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_2": # Type 2 constraint using the Indicator constraint provided by Gurobi
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p, var_d = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize-1)], [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(self.input_vars[0].bitsize-1) :
                    b = [var_in1[i],var_in2[i],var_out[i]]
                    a = [var_in1[i+1],var_in2[i+1],var_out[i+1]]
                    model_list += [a[1]+' - '+a[2]+' + '+var_p[i]+' >= 0']
                    model_list += [a[0]+' - '+a[1]+' + '+var_p[i]+' >= 0']
                    model_list += [a[2]+' - '+a[0]+' + '+var_p[i]+' >= 0']
                    model_list += [a[0]+' + '+a[1]+' + '+a[2]+' + '+var_p[i]+' <= 3 ']
                    model_list += [a[0]+' + '+a[1]+' + '+a[2]+' - '+var_p[i]+' >= 0 ']
                    model_list += [f"{var_p[i]} = 0 -> {b[0]} + {b[1]} + {b[2]} - {a[2]} - 2 {var_d[i+1]} = 0"]
                    model_list += [f"{var_p[i]} = 1 -> {var_d[i+1]} = 0"]
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' <= 2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' - 2 ' + var_d[0] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in1[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in2[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_out[-1] + ' >= 0 ']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p + var_d))
                self.weight = [" + ".join(var_p)]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_3": # Type 3 constraint using the Indicator constraint provided by Gurobi
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p, var_d = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize-1)], [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(self.input_vars[0].bitsize-1) :
                    b = [var_in1[i],var_in2[i],var_out[i]]
                    a = [var_in1[i+1],var_in2[i+1],var_out[i+1]]
                    model_list += [f"{var_p[i]} = 0 -> {a[0]} - {a[1]} = 0", f"{var_p[i]} = 0 -> {a[0]} - {a[2]} = 0"]
                    model_list += [f"{var_p[i]} = 1 -> {a[0]} + {a[1]} + {a[2]} >= 1", f"{var_p[i]} = 1 -> {a[0]} + {a[1]} + {a[2]} <= 2"]
                    model_list += [f"{var_p[i]} = 0 -> {b[0]} + {b[1]} + {b[2]} - {a[2]} - 2 {var_d[i+1]} = 0"]
                    model_list += [f"{var_p[i]} = 1 -> {var_d[i+1]} = 0"]
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' <= 2 ']
                model_list += [var_in1[-1]+' + '+var_in2[-1]+' + '+var_out[-1] + ' - 2 ' + var_d[0] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in1[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_in2[-1] + ' >= 0 ']
                model_list += [var_d[0] + ' - ' + var_out[-1] + ' >= 0 ']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p + var_d))
                self.weight = [" + ".join(var_p)]
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_LINEAR": # Reference: Kai Fu, Meiqin Wang, Yinghua Guo, Siwei Sun, Lei Hu. MILP-Based Automatic Search Algorithms for Differential and Linear Trails for Speck
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize+1)]
                model_list = [f'{var_p[0]} = 0']
                for i in range(self.input_vars[0].bitsize):
                    a = [var_out[i],var_in1[i],var_in2[i]]
                    model_list += [var_p[i]+' - '+a[0]+' - '+a[1]+' + '+a[2]+' + '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' + '+a[0]+' + '+a[1]+' - '+a[2]+' - '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' + '+a[0]+' - '+a[1]+' - '+a[2]+' + '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' - '+a[0]+' + '+a[1]+' - '+a[2]+' + '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' + '+a[0]+' - '+a[1]+' + '+a[2]+' - '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' - '+a[0]+' + '+a[1]+' + '+a[2]+' - '+var_p[i+1]+' >= 0']
                    model_list += [a[0]+' - '+var_p[i]+' + '+a[1]+' + '+a[2]+' + '+var_p[i+1]+' >= 0']
                    model_list += [var_p[i]+' + '+a[0]+' + '+a[1]+' + '+a[2]+' + '+var_p[i+1]+' <= 4']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p[:self.input_vars[0].bitsize])]
                return model_list
            else:
                RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class ModMul(BinaryOperator):  # Operator for the modular multiplication: multiply the two input variables together towards the output variable
                               # (optional "modulo" defines the modular value in case of a modular addition, by default it uses 2^bitsize as modular value)
    def __init__(self, input_vars, output_vars, modulo = None, ID = None):
        super().__init__(input_vars, output_vars, ID = ID )
        self.modulo = None
        pass # TODO

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ') % ' + str(self.modulo)]
        elif implementation_type == 'c':
            if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ';']
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ') & ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' * ' + self.get_var_ID('in', 1, unroll) + ') % ' + str(self.modulo) + ';']
        elif implementation_type == 'verilog':
            raise Exception(str(self.__class__.__name__) + ": multiplication is not yet handled for verilog '")
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class ConstantAdd(UnaryOperator): # Operator for the constant addition: use modular addition to incorporate the constant with value "constant" to the input variable and result is stored in the output variable
                                  # (optional "modulo" defines the modular value in case of a modular addition, by default it uses 2^bitsize as modular value)
    def __init__(self, input_vars, output_vars, constant_table, round = 0, index = 0, modulo = None, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.modulo = modulo
        self.table = constant_table
        self.table_r, self.table_i = round, index

    def generate_implementation(self, implementation_type='python', unroll=False):
        if unroll==True: my_constant=hex(self.table[self.table_r-1][self.table_i])
        else: my_constant=f"RC[i][{self.table_i}]"
        if implementation_type == 'python':
            if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + " + " + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') % ' + str(self.modulo)]
        elif implementation_type == 'c':
            if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ';']
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
                else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') % ' + str(self.modulo) + ';']
        elif implementation_type == 'verilog':
            if self.modulo == None: return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ';']
            else:
                if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return ["assign " + self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
                else: return ["assign " + self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') % ' + str(self.modulo) + ';']
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
        if model_type == 'sat': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
