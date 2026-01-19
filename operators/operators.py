from abc import ABC, abstractmethod


def RaiseExceptionVersionNotExisting(class_name, model_version, model_type):
    raise Exception(class_name + ": version " + str(model_version) + " not existing for " + model_type)


# ********************* OPERATORS ********************* #
# Class that represents a constraint/operator object, i.e. a type of node in our graph modeling (the other type being the variables)
# An Operator/Constraint node can only be linked to a Variable node in the graph representation
# Operators/Constraints are relationships between a group of variables

class Operator(ABC):
    def __init__(self, input_vars, output_vars, model_version=None, ID=None):
        self.input_vars = input_vars        # input variables of that operator
        self.output_vars = output_vars      # output variables of that operator
        self.model_version = model_version  # model version that will be used for that operator
        self.ID = ID                        # ID of the operator
        self.is_ghost = False               # indicates whether that operator is a ghost operator (i.e., an operator that has been marked as ghost during the dead-end removal process)

        # For this new operator created, update the connected_vars list for each input and output variables
        if self.__class__.__name__!="NoneOperator":
            for var_in in input_vars:
                for var_out in output_vars:
                    var_in.connected_vars.append((var_out,self,'in'))
                    var_out.connected_vars.append((var_in,self,'out'))

    def display(self):
        print("ID: ", self.ID)

        print("Input:")
        for i in range(len(self.input_vars)):
            if not isinstance(self.input_vars[i], list):
                self.input_vars[i].display()
            else:
                for j in range(len(self.input_vars[i])):
                    self.input_vars[i][j].display()

        print("Output:")
        for i in range(len(self.output_vars)):
            if not isinstance(self.output_vars[i], list):
                self.output_vars[i].display()
            else:
                for j in range(len(self.output_vars[i])):
                    self.output_vars[i][j].display()
        return self.__class__.__name__

    # obtain the ID of the variable located at "index" of input or output (in_out) for that operator. Compresses the ID if unroll is False
    def get_var_ID(self, in_out, index, unroll=False):    
        if in_out == 'out':
            return self.output_vars[index].ID if unroll else self.output_vars[index].remove_round_from_ID()
        elif in_out == 'in':
            return self.input_vars[index].ID if unroll else self.input_vars[index].remove_round_from_ID()
        else:
            raise Exception(str(self.__class__.__name__) + ": unknown in_out type '" + in_out + "'")    

    def get_header_ID(self):
        return [self.__class__.__name__, self.model_version]

    def generate_implementation_header(self, implementation_type='python'):    # generic method that generates the code for the header of the modeling of that operator
        return None

    # method that returns the ID of the variable located at "index" of either the input or output of the operator, with options for bitwise listing and dimension unrolling
    def get_var_model(self, in_out, index, bitwise=True, dim=1):
        var = self.input_vars[index] if in_out == 'in' else self.output_vars[index]
        if bitwise and var.bitsize > 1:
            return [f"{var.ID}_{i}_{j}" for i in range(var.bitsize) for j in range(dim)] if dim > 1 else [f"{var.ID}_{i}" for i in range(var.bitsize)]  
        else:
            return [f"{var.ID}_{j}" for j in range(dim)] if dim > 1 else [f"{var.ID}"]

    @abstractmethod
    def generate_implementation(self, implementation_type='python'):  # generic method (abstract) that generates the code for the implementation of that operator
        pass

    @abstractmethod
    def generate_model(self, model_type='python'):  # generic method (abstract) that generates the code for the modeling of that operator
        pass


class CastingOperator(Operator):    # Operator for casting from on type to another
    def __init__(self, input_vars, output_vars, ID = None):
        if sum([input_vars[i].bitsize for i in range(len(input_vars))]) != sum([output_vars[i].bitsize for i in range(len(output_vars))]): raise Exception("CastingOperator: the total input size does not match the total output size")
        super().__init__(input_vars, output_vars, ID = ID)
        pass   # TODO


class CastingWordtoBitVector(CastingOperator):   # Operator for casting a bit word to a vector of bits
    def __init__(self, input_vars, output_vars, ID = None):
        pass   # TODO


class UnaryOperator(Operator):   # Generic operator taking one input and one output (must be of same bitsize)
    def __init__(self, input_vars, output_vars, ID = None):
        if len(input_vars) != 1: raise Exception(str(self.__class__.__name__) + ": your input does not contain exactly 1 element")
        if len(output_vars) != 1: raise Exception(str(self.__class__.__name__) + ": your output does not contain exactly 1 element")
        # if input_vars[0].bitsize != output_vars[0].bitsize: raise Exception(str(self.__class__.__name__) + ": your input and output sizes do not match") zcn: can be removed because the input size and output size of sbox may be different
        super().__init__(input_vars, output_vars, ID = ID)


class BinaryOperator(Operator):   # Generic operator taking two inputs and one output (must be of same bitsize)
    def __init__(self, input_vars, output_vars, ID = None):
        if len(input_vars) != 2: raise Exception(str(self.__class__.__name__) + ": your input does not contain exactly 2 element")
        if len(output_vars) != 1: raise Exception(str(self.__class__.__name__) + ": your output does not contain exactly 1 element")
        if input_vars[0].bitsize != input_vars[1].bitsize: raise Exception(str(self.__class__.__name__) + ": your inputs sizes do not match")
        if input_vars[0].bitsize != output_vars[0].bitsize: raise Exception(str(self.__class__.__name__) + ": your input and output sizes do not match")
        super().__init__(input_vars, output_vars, ID = ID)


class NoneOperator(Operator):  # Ghost Operator, does nothing (just a placeholder)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        return []

    def generate_model(self, model_type='sat'):
        return []


class CopyOperator(Operator):  # Operator that duplicates one input into multiple outputs: b_0, b_1, ..., b_n = a
    def __init__(self, input_vars, output_vars, ID = None):
        if len(input_vars) != 1:
            raise Exception(f"{self.__class__.__name__}: your input does not contain exactly 1 element")
        if len(output_vars) < 2:
            raise Exception(f"{self.__class__.__name__}: your output must contain at least 2 element")
        super().__init__(input_vars, output_vars, ID=ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        in_id = self.get_var_ID('in', 0, unroll)
        if implementation_type == 'python':
            return [f"{self.get_var_ID('out', j, unroll)} = {in_id}" for j in range(len(self.output_vars))]
        elif implementation_type == 'c':
            return [f"{self.get_var_ID('out', j, unroll)} = {in_id};" for j in range(len(self.output_vars))]
        elif implementation_type == 'verilog':
            return [f"assign {self.get_var_ID('out', j, unroll)} = {in_id};" for j in range(len(self.output_vars))]
        else:
            raise Exception(f"{self.__class__.__name__}: unknown implementation type '{implementation_type}'")

    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            if self.model_version in [self.__class__.__name__ + "_XORDIFF"]:
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                for i in range(self.input_vars[0].bitsize):
                    for j in range(len(var_out)):
                        model_list += [f"{var_out[j][i]} -{var_in[i]}", f"-{var_out[j][i]} {var_in[i]}"]
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":
                var_in, var_out1, var_out2 = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("out", 0, bitwise=False), self.get_var_model("out", 1, bitwise=False))
                model_list = [f'{var_in[0]} -{var_out1[0]}', f'-{var_in[0]} {var_out1[0]}', f'{var_in[0]} -{var_out2[0]}', f'-{var_in[0]} {var_out2[0]}']
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                var_out = [list(group) for group in zip(*var_out)]
                for i in range(self.output_vars[0].bitsize):
                    current_var_in = var_in[i]
                    current_var_out = var_out[i]
                    n = len(current_var_out)
                    for k in range(0, n + 1):
                        for comb in combinations(current_var_out, k):
                            is_odd_parity = (len(comb) % 2 == 1)
                            clause = [f"{current_var_in}" if is_odd_parity else f"-{current_var_in}"]
                            clause += [f"-{v}" if v in comb else f"{v}" for v in current_var_out]
                            model_list.append(" ".join(clause))
                return model_list
            elif len(self.output_vars) == 2 and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out1, var_out2 = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("out", 0, bitwise=False), self.get_var_model("out", 1, bitwise=False))
                model_list = [f'{var_in[0]} {var_out1[0]} -{var_out2[0]}', f'{var_in[0]} -{var_out1[0]} {var_out2[0]}', f'-{var_in[0]} {var_out1[0]} {var_out2[0]}']
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            # Modeling for differential cryptanalysis
            if self.model_version in [self.__class__.__name__ + "_XORDIFF"]:
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                for i in range(self.output_vars[0].bitsize):
                    for j in range(len(var_out)):
                        model_list += [f"{var_out[j][i]} - {var_in[i]} = 0"]
                model_list.append('Binary\n' + ' '.join(var_in + sum(var_out, [])))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), [self.get_var_model("out", i, bitwise=False) for i in range(len(self.output_vars))])
                for j in range(len(var_out)):
                    model_list += [f"{var_out[j][0]} - {var_in[0]} = 0"]
                model_list.append('Binary\n' + ' '.join(var_in + sum(var_out, [])))
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                var_d = [f"{self.ID}_d_{i}" for i in range(self.output_vars[0].bitsize)]
                if len(var_out) == 2:
                    for i in range(self.output_vars[0].bitsize):
                        i, o1, o2, d = var_in[i], var_out[0][i], var_out[1][i], var_d[i]
                        model_list += [f'{i} + {o1} + {o2} - 2 {d} >= 0', f'{i} + {o1} + {o2} <= 2', f'{d} - {i} >= 0', f'{d} - {o1} >= 0', f'{d} - {o2} >= 0']
                    model_list.append('Binary\n' + ' '.join(sum(var_out, []) + var_in + var_d))
                else:
                    var_out = [list(group) for group in zip(*var_out)]
                    for i in range(self.output_vars[0].bitsize):
                        model_list += [" + ".join(v for v in (var_out[i])) + " + " + var_in[i] + f" - 2 {var_d[i]} = 0"]
                        model_list += [f"{var_d[i]} >= 0"]
                        model_list += [f"{var_d[i]} <= {int((len(var_out[0])+1)/2)}"]
                    model_list.append('Binary\n' + ' '.join(sum(var_out, []) + var_in))
                    model_list.append('Integer\n' + ' '.join(var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR_1":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                if len(var_out) == 2:
                    for i in range(self.output_vars[0].bitsize):
                        i, o1, o2 = var_in[i], var_out[0][i], var_out[1][i]
                        model_list += [f'{i} + {o1} - {o2} >= 0', f'{o1} + {o2} - {i} >= 0', f'{i} + {o2} - {o1} >= 0', f'{i} + {o1} + {o2} <= 2']
                    model_list.append('Binary\n' + ' '.join(var_in + sum(var_out, [])))
                else:
                    var_out = [list(group) for group in zip(*var_out)]
                    var_d = [[f"{self.ID}_d_{i}_{j}" for i in range(int((len(self.output_vars)+1)/2))] for j in range(self.output_vars[0].bitsize)]
                    for i in range(self.output_vars[0].bitsize):
                        s = " + ".join(var_out[i]) + f" + {var_in[i]} - {2 * len(var_d[i])} {var_d[i][0]}"
                        s += " - " + " - ".join(f"{2 * (len(var_d[i]) - j)} {var_d[i][j]}" for j in range(1, len(var_d[i]))) if len(var_d[i]) > 1 else ""
                        s += " = 0"
                        model_list += [s]
                    model_list.append('Binary\n' + ' '.join(var_in + sum(var_out, []) + sum(var_d, [])))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR_2":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                if len(var_out) == 2:
                    var_d = [self.ID + '_d_' + str(i) for i in range(self.output_vars[0].bitsize)]
                    for i in range(self.output_vars[0].bitsize):
                        i, o1, o2, d = var_in[i], var_out[0][i], var_out[1][i], var_d[i]
                        model_list += [f'{i} + {o1} + {o2} - 2 {d} = 0']
                    model_list.append('Binary\n' + ' '.join(var_in + sum(var_out, []) + var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), [self.get_var_model("out", i, bitwise=False) for i in range(len(self.output_vars))])
                outputs = [iv[0] for iv in var_out]
                input = var_in[0]
                model_list.append(f"{' + '.join(outputs)} - {input} >= 0")
                for k, ik in enumerate(outputs):
                    others = [x for j, x in enumerate(outputs) if j != k]
                    model_list.append(f"{' + '.join(others)} + {input} - {ik} >= 0")
                model_list.append('Binary\n' +  ' '.join(var_in + outputs))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")



class Equal(UnaryOperator):  # Operator assigning equality between the input variable and output variable (must be of same bitsize)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

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
                model_list = [f"{vin} - {vout} = 0" for vin, vout in zip(var_in, var_out)]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f"{var_in[0]} - {var_out[0]} = 0"]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class Rot(UnaryOperator):     # Operator for the rotation function: rotation of the input variable to the output variable with "direction" ('l' or 'r') and "amount" of bits
    def __init__(self, input_vars, output_vars, direction, amount, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        if direction!='l' and direction!='r': raise Exception(str(self.__class__.__name__) + ": unknown direction value")
        self.direction = direction
        if amount<=0 or amount>= input_vars[0].bitsize: raise Exception(str(self.__class__.__name__) + ": wrong amount value")
        self.amount = amount

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            if self.direction == 'r': return [self.get_var_ID('out', 0, unroll) + ' = ROTR(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ')']
            else: return [self.get_var_ID('out', 0, unroll) + ' = ROTL(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ')']
        elif implementation_type == 'c':
            if self.direction == 'r': return [self.get_var_ID('out', 0, unroll) + ' = ROTR(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ');']
            else: return [self.get_var_ID('out', 0, unroll) + ' = ROTL(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ');']
        elif implementation_type == 'verilog':
            if self.direction == 'r': return ["assign " + self.get_var_ID('out', 0, unroll) + ' = `ROTR(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ');']
            else: return ["assign " + self.get_var_ID('out', 0, unroll) + ' = `ROTL(' + self.get_var_ID('in', 0, unroll) + ', ' + str(self.amount) + ', ' + str(self.input_vars[0].bitsize) + ');']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_implementation_header_unique(self, implementation_type='python'):
        if implementation_type == 'python':
            return ["#Rotation Macros ", "def ROTL(n, d, bitsize): return ((n << d) | (n >> (bitsize - d))) & (2**bitsize - 1)", "def ROTR(n, d, bitsize): return ((n >> d) | (n << (bitsize - d))) & (2**bitsize - 1)"]
        elif implementation_type == 'c':
            if self.input_vars[0].bitsize < 32:
                return ["//Rotation Macros", "#define ROTL(n, d, bitsize) (((n << d) | (n >> (bitsize - d))) & ((1<<bitsize) - 1)) ", "#define ROTR(n, d, bitsize) (((n >> d) | (n << (bitsize - d))) & ((1<<bitsize) - 1))"]
            elif 32 <= self.input_vars[0].bitsize < 64:
                return ["//Rotation Macros", "#define ROTL(n, d, bitsize) (((n << d) | (n >> ((unsigned long long)(bitsize) - d))) & ((1ULL << (bitsize)) - 1))", "#define ROTR(n, d, bitsize) (((n >> d) | (n << ((unsigned long long)(bitsize) - d))) & ((1ULL << (bitsize)) - 1))"]
            else:
                return ["//Rotation Macros", "#define ROTL(n, d, bitsize) (((n << d) | (n >> ((__uint128_t)(bitsize) - d))) & (((__uint128_t)1 << (bitsize)) - 1))", "#define ROTR(n, d, bitsize) (((n >> d) | (n << ((__uint128_t)(bitsize) - d))) & (((__uint128_t)1 << (bitsize)) - 1))"]
        elif implementation_type == 'verilog':
            return ["//Rotation Macros", "`define ROTL(n, d, bitsize) {n[bitsize-1-d:0],n[bitsize-1:bitsize-d]}", "`define ROTR(n, d, bitsize) {n[d-1:0],n[bitsize-1:d]}"]
        else: return None

    def generate_model(self, model_type='sat'):
        if model_type == 'sat':
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[i]} {var_out[(i+self.amount)%len(var_in)]}", f"{var_in[i]} -{var_out[(i+self.amount)%len(var_in)]}")]
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[(i+self.amount)%len(var_in)]} {var_out[i]}", f"{var_in[(i+self.amount)%len(var_in)]} -{var_out[i]}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction == 'r' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction == 'l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f'{var_in[i]} - {var_out[(i + self.amount) % len(var_in)]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f'{var_in[(i+self.amount)%len(var_in)]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return  model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class Shift(UnaryOperator):    # Operator for the shift function: shift of the input variable to the output variable with "direction" ('l' or 'r') and "amount" of bits
    def __init__(self, input_vars, output_vars, direction, amount, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        if direction!='l' and direction!='r': raise Exception(str(self.__class__.__name__) + ": unknown direction value")
        self.direction = direction
        if amount<=0 or amount>= input_vars[0].bitsize: raise Exception(str(self.__class__.__name__) + ": wrong amount value")
        self.amount = amount

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + [" >> " if self.direction == 'r' else " << "][0] + str(self.amount) + ") & (2**" + str(self.input_vars[0].bitsize) + " - 1)"]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + [" >> " if self.direction == 'r' else " << "][0] + str(self.amount) + ') & ((1<<' + str(self.input_vars[0].bitsize) + ') - 1);']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + [" >> " if self.direction == 'r' else " << "][0] + str(self.amount) + ') & ((1<<' + str(self.input_vars[0].bitsize) + ') - 1);']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_model(self, model_type='sat'):
        if model_type == 'sat':
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f"-{var_out[i]}" for i in range(self.amount)]
                model_list += [clause for i in range(len(var_in)-self.amount) for clause in (f"-{var_in[i]} {var_out[i+self.amount]}", f"{var_in[i]} -{var_out[i+self.amount]}")]
                model_list += [f"{var_in[i]} -{var_in[i]}" for i in range(len(var_in)-self.amount, len(var_in))]
                return model_list
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f"{var_in[i]} -{var_in[i]}" for i in range(self.amount)]
                model_list += [clause for i in range(len(var_in) - self.amount) for clause in (f"-{var_in[i+self.amount]} {var_out[i]}", f"{var_in[i+self.amount]} -{var_out[i]}")]
                model_list += [f"-{var_out[i]}" for i in range(len(var_in)-self.amount, len(var_in))]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp':
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f'{var_out[i]} = 0' for i in range(self.amount)]
                model_list += [f'{var_in[i]} - {var_out[i+self.amount]} = 0' for i in range(len(var_in)-self.amount)]
                model_list += [f"{var_in[i]} - {var_in[i]} = 0" for i in range(len(var_in)-self.amount, len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF"])  or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]):
                model_list = [f"{var_in[i]} - {var_in[i]} = 0" for i in range(self.amount)]
                model_list += [f'{var_in[i+self.amount]} - {var_out[i]} = 0' for i in range(len(var_in)-self.amount)]
                model_list += [f'{var_out[i]} = 0' for i in range(len(var_in)-self.amount, len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class CustomOP(Operator):   # generic custom operator (to be defined by the user)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID=ID)
        pass # TODO
