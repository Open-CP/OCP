from abc import ABC, abstractmethod
import math


def RaiseExceptionVersionNotExisting(class_name, model_version, model_type):
    raise Exception(class_name + ": version " + str(model_version) + " not existing for " + model_type)


# ********************* OPERATORS ********************* # 
# Class that represents a constraint/operator object, i.e. a type of node in our graph modeling (the other type being the variables)
# An Operator/Constraint node can only be linked to a Variable node in the graph representation 
# Operators/Constraints are relationships between a group of variables 

class Operator(ABC):
    def __init__(self, input_vars, output_vars, model_version="DEFAULT", ID = None):
        self.input_vars = input_vars        # input variables of that operator
        self.output_vars = output_vars      # output variables of that operator
        self.model_version = model_version  # model version that will be used for that operator
        self.ID = ID                        # ID of the operator
        
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
        
    def get_var_ID(self, in_out, index, unroll=False, index2=None):    # obtain the ID of the variable located at "index" of input or output (in_out) for that operator. Compresses the ID if unroll is False
        if in_out == 'out': 
            if index2 is not None: return self.output_vars[index][index2].ID if unroll else self.output_vars[index][index2].remove_round_from_ID()
            else: return self.output_vars[index].ID if unroll else self.output_vars[index].remove_round_from_ID()
        if in_out == 'in':
            if index2 is not None: return self.input_vars[index][index2].ID if unroll else self.input_vars[index][index2].remove_round_from_ID()
            else: return self.input_vars[index].ID if unroll else self.input_vars[index].remove_round_from_ID()

    def get_header_ID(self): 
        return [self.__class__.__name__, self.model_version]

    def generate_implementation_header(self, implementation_type='python'):    # generic method that generates the code for the header of the modeling of that operator
        return None
    
    def get_var_model(self, in_out, index, bitwise=True):
        var = self.input_vars[index] if in_out == 'in' else self.output_vars[index]
        if not isinstance(var, list):
            var_ID = self.get_var_ID(in_out, index, unroll=True)
            if bitwise and var.bitsize > 1:
                return [f"{var_ID}_{i}" for i in range(var.bitsize)]
            return [var_ID]
        elif isinstance(var, list):
            vars = []
            for i, v in enumerate(var):
                var_ID = self.get_var_ID(in_out, index, unroll=True, index2=i)
                if bitwise and v.bitsize > 1:
                    vars.extend([f"{var_ID}_{j}" for j in range(v.bitsize)])
                else:
                    vars.append(var_ID)
            return vars
        else:
            raise TypeError(f"{self.__class__.__name__}: Expected Variable or list of Variable at index {index}, "f"got {type(var)} instead.")

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

        
class Equal(UnaryOperator):  # Operator assigning equality between the input variable and output variable (must be of same bitsize)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
    
    def generate_implementation(self, implementation_type='python', unroll=False):   
        if implementation_type == 'python': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll)]
        elif implementation_type == 'c': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")
            
    def generate_model(self, model_type='sat'):   
        if model_type == 'sat': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:  
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))                
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_RUNCATEDLINEAR"]: 
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))                
                return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]: 
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = [f"{vin} - {vout} = 0" for vin, vout in zip(var_in, var_out)]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_RUNCATEDLINEAR"]: 
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f"{var_in[0]} - {var_out[0]} = 0"]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self._class_._name_), self.model_version, model_type)
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
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
        else: return None
        
    def generate_model(self, model_type='sat'):
        if model_type == 'sat': 
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[i]} {var_out[(i+self.amount)%len(var_in)]}", f"{var_in[i]} -{var_out[(i+self.amount)%len(var_in)]}")]
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[(i+self.amount)%len(var_in)]} {var_out[i]}", f"{var_in[(i+self.amount)%len(var_in)]} -{var_out[i]}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction == 'r' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction == 'l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                model_list = [f'{var_in[i]} - {var_out[(i + self.amount) % len(var_in)]} = 0' for i in range(len(var_in))] 
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat'):
        if model_type == 'sat': 
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                model_list = [f"-{var_out[i]}" for i in range(self.amount)]              
                model_list += [clause for i in range(len(var_in)-self.amount) for clause in (f"-{var_in[i]} {var_out[i+self.amount]}", f"{var_in[i]} -{var_out[i+self.amount]}")]
                model_list += [f"{var_in[i]} -{var_in[i]}" for i in range(len(var_in)-self.amount, len(var_in))]
                return model_list        
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                model_list = [f"{var_in[i]} -{var_in[i]}" for i in range(self.amount)]
                model_list += [clause for i in range(len(var_in) - self.amount) for clause in (f"-{var_in[i+self.amount]} {var_out[i]}", f"{var_in[i+self.amount]} -{var_out[i]}")]
                model_list += [f"-{var_out[i]}" for i in range(len(var_in)-self.amount, len(var_in))]              
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            if (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"]) or (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                model_list = [f'{var_out[i]} = 0' for i in range(self.amount)]
                model_list += [f'{var_in[i]} - {var_out[i+self.amount]} = 0' for i in range(len(var_in)-self.amount)]                    
                model_list += [f"{var_in[i]} - {var_in[i]} = 0" for i in range(len(var_in)-self.amount, len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))        
                return model_list  
            elif (self.direction =='l' and self.model_version in [self.__class__.__name__ + "_XORDIFF", "DEFAULT"])  or (self.direction =='r' and self.model_version in [self.__class__.__name__ + "_LINEAR"]): 
                model_list = [f"{var_in[i]} - {var_in[i]} = 0" for i in range(self.amount)]
                model_list += [f'{var_in[i+self.amount]} - {var_out[i]} = 0' for i in range(len(var_in)-self.amount)]
                model_list += [f'{var_out[i]} = 0' for i in range(len(var_in)-self.amount, len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list         
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class ConstantAdd(UnaryOperator): # Operator for the constant addition: use add_type ('xor' or '+') to incorporate the constant with value "constant" to the input variable and result is stored in the output variable 
                                  # (optional "modulo" defines the modular value in case of a modular addition, by default it uses 2^bitsize as modular value)
    def __init__(self, input_vars, output_vars, add_type, constant_table, round = 0, index = 0, modulo = None, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        if add_type!='xor' and add_type!='modadd': raise Exception(str(self.__class__.__name__) + ": unknown add_type value")
        self.add_type = add_type
        self.modulo = modulo
        self.table = constant_table
        self.table_r, self.table_i = round, index

    def generate_implementation(self, implementation_type='python', unroll=False):
        if unroll==True: my_constant=hex(self.table[self.table_r-1][self.table_i])
        else: my_constant=f"RC[i][{self.table_i}]"
        if implementation_type == 'python': 
            if self.add_type == 'xor': return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant]
            elif self.add_type == "+":
                if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + " + " + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
                else: 
                    if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1)]
                    else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') % ' + str(self.modulo)]
        elif implementation_type == 'c': 
            if self.add_type == 'xor': return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + my_constant.replace("//", "/") + ';']
            elif self.add_type == "+":
                if self.modulo == None: return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ';']
                else: 
                    if int(math.log2(self.input_vars[0].bitsize))==math.log2(self.input_vars[0].bitsize): return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') & ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
                    else: return [self.get_var_ID('out', 0, unroll) + ' = (' + self.get_var_ID('in', 0, unroll) + ' + ' + my_constant + ') % ' + str(self.modulo) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")  
            
    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python': 
            return [f"#Constraints List\nRC={self.table}"]
        elif implementation_type == 'c': 
            bit_size = max(max(row) for row in self.table).bit_length()
            var_def_c = 'uint8_t' if bit_size <= 8 else "uint32_t" if bit_size <= 32 else "uint64_t" if bit_size <= 64 else "uint128_t"
            return [f"// Constraints List\n{var_def_c} RC[][{len(self.table[0])}] = {{\n    " + ", ".join("{ " + ", ".join(map(str, row)) + " }" for row in self.table) + "\n};"]
        else: return None
        
    def generate_model(self, model_type='sat'):
        if model_type == 'sat': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"] and self.add_type == 'xor': 
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"] and self.add_type == 'xor': 
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"] and self.add_type == 'xor': 
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"] and self.add_type == 'xor': 
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f'{var_in[0]} - {var_out[0]} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class CustomOP(Operator):   # generic custom operator (to be defined by the user)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID=ID)
        pass # TODO       
