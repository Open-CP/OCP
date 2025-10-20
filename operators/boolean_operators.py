from itertools import combinations
from operators.operators import Operator, BinaryOperator, UnaryOperator, RaiseExceptionVersionNotExisting
  

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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat': 
            if self.model_version == "DEFAULT" or self.model_version == self.__class__.__name__ + "_XORDIFF": 
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
            if self.model_version == "DEFAULT" or self.model_version == self.__class__.__name__ + "_XORDIFF": 
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
            
    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat': 
            if self.model_version == "DEFAULT" or self.model_version == self.__class__.__name__ + "_XORDIFF": 
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
            if self.model_version == "DEFAULT" or self.model_version == self.__class__.__name__+"_XORDIFF": 
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
    def __init__(self, input_vars, output_vars, ID = None, mat=None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.mat = mat

    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python': 
            if self.mat:
                n = len(self.mat)
                s = self.get_var_ID('out', 0, unroll) + ' = ' 
                for i in range(n):
                    s += "(("
                    if self.mat[i][0] != None:
                        s += f"(({self.get_var_ID('in', 0, unroll)} >> {n-self.mat[i][0]-1}) & 1)"
                    if self.mat[i][1] != None:
                        s += f" ^ (({self.get_var_ID('in', 1, unroll)} >> {n-self.mat[i][1]-1}) & 1)"
                    s += f") << {n-i-1}) | "
                return [s[:-2]]
            else: return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c': 
            if self.mat:
                n = len(self.mat)
                s = self.get_var_ID('out', 0, unroll) + ' = '
                for i in range(n):
                    s += "("
                    if self.mat[i][0] != None:
                        s += f"(({self.get_var_ID('in', 0, unroll)} >> {n-self.mat[i][0]-1}) & 1)"
                    if self.mat[i][1] != None:
                        s += f" ^ (({self.get_var_ID('in', 1, unroll)} >> {n-self.mat[i][1]-1}) & 1)"
                    s += f") << {n-i-1} | "
                s = s.rstrip(' | ') + ';'
                return [s]
            else: return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll) + ';']
        elif implementation_type == 'verilog': 
            if self.mat:
                n = len(self.mat)
                s = "assign " + self.get_var_ID('out', 0, unroll) + ' = '
                for i in range(n):
                    s += "("
                    if self.mat[i][0] != None:
                        s += f"(({self.get_var_ID('in', 0, unroll)} >> {n-self.mat[i][0]-1}) & 1)"
                    if self.mat[i][1] != None:
                        s += f" ^ (({self.get_var_ID('in', 1, unroll)} >> {n-self.mat[i][1]-1}) & 1)"
                    s += f") << {n-i-1} | "
                s = s.rstrip(' | ') + ';'
                return [s]
            else: return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
    
    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat': 
            if self.mat and (self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]):
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(self.mat)): 
                    if self.mat[i][0] != None and self.mat[i][1] != None:
                        i1, i2, o = var_in1[self.mat[i][0]], var_in2[self.mat[i][1]], var_out[i]
                        if self.model_version == "DEFAULT" or self.model_version == self.__class__.__name__ + "_XORDIFF": 
                            model_list += [f'{i1} {i2} -{o}', f'{i1} -{i2} {o}', f'-{i1} {i2} {o}', f'-{i1} -{i2} -{o}']
                        elif self.model_version == self.__class__.__name__ + "_LINEAR":
                            model_list += [f'{i1} -{o}', f'-{i1} {o}', f'{i2} -{o}', f'-{i2} {o}']
                        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")  
                    elif self.mat[i][0] != None: 
                        model_list += [f'{var_in1[self.mat[i][0]]} -{var_out[i]}', f'-{var_in1[self.mat[i][0]]} {var_out[i]}']
                    elif self.mat[i][1] != None: 
                        model_list += [f'{var_in2[self.mat[i][1]]} -{var_out[i]}', f'-{var_in2[self.mat[i][1]]} {var_out[i]}']                        
                return model_list
            # Modeling for differential cryptanalysis
            elif self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]:
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):
                    i1, i2, o = var_in1[i],var_in2[i],var_out[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} -{i2} {o}', f'-{i1} {i2} {o}', f'-{i1} -{i2} -{o}']
                return model_list      
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f'{var_in1[0]} {var_in2[0]} -{var_out[0]}', f'{var_in1[0]} -{var_in2[0]} {var_out[0]}', f'-{var_in1[0]} {var_in2[0]} {var_out[0]}']
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):
                    i1, i2, o = var_in1[i],var_in2[i],var_out[i]
                    model_list += [f'{i1} -{o}', f'-{i1} {o}', f'{i2} -{o}', f'-{i2} {o}']
                return model_list      
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list = [f'{var_in1[0]} -{var_out[0]}', f'-{var_in1[0]} {var_out[0]}', f'{var_in2[0]} -{var_out[0]}', f'-{var_in2[0]} {var_out[0]}']
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.mat and (self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1", self.__class__.__name__ + "_XORDIFF_2", self.__class__.__name__ + "_LINEAR"]):
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                bin_vars = []
                for i in range(len(self.mat)): 
                    if self.mat[i][0] != None and self.mat[i][1] != None:
                        i1, i2, o = var_in1[self.mat[i][0]], var_in2[self.mat[i][1]], var_out[i]
                        bin_vars += [i1, i2, o]
                        if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
                            d = self.ID + '_d_' + str(i)
                            model_list += [f'{i1} + {i2} + {o} - 2 {d} >= 0', f'{i1} + {i2} + {o} <= 2', f'{d} - {i1} >= 0', f'{d} - {i2} >= 0', f'{d} - {o} >= 0']
                            bin_vars.append(d)
                        elif self.model_version == self.__class__.__name__ + "_XORDIFF_1":                             
                            model_list += [f'{i1} + {i2} - {o} >= 0', f'{i2} + {o} - {i1} >= 0', f'{i1} + {o} - {i2} >= 0', f'{i1} + {i2} + {o} <= 2']
                        elif self.model_version == self.__class__.__name__ + "_XORDIFF_2": 
                            d = self.ID + '_d_' + str(i)
                            model_list += [f'{i1} + {i2} + {o} - 2 {d} = 0']
                            bin_vars.append(d)
                        elif self.model_version == self.__class__.__name__ + "_LINEAR":
                            model_list += [f'{i1} - {o} = 0', f'{i2} - {o} = 0']
                    elif self.mat[i][0] != None: 
                        model_list += [f'{var_in1[self.mat[i][0]]} - {var_out[i]} = 0']
                        bin_vars += [var_in1[self.mat[i][0]], var_out[i]]
                    elif self.mat[i][1] != None : 
                        model_list += [f'{var_in2[self.mat[i][1]]} - {var_out[i]} = 0']  
                        bin_vars += [var_in2[self.mat[i][1]], var_out[i]]     
                    else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)   
                model_list.append('Binary\n' +  ' '.join(v for v in bin_vars))
                return model_list
            # Modeling for differential cryptanalysis
            elif self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
                    var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                    var_d = [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                    for i in range(len(var_in1)): 
                        i1, i2, o, d = var_in1[i], var_in2[i], var_out[i], var_d[i]
                        model_list += [f'{i1} + {i2} + {o} - 2 {d} >= 0', f'{i1} + {i2} + {o} <= 2', f'{d} - {i1} >= 0', f'{d} - {i2} >= 0', f'{d} - {o} >= 0']
                    model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                    return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_1": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):  
                    i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                    model_list += [f'{i1} + {i2} - {o} >= 0', f'{i2} + {o} - {i1} >= 0', f'{i1} + {o} - {i2} >= 0', f'{i1} + {i2} + {o} <= 2']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_2": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                var_d = [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):  
                    i1, i2, o, d = var_in1[i], var_in2[i], var_out[i], var_d[i]
                    model_list += [f'{i1} + {i2} + {o} - 2 {d} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                var_d = [self.ID + '_d']
                model_list += [f'{var_in1[0]} + {var_in2[0]} + {var_out[0]} - 2 {var_d[0]} >= 0', f'{var_d[0]} - {var_in1[0]} >= 0', f'{var_d[0]} - {var_in2[0]} >= 0', f'{var_d[0]} - {var_out[0]} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF_1": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("in", 1, bitwise=False), self.get_var_model("out", 0, bitwise=False))
                model_list += [f'{var_in1[0]} + {var_in2[0]} - {var_out[0]} >= 0', f'{var_in2[0]} + {var_out[0]} - {var_in1[0]} >= 0', f'{var_in1[0]} + {var_out[0]} - {var_in2[0]} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version == self.__class__.__name__ + "_LINEAR": 
                var_in1, var_in2, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("out", 0))
                for i in range(len(var_in1)):  
                    i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                    model_list += [f'{i1} - {o} = 0', f'{i2} - {o} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR": 
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
    
    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat':
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                var_in = [list(group) for group in zip(*var_in)]
                for i in range(self.input_vars[0].bitsize):
                    current_var_in = var_in[i]
                    current_var_out = var_out[i]
                    n = len(current_var_in)
                    for k in range(0, n + 1):  # All subsets (0 to n elements)
                        for comb in combinations(current_var_in, k):
                            is_odd_parity = (len(comb) % 2 == 1)
                            clause = [f"{current_var_out}" if is_odd_parity else f"-{current_var_out}"]
                            clause += [f"-{v}" if v in comb else f"{v}" for v in current_var_in]
                            model_list.append(" ".join(clause))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_LINEAR": 
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                for i in range(self.input_vars[0].bitsize):
                    for j in range(len(var_in)):
                        model_list += [f"{var_out[i]} -{var_in[j][i]}", f"-{var_out[i]} {var_in[j][i]}"]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            # Modeling for differential cryptanalysis
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                var_in = [list(group) for group in zip(*var_in)]
                var_d = [f"{self.ID}_d_{i}" for i in range(self.input_vars[0].bitsize)] 
                for i in range(self.input_vars[0].bitsize):
                    model_list += [" + ".join(v for v in (var_in[i])) + " + " + var_out[i] + f" - 2 {var_d[i]} = 0"]
                    model_list += [f"{var_d[i]} >= 0"]
                    model_list += [f"{var_d[i]} <= {int((len(var_in[0])+1)/2)}"]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + var_out))
                model_list.append('Integer\n' + ' '.join(var_d))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_XORDIFF_1":  # Reference: Milp-aided cryptanalysis of the future block cipher.
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                var_in = [list(group) for group in zip(*var_in)]
                var_d = [[f"{self.ID}_d_{i}_{j}" for i in range(int((len(self.input_vars)+1)/2))] for j in range(self.input_vars[0].bitsize)] 
                for i in range(self.input_vars[0].bitsize):
                    s = " + ".join(var_in[i]) + f" + {var_out[i]} - {2 * len(var_d[i])} {var_d[i][0]}"
                    s += " - " + " - ".join(f"{2 * (len(var_d[i]) - j)} {var_d[i][j]}" for j in range(1, len(var_d[i]))) if len(var_d[i]) > 1 else ""
                    s += " = 0"
                    model_list += [s]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + sum(var_d, []) + var_out))
                return model_list
            elif len(self.input_vars) >= 2 and self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":  # Reference: Related-Key Differential Analysis of the AES.
                var_in, var_out = ([self.get_var_model("in", i, bitwise=False) for i in range(len(self.input_vars))], self.get_var_model("out", 0, bitwise=False))
                inputs = [iv[0] for iv in var_in]
                output = var_out[0]
                model_list.append(f"{' + '.join(inputs)} - {output} >= 0")
                for k, ik in enumerate(inputs):
                    others = [x for j, x in enumerate(inputs) if j != k]
                    model_list.append(f"{' + '.join(others)} + {output} - {ik} >= 0")
                model_list.append('Binary\n' +  ' '.join(inputs + var_out))
                return model_list
            # Modeling for linear cryptanalysis
            elif self.model_version in [self.__class__.__name__ + "_LINEAR"]: 
                var_in, var_out = ([self.get_var_model("in", i) for i in range(len(self.input_vars))], self.get_var_model("out", 0))
                for i in range(self.input_vars[0].bitsize):
                    for j in range(len(var_in)):
                        model_list += [f"{var_out[i]} - {var_in[j][i]} = 0"]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + var_out))
                return model_list
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
            
    def generate_model(self, model_type='sat'):
        if model_type == 'sat': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]: 
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]: 
                var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
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
    
        
    def generate_model(self, model_type='sat'):
        model_list = []
        if model_type == 'sat': 
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_1", self.__class__.__name__ + "_XORDIFF_2", self.__class__.__name__ + "_XORDIFF_3"]: 
                var_in1, var_in2, var_in3, var_out = (self.get_var_model("in", 0),  self.get_var_model("in", 1), self.get_var_model("in", 2), self.get_var_model("out", 0))
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                for i in range(len(var_in1)):
                    i1, i2, i3, o, p = var_in1[i], var_in2[i], var_in3[i], var_out[i], var_p[i]
                    if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]:
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


class COPY(Operator): # Operator that duplicates one input into multiple outputs: b_0, b_1, ..., b_n = a
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
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
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
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]:
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