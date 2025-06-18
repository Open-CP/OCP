import math
from operators.operators import BinaryOperator, RaiseExceptionVersionNotExisting


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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
    
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat': 
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]:
                # reference: Ling Sun, et al. Accelerating the Search of Differential and Linear Characteristics with the SAT Method
                model_list = []
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize-1)]
                for i in range(self.input_vars[0].bitsize-1):
                    alpha, beta, gamma, alpha1, beta1, gamma1 = var_in1[i],var_in2[i],var_out[i],var_in1[i+1],var_in2[i+1],var_out[i+1]
                    model_list += [f'{alpha} {beta} -{gamma} {alpha1} {beta1} {gamma1}', 
                                   f'{alpha} -{beta} {gamma} {alpha1} {beta1} {gamma1}',
                                   f'-{alpha} {beta} {gamma} {alpha1} {beta1} {gamma1}',
                                   f'-{alpha} -{beta} -{gamma} {alpha1} {beta1} {gamma1}',
                                   f'{alpha} {beta} {gamma} -{alpha1} -{beta1} -{gamma1}',
                                   f'{alpha} -{beta} -{gamma} -{alpha1} -{beta1} -{gamma1}',
                                   f'-{alpha} {beta} -{gamma} -{alpha1} -{beta1} -{gamma1}',
                                   f'-{alpha} -{beta} {gamma} -{alpha1} -{beta1} -{gamma1}']
                alpha, beta, gamma = var_in1[-1],var_in2[-1],var_out[-1]
                model_list += [f'{alpha} {beta} -{gamma}', f'{alpha} -{beta} {gamma}', f'-{alpha} {beta} {gamma}', f'-{alpha} -{beta} -{gamma}']
                for i in range(self.input_vars[0].bitsize-1):
                    alpha, beta, gamma, w = var_in1[i+1],var_in2[i+1],var_out[i+1],var_p[i]
                    model_list += [f'-{alpha} {gamma} {w}', f'{beta} -{gamma} {w}', f'{alpha} -{beta} {w}', f'{alpha} {beta} {gamma} -{w}', f'-{alpha} -{beta} -{gamma} -{w}']
                self.weight = var_p
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            model_list = []
            if self.model_version in ["DEFAULT", self.__class__.__name__ + "_XORDIFF"]: 
                # reference: Fu, K., Wang, M., Guo, Y., Sun, S., Hu, L. (2016). MILP-Based Automatic Search Algorithms for Differential and Linear Trails for Speck 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
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
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
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
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
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
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
    
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")