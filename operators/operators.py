from abc import ABC, abstractmethod
import math
import subprocess
import time 
import copy
import operators.matrix as matrix
from itertools import combinations
import os


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
            self.input_vars[i].display()
        print("Output:")
        for i in range(len(self.output_vars)):
            self.output_vars[i].display()
        return self.__class__.__name__    
        
    def get_var_ID(self, in_out, index, unroll=False, index2=None):    # obtain the ID of the variable located at "index" of input or output (in_out) for that operator. Compresses the ID if unroll is False
        if in_out == 'out': 
            if index2 is not None: return self.output_vars[index][index2].ID if unroll else self.output_vars[index][index2].remove_round_from_ID()
            else: return self.output_vars[index].ID if unroll else self.output_vars[index].remove_round_from_ID()
        if in_out == 'in':
            if index2 is not None: return self.input_vars[index][index2].ID if unroll else self.input_vars[index][index2].remove_round_from_ID()
            else: return self.input_vars[index].ID if unroll else self.input_vars[index].remove_round_from_ID()
            
    def generate_implementation_header(self, implementation_type='python'):    # generic method that generates the code for the header of the modeling of that operator
        return None
    
    def get_vars(self, in_out, index=0, unroll=False):
        if not isinstance(self.input_vars[index], list):
            if in_out == 'in':
                return [self.get_var_ID('in', index, unroll) + '_' + str(i) for i in range(self.input_vars[index].bitsize)]
            elif in_out == 'out':
                return [self.get_var_ID('out', index, unroll) + '_' + str(i) for i in range(self.output_vars[index].bitsize)]
        elif isinstance(self.input_vars[index], list):
            if in_out == 'in':
                return [self.get_var_ID('in', index, unroll=unroll, index2=i) + '_' + str(j) for i in range(len(self.input_vars[index])) for j in range(self.input_vars[index][i].bitsize)]
            elif in_out == 'out':
                return [self.get_var_ID('out', index, unroll=unroll, index2=i) + '_' + str(j) for i in range(len(self.output_vars[index])) for j in range(self.output_vars[index][i].bitsize)]
    
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
            
    def generate_model(self, model_type='sat', unroll=False):   
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT":  
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            elif self.model_version == "truncated_diff": 
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = [f"{vin} - {vout} = 0" for vin, vout in zip(var_in, var_out)]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version == "truncated_diff": 
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                model_list = [f"{var_in[0]} - {var_out[0]} = 0"]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self._class_._name_), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
    

class Sbox(UnaryOperator):  # Generic operator assigning a Sbox relationship between the input variable and output variable (must be of same bitsize)
    def __init__(self, input_vars, output_vars, input_bitsize, output_bitsize, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.input_bitsize = input_bitsize
        self.output_bitsize = output_bitsize
        self.table = None
        self.table_inv = None
        
    def computeDDT(self):  # method computing the DDT of the Sbox
        DDT = [[0]*(2**self.output_bitsize) for _ in range(2**self.input_bitsize)] 
        for in_diff in range(2**self.input_bitsize):
            for j in range(2**self.input_bitsize):
                out_diff = self.table[j] ^ self.table[j^in_diff]
                DDT[in_diff][out_diff] += 1 
        return DDT
    

    def star_ddt_to_truthtable(self, ddt): # Convert star-DDT into a truthtable, which encode the differential propagations without probalities
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            dx = n >> self.output_bitsize
            dy = n & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] > 0: ttable += '0'
            else: ttable += '1'
        return ttable


    def pddt_to_truthtable(self, ddt, p): # Convert p-DDT into a truthtable, which encode the differential propagations with the item in ddt equal to p.  
        ttable = ''    
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            dx = n >> self.output_bitsize
            dy = n & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] == p: ttable += '0'
            else: ttable += '1'
        return ttable


    def ddt_to_truthtable(self, len_diff_spectrum, diff_weights, ddt): # Convert the DDT into a truthtable, which encode the differential propagations with probalities.  
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize+len_diff_spectrum)):
            dx = n >> (self.output_bitsize + len_diff_spectrum)
            dy = (n >> len_diff_spectrum) & ((1 << self.output_bitsize) - 1)
            p = bin(n & ((1 << (len_diff_spectrum)) - 1))[2:].zfill(len_diff_spectrum)
            w = 0
            for i in range(len_diff_spectrum): w += diff_weights[i] * int(p[i])
            if ddt[dx][dy] > 0 and abs(float(math.log(ddt[dx][dy]/(2**self.input_bitsize), 2))) == w: ttable += '0'
            else: ttable += '1'
        return ttable


    def ddt_to_truthtable_sat(self, diff_weights, max_diff_weights, ddt): # Convert the DDT, which encode the differential propagations with probalities into a truthtable in sat. All weights should be integers in this case.  
        if any([i != int(i) for i in diff_weights]):
            raise ValueError("All transition's weights should be integers")
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize+max_diff_weights)):
            dx = n >> (self.output_bitsize + max_diff_weights)
            dy = (n >> max_diff_weights) & ((1 << self.output_bitsize) - 1)
            p = tuple(int(x) for x in bin(n & ((1 << max_diff_weights) - 1))[2:].zfill(max_diff_weights))
            if ddt[dx][dy] > 0:
                w = int(abs(float(math.log(ddt[dx][dy]/(2**self.input_bitsize), 2))))
                if p == tuple([0]*(max_diff_weights - w) + [1]*w): ttable += '0' 
                else: ttable += '1'                
            else: ttable += '1'
        return ttable
    

    def gen_constraints_espresso(self, filename, model_type='milp', mode=0):
        try:
            from pyeda.inter import espresso_tts
        except ImportError:
            print("pyeda is not installed, installing it by 'pip3 install pyeda', https://pyeda.readthedocs.io/en/latest/install.html...\n")
            return ""
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                for line in file:
                    if 'Constraints:' in line:
                        constraints = eval(line.split(':', 1)[1].strip())
                    if 'Weight:' in line:
                        objective_fun = line[len("Weight: "):]
            return constraints, objective_fun
        input_variables = [f'a{i}' for i in range(self.input_bitsize)]
        output_variables = [f'b{i}' for i in range(self.output_bitsize)]
        if "diff" in self.model_version or self.model_version == "DEFAULT": ddt = self.computeDDT()
        objective_fun = ''
        cont_ttable = ''
        if self.model_version == "diff_0" or self.model_version == "DEFAULT":
            diff_spectrum = sorted(list(set([ddt[i][j] for i in range(2**self.input_bitsize) for j in range(2**self.output_bitsize)]) - {0, 2**self.input_bitsize}))
            diff_weights = [abs(float(math.log(d/(2**self.input_bitsize), 2))) for d in diff_spectrum]  
            len_diff_spectrum = len(diff_spectrum)   
            if model_type == 'milp':
                pr_variables = [f'p{i}' for i in range(len_diff_spectrum)]  
                complexity = self.input_bitsize + self.output_bitsize + len_diff_spectrum
                ttable = self.ddt_to_truthtable(len_diff_spectrum, diff_weights, ddt)
                objective_fun = "{}".format(" + ".join(["{:0.04f} p{:d}".format(diff_weights[i], i) for i in range(len_diff_spectrum)]))
            elif model_type == 'sat':
                max_diff_weights = int(max(diff_weights))
                pr_variables = [f'p{i}' for i in range(max_diff_weights)] 
                complexity = self.input_bitsize + self.output_bitsize + max_diff_weights
                ttable = self.ddt_to_truthtable_sat(diff_weights, max_diff_weights, ddt)
                objective_fun = "{}".format(" + ".join(["p{:d}".format(i) for i in range(max_diff_weights)])) 
            variables = input_variables + output_variables + pr_variables
            for n in range(2**(complexity)): cont_ttable += f'{bin(n)[2:].zfill(complexity)} {ttable[n]}\n'  
        elif self.model_version == "diff_1":
            complexity = self.input_bitsize + self.output_bitsize
            variables = input_variables + output_variables
            ttable = self.star_ddt_to_truthtable(ddt)
            for n in range(2**(complexity)): cont_ttable += f'{bin(n)[2:].zfill(complexity)} {ttable[n]}\n' 
        elif "diff_p" in self.model_version:
            complexity = self.input_bitsize + self.output_bitsize
            variables = input_variables + output_variables
            ttable = self.pddt_to_truthtable(ddt, int(self.model_version.replace("diff_p", "")))
            for n in range(2**(complexity)): cont_ttable += f'{bin(n)[2:].zfill(complexity)} {ttable[n]}\n' 
        file_contents = f".i {complexity}\n"
        file_contents += ".o 1\n"
        file_contents += f".p {2**(complexity)}\n"
        file_contents += ".ilb " + " ".join(variables) + "\n"     
        file_contents += ".ob F\n"
        file_contents += ".type fr\n" 
        file_contents += cont_ttable
        with open('files/ttable.txt', "w") as fw:
            fw.write(file_contents)
            fw.write(".e\n")
        espresso_options =  [['-estrong', '-eonset'], [], ['-eonset']]
        espresso_command = ['espresso', *espresso_options[mode], 'files/ttable.txt'] # Espresso Script of Pyeda provides the parameters: "-e {fast,ness,nirr,nunwrap,onset,strong}"
        time_start = time.time()
        result = subprocess.run(espresso_command, capture_output=True, text=True)
        time_end = time.time()
        espresso_output = result.stdout
        with open('files/sttable.txt', 'w') as fw: fw.write(espresso_output)
        with open('files/sttable.txt', 'r') as fr: espresso_output = fr.readlines()   
        constraints = []
        starting_point = 0
        end_point = 0
        for i in range(len(espresso_output)):
            if ".p" in espresso_output[i]: starting_point = i + 1
            if ".e" in espresso_output[i]: end_point = i
        for l in espresso_output[starting_point:end_point]:
            line = l[0:len(variables)]
            constraint = ''
            lp_rhs = 0
            for i in range(len(variables)):
                if model_type == 'milp':
                    if line[i] == '0': constraint += " + " + variables[i]
                    elif line[i] == '1': 
                        constraint += " - "+ variables[i]
                        lp_rhs += 1
                elif model_type == 'sat':
                    if line[i] == '0': constraint += ' ' + variables[i]
                    elif line[i] == '1': constraint += ' -' + variables[i]
            if model_type == 'milp':
                constraint = constraint.strip(" + ")
                constraint += ' >= ' + str(-(lp_rhs - 1))
            elif model_type == 'sat': constraint = constraint.strip(" ")
            constraints.append(constraint)
        len_constraints = len(constraints)
        variables_mapping = "Input: {0}; msb: {1}".format("||".join(input_variables), input_variables[0])
        variables_mapping += "\nOutput: {0}; msb: {1}".format("||".join(output_variables), output_variables[0])
        with open(filename, 'w') as file:
            file.write(f"{variables_mapping}\n")
            file.write("Time used to simplify the constraints: {:0.04f} s\n".format(time_end - time_start))
            file.write(f"Number of constraints: {len_constraints}\n") 
            file.write(f"Constraints: {constraints}\n") 
            file.write(f"Weight: {objective_fun}\n") 
        return constraints, objective_fun
    
    
    def differential_branch_number(self): # Return differential branch number of the S-Box.
        ret = (1 << self.input_bitsize) + (1 << self.output_bitsize)
        for a in range(1 << self.input_bitsize):
            for b in range(1 << self.output_bitsize):
                if a != b:
                    x = a ^ b
                    y = self.table[a] ^ self.table[b]
                    w = bin(x).count('1') + bin(y).count('1')
                    if w < ret: ret = w
        return ret
    
    def is_bijective(self): # Check if the length of the set of s_box is equal to the length of s_box. The set will contain only unique elements
        return len(set(self.table)) == len(self.table) and all(i in self.table for i in range(len(self.table)))

    def generate_implementation(self, implementation_type='python', mode = 0, unroll=False, weight=True):
        if implementation_type == 'python': 
            if not isinstance(self.input_vars[0], list):
                return [self.get_var_ID('out', 0, unroll) + ' = ' + str(self.__class__.__name__) + '[' + self.get_var_ID('in', 0, unroll) + ']']
            elif isinstance(self.input_vars[0], list):
                x_bits = len(self.input_vars[0])
                x_expr = 'x = ' + ' | '.join(f'({self.get_var_ID("in", 0, unroll=unroll, index2=i)} << {x_bits - 1 - i})'for i in range(x_bits))
                model_list = [x_expr]
                model_list.append(f'y = {self.__class__.__name__}[x]')
                y_vars = ', '.join(f'{self.get_var_ID("out", 0, unroll=unroll, index2=i)}' for i in range(x_bits))
                y_bits = ', '.join(f'(y >> {x_bits - 1 - i}) & 1' for i in range(x_bits))
                model_list.append(f'{y_vars} = {y_bits}')
                return model_list
        elif implementation_type == 'c': 
            if not isinstance(self.input_vars[0], list):
                return [self.get_var_ID('out', 0, unroll) + ' = ' + str(self.__class__.__name__) + '[' + self.get_var_ID('in', 0, unroll) + '];']
            elif isinstance(self.input_vars[0], list):
                x_bits = len(self.input_vars[0])
                x_expr = 'x = ' + ' | '.join(f'({self.get_var_ID("in", 0, unroll=unroll, index2=i)} << {x_bits - 1 - i})'for i in range(x_bits))+ ";"
                model_list = [x_expr]
                model_list.append(f'y = {str(self.__class__.__name__)}[x];')
                for i in range(x_bits):
                    y_vars = self.get_var_ID("out", 0, unroll=unroll, index2=i)
                    y_bits = f'(y >> {x_bits - 1 - i}) & 1'
                    model_list.append(f'{y_vars} = {y_bits};')
                return model_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
        
    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python': 
            return [str(self.__class__.__name__) + ' = ' + str(self.table)]
        elif implementation_type == 'c': 
            if self.input_bitsize <= 8: 
                if isinstance(self.input_vars[0], list): return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint8_t ' + 'x;'] + ['uint8_t ' + 'y;']
                else: return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
            else: 
                if isinstance(self.input_vars[0], list): return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint32_t ' + 'x;'] + ['uint32_t ' + 'y;']
                else: return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
        else: return None
            
            
    def generate_model(self, model_type='python', mode = 0, unroll=False, weight=True):
        if model_type == 'sat': 
            filename = f'files/constraints_sbox_{str(self.__class__.__name__)}_{model_type}_{self.model_version}.txt'
            if self.model_version == "diff_0" or self.model_version == "DEFAULT":
                # modeling all possible (input difference, output difference, probablity) to search for the best differential characteristic
                sbox_inequalities, sbox_weight = self.gen_constraints_espresso(filename, model_type, mode)
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                model_list = []
                var_p = []
                for i in range(sbox_weight.count('+') + 1): var_p.append(f"{self.ID}_p{i}")
                for ineq in sbox_inequalities:
                    temp = ineq
                    for i in range(self.input_bitsize): temp = temp.replace(f"a{i}", var_in[i])
                    for i in range(self.output_bitsize): temp = temp.replace(f"b{i}", var_out[i])
                    for i in range(sbox_weight.count('+')+1): temp = temp.replace(f"p{i}", var_p[i])
                    model_list += [temp]
                self.weight = var_p                
                return model_list
            elif self.model_version == "diff_1":
                # modeling all possible (input difference, output difference)
                sbox_inequalities, sbox_weight = self.gen_constraints_espresso(filename, model_type, mode)
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                model_list = []
                for ineq in sbox_inequalities:
                    temp = ineq
                    for i in range(self.input_bitsize): temp = temp.replace(f"a{i}", var_in[i])
                    for i in range(self.output_bitsize): temp = temp.replace(f"b{i}", var_out[i])
                    model_list += [temp]
                if weight:
                    var_At = [self.ID + '_At']    
                    model_list += [f"-{var} {var_At[0]}" for var in var_in]
                    model_list += [" ".join(var_in) + ' -' + var_At[0]]
                    self.weight = var_At         
                return model_list
            elif self.model_version == "truncated_diff" and (not isinstance(self.input_vars[0], list)):
                # word-wise difference propagations, the input difference equals the ouput difference
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                model_list = [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
                if weight: self.weight = var_in               
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT":
                # modeling all possible (input difference, output difference, probablity)
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                model_list = []
                var_p = []    
                filename = f'files/constraints_sbox_{str(self.__class__.__name__)}_{model_type}_{self.model_version}.txt'
                sbox_inequalities, sbox_weight = self.gen_constraints_espresso(filename, model_type, mode)
                for i in range(sbox_weight.count('+') + 1):
                    var_p.append(f"{self.ID}_p{i}")
                    sbox_weight = sbox_weight.replace(f"p{i}", f"{self.ID}_p{i}")
                for ineq in sbox_inequalities:
                    temp = ineq
                    for i in range(self.input_bitsize): temp = temp.replace(f"a{i}", var_in[i])
                    for i in range(self.output_bitsize): temp = temp.replace(f"b{i}", var_out[i])
                    for i in range(sbox_weight.count('+')+1): temp = temp.replace(f"p{i}", var_p[i])
                    model_list += [temp]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out + var_p))
                self.weight = [sbox_weight]
                return model_list
            elif self.model_version == "diff_1":
                # modeling all possible (input difference, output difference)
                filename = f'files/constraints_sbox_{str(self.__class__.__name__)}_{model_type}_{self.model_version}.txt'
                sbox_inequalities, sbox_weight = self.gen_constraints_espresso(filename, model_type, mode)
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                vars = var_in + var_out
                model_list = []
                for ineq in sbox_inequalities:
                    temp = ineq
                    for i in range(self.input_bitsize): 
                        temp = temp.replace(f"a{i}", var_in[i])
                    for i in range(self.output_bitsize): 
                        temp = temp.replace(f"b{i}", var_out[i])
                    model_list += [temp]
                if weight:
                    var_At = [self.ID + '_At']    
                    model_list += [f"{var_At[0]} - {var_in[i]} >= 0" for i in range(len(var_in))]
                    model_list += [" + ".join(var_in) + ' - ' + var_At[0] + ' >= 0']
                    vars += var_At
                    self.weight = var_At
                model_list.append('Binary\n' +  ' '.join(v for v in vars))
                return model_list                
            elif self.model_version == "diff_p":
                # for large sbox, self.input_bitsize >= 8, e.g., skinny, cite from: MILP Modeling for (Large) S-boxes to Optimize Probability of Differential Characteristics. (2017). IACR Transactions on Symmetric Cryptology, 2017(4), 99-129.
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                model_list = []
                var_p = []    
                ddt = self.computeDDT()
                diff_spectrum = sorted(list(set([ddt[i][j] for i in range(2**self.input_bitsize) for j in range(2**self.output_bitsize)]) - {0, 2**self.input_bitsize}))
                weight = ''
                for w in range(len(diff_spectrum)):
                    var_p.append(f"{self.ID}_p{w}")
                    filename = f'files/constraints_sbox_{str(self.__class__.__name__)}_{model_type}_diff_1_p{diff_spectrum[w]}.txt'
                    sbox_inequalities, sbox_weight = self.gen_constraints_espresso(filename, model_type, f"diff_p{diff_spectrum[w]}", mode)
                    for ineq in sbox_inequalities:
                        temp = ineq
                        for i in range(self.input_bitsize): temp = temp.replace(f"a{i}", var_in[i])
                        for i in range(self.output_bitsize): temp = temp.replace(f"b{i}", var_out[i])
                        temp_0, temp_1 = temp.split(">=")[0], int(temp.split(" >= ")[1])
                        temp = temp_0 + f"- 10000 {var_p[w]} >= {temp_1-10000}"
                        model_list += [temp]
                    weight += " + " + "{:0.04f} ".format(abs(float(math.log(diff_spectrum[w]/(2**self.input_bitsize), 2)))) + var_p[w]
                self.weight = [weight]
                model_list += [' + '.join(var_p) + ' = 1\n']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out + var_p))
                return model_list
            elif self.model_version == "truncated_diff" and (not isinstance(self.input_vars[0], list)):
                # word-wise difference propagations, the input difference equals the ouput difference
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                model_list = [f'{var_in[0]} - {var_out[0]} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                if weight: self.weight = var_in           
                return model_list
            elif self.model_version == "truncated_diff_1":
                # modeling the differential branch number of sbox to calculate the minimum number of differentially active s-boxes
                branch_num = self.differential_branch_number()
                var_in, var_out = self.get_vars("in", unroll=unroll), self.get_vars("out", unroll=unroll)
                vars = var_in + var_out
                if branch_num >= 3: # modeling the differential branch number of sbox 
                    var_d = [self.ID + '_d'] 
                    model_list = [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                    model_list += [" + ".join(var_in + var_out) + ' - ' + str(branch_num) + ' ' + var_d[0] + ' >= 0']
                    vars += var_d
                if self.is_bijective(): # For bijective S-boxes, nonzero input difference must result in nonzero output difference and vice versa
                    model_list += [" {len(var_in)} + ".join(var_out) +  " - ".join(var_in) + ' >= 0']
                    model_list += [" {len(var_out)} + ".join(var_in) +  " - ".join(var_out) + ' >= 0']                    
                if weight:
                    var_At = [self.ID + '_At'] 
                    model_list += [f"{var_At[0]} - {var_in[i]} >= 0" for i in range(len(var_in))]
                    model_list += [" + ".join(var_in) + ' - ' + var_At[0] + ' >= 0']
                    self.weight = var_At  
                    vars += var_At
                model_list.append('Binary\n' +  ' '.join(v for v in vars))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")



class Skinny_4bit_Sbox(Sbox):         # Operator of the Skinny 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [12, 6, 9, 0, 1, 10, 2, 11, 3, 8, 5, 13, 4, 14, 7, 15]
        self.table_inv = [3, 4, 6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15]


class Skinny_8bit_Sbox(Sbox):         # Operator of the Skinny 8 -bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):      
        super().__init__(input_vars, output_vars, 8, 8, ID = ID)
        self.table = [0x65, 0x4c, 0x6a, 0x42, 0x4b, 0x63, 0x43, 0x6b, 0x55, 0x75, 0x5a, 0x7a, 0x53, 0x73, 0x5b, 0x7b, 
            0x35, 0x8c, 0x3a, 0x81, 0x89, 0x33, 0x80, 0x3b, 0x95, 0x25, 0x98, 0x2a, 0x90, 0x23, 0x99, 0x2b, 
            0xe5, 0xcc, 0xe8, 0xc1, 0xc9, 0xe0, 0xc0, 0xe9, 0xd5, 0xf5, 0xd8, 0xf8, 0xd0, 0xf0, 0xd9, 0xf9, 
            0xa5, 0x1c, 0xa8, 0x12, 0x1b, 0xa0, 0x13, 0xa9, 0x05, 0xb5, 0x0a, 0xb8, 0x03, 0xb0, 0x0b, 0xb9, 
            0x32, 0x88, 0x3c, 0x85, 0x8d, 0x34, 0x84, 0x3d, 0x91, 0x22, 0x9c, 0x2c, 0x94, 0x24, 0x9d, 0x2d, 
            0x62, 0x4a, 0x6c, 0x45, 0x4d, 0x64, 0x44, 0x6d, 0x52, 0x72, 0x5c, 0x7c, 0x54, 0x74, 0x5d, 0x7d, 
            0xa1, 0x1a, 0xac, 0x15, 0x1d, 0xa4, 0x14, 0xad, 0x02, 0xb1, 0x0c, 0xbc, 0x04, 0xb4, 0x0d, 0xbd, 
            0xe1, 0xc8, 0xec, 0xc5, 0xcd, 0xe4, 0xc4, 0xed, 0xd1, 0xf1, 0xdc, 0xfc, 0xd4, 0xf4, 0xdd, 0xfd, 
            0x36, 0x8e, 0x38, 0x82, 0x8b, 0x30, 0x83, 0x39, 0x96, 0x26, 0x9a, 0x28, 0x93, 0x20, 0x9b, 0x29, 
            0x66, 0x4e, 0x68, 0x41, 0x49, 0x60, 0x40, 0x69, 0x56, 0x76, 0x58, 0x78, 0x50, 0x70, 0x59, 0x79, 
            0xa6, 0x1e, 0xaa, 0x11, 0x19, 0xa3, 0x10, 0xab, 0x06, 0xb6, 0x08, 0xba, 0x00, 0xb3, 0x09, 0xbb, 
            0xe6, 0xce, 0xea, 0xc2, 0xcb, 0xe3, 0xc3, 0xeb, 0xd6, 0xf6, 0xda, 0xfa, 0xd3, 0xf3, 0xdb, 0xfb, 
            0x31, 0x8a, 0x3e, 0x86, 0x8f, 0x37, 0x87, 0x3f, 0x92, 0x21, 0x9e, 0x2e, 0x97, 0x27, 0x9f, 0x2f, 
            0x61, 0x48, 0x6e, 0x46, 0x4f, 0x67, 0x47, 0x6f, 0x51, 0x71, 0x5e, 0x7e, 0x57, 0x77, 0x5f, 0x7f, 
            0xa2, 0x18, 0xae, 0x16, 0x1f, 0xa7, 0x17, 0xaf, 0x01, 0xb2, 0x0e, 0xbe, 0x07, 0xb7, 0x0f, 0xbf, 
            0xe2, 0xca, 0xee, 0xc6, 0xcf, 0xe7, 0xc7, 0xef, 0xd2, 0xf2, 0xde, 0xfe, 0xd7, 0xf7, 0xdf, 0xff]
        self.table_inv = [0xac, 0xe8, 0x68, 0x3c, 0x6c, 0x38, 0xa8, 0xec, 0xaa, 0xae, 0x3a, 0x3e, 0x6a, 0x6e, 0xea, 0xee, 
            0xa6, 0xa3, 0x33, 0x36, 0x66, 0x63, 0xe3, 0xe6, 0xe1, 0xa4, 0x61, 0x34, 0x31, 0x64, 0xa1, 0xe4, 
            0x8d, 0xc9, 0x49, 0x1d, 0x4d, 0x19, 0x89, 0xcd, 0x8b, 0x8f, 0x1b, 0x1f, 0x4b, 0x4f, 0xcb, 0xcf, 
            0x85, 0xc0, 0x40, 0x15, 0x45, 0x10, 0x80, 0xc5, 0x82, 0x87, 0x12, 0x17, 0x42, 0x47, 0xc2, 0xc7, 
            0x96, 0x93, 0x03, 0x06, 0x56, 0x53, 0xd3, 0xd6, 0xd1, 0x94, 0x51, 0x04, 0x01, 0x54, 0x91, 0xd4, 
            0x9c, 0xd8, 0x58, 0x0c, 0x5c, 0x08, 0x98, 0xdc, 0x9a, 0x9e, 0x0a, 0x0e, 0x5a, 0x5e, 0xda, 0xde, 
            0x95, 0xd0, 0x50, 0x05, 0x55, 0x00, 0x90, 0xd5, 0x92, 0x97, 0x02, 0x07, 0x52, 0x57, 0xd2, 0xd7, 
            0x9d, 0xd9, 0x59, 0x0d, 0x5d, 0x09, 0x99, 0xdd, 0x9b, 0x9f, 0x0b, 0x0f, 0x5b, 0x5f, 0xdb, 0xdf, 
            0x16, 0x13, 0x83, 0x86, 0x46, 0x43, 0xc3, 0xc6, 0x41, 0x14, 0xc1, 0x84, 0x11, 0x44, 0x81, 0xc4, 
            0x1c, 0x48, 0xc8, 0x8c, 0x4c, 0x18, 0x88, 0xcc, 0x1a, 0x1e, 0x8a, 0x8e, 0x4a, 0x4e, 0xca, 0xce, 
            0x35, 0x60, 0xe0, 0xa5, 0x65, 0x30, 0xa0, 0xe5, 0x32, 0x37, 0xa2, 0xa7, 0x62, 0x67, 0xe2, 0xe7, 
            0x3d, 0x69, 0xe9, 0xad, 0x6d, 0x39, 0xa9, 0xed, 0x3b, 0x3f, 0xab, 0xaf, 0x6b, 0x6f, 0xeb, 0xef, 
            0x26, 0x23, 0xb3, 0xb6, 0x76, 0x73, 0xf3, 0xf6, 0x71, 0x24, 0xf1, 0xb4, 0x21, 0x74, 0xb1, 0xf4, 
            0x2c, 0x78, 0xf8, 0xbc, 0x7c, 0x28, 0xb8, 0xfc, 0x2a, 0x2e, 0xba, 0xbe, 0x7a, 0x7e, 0xfa, 0xfe, 
            0x25, 0x70, 0xf0, 0xb5, 0x75, 0x20, 0xb0, 0xf5, 0x22, 0x27, 0xb2, 0xb7, 0x72, 0x77, 0xf2, 0xf7, 
            0x2d, 0x79, 0xf9, 0xbd, 0x7d, 0x29, 0xb9, 0xfd, 0x2b, 0x2f, 0xbb, 0xbf, 0x7b, 0x7f, 0xfb, 0xff]


class GIFT_Sbox(Sbox):              # Operator of the GIFT 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [1, 10, 4, 12, 6, 15, 3, 9, 2, 13, 11, 7, 5, 0, 8, 14]
        self.table_inv = [13, 0, 8, 6, 2, 12, 4, 11, 14, 7, 1, 10, 3, 9, 15, 5]
        

class ASCON_Sbox(Sbox):             # Operator of the ASCON 5-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 5, 5, ID = ID)
        self.table = [4, 11, 31, 20, 26, 21, 9, 2, 27, 5, 8, 18, 29, 3, 6, 28, 30, 19, 7, 14, 0, 13, 17, 24, 16, 12, 1, 25, 22, 10, 15, 23]
        self.table_inv = [20, 26, 7, 13, 0, 9, 14, 18, 10, 6, 29, 1, 25, 21, 19, 30, 24, 22, 11, 17, 3, 5, 28, 31, 23, 27, 4, 8, 15, 12, 16, 2]
        

class AES_Sbox(Sbox):               # Operator of the AES 8-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):      
        super().__init__(input_vars, output_vars, 8, 8, ID = ID)
        self.table = [0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
            0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
            0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
            0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
            0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
            0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
            0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
            0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
            0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
            0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
            0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
            0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
            0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
            0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
            0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
            0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16]
        self.table_inv = [0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
            0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
            0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
            0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
            0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
            0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
            0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
            0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
            0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
            0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
            0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
            0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
            0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
            0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
            0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D]


class TWINE_Sbox(Sbox):             # Operator of the TWINE 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [12, 0, 15, 10, 2, 11, 9, 5, 8, 3, 13, 7, 1, 14, 6, 4]


class PRESENT_Sbox(Sbox):           # Operator of the PRESENT 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2]


class KNOT_Sbox(Sbox):             # Operator of the KNOT 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [4, 0, 10, 7, 11, 14, 1, 13, 9, 15, 6, 8, 5, 2, 12, 3]


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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
    
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            var_in, var_out = [[f"{self.get_var_ID('in', i, unroll)}_{j}" for i in range(len(self.input_vars))] for j in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
            model_list = []
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                for i in range(self.input_vars[0].bitsize):
                    current_var_in = var_in[i]
                    current_var_out = var_out[i]
                    n = len(current_var_in)
                    for k in range(0, n + 1):  # All subsets (0 to n elements)
                        for comb in combinations(current_var_in, k):
                            is_even_parity = (len(comb) % 2 == 0)
                            clause = [f"{current_var_out}" if is_even_parity else f"-{current_var_out}"]
                            clause += [f"-{v}" if v in comb else f"{v}" for v in current_var_in]
                            model_list.append(" ".join(clause))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            model_list = []
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in, var_out = [[f"{self.get_var_ID('in', i, unroll)}_{j}" for i in range(len(self.input_vars))] for j in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_d = [f"{self.ID}_d_{i}" for i in range(self.input_vars[0].bitsize)] 
                model_list = []
                for i in range(self.input_vars[0].bitsize):
                    model_list += [" + ".join(v for v in (var_in[i])) + " + " + var_out[i] + f" - 2 {var_d[i]} = 0"]
                    model_list += [f"{var_d[i]} >= 0"]
                    model_list += [f"{var_d[i]} <= {int((len(var_in[0])+1)/2)}"]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + var_out))
                model_list.append('Integer\n' + ' '.join(var_d))
                return model_list
            elif self.model_version == "diff_1":  # cited from: "Milp-aided cryptanalysis of the future block cipher".
                var_in, var_out = [[f"{self.get_var_ID('in', i, unroll)}_{j}" for i in range(len(self.input_vars))] for j in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_d = [[f"{self.ID}_d_{i}_{j}" for i in range(int((len(self.input_vars)+1)/2))] for j in range(self.input_vars[0].bitsize)] 
                model_list = []
                for i in range(self.input_vars[0].bitsize):
                    s = " + ".join(var_in[i]) + f" + {var_out[i]} - {2 * len(var_d[i])} {var_d[i][0]}"
                    s += " - " + " - ".join(f"{2 * (len(var_d[i]) - j)} {var_d[i][j]}" for j in range(1, len(var_d[i]))) if len(var_d[i]) > 1 else ""
                    s += " = 0"
                    model_list += [s]
                model_list.append('Binary\n' + ' '.join(sum(var_in, []) + sum(var_d, []) + var_out))
                return model_list
            elif len(self.input_vars) == 3 and self.model_version == "truncated_diff":  # cited from: "Related-Key Differential Analysis of the AES".
                var_in1, var_in2, var_in3, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('in', 1, unroll)], [self.get_var_ID('in', 2, unroll)], [self.get_var_ID('out', 0, unroll)]
                i1, i2, i3, o = var_in1[0], var_in2[0], var_in3[0], var_out[0]
                model_list += [f'{i1} + {i2} + {i3} - {o} >= 0', f'{i2} + {i3} + {o} - {i1} >= 0', f'{i1} + {i3} + {o} - {i2} >= 0', f'{i1} + {i2} + {o} - {i3} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_in3 + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")



class Matrix(Operator):   # Operator of the Matrix multiplication: appplies the matrix "mat" (stored as a list of lists) to the input vector of variables, towards the output vector of variables
                          # The optional "polynomial" allors to define the polynomial reduction (not implemted yet)
    def __init__(self, name, input_vars, output_vars, mat, polynomial = None, ID = None):
        r, c = len(mat), len(mat[0])
        for i in mat: 
            if len(i)!=c: raise Exception(str(self.__class__.__name__) + ": matrix size not consistent")
        if len(input_vars)!=c: raise Exception(str(self.__class__.__name__) + ": input vector does not match matrix size")
        if len(output_vars)!=r: raise Exception(str(self.__class__.__name__) + ": output vector does not match matrix size")
        super().__init__(input_vars, output_vars, ID = ID)
        self.name = name
        self.mat = mat
        self.polynomial = polynomial
    
    def differential_branch_number(self):
        # Return differential branch number of the Matrix. TO DO
        return 5 # the branch number of matrix for aes is 5, to do for other ciphers
        
    def generate_implementation(self, implementation_type='python', unroll=False, branch_num=None):
        if implementation_type == 'python': 
            return ['(' + ''.join([self.get_var_ID('out', i, unroll) + ", " for i in range(len(self.output_vars))])[:-2] + ") = " + self.name + "(" + ''.join([self.get_var_ID('in', i, unroll) + ", " for i in range(len(self.input_vars))])[:-2] + ")"]
        elif implementation_type == 'c': 
            return [self.name + "(" + ''.join([self.get_var_ID('in', i, unroll) + ", " for i in range(len(self.input_vars))])[:-2] + ", " + ''.join([self.get_var_ID('out', i, unroll) + ", " for i in range(len(self.output_vars))])[:-2] + ");"]
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python': 
            model_list = ["#Galois Field Multiplication Macro", "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
            model_list.append("#Matrix Macro ")
            model_list.append("def " + self.name + "(" + ''.join(["x" + str(i) + ", " for i in range (len(self.mat[0]))])[:-2]  + "):")      
            for i, out_v in enumerate(self.output_vars):
                model = '\t' + 'y' + str(i) + ' = ' 
                first = True
                for j, in_v in enumerate(self.input_vars):
                    if self.mat[i][j] == 1: 
                        if first: 
                            model = model + "x" + str(j)
                            first = False
                        else: model = model + " ^ " + "x" + str(j)
                    elif self.mat[i][j] != 0:
                        if first: 
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")" 
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")" 
                model_list.append(model)
            model_list.append("\treturn (" + ''.join(["y" + str(i) + ", " for i in range (len(self.mat))])[:-2]  + ")")
            return model_list
        elif implementation_type == 'c': 
            model_list = ["//Galois Field Multiplication Macro", "#define GMUL(a, b, p, d) ({ \\", "\tunsigned int result = 0; \\", "\tunsigned int temp_a = a; \\", "\tunsigned int temp_b = b; \\", "\twhile (temp_b > 0) { \\", "\t\tif (temp_b & 1) \\", "\t\t\tresult ^= temp_a; \\", "\t\ttemp_a <<= 1; \\", "\t\tif (temp_a & (1 << d)) \\", "\t\t\ttemp_a ^= p; \\", "\t\ttemp_b >>= 1; \\", "\t} \\", "\tresult & ((1 << d) - 1); \\","})"];
            model_list.append("//Matrix Macro ")
            model_list.append("#define " + self.name + "(" + ''.join(["x" + str(i) + ", " for i in range (len(self.mat[0]))])[:-2] + ", "  + ''.join(["y" + str(i) + ", " for i in range (len(self.mat))])[:-2] + ")  { \\")      
            for i, out_v in enumerate(self.output_vars):
                model = '\t' + 'y' + str(i) + ' = ' 
                first = True
                for j, in_v in enumerate(self.input_vars):
                    if self.mat[i][j] == 1: 
                        if first: 
                            model = model + "x" + str(j)
                            first = False
                        else: model = model + " ^ " + "x" + str(j)
                    elif self.mat[i][j] != 0:
                        if first: 
                            model = model + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")" 
                            first = False
                        else: model = model + " ^ " + "GMUL(" + "x" + str(j) + "," + str(self.mat[i][j]) + "," + self.polynomial + "," + str(self.input_vars[0].bitsize) + ")" 
                model_list.append(model + "; \\")
            model_list.append("} ")
            return model_list
            
    def generate_model(self, model_type='python', unroll=False, branch_num=None):
        if model_type == 'milp' or model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "diff_1" or self.model_version == "DEFAULT":
                model_list = []
                if self.polynomial: bin_matrix = matrix.generate_pmr_for_mds(self.mat, self.polynomial, self.input_vars[0].bitsize)
                elif self.input_vars[0].bitsize * len(self.input_vars) > len(self.mat):
                    bin_matrix = matrix.generate_bin_matrix(self.mat, self.input_vars[0].bitsize)
                elif self.input_vars[0].bitsize * len(self.input_vars) == len(self.mat):
                    bin_matrix = self.mat
                for i in range(len(self.mat)):
                    for j in range(self.input_vars[0].bitsize):
                        var_in = []
                        var_out = []
                        for k in range(len(self.mat)):
                            for l in range(self.input_vars[0].bitsize):
                                if bin_matrix[self.input_vars[0].bitsize*i+j][self.input_vars[0].bitsize*k+l] == 1:
                                    vi = copy.deepcopy(self.input_vars[i])
                                    vi.bitsize = 1
                                    vi.ID = self.input_vars[k].ID + '_' + str(l)
                                    var_in.append(vi)
                        vo = copy.deepcopy(self.output_vars[i])
                        vo.bitsize = 1
                        vo.ID = self.output_vars[i].ID + '_' + str(j)
                        var_out.append(vo)
                        n_xor = N_XOR(var_in, var_out, ID=self.ID+"_"+str(self.input_vars[0].bitsize*i+j))
                        n_xor.model_version = self.model_version
                        cons = n_xor.generate_model(model_type, unroll)
                        model_list += cons
                return model_list
            elif model_type == 'milp' and self.model_version == "truncated_diff":
                var_in, var_out = [self.get_var_ID('in', i, unroll) for i in range(len(self.input_vars))], [self.get_var_ID('out', i, unroll)for i in range (len(self.output_vars))]
                var_d = [f"{self.ID}_d"] 
                if branch_num == None: branch_num =self.differential_branch_number() 
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [f"{var_d[0]} - {var} >= 0" for var in var_in + var_out]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            elif model_type == 'milp' and self.model_version == "truncated_diff_1":
                var_in, var_out = [self.get_var_ID('in', i, unroll) for i in range(len(self.input_vars))], [self.get_var_ID('out', i, unroll)for i in range (len(self.output_vars))]
                var_d = [f"{self.ID}_d"] 
                if branch_num == None: branch_num =self.differential_branch_number() 
                model_list = [" + ".join(var_in + var_out) + f" - {branch_num} {var_d[0]} >= 0"]
                model_list += [" + ".join(var_in + var_out) + f" - {len(var_in+var_out)} {var_d[0]} <= 0"]
                model_list.append('Binary\n' + ' '.join(var_in + var_out + var_d))
                return model_list
            else:  RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
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
        
    def generate_implementation_header(self, implementation_type='python'):
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
        
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
            if self.direction =='r' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[i]} {var_out[(i+self.amount)%len(var_in)]}", f"{var_in[i]} -{var_out[(i+self.amount)%len(var_in)]}")]
            elif self.direction =='l' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                return [clause for i in range(len(var_in)) for clause in (f"-{var_in[(i+self.amount)%len(var_in)]} {var_out[i]}", f"{var_in[(i+self.amount)%len(var_in)]} -{var_out[i]}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
            if self.direction == 'r' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                model_list = [f'{var_in[i]} - {var_out[(i + self.amount) % len(var_in)]} = 0' for i in range(len(var_in))] 
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.direction =='l' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
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
        
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
            if self.direction =='r' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                model_list = [f"-{var_out[i]}" for i in range(self.amount)]              
                model_list += [clause for i in range(len(var_in)-self.amount) for clause in (f"-{var_in[i]} {var_out[i+self.amount]}", f"{var_in[i]} -{var_out[i+self.amount]}")]
                model_list += [f"{var_in[i]} -{var_in[i]}" for i in range(len(var_in)-self.amount, len(var_in))]    
                return model_list        
            elif self.direction =='l' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                model_list = [f"{var_in[i]} -{var_in[i]}" for i in range(self.amount)]    
                model_list += [clause for i in range(len(var_in) - self.amount) for clause in (f"-{var_in[i+self.amount]} {var_out[i]}", f"{var_in[i+self.amount]} -{var_out[i]}")]
                model_list += [f"-{var_out[i]}" for i in range(len(var_in)-self.amount, len(var_in))]              
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
            if self.direction =='r' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                model_list = [f'{var_out[i]} = 0' for i in range(self.amount)]
                model_list += [f'{var_in[i]} - {var_out[i+self.amount]} = 0' for i in range(len(var_in)-self.amount)]                    
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))        
                return model_list  
            elif self.direction =='l' and (self.model_version == "diff_0" or self.model_version == "DEFAULT"): 
                model_list = [f'{var_in[i+self.amount]} - {var_out[i]} = 0' for i in range(len(var_in)-self.amount)]
                model_list += [f'{var_out[i]} = 0' for i in range(len(var_in)-self.amount, len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list         
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

class ConstantAdd(UnaryOperator): # Operator for the constant addition: use add_type ('xor' or '+') to incorporate the constant with value "constant" to the input variable and result is stored in the output variable 
                                  # (optional "modulo" defines the modular value in case of a modular addition, by default it uses 2^bitsize as modular value)
    def __init__(self, input_vars, output_vars, add_type, constant_table, round, index, modulo = None, ID = None):
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
        
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if (self.model_version == "diff_0" or self.model_version == "DEFAULT") and self.add_type == 'xor': 
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            elif self.model_version == "truncated_diff" and self.add_type == 'xor': 
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if (self.model_version == "diff_0" or self.model_version == "DEFAULT") and self.add_type == 'xor': 
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            elif self.model_version == "truncated_diff" and self.add_type == 'xor': 
                var_in, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('out', 0, unroll)]
                model_list = [f'{var_in[0]} - {var_out[0]} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


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
    
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT":
                # cite: Ling Sun, et al. Accelerating the Search of Differential and Linear Characteristics with the SAT Method
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
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                # cite: Fu, K., Wang, M., Guo, Y., Sun, S., Hu, L. (2016). MILP-Based Automatic Search Algorithms for Differential and Linear Trails for Speck 
                model_list = []
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
    
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

class bitwiseAND(BinaryOperator):  # Operator for the bitwise AND operation: compute the bitwise AND on the two input variables towards the output variable 
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' & ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = []
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} {i2} -{p}', f'-{i1} {p}', f'-{i2} {p}']
                self.weight = var_p
                return model_list        
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = []
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} + {i2} - {o} >= 0', f'{i1} + {i2} - {p} >= 0', f'- {i1} + {p} >= 0', f'- {i2} + {p} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

class bitwiseOR(BinaryOperator):  # Operator for the bitwise OR operation: compute the bitwise OR on the two input variables towards the output variable 
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' | ' + self.get_var_ID('in', 1, unroll)]
        elif implementation_type == 'c': 
           return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' | ' + self.get_var_ID('in', 1, unroll) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
            
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = []
                for i in range(len(var_in1)):
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} {i2} -{p}', f'-{i1} {p}', f'-{i2} {p}']
                self.weight = var_p
                return model_list        
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                var_p = [self.ID + '_p_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = []
                for i in range(len(var_in1)):   
                    i1, i2, o, p = var_in1[i], var_in2[i], var_out[i], var_p[i]
                    model_list += [f'{i1} + {i2} - {o} >= 0', f'{i1} + {i2} - {p} >= 0',  f'- {i1} + {p} >= 0', f'- {i2} + {p} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_p))
                self.weight = [" + ".join(var_p)]
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

class bitwiseXOR(BinaryOperator):  # Operator for the bitwise XOR operation: compute the bitwise XOR on the two input variables towards the output variable 
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
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
            
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = []
                for i in range(len(var_in1)):
                    i1, i2, o = var_in1[i],var_in2[i],var_out[i]
                    model_list += [f'{i1} {i2} -{o}', f'{i1} -{i2} {o}', f'-{i1} {i2} {o}', f'-{i1} -{i2} -{o}']
                return model_list        
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            model_list = []
            if self.model_version == "diff_0" or self.model_version == "DEFAULT": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                if self.mat:
                    var_d = []
                    for i in range(len(self.mat)): 
                        if self.mat[i][0] and self.mat[i][1]:
                            d = self.ID + '_d_' + str(i)
                            var_d += [d]
                            i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                            model_list += [f'{i1} + {i2} + {o} - 2 {d} >= 0', f'{i1} + {i2} + {o} <= 2', f'{d} - {i1} >= 0', f'{d} - {i2} >= 0', f'{d} - {o} >= 0']
                        elif self.mat[i][0]: model_list += [f'{var_in1[i]} - {var_out[i]} = 0']
                        elif self.mat[i][1]: model_list += [f'{var_in2[i]} - {var_out[i]} = 0']   
                        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")                         
                else:
                    var_d = [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                    for i in range(len(var_in1)): 
                        i1, i2, o, d = var_in1[i], var_in2[i], var_out[i], var_d[i]
                        model_list += [f'{i1} + {i2} + {o} - 2 {d} >= 0', f'{i1} + {i2} + {o} <= 2', f'{d} - {i1} >= 0', f'{d} - {i2} >= 0', f'{d} - {o} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                return model_list
            elif self.model_version == "diff_1": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                if self.mat:
                    for i in range(len(self.mat)): 
                        if self.mat[i][0] and self.mat[i][1]:
                            i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                            model_list += [f'{i1} + {i2} - {o} >= 0', f'{i2} + {o} - {i1} >= 0', f'{i1} + {o} - {i2} >= 0', f'{i1} + {i2} + {o} <= 2']
                        elif self.mat[i][0]: model_list += [f'{var_in1[i]} - {var_out[i]} = 0']
                        elif self.mat[i][1]: model_list += [f'{var_in2[i]} - {var_out[i]} = 0']   
                        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")                         
                else:
                    for i in range(len(var_in1)):  
                        i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                        model_list += [f'{i1} + {i2} - {o} >= 0', f'{i2} + {o} - {i1} >= 0', f'{i1} + {o} - {i2} >= 0', f'{i1} + {i2} + {o} <= 2']
                    model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            elif self.model_version == "diff_2": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('in', 1, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                if self.mat:
                    var_d = []
                    for i in range(len(var_in1)): 
                        if self.mat[i][0] and self.mat[i][1]:
                            d = self.ID + '_d_' + str(i)
                            var_d += [d]
                            i1, i2, o = var_in1[i], var_in2[i], var_out[i]
                            model_list += [f'{i1} + {i2} + {o} - 2 {d} = 0']
                        elif self.mat[i][0]: model_list += [f'{var_in1[i]} - {var_out[i]} = 0']
                        elif self.mat[i][1]: model_list += [f'{var_in2[i]} - {var_out[i]} = 0']   
                        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")                         
                else:
                    var_d = [self.ID + '_d_' + str(i) for i in range(self.input_vars[0].bitsize)]
                    for i in range(len(var_in1)):  
                        i1, i2, o, d = var_in1[i], var_in2[i], var_out[i], var_d[i]
                        model_list += [f'{i1} + {i2} + {o} - 2 {d} = 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                return model_list
            elif self.model_version == "truncated_diff": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('in', 1, unroll)], [self.get_var_ID('out', 0, unroll)]
                var_d = [self.ID + '_d']
                i1, i2, o, d = var_in1[0], var_in2[0], var_out[0], var_d[0]
                model_list += [f'{i1} + {i2} + {o} - 2 {d} >= 0', f'{d} - {i1} >= 0', f'{d} - {i2} >= 0', f'{d} - {o} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out + var_d))
                return model_list
            elif self.model_version == "truncated_diff_1": 
                var_in1, var_in2, var_out = [self.get_var_ID('in', 0, unroll)], [self.get_var_ID('in', 1, unroll)], [self.get_var_ID('out', 0, unroll)]
                i1, i2, o = var_in1[0], var_in2[0], var_out[0]
                model_list += [f'{i1} + {i2} - {o} >= 0', f'{i2} + {o} - {i1} >= 0', f'{i1} + {o} - {i2} >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_in1 + var_in2 + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

class bitwiseNOT(UnaryOperator): # Operator for the bitwise NOT operation: compute the bitwise NOT operation on the input variable towards the output variable 
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1)] 
        elif implementation_type == 'c': 
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ' ^ ' + hex(2**self.input_vars[0].bitsize - 1) + ';']
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
            
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': 
            if self.model_version == "diff_0" or self.model_version =="DEFAULT": 
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                return [clause for vin, vout in zip(var_in, var_out) for clause in (f"-{vin} {vout}", f"{vin} -{vout}")]
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': 
            if self.model_version == "diff_0" or self.model_version =="DEFAULT": 
                var_in, var_out = [self.get_var_ID('in', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)], [self.get_var_ID('out', 0, unroll) + '_' + str(i) for i in range(self.input_vars[0].bitsize)]
                model_list = [f'{var_in[i]} - {var_out[i]} = 0' for i in range(len(var_in))]
                model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class AESround(Operator): # Operator for the AES round
    def __init__(self, input_vars, output_vars, ID = None, subkey=None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.subkey = subkey
        pass # TODO
        
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python': 
            pass
        elif implementation_type == 'c': 
            pass
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='python', unroll=False):
        if model_type == 'sat': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'milp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

        
class CustomOP(Operator):   # generic custom operator (to be defined by the user)
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID)
        pass # TODO
        
