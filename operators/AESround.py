import copy
from operators.operators import Operator, Equal, RaiseExceptionVersionNotExisting
from operators.Sbox import AES_Sbox
from operators.matrix import Matrix
from operators.boolean_operators import XOR


class AESround(Operator): # Operator for the AES round
    def __init__(self, input_vars, output_vars, subkey=None, ID = None):
        if len(input_vars) != 16: raise Exception(str(self.__class__.__name__) + ": your input does not contain exactly 16 element")
        super().__init__(input_vars, output_vars, ID = ID)
        self.subkey = subkey
        self.layers = []
        self.vars = []

        # create intermediate variables
        self.vars.append(input_vars)
        suffixes = ["_SB", "_SR"] + (["_MC"] if subkey else [])
        for suffix in suffixes:
            temp_vars = copy.deepcopy(input_vars)
            for var in temp_vars:
                var.ID += suffix
            self.vars.append(temp_vars)
        self.vars.append(output_vars)

        # create intermediate layers
        self.layers.append([AES_Sbox([self.vars[0][i]], [self.vars[1][i]], ID + "_SB") for i in range(16)]) # S-box Layer

        perm_s = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11] # ShiftRows Layer
        self.layers.append([Equal([self.vars[1][perm_s[i]]], [self.vars[2][i]], ID + "_SR") for i in range(16)])

        mat = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]] # MixColumns Layer
        for indexes in [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]:
            self.layers.append([Matrix("MC", [self.vars[2][x] for x in indexes], [self.vars[3][x] for x in indexes], mat=mat, polynomial="0x1B", ID=ID + "_MC")])

        if subkey: # AddRoundKey Layer (only if subkey is provided)
            self.layers.append([XOR([self.vars[3][i], subkey[i]], [self.vars[4][i]], ID + "_AK") for i in range(16)])        
        
    def generate_implementation_header_unique(self, implementation_type='python'):
        if implementation_type == 'python': 
            model_list = ["#Galois Field Multiplication Macro", "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
        elif implementation_type == 'c': 
            model_list = ["//Galois Field Multiplication Macro", "#define GMUL(a, b, p, d) ({ \\", "\tunsigned int result = 0; \\", "\tunsigned int temp_a = a; \\", "\tunsigned int temp_b = b; \\", "\twhile (temp_b > 0) { \\", "\t\tif (temp_b & 1) \\", "\t\t\tresult ^= temp_a; \\", "\t\ttemp_a <<= 1; \\", "\t\tif (temp_a & (1 << d)) \\", "\t\t\ttemp_a ^= p; \\", "\t\ttemp_b >>= 1; \\", "\t} \\", "\tresult & ((1 << d) - 1); \\","})"];
        return model_list
    
    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat'):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")