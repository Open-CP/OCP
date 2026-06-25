from operators.operators import Operator, Equal, RaiseExceptionVersionNotExisting
from operators.Sbox import AES_Sbox
from operators.matrix import Matrix
from operators.boolean_operators import XOR
from variables.variables import Variable


class AESround(Operator): # One full AES round (16-byte state) as a composite operator: SubBytes -> ShiftRows -> MixColumns -> AddRoundKey (only if a subkey is given). 
    def __init__(self, input_vars, output_vars, subkey=None, ID = None):
        if len(input_vars) != 16: raise Exception(str(self.__class__.__name__) + ": Input does not contain exactly 16 elements")
        if len(output_vars) != 16: raise Exception(str(self.__class__.__name__) + ": Output does not contain exactly 16 elements")
        if subkey is not None and len(subkey) != 16: raise Exception(str(self.__class__.__name__) + ": Subkey does not contain exactly 16 elements")
        super().__init__(input_vars, output_vars, ID = ID)
        self.subkey = subkey
        self.layers = []
        self.vars = []
        base_id = self.ID if self.ID is not None else self.__class__.__name__

        # create intermediate variables
        self.vars.append(input_vars)
        suffixes = ["_SB", "_SR"] + (["_MC"] if subkey is not None else [])
        for suffix in suffixes:
            temp_vars = []
            for i, var in enumerate(input_vars):
                var_id = var.ID if var.ID is not None else f"{base_id}_in_{i}"
                new_var = Variable(var.bitsize, ID=var_id + suffix)
                temp_vars.append(new_var)
            self.vars.append(temp_vars)
        self.vars.append(output_vars)

        # create intermediate layers
        self.layers.append([AES_Sbox([self.vars[0][i]], [self.vars[1][i]], f"{base_id}_SB_{i}") for i in range(16)]) # S-box Layer

        perm_s = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11] # ShiftRows Layer
        self.layers.append([Equal([self.vars[1][perm_s[i]]], [self.vars[2][i]], f"{base_id}_SR_{i}") for i in range(16)])

        mat = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]] # MixColumns Layer
        for i, indexes in enumerate([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]):
            self.layers.append([Matrix("aes_matrix", [self.vars[2][x] for x in indexes], [self.vars[3][x] for x in indexes], mat=mat, polynomial="0x1B", ID=f"{base_id}_MC_{i}")])

        if subkey is not None: # AddRoundKey Layer (only if subkey is provided)
            self.layers.append([XOR([self.vars[3][i], subkey[i]], [self.vars[4][i]], f"{base_id}_AK_{i}") for i in range(16)])

    def generate_implementation_header_unique(self, implementation_type='python'):
        if implementation_type == 'python':
            model_list = ["#Galois Field Multiplication Macro", "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
        elif implementation_type == 'c':
            model_list = ["//Galois Field Multiplication Macro", "#define GMUL(a, b, p, d) ({ \\", "\tunsigned int result = 0; \\", "\tunsigned int temp_a = a; \\", "\tunsigned int temp_b = b; \\", "\twhile (temp_b > 0) { \\", "\t\tif (temp_b & 1) \\", "\t\t\tresult ^= temp_a; \\", "\t\ttemp_a <<= 1; \\", "\t\tif (temp_a & (1 << d)) \\", "\t\t\ttemp_a ^= p; \\", "\t\ttemp_b >>= 1; \\", "\t} \\", "\tresult & ((1 << d) - 1); \\","})"];
        else:
            raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")
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

    def _get_inner_model_version(self, cons):
        if self.model_version is None:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "milp/sat")
        prefix = self.__class__.__name__ + "_"
        if not self.model_version.startswith(prefix):
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "milp/sat")

        suffix = self.model_version[len(prefix):]
        is_sbox = "Sbox" in cons.__class__.__name__
        if not is_sbox: # Non-S-box operators (Equal/Matrix/XOR) only apply the base versions (e.g. "XORDIFF", "LINEAR"), so strip the trailing "_A"/"_PR" tag.
            for tag in ("_PR", "_P", "_A"):
                if suffix.endswith(tag):
                    suffix = suffix[:-len(tag)]
                    break
        return cons.__class__.__name__ + "_" + suffix

    def generate_model(self, model_type='sat'):
        model_list = []
        self.weight = []
        if model_type == 'sat' or model_type == 'milp':
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self._get_inner_model_version(cons)
                    if cons.__class__.__name__ == "Matrix" and model_type == 'milp' and ("TRUNCATEDDIFF" in self.model_version or "TRUNCATEDLINEAR" in self.model_version):
                        model_list += cons.generate_model(model_type, branch_num=5) # Use the branch_num=5 constraint for the MixColumns layer in the truncated differential/linear models.
                    else:
                        model_list += cons.generate_model(model_type)
                    if hasattr(cons, 'weight'):
                        self.weight += cons.weight
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
