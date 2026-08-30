"""Composite AES round operator.

``AESround`` assembles one AES round from sub-operators -- SubBytes (AES_Sbox), ShiftRows
(Equal wiring), MixColumns (Matrix over GF(2^8)), and an optional AddRoundKey (XOR) -- and
delegates code/model generation to them, propagating its model version to each inner operator
(the S-box keeps the full version suffix; the other operators take the base version).
"""

from operators.operators import Operator, Equal, RaiseExceptionVersionNotExisting, raise_unknown_implementation_type, raise_unknown_model_type
from operators.Sbox import AES_Sbox
from operators.matrix import Matrix
from operators.boolean_operators import XOR
from variables.variables import Variable


def _iter_layer_constraints(layers):
    for layer in layers:
        for constraint in layer:
            yield constraint


class AESround(Operator):
    """One AES round (SubBytes/ShiftRows/MixColumns/AddRoundKey) over a 16-byte state."""

    # The round propagates each version to its inner operators (see _inner_model_version):
    # S-boxes keep the full suffix, other operators take the base version.
    _MODEL_VERSIONS = (
        "XORDIFF", "XORDIFF_A", "XORDIFF_PR",
        "LINEAR", "LINEAR_A", "LINEAR_PR",
        "TRUNCATEDDIFF", "TRUNCATEDDIFF_A",
        "TRUNCATEDLINEAR", "TRUNCATEDLINEAR_A",
    )
    SUPPORTED_MODEL_VERSIONS = {
        "sat":  _MODEL_VERSIONS,
        "milp": _MODEL_VERSIONS,
    }
    SUPPORTED_IMPLEMENTATIONS = ("python", "c")

    def __init__(self, input_vars, output_vars, subkey=None, ID = None):
        if len(input_vars) != 16:
            raise ValueError(f"{self.__class__.__name__}: expected exactly 16 input variables, got {len(input_vars)}")
        if len(output_vars) != 16:
            raise ValueError(f"{self.__class__.__name__}: expected exactly 16 output variables, got {len(output_vars)}")
        if subkey is not None and len(subkey) != 16:
            raise ValueError(f"{self.__class__.__name__}: expected exactly 16 subkey variables, got {len(subkey)}")
        super().__init__(input_vars, output_vars, ID = ID)
        self.subkey = subkey
        self.layers = []
        self.vars = []
        base_id = self.ID if self.ID is not None else self.__class__.__name__

        # create intermediate variables (each layer operator gets a unique per-position ID so that
        # per-operator auxiliary variables, e.g. the MixColumns branch dummies, never collide)
        self.vars.append(input_vars)
        suffixes = ["_SB", "_SR"] + (["_MC"] if subkey is not None else [])
        for suffix in suffixes:
            temp_vars = []
            for i, var in enumerate(input_vars):
                var_id = var.ID if var.ID is not None else f"{base_id}_in_{i}"
                temp_vars.append(Variable(var.bitsize, ID=var_id + suffix))
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
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)
        return model_list

    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for cons in _iter_layer_constraints(self.layers):
            if [cons.__class__.__name__] not in header_set:
                header_set.append([cons.__class__.__name__])
                header = cons.generate_implementation_header(implementation_type)
                if header is not None:
                    code_list += header
        return code_list

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        if implementation_type == 'python' or implementation_type == 'c':
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for cons in _iter_layer_constraints(self.layers):
                code_list += cons.generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else:
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def _inner_model_version(self, cons):
        """Map this round's ``model_version`` onto an inner layer constraint.

        The round-level suffix (e.g. ``XORDIFF_PR``) is an S-box tag. S-box constraints
        keep the full suffix; the other operators (Equal/Matrix/XOR) only implement the
        base versions (``XORDIFF``/``LINEAR``/``TRUNCATEDDIFF``/``TRUNCATEDLINEAR``), so a
        trailing ``_A``/``_P``/``_PR`` is stripped for them.
        """
        prefix = self.__class__.__name__ + "_"
        suffix = self.model_version[len(prefix):]
        if "Sbox" not in cons.__class__.__name__:
            for tag in ("_PR", "_P", "_A"):
                if suffix.endswith(tag):
                    suffix = suffix[:-len(tag)]
                    break
        return cons.__class__.__name__ + "_" + suffix

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        self.weight = []
        if model_type == 'sat' or model_type == 'milp':
            # The MixColumns matrix uses its branch-number model in the truncated settings.
            use_branch_number_model = "TRUNCATEDDIFF" in self.model_version or "TRUNCATEDLINEAR" in self.model_version
            for cons in _iter_layer_constraints(self.layers):
                cons.model_version = self._inner_model_version(cons)
                if cons.__class__.__name__ == "Matrix" and use_branch_number_model:
                    model_list += cons.generate_model(model_type, branch_num=5)
                else:
                    model_list += cons.generate_model(model_type)
                if hasattr(cons, 'weight'):
                    self.weight += cons.weight
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else:
            raise_unknown_model_type(self.__class__.__name__, model_type)
