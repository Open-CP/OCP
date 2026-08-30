"""Operator (constraint) nodes of OCP's graph model.

An Operator relates a group of input variables to a group of output variables. 
Every concrete operator provides two code generators:

- :meth:`Operator.generate_implementation` -- executable code for the operator in a
  target language (python/c/verilog), selected by the operator's ``implementation_type``.
- :meth:`Operator.generate_model` -- SAT/MILP constraint strings for a cryptanalysis
  model (differential / linear / integral), selected by the operator's ``model_version``.
"""

from abc import ABC, abstractmethod
from tools.operator_constraints import gen_xor_constraints, gen_word_xor_constraints, gen_nxor_constraints, gen_word_nxor_constraints, binary_declaration, integer_declaration, gen_equivalence_constraints


def RaiseExceptionVersionNotExisting(class_name, model_version, model_type):
    """Raise for a ``model_version`` an operator does not model under ``model_type``."""
    raise ValueError(f"{class_name}: version {model_version} not existing for {model_type}")


def raise_unknown_model_type(class_name, model_type, context=None):
    """Raise for an unsupported ``model_type`` (i.e. not 'sat'/'milp'/'cp')."""
    context_message = f" for {context}" if context is not None else ""
    raise ValueError(f"{class_name}: unknown model type '{model_type}'{context_message}")


def raise_unknown_implementation_type(class_name, implementation_type):
    """Raise for an unsupported ``implementation_type`` (i.e. not 'python'/'c'/'verilog')."""
    raise ValueError(f"{class_name}: unknown implementation type '{implementation_type}'")


def require_variable_count(class_name, variables, expected_count, side):
    """Validate that ``side`` ('input'/'output') has exactly ``expected_count`` variables."""
    if len(variables) != expected_count:
        raise ValueError(f"{class_name}: expected exactly {expected_count} {side} variable(s), got {len(variables)}")


def require_min_variable_count(class_name, variables, min_count, side):
    """Validate that ``side`` ('input'/'output') has at least ``min_count`` variables."""
    if len(variables) < min_count:
        raise ValueError(f"{class_name}: expected at least {min_count} {side} variable(s), got {len(variables)}")


def require_same_bitsize(class_name, left_var, right_var, message):
    """Validate that two variables share a bit-width, raising ``message`` otherwise."""
    if left_var.bitsize != right_var.bitsize:
        raise ValueError(f"{class_name}: {message}")


class Operator(ABC):
    """Abstract base for every operator (constraint) node in the cipher graph.

    An Operator relates its ``input_vars`` to its ``output_vars`` and knows how to
    emit both an executable implementation and a cryptanalysis model of that relation.

    Attributes:
        input_vars (list): The operator's input variables (each entry a Variable, or a
            list of Variables for word/vector operands).
        output_vars (list): The operator's output variables (each entry a Variable, or a
            list of Variables for word/vector operands).
        model_version (str): The model to emit, as ``'<ClassName>_<SUFFIX>'`` (e.g.
            ``'XOR_XORDIFF'``); set before calling :meth:`generate_model`.
        ID (str): Identifier for the operator instance.
        is_ghost (bool): Whether this operator was marked ghost during dead-end removal.
    """

    # Declared, enforced capability surface (see the check_* methods below).
    SUPPORTED_MODEL_VERSIONS = {}                          # {model_type: version suffixes generate_model accepts}
    SUPPORTED_IMPLEMENTATIONS = ("python", "c", "verilog")  # languages generate_implementation supports

    def check_supported_model_version(self, model_type):
        """Reject a ``model_version`` not declared for ``model_type`` in SUPPORTED_MODEL_VERSIONS.

        SUPPORTED_MODEL_VERSIONS maps each model_type ('sat'/'milp') to the version suffixes it
        accepts; each is prefixed with the class name to form the ``'<ClassName>_<SUFFIX>'`` value
        that ``generate_model`` compares ``model_version`` against.
        """
        supported = self.SUPPORTED_MODEL_VERSIONS.get(model_type, ())
        if not supported or self.model_version is None:
            return
        allowed = {f"{self.__class__.__name__}_{suffix}" for suffix in supported}
        if self.model_version not in allowed:
            RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)

    def check_supported_implementation(self, implementation_type):
        """Reject an ``implementation_type`` not declared in SUPPORTED_IMPLEMENTATIONS."""
        if self.SUPPORTED_IMPLEMENTATIONS and implementation_type not in self.SUPPORTED_IMPLEMENTATIONS:
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def __init__(self, input_vars, output_vars, model_version=None, ID=None):
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.model_version = model_version
        self.ID = ID
        self.is_ghost = False   # set when the operator is pruned during dead-end removal

        # For this new operator, update the connected_vars list of each input and output variable
        if self.__class__.__name__!="NoneOperator":
            for var_in in input_vars:
                for var_out in output_vars:
                    var_in.connected_vars.append((var_out,self,'in'))
                    var_out.connected_vars.append((var_in,self,'out'))

    def format_display(self):
        """Return a multi-line string describing the operator's ID and its operands."""
        lines = [f"ID: {self.ID}", "Input:"]
        for input_var in self.input_vars:
            if not isinstance(input_var, list):
                lines.append(input_var.format_display())
            else:
                lines.extend(var.format_display() for var in input_var)

        lines.append("Output:")
        for output_var in self.output_vars:
            if not isinstance(output_var, list):
                lines.append(output_var.format_display())
            else:
                lines.extend(var.format_display() for var in output_var)
        return "\n".join(lines)

    def display(self, output_func=None):
        """Print (or route via ``output_func``) the operator description; return its class name."""
        text = self.format_display()
        if output_func is None:
            print(text)
        else:
            output_func(text)
        return self.__class__.__name__

    def get_var_ID(self, in_out, index, unroll=False):
        """Return the ID of the input/output ('in'/'out') variable at ``index``.

        When ``unroll`` is False the round index is stripped from the ID (so the same
        code can be reused across rounds); when True the fully-qualified ID is returned.
        """
        if in_out == 'out':
            return self.output_vars[index].ID if unroll else self.output_vars[index].remove_round_from_ID()
        elif in_out == 'in':
            return self.input_vars[index].ID if unroll else self.input_vars[index].remove_round_from_ID()
        else:
            raise ValueError(f"{self.__class__.__name__}: unknown in_out type '{in_out}'")

    def get_header_ID(self):
        """Return ``[class name, model_version]``, identifying the operator's model header."""
        return [self.__class__.__name__, self.model_version]

    def generate_implementation_header(self, implementation_type='python'):
        """Return code emitted once before the operator's implementation (default: none)."""
        return None

    def get_var_model(self, in_out, index, bitwise=True, dim=1):
        """Return the model-variable name(s) for the input/output variable at ``index``.

        With ``bitwise`` True and a multi-bit variable, one name per bit is returned;
        ``dim`` > 1 additionally expands a trailing dimension (e.g. parallel words).
        """
        if in_out == 'in':
            var = self.input_vars[index]
        elif in_out == 'out':
            var = self.output_vars[index]
        else:
            raise ValueError(f"{self.__class__.__name__}: unknown in_out type '{in_out}'")
        if bitwise and var.bitsize > 1:
            return [f"{var.ID}_{i}_{j}" for i in range(var.bitsize) for j in range(dim)] if dim > 1 else [f"{var.ID}_{i}" for i in range(var.bitsize)]
        else:
            return [f"{var.ID}_{j}" for j in range(dim)] if dim > 1 else [f"{var.ID}"]

    @abstractmethod
    def generate_implementation(self, implementation_type='python'):
        """Return executable code (a list of source lines) computing this operator.

        ``implementation_type`` selects the target language (one of
        ``SUPPORTED_IMPLEMENTATIONS``, typically 'python'/'c'/'verilog').
        """
        pass

    @abstractmethod
    def generate_model(self, model_type='sat'):
        """Return the cryptanalysis-model constraints (a list of strings) for this operator.

        ``model_type`` is 'sat' or 'milp'; the cryptanalysis technique (differential / linear /
        integral) is selected by ``self.model_version`` (see ``SUPPORTED_MODEL_VERSIONS``).
        Implementations may also set ``self.weight`` as a side effect.
        """
        pass


class CastingOperator(Operator):
    """Abstract base for casting operators (casting from one type to another).

    Casting operators must preserve the total bit width. Concrete subclasses are
    expected to define implementation/model generation for their specific layout.
    """

    def __init__(self, input_vars, output_vars, ID = None):
        if sum(input_var.bitsize for input_var in input_vars) != sum(output_var.bitsize for output_var in output_vars):
            raise ValueError("CastingOperator: the total input size does not match the total output size")
        super().__init__(input_vars, output_vars, ID = ID)
        pass   # TODO


class CastingWordtoBitVector(CastingOperator):
    """Abstract base for casting a bit word to a vector of bits."""
    pass   # TODO


class UnaryOperator(Operator):
    """Generic operator with a single input and output of the same bit-width."""

    def __init__(self, input_vars, output_vars, ID = None):
        require_variable_count(self.__class__.__name__, input_vars, 1, "input")
        require_variable_count(self.__class__.__name__, output_vars, 1, "output")
        require_same_bitsize(self.__class__.__name__, input_vars[0], output_vars[0], "input and output sizes do not match")
        super().__init__(input_vars, output_vars, ID = ID)


class BinaryOperator(Operator):
    """Generic operator with two inputs and one output, all of the same bit-width."""

    def __init__(self, input_vars, output_vars, ID = None):
        require_variable_count(self.__class__.__name__, input_vars, 2, "input")
        require_variable_count(self.__class__.__name__, output_vars, 1, "output")
        require_same_bitsize(self.__class__.__name__, input_vars[0], input_vars[1], "input sizes do not match")
        require_same_bitsize(self.__class__.__name__, input_vars[0], output_vars[0], "input and output sizes do not match")
        super().__init__(input_vars, output_vars, ID = ID)


class NoneOperator(Operator):
    """Ghost operator that does nothing; a placeholder emitting no code or constraints."""

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        return []

    def generate_model(self, model_type='sat'):
        return []


class CopyOperator(Operator):
    """Duplicate one input into multiple outputs (b_0, b_1, ..., b_n = a)."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR", "INTEGRAL_TWOSUBSET"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        require_variable_count(self.__class__.__name__, input_vars, 1, "input")
        require_min_variable_count(self.__class__.__name__, output_vars, 2, "output")
        super().__init__(input_vars, output_vars, ID=ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        in_id = self.get_var_ID('in', 0, unroll)
        if implementation_type == 'python':
            return [f"{self.get_var_ID('out', j, unroll)} = {in_id}" for j in range(len(self.output_vars))]
        elif implementation_type == 'c':
            return [f"{self.get_var_ID('out', j, unroll)} = {in_id};" for j in range(len(self.output_vars))]
        elif implementation_type == 'verilog':
            return [f"assign {self.get_var_ID('out', j, unroll)} = {in_id};" for j in range(len(self.output_vars))]
        else:
            raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        model_list = []
        if model_type in ['sat', 'milp']:
            # XORDIFF: a difference copies unchanged, so every output bit equals the input bit.
            if self.model_version == self.__class__.__name__ + "_XORDIFF":
                var_in = self.get_var_model("in", 0)
                var_out = [self.get_var_model("out", i) for i in range(len(self.output_vars))]
                for output_vars in var_out:
                    model_list.extend(gen_equivalence_constraints(output_vars, var_in, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, *var_out))
                return model_list
            # TRUNCATEDDIFF: the same copy, but at word-level (activity) granularity.
            elif self.model_version == self.__class__.__name__ + "_TRUNCATEDDIFF":
                var_in = self.get_var_model("in", 0, bitwise=False)
                var_out = [self.get_var_model("out", i, bitwise=False) for i in range(len(self.output_vars))]
                for output_vars in var_out:
                    model_list.extend(gen_equivalence_constraints(output_vars, var_in, model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, *var_out))
                return model_list
            # LINEAR: the input mask is the XOR of the output masks
            elif self.model_version == self.__class__.__name__ + "_LINEAR":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                integer_dummies = []
                if len(var_out) == 2: # Two outputs: out1, out2 = in
                    for i in range(self.input_vars[0].bitsize):
                        model_list.extend(gen_xor_constraints(var_in[i], var_out[0][i], var_out[1][i], model_type))
                elif len(var_out) >= 3: # n outputs: out1, out2, ..., outn = in
                    for i in range(self.input_vars[0].bitsize):
                        if model_type == 'milp':
                            v_dummy = f"{self.ID}_d_{i}"
                            integer_dummies.append(v_dummy)  # n-XOR version 0 uses an integer dummy
                        else:
                            v_dummy = None
                        model_list.extend(gen_nxor_constraints([var_out[j][i] for j in range(len(var_out))], var_in[i], model_type=model_type, v_dummy=v_dummy))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, *var_out))
                    if integer_dummies:
                        model_list.append(integer_declaration(integer_dummies))
                return model_list
            # TRUNCATEDLINEAR (2 outputs): the input activity is the word-XOR of the two output activities.
            elif len(self.output_vars) == 2 and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out1, var_out2 = (self.get_var_model("in", 0, bitwise=False),  self.get_var_model("out", 0, bitwise=False), self.get_var_model("out", 1, bitwise=False))
                model_list.extend(gen_word_xor_constraints(var_out1[0], var_out2[0], var_in[0], model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, var_out1, var_out2))
                return model_list
            # TRUNCATEDLINEAR (n>=3 outputs): word-level n-XOR of the output activities into the input.
            elif len(self.output_vars) >= 3 and self.model_version == self.__class__.__name__ + "_TRUNCATEDLINEAR":
                var_in, var_out = (self.get_var_model("in", 0, bitwise=False), [self.get_var_model("out", i, bitwise=False) for i in range(len(self.output_vars))])
                out_words = [var_out[j][0] for j in range(len(var_out))]
                model_list.extend(gen_word_nxor_constraints(out_words, var_in[0], model_type))
                if model_type == 'milp':
                    model_list.append(binary_declaration(out_words, [var_in[0]]))
                return model_list
            # INTEGRAL_TWOSUBSET: the input division-property bit splits over the outputs
            # (input bit = sum of the corresponding output bits).
            elif model_type == "milp" and self.model_version == self.__class__.__name__ + "_INTEGRAL_TWOSUBSET":
                var_in, var_out = (self.get_var_model("in", 0), [self.get_var_model("out", i) for i in range(len(self.output_vars))])
                for i in range(self.input_vars[0].bitsize):
                    model_list.append(f"{var_in[i]} - " + " - ".join(var_out[j][i] for j in range(len(var_out))) + " = 0")
                model_list.append(binary_declaration(var_in, sum(var_out, [])))
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class Equal(UnaryOperator):
    """Assign equality between the input and output variable (same bit-width)."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR"),
        "milp": ("XORDIFF", "LINEAR", "TRUNCATEDDIFF", "TRUNCATEDLINEAR", "INTEGRAL_TWOSUBSET"),
    }

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        if implementation_type == 'python':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll)]
        elif implementation_type == 'c':
            return [self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ';']
        elif implementation_type == 'verilog':
            return ["assign " + self.get_var_ID('out', 0, unroll) + ' = ' + self.get_var_ID('in', 0, unroll) + ';']
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        if model_type in ('sat', 'milp'):
            # Each branch only selects the variable granularity; the equivalence and the single
            # Binary declaration are emitted once below.
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR"]:
                var_in, var_out = self.get_var_model("in", 0), self.get_var_model("out", 0)
            elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDLINEAR"]:
                var_in, var_out = self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False)
            elif model_type == 'milp' and self.model_version == self.__class__.__name__ + "_INTEGRAL_TWOSUBSET":
                var_in, var_out = self.get_var_model("in", 0), self.get_var_model("out", 0)
            else:
                RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
            model_list = gen_equivalence_constraints(var_in, var_out, model_type)
            if model_type == 'milp':
                model_list.append(binary_declaration(var_in, var_out))
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class Rot(UnaryOperator):
    """Bitwise rotation of the input by ``amount`` bits in ``direction`` ('l' or 'r')."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "LINEAR", "INTEGRAL_TWOSUBSET"),
    }

    def __init__(self, input_vars, output_vars, direction, amount, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        if direction not in {'l', 'r'}:
            raise ValueError(f"{self.__class__.__name__}: direction must be 'l' or 'r', got '{direction}'")
        self.direction = direction
        if amount <= 0 or amount >= input_vars[0].bitsize:
            raise ValueError(f"{self.__class__.__name__}: amount must satisfy 0 < amount < bitsize ({input_vars[0].bitsize}), got {amount}")
        self.amount = amount

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        lhs = self.get_var_ID('out', 0, unroll)
        source = self.get_var_ID('in', 0, unroll)
        bitsize = self.input_vars[0].bitsize
        macro = "ROTR" if self.direction == 'r' else "ROTL"

        if implementation_type == 'python':
            return [f"{lhs} = {macro}({source}, {self.amount}, {bitsize})"]
        elif implementation_type == 'c':
            return [f"{lhs} = {macro}({source}, {self.amount}, {bitsize});"]
        elif implementation_type == 'verilog':
            return [f"assign {lhs} = `{macro}({source}, {self.amount}, {bitsize});"]
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_implementation_header_unique(self, implementation_type='python'):
        """Return the rotation-macro definitions (ROTL/ROTR) emitted once per implementation.

        ``_unique`` marks a header that must be emitted at most once even when many Rot
        operators are present, so the macros are not redefined per operator.
        """
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
        self.check_supported_model_version(model_type)
        if model_type in ('sat', 'milp'):
            var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
            n = len(var_in)
            # A rotation is a bit permutation: every version is the equivalence between input bit i
            # and its rotated output bit (INTEGRAL is milp-only, guaranteed by the guard).
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_INTEGRAL_TWOSUBSET"]:
                if self.direction == 'r':   # var_in[i] == var_out[(i + amount) % n]
                    left, right = var_in, [var_out[(i + self.amount) % n] for i in range(n)]
                else:                       # var_in[(i + amount) % n] == var_out[i]
                    left, right = [var_in[(i + self.amount) % n] for i in range(n)], var_out
                model_list = gen_equivalence_constraints(left, right, model_type)
                if model_type == 'milp':
                    model_list.append(binary_declaration(var_in, var_out))
                return model_list
            else:
                RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class Shift(UnaryOperator):
    """Bitwise shift of the input by ``amount`` bits in ``direction`` ('l' or 'r')."""

    SUPPORTED_MODEL_VERSIONS = {
        "sat":  ("XORDIFF", "LINEAR"),
        "milp": ("XORDIFF", "LINEAR"),
    }

    def __init__(self, input_vars, output_vars, direction, amount, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        if direction not in {'l', 'r'}:
            raise ValueError(f"{self.__class__.__name__}: direction must be 'l' or 'r', got '{direction}'")
        self.direction = direction
        if amount <= 0 or amount >= input_vars[0].bitsize:
            raise ValueError(f"{self.__class__.__name__}: amount must satisfy 0 < amount < bitsize ({input_vars[0].bitsize}), got {amount}")
        self.amount = amount

    def generate_implementation(self, implementation_type='python', unroll=False):
        self.check_supported_implementation(implementation_type)
        lhs = self.get_var_ID('out', 0, unroll)
        source = self.get_var_ID('in', 0, unroll)
        shift_operator = ">>" if self.direction == 'r' else "<<"
        bitsize = self.input_vars[0].bitsize

        if implementation_type == 'python':
            return [f"{lhs} = ({source} {shift_operator} {self.amount}) & (2**{bitsize} - 1)"]
        elif implementation_type == 'c':
            return [f"{lhs} = ({source} {shift_operator} {self.amount}) & ((1<<{bitsize}) - 1);"]
        elif implementation_type == 'verilog':
            return [f"assign {lhs} = ({source} {shift_operator} {self.amount}) & ((1<<{bitsize}) - 1);"]
        else: raise_unknown_implementation_type(self.__class__.__name__, implementation_type)

    def generate_model(self, model_type='sat'):
        self.check_supported_model_version(model_type)
        var_in, var_out = (self.get_var_model("in", 0), self.get_var_model("out", 0))
        n = len(var_in)
        s = self.amount
        if model_type == 'sat':
            # XORDIFF: zero the shifted-out output bits; window maps in->out; input tail is free.
            if self.direction == 'r' and self.model_version == self.__class__.__name__ + "_XORDIFF":
                model_list = [f"-{var_out[i]}" for i in range(s)]
                model_list += gen_equivalence_constraints(var_in[:n - s], var_out[s:], "sat")
                model_list += [f"{var_in[i]} -{var_in[i]}" for i in range(n - s, n)]
                return model_list
            elif self.direction == 'l' and self.model_version == self.__class__.__name__ + "_XORDIFF":
                model_list = [f"{var_in[i]} -{var_in[i]}" for i in range(s)]
                model_list += gen_equivalence_constraints(var_in[s:], var_out[:n - s], "sat")
                model_list += [f"-{var_out[i]}" for i in range(n - s, n)]
                return model_list

            # LINEAR: zero the shifted-out input bits; the output tail is free.
            elif self.direction == 'r' and self.model_version == self.__class__.__name__ + "_LINEAR":
                model_list = [f"{var_out[i]} -{var_out[i]}" for i in range(s)]
                model_list += gen_equivalence_constraints(var_in[:n - s], var_out[s:], "sat")
                model_list += [f"-{var_in[i]}" for i in range(n - s, n)]
                return model_list

            elif self.direction == 'l' and self.model_version == self.__class__.__name__ + "_LINEAR":
                model_list = [f"-{var_in[i]}" for i in range(s)]
                model_list += gen_equivalence_constraints(var_in[s:], var_out[:n - s], "sat")
                model_list += [f"{var_out[i]} -{var_out[i]}" for i in range(n - s, n)]
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'milp':
            # XORDIFF: zero the shifted-out output bits; window maps in->out; input tail is free.
            if self.direction == 'r' and self.model_version == self.__class__.__name__ + "_XORDIFF":
                model_list = [f'{var_out[i]} = 0' for i in range(s)]
                model_list += [f'{var_in[i]} - {var_out[i + s]} = 0' for i in range(n - s)]
                model_list.append(binary_declaration(var_in, var_out))
                return model_list
            elif self.direction == 'l' and self.model_version == self.__class__.__name__ + "_XORDIFF":
                model_list = [f'{var_in[i + s]} - {var_out[i]} = 0' for i in range(n - s)]
                model_list += [f'{var_out[i]} = 0' for i in range(n - s, n)]
                model_list.append(binary_declaration(var_in, var_out))
                return model_list

            # LINEAR: zero the shifted-out input bits; the output tail is free.
            elif self.direction == 'r' and self.model_version == self.__class__.__name__ + "_LINEAR":
                model_list = [f'{var_in[i]} - {var_out[i + s]} = 0' for i in range(n - s)]
                model_list += [f'{var_in[i]} = 0' for i in range(n - s, n)]
                model_list.append(binary_declaration(var_in, var_out))
                return model_list
            elif self.direction == 'l' and self.model_version == self.__class__.__name__ + "_LINEAR":
                model_list = [f'{var_in[i]} = 0' for i in range(s)]
                model_list += [f'{var_in[i + s]} - {var_out[i]} = 0' for i in range(n - s)]
                model_list.append(binary_declaration(var_in, var_out))
                return model_list
            else: RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(self.__class__.__name__, self.model_version, model_type)
        else: raise_unknown_model_type(self.__class__.__name__, model_type)


class CustomOP(Operator):
    """Abstract base for a generic custom operator, to be defined by the user.

    Subclasses should provide their own implementation and model generation.
    """

    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, ID=ID)
        pass # TODO
