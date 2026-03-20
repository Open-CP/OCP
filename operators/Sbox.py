import math
import os
from pathlib import Path
from operators.operators import Operator, RaiseExceptionVersionNotExisting
from tools.model_constraints import generate_and_save_constraints, gen_constraints_obj_func_from_template

ROOT = Path(__file__).resolve().parents[1]  # this file -> operators -> <ROOT>
BASE_PATH = ROOT / "files/sbox_modeling"
BASE_PATH.mkdir(parents=True, exist_ok=True)

class Sbox(Operator):  # Generic operator assigning a Sbox relationship between the input variable and output variable (must be of same bitsize)
    def __init__(self, input_vars, output_vars, input_bitsize, output_bitsize, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.input_bitsize = input_bitsize
        self.output_bitsize = output_bitsize
        self.table = None
        self.table_inv = None
        self.ddt = None
        self.lat = None

    def computeDDT(self): # Compute the differential Distribution Table (DDT) of the Sbox
        if self.ddt is not None: return self.ddt
        ddt = [[0]*(2**self.output_bitsize) for _ in range(2**self.input_bitsize)]
        for in_diff in range(2**self.input_bitsize):
            for j in range(2**self.input_bitsize):
                out_diff = self.table[j] ^ self.table[j^in_diff]
                ddt[in_diff][out_diff] += 1
        self.ddt = ddt
        return ddt

    def computeLAT(self): # Compute the Linear Approximation Table (LAT) of the S-box.
        if self.lat is not None: return self.lat
        lat = [[0] * 2**self.output_bitsize for _ in range(2**self.input_bitsize)]
        for a in range(2**self.input_bitsize):
            for b in range(2**self.output_bitsize):
                acc = 0
                for x in range(2**self.input_bitsize):
                    ax = bin(a & x).count("1") & 1
                    bs = bin(b & self.table[x]).count("1") & 1
                    acc += 1 if (ax ^ bs) == 0 else -1
                lat[a][b] = acc
        self.lat = lat
        return lat

    def differential_branch_number(self): # Return differential branch number of the S-Box.
        ret = (1 << self.input_bitsize) + (1 << self.output_bitsize)
        for a in range(1 << self.input_bitsize):
            for b in range(1 << self.input_bitsize):
                if a != b:
                    x = a ^ b
                    y = self.table[a] ^ self.table[b]
                    w = bin(x).count('1') + bin(y).count('1')
                    if w < ret: ret = w
        return ret

    def linear_branch_number(self):
        m, n = self.input_bitsize, self.output_bitsize
        lat = self.computeLAT()
        ret = (1 << m) + (1 << n)
        for a in range(1 << m):
            for b in range(1, 1 << n):
                if lat[a][b] != 0:
                    w = bin(a).count("1") + bin(b).count("1")
                    if w < ret:
                        ret = w
        return ret

    def is_bijective(self): # Check if the length of the set of s_box is equal to the length of s_box. The set will contain only unique elements
        return len(set(self.table)) == len(self.table) and all(i in self.table for i in range(len(self.table)))

    # ---------------- Truth Table Generation ---------------- #
    def star_ddt_to_truthtable(self): # Convert star-DDT into a truthtable, which encode the differential propagations without probalities
        ddt = self.computeDDT()
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            dx = n >> self.output_bitsize
            dy = n & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] > 0: ttable += '1'
            else: ttable += '0'
        return ttable

    def pddt_to_truthtable(self, p): # Convert p-DDT into a truthtable, which encode the differential propagations with the item in ddt equal to p.
        ddt = self.computeDDT()
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            dx = n >> self.output_bitsize
            dy = n & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] == p: ttable += '1'
            else: ttable += '0'
        return ttable

    def ddt_to_truthtable_milp(self): # Convert the DDT into a truthtable, which encode the differential propagations with probalities.
        ddt = self.computeDDT()
        ttable = ''
        diff_weights = self.gen_weights(ddt)
        len_diff_weights = len(diff_weights)
        for n in range(2**(self.input_bitsize+self.output_bitsize+len_diff_weights)):
            dx = n >> (self.output_bitsize + len_diff_weights)
            dy = (n >> len_diff_weights) & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] > 0:
                p = bin(n & ((1 << (len_diff_weights)) - 1))[2:].zfill(len_diff_weights)
                w = 0
                for i in range(len_diff_weights):
                    w += diff_weights[i] * int(p[i])
                if abs(float(math.log(ddt[dx][dy]/(2**self.input_bitsize), 2))) == w: ttable += '1'
                else: ttable += '0'
            else: ttable += '0'
        return ttable

    def ddt_to_truthtable_sat(self): # Convert the DDT, which encode the differential propagations with probalities into a truthtable in sat.
        ddt = self.computeDDT()
        ttable = ''
        integers_weight, floats_weight = self.gen_integer_float_weight(ddt)
        len_diff_weights = int(max(integers_weight)+len(floats_weight))
        for n in range(2**(self.input_bitsize+self.output_bitsize+len_diff_weights)):
            dx = n >> (self.output_bitsize + len_diff_weights)
            dy = (n >> len_diff_weights) & ((1 << self.output_bitsize) - 1)
            if ddt[dx][dy] > 0:
                p = tuple(int(x) for x in bin(n & ((1 << len_diff_weights) - 1))[2:].zfill(len_diff_weights))
                w = abs(float(math.log(ddt[dx][dy]/(2**self.input_bitsize), 2)))
                pattern = self.gen_weight_pattern_sat(integers_weight, floats_weight, w)
                if p == tuple(pattern):  ttable += '1'
                else: ttable += '0'
            else: ttable += '0'
        return ttable

    def star_lat_to_truthtable(self): # Convert star-LAT into a truthtable, which encode the linear mask propagations without correlations.
        lat = self.computeLAT()
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            lx = n >> self.output_bitsize
            ly = n & ((1 << self.output_bitsize) - 1)
            if lat[lx][ly] != 0: ttable += '1'
            else: ttable += '0'
        return ttable

    def plat_to_truthtable(self, p): # Convert p-LAT into a truthtable, which encode the linear mask propagations with the item in lat equal to p.
        lat = self.computeLAT()
        ttable = ''
        for n in range(2**(self.input_bitsize+self.output_bitsize)):
            lx = n >> self.output_bitsize
            ly = n & ((1 << self.output_bitsize) - 1)
            if lat[lx][ly] == p or lat[lx][ly] == -p: ttable += '1'
            else: ttable += '0'
        return ttable

    def lat_to_truthtable_milp(self): # Convert the LAT into a truthtable, which encode the linear mask propagations with correlations.
        lat = self.computeLAT()
        ttable = ''
        linear_weights = self.gen_weights(lat)
        len_linear_weights = len(linear_weights)
        for n in range(2**(self.input_bitsize+self.output_bitsize+len_linear_weights)):
            lx = n >> (self.output_bitsize + len_linear_weights)
            ly = (n >> len_linear_weights) & ((1 << self.output_bitsize) - 1)
            if lat[lx][ly] != 0:
                p = bin(n & ((1 << (len_linear_weights)) - 1))[2:].zfill(len_linear_weights)
                w = 0
                for i in range(len_linear_weights):
                    w += linear_weights[i] * int(p[i])
                if abs(float(math.log(abs(lat[lx][ly])/(2**self.input_bitsize), 2))) == w: ttable += '1'
                else: ttable += '0'
            else: ttable += '0'
        return ttable

    def lat_to_truthtable_sat(self): # Convert the LAT, which encode the linear mask propagations with correlations into a truthtable in sat.
        lat = self.computeLAT()
        ttable = ''
        integers_weight, floats_weight = self.gen_integer_float_weight(lat)
        len_linear_weights = int(max(integers_weight)+len(floats_weight))
        for n in range(2**(self.input_bitsize+self.output_bitsize+len_linear_weights)):
            lx = n >> (self.output_bitsize + len_linear_weights)
            ly = (n >> len_linear_weights) & ((1 << self.output_bitsize) - 1)
            if lat[lx][ly] != 0:
                p = tuple(int(x) for x in bin(n & ((1 << len_linear_weights) - 1))[2:].zfill(len_linear_weights))
                w = abs(float(math.log(abs(lat[lx][ly])/(2**self.input_bitsize), 2)))
                pattern = self.gen_weight_pattern_sat(integers_weight, floats_weight, w)
                if p == tuple(pattern):  ttable += '1'
                else: ttable += '0'
            else: ttable += '0'
        return ttable

    def gen_spectrum(self, table):
        spectrum = sorted(list(set([abs(table[i][j]) for i in range(2**self.input_bitsize) for j in range(2**self.output_bitsize)]) - {0, 2**self.input_bitsize}))
        return spectrum

    def gen_weights(self, table):
        spectrum = self.gen_spectrum(table)
        weights = [abs(float(math.log(i/(2**self.input_bitsize), 2))) for i in spectrum]
        return weights

    def gen_integer_float_weight(self, table):
        weights = self.gen_weights(table)
        integers = sorted(set([int(x) for x in weights]))
        floats = sorted(set([x-int(x) for x in weights if x != int(x)]))
        return integers, floats

    def gen_weight_pattern_sat(self, integers_weight, floats_weight, w):
        int_w = int(w)
        float_w = w - int_w
        return [0] * (max(integers_weight) - int_w) + [1] * int_w + [1 if f == float_w else 0 for f in floats_weight]

    # ---------------- Implementation Code Generation ---------------- #
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python':
            if len(self.input_vars) == 1 and len(self.output_vars) == 1:
                return [self.get_var_ID('out', 0, unroll) + ' = ' + str(self.__class__.__name__) + '[' + self.get_var_ID('in', 0, unroll) + ']']
            elif len(self.input_vars) > 1 and len(self.output_vars) > 1:
                x_bits = len(self.input_vars)
                x_expr = 'x = ' + ' | '.join(f'({self.get_var_ID("in", i, unroll=unroll)} << {x_bits - 1 - i})'for i in range(x_bits))
                model_list = [x_expr]
                model_list.append(f'y = {self.__class__.__name__}[x]')
                y_vars = ', '.join(f'{self.get_var_ID("out", i, unroll=unroll)}' for i in range(x_bits))
                y_bits = ', '.join(f'(y >> {x_bits - 1 - i}) & 1' for i in range(x_bits))
                model_list.append(f'{y_vars} = {y_bits}')
                return model_list
            else: raise Exception(str(self.__class__.__name__) + ": unsupported number of input/output variables for 'python' implementation")
        elif implementation_type == 'c':
            if len(self.input_vars) == 1 and len(self.output_vars) == 1:
                return [self.get_var_ID('out', 0, unroll) + ' = ' + str(self.__class__.__name__) + '[' + self.get_var_ID('in', 0, unroll) + '];']
            elif len(self.input_vars) > 1 and len(self.output_vars) > 1:
                x_bits = len(self.input_vars)
                x_expr = 'x = ' + ' | '.join(f'({self.get_var_ID("in", i, unroll=unroll)} << {x_bits - 1 - i})'for i in range(x_bits))+ ";"
                model_list = [x_expr]
                model_list.append(f'y = {str(self.__class__.__name__)}[x];')
                for i in range(x_bits):
                    y_vars = self.get_var_ID("out", i, unroll=unroll)
                    y_bits = f'(y >> {x_bits - 1 - i}) & 1'
                    model_list.append(f'{y_vars} = {y_bits};')
                return model_list
            else: raise Exception(str(self.__class__.__name__) + ": unsupported number of input/output variables for 'c' implementation")
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def get_header_ID(self):
        return [self.__class__.__name__, self.model_version, self.input_bitsize, self.output_bitsize, self.table]

    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python':
            return [str(self.__class__.__name__) + ' = ' + str(self.table)]
        elif implementation_type == 'c':
            if self.input_bitsize <= 8:
                if len(self.input_vars) > 1 and len(self.output_vars) > 1: return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint8_t ' + 'x;'] + ['uint8_t ' + 'y;']
                else: return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
            else:
                if len(self.input_vars) > 1 and len(self.output_vars) > 1: return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint32_t ' + 'x;'] + ['uint32_t ' + 'y;']
                else: return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
        else: return None


    # ---------------- Modeling Interface ---------------- #
    def generate_model(self, model_type='sat', tool_type="minimize_logic", mode = 0, filename_load=True):
        self.model_filename = str(BASE_PATH / f"constraints_{model_type}_{self.model_version}_{tool_type}_{mode}.txt")
        self.filename_load = filename_load
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR", self.__class__.__name__ + "_LINEAR_PR"]:
            return self._generate_model_diff_linear_pr(model_type, tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
            return self._generate_model_diff_linear(model_type, tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF_P", self.__class__.__name__ + "_LINEAR_P"]:
            return self._generate_model_diff_linear_p(model_type, tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"] and (not isinstance(self.input_vars[0], list)):
            return self._generate_model_diff_linear_word_truncated(model_type)
        else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

    def _generate_model_diff_linear_pr(self, model_type, tool_type, mode):
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
        for i in range(len(self.output_vars)):
            var_out += self.get_var_model("out", i)

        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
            table = self.computeDDT()
        elif self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
            table = self.computeLAT()
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        if model_type == "sat":
            integers_weight, floats_weight = self.gen_integer_float_weight(table)
            var_p = [f"{self.ID}_p{i}" for i in range(max(integers_weight)+len(floats_weight))]
            pr_variables = [f"p{i}" for i in range(len(var_p))]
            objective_fun = " + ".join(pr_variables[:max(integers_weight)])
            if floats_weight:
                objective_fun += " + " + " + ".join(f"{w:.4f} {v}" for w, v in zip(floats_weight, pr_variables[max(integers_weight):]))
        elif model_type == "milp":
            weights = self.gen_weights(table)
            var_p = [f"{self.ID}_p{i}" for i in range(len(weights))]
            pr_variables = [f"p{i}" for i in range(len(var_p))]
            objective_fun = " + ".join(f"{w:.4f} {v}" for w, v in zip(weights, pr_variables))
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

        if self.filename_load and os.path.exists(self.model_filename):
            model_list, obj_fun = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out, var_p)
        else:
            if model_type == "sat" and self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
                ttable = self.ddt_to_truthtable_sat()
            elif model_type == "sat" and self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
                ttable = self.lat_to_truthtable_sat()
            elif model_type == "milp" and self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
                ttable = self.ddt_to_truthtable_milp()
            elif model_type == "milp" and self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
                ttable = self.lat_to_truthtable_milp()
            else:
                RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

            input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
            generate_and_save_constraints(model_type, tool_type, mode, ttable, input_variables, output_variables, pr_variables, objective_fun=objective_fun, model_filename=self.model_filename)
            model_list, obj_fun = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out, var_p)
        self.weight = [obj_fun]
        return model_list

    def _generate_model_diff_linear(self, model_type, tool_type, mode): # modeling all possible (input difference, output difference)
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]:
            self.model_filename = str(BASE_PATH / f"constraints_{model_type}_{self.model_version.replace('_A', '')}_{tool_type}_{mode}.txt")

        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
        for i in range(len(self.output_vars)):
            var_out += self.get_var_model("out", i)

        if self.filename_load and os.path.exists(self.model_filename):
            model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
        else:
            if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A"]:
                ttable = self.star_ddt_to_truthtable()
            elif self.model_version in [self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
                ttable = self.star_lat_to_truthtable()
            else:
                 RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
            input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
            generate_and_save_constraints(model_type, tool_type, mode, ttable, input_variables, output_variables, model_filename=self.model_filename)
            model_list, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)

        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]: # to calculate the minimum number of active S-boxes
            var_At = [self.ID + '_At']
            if model_type == "sat":
                model_list += [f"-{var} {var_At[0]}" for var in var_in] + [" ".join(var_in) + ' -' + var_At[0]]
            elif model_type == "milp":
                model_list += [f"{var_At[0]} - {var_in[i]} >= 0" for i in range(len(var_in))] + [" + ".join(var_in) + ' - ' + var_At[0] + ' >= 0']
                model_list.append('Binary\n' +  ' '.join(v for v in var_At))
            self.weight = var_At

        return model_list

    def _generate_model_diff_linear_p(self, model_type, tool_type, mode): # for large sbox, self.input_bitsize >= 8, e.g., skinny, use teh method from: MILP Modeling for (Large) S-boxes to Optimize Probability of Differential Characteristics. (2017). IACR Transactions on Symmetric Cryptology, 2017(4), 99-129.
        model_list = []

        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
        for i in range(len(self.output_vars)):
            var_out += self.get_var_model("out", i)

        if self.model_version in [self.__class__.__name__ + "_XORDIFF_P"]:
            table = self.computeDDT()
        elif self.model_version in [self.__class__.__name__ + "_LINEAR_P"]:
            table = self.computeLAT()
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        spectrum = self.gen_spectrum(table) + [2**self.input_bitsize]
        var_p = [f"{self.ID}_p{w}" for w in spectrum]
        model_v = self.model_version
        weight = ''

        for i in range(len(spectrum)):
            self.model_version = model_v + str(spectrum[i])
            self.model_filename = str(BASE_PATH / f"constraints_{model_type}_{self.model_version}_{tool_type}_{mode}.txt")

            if self.filename_load and os.path.exists(self.model_filename):
                sbox_inequalities, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)
            else:
                if "XORDIFF" in self.model_version:
                    ttable = self.pddt_to_truthtable(spectrum[i])
                elif "LINEAR" in self.model_version:
                    ttable = self.plat_to_truthtable(spectrum[i])
                else:
                    RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
                input_variables, output_variables = [f"a{i}" for i in range(len(var_in))], [f"b{i}" for i in range(len(var_out))]
                generate_and_save_constraints(model_type, tool_type, mode, ttable, input_variables, output_variables, model_filename=self.model_filename)
                sbox_inequalities, _ = gen_constraints_obj_func_from_template(self.model_filename, var_in, var_out)

            for ineq in sbox_inequalities:
                temp = ineq
                if ">=" in temp:
                    temp_0, temp_1 = temp.split(">=")[0], int(temp.split(" >= ")[1])
                    temp = temp_0 + f"- 10000 {var_p[i]} >= {temp_1-10000}"
                model_list += [temp]
            weight += " + " + "{:0.04f} ".format(abs(float(math.log(spectrum[i]/(2**self.input_bitsize), 2)))) + var_p[i]
        weight = weight[3:]
        model_list += [' + '.join(var_p) + ' = 1\n']
        model_list.append('Binary\n' +  ' '.join(v for v in var_p))
        self.weight = [weight]
        return model_list

    def _generate_model_diff_linear_word_truncated(self, model_type): # word-wise difference/linear propagations, the input difference equals the ouput difference
        var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))

        if model_type == "sat":
            model_list = [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]
        elif model_type == "milp":
            model_list = [f'{var_in[0]} - {var_out[0]} = 0']
            model_list.append('Binary\n' +  ' '.join(v for v in var_in + var_out))
        else:
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)

        if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"]: # to calculate the minimum number of active S-boxes
            self.weight = var_in

        return model_list


# ---------------- Cipher Sbox ---------------- #
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


class PRINCE_Sbox(Sbox):          # Operator of the PRINCE 4-bit Sbox
    def __init__(self, input_vars, output_vars, ID = None):
        super().__init__(input_vars, output_vars, 4, 4, ID = ID)
        self.table = [0xb, 0xf, 0x3, 0x2, 0xa, 0xc, 0x9, 0x1, 0x6, 0x7, 0x8, 0x0, 0xe, 0x5, 0xd, 0x4]
