import math
import os
import sys
import time
import copy
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from operators.operators import Operator, RaiseExceptionVersionNotExisting
from tools.minimize_logic import ttb_to_ineq_logic
from tools.polyhedron import ttb_to_ineq_convex_hull
from tools.inequality import inequality_to_constraint_sat, inequality_to_constraint_milp
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'files/sbox_modeling/'))
if not os.path.exists(base_path):
    os.makedirs(base_path, exist_ok=True)


class Sbox(Operator):  # Generic operator assigning a Sbox relationship between the input variable and output variable (must be of same bitsize)
    def __init__(self, input_vars, output_vars, input_bitsize, output_bitsize, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.input_bitsize = input_bitsize
        self.output_bitsize = output_bitsize
        self.table = None
        self.table_inv = None

    def computeDDT(self): # Compute the differential Distribution Table (DDT) of the Sbox
        ddt = [[0]*(2**self.output_bitsize) for _ in range(2**self.input_bitsize)]
        for in_diff in range(2**self.input_bitsize):
            for j in range(2**self.input_bitsize):
                out_diff = self.table[j] ^ self.table[j^in_diff]
                ddt[in_diff][out_diff] += 1
        return ddt

    def computeLAT(self): # Compute the Linear Approximation Table (LAT) of the S-box.
        lat = [[0] * 2**self.output_bitsize for _ in range(2**self.input_bitsize)]
        for a in range(2**self.input_bitsize):
            for b in range(2**self.output_bitsize):
                acc = 0
                for x in range(2**self.input_bitsize):
                    ax = bin(a & x).count("1") & 1
                    bs = bin(b & self.table[x]).count("1") & 1
                    acc += 1 if (ax ^ bs) == 0 else -1
                lat[a][b] = acc
        return lat

    def linearDistributionTable(self):
        # storing the correlation (correlation = bias * 2)
        input_size = self.input_bitsize
        output_size = self.output_bitsize
        ldt = [[0 for i in range(2 ** output_size)] for j in range(2 ** input_size)]
        for output_mask in range(2 ** output_size):
            for input_mask in range(2 ** input_size):
                sum = 0
                for input in range(2 ** input_size):
                    output_mul = 0
                    for i in range(output_size):
                        output_mul = output_mul + int(bin(output_mask).replace("0b","").zfill(4)[i]) * int(bin(self.table[input]).replace("0b","").zfill(4)[i])
                    input_mul = 0
                    for i in range(input_size):
                        input_mul = input_mul + int(bin(input_mask).replace("0b","").zfill(4)[i]) * int(bin(input).replace("0b","").zfill(4)[i])
                    sum = sum + math.pow(-1, output_mul%2) * math.pow(-1, input_mul%2)
                ldt[input_mask][output_mask] = int(sum)
        return ldt


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
            if lat[lx][ly] == p: ttable += '1'
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
        self.model_filename = os.path.join(base_path, f'constraints_{model_type}_{self.model_version}_{tool_type}_{mode}.txt')
        self.filename_load = filename_load
        if model_type == 'sat':
            return self.generate_model_sat(tool_type, mode)
        elif model_type == 'milp':
            return self.generate_model_milp(tool_type, mode)
        elif model_type == 'cp':
            RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")

    # ---------------- Common utilities in SAT and MILP modeling ---------------- #
    def _reload_constraints_objfun_from_file(self):
        if os.path.exists(self.model_filename):
            with open(self.model_filename, 'r') as file:
                for line in file:
                    if 'Constraints:' in line:
                        constraints = eval(line.split(':', 1)[1].strip())
                    if 'Weight:' in line:
                        objective_fun = line[len("Weight: "):]
            return constraints, objective_fun
        else:
            return None, None

    def _trans_template_ineq(self, template_inequalities, template_weight, var_in, var_out, var_p=None):
        a, b, p = "a", "b", "p" # Variable prefixes for input (a), output (b), and probability (p) in modeling
        inequalities = []
        for ineq in template_inequalities:
            temp = ineq
            for i in range(self.input_bitsize):
                temp = temp.replace(f"{a}{i}", var_in[i])
            for i in range(self.output_bitsize):
                temp = temp.replace(f"{b}{i}", var_out[i])
            if var_p:
                for i in range(template_weight.count('+')+1):
                    temp = temp.replace(f"{p}{i}", var_p[i])
            inequalities += [temp]
        return inequalities

    def _trans_template_weight(self, template_weight, var_p):
        p = "p" # Variable prefixes for probability (p) in modeling
        weight = copy.deepcopy(template_weight)
        for i in range(weight.count('+') + 1):
            weight = weight.replace(f"{p}{i}", f"{var_p[i]}")
        weight = weight.replace("\n", "")
        return weight

    def _gen_model_input_output_variables(self):
        input_variables = [f'a{i}' for i in range(self.input_bitsize)]
        output_variables = [f'b{i}' for i in range(self.output_bitsize)]
        return input_variables, output_variables

    def _write_model_constraints(self, input_variables, output_variables, constraints, objective_fun, time):
        variables_mapping = "Input: {0}; msb: {1}".format("||".join(input_variables), input_variables[0])
        variables_mapping += "\nOutput: {0}; msb: {1}".format("||".join(output_variables), output_variables[0])
        with open(self.model_filename, 'w') as file:
            file.write(f"{variables_mapping}\n")
            file.write(f"Time used to simplify the constraints: {time:.4f} s\n")
            file.write(f"Number of constraints: {len(constraints)}\n")
            file.write(f"Constraints: {constraints}\n")
            file.write(f"Weight: {objective_fun}\n")

    # ---------------- SAT Model Generation ---------------- #
    def generate_model_sat(self, tool_type="minimize_logic", mode = 0):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR", self.__class__.__name__ + "_LINEAR_PR"]:
            return self._gen_model_sat_diff_linear_pr(tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
            return self._gen_model_sat_diff_linear(tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"] and (not isinstance(self.input_vars[0], list)):
            return self._gen_model_sat_diff_linear_word_truncated()
        else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "sat")

    def _gen_model_sat_diff_linear_pr(self, tool_type, mode): # model all possible (input difference, output difference, probablity) to search for the best differential/linear characteristic
        sbox_inequalities, sbox_weight = self._gen_model_constraints_sat(tool_type, mode)
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        var_p = [f"{self.ID}_p{i}" for i in range(sbox_weight.count('+') + 1)]
        self.weight = [self._trans_template_weight(sbox_weight, var_p)]
        return self._trans_template_ineq(sbox_inequalities, sbox_weight, var_in, var_out, var_p)

    def _gen_model_sat_diff_linear(self, tool_type, mode): # modeling all possible (input difference, output difference)
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]:
            self.model_filename = os.path.join(base_path, f'constraints_sat_{self.model_version.replace("_A", "")}_{tool_type}_{mode}.txt')
        sbox_inequalities, sbox_weight = self._gen_model_constraints_sat(tool_type, mode)
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        model_list = self._trans_template_ineq(sbox_inequalities, sbox_weight, var_in, var_out)
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]: # to calculate the minimum number of active S-boxes
            var_At = [self.ID + '_At']
            model_list += self._model_count_active_sbox_sat(var_in, var_At[0])
            self.weight = var_At
        return model_list

    def _gen_model_sat_diff_linear_word_truncated(self): # word-wise difference/linear propagations, the input difference equals the ouput difference
        var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
        if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"]:
            self.weight = var_in
        return [f"-{var_in[0]} {var_out[0]}", f"{var_in[0]} -{var_out[0]}"]

    def _gen_model_constraints_sat(self, tool_type, mode):
        if self.filename_load and os.path.exists(self.model_filename):
            return self._reload_constraints_objfun_from_file()
        ttable = self._gen_model_ttable_sat()
        input_variables, output_variables = self._gen_model_input_output_variables()
        pr_variables, objective_fun = self._gen_model_pr_variables_objective_fun_sat()
        variables = input_variables + output_variables + pr_variables
        time_start = time.time()
        if tool_type == "minimize_logic":
            inequalities = ttb_to_ineq_logic(ttable, variables, mode=mode)
        else: raise Exception(str(self.__class__.__name__) + ": unknown tool type '" + tool_type + "'")
        constraints = [inequality_to_constraint_sat(ineq, variables) for ineq in inequalities]
        time_end = time.time()
        self._write_model_constraints(input_variables, output_variables, constraints, objective_fun, time_end-time_start)
        return constraints, objective_fun

    def _gen_model_ttable_sat(self):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
            return self.ddt_to_truthtable_sat()
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A"]:
            return self.star_ddt_to_truthtable()
        elif self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
            return self.lat_to_truthtable_sat()
        elif self.model_version in [self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
            return self.star_lat_to_truthtable()
        else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "sat")

    def _gen_model_pr_variables_objective_fun_sat(self):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR", self.__class__.__name__ + "_LINEAR_PR"]:
            if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
                table = self.computeDDT()
            elif self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
                table = self.computeLAT()
            integers_weight, floats_weight = self.gen_integer_float_weight(table)
            pr_variables = [f'p{i}' for i in range(max(integers_weight)+len(floats_weight))]
            objective_fun = " + ".join(pr_variables[:max(integers_weight)])
            if floats_weight:
                objective_fun += " + " + " + ".join(f"{w:.4f} {v}" for w, v in zip(floats_weight, pr_variables[max(integers_weight):]))
            return pr_variables, objective_fun
        return [], ""

    def _model_count_active_sbox_sat(self, var_in, var_At):
        return [f"-{var} {var_At}" for var in var_in] + [" ".join(var_in) + ' -' + var_At]

    # ---------------- MILP Model Generation ---------------- #
    def generate_model_milp(self, tool_type="polyhedron", mode = 0):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR", self.__class__.__name__ + "_LINEAR_PR"]:
            return self._generate_model_milp_diff_linear_pr(tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
            return self._generate_model_milp_diff_linear(tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF_P", self.__class__.__name__ + "_LINEAR_P"]:
            return self._generate_model_milp_diff_linear_p(tool_type, mode)
        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF", self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"] and (not isinstance(self.input_vars[0], list)): # word-wise difference propagations, the input difference equals the ouput difference
            return self._generate_model_milp_diff_linear_word_truncated()
        elif self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_1", self.__class__.__name__ + "_TRUNCATEDDIFF_A_1", self.__class__.__name__ + "_TRUNCATEDLINEAR_1", self.__class__.__name__ + "_TRUNCATEDLINEAR_A_1"]: #  bit-wise truncated difference propagations
            return self._generate_model_milp_diff_linear_bit_truncated()
        else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "milp")

    def _generate_model_milp_diff_linear_pr(self, tool_type, mode): # modeling all possible (input difference, output difference, probablity)
        sbox_inequalities, sbox_weight = self._gen_model_constraints_milp(tool_type, mode)
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        var_p = [f"{self.ID}_p{i}" for i in range(sbox_weight.count('+') + 1)]
        model_list = self._trans_template_ineq(sbox_inequalities, sbox_weight, var_in, var_out, var_p)
        model_list += self._declare_vars_type_milp('Binary', var_in + var_out + var_p)
        self.weight = [self._trans_template_weight(sbox_weight, var_p)]
        return model_list

    def _generate_model_milp_diff_linear(self, tool_type, mode):  # modeling all possible (input difference, output difference)
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]:
            self.model_filename = os.path.join(base_path, f'constraints_milp_{self.model_version.replace("_A", "")}_{tool_type}_{mode}.txt')
        sbox_inequalities, sbox_weight = self._gen_model_constraints_milp(tool_type, mode)
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        model_list = self._trans_template_ineq(sbox_inequalities, sbox_weight, var_in, var_out)
        all_vars = var_in + var_out
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_A", self.__class__.__name__ + "_LINEAR_A"]: # to calculate the minimum number of active S-boxes
            var_At = [self.ID + '_At']
            model_list += self._model_count_active_sbox_milp(var_in, var_At[0])
            all_vars += var_At
            self.weight = var_At
        model_list += self._declare_vars_type_milp('Binary', all_vars)
        return model_list

    def _generate_model_milp_diff_linear_p(self, tool_type, mode): # for large sbox, self.input_bitsize >= 8, e.g., skinny, cite from: MILP Modeling for (Large) S-boxes to Optimize Probability of Differential Characteristics. (2017). IACR Transactions on Symmetric Cryptology, 2017(4), 99-129.
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        ddt = self.computeDDT()
        diff_spectrum = self.gen_spectrum(ddt) + [2**self.input_bitsize]
        var_p = [f"{self.ID}_p{w}" for w in range(len(diff_spectrum))]
        weight = ''
        model_list = []
        model_v = self.model_version
        mode = mode if isinstance(mode, list) else [mode] * len(diff_spectrum)
        for w in range(len(diff_spectrum)):
            self.model_version = model_v + str(diff_spectrum[w])
            self.model_filename = os.path.join(base_path, f'constraints_milp_{self.model_version}_{tool_type}_{mode[w]}.txt')
            sbox_inequalities, sbox_weight = self._gen_model_constraints_milp(tool_type, mode[w])
            for ineq in sbox_inequalities:
                temp = ineq
                for i in range(self.input_bitsize): temp = temp.replace(f"a{i}", var_in[i])
                for i in range(self.output_bitsize): temp = temp.replace(f"b{i}", var_out[i])
                temp_0, temp_1 = temp.split(">=")[0], int(temp.split(" >= ")[1])
                temp = temp_0 + f"- 10000 {var_p[w]} >= {temp_1-10000}"
                model_list += [temp]
            weight += " + " + "{:0.04f} ".format(abs(float(math.log(diff_spectrum[w]/(2**self.input_bitsize), 2)))) + var_p[w]
        model_list += [' + '.join(var_p) + ' = 1\n']
        model_list += self._declare_vars_type_milp('Binary', var_in + var_out + var_p)
        self.weight = [weight]
        return model_list

    def _generate_model_milp_diff_linear_word_truncated(self): # word-wise truncated difference propagations, the input difference equals the ouput difference
        var_in, var_out = (self.get_var_model("in", 0, bitwise=False), self.get_var_model("out", 0, bitwise=False))
        model_list = [f'{var_in[0]} - {var_out[0]} = 0']
        model_list += self._declare_vars_type_milp('Binary', var_in + var_out)
        if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_A", self.__class__.__name__ + "_TRUNCATEDLINEAR_A"]: # to calculate the minimum number of active S-boxes
            self.weight = var_in
        return model_list

    def _generate_model_milp_diff_linear_bit_truncated(self): #  bit-wise truncated difference propagations
        if "DIFF" in self.model_version:
            branch_num = self.differential_branch_number()
        elif "LINEAR" in self.model_version:
            branch_num = self.linear_branch_number()
        var_in, var_out = [], []
        for i in range(len(self.input_vars)):
            var_in += self.get_var_model("in", i)
            var_out += self.get_var_model("out", i)
        all_vars = var_in + var_out
        model_list = []
        if branch_num >= 3: # model the differential/linear branch number of sbox
            var_d = [self.ID + '_d']
            model_list += self._model_branch_num_milp(var_in, var_out, var_d[0], branch_num)
            all_vars += var_d
        if self.is_bijective(): # for bijective S-boxes, nonzero input difference must result in nonzero output difference and vice versa
            model_list += self._model_bijective_milp(var_in, var_out)
        if self.model_version in [self.__class__.__name__ + "_TRUNCATEDDIFF_A_1", self.__class__.__name__ + "_TRUNCATEDLINEAR_A_1"]: # to calculate the minimum number of differentially active s-boxes
            var_At = [self.ID + '_At']
            model_list += self._model_count_active_sbox_milp(var_in, var_At[0])
            self.weight = var_At
            all_vars += var_At
        model_list += self._declare_vars_type_milp('Binary', all_vars)
        return model_list

    def _gen_model_constraints_milp(self, tool_type="polyhedron", mode=0):
        if self.filename_load and os.path.exists(self.model_filename):
            return self._reload_constraints_objfun_from_file()
        ttable = self._gen_model_ttable_milp()
        input_variables, output_variables = self._gen_model_input_output_variables()
        pr_variables, objective_fun = self._gen_model_pr_variables_objective_fun_milp()
        variables = input_variables + output_variables + pr_variables
        time_start = time.time()
        if tool_type=="minimize_logic":
            inequalities = ttb_to_ineq_logic(ttable, variables, mode=mode)
        elif tool_type=="polyhedron":
            inequalities = ttb_to_ineq_convex_hull(ttable, variables)
        constraints = [inequality_to_constraint_milp(ineq, variables) for ineq in inequalities]
        time_end = time.time()
        self._write_model_constraints(input_variables, output_variables, constraints, objective_fun, time_end-time_start)
        return constraints, objective_fun

    def _declare_vars_type_milp(self, var_type, variables):
        return [f'{var_type}\n' +  ' '.join(variables)]

    def _model_count_active_sbox_milp(self, var_in, var_At):
        return [f"{var_At} - {var_in[i]} >= 0" for i in range(len(var_in))] + [" + ".join(var_in) + ' - ' + var_At + ' >= 0']

    def _model_branch_num_milp(self, var_in, var_out, var_d, branch_num):
        return [f"{var_d} - {var} >= 0" for var in var_in + var_out] + [" + ".join(var_in + var_out) + ' - ' + str(branch_num) + ' ' + var_d + ' >= 0']

    def _model_bijective_milp(self, var_in, var_out):
        model_list = [f"{len(var_in)} " + f" + {len(var_in)} " .join(var_out) +  " - " + " - ".join(var_in) + ' >= 0']
        model_list += [f"{len(var_out)} " + f" + {len(var_out)} ".join(var_in) +  " - " +  " - ".join(var_out) + ' >= 0']
        return model_list

    def _gen_model_ttable_milp(self):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF", self.__class__.__name__ + "_XORDIFF_A"]:
            return self.star_ddt_to_truthtable()
        elif self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
            return self.ddt_to_truthtable_milp()
        elif self.model_version[:len(self.__class__.__name__ + "_XORDIFF_P")] == self.__class__.__name__ + "_XORDIFF_P" and self.model_version[len(self.__class__.__name__ + "_XORDIFF_P"):].isdigit():
            return self.pddt_to_truthtable(int(self.model_version[len(self.__class__.__name__ + "_XORDIFF_P"):]))
        elif self.model_version in [self.__class__.__name__ + "_LINEAR", self.__class__.__name__ + "_LINEAR_A"]:
            return self.star_lat_to_truthtable()
        elif self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
            return self.lat_to_truthtable_milp()
        elif self.model_version[:len(self.__class__.__name__ + "_LINEAR_P")] == self.__class__.__name__ + "_LINEAR_P" and self.model_version[len(self.__class__.__name__ + "_LINEAR_P"):].isdigit():
            return self.plat_to_truthtable(int(self.model_version[len(self.__class__.__name__ + "_LINEAR_P"):]))
        else: RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, "milp")

    def _gen_model_pr_variables_objective_fun_milp(self):
        if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR", self.__class__.__name__ + "_LINEAR_PR"]:
            if self.model_version in [self.__class__.__name__ + "_XORDIFF_PR"]:
                table = self.computeDDT()
            elif self.model_version in [self.__class__.__name__ + "_LINEAR_PR"]:
                table = self.computeLAT()
            weights = self.gen_weights(table)
            pr_variables = [f'p{i}' for i in range(len(weights))]
            objective_fun = " + ".join(f"{w:.4f} {v}" for w, v in zip(weights, pr_variables))
            return pr_variables, objective_fun
        return [], ""


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


class AES_TTable(Sbox):
    def __init__(self, input_vars, output_vars, ID = None):
        """
        split input var into 2d arrays 
        input put into [[], [], [], []]
        output into [,,,,,,] again 
        but swigth the usula shift 
        """
        super().__init__(input_vars, output_vars, 8, 8, ID = ID)
        self.table = [[3328402341, 4168907908, 4000806809, 4135287693, 4294111757, 3597364157, 3731845041, 2445657428, 1613770832, 33620227, 3462883241, 1445669757, 3892248089, 3050821474, 1303096294, 3967186586, 2412431941, 528646813, 2311702848, 4202528135, 4026202645, 2992200171, 2387036105, 4226871307, 1101901292, 3017069671, 1604494077, 1169141738, 597466303, 1403299063, 3832705686, 2613100635, 1974974402, 3791519004, 1033081774, 1277568618, 1815492186, 2118074177, 4126668546, 2211236943, 1748251740, 1369810420, 3521504564, 4193382664, 3799085459, 2883115123, 1647391059, 706024767, 134480908, 2512897874, 1176707941, 2646852446, 806885416, 932615841, 168101135, 798661301, 235341577, 605164086, 461406363, 3756188221, 3454790438, 1311188841, 2142417613, 3933566367, 302582043, 495158174, 1479289972, 874125870, 907746093, 3698224818, 3025820398, 1537253627, 2756858614, 1983593293, 3084310113, 2108928974, 1378429307, 3722699582, 1580150641, 327451799, 2790478837, 3117535592, 0, 3253595436, 1075847264, 3825007647, 2041688520, 3059440621, 3563743934, 2378943302, 1740553945, 1916352843, 2487896798, 2555137236, 2958579944, 2244988746, 3151024235, 3320835882, 1336584933, 3992714006, 2252555205, 2588757463, 1714631509, 293963156, 2319795663, 3925473552, 67240454, 4269768577, 2689618160, 2017213508, 631218106, 1269344483, 2723238387, 1571005438, 2151694528, 93294474, 1066570413, 563977660, 1882732616, 4059428100, 1673313503, 2008463041, 2950355573, 1109467491, 537923632, 3858759450, 4260623118, 3218264685, 2177748300, 403442708, 638784309, 3287084079, 3193921505, 899127202, 2286175436, 773265209, 2479146071, 1437050866, 4236148354, 2050833735, 3362022572, 3126681063, 840505643, 3866325909, 3227541664, 427917720, 2655997905, 2749160575, 1143087718, 1412049534, 999329963, 193497219, 2353415882, 3354324521, 1807268051, 672404540, 2816401017, 3160301282, 369822493, 2916866934, 3688947771, 1681011286, 1949973070, 336202270, 2454276571, 201721354, 1210328172, 3093060836, 2680341085, 3184776046, 1135389935, 3294782118, 965841320, 831886756, 3554993207, 4068047243, 3588745010, 2345191491, 1849112409, 3664604599, 26054028, 2983581028, 2622377682, 1235855840, 3630984372, 2891339514, 4092916743, 3488279077, 3395642799, 4101667470, 1202630377, 268961816, 1874508501, 4034427016, 1243948399, 1546530418, 941366308, 1470539505, 1941222599, 2546386513, 3421038627, 2715671932, 3899946140, 1042226977, 2521517021, 1639824860, 227249030, 260737669, 3765465232, 2084453954, 1907733956, 3429263018, 2420656344, 100860677, 4160157185, 470683154, 3261161891, 1781871967, 2924959737, 1773779408, 394692241, 2579611992, 974986535, 664706745, 3655459128, 3958962195, 731420851, 571543859, 3530123707, 2849626480, 126783113, 865375399, 765172662, 1008606754, 361203602, 3387549984, 2278477385, 2857719295, 1344809080, 2782912378, 59542671, 1503764984, 160008576, 437062935, 1707065306, 3622233649, 2218934982, 3496503480, 2185314755, 697932208, 1512910199, 504303377, 2075177163, 2824099068, 1841019862, 739644986], [2781242211, 2230877308, 2582542199, 2381740923, 234877682, 3184946027, 2984144751, 1418839493, 1348481072, 50462977, 2848876391, 2102799147, 434634494, 1656084439, 3863849899, 2599188086, 1167051466, 2636087938, 1082771913, 2281340285, 368048890, 3954334041, 3381544775, 201060592, 3963727277, 1739838676, 4250903202, 3930435503, 3206782108, 4149453988, 2531553906, 1536934080, 3262494647, 484572669, 2923271059, 1783375398, 1517041206, 1098792767, 49674231, 1334037708, 1550332980, 4098991525, 886171109, 150598129, 2481090929, 1940642008, 1398944049, 1059722517, 201851908, 1385547719, 1699095331, 1587397571, 674240536, 2704774806, 252314885, 3039795866, 151914247, 908333586, 2602270848, 1038082786, 651029483, 1766729511, 3447698098, 2682942837, 454166793, 2652734339, 1951935532, 775166490, 758520603, 3000790638, 4004797018, 4217086112, 4137964114, 1299594043, 1639438038, 3464344499, 2068982057, 1054729187, 1901997871, 2534638724, 4121318227, 1757008337, 0, 750906861, 1614815264, 535035132, 3363418545, 3988151131, 3201591914, 1183697867, 3647454910, 1265776953, 3734260298, 3566750796, 3903871064, 1250283471, 1807470800, 717615087, 3847203498, 384695291, 3313910595, 3617213773, 1432761139, 2484176261, 3481945413, 283769337, 100925954, 2180939647, 4037038160, 1148730428, 3123027871, 3813386408, 4087501137, 4267549603, 3229630528, 2315620239, 2906624658, 3156319645, 1215313976, 82966005, 3747855548, 3245848246, 1974459098, 1665278241, 807407632, 451280895, 251524083, 1841287890, 1283575245, 337120268, 891687699, 801369324, 3787349855, 2721421207, 3431482436, 959321879, 1469301956, 4065699751, 2197585534, 1199193405, 2898814052, 3887750493, 724703513, 2514908019, 2696962144, 2551808385, 3516813135, 2141445340, 1715741218, 2119445034, 2872807568, 2198571144, 3398190662, 700968686, 3547052216, 1009259540, 2041044702, 3803995742, 487983883, 1991105499, 1004265696, 1449407026, 1316239930, 504629770, 3683797321, 168560134, 1816667172, 3837287516, 1570751170, 1857934291, 4014189740, 2797888098, 2822345105, 2754712981, 936633572, 2347923833, 852879335, 1133234376, 1500395319, 3084545389, 2348912013, 1689376213, 3533459022, 3762923945, 3034082412, 4205598294, 133428468, 634383082, 2949277029, 2398386810, 3913789102, 403703816, 3580869306, 2297460856, 1867130149, 1918643758, 607656988, 4049053350, 3346248884, 1368901318, 600565992, 2090982877, 2632479860, 557719327, 3717614411, 3697393085, 2249034635, 2232388234, 2430627952, 1115438654, 3295786421, 2865522278, 3633334344, 84280067, 33027830, 303828494, 2747425121, 1600795957, 4188952407, 3496589753, 2434238086, 1486471617, 658119965, 3106381470, 953803233, 334231800, 3005978776, 857870609, 3151128937, 1890179545, 2298973838, 2805175444, 3056442267, 574365214, 2450884487, 550103529, 1233637070, 4289353045, 2018519080, 2057691103, 2399374476, 4166623649, 2148108681, 387583245, 3664101311, 836232934, 3330556482, 3100665960, 3280093505, 2955516313, 2002398509, 287182607, 3413881008, 4238890068, 3597515707, 975967766], [1671808611, 2089089148, 2006576759, 2072901243, 4061003762, 1807603307, 1873927791, 3310653893, 810573872, 16974337, 1739181671, 729634347, 4263110654, 3613570519, 2883997099, 1989864566, 3393556426, 2191335298, 3376449993, 2106063485, 4195741690, 1508618841, 1204391495, 4027317232, 2917941677, 3563566036, 2734514082, 2951366063, 2629772188, 2767672228, 1922491506, 3227229120, 3082974647, 4246528509, 2477669779, 644500518, 911895606, 1061256767, 4144166391, 3427763148, 878471220, 2784252325, 3845444069, 4043897329, 1905517169, 3631459288, 827548209, 356461077, 67897348, 3344078279, 593839651, 3277757891, 405286936, 2527147926, 84871685, 2595565466, 118033927, 305538066, 2157648768, 3795705826, 3945188843, 661212711, 2999812018, 1973414517, 152769033, 2208177539, 745822252, 439235610, 455947803, 1857215598, 1525593178, 2700827552, 1391895634, 994932283, 3596728278, 3016654259, 695947817, 3812548067, 795958831, 2224493444, 1408607827, 3513301457, 0, 3979133421, 543178784, 4229948412, 2982705585, 1542305371, 1790891114, 3410398667, 3201918910, 961245753, 1256100938, 1289001036, 1491644504, 3477767631, 3496721360, 4012557807, 2867154858, 4212583931, 1137018435, 1305975373, 861234739, 2241073541, 1171229253, 4178635257, 33948674, 2139225727, 1357946960, 1011120188, 2679776671, 2833468328, 1374921297, 2751356323, 1086357568, 2408187279, 2460827538, 2646352285, 944271416, 4110742005, 3168756668, 3066132406, 3665145818, 560153121, 271589392, 4279952895, 4077846003, 3530407890, 3444343245, 202643468, 322250259, 3962553324, 1608629855, 2543990167, 1154254916, 389623319, 3294073796, 2817676711, 2122513534, 1028094525, 1689045092, 1575467613, 422261273, 1939203699, 1621147744, 2174228865, 1339137615, 3699352540, 577127458, 712922154, 2427141008, 2290289544, 1187679302, 3995715566, 3100863416, 339486740, 3732514782, 1591917662, 186455563, 3681988059, 3762019296, 844522546, 978220090, 169743370, 1239126601, 101321734, 611076132, 1558493276, 3260915650, 3547250131, 2901361580, 1655096418, 2443721105, 2510565781, 3828863972, 2039214713, 3878868455, 3359869896, 928607799, 1840765549, 2374762893, 3580146133, 1322425422, 2850048425, 1823791212, 1459268694, 4094161908, 3928346602, 1706019429, 2056189050, 2934523822, 135794696, 3134549946, 2022240376, 628050469, 779246638, 472135708, 2800834470, 3032970164, 3327236038, 3894660072, 3715932637, 1956440180, 522272287, 1272813131, 3185336765, 2340818315, 2323976074, 1888542832, 1044544574, 3049550261, 1722469478, 1222152264, 50660867, 4127324150, 236067854, 1638122081, 895445557, 1475980887, 3117443513, 2257655686, 3243809217, 489110045, 2662934430, 3778599393, 4162055160, 2561878936, 288563729, 1773916777, 3648039385, 2391345038, 2493985684, 2612407707, 505560094, 2274497927, 3911240169, 3460925390, 1442818645, 678973480, 3749357023, 2358182796, 2717407649, 2306869641, 219617805, 3218761151, 3862026214, 1120306242, 1756942440, 1103331905, 2578459033, 762796589, 252780047, 2966125488, 1425844308, 3151392187, 372911126], [1667474886, 2088535288, 2004326894, 2071694838, 4075949567, 1802223062, 1869591006, 3318043793, 808472672, 16843522, 1734846926, 724270422, 4278065639, 3621216949, 2880169549, 1987484396, 3402253711, 2189597983, 3385409673, 2105378810, 4210693615, 1499065266, 1195886990, 4042263547, 2913856577, 3570689971, 2728590687, 2947541573, 2627518243, 2762274643, 1920112356, 3233831835, 3082273397, 4261223649, 2475929149, 640051788, 909531756, 1061110142, 4160160501, 3435941763, 875846760, 2779116625, 3857003729, 4059105529, 1903268834, 3638064043, 825316194, 353713962, 67374088, 3351728789, 589522246, 3284360861, 404236336, 2526454071, 84217610, 2593830191, 117901582, 303183396, 2155911963, 3806477791, 3958056653, 656894286, 2998062463, 1970642922, 151591698, 2206440989, 741110872, 437923380, 454765878, 1852748508, 1515908788, 2694904667, 1381168804, 993742198, 3604373943, 3014905469, 690584402, 3823320797, 791638366, 2223281939, 1398011302, 3520161977, 0, 3991743681, 538992704, 4244381667, 2981218425, 1532751286, 1785380564, 3419096717, 3200178535, 960056178, 1246420628, 1280103576, 1482221744, 3486468741, 3503319995, 4025428677, 2863326543, 4227536621, 1128514950, 1296947098, 859002214, 2240123921, 1162203018, 4193849577, 33687044, 2139062782, 1347481760, 1010582648, 2678045221, 2829640523, 1364325282, 2745433693, 1077985408, 2408548869, 2459086143, 2644360225, 943212656, 4126475505, 3166494563, 3065430391, 3671750063, 555836226, 269496352, 4294908645, 4092792573, 3537006015, 3452783745, 202118168, 320025894, 3974901699, 1600119230, 2543297077, 1145359496, 387397934, 3301201811, 2812801621, 2122220284, 1027426170, 1684319432, 1566435258, 421079858, 1936954854, 1616945344, 2172753945, 1330631070, 3705438115, 572679748, 707427924, 2425400123, 2290647819, 1179044492, 4008585671, 3099120491, 336870440, 3739122087, 1583276732, 185277718, 3688593069, 3772791771, 842159716, 976899700, 168435220, 1229577106, 101059084, 606366792, 1549591736, 3267517855, 3553849021, 2897014595, 1650632388, 2442242105, 2509612081, 3840161747, 2038008818, 3890688725, 3368567691, 926374254, 1835907034, 2374863873, 3587531953, 1313788572, 2846482505, 1819063512, 1448540844, 4109633523, 3941213647, 1701162954, 2054852340, 2930698567, 134748176, 3132806511, 2021165296, 623210314, 774795868, 471606328, 2795958615, 3031746419, 3334885783, 3907527627, 3722280097, 1953799400, 522133822, 1263263126, 3183336545, 2341176845, 2324333839, 1886425312, 1044267644, 3048588401, 1718004428, 1212733584, 50529542, 4143317495, 235803164, 1633788866, 892690282, 1465383342, 3115962473, 2256965911, 3250673817, 488449850, 2661202215, 3789633753, 4177007595, 2560144171, 286339874, 1768537042, 3654906025, 2391705863, 2492770099, 2610673197, 505291324, 2273808917, 3924369609, 3469625735, 1431699370, 673740880, 3755965093, 2358021891, 2711746649, 2307489801, 218961690, 3217021541, 3873845719, 1111672452, 1751693520, 1094828930, 2576986153, 757954394, 252645662, 2964376443, 1414855848, 3149649517, 370555436]]
        self.table_inv = []

    def generate_implementation_header(self, implementation_type='python'):
        if implementation_type == 'python': 
            return [str(self.__class__.__name__) + ' = ' + str(self.table)]       
        elif implementation_type == 'c': 
            return None
            if self.input_bitsize <= 8: 
                if isinstance(self.input_vars[0], list): return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint8_t ' + 'x;'] + ['uint8_t ' + 'y;']
                else: return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
            else: 
                if isinstance(self.input_vars[0], list): return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint32_t ' + 'x;'] + ['uint32_t ' + 'y;']
                else: return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
        else: return None
    
    def generate_implementation_header_unique(self, implementation_type='python'):
        print("CALLED THERE model list")
        #the input var is diff
        if implementation_type == 'python': 
            model_list = ["#TTable generation function", \
                          "def GMUL(a, b, p, d):\n\tresult = 0\n\twhile b > 0:\n\t\tif b & 1:\n\t\t\tresult ^= a\n\t\ta <<= 1\n\t\tif a & (1 << d):\n\t\t\ta ^= p\n\t\tb >>= 1\n\treturn result & ((1 << d) - 1)\n\n"]
        else:
            raise Exception("TO BE DONE LATER")
        return model_list    

    def generate_implementation(self, implementation_type='python', unroll=False):
        #differ greatly from parent class here we enforce strictly 2d array in inputs
        #we will be taking in a 2d list of vinput
        if implementation_type == 'python': 
            name = str(self.__class__.__name__)
            #str(self.__class__.__name__) + '[' + self.get_var_ID('in', 0, unroll) + ']'
            return ['[' + ','.join([self.get_var_ID('out', i, unroll) for i in range(len(self.output_vars))]) + "] = " + "int("+'^'.join([ name+f"[{i}]"+"["+self.get_var_ID('in', i, unroll)+"]"  for i in range(len(self.input_vars))])+')'+ ".to_bytes(4, 'big')" ] 
        elif implementation_type == 'c': 
            raise Exception(str(self.__class__.__name__) + ": NOT SUPPORTED YET LATER DO '" + implementation_type + "'")
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def computeDDT(self): # Compute the differential Distribution Table (DDT) of the Sbox
        ddt = [[0]*(2**self.output_bitsize) for _ in range(2**self.input_bitsize)] 
        return ddt
        
    def computeLAT(self): # Compute the Linear Approximation Table (LAT) of the S-box.
        lat = [[0] * 2**self.output_bitsize for _ in range(2**self.input_bitsize)]
        return lat 
        
    def linearDistributionTable(self):
        # storing the correlation (correlation = bias * 2)
        input_size = self.input_bitsize
        output_size = self.output_bitsize
        ldt = [[0 for i in range(2 ** output_size)] for j in range(2 ** input_size)]
        return ldt
        
    def differential_branch_number(self): # Return differential branch number of the S-Box.
        return 0
        
    
    def is_bijective(self): # Check if the length of the set of s_box is equal to the length of s_box. The set will contain only unique elements
        return 0 
        #return len(set(self.table)) == len(self.table) and all(i in self.table for i in range(len(self.table)))

    def get_header_ID(self): 
        return [self.__class__.__name__, self.model_version, self.input_bitsize, self.output_bitsize, self.table]
        
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
