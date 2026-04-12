from implementations.t_table.helper import generate_ttable

class TTable:
    def __init__(self, mc, sbox,table_name, word_size, poly):
        self.mc = mc.copy()
        self.sbox = sbox.copy()
        self.table_name = table_name
        self.word_size = word_size
        self.poly = poly

    def generate_implementation_header(self, implementation_type='python'):
        from math import sqrt
        n = int(sqrt(len(self.mc)))
        self.table = generate_ttable(self.mc, self.sbox,n, self.word_size,self.poly)
        if implementation_type == 'python': 
            return str(self.table_name) + ' = ' + str(self.table)
        elif implementation_type == 'c': 
            return "int "+ str(self.table_name)+f"[{len(self.table)}][{len(self.table[0])}]" + " = " + str([[c for c in r] for r in self.table]).replace('[', '{').replace(']', '}')+";"
        else: return None 
    
    def generate_implementation(self, input_vars, output_vars,name_list, implementation_type='python', unroll=True):
        if implementation_type == 'python': 
            return '[' + ','.join([output_vars[i].ID for i in range(len(output_vars))]) + "] = " + "int("+'^'.join([ name_list[i]+f"[{i}]"+"["+input_vars[i].ID+"]"  for i in range(len(input_vars))])+')'+ ".to_bytes(4, 'big')" 
        elif implementation_type == 'c': 
            return "x = "+ '^'.join([ name_list[i]+f"[{i}]"+"["+input_vars[i].ID+"]"  for i in range(len(input_vars))]) + "; " + ';'.join([output_vars[i].ID + f" = x >> {32 - (i+1)*8}"  for i in range(len(output_vars))])+";"
            
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")

    def generate_implementation_xor(self, input_vars, output_vars,name_list, implementation_type='python', unroll=True):
        #will ^= the out put varaibels 
        #if it is cont hte input_var is a list of integers 
        #input either is string or integer
        if implementation_type == 'python': 
            a = "int("+'^'.join([ name_list[i]+f"[{i}]"+"["+"^".join(input_vars[i])+"]"  for i in range(len(input_vars))])+')'+ ".to_bytes(4, 'big')"
            b="[" + ",".join([output_vars[i].ID for i in range(len(output_vars))]) +"]"
            rhs = f" = [a^b for a,b in zip({a},{b})]"
            lhs = b
            return lhs + rhs 
        elif implementation_type == 'c': 
            a = "int("+'^'.join([ name_list[i]+f"[{i}]"+"["+"^".join(input_vars[i])+"]"  for i in range(len(input_vars))])+')'+ ".to_bytes(4, 'big')"
            b = "[" + ",".join([output_vars[i].ID for i in range(len(output_vars))]) +"]"
            rhs = '^'.join([ name_list[i]+f"[{i}]"+"["+"^".join(input_vars[i])+"]"  for i in range(len(input_vars))])
            rtn = "x = " + rhs+"; " +  ';'.join([output_vars[i].ID + f" ^= x >> {32 - (i+1)*8}"  for i in range(len(output_vars))])
            return rtn  + ";"
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")