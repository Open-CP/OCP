from implementations.t_table.initializer import generate_ttable

class TTable:
    def __init__(self, mc, sbox,table_name):
        self.mc = mc.copy()
        self.sbox = sbox.copy()
        self.table_name = table_name

    def generate_implementation_header(self, implementation_type='python'):
        from math import sqrt
        n = int(sqrt(len(self.mc)))
        self.table = generate_ttable(self.mc, self.sbox,n)
        if implementation_type == 'python': 
            return str(self.table_name) + ' = ' + str(self.table)       
        elif implementation_type == 'c': 
            return None
            if self.input_bitsize <= 8: 
                if isinstance(self.input_vars[0], list): return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint8_t ' + 'x;'] + ['uint8_t ' + 'y;']
                else: return ['uint8_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
            else: 
                if isinstance(self.input_vars[0], list): return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};'] + ['uint32_t ' + 'x;'] + ['uint32_t ' + 'y;']
                else: return ['uint32_t ' + str(self.__class__.__name__) + '[' + str(2**self.input_bitsize) + '] = {' + str(self.table)[1:-1] + '};']
        else: return None 
    
    def generate_implementation(self, input_vars, output_vars, implementation_type='python', unroll=False):
        #differ greatly from parent class here we enforce strictly 2d array in inputs
        #we will be taking in a 2d list of vinput
        if implementation_type == 'python': 
            name = self.table_name
            return '[' + ','.join([output_vars[i].ID for i in range(len(output_vars))]) + "] = " + "int("+'^'.join([ name+f"[{i}]"+"["+input_vars[i].ID+"]"  for i in range(len(input_vars))])+')'+ ".to_bytes(4, 'big')" 
        elif implementation_type == 'c': 
            raise Exception(str(self.__class__.__name__) + ": NOT SUPPORTED YET LATER DO '" + implementation_type + "'")
        else: raise Exception(str(self.__class__.__name__) + ": unknown implementation type '" + implementation_type + "'")