"""
Docstring for implementations.t_table.convertion

This is to convert layers per round to ttable 
format.

only want to play with constraits here

must also deal with headers

things of concern are the operators 

Matrix (for matrix layer)
Equal (for permuation must extract the in/out variables number ting)
Sbox (for sbox layer)

after the convertion 

need to say which round and which layer
needs the oritinal line 
and which one isdeleted
which one is ttable guy 

only study the permutation

three posible statea
original : 0 
delelte : 1
ttable : 2 

"""
from implementations.t_table.helper import flatten, permute, transpose 
from primitives.primitives import Layered_Function
import operators.operators as op
from operators.boolean_operators import ConstantXOR, XOR

from operators.Sbox import Sbox
from operators.matrix import Matrix
from implementations.t_table.implementation import TTable
DELETE = 1 
TTABLE = 2 
class TTable_Conversion:
    #these layers pairwise commutative(under composition) and the functions are additive
    #additional we know only commutative if all same operator(i.e. (equal and ^ only) or (equal and & only))
    COMMUTATIVE_LAYERS = ["ConstantXor", "AddRoundKey"] 
    def __init__(self, layer: Layered_Function):
        self.layer = layer 
        con = self.layer.constraints
        self.states = [] 
        self.con_list = [] 
        for rd in range(len(con)):
            self.states.append([])
            self.con_list.append([])
            for lyr in range(len(con[rd])):
                self.states[-1].append([])
                self.con_list[-1].append([])
                for _ in range(len(con[rd][lyr])):
                    self.states[-1][-1].append(0)
                    self.con_list[-1][-1].append(None)
        self.sboxs = []
        self.hasNoSboxCase = False 
        self.poly = 0x1B
        self.word_size = layer.word_bitsize
        for rd in range(len(con)): self.convert_round(rd)
        
        
    def getPermMapping(self, arr):
        return [int(c.input_vars[0].ID.split("_")[-1]) for c in arr]
    def sort_vars(self, arr):
        rtn = arr.copy()
        rtn.sort(key=lambda x: int(x.ID.split("_")[-1]) )
        return rtn 
    def getMcMapping(self, arr):
        rtn = [] 
        for a in arr:
            rtn.extend(int(v.ID.split("_")[-1]) for v in a.input_vars)
        return rtn      
    def getLayerName(self, rd, lyr):
        con = self.layer.constraints[rd][lyr]
        if not con: return "NoTouch"
        nbr_words = self.layer.nbr_words
        if all([type(c) is op.Equal for c in con]):
            self.getPermMapping(con)
            perm = self.getPermMapping(con)
            if not all([perm[i]==i for i in range(nbr_words)]): return "Permutation" 
            return "Identity"
        if any([isinstance(c,Sbox) for c in con]): return "Sbox"
        if any([type(c) is Matrix for c in con]): return "Matrix"
        if any([type(c) is ConstantXOR for c in con]): return "ConstantXor"
        if any([type(c) is XOR for c in con]): return "AddRoundKey"
        return "NoTouch"   
    def matrix_ttable_conversion(self, output_vars, input_vars, in_idx, out_idx, have_sbox, round, layer):
        """
        Docstring for matrix_ttable_conversion
        
        :param self: Description
        :param output_vars: output variables 
        :param input_vars: input variables
        :param in_idx: input index for ttable
        :param out_indx: output index for ttable
        :param have_sbox: ttable with sbox flag or not
        :param mat: the matrix being used 
        This method is to update the con_list
        """
        table_name = "TTABLE" if have_sbox else "TTABLE_NOSBOX"
        if not have_sbox: self.hasNoSboxCase=True
        #here need to see which sbox is used 
        obj = TTable(self.mat, self.sbox, table_name, self.word_size, self.poly)
        n = 4
        for r in range(n):
            tmp_in,tmp_out = [],[]
            for c in range(n):
                tmp_in.append(input_vars[in_idx[r*n + c]])
                tmp_out.append(output_vars[out_idx[r*n + c]])
            self.con_list[round][layer][r] = obj.generate_implementation(tmp_in, tmp_out)
    def xor_conversion(self, output_vars, input_variables, in_idx, out_idx, have_sbox, round, layer):
        #my_constant=hex(self.table[self.table_r-1][self.table_i])
        #if constant the generaterimpleentent taio is hex of hte table
        #other wise 
        table_name = "TTABLE" if have_sbox else "TTABLE_NOSBOX"
        if not have_sbox: self.hasNoSboxCase=True 
        obj = TTable(self.mat, list(range(256)), table_name, self.word_size, self.poly)
        n = 4 
        for r in range(n):
            tmp_in,tmp_out = [],[]
            for c in range(n):
                tmp_in.append([input_variables[o][in_idx[o][r*n + c]] for o in range(len(input_variables))]  )
                tmp_out.append(output_vars[out_idx[r*n + c]])
            self.con_list[round][layer][r] = obj.generate_implementation_xor(tmp_in, tmp_out)
                
        
        """
        much change a titllti e hte genreat implemetnatio 
        is not varible name no mo 
        and it is ^=the out varable 
        """
    
    def set_layer(self, rd,lyr, status):
        for c in range(len(self.states[rd][lyr])):
            self.states[rd][lyr][c] = status[c]
             
    def convert_round(self, round):
        layer_names = [self.getLayerName(round, lyr) for lyr in range(self.layer.nbr_layers)] 
        if "Matrix" in layer_names:
            midx = layer_names.index("Matrix")
            if self.layer.constraints[round][midx][0].polynomial:
                self.poly = int(self.layer.constraints[round][midx][0].polynomial,16)
            else: self.poly=0
            self.mat = flatten(self.layer.constraints[round][midx][0].mat)
            sidx = -1 
            pidx = -1 
            if "Permutation" in layer_names: pidx = layer_names.index("Permutation")
            if "Sbox" in layer_names: sidx = layer_names.index("Sbox")
            #apply perm to layer perm and down 
            if sidx!=-1:#have sbox case 
                self.sbox = self.layer.constraints[round][sidx][0].table
                if pidx==-1 or pidx > midx: 
                    perm = list(range(16))
                else:
                    perm = self.getPermMapping(self.layer.constraints[round][pidx])
                    self.set_layer(round,pidx,[DELETE]*16)
                mc_idx = self.getMcMapping(self.layer.constraints[round][midx])
                idxs = permute(perm, mc_idx)
                oidx = mc_idx 
                if sidx < midx:
                    """
                    assume that sbox is contiguous 

                    need to apply to the 

                    sbox to the neccessary spots
                    """
                    srr = [j for j in range(sidx, midx) if layer_names[j]=="Sbox"]
                    #sbox merged with matrix layer nomater what if have perm before matrix we good
                    self.set_layer(round,sidx,[TTABLE]*4 +[DELETE]*12)
                    for lr in srr[1:]: self.set_layer(round, lr, [DELETE]*16)
                    self.set_layer(round, midx, [DELETE]*4)
                    sbox_case = True 
                    input_vars = self.sort_vars(flatten(con.input_vars for con in self.layer.constraints[round][sidx]))
                    output_vars = self.sort_vars(flatten(con.output_vars for con in self.layer.constraints[round][midx]))
                    self.matrix_ttable_conversion(output_vars, input_vars, idxs,oidx,sbox_case,round, sidx)
                input_variables = []
                input_idxs = []
                output_vars = self.sort_vars(flatten(con.output_vars for con in self.layer.constraints[round][midx]))
                chosen_layer = -1
                for lyr in range(sidx+1, midx):
                    name = self.getLayerName(round, lyr)
                    if name in self.COMMUTATIVE_LAYERS:  #linearity here assumed 
                        chosen_layer = lyr
                        #the input should be all strings
                        self.set_layer(round, lyr, [DELETE]*16)
                        cons = [con for con in self.layer.constraints[round][lyr]]
                        input_vars = []
                        if name=="ConstantXor":
                            for c in cons:
                                if type(c) is ConstantXOR: input_vars.append(hex(c.table[c.table_r-1][c.table_i]))
                                elif type(c) is op.Equal: input_vars.append(hex(0))
                        elif name=="AddRoundKey":
                            for c in cons:
                                if type(c) is XOR: input_vars.append(c.get_var_ID("in", 1, True))
                                elif type(c) is op.Equal: input_vars.append(hex(0))
                        else:
                            raise Exception(f"{name} Layer is not supported for TTable conversion yet")
                        input_variables.append(input_vars)
                        input_idxs.append(idxs.copy())
                    else:
                        if name=="Permutation": idxs = mc_idx.copy()
                        else:
                            raise Exception(f"{name} Layer is not supported for TTAble conversion yet")
                if chosen_layer!=-1:
                    self.set_layer(round, chosen_layer, [TTABLE]*4 + [DELETE]*12)
                    self.xor_conversion(output_vars, input_variables, input_idxs, oidx,False,round,chosen_layer)
                
    def generate_headers(self, language="python"):
        word_size = self.layer.word_bitsize
        rtn = [] 
        if self.hasNoSboxCase:
            tt = TTable(self.mat, list(range(len(self.sbox))), "TTABLE_NOSBOX",word_size,self.poly)
            rtn.append(tt.generate_implementation_header(language))
        tt = TTable(self.mat, self.sbox, "TTABLE", word_size,self.poly)
        rtn.append(tt.generate_implementation_header(language))
        return rtn 
            
