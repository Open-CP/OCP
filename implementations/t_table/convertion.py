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
from implementations.t_table.initializer import flatten, permute, transpose 
from primitives.primitives import Layered_Function
import operators.operators as op
from operators.Sbox import Sbox
from operators.matrix import Matrix
from implementations.t_table.implementation import TTable
DELETE = 1 
TTABLE = 2 
class TTable_Conversion:
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
        self.sbox = []
        self.hasNoSboxCase = False 
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
        if all([isinstance(c,Sbox) for c in con]): return "Sbox"
        if any([type(c) is Matrix for c in con]): return "Matrix"
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
        obj = TTable(self.mat, self.sbox, table_name)
        n = 4
        for r in range(n):
            tmp_in,tmp_out = [],[]
            for c in range(n):
                tmp_in.append(input_vars[in_idx[r*n + c]])
                tmp_out.append(output_vars[out_idx[r*n + c]])
            self.con_list[round][layer][r] = obj.generate_implementation(tmp_in, tmp_out)
    def set_layer(self, rd,lyr, status):
        for c in range(len(self.states[rd][lyr])):
            self.states[rd][lyr][c] = status 
    def convert_round(self, round):
        """
        Docstring for convert_round
        
        :param self: Description
        :param round: The round i want to convert 
        
        for now just assume each appears only once 
        wnat to squeeze perm sbox matrix -> ttablelayer

        will delete sbox, perm,
        replace matrix layer 

        mutsl also determine how many ttable must ge gneerateraed 
        will create a ttable layer here 
        but will throw its own con list
        so have a contlist here as well 
        """
        layer_names = [self.getLayerName(round, lyr) for lyr in range(self.layer.nbr_layers)] 
        if "Matrix" in layer_names:
            midx = layer_names.index("Matrix")
            self.mat = flatten(self.layer.constraints[round][midx][0].mat)
            sidx = -1 
            pidx = -1 
            if "Permutation" in layer_names: pidx = layer_names.index("Permutation")
            if "Sbox" in layer_names: sidx = layer_names.index("Sbox")
            #apply perm to layer perm and down 
            if sidx!=-1:#have sbox case 
                self.sbox = self.layer.constraints[round][sidx][0].table
                if sidx < midx:
                    for lyr in range(sidx+1, midx+1):
                        if lyr > pidx:
                            perm = self.getPermMapping(self.layer.constraints[round][pidx])
                            self.set_layer(round,pidx,DELETE)
                            if lyr==midx:
                                #sbox also need to go 
                                self.set_layer(round, sidx, DELETE)
                                self.set_layer(round,midx,TTABLE)
                                mc_idx = self.getMcMapping(self.layer.constraints[round][midx])
                                idxs = permute(perm, mc_idx)
                                oidx = mc_idx 
                                sbox_case = True 
                                input_vars = self.sort_vars(flatten(con.input_vars for con in self.layer.constraints[round][sidx]))
                                output_vars = self.sort_vars(flatten(con.output_vars for con in self.layer.constraints[round][midx]))
                                self.matrix_ttable_conversion(output_vars, input_vars, idxs,oidx,sbox_case,round, lyr)

                            #will apply perm here 
                            #give me the ttable equivalent 
            else:#dont have sbox case
                #this is only a sufficient condition for no sbox case 
                #there may be no sbox case in above case particularly for skinny
                pass 
    def generate_headers(self, language="python"):
        rtn = [] 
        if self.hasNoSboxCase:
            tt = TTable(self.mat, list(len(self.sbox)), "TTABLE_NOSBOX")
            rtn.append(tt.generate_implementation_header(language))
        tt = TTable(self.mat, self.sbox, "TTABLE")
        rtn.append(tt.generate_implementation_header(language))
        return rtn 
            
