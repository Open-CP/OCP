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
from typing import List

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
        self.sboxs:List[Sbox] = self.getSboxes()#just a list of sbox object
        self.sboxPerms:List[int] = []
        self.hasNoSboxCase = False 
        self.poly = 0x1B
        self.word_size = layer.word_bitsize
        for rd in range(len(con)): self.convert_round(rd)
    def getSboxIndex(self, sboxObj:Sbox):
        return [obj.get_header_ID() for obj in self.sboxs].index(sboxObj.get_header_ID())
    def getSboxes(self):
        con = self.layer.constraints
        rtn= [] 
        for rd in range(len(con)):
            names = [self.getLayerName(rd, lyr) for lyr in range(self.layer.nbr_layers)]
            for i,e in enumerate(names):
                if e=="Sbox":
                    #view all constraints to get sbox 
                    for c in con[rd][i]:
                        if isinstance(c,Sbox):
                            if not rtn or all(r.get_header_ID()!=c.get_header_ID() for r in rtn):
                                rtn.append(c)
        return rtn #distinct sbox objects based on get_header_id()    
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
    def generateSbox(self, permList):
        sz = 1<<self.layer.word_bitsize
        rtn = list(range(sz))
        for c in range(sz):
            for pl in permList:
                rtn[c] = self.sboxs[pl].table[c]
                c = rtn[c]
        return rtn 
    def matrix_ttable_conversion(self, output_vars, input_vars, in_idx, out_idx, have_sbox, round, layer, sbox_perm):
        if not have_sbox: self.hasNoSboxCase=True  
        name_list = ["TTABLE_NOSBOX"]*16#initialized to all no ttable  
        #from the sbox_perm gethe number
        for v in range(len(sbox_perm)):
            name_list[v] = self.get_ttable_name(sbox_perm[v])
        sz = 1<<self.layer.word_bitsize
        n = 4
        for r in range(n):
            tmp_in,tmp_out = [],[]
            tmp_name_list = []
            for c in range(n):
                tmp_in.append(input_vars[in_idx[r*n + c]])
                tmp_out.append(output_vars[out_idx[r*n + c]])
                tmp_name_list.append(name_list[in_idx[r*n+c]])
            obj = TTable(self.mat, list(range(sz)), "PLACE HOLDER", self.word_size, self.poly)
            self.con_list[round][layer][r] = obj.generate_implementation(tmp_in, tmp_out, tmp_name_list)
    def xor_conversion(self, output_vars, input_variables, in_idx, out_idx, have_sbox, round, layer, sbox_perm):
        if not have_sbox: self.hasNoSboxCase=True 
        name_list = ["TTABLE_NOSBOX"]*16#initialized to all no ttable  
        sz = 1<<self.layer.word_bitsize
        if have_sbox:
            for v in range(len(sbox_perm)):
                name_list[v] = self.get_ttable_name(sbox_perm[v])
        n = 4 
        for r in range(n):
            tmp_in,tmp_out = [],[]
            tmp_name_list = []
            for c in range(n):
                tmp_in.append([input_variables[o][in_idx[o][r*n + c]] for o in range(len(input_variables))])
                tmp_out.append(output_vars[out_idx[r*n + c]])
                tmp_name_list.extend([name_list[in_idx[o][r*n+c]] for o in range(len(input_variables))])
            obj = TTable(self.mat, list(range(sz)), "PLACE HOLDER", self.word_size, self.poly)
            self.con_list[round][layer][r] = obj.generate_implementation_xor(tmp_in, tmp_out, tmp_name_list)
    def set_layer(self, rd,lyr, status):
        for c in range(len(self.states[rd][lyr])):
            self.states[rd][lyr][c] = status[c]
    def getSboxPermIndex(self, perm):
        return self.sboxPerms.index(perm)
    def get_sbox_mapping(self, l,r, rd):
        con = self.layer.constraints
        hm = [[] for _ in range(16)]#empty means no sbox case for that variable this round
        rtn = [-1]*16
        for v in range(16): 
            for lyr in range(l,r+1):
                c = con[rd][lyr][v]
                if isinstance(c, Sbox):
                    hm[v].append(self.getSboxIndex(c))
            if hm[v] and (hm[v] not in self.sboxPerms): self.sboxPerms.append(hm[v].copy())
        for v,perm in enumerate(hm):
            if perm:rtn[v] = self.getSboxPermIndex(perm)
        return rtn 
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
                    srr = [j for j in range(sidx, midx) if layer_names[j]=="Sbox"]
                    #sbox merged with matrix layer nomater what if have perm before matrix we good
                    self.set_layer(round,srr[0],[TTABLE]*4 +[DELETE]*12)
                    for lr in srr[1:]: self.set_layer(round, lr, [DELETE]*16)
                    self.set_layer(round, midx, [DELETE]*4)
                    sbox_case = True
                    sbox_perm = self.get_sbox_mapping(srr[0], srr[-1], round)#variable -> sbox permutation mapping 
                    input_vars = self.sort_vars(flatten(con.input_vars for con in self.layer.constraints[round][srr[0]]))
                    output_vars = self.sort_vars(flatten(con.output_vars for con in self.layer.constraints[round][midx]))
                    self.matrix_ttable_conversion(output_vars, input_vars, idxs,oidx,sbox_case,round, srr[0], sbox_perm)#instead of sbox_case need to tell which sbox index to use 
                    #if no sbox is used then -1 for hte sbox_case 
                input_variables = []
                input_idxs = []
                output_vars = self.sort_vars(flatten(con.output_vars for con in self.layer.constraints[round][midx]))
                chosen_layer = -1
                #start fomr the last of hte sbox layer 
                st = max([t for t in range(sidx, midx) if layer_names[t]=="Sbox"])
                for lyr in range(st+1, midx):
                    name = self.getLayerName(round, lyr)
                    if name in self.COMMUTATIVE_LAYERS:
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
                    srr = [j for j in range(sidx, midx) if layer_names[j]=="Sbox"]
                    sbox_perm = self.get_sbox_mapping(srr[0], srr[-1], round)#variable -> sbox permutation mapping 
                    self.set_layer(round, chosen_layer, [TTABLE]*4 + [DELETE]*12)
                    self.xor_conversion(output_vars, input_variables, input_idxs, oidx,False,round,chosen_layer,sbox_perm)
                
    def generate_headers(self, language="python"):
        word_size = self.layer.word_bitsize
        rtn = [] 
        if self.hasNoSboxCase:
            tt = TTable(self.mat, list(range(len(self.sbox))), "TTABLE_NOSBOX",word_size,self.poly)
            rtn.append(tt.generate_implementation_header(language)) 
        for perm in self.sboxPerms:
            sbox = self.generateSbox(perm)
            tt = TTable(self.mat, sbox, self.get_ttable_name(perm), word_size,self.poly)
            rtn.append(tt.generate_implementation_header(language))
        return rtn 

    def get_ttable_name(self, perm):
        if type(perm) is int: return f"TTABLE_{perm}"
        idx = self.sboxPerms.index(perm)
        return f"TTABLE_{idx}"

            
