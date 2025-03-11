from abc import ABC
import variables.variables as var
import operators.operators as op

def generateID(name, round_nb, layer, position):
    return name + '_' + str(round_nb) + '_' + str(layer) + '_' + str(position)

# ********************* STATES ********************* #
# Class that represents a state object, i.e. a collection of words of the same bitsize that will be updated through a certain number of rounds each composed of a certain number of layers
# This object will contain the list of variables representing the words at each stage of the computation
# This object will contain the list of constraints linking the variables together

class State:
    def __init__(self, name, label, nbr_rounds, nbr_layers, nbr_words, nbr_temp_words, word_bitsize):
        self.name = name                      # name of the state 
        self.label = label                    # label for display when refering that state 
        self.nbr_rounds = nbr_rounds          # number of layers per round in that state 
        self.nbr_layers = nbr_layers          # number of layers per round in that state 
        self.nbr_words = nbr_words            # number of words in that state
        self.nbr_temp_words = nbr_temp_words  # number of temporary words in that state
        self.word_bitsize = word_bitsize      # number of bits per word in that state
        self.vars = []                    
        self.constraints = []  
        
        # list of variables for that state (indexed with vars[r][l][n] where r is the round number, l the layer number, n the word number)
        self.vars = [[[] for i in range(nbr_layers+1)] for j in range(nbr_rounds+1)] 
        
        # list of constraints for that state (indexed with constraints[r][l][i] where r is the round number, l the layer number, i the constraint number)
        self.constraints = [[[] for i in range(nbr_layers+1)] for j in range(nbr_rounds+1)]  
    
        # create variables
        for i in range(0,nbr_rounds+1):  
            for l in range(0,nbr_layers+1):
                self.vars[i][l] = [var.Variable(word_bitsize, ID = generateID('v' + label,i,l,j)) for j in range(nbr_words + nbr_temp_words)]
                
        # create initial constraints
        for i in range(0,nbr_rounds):  
            self.constraints[i][nbr_layers] = [op.Equal([self.vars[i][nbr_layers][j]], [self.vars[i+1][0][j]], ID=generateID('LINK_' + label,i,nbr_layers,j)) for j in range(nbr_words + nbr_temp_words)]
            

    def display(self, representation='binary'):   # method that displays in details the state
        print("Name: " + str(self.name), " / nbr_words: " + str(self.nbr_words), " / word_bitsize: " + str(self.word_bitsize))
        print("Vars: [" + str([ len(self.vars[i]) for i in range(len(self.vars))])   + "]")
        print("Constraints: [" + str([ len(self.constraints[i]) for i in range(len(self.constraints))])  + "]")
        
    # apply a layer "name" of an Sbox, at the round "crt_round", at the layer "crt_layer", with the Sbox operator "sbox_operator". Only the positions where mask=1 will have the Sbox applied, the rest being just identity  
    def SboxLayer(self, name, crt_round, crt_layer, sbox_operator, mask = None, index=None):
        if index is not None:
            bitsize = len(index[0])
            n_words = int((self.nbr_words+self.nbr_temp_words)/bitsize)
            if mask is None: mask = [1]*int(self.nbr_words/bitsize) 
            if len(mask)<n_words: mask = mask + [0]*(n_words - len(mask))
            for j in range(n_words):
                if mask[j]==1: 
                    in_var, out_var = [self.vars[crt_round][crt_layer][i] for i in index[j]], [self.vars[crt_round][crt_layer+1][i] for i in index[j]]
                    self.constraints[crt_round][crt_layer].append(sbox_operator([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
                else: 
                    for i in range(bitsize):
                        in_var, out_var = self.vars[crt_round][crt_layer][j*bitsize+i], self.vars[crt_round][crt_layer+1][j*bitsize+i]
                        self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
        else:
            if mask is None: mask = [1]*self.nbr_words 
            if len(mask)<(self.nbr_words + self.nbr_temp_words): mask = mask + [0]*(self.nbr_words + self.nbr_temp_words - len(mask))
            for j in range(self.nbr_words + self.nbr_temp_words):
                in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
                if mask[j]==1: self.constraints[crt_round][crt_layer].append(sbox_operator([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
       
    # apply a layer "name" of a Permutation, at the round "crt_round", at the layer "crt_layer", with the permutation "permutation". 
    def PermutationLayer(self, name, crt_round, crt_layer, permutation):
        if len(permutation)<(self.nbr_words + self.nbr_temp_words): permutation = permutation + [i for i in range(len(permutation), self.nbr_words + self.nbr_temp_words)] 
        for j in range(len(permutation)):
            in_var, out_var = self.vars[crt_round][crt_layer][permutation[j]], self.vars[crt_round][crt_layer+1][j]
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))

    # apply a layer "name" of a Rotation, at the round "crt_round", at the layer "crt_layer" on the word of the state located at position "index". The rotation direction is "direction" and the rotation amount is "amount". If "index_out" is specified, then the output is placed in index_out
    def RotationLayer(self, name, crt_round, crt_layer, rot, index_in, index_out=None):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if index_out==None:
                if j in index_in: self.constraints[crt_round][crt_layer].append(op.Rot([in_var], [out_var], rot[0], rot[1], ID=generateID(name,crt_round,crt_layer,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            else:
                if j in index_out: 
                    in_var = self.vars[crt_round][crt_layer][index_in[index_out.index(j)]]
                    self.constraints[crt_round][crt_layer].append(op.Rot([in_var], [out_var], rot[0], rot[1], ID=generateID(name,crt_round,crt_layer,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            
    # apply a layer "name" of a simple identity at the round "crt_round", at the layer "crt_layer". 
    def AddIdentityLayer(self, name, crt_round, crt_layer):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
        
    # apply a layer "name" of a Constant addition, at the round "crt_round", at the layer "crt_layer", with the adding "add_type" and the constant value "constant". 
    def AddConstantLayer(self, name, crt_round, crt_layer, add_type, constant, constant_table):
        if len(constant)<(self.nbr_words + self.nbr_temp_words): constant = constant + [None]*(self.nbr_words + self.nbr_temp_words - len(constant))
        i = 0
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if constant[j]!=None: 
                self.constraints[crt_round][crt_layer].append(op.ConstantAdd([in_var], [out_var], add_type, constant_table, crt_round, i, ID=generateID(name,crt_round,crt_layer,j)))  
                i += 1
            else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
    
    # apply a layer "name" of a single operator "my_operator" with input indexes "index_in" and output indexes "index_out", at the round "crt_round", at the layer "crt_layer". The other output indexes are just being applied identity
    def SingleOperatorLayer(self, name, crt_round, crt_layer, my_operator, index_in, index_out, mat=None):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if j not in index_out: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            else: 
                in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[index_out.index(j)]]
                if mat: self.constraints[crt_round][crt_layer].append(my_operator(in_vars, [out_var], ID=generateID(name,crt_round,crt_layer,j), mat=mat))
                else: self.constraints[crt_round][crt_layer].append(my_operator(in_vars, [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            
    # apply a layer "name" of a Matrix "mat" (only square matrix), at the round "crt_round", at the layer "crt_layer", operating in the field GF(2^"bitsize") with polynomial "polynomial"
    def MatrixLayer(self, name, crt_round, crt_layer, mat, indexes_list, polynomial = None):
        m = len(mat)
        for i in mat: 
            if len(i)!=m: raise Exception("MatrixLayer: matrix shape is not square") 
        flat_indexes = [x for sublist in indexes_list for x in sublist]
        for j in range(self.nbr_words + self.nbr_temp_words):
            if j not in flat_indexes: 
                self.constraints[crt_round][crt_layer].append(op.Equal([self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]], ID=generateID(name,crt_round,crt_layer,j)))
        for j, indexes in enumerate(indexes_list): 
            if len(indexes)!=m: raise Exception("MatrixLayer: input vector does not match matrix size") 
            self.constraints[crt_round][crt_layer].append(op.Matrix(name, [self.vars[crt_round][crt_layer][x] for x in indexes], [self.vars[crt_round][crt_layer+1][x] for x in indexes], mat = mat, polynomial = polynomial, ID=generateID(name,crt_round,crt_layer,j)) )
       
    # extract a subkey from the external variable, determined by "extraction_mask"
    def ExtractionLayer(self, name, crt_round, crt_layer, extraction_indexes, external_variable):
        for j, indexes in enumerate(extraction_indexes):
            in_var, out_var = external_variable[indexes], self.vars[crt_round][crt_layer+1][j] 
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))   
    
    # apply a layer "name" of an AddRoundKeyLayer addition, at the round "crt_round", at the layer "crt_layer", with the adding operator "my_operator". Only the positions where mask=1 will have the AddRoundKey applied, the rest being just identity  
    def AddRoundKeyLayer(self, name, crt_round, crt_layer, my_operator, sk_state, mask = None):
        if sum(mask)!=sk_state.nbr_words: raise Exception("AddRoundKeyLayer: subkey size does not match the mask") 
        if len(mask)<(self.nbr_words + self.nbr_temp_words): mask += [0]*(self.nbr_words + self.nbr_temp_words - len(mask))
        cpt = 0
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if mask[j]==1: 
                sk_var = sk_state.vars[crt_round][-1][cpt]                
                self.constraints[crt_round][crt_layer].append(my_operator([in_var, sk_var], [out_var], ID=generateID(name,crt_round,crt_layer,cpt))) 
                cpt = cpt + 1
            else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,cpt))) 
                   
  
                                     

# ********************* PRIMITIVES ********************* #
# Class that represents a primitive object, i.e. a cryptographic algorithm such as a permutation, a block cipher etc. 
# This object makes the link between what is a specific cryptographic primitive and various "states", "variables", "operators"
# These objects are the ones to be instantiated by the user for analysing a cipher

class Primitive(ABC):
    def __init__(self, name, inputs, outputs):
        self.name = name                # name of the primitive
        self.inputs = inputs            # list of the inputs of the primitive
        self.outputs = outputs          # list of the outputs of the primitive
        self.states = []                # list of states used by the primitive
        self.inputs_constraints = []    # constraints linking the primitive inputs to the states input variables
        self.outputs_constraints = []   # constraints linking the primitive outputs to the states output variables
        self.rounds_python_code_if_unrolled = {}
        self.rounds_c_code_if_unrolled = {}
        
        
# ********************************************** PERMUTATIONS **********************************************
# Subclass that represents a permutation object    
# A permutation is composed of a single state 
     
class Permutation(Primitive):
    def __init__(self, name, s_input, s_output, nbr_rounds, config):
        super().__init__(name, {"IN":s_input}, {"OUT":s_output})
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = config[0], config[1], config[2], config[3]
        self.nbr_rounds = nbr_rounds
        self.states = {"STATE": State("STATE", "", nbr_rounds, nbr_layers, nbr_words, nbr_temp_words, word_bitsize)}
        self.states_implementation_order = ["STATE"]
        self.states_display_order = ["STATE"]
        
        if len(s_input)!=nbr_words: raise Exception("Permutation: the number of input words does not match the number of words in state") 
        for i in range(len(s_input)): self.inputs_constraints.append(op.Equal([s_input[i]], [self.states["STATE"].vars[1][0][i]], ID='IN_LINK_'+str(i)))
            
        if len(s_output)!=nbr_words: raise Exception("Permutation: the number of output words does not match the number of words in state") 
        for i in range(len(s_output)): self.outputs_constraints.append(op.Equal([self.states["STATE"].vars[nbr_rounds][nbr_layers][i]], [s_output[i]], ID='OUT_LINK_'+str(i)))
        

# The Skinny internal permutation       
class Skinny_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, model_type=0):
        
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==64 else 64 if version==128 else None
        if model_type==0:  nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, int(p_bitsize/16)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                  0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                  0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                  0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                  0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):              
                if word_bitsize==4: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_4bit_Sbox) 
                else: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_8bit_Sbox)  # Sbox layer            
                rc = round_constants[i-1]
                c0, c1, c2 = rc & 0xF, rc >> 4, 0x2       
                self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [c0,None,None,None, c1,None,None,None, c2], code_if_unrolled=["RC[i] & 0xF", "RC[i] >> 4", "0X2"], constants_if_unrolled=f"RC={round_constants}")  # Constant layer            
                self.states["STATE"].PermutationLayer("SR", i, 2, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                self.states["STATE"].MatrixLayer("MC", i, 3, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer


# The Speck internal permutation  
class Speck_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, model_type=0):
                
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=22 if version==32 else 22 if version==48 else 26 if version==64 else 28 if version==96 else 32 if version==128 else None
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 2, 0, p_bitsize>>1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        if version==32: rotr, rotl = 7, 2
        else: rotr, rotl = 8, 3
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], [0]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer   
                self.states["STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], [1]) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[0,1]], [1]) # XOR layer 
  
    
# The Simon internal permutation  
class Simon_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, model_type=0):
                
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==32 else 36 if version==48 else 42 if version==64 else 52 if version==96 else 68 if version==128 else None
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 7, 2, 2, p_bitsize>>1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):                
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], [0], index_out=[2]) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], [0], index_out=[3]) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], [0], index_out=[2]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].PermutationLayer("PERM", i, 6, [1,0]) # Permutation layer
            
                
# The ASCON internal permutation             
class ASCON_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=12
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 320, 320, 1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        cons = [0xf0 - r*0x10 + r*0x1 for r in range(12)]
        # create constraints
        if model_type==0: 
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].AddConstantLayer("C", i, 0, "xor", [None]*184+[int(bit) for bit in format(cons[12-nbr_rounds+i-1], '08b')], code_if_unrolled=[f"((RC[{12-nbr_rounds}+i] >> {7-l}) & 1)" for l in range(8)], constants_if_unrolled=f"RC={cons}")  # Constant layer      
                self.states["STATE"].SboxLayer("SB", i, 1, op.ASCON_Sbox, index=[[k+j*64 for j in range(5)] for k in range(64)])  # Sbox layer            
                self.states["STATE"].SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[(45+j)%64, (36+j)%64] for j in range(64)]+[[(3+j)%64+64, (25+j)%64+64] for j in range(64)]+[[(63+j)%64+128, (58+j)%64+128] for j in range(64)]+[[(54+j)%64+192, (47+j)%64+192] for j in range(64)]+[[(57+j)%64+256, (23+j)%64+256] for j in range(64)], [j for j in range(320,640)]) # XOR layer 
                self.states["STATE"].SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[j, j+320] for j in range(320)], [j for j in range(320)]) # XOR layer 
                

# The GIFT internal permutation               
class GIFT_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=28 if version==64 else 40 if version==128 else None
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 3, version, 0, 1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        const = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
        if version == 64: perm = [12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63]
        elif version == 128: perm = [12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 76, 65, 70, 75, 92, 81, 86, 91, 108, 97, 102, 107, 124, 113, 118, 123, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 72, 77, 66, 71, 88, 93, 82, 87, 104, 109, 98, 103, 120, 125, 114, 119, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 68, 73, 78, 67, 84, 89, 94, 83, 100, 105, 110, 99, 116, 121, 126, 115, 0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63, 64, 69, 74, 79, 80, 85, 90, 95, 96, 101, 106, 111, 112, 117, 122, 127]
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):              
                self.states["STATE"].SboxLayer("SB", i, 0, op.GIFT_Sbox,index=[list(range(i, i + 4)) for i in range(0, nbr_words, 4)])  # Sbox layer            
                self.states["STATE"].PermutationLayer("P", i, 1, perm) # Permutation layer
                if version == 64: self.states["STATE"].AddConstantLayer("C", i, 2, "xor", [1]+[None]*39+[(const[i-1]>>5)&1]+[None]*3+[(const[i-1]>>4)&1]+[None]*3+[(const[i-1]>>3)&1]+[None]*3+[(const[i-1]>>2)&1]+[None]*3 +[(const[i-1]>>1)&1]+[None]*3+[(const[i-1])&1], code_if_unrolled=["1"] + [f"(RC[i]>>{5-a})&1" for a in range(6)], constants_if_unrolled=f"RC={const}")# Constant layer                      
                elif version == 128: self.states["STATE"].AddConstantLayer("C", i, 2, "xor", [1]+[None]*103+[(const[i-1]>>5)&1]+[None]*3+[(const[i-1]>>4)&1]+[None]*3+[(const[i-1]>>3)&1]+[None]*3+[(const[i-1]>>2)&1]+[None]*3 +[(const[i-1]>>1)&1]+[None]*3+[(const[i-1])&1], code_if_unrolled=["1"] + [f"(RC[i]>>{5-a})&1" for a in range(6)], constants_if_unrolled=f"RC={const}")# Constant layer            

# The AES internal permutation  
class AES_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=10
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, 8
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        full_rounds = 10
        if nbr_rounds == full_rounds: self.rounds_python_code_if_unrolled, self.rounds_c_code_if_unrolled = {"STATE": [[1, f"if i < {full_rounds-1}:"], [full_rounds, f"elif i == {full_rounds-1}:"]]}, {"STATE": [[1, f"if (i < {full_rounds-1})"+"{"], [full_rounds, f"else if (i == {full_rounds-1})"+"{"]]}

        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):             
                self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                self.states["STATE"].PermutationLayer("SR", i, 1, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]) # Shiftrows layer
                if i != full_rounds: self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]], "0x1B")  #Mixcolumns layer
                else: self.states["STATE"].AddIdentityLayer("ID", i, 2)     # Identity layer 
                self.states["STATE"].AddConstantLayer("AC", i, 3, "xor", [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14], ["0x0","0x1","0x2","0x3","0x5","0x6","0x7","0x4","0xa","0xb","0x8","0x9","0xf","0xc","0xd","0xe"])  # Constant layer            
                    
                    
class Rocca_AD_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=20
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 128+32*nbr_rounds, 32, 0
        perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if model_type==0:
            for r in range(1, nbr_rounds+1):
                self.states["STATE"].PermutationLayer("P", r, 0, [i for i in range(16*7,16*8)] + perm_s + [i for i in range(16,16*2)] + [perm_s[i]+16*2 for i in range(16)] + [i for i in range(16*3,16*4)] + [perm_s[i]+16*4 for i in range(16)] + [perm_s[i]+16*5 for i in range(16)] + [i for i in range(16*6, 16*7)] + [i for i in range(16*8,16*(8+2*nbr_rounds))] + [i for i in range(16)] + [i for i in range(16*4, 16*5)])
                self.states["STATE"].SboxLayer("SB", r, 1, op.AES_Sbox, mask=[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[1 for i in range(16)]) # Sbox layer 
                self.states["STATE"].MatrixLayer("MC", r, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0+16,1+16,2+16,3+16], [4+16,5+16,6+16,7+16], [8+16,9+16,10+16,11+16], [12+16,13+16,14+16,15+16],[0+16*3,1+16*3,2+16*3,3+16*3], [4+16*3,5+16*3,6+16*3,7+16*3], [8+16*3,9+16*3,10+16*3,11+16*3], [12+16*3,13+16*3,14+16*3,15+16*3],[0+16*5,1+16*5,2+16*5,3+16*5], [4+16*5,5+16*5,6+16*5,7+16*5], [8+16*5,9+16*5,10+16*5,11+16*5], [12+16*5,13+16*5,14+16*5,15+16*5], [0+16*6,1+16*6,2+16*6,3+16*6], [4+16*6,5+16*6,6+16*6,7+16*6], [8+16*6,9+16*6,10+16*6,11+16*6], [12+16*6,13+16*6,14+16*6,15+16*6]], "0x1B")  #Mixcolumns layer
                self.states["STATE"].SingleOperatorLayer("XOR", r, 3, op.bitwiseXOR, [[i, i+16*(8+2*(r-1))] for i in range(16)] + [[i+16, i] for i in range(16)] + [[i+16*2, i+16*7] for i in range(16)] + [[i+16*3, i+16*2] for i in range(16)] + [[i+16*4, i+16*(8+2*(r-1)+1)] for i in range(16)] + [[i+16*5, i+16*4]  for i in range(16)] + [[i+16*6, i+16*(8+2*nbr_rounds+1)] for i in range(16)] + [[i+16*7, i+16*(8+2*nbr_rounds)] for i in range(16)],[i for i in range(16*8)]) # XOR layer 
               
            

# ********************************************** BLOCK CIPHERS **********************************************
# Subclass that represents a block cipher object 
# A block cipher is composed of three states: a permutation (to update the cipher state), a key schedule (to update the key state), and a round-key computation (to compute the round key from the key state)

class Block_cipher(Primitive):
    def __init__(self, name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, s_config, k_config, sk_config):
        super().__init__(name, {"plaintext":p_input, "key":k_input}, {"ciphertext":c_output})
        s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize = s_config[0], s_config[1], s_config[2], s_config[3]
        k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize = k_config[0], k_config[1], k_config[2], k_config[3]
        sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize = sk_config[0], sk_config[1], sk_config[2], sk_config[3]
        self.nbr_rounds = nbr_rounds
        self.states = {"STATE": State("STATE", 's', nbr_rounds, s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), "KEY_STATE": State("KEY_STATE", 'k', k_nbr_rounds, k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), "SUBKEYS": State("SUBKEYS", 'sk', nbr_rounds, sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize)}
        self.states_implementation_order = ["SUBKEYS", "KEY_STATE", "STATE"]
        self.states_display_order = ["STATE", "KEY_STATE", "SUBKEYS"]
        
        if (len(k_input)!=k_nbr_words) or (len(p_input)!=s_nbr_words): raise Exception("Block_cipher: the number of input plaintext/key words does not match the number of plaintext/key words in state") 
        
        if len(p_input)!=s_nbr_words: raise Exception("Block_cipher: the number of plaintext words does not match the number of words in state") 
        for i in range(len(p_input)): self.inputs_constraints.append(op.Equal([p_input[i]], [self.states["STATE"].vars[1][0][i]], ID='IN_LINK_P_'+str(i)))
        
        if len(k_input)!=k_nbr_words: raise Exception("Block_cipher: the number of key words does not match the number of words in state") 
        for i in range(len(k_input)): self.inputs_constraints.append(op.Equal([k_input[i]], [self.states["KEY_STATE"].vars[1][0][i]], ID='IN_LINK_K_'+str(i)))
            
        if len(c_output)!=s_nbr_words: raise Exception("Block_cipher: the number of ciphertext words does not match the number of words in state") 
        for i in range(len(c_output)): self.outputs_constraints.append(op.Equal([self.states["STATE"].vars[nbr_rounds][s_nbr_layers][i]], [c_output[i]], ID='OUT_LINK_C_'+str(i)))
          
       
# The Skinny block cipher        
class Skinny_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
        
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(64,64) else 36 if (version[0],version[1])==(64,128) else 40 if (version[0],version[1])==(64,192)  else 40 if (version[0],version[1])==(128,128)  else 48 if (version[0],version[1])==(128,256)  else 56 if (version[0],version[1])==(128,384) else None
        tweak_size = int(k_bitsize/p_bitsize)
        if model_type==0: 
            if tweak_size ==1: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (1, int(16*k_bitsize / p_bitsize), 0, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif tweak_size == 2: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (3, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif tweak_size ==3: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (5, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
        if tweak_size == 1: k_nbr_rounds = nbr_rounds
        else: k_nbr_rounds = nbr_rounds + 1
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                  0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                  0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                  0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                  0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
        k_perm = [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]
        if tweak_size >= 2: 
            self.states_implementation_order = ["KEY_STATE", "SUBKEYS", "STATE"]
        
        # create constraints
        if model_type==0:

            # subkeys extraction
            for i in range(1,nbr_rounds+1): 
                if tweak_size == 1:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(8)], self.states["KEY_STATE"].vars[i][0])
                elif tweak_size >= 2:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(tweak_size*16,tweak_size*16+8)], self.states["KEY_STATE"].vars[i+1][0])
            

            # key schedule
            k_perm_T = [i + 16 * j for j in range(int(k_bitsize/p_bitsize)) for i in k_perm]
            if tweak_size == 1:
                for i in range(1, k_nbr_rounds): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
            elif tweak_size == 2: 
                for i in range(1, k_nbr_rounds): 
                    if i == 1: 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 0)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 1)     # Identity layer 
                    else:
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        if int(p_bitsize/16) == 4: self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=[[1,None],[2,None],[3,None],[0,1]])
                        elif int(p_bitsize/16) == 8: self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=[[1,None],[2,None],[3,None],[4,None],[5,None],[6,None],[7,None],[0,2]])
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 2, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(32,40)]) # XOR layer 
                if k_nbr_rounds >= 2: 
                    self.rounds_python_code_if_unrolled = {"KEY_STATE": [[1, "if i == 0:"], [2, "else:"]]}
                    self.rounds_c_code_if_unrolled = {"KEY_STATE": [[1, "if (i == 0)"+"{"], [2, "else{"]]}
            elif tweak_size == 3:
                for i in range(1, k_nbr_rounds): 
                    if i == 1: 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 0)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 1)     # Identity layer
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 2)     # Identity layer 
                    else:
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        if int(p_bitsize/16) == 4: 
                            self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=[[1,None],[2,None],[3,None],[0,1]])
                            self.states["KEY_STATE"].SingleOperatorLayer("K_XOR2", i, 2, op.bitwiseXOR, [[j,j] for j in range(32,40)], [j for j in range(32,40)], mat=[[0,3],[0,None],[1,None],[2,None]])
                        elif int(p_bitsize/16) == 8: 
                            self.states["KEY_STATE"].SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=[[1,None],[2,None],[3,None],[4,None],[5,None],[6,None],[7,None],[0,2]])
                            self.states["KEY_STATE"].SingleOperatorLayer("K_XOR2", i, 2, op.bitwiseXOR, [[j,j] for j in range(32,40)], [j for j in range(32,40)], mat=[[1,7],[0,None],[1,None],[2,None],[3,None],[4,None],[5,None],[6,None]])
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(48,56)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j,16+j] for j in range(32,40)], [j for j in range(48,56)]) # XOR layer 
                if k_nbr_rounds >= 2: 
                    self.rounds_python_code_if_unrolled = {"KEY_STATE": [[1, "if i == 0:"], [2, "else:"]]}
                    self.rounds_c_code_if_unrolled = {"KEY_STATE": [[1, "if (i == 0)"+"{"], [2, "else{"]]}
                
            # Internal permutation
            for i in range(1,nbr_rounds+1):  
                if s_word_bitsize==4: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_4bit_Sbox) 
                else: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_8bit_Sbox)  # Sbox layer                        
                rc = round_constants[i-1]
                c0 = rc & 0xF
                c1 = rc >> 4
                c2 = 0x2     
                self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [c0,None,None,None, c1,None,None,None, c2], code_if_unrolled=["RC[i] & 0xF", "RC[i] >> 4", "0X2"], constants_if_unrolled=f"RC={round_constants}")  # Constant layer            
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 2, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(8)])  # AddRoundKey layer   
                self.states["STATE"].PermutationLayer("SR", i, 3, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                self.states["STATE"].MatrixLayer("MC", i, 4, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer
                

# The AES block cipher
class AES_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
        
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=10 if version[1]==128 else 12 if version[1]==192 else 14 if version[1]==256  else None
        nbr_rounds += 1
        full_rounds=11 if version[1]==128 else 13 if version[1]==192 else 15 if version[1]==256  else None
        if model_type==0: 
            if k_bitsize==128:
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (7, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==192: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (9, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==256: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (13, int(16*k_bitsize / p_bitsize), 8, 8),  (1, 16, 0, 8) 
        self.p_bitsize, self.k_bitsize = version[0], version[1]
        
        perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        Rcon = [0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000]
        if k_bitsize==128: k_nbr_rounds, k_perm = nbr_rounds, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,13,14,15,12]
        elif k_bitsize==192: k_nbr_rounds, k_perm = nbr_rounds,  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,21,22,23,20]
        elif k_bitsize==256: k_nbr_rounds, k_perm = int(nbr_rounds/2) if nbr_rounds % 2 == 0 else int(nbr_rounds/2)+1,  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,29,30,31,28]
        nk = int(k_bitsize/32)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        # create constraints
        if model_type==0:
             # subkeys extraction
            if k_bitsize==128:
                for i in range(1,nbr_rounds+1): 
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[i][0])
            elif k_bitsize==192: 
                if k_nbr_rounds == 2: 
                    self.rounds_python_code_if_unrolled = {"SUBKEYS": [[1, "if i%3 == 0:"], [2, "elif i%3 == 1:"]]}
                    self.rounds_c_code_if_unrolled["SUBKEYS"] = [[1, "if (i % 3 == 0)"+"{"], [2, "else if (i % 3 == 1)"+"{"]]
                if k_nbr_rounds >= 3: 
                    self.rounds_python_code_if_unrolled = {"SUBKEYS": [[1, "if i%3 == 0:"], [2, "elif i%3 == 1:"], [3, "elif i%3 == 2:"]]}
                    self.rounds_c_code_if_unrolled["SUBKEYS"] = [[1, "if (i % 3 == 0)"+"{"], [2, "if (i % 3 == 1)"+"{"], [3, "if (i % 3 == 2)"+"{"]]
                for i in range(1, nbr_rounds+1):
                    if i%3 == 1: 
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[i][0][0:16])
                    elif i%3 == 2:
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[i][0][16:24]+self.states["KEY_STATE"].vars[i][0][0:8])
                    elif i%3 == 0:
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[i][0][8:24])
            elif k_bitsize==256: 
                if k_nbr_rounds >= 2: 
                    self.rounds_python_code_if_unrolled = {"SUBKEYS": [[1, "if i%2 == 0:"], [2, "elif i%2 == 1:"]]}
                    self.rounds_c_code_if_unrolled["SUBKEYS"] = [[1, "if (i % 2 == 0)"+"{"], [2, "if (i % 2 == 1)"+"{"]]
                for i in range(1, nbr_rounds+1):
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [j for j in range(16)], self.states["KEY_STATE"].vars[int((i+1)/2)][0] if i%2 != 0 else self.states["KEY_STATE"].vars[int((i+1)/2)][0][16:32])
                
            # key schedule    
            if k_bitsize==128: 
                self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, f"if i < {k_nbr_rounds-1}:"]]
                self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, f"if (i < {k_nbr_rounds-1})"+"{"]]
                for i in range(1,k_nbr_rounds): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff], code_if_unrolled=[f"((RC[i]>>{24-8*a})&0xff)" for a in range(4)], constants_if_unrolled=f"RC={Rcon}")  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"] TO DO
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
            elif k_bitsize==192: 
                count_rc = 0
                if k_nbr_rounds == 2: 
                    self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, f"if i < {k_nbr_rounds-1} and i % 3 == 0:"], [2, f"if i < {k_nbr_rounds-1} and i % 3 == 1:"]]
                    self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, f"if (i < {k_nbr_rounds-1} && i % 3 == 0)"+"{"], [2, f"if (i < {k_nbr_rounds-1} && i % 3 == 1)"+"{"]]
                if k_nbr_rounds >= 3: 
                    self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, "if i % 3 == 0:"], [2, f"if i < {k_nbr_rounds-1} and i % 3 == 1:"], [3, f"if i < {k_nbr_rounds-1} and i % 3 == 2:"]]
                    self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, f"if (i < {k_nbr_rounds-1} && i % 3 == 0)"+"{"], [2, f"if (i < {k_nbr_rounds-1} && i % 3 == 1)"+"{"], [3, f"if (i < {k_nbr_rounds-1} && i % 3 == 2)"+"{"]]
                for i in range(1,k_nbr_rounds): 
                    if i % 3 == 1: 
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                        self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                        self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[Rcon[count_rc]>>24&0xff, Rcon[count_rc]>>16&0xff, Rcon[count_rc]>>8&0xff, Rcon[count_rc]&0xff], code_if_unrolled=[f"(RC[2*(i//3)]>>{24-8*a})&0xff" for a in range(4)], constants_if_unrolled=f"RC={Rcon}")  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"] TO DO
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 5)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 6)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 7)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 8)     # Identity layer 
                        count_rc += 1
                    elif i % 3 == 2:  
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 0, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 1, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 2, op.bitwiseXOR, [[j, j+4] for j in range(12,16)],  [j for j in range(16,20)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 4)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 5)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 6)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 7)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 8)     # Identity layer 
                    elif i % 3 == 0:
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                        self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                        self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[Rcon[count_rc]>>24&0xff, Rcon[count_rc]>>16&0xff, Rcon[count_rc]>>8&0xff, Rcon[count_rc]&0xff], code_if_unrolled=[f"(RC[2 * (i//3)+1]>>{24-8*a})&0xff" for a in range(4)], constants_if_unrolled=f"RC={Rcon}")  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"] TO DO
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 7, op.bitwiseXOR, [[j, j+4] for j in range(12,16)],  [j for j in range(16,20)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 8, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                        count_rc += 1
                        
            elif k_bitsize==256:
                if k_nbr_rounds >= 2: 
                    self.rounds_python_code_if_unrolled["KEY_STATE"] = [[1, "if i % 2 == 1:"]]
                    self.rounds_c_code_if_unrolled["KEY_STATE"] = [[1, "if (i % 2 == 1)"+"{"]]
                for i in range(1,k_nbr_rounds): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for _ in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for _ in range(4*nk)]+[Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff], code_if_unrolled=[f"(RC[i//2]>>{24-8*a})&0xff" for a in range(4)], constants_if_unrolled=f"RC={Rcon}")  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"], TO DO
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0, 1, 2, 3]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                    if i == (k_nbr_rounds - 1) and nbr_rounds%2 == 1:
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 7)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 8)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 9)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 10)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 11)     # Identity layer 
                        self.states["KEY_STATE"].AddIdentityLayer("ID", i, 12)     # Identity layer 
                    else:
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 7, [i for i in range(36)]+[12,13,14,15]) # Permutation layer
                        self.states["KEY_STATE"].SboxLayer("K_SB", i, 8, op.AES_Sbox, mask=([0 for i in range(36)] + [1, 1, 1, 1])) # Sbox layer   
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 9, op.bitwiseXOR, [[j, j+20] for j in range(16, 20)],  [j for j in range(16, 20)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 10, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 11, op.bitwiseXOR, [[j, j+4] for j in range(20,24)],  [j for j in range(24,28)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 12, op.bitwiseXOR, [[j, j+4] for j in range(24,28)],  [j for j in range(28,32)]) # XOR layer 
                    
            # Internal permutation
            self.states["STATE"].AddRoundKeyLayer("ARK", 1, 0, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
            self.states["STATE"].AddIdentityLayer("ID", 1, 1)     # Identity layer 
            self.states["STATE"].AddIdentityLayer("ID", 1, 2)     # Identity layer 
            self.states["STATE"].AddIdentityLayer("ID", 1, 3)     # Identity layer 
            for i in range(2,nbr_rounds+1):   
                if i < full_rounds:  
                    self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                    self.states["STATE"].PermutationLayer("SR", i, 1, perm_s) # Shiftrows layer
                    self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]], "0x1B")  #Mixcolumns layer
                    self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
                elif i == full_rounds:
                    self.states["STATE"].SboxLayer("SB", nbr_rounds, 0, op.AES_Sbox) # Sbox layer   
                    self.states["STATE"].PermutationLayer("SR", nbr_rounds, 1, perm_s) # Shiftrows layer
                    self.states["STATE"].AddRoundKeyLayer("ARK", nbr_rounds, 2, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer            
                    self.states["STATE"].AddIdentityLayer("ID", nbr_rounds, 3)     # Identity layer 
            if nbr_rounds >= 2: 
                self.rounds_python_code_if_unrolled["STATE"] = [[1, "if i == 0:"]]
                self.rounds_c_code_if_unrolled["STATE"] = [[1, "if (i == 0)"+"{"]]
                if nbr_rounds < full_rounds: 
                    self.rounds_python_code_if_unrolled["STATE"].append([2, f"elif 1 <= i <= {nbr_rounds-1}:"])
                    self.rounds_c_code_if_unrolled["STATE"].append([2, f"else if (i <= {nbr_rounds-1})"+"{"])
                elif nbr_rounds == full_rounds:  
                    self.rounds_python_code_if_unrolled["STATE"].append([2, f"elif 1 <= i <= {nbr_rounds-2}:"])
                    self.rounds_python_code_if_unrolled["STATE"].append([nbr_rounds, f"elif i == {nbr_rounds-1}:"])
                    self.rounds_c_code_if_unrolled["STATE"].append([2, f"else if (i <= {nbr_rounds-2})"+"{"])
                    self.rounds_c_code_if_unrolled["STATE"].append([nbr_rounds, f"else if (i == {nbr_rounds-1})"+"{"])
 
               
# The Speck block cipher        
class Speck_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
                
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=22 if (version[0],version[1])==(32,64) else 22 if (version[0],version[1])==(48,72) else 23 if (version[0],version[1])==(48,96)  else 26 if (version[0],version[1])==(64,96)  else 27 if (version[0],version[1])==(64,128)  else 28 if (version[0],version[1])==(96,96) else 29 if (version[0],version[1])==(96,144) else 32 if (version[0],version[1])==(128,128) else 33 if (version[0],version[1])==(128,192) else 34 if (version[0],version[1])==(128,256) else None
        if model_type==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 2, 0, p_bitsize>>1),  (6, int(2*k_bitsize / p_bitsize), 0, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        if version[0]==32: rotr, rotl = 7, 2
        else: rotr, rotl = 8, 3
        if k_bitsize==p_bitsize: perm, left_k_index, right_k_index = [0,1], 0, 1
        elif k_bitsize==1.5*p_bitsize: perm, left_k_index, right_k_index = [1,0,2], 1, 2
        elif k_bitsize==2*p_bitsize: perm, left_k_index, right_k_index = [2,0,1,3], 2, 3
                
        # create constraints
        if model_type==0:         

            for i in range(1,nbr_rounds+1):
                # subkeys extraction
                self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [right_k_index], self.states["KEY_STATE"].vars[i][0])

            for i in range(1,nbr_rounds):    
                # key schedule
                self.states["KEY_STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], [left_k_index]) # Rotation layer
                self.states["KEY_STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[left_k_index, right_k_index]], [left_k_index]) # Modular addition layer   
                self.states["KEY_STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], [right_k_index]) # Rotation layer 
                self.states["KEY_STATE"].AddConstantLayer("C", i, 3, "xor", [(i-1) if e==left_k_index else None for e in range(self.states["KEY_STATE"].nbr_words)], code_if_unrolled="i")  # Constant layer
                self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[left_k_index, right_k_index]], [right_k_index]) # XOR layer 
                self.states["KEY_STATE"].PermutationLayer("SHIFT", i, 5, perm) # key schedule word shift
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], [0]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer  
                self.states["STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], [1]) # Rotation layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], [1,0]) # Addroundkey layer 
                self.states["STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[0,1]], [1]) # XOR layer
                

# The Simon block cipher 
class Simon_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
                
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(32,64) else 36 if (version[0],version[1])==(48,72) else 36 if (version[0],version[1])==(48,96)  else 42 if (version[0],version[1])==(64,96)  else 44 if (version[0],version[1])==(64,128)  else 52 if (version[0],version[1])==(96,96) else 54 if (version[0],version[1])==(96,144) else 68 if (version[0],version[1])==(128,128) else 69 if (version[0],version[1])==(128,192) else 72 if (version[0],version[1])==(128,256) else None
        if model_type==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (8, 2, 2, p_bitsize>>1),  (6, int(2*k_bitsize/p_bitsize), 2, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        if k_nbr_words == 4: k_nbr_layers += 1
        k_nbr_rounds = max(1, nbr_rounds - k_nbr_words + 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        # Z Arrays (stored bit reversed for easier usage)
        z0 = 0b01100111000011010100100010111110110011100001101010010001011111
        z1 = 0b01011010000110010011111011100010101101000011001001111101110001
        z2 = 0b11001101101001111110001000010100011001001011000000111011110101
        z3 = 0b11110000101100111001010001001000000111101001100011010111011011
        z4 = 0b11110111001001010011000011101000000100011011010110011110001011
        round_constant = (2 ** (p_bitsize >> 1) - 1) ^ 3
        z=z0 if (version[0],version[1])==(32,64) else z0 if (version[0],version[1])==(48,72) else z1 if (version[0],version[1])==(48,96)  else z2 if (version[0],version[1])==(64,96)  else z3 if (version[0],version[1])==(64,128)  else z2 if (version[0],version[1])==(96,96) else z3 if (version[0],version[1])==(96,144) else z2 if (version[0],version[1])==(128,128) else z3 if (version[0],version[1])==(128,192) else z4 if (version[0],version[1])==(128,256) else None
        
        # create constraints
        if model_type==0:
            
            for i in range(1,nbr_rounds+1):    
                # subkeys extraction
                if i <= k_nbr_words:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [(k_nbr_words-i%k_nbr_words)%k_nbr_words], self.states["KEY_STATE"].vars[1][0])
                else:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [0], self.states["KEY_STATE"].vars[i-k_nbr_words+1][0])
                                
            for i in range(1,k_nbr_rounds): 
                # key schedule
                if k_nbr_words == 2 or k_nbr_words == 3:
                    self.states["KEY_STATE"].RotationLayer("ROT1", i, 0, ['r', 3], [0], [k_nbr_words]) # Rotation layer
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words-1]) # XOR layer 
                    self.states["KEY_STATE"].RotationLayer("ROT2", i, 2, ['r', 1], [k_nbr_words], [k_nbr_words]) # Rotation layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words]) # XOR layer 
                    self.states["KEY_STATE"].AddConstantLayer("C", i, 4, "xor", [round_constant ^ ((z >> ((i-1) % 62)) & 1) if e==k_nbr_words else None for e in range((self.states["KEY_STATE"].nbr_words+self.states["KEY_STATE"].nbr_temp_words))], code_if_unrolled=["RC ^ (z >> (i % 62)) & 1"], constants_if_unrolled=f"RC = {hex(round_constant)}\nz={bin(z)}")  # Constant layer
                    self.states["KEY_STATE"].PermutationLayer("PERM", i, 5, [k_nbr_words]+[i for i in range(k_nbr_words)]) # Shiftrows layer
                
                elif k_nbr_words == 4:
                    self.states["KEY_STATE"].RotationLayer("ROT1", i, 0, ['r', 3], [0], [4]) # Rotation layer
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[2, 4]], [4]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[3, 4]], [5]) # XOR layer 
                    self.states["KEY_STATE"].RotationLayer("ROT2", i, 3, ['r', 1], [4], [4]) # Rotation layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[4, 5]], [4]) # XOR layer 
                    self.states["KEY_STATE"].AddConstantLayer("C", i, 5, "xor", [round_constant ^ ((z >> ((i-1) % 62)) & 1) if e==4 else None for e in range((self.states["KEY_STATE"].nbr_words+self.states["KEY_STATE"].nbr_temp_words))], code_if_unrolled=["RC ^ (z >> (i % 62)) & 1"], constants_if_unrolled=f"RC = {hex(round_constant)}\nz={bin(z)}")  # Constant layer
                    self.states["KEY_STATE"].PermutationLayer("PERM", i, 6, [4,0,1,2]) # Shiftrows layer           
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], [0], [2]) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], [0], [3]) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], [0], [2]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 6, op.bitwiseXOR, self.states["SUBKEYS"], [0,1]) # Addroundkey layer 
                self.states["STATE"].PermutationLayer("PERM", i, 7, [1,0]) # Permutation layer


# The GIFT block cipher              
class GIFT_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):

        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=28 if p_bitsize==64 else 40 if p_bitsize==128 else None
        if model_type==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, p_bitsize, 0, 1),  (1, k_bitsize, 0, 1),  (1, int(p_bitsize/2), 0, 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        const = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
        # perm_64 = [0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3, 4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7, 8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11, 12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15]
        # perm_128 = [0, 33, 66, 99, 96,  1, 34, 67, 64, 97,  2, 35, 32, 65, 98,  3, 4, 37, 70,103,100,  5, 38, 71, 68,101,  6, 39, 36, 69, 102, 7, 8, 41, 74,107,104,  9, 42, 75, 72,105, 10, 43, 40, 73,106, 11, 12, 45, 78,111,108, 13, 46, 79, 76,109, 14, 47, 44, 77,110, 15, 16, 49, 82,115,112, 17, 50, 83, 80,113, 18, 51, 48, 81,114, 19, 20, 53, 86,119,116, 21, 54, 87, 84,117, 22, 55, 52, 85,118, 23, 24, 57, 90,123,120, 25, 58, 91, 88,121, 26, 59, 56, 89,122, 27, 28, 61, 94,127,124, 29, 62, 95, 92,125, 30, 63, 60, 93,126, 31]
        if p_bitsize == 64: perm = [12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63]
        elif p_bitsize == 128: perm = [12, 1, 6, 11, 28, 17, 22, 27, 44, 33, 38, 43, 60, 49, 54, 59, 76, 65, 70, 75, 92, 81, 86, 91, 108, 97, 102, 107, 124, 113, 118, 123, 8, 13, 2, 7, 24, 29, 18, 23, 40, 45, 34, 39, 56, 61, 50, 55, 72, 77, 66, 71, 88, 93, 82, 87, 104, 109, 98, 103, 120, 125, 114, 119, 4, 9, 14, 3, 20, 25, 30, 19, 36, 41, 46, 35, 52, 57, 62, 51, 68, 73, 78, 67, 84, 89, 94, 83, 100, 105, 110, 99, 116, 121, 126, 115, 0, 5, 10, 15, 16, 21, 26, 31, 32, 37, 42, 47, 48, 53, 58, 63, 64, 69, 74, 79, 80, 85, 90, 95, 96, 101, 106, 111, 112, 117, 122, 127]
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):    
                # subkeys extraction
                if p_bitsize == 64:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [item for pair in zip(range(96, 112), range(112, 128)) for item in pair], self.states["KEY_STATE"].vars[i][0])
                elif p_bitsize == 128:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [item for pair in zip(range(32, 64), range(96, 128)) for item in pair], self.states["KEY_STATE"].vars[i][0])

            for i in range(1,nbr_rounds):        
                # key schedule
                self.states["KEY_STATE"].PermutationLayer("PERM", i, 0, [110,111]+[i for i in range(96,110)]+[i for i in range(116,128)]+[i for i in range(112,116)]+[i for i in range(96)]) # key schedule 
        
            for i in range(1,nbr_rounds+1):              
                self.states["STATE"].SboxLayer("SB", i, 0, op.GIFT_Sbox,index=[list(range(i, i + 4)) for i in range(0, s_nbr_words, 4)])  # Sbox layer            
                self.states["STATE"].PermutationLayer("P", i, 1, perm) # Permutation layer
                if p_bitsize == 64:
                    self.states["STATE"].AddConstantLayer("C", i, 2, "xor", [1]+[None]*39+[(const[i-1]>>5)&1]+[None]*3+[(const[i-1]>>4)&1]+[None]*3+[(const[i-1]>>3)&1]+[None]*3+[(const[i-1]>>2)&1]+[None]*3 +[(const[i-1]>>1)&1]+[None]*3+[(const[i-1])&1], code_if_unrolled=["1"] + [f"(RC[i]>>{5-a})&1" for a in range(6)], constants_if_unrolled=f"RC={const}")# Constant layer            
                    self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], [0,0,1,1]*16) # Addroundkey layer 
                elif p_bitsize == 128:
                    self.states["STATE"].AddConstantLayer("C", i, 2, "xor", [1]+[None]*103+[(const[i-1]>>5)&1]+[None]*3+[(const[i-1]>>4)&1]+[None]*3+[(const[i-1]>>3)&1]+[None]*3+[(const[i-1]>>2)&1]+[None]*3 +[(const[i-1]>>1)&1]+[None]*3+[(const[i-1])&1], code_if_unrolled=["1"] + [f"(RC[i]>>{5-a})&1" for a in range(6)], constants_if_unrolled=f"RC={const}")# Constant layer            
                    self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], [0,1,1,0]*32) # Addroundkey layer 

