from abc import ABC
import variables.variables as var
import operators.operators as op
from operators.matrix import Matrix

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

    # apply a layer "name" of Rotation, at the round "crt_round", at the layer "crt_layer". Each rot is a list of rotation executions, each execution is composed of three elements plus an optional fourth: [direction, amount, index_in, (index_out)]. A rotation execution will take the word of the state located at position "index_in", apply the rotation direction "direction" and amount "amount" and place it in state located at position "index_out" (if defined, "index_in" otherwise). The state words receiving no rotation are applied identity.
    def RotationLayer(self, name, crt_round, crt_layer, rot):
        if type(rot[0]) is not list: rot = [rot]
        identity_indexes = list(range(self.nbr_words + self.nbr_temp_words))
        for r in rot:
            in_index, out_index = r[2], r[2] if len(r)==3 else r[3]
            self.constraints[crt_round][crt_layer].append(op.Rot([self.vars[crt_round][crt_layer][in_index]], [self.vars[crt_round][crt_layer+1][out_index]], r[0], r[1], ID=generateID(name,crt_round,crt_layer,in_index)))
            if out_index in identity_indexes: identity_indexes.remove(out_index)

        for j in identity_indexes:
            self.constraints[crt_round][crt_layer].append(op.Equal([self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]], ID=generateID(name,crt_round,crt_layer,j)))

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
        flat_index_out = [idx for sub in index_out for idx in (sub if isinstance(sub, list) else [sub])]
        for j in range(self.nbr_words + self.nbr_temp_words):
            if j not in flat_index_out:
                in_var, out_var = [self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]]
                self.constraints[crt_round][crt_layer].append(op.Equal(in_var, out_var, ID=generateID(name,crt_round,crt_layer,j)))
            else:
                if isinstance(index_out[0], int):
                    in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[index_out.index(j)]]
                    out_vars = [self.vars[crt_round][crt_layer+1][j]]
                    if mat: self.constraints[crt_round][crt_layer].append(my_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer,j), mat=mat))
                    else: self.constraints[crt_round][crt_layer].append(my_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer,j)))       
                elif isinstance(index_out[0], list):
                    for id, sub_index in enumerate(index_out):
                        if j == sub_index[0]:
                            in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[id]]    
                            out_vars = [self.vars[crt_round][crt_layer + 1][i] for i in sub_index]
                            self.constraints[crt_round][crt_layer].append(my_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer,j)))
        
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
            self.constraints[crt_round][crt_layer].append(Matrix(name, [self.vars[crt_round][crt_layer][x] for x in indexes], [self.vars[crt_round][crt_layer+1][x] for x in indexes], mat = mat, polynomial = polynomial, ID=generateID(name,crt_round,crt_layer,j)) )
       
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
        self.test_vectors = []
        
        
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
          
  