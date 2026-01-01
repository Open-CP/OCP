from abc import ABC
import variables.variables as var
import operators.operators as op
from operators.matrix import Matrix, GF2Linear_Trans
from operators.boolean_operators import ConstantXOR
from operators.modular_operators import ConstantAdd

def generateID(name, round_nb, layer, position):
    return name + '_' + str(round_nb) + '_' + str(layer) + '_' + str(position)

# ********************* LAYERED_FUNCTION ********************* #
# Class that represents a layered function object, i.e. a collection of functions that will be updated through a certain number of rounds each composed of a certain number of layers
# This object will contain the list of variables representing the functions at each stage of the computation
# This object will contain the list of constraints linking the variables together

class Layered_Function:
    def __init__(self, name, label, nbr_rounds, nbr_layers, nbr_words, nbr_temp_words, word_bitsize):
        self.name = name                      # name of the function
        self.label = label                    # label for display when refering that function
        self.nbr_rounds = nbr_rounds          # number of rounds in that function
        self.nbr_layers = nbr_layers          # number of layers per round in that function
        self.nbr_words = nbr_words            # number of words in that function
        self.nbr_temp_words = nbr_temp_words  # number of temporary words in that function
        self.word_bitsize = word_bitsize      # number of bits per word in that function
        self.vars = []
        self.constraints = []

        # list of variables for that function (indexed with vars[r][l][n] where r is the round number, l the layer number, n the word number)
        self.vars = [[[] for i in range(nbr_layers+1)] for j in range(nbr_rounds+1)]

        # list of constraints for that function (indexed with constraints[r][l][n] where r is the round number, l the layer number, n the constraint number)
        self.constraints = [[[] for i in range(nbr_layers+1)] for j in range(nbr_rounds+1)]

        # create variables
        for i in range(0,nbr_rounds+1):
            for l in range(0,nbr_layers+1):
                self.vars[i][l] = [var.Variable(word_bitsize, ID = generateID('v' + label,i,l,j)) for j in range(nbr_words + nbr_temp_words)]

        # create initial constraints
        for i in range(0,nbr_rounds):
            self.constraints[i][nbr_layers] = [op.Equal([self.vars[i][nbr_layers][j]], [self.vars[i+1][0][j]], ID=generateID('LINK_EQ_' + label,i,nbr_layers+1,j)) for j in range(nbr_words + nbr_temp_words)]

    def display(self, representation='binary'):   # method that displays in details the function
        print("Name: " + str(self.name), " / nbr_words: " + str(self.nbr_words), " / word_bitsize: " + str(self.word_bitsize))
        print("Vars: [" + str([ len(self.vars[i]) for i in range(len(self.vars))])   + "]")
        print("Constraints: [" + str([ len(self.constraints[i]) for i in range(len(self.constraints))])  + "]")

    # apply a layer "name" of an Sbox, at the round "crt_round", at the layer "crt_layer", with the Sbox operator "sbox_operator". Only the positions where mask=1 will have the Sbox applied, the rest being just identity
    def SboxLayer(self, name, crt_round, crt_layer, sbox_operator, mask = None, index=None):
        """
        Apply a layer "name" of an Sbox, at the round "crt_round", at the layer "crt_layer", with the Sbox operator "sbox_operator".

        Parameters:
            name (str): Name of the layer.
            crt_round (int): Round number.
            crt_layer (int): Layer number.
            sbox_operator (callable): Operator to apply as the S-box.
            mask (list, optional): List indicating which positions to apply the S-box (1) or identity (0).
                - If mask is None, S-box is applied to all positions.
                - If mask is provided, S-box is applied where mask[i] = 1, identity where mask[i] = 0.
                Example: mask = [1, 0, 1], S-box is applied to positions 0 and 2, identity to position 1.
            index (list of lists, optional): Index mapping that specifies how input and output variables are grouped for S-box application.
                - If index is None, S-box is applied to each variable individually.
                - If index is provided, S-box is applied according to the specified grouping in index. This allows flexible grouping of variables for S-box operations.
                Example: index = [[0,1,2,3], [4,5,6,7]] apply S-box to variables at positions 0-3 and 4-7 respectively.

        Returns:
            None
        """
        if index is not None:
            bitsize = len(index[0])
            n_words = int((self.nbr_words+self.nbr_temp_words)/bitsize)
            if mask is None: mask = [1]*int(self.nbr_words/bitsize)
            if len(mask)<n_words: mask = mask + [0]*(n_words - len(mask))
            for j in range(n_words):
                if mask[j]==1:
                    in_vars, out_vars = [self.vars[crt_round][crt_layer][i] for i in index[j]], [self.vars[crt_round][crt_layer+1][i] for i in index[j]]
                    self.constraints[crt_round][crt_layer].append(sbox_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer+1,j)))
                else:
                    for i in range(bitsize):
                        in_var, out_var = self.vars[crt_round][crt_layer][j*bitsize+i], self.vars[crt_round][crt_layer+1][j*bitsize+i]
                        self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
        else:
            if mask is None: mask = [1]*self.nbr_words
            if len(mask)<(self.nbr_words + self.nbr_temp_words): mask = mask + [0]*(self.nbr_words + self.nbr_temp_words - len(mask))
            for j in range(self.nbr_words + self.nbr_temp_words):
                in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
                if mask[j]==1: self.constraints[crt_round][crt_layer].append(sbox_operator([in_var], [out_var], ID=generateID(name,crt_round,crt_layer+1,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))

    # apply a layer "name" of a Permutation, at the round "crt_round", at the layer "crt_layer", with the permutation "permutation".
    def PermutationLayer(self, name, crt_round, crt_layer, permutation):
        """
        Apply a layer "name" of a Permutation, at the round "crt_round", at the layer "crt_layer", with the permutation "permutation"

        Parameters:
            name (str): Name of the layer.
            crt_round (int): Round number.
            crt_layer (int): Layer number.
            permutation (list): List defining the permutation. Each element at index j indicates the position from which the value should be taken for position j in the output.

        Returns:
            None
        """
        if len(permutation)<(self.nbr_words + self.nbr_temp_words): permutation = permutation + [i for i in range(len(permutation), self.nbr_words + self.nbr_temp_words)]
        for j in range(len(permutation)):
            in_var, out_var = self.vars[crt_round][crt_layer][permutation[j]], self.vars[crt_round][crt_layer+1][j]
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], simple_connect=False, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))

    # apply a layer "name" of Rotation, at the round "crt_round", at the layer "crt_layer". Each rot is a list of rotation executions, each execution is composed of three elements plus an optional fourth: [direction, amount, index_in, (index_out)]. A rotation execution will take the word of the state located at position "index_in", apply the rotation direction "direction" and amount "amount" and place it in state located at position "index_out" (if defined, "index_in" otherwise). The state words receiving no rotation are applied identity.
    def RotationLayer(self, name, crt_round, crt_layer, rot):
        """
        Apply a layer "name" of Rotation, at the round "crt_round", at the layer "crt_layer".

        Parameters:
            name (str): Name of the layer.
            crt_round (int): Round number.
            crt_layer (int): Layer number.
            rot (list): A list of rotation executions, each execution is composed of three elements plus an optional fourth: [direction, amount, index_in, (index_out)]. A rotation execution will take the word of the function located at position "index_in", apply the rotation direction "direction" and amount "amount" and place it in function located at position "index_out" (if defined, "index_in" otherwise). The function words receiving no rotation are applied identity.
            Example: rot = [["l", 1, 2], ["r", 1, 2, 0]] will apply a left rotation of 1 to the word at index 2 and place it back at index 2, and a right rotation of 1 to the word at index 2 and place it at index 0. The other words (1,3,4...) will be applied identity.

        Returns:
            None
        """
        if type(rot[0]) is not list: rot = [rot]  # if only one rotation is given, transform it into a list of one rotation
        table = [None]*(self.nbr_words + self.nbr_temp_words) # prepare a table to identify which output indexes are rotated values and which are not
        for r in rot:
            index_in, out_index = r[2], r[2] if len(r)==3 else r[3]
            table[out_index] = (r[0], r[1], index_in, out_index)

        for j in range(self.nbr_words + self.nbr_temp_words): # apply the rotations and the identity where no rotation is applied
            if table[j] is not None:
                self.constraints[crt_round][crt_layer].append(op.Rot([self.vars[crt_round][crt_layer][table[j][2]]], [self.vars[crt_round][crt_layer+1][table[j][3]]], table[j][0], table[j][1], ID=generateID(name,crt_round,crt_layer+1,table[j][3])))
            else:
                self.constraints[crt_round][crt_layer].append(op.Equal([self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))

    # apply a layer "name" of a simple identity at the round "crt_round", at the layer "crt_layer".
    def AddIdentityLayer(self, name, crt_round, crt_layer):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], simple_connect=False, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))

    # apply a layer "name" of a Constant addition, at the round "crt_round", at the layer "crt_layer", with the adding "add_type" and the constant value "constant".
    def AddConstantLayer(self, name, crt_round, crt_layer, add_type, constant, constant_table, modulo=None):
        if len(constant)<(self.nbr_words + self.nbr_temp_words): constant = constant + [None]*(self.nbr_words + self.nbr_temp_words - len(constant))
        i = 0
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if constant[j]!=None:
                if add_type == 'xor':
                    self.constraints[crt_round][crt_layer].append(ConstantXOR([in_var], [out_var], constant_table, crt_round, i, ID=generateID(name,crt_round,crt_layer+1,j)))
                elif add_type == 'modadd':
                    self.constraints[crt_round][crt_layer].append(ConstantAdd([in_var], [out_var], constant_table, crt_round, i, modulo=modulo, ID=generateID(name,crt_round,crt_layer+1,j)))
                i += 1
            else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))

    # apply a layer "name" of a single operator "my_operator" with input indexes "index_in" and output indexes "index_out", at the round "crt_round", at the layer "crt_layer". The other output indexes are just being applied identity
    def SingleOperatorLayer(self, name, crt_round, crt_layer, my_operator, index_in, index_out):
        flat_index_out = [idx for sub in index_out for idx in (sub if isinstance(sub, list) else [sub])]
        for j in range(self.nbr_words + self.nbr_temp_words):
            if j not in flat_index_out:
                in_var, out_var = [self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]]
                self.constraints[crt_round][crt_layer].append(op.Equal(in_var, out_var, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
            else:
                if isinstance(index_out[0], int):
                    in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[index_out.index(j)]]
                    out_vars = [self.vars[crt_round][crt_layer+1][j]]
                    self.constraints[crt_round][crt_layer].append(my_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer+1,j)))
                elif isinstance(index_out[0], list):
                    for id, sub_index in enumerate(index_out):
                        if j == sub_index[0]:
                            in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[id]]
                            out_vars = [self.vars[crt_round][crt_layer + 1][i] for i in sub_index]
                            self.constraints[crt_round][crt_layer].append(my_operator(in_vars, out_vars, ID=generateID(name,crt_round,crt_layer+1,j)))

    # apply a layer "name" of a GF2Linear_Trans at the round "crt_round", at the layer "crt_layer"
    def GF2Linear_TransLayer(self, name, crt_round, crt_layer, index_in, index_out, mat, constants=None):
        flat_index_out = [idx for sub in index_out for idx in (sub if isinstance(sub, list) else [sub])]
        for j in range(self.nbr_words + self.nbr_temp_words):
            if j not in flat_index_out:
                in_var, out_var = [self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]]
                self.constraints[crt_round][crt_layer].append(op.Equal(in_var, out_var, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
            else:
                in_vars = [self.vars[crt_round][crt_layer][index_in[index_out.index(j)]]]
                out_vars = [self.vars[crt_round][crt_layer+1][j]]
                self.constraints[crt_round][crt_layer].append(GF2Linear_Trans(in_vars, out_vars, mat, ID=generateID(name,crt_round,crt_layer+1,j), constants=constants))
    #AES_TTABLE
    def TTableLayer(self, name, crt_round, crt_layer, ttable_operator, idxs):
        from math import sqrt
        n = int(sqrt(self.nbr_words))
        if  n*n != self.nbr_words: raise Exception("Layered Function: TTableLayer requires that tye nbr words is square rootable!")

        for r in range(n):
            tmp_in = []
            tmp_out = [] 
            for c in range(n):
                tmp_in.append(self.vars[crt_round][crt_layer][idxs[r*n + c]])
                #output is the transpose 
                tmp_out.append(self.vars[crt_round][crt_layer+1][r*n+c])
        #got to transpose one more time at the end 

            self.constraints[crt_round][crt_layer].append(ttable_operator(tmp_in, tmp_out, ID=generateID(name,crt_round,crt_layer+1,r)))

    # apply a layer "name" of a Matrix "mat" (only square matrix), at the round "crt_round", at the layer "crt_layer", operating in the field GF(2^"bitsize") with polynomial "polynomial"
    def MatrixLayer(self, name, crt_round, crt_layer, mat, indexes_list, polynomial = None):
        m = len(mat)
        for i in mat:
            if len(i)!=m: raise Exception("MatrixLayer: matrix shape is not square")
        flat_indexes = [x for sublist in indexes_list for x in sublist]
        for j in range(self.nbr_words + self.nbr_temp_words):
            if j not in flat_indexes:
                self.constraints[crt_round][crt_layer].append(op.Equal([self.vars[crt_round][crt_layer][j]], [self.vars[crt_round][crt_layer+1][j]], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
        for j, indexes in enumerate(indexes_list):
            if len(indexes)!=m: raise Exception("MatrixLayer: input vector does not match matrix size")
            self.constraints[crt_round][crt_layer].append(Matrix(name, [self.vars[crt_round][crt_layer][x] for x in indexes], [self.vars[crt_round][crt_layer+1][x] for x in indexes], mat = mat, polynomial = polynomial, ID=generateID(name,crt_round,crt_layer+1,j)) )

    # extract a subkey from the external variable, determined by "extraction_mask"
    def ExtractionLayer(self, name, crt_round, crt_layer, extraction_indexes, external_variable):
        for j, indexes in enumerate(extraction_indexes):
            in_var, out_var = external_variable[indexes], self.vars[crt_round][crt_layer+1][j] 
            if name=="TT_EX": print("HERE: ",op.Equal([in_var], [out_var], simple_connect=False, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], simple_connect=False, ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))
    # apply a layer "name" of an AddRoundKeyLayer addition, at the round "crt_round", at the layer "crt_layer", with the adding operator "my_operator". Only the positions where mask=1 will have the AddRoundKey applied, the rest being just identity  
    def AddRoundKeyLayer(self, name, crt_round, crt_layer, my_operator, sk_function, mask = None):
        if sum(mask)!=sk_function.nbr_words: raise Exception("AddRoundKeyLayer: subkey size does not match the mask")
        if len(mask)<(self.nbr_words + self.nbr_temp_words): mask += [0]*(self.nbr_words + self.nbr_temp_words - len(mask))
        cpt = 0
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if mask[j]==1:
                sk_var = sk_function.vars[crt_round][-1][cpt]
                self.constraints[crt_round][crt_layer].append(my_operator([in_var, sk_var], [out_var], ID=generateID(name,crt_round,crt_layer+1,j)))
                cpt = cpt + 1
            else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name + "_EQ",crt_round,crt_layer+1,j)))


# ********************* PRIMITIVES ********************* #
# Class that represents a primitive object, i.e. a cryptographic algorithm such as a permutation, a block cipher etc.
# This object makes the link between what is a specific cryptographic primitive and various "functions", "variables", "operators"
# These objects are the ones to be instantiated by the user for analysing a cipher

class Primitive(ABC):
    def __init__(self, name, inputs, outputs):
        self.name = name                # name of the primitive
        self.inputs = inputs            # list of the inputs of the primitive
        self.outputs = outputs          # list of the outputs of the primitive
        self.functions = []             # list of functions used by the primitive
        self.inputs_constraints = []    # constraints linking the primitive inputs to the functions input variables
        self.outputs_constraints = []   # constraints linking the primitive outputs to the functions output variables
        self.copy_constraints = []      # constraints for all the copy operators
        self.test_vectors = []

    # method that cleans the graph from dead-end variables linked only to Equal operators
    def clean_graph(self):
        changed = True
        while changed:
            changed = False
            for f in self.functions.values():       # for all functions of the primitive
                for r in range(f.nbr_rounds+1):       # for all the rounds
                    for l in range(f.nbr_layers+1):   # for all the layers
                        for v in f.vars[r][l]:      # for all variables in that function
                            # find dead-end variables in the graph
                            if len(v.connected_vars)==1 and v.connected_vars[0][1].__class__.__name__=="Equal":
                                v_temp=v
                                # follow the chain and remove the corresponding Equal operators
                                while len(v_temp.connected_vars)==1 and v_temp.connected_vars[0][1].__class__.__name__=="Equal":
                                    (new_v, new_op, direction) = v_temp.connected_vars[0]
                                    v_temp.connected_vars.pop(0)
                                    index = new_v.connected_vars.index((v_temp,new_op, "in" if direction=="out" else "out"))
                                    new_v.connected_vars.pop(index)
                                    new_op.is_ghost = True  # mark the Equal operator as ghost
                                    v_temp = new_v
                                    changed = True

        # remove ghost operators from the constraints lists
        for f in self.functions.values():       # for all functions of the primitive
            for r in range(f.nbr_rounds+1):       # for all the rounds
                for l in range(f.nbr_layers+1):   # for all the layers
                    for n in range(len(f.constraints[r][l])):
                        if f.constraints[r][l][n].is_ghost:
                            f.constraints[r][l][n] = op.NoneOperator(input_vars=f.constraints[r][l][n].input_vars, output_vars=f.constraints[r][l][n].output_vars, ID=generateID("NONE",r,l,n))  # replace the ghost operator by a NoneOperator

        for n in range(len(self.inputs_constraints)):
            if self.inputs_constraints[n].is_ghost:
                self.inputs_constraints[n] = op.NoneOperator(input_vars=self.inputs_constraints[n].input_vars, output_vars=self.inputs_constraints[n].output_vars, ID="NONE_INPUT_" + str(n))  # replace the ghost operator by a NoneOperator

        for n in range(len(self.outputs_constraints)):
            if self.outputs_constraints[n].is_ghost:
                self.outputs_constraints[n] = op.NoneOperator(input_vars=self.outputs_constraints[n].input_vars, output_vars=self.outputs_constraints[n].output_vars, ID="NONE_OUTPUT_" + str(n))  # replace the ghost operator by a NoneOperator

    # method that add the copy operators where needed in the graph
    def add_copy_operators(self, functions_list=None):
        if functions_list is None:
            functions_list = self.functions.values()
        for f in functions_list:
            for r in range(f.nbr_rounds+1):       # for all the rounds
                for l in range(f.nbr_layers+1):   # for all the layers
                    for v in f.vars[r][l]:      # for all variables in that function
                        # find variables that need copy operators
                        connected_vars_with_unique_operator = []
                        added_operators = []
                        for (vv,opop,direction) in v.connected_vars:
                            if direction=='in':  # we only consider the operators where v is an input variable
                                if opop not in added_operators:
                                    added_operators.append(opop)
                                    connected_vars_with_unique_operator.append((vv,opop,direction))

                        # if more than one unique operator is connected to that variable, then we need copy operators
                        if len(connected_vars_with_unique_operator)>1:
                            
                            #if there is a direct Equal operator, in connected_vars_with_unique_operator, put it on first position
                            for i in range(1,len(connected_vars_with_unique_operator)):
                                if connected_vars_with_unique_operator[i][1].__class__.__name__=="Equal":
                                    if connected_vars_with_unique_operator[i][1].simple_connect==True:
                                        connected_vars_with_unique_operator[0], connected_vars_with_unique_operator[i] = connected_vars_with_unique_operator[i], connected_vars_with_unique_operator[0]
                                    break

                            # create new variables and the copy operator
                            v_new = [var.Variable(v.bitsize, ID=v.ID + "_COPY_" + str(i)) for i in range(len(connected_vars_with_unique_operator))] 
                            op_new = op.CopyOperator([v], v_new, ID= "COPYOPERATOR_" + v.ID)
                            self.copy_constraints.append(op_new)      # save this new operator in the copy operator list
                            for i in range(len(connected_vars_with_unique_operator)):
                                v.copied_vars.append((v_new[i], connected_vars_with_unique_operator[i][1], op_new))     # save these new variables and operators
                                
                            # update the graph connections    
                            for i in range(len(connected_vars_with_unique_operator)):
                                (vv, opop, direction) = connected_vars_with_unique_operator[i]
                                for v_index in range(len(opop.input_vars)): # update the input of the operator with the new variable
                                    if opop.input_vars[v_index]==vv: opop.input_vars[v_index] = v_new[i]

                                ## remove vv from connected vars in v
                                index = vv.connected_vars.index((v, opop, "out"))
                                vv.connected_vars.pop(index)

                                ## remove v from connected vars in vv
                                index = v.connected_vars.index((vv, opop, "in"))
                                v.connected_vars.pop(index)

                                ## add v_new in connected vars of vv
                                vv.connected_vars.append((v_new[i], opop, "out"))

                                ## add vv in connected vars of v_new
                                v_new[i].connected_vars.append((vv, opop, "in"))


# ********************************************** FUNCTIONS **********************************************
# Subclass that represents a function object
# A function is composed of a single internal function

class Function(Primitive):
    def __init__(self, name, s_input, s_output, nbr_rounds, config):
        super().__init__(name, {"IN_":s_input}, {"OUT_":s_output})
        nbr_layers, nbr_words_input, nbr_words_output, nbr_temp_words, word_bitsize = config[0], config[1], config[2], config[3], config[4]
        self.nbr_rounds = nbr_rounds
        self.functions = {"FUNCTION": Layered_Function("FUNCTION", "", nbr_rounds, nbr_layers, max(nbr_words_input, nbr_words_output), nbr_temp_words, word_bitsize)}
        self.functions_implementation_order = ["FUNCTION"]
        self.functions_display_order = ["FUNCTION"]

        if len(s_input)!=nbr_words_input: raise Exception("Function: the number of input words does not match the number of input words in function")
        for i in range(len(s_input)): self.inputs_constraints.append(op.Equal([s_input[i]], [self.functions["FUNCTION"].vars[1][0][i]], ID='IN_LINK_EQ_'+str(i)))

        if len(s_output)!=nbr_words_output: raise Exception("Function: the number of output words does not match the number of output words in function")
        for i in range(len(s_output)): self.outputs_constraints.append(op.Equal([self.functions["FUNCTION"].vars[nbr_rounds][nbr_layers][i]], [s_output[i]], ID='OUT_LINK_EQ_'+str(i)))


# ********************************************** PERMUTATIONS **********************************************
# Subclass that represents a permutation object
# A permutation is composed of a single function

class Permutation(Primitive):
    def __init__(self, name, s_input, s_output, nbr_rounds, config):
        super().__init__(name, {"IN_":s_input}, {"OUT_":s_output})
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = config[0], config[1], config[2], config[3]
        self.nbr_rounds = nbr_rounds
        self.functions = {"PERMUTATION": Layered_Function("PERMUTATION", "", nbr_rounds, nbr_layers, nbr_words, nbr_temp_words, word_bitsize)}
        self.functions_implementation_order = ["PERMUTATION"]
        self.functions_display_order = ["PERMUTATION"]

        if len(s_input)!=nbr_words: raise Exception("Permutation: the number of input words does not match the number of words in function")
        for i in range(len(s_input)): self.inputs_constraints.append(op.Equal([s_input[i]], [self.functions["PERMUTATION"].vars[1][0][i]], ID='IN_LINK_EQ_'+str(i)))

        if len(s_output)!=nbr_words: raise Exception("Permutation: the number of output words does not match the number of words in function")
        for i in range(len(s_output)): self.outputs_constraints.append(op.Equal([self.functions["PERMUTATION"].vars[nbr_rounds][nbr_layers][i]], [s_output[i]], ID='OUT_LINK_EQ_'+str(i)))


# ********************************************** BLOCK CIPHERS **********************************************
# Subclass that represents a block cipher object
# A block cipher is composed of three functions: a permutation (to update the cipher function), a key schedule (to update the key schedule), and a round-key computation (to compute the round key from the key schedule)

class Block_cipher(Primitive):
    def __init__(self, name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, s_config, k_config, sk_config):
        super().__init__(name, {"plaintext":p_input, "key":k_input}, {"ciphertext":c_output})
        s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize = s_config[0], s_config[1], s_config[2], s_config[3]
        k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize = k_config[0], k_config[1], k_config[2], k_config[3]
        sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize = sk_config[0], sk_config[1], sk_config[2], sk_config[3]
        self.nbr_rounds = nbr_rounds
        self.functions = {"PERMUTATION": Layered_Function("PERMUTATION", 's', nbr_rounds, s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), "KEY_SCHEDULE": Layered_Function("KEY_SCHEDULE", 'k', k_nbr_rounds, k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), "SUBKEYS": Layered_Function("SUBKEYS", 'sk', nbr_rounds, sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize)}
        self.functions_implementation_order = ["SUBKEYS", "KEY_SCHEDULE", "PERMUTATION"]
        self.functions_display_order = ["PERMUTATION", "KEY_SCHEDULE", "SUBKEYS"]

        if (len(k_input)!=k_nbr_words) or (len(p_input)!=s_nbr_words): raise Exception("Block_cipher: the number of input plaintext/key words does not match the number of plaintext/key words in function")

        if len(p_input)!=s_nbr_words: raise Exception("Block_cipher: the number of plaintext words does not match the number of words in the permutation")
        for i in range(len(p_input)): self.inputs_constraints.append(op.Equal([p_input[i]], [self.functions["PERMUTATION"].vars[1][0][i]], ID='IN_LINK_P_EQ_'+str(i)))

        if len(k_input)!=k_nbr_words: raise Exception("Block_cipher: the number of key words does not match the number of words in the")
        for i in range(len(k_input)): self.inputs_constraints.append(op.Equal([k_input[i]], [self.functions["KEY_SCHEDULE"].vars[1][0][i]], ID='IN_LINK_K_EQ_'+str(i)))

        if len(c_output)!=s_nbr_words: raise Exception("Block_cipher: the number of ciphertext words does not match the number of words in the permutation")
        for i in range(len(c_output)): self.outputs_constraints.append(op.Equal([self.functions["PERMUTATION"].vars[nbr_rounds][s_nbr_layers][i]], [c_output[i]], ID='OUT_LINK_C_EQ_'+str(i)))


