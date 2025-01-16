from abc import ABC, abstractmethod
import os, os.path
from numpy import linspace
import variables as var
import operators as op


import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


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
    def SboxLayer(self, name, crt_round, crt_layer, sbox_operator, mask = None):
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
    def RotationLayer(self, name, crt_round, crt_layer, rot, index, index_out=None):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if index_out==None:
                if j==index: self.constraints[crt_round][crt_layer].append(op.Rot([in_var], [out_var], rot[0], rot[1], ID=generateID(name,crt_round,crt_layer,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            else:
                if j==index_out: 
                    in_var = self.vars[crt_round][crt_layer][index]
                    self.constraints[crt_round][crt_layer].append(op.Rot([in_var], [out_var], rot[0], rot[1], ID=generateID(name,crt_round,crt_layer,j)))
                else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            
    # apply a layer "name" of a simple identity at the round "crt_round", at the layer "crt_layer". 
    def AddIndentityLayer(self, name, crt_round, crt_layer):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
        
    # apply a layer "name" of a Constant addition, at the round "crt_round", at the layer "crt_layer", with the adding "add_type" and the constant value "constant". 
    def AddConstantLayer(self, name, crt_round, crt_layer, add_type, constant, code_if_unrolled=None):
        if len(constant)<(self.nbr_words + self.nbr_temp_words): constant = constant + [0]*(self.nbr_words + self.nbr_temp_words - len(constant))
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if constant[j]!=None: self.constraints[crt_round][crt_layer].append(op.ConstantAdd([in_var], [out_var], constant[j], add_type, code_if_unrolled=code_if_unrolled, ID=generateID(name,crt_round,crt_layer,j)))  
            else: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
    
    # apply a layer "name" of a single operator "my_operator" with input indexes "index_in" and output indexes "index_out", at the round "crt_round", at the layer "crt_layer". The other output indexes are just being applied identity
    def SingleOperatorLayer(self, name, crt_round, crt_layer, my_operator, index_in, index_out):
        for j in range(self.nbr_words + self.nbr_temp_words):
            in_var, out_var = self.vars[crt_round][crt_layer][j], self.vars[crt_round][crt_layer+1][j]
            if j not in index_out: self.constraints[crt_round][crt_layer].append(op.Equal([in_var], [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            else: 
                in_vars = [self.vars[crt_round][crt_layer][k] for k in index_in[index_out.index(j)]]
                self.constraints[crt_round][crt_layer].append(my_operator(in_vars, [out_var], ID=generateID(name,crt_round,crt_layer,j)))
            
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
        if mask is None: mask = [1]*self.nbr_words 
        cpt = 0
        for j in range(self.nbr_words):
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
    
    def get_var_def_c(self, word_bitsize):   # select the variable bitsize when generating C code
        if word_bitsize <= 8: return 'uint8_t'
        elif word_bitsize <= 32: return 'uint32_t'
        elif word_bitsize <= 64: return 'uint64_t'
        else: return 'uint128_t'
    
    def generate_code(self, filename, language = 'python', unroll = False):  # method that generates the code defining the primitive
        
        nbr_rounds = self.nbr_rounds
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as myfile:
            
            if language == 'c': myfile.write("#include <stdint.h>\n#include <stdio.h>\n\n")
            
            header_set = []
            nbr_layers_table = [self.states[s].nbr_layers for s in self.states]
            constraints_table = [self.states[s].constraints for s in self.states]
            for i in range(len(self.states)):
               for r in range(1,nbr_rounds+1):
                   for l in range(nbr_layers_table[i]+1):
                       for cons in constraints_table[i][r][l]:
                           if [cons.__class__.__name__, cons.model_version] not in header_set:
                               header_set.append([cons.__class__.__name__, cons.model_version]) 
                               if cons.generate_header(language) != None: 
                                   for line in cons.generate_header(language): myfile.write(line + '\n')
                                   myfile.write('\n')
                            
            if language == 'python':
                                               
                myfile.write("# Function implementing the " + self.name + " function\n")
                myfile.write("# Input:\n")
                for my_input in self.inputs: myfile.write("#   " + my_input + ": a list of " + str(len(self.inputs[my_input])) + " words of " + str(self.inputs[my_input][0].bitsize) + " bits \n")
                myfile.write("# Output:\n") 
                for my_output in self.outputs: myfile.write("#   " + my_output + ": a list of " + str(len(self.outputs[my_output])) + " words of " + str(self.outputs[my_output][0].bitsize) + " bits \n") 
                myfile.write("def " + self.name + "(" + ", ".join(self.inputs) + ", " + ", ".join(self.outputs) + "): \n")
                myfile.write("\n\t# Input \n")

                
                cpt, cptw = 0, 0
                my_input_name = sum([[i]*len(self.inputs[i]) for i in self.inputs], [])
                for s in self.states: 
                    for w in range(self.states[s].nbr_words): 
                        if unroll: myfile.write("\t" + self.states[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                        else: myfile.write("\t" + self.states[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                        cptw = cptw+1
                        if cptw>=len(self.inputs[my_input_name[cpt]]): cptw=0
                        cpt = cpt+1
                        if cpt>=sum(len(self.inputs[a]) for a in self.inputs): break
                    if cpt>=sum(len(self.inputs[a]) for a in self.inputs): break
                    myfile.write("\n")

                
                
                for s in self.states: 
                    if self.states[s].nbr_temp_words!=0: myfile.write("\t")
                    for w in range(self.states[s].nbr_words, self.states[s].nbr_words + self.states[s].nbr_temp_words): 
                        if unroll: myfile.write(self.states[s].vars[1][0][w].ID + " = ")
                        else: myfile.write(self.states[s].vars[1][0][w].remove_round_from_ID() + " = ")
                    if self.states[s].nbr_temp_words!=0: myfile.write("0 \n")    
               
                
                if unroll: 
                    for r in range(1,nbr_rounds+1):
                        myfile.write("\n\t# Round " + str(r) + "\n")
                        for s in self.states_implementation_order: 
                            for l in range(self.states[s].nbr_layers+1):                        
                                for cons in self.states[s].constraints[r][l]: 
                                    for line in cons.generate_model("python", unroll=True): myfile.write("\t" + line + "\n")      
                            myfile.write("\n")
                else: 
                    myfile.write("\n\t# Round function \n")
                    myfile.write("\tfor i in range(" + str(nbr_rounds) + "):\n")            
                    for s in self.states_implementation_order: 
                        for l in range(self.states[s].nbr_layers+1):                        
                            for cons in self.states[s].constraints[1][l]: 
                                for line in cons.generate_model("python"): myfile.write("\t\t" + line + "\n")      
                        myfile.write("\n")
                        
                myfile.write("\t# Output \n")
                cpt, cptw = 0, 0
                my_output_name = sum([[i]*len(self.outputs[i]) for i in self.outputs], [])
                for s in self.states: 
                    for w in range(self.states[s].nbr_words):
                        if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + self.states[s].vars[nbr_rounds][self.states[s].nbr_layers][w].ID + "\n")
                        else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + self.states[s].vars[nbr_rounds][self.states[s].nbr_layers][w].remove_round_from_ID() + "\n")
                        cptw = cptw+1
                        if cptw>=len(self.outputs[my_output_name[cpt]]): cptw=0
                        cpt = cpt+1
                        if cpt>=sum(len(self.outputs[a]) for a in self.outputs): break                           
                    if cpt>=sum(len(self.outputs[a]) for a in self.outputs): break
                    myfile.write("\n")
                
                myfile.write("\n# test implementation\n")
                for my_input in self.inputs: myfile.write(my_input + " = [" + ", ".join(["0x0"]*len(self.inputs[my_input])) + "] \n")
                for my_output in self.outputs: myfile.write(my_output + " = [" + ", ".join(["0x0"]*len(self.outputs[my_output])) + "] \n")
                myfile.write(self.name + "(" + ", ".join(self.inputs) + ", " + ", ".join(self.outputs) + ")\n")
                for my_input in self.inputs: myfile.write("print('" + my_input + "', str([hex(i) for i in " + my_input + "]))\n") 
                for my_output in self.outputs: myfile.write("print('" + my_output + "', str([hex(i) for i in " + my_output + "]))\n")         
               
              
            elif language == 'c':
                                     
                 myfile.write("// Function implementing the " + self.name + " function\n")
                 myfile.write("// Input:\n")
                 for my_input in self.inputs: myfile.write("//   " + my_input + ": an array of " + str(len(self.inputs[my_input])) + " words of " + str(self.inputs[my_input][0].bitsize) + " bits \n")
                 myfile.write("// Output:\n") 
                 for my_output in self.outputs: myfile.write("//   " + my_output + ": an array of " + str(len(self.outputs[my_output])) + " words of " + str(self.outputs[my_output][0].bitsize) + " bits \n") 
                 myfile.write("void " + self.name + "(" + ", ".join([self.get_var_def_c(self.inputs[i][0].bitsize) + "* " + i for i in self.inputs]) + ", " +  ", ".join([self.get_var_def_c(self.outputs[i][0].bitsize) + "* " + i for i in self.outputs]) + "){ \n")
                 
                 
                 for s in self.states_implementation_order: 
                     if unroll:  myfile.write("\t" + self.get_var_def_c(self.states[s].word_bitsize) + " " + ', '.join([self.states[s].vars[i][j][k].ID for i in range(nbr_rounds+1) for j in range(self.states[s].nbr_layers+1) for k in range(self.states[s].nbr_words + + self.states[s].nbr_temp_words)]  ) + ";\n")
                     else: myfile.write("\t" + self.get_var_def_c(self.states[s].word_bitsize) + " " + ', '.join([self.states[s].vars[1][j][k].remove_round_from_ID() for j in range(self.states[s].nbr_layers+1) for k in range(self.states[s].nbr_words + + self.states[s].nbr_temp_words)]  ) + ";\n")
                 myfile.write("\n\t// Input \n")
                 
                 cpt, cptw = 0, 0
                 my_input_name = sum([[i]*len(self.inputs[i]) for i in self.inputs], [])
                 for s in self.states: 
                     for w in range(self.states[s].nbr_words): 
                         if unroll: myfile.write("\t" + self.states[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                         else: myfile.write("\t" + self.states[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                         cptw = cptw+1
                         if cptw>=len(self.inputs[my_input_name[cpt]]): cptw=0
                         cpt = cpt+1
                         if cpt>=sum(len(self.inputs[a]) for a in self.inputs): break
                     if cpt>=sum(len(self.inputs[a]) for a in self.inputs): break
                     myfile.write("\n")
                       
                 if unroll:  
                     for r in range(1,nbr_rounds+1):
                         myfile.write("\n\t// Round " + str(r) + "\n")
                         for s in self.states_implementation_order:
                             for l in range(self.states[s].nbr_layers+1):
                                 for cons in self.states[s].constraints[r][l]: 
                                     for line in cons.generate_model('c', unroll=True): myfile.write("\t" + line + "\n")
                             myfile.write("\n")
                 else:
                     myfile.write("\n\t// Round function \n")
                     myfile.write("\tfor (int i=0; i<" + str(nbr_rounds) + "; i++) {\n")                     
                     for s in self.states_implementation_order:
                         for l in range(self.states[s].nbr_layers+1):
                             for cons in self.states[s].constraints[1][l]: 
                                 for line in cons.generate_model('c'): myfile.write("\t\t" + line + "\n")
                         myfile.write("\n")
                     myfile.write("\t}\n")     
                     
                 myfile.write("\n\t// Output \n")
                 cpt, cptw = 0, 0
                 my_output_name = sum([[i]*len(self.outputs[i]) for i in self.outputs], [])
                 for s in self.states: 
                     for w in range(self.states[s].nbr_words): 
                         if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + self.states[s].vars[nbr_rounds][self.states[s].nbr_layers][w].ID + "; \n")
                         else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + self.states[s].vars[nbr_rounds][self.states[s].nbr_layers][w].remove_round_from_ID() + "; \n")
                         cptw = cptw+1
                         if cptw>=len(self.outputs[my_output_name[cpt]]): cptw=0
                         cpt = cpt + 1
                         if cpt>=sum(len(self.outputs[a]) for a in self.outputs): break
                     if cpt>=sum(len(self.outputs[a]) for a in self.outputs): break
                     myfile.write("\n")
                         
                 myfile.write("} \n")
                 
                 myfile.write("\n// test implementation\n")
                 myfile.write("void main() {\n")
                 for my_input in self.inputs: myfile.write("\t" + self.get_var_def_c(self.inputs[my_input][0].bitsize) + " " + my_input + "[" + str(len(self.inputs[my_input])) + "] = {" + ", ".join(["0x0"]*len(self.inputs[my_input])) + "}; \n") 
                 for my_output in self.outputs: myfile.write("\t" + self.get_var_def_c(self.outputs[my_output][0].bitsize) + " " + my_output + "[" + str(len(self.outputs[my_output])) + "] = {" + ", ".join(["0x0"]*len(self.outputs[my_output])) + "}; \n") 
                 myfile.write("\t" + self.name + "(" + ", ".join(self.inputs) + ", " + ", ".join(self.outputs) + ");\n")
                 for my_input in self.inputs: 
                     myfile.write('\tprintf("' + my_input + ': ");') 
                     myfile.write('\tfor (int i=0;i<' + str(len(self.inputs[my_input])) + ';i++){ printf("0x%x, ", ' + my_input + '[i]);} printf("\\n");\n')                       
                 for my_output in self.outputs: 
                     myfile.write('\tprintf("' + my_output + ': ");') 
                     myfile.write('\tfor (int i=0;i<' + str(len(self.outputs[my_output])) + ';i++){ printf("0x%x, ", ' + my_output + '[i]);} printf("\\n");\n')     
                 myfile.write('}\n')
                
                

                    
    def generate_figure(self, filename):  # method that generates the figure describing the primitive       
        
        var_font_size = 2    # controls the font size of the variables
        op_font_size = 2     # controls the font size of the operators
        x_space_state = 20   # controls the x-axis space between the states
        x_space = 10         # controls the x-axis space between the variables/operators
        y_space_rounds = 5   # controls the y-axis space between the rounds
        y_space_layer = 5    # controls the y-axis space between the layers
        y_space_in_out = 25  # controls the y-axis space between the input/output and the rest
        elements_height = 8  # controls the height of the displayed elements
        var_length = 10      # controls the length of the displayed variables
        op_length = 15       # controls the length of the displayed operators
        var_colors = ['lightcyan','lightgreen','gray']  # controls the displayed colors for the variables
        op_colors = ['red', 'pink']                     # controls the displayed colors for the operators
        in_color = "orange"                             # controls the displayed colors for the input variables
        out_color = "orange"                            # controls the displayed colors for the output variables
        
        nbr_rounds = self.nbr_rounds
        nbr_layers_table = [self.states[s].nbr_layers for s in self.states_display_order]
        nbr_words_table = [self.states[s].nbr_words for s in self.states_display_order]
        constraints_table = [self.states[s].constraints for s in self.states_display_order]
        vars_table = [self.states[s].vars for s in self.states_display_order]
        vars_coord = []
        
        ax = plt.gca()
        
        ax.annotate(self.name, xy=(-op_length, 4*elements_height), fontsize=3*var_font_size, ha="center")
        
        # computation of the maximum x-lenghth for each state
        max_length = [0]*len(self.states)
        for i in range(len(self.states)):
           for r in range(1,nbr_rounds+1):
               for l in range(nbr_layers_table[i]+1):
                   temp = x_space*(len(vars_table[i][r][l])-1) + var_length*len(vars_table[i][r][l])
                   if temp > max_length[i]: max_length[i] = temp
                   temp = x_space*(len(constraints_table[i][r][l])-1) + op_length*len(constraints_table[i][r][l])
                   if temp > max_length[i]: max_length[i] = temp
         
        # display of the round delimitation lines
        for r in range(0,nbr_rounds): 
            y_coord = 2*elements_height-r*(y_space_rounds + 2*(max(nbr_layers_table)+1)*(y_space_layer + elements_height))
            plt.plot([-op_length-40, sum(max_length)+x_space_state*(len(self.states)-1)], [y_coord, y_coord], linewidth=0.1, linestyle='dashed', color='gray')
            ax.annotate("Round " + str(r+1), xy=(-op_length,y_coord-8), fontsize=2*var_font_size, ha="center")
                    
        # display the input variables
        y_coord = -y_space_in_out
        x_shift_input = 0
        cpt = 0
        for my_input in self.inputs:
            for w in range(len(self.inputs[my_input])):
                x_coord = x_shift_input + w*(x_space + op_length)
                ax.add_patch(Ellipse((x_coord,-y_coord), var_length, elements_height, facecolor=in_color))
                ax.annotate(self.inputs[my_input][w].ID, xy=(x_coord,-y_coord), fontsize=var_font_size, ha="center")
                vars_coord.append((self.inputs[my_input][w].ID,(x_coord,-y_coord)))
            x_shift_input = x_shift_input + x_space_state + max_length[cpt]
            cpt = cpt + 1
                    
        # diplay the variables  
        max_y_space = 0
        x_shift_state = 0
        for i in range(len(self.states)):
           y_shift_round = 0
           for r in range(1,nbr_rounds+1):
               for l in range(nbr_layers_table[i]+1):
                   
                   # display the variables
                   y_coord = y_shift_round + (y_space_layer + elements_height)*2*l 
                   for w in range(len(vars_table[i][r][l])): 
                       x_coord = x_shift_state + w*(x_space + op_length)
                       ax.add_patch(Ellipse((x_coord,-y_coord), var_length, elements_height, facecolor=adjust_lightness(var_colors[(i)%len(var_colors)], (0.8 if w >= nbr_words_table[i] else 1))))
                       ax.annotate(vars_table[i][r][l][w].ID, xy=(x_coord,-y_coord), fontsize=var_font_size, ha="center")
                       vars_coord.append((vars_table[i][r][l][w].ID,(x_coord,-y_coord)))
                       
               y_shift_round = y_shift_round + y_space_rounds + 2*(max(nbr_layers_table)+1)*(y_space_layer + elements_height)
               if y_shift_round > max_y_space: max_y_space = y_shift_round 
               
           x_shift_state = x_shift_state + x_space_state + max_length[i]
           
        # display the output variables
        y_coord = y_out_coord = y_space_in_out + max_y_space - y_space_rounds - 2*(y_space_layer + elements_height)
        x_shift_input = 0
        cpt = 0
        for my_output in self.outputs:
            for w in range(len(self.outputs[my_output])):
                x_coord = x_shift_input + w*(x_space + op_length)
                ax.add_patch(Ellipse((x_coord,-y_coord), var_length, elements_height, facecolor=out_color))
                ax.annotate(self.outputs[my_output][w].ID, xy=(x_coord,-y_coord), fontsize=var_font_size, ha="center")
                vars_coord.append((self.outputs[my_output][w].ID,(x_coord,-y_coord)))
            x_shift_input = x_shift_input + x_space_state + max_length[cpt]
            cpt = cpt + 1  
            
        # diplay the operators and links to the variables 
        vars_coord = dict(vars_coord)
        x_shift_state = 0
        for i in range(len(self.states)):
           y_shift_round = 0
           for r in range(1,nbr_rounds+1):
               for l in range(nbr_layers_table[i]+1):
                       
                   # display the operators and the links with the variables
                   y_coord = y_shift_round + (y_space_layer + elements_height)*(2*l+1) 
                   factor = 1
                   if len(constraints_table[i][r][l])!=0 and len(constraints_table[i][r][l]) < len(vars_table[i][r][l]): factor =  len(vars_table[i][r][l])/len(constraints_table[i][r][l])
                   
                   for w in range(len(constraints_table[i][r][l])):
                       # display the operators boxes
                       x_coord = x_shift_state + factor*w*(x_space + op_length) - op_length/2
                       if constraints_table[i][r][l][w].__class__.__name__ != "Equal":
                           ax.add_patch(Rectangle((x_coord,-y_coord-elements_height/2), op_length, elements_height, facecolor=op_colors[(i)%len(op_colors)], label='Label'))
                           ax.annotate(constraints_table[i][r][l][w].ID, xy=(x_coord+op_length/2,-y_coord), fontsize=op_font_size, ha="center")
                           
                           # display the links with the variables
                           my_inputs = constraints_table[i][r][l][w].input_vars
                           my_outputs = constraints_table[i][r][l][w].output_vars
                           in_x_coord, in_y_coord = list(linspace(x_coord,x_coord+op_length,len(my_inputs)+2)[1:-1]), -y_coord+elements_height/2
                           out_x_coord, out_y_coord = list(linspace(x_coord,x_coord+op_length,len(my_outputs)+2)[1:-1]), -y_coord-elements_height/2
                           for j in range(len(my_inputs)):
                               (var_x_coord, var_y_coord) = vars_coord[my_inputs[j].ID]
                               ax.arrow(var_x_coord, var_y_coord, in_x_coord[j]-var_x_coord, in_y_coord-var_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                           for j in range(len(my_outputs)):
                               (var_x_coord, var_y_coord) = vars_coord[my_outputs[j].ID]
                               ax.arrow(out_x_coord[j], out_y_coord, var_x_coord-out_x_coord[j], var_y_coord-out_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                       else:
                           (var_in_x_coord, var_in_y_coord) = vars_coord[constraints_table[i][r][l][w].input_vars[0].ID]
                           (var_out_x_coord, var_out_y_coord) = vars_coord[constraints_table[i][r][l][w].output_vars[0].ID]
                           ax.arrow(var_in_x_coord, var_in_y_coord, var_out_x_coord-var_in_x_coord, var_out_y_coord-var_in_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                           
               y_shift_round = y_shift_round + y_space_rounds + 2*(max(nbr_layers_table)+1)*(y_space_layer + elements_height)
               
           x_shift_state = x_shift_state + x_space_state + max_length[i]
           
        # display the input and output links   
        for j in range(len(self.inputs_constraints)):
            (x_in, y_in) = vars_coord[self.inputs_constraints[j].input_vars[0].ID]
            (x_state_in_coord, y_state_in_coord) = vars_coord[self.inputs_constraints[j].output_vars[0].ID]
            ax.arrow(x_in, y_in, x_state_in_coord-x_in, y_state_in_coord-y_in, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
        for j in range(len(self.outputs_constraints)):
            (x_state_out_coord, y_state_out_coord) = vars_coord[self.outputs_constraints[j].input_vars[0].ID]
            (x_out, y_out) = vars_coord[self.outputs_constraints[j].output_vars[0].ID]
            ax.arrow(x_state_out_coord, y_state_out_coord, x_out-x_state_out_coord, y_out-y_state_out_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
        
            
        #ax.autoscale_view()
        #ax.autoscale(tight=True)
        ax.set_xlim(-op_length, x_shift_state)
        ax.set_ylim(-elements_height-y_out_coord,elements_height+y_space_in_out)
        ax.set_axis_off()
        ax.axes.set_aspect('equal')
        
        my_fig = plt.gcf()
        my_fig.set_size_inches(0.02*(x_shift_state+max(op_length,op_length)),0.02*(2*elements_height+y_space_in_out+y_out_coord))
        my_fig.savefig(filename, bbox_inches='tight')
        plt.show()
        
        
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
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):              
                if word_bitsize==4: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_4bit_Sbox) 
                else: self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_8bit_Sbox)  # Sbox layer            
                self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [0,0,0,0, 0,0,0,0, 2,0,0,0, 0,0,0,0])  # Constant layer            
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
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], 0) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer   
                self.states["STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], 1) # Rotation layer 
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
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], 0, index_out=2) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], 0, index_out=3) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], 0, index_out=2) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].PermutationLayer("PERM", i, 6, [1,0]) # Permutation layer
            
                
# The ASCON internal permutation - NOT READY             
class ASCON_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=12
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 5, 64, 0, 5
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if model_type==0: 
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].SboxLayer("SB", i, 0, op.ASCON_Sbox) # Sbox layer   
                pass #TODO
                

# The GIFT internal permutation - NOT READY              
class GIFT_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=28 if version==64 else 40 if version==128 else None
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 3, version, 0, 1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        const = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a, 0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):              
                self.states["STATE"].SboxLayer("SB", i, 0, op.GIFT_Sbox)  # Sbox layer            
                self.states["STATE"].PermutationLayer("P", i, 1, [0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3, 4, 21, 38, 55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7, 8, 25, 42, 59, 56, 9, 26, 43, 40, 57, 10, 27, 24, 41, 58, 11, 12, 29, 46, 63, 60, 13, 30, 47, 44, 61, 14, 31, 28, 45, 62, 15]) # Permutation layer
                self.states["STATE"].AddConstantLayer("C", i, 2, "xor", [0,0,0,const[i-1]&1, 0,0,0,(const[i-1]>>1)&1, 0,0,0,(const[i-1]>>2)&1, 0,0,0,(const[i-1]>>3)&1, 0,0,0,(const[i-1]>>4)&1, 0,0,0,(const[i-1]>>5)&1, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,1])  # Constant layer            
                

# The AES internal permutation  
class AES_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=10
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, 8
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds):             
                self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                self.states["STATE"].PermutationLayer("SR", i, 1, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]) # Shiftrows layer
                self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]], "0x1B")  #Mixcolumns layer
                self.states["STATE"].AddConstantLayer("AC", i, 3, "xor", [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14])  # Constant layer            
            
            self.states["STATE"].SboxLayer("SB", nbr_rounds, 0, op.AES_Sbox) # Sbox layer   
            self.states["STATE"].PermutationLayer("SR", nbr_rounds, 1, [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14]) # Shiftrows layer
            self.states["STATE"].AddConstantLayer("AC", nbr_rounds, 2, "xor", [0,1,2,3, 5,6,7,4, 10,11,8,9, 15,12,13,14])  # Constant layer            
            self.states["STATE"].AddIndentityLayer("ID", nbr_rounds, 3)     # Identity layer 



# ********************************************** BLOCK CIPHERS **********************************************
# Subclass that represents a block cipher object 
# A block cipher is composed of three states: a permutation (to update the cipher state), a key schedule (to update the key state), and a round-key computation (to compute the round key from the key state)

class Block_cipher(Primitive):
    def __init__(self, name, p_input, k_input, c_output, nbr_rounds, s_config, k_config, sk_config):
        super().__init__(name, {"plaintext":p_input, "key":k_input}, {"ciphertext":c_output})
        s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize = s_config[0], s_config[1], s_config[2], s_config[3]
        k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize = k_config[0], k_config[1], k_config[2], k_config[3]
        sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize = sk_config[0], sk_config[1], sk_config[2], sk_config[3]
        self.nbr_rounds = nbr_rounds
        self.states = {"STATE": State("STATE", 's', nbr_rounds, s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), "KEY_STATE": State("KEY_STATE", 'k', nbr_rounds, k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), "SUBKEYS": State("SUBKEYS", 'sk', nbr_rounds, sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize)}
        self.states_implementation_order = ["SUBKEYS", "KEY_STATE", "STATE"]
        self.states_display_order = ["STATE", "KEY_STATE", "SUBKEYS"]
        
        if (len(k_input)!=k_nbr_words) or (len(p_input)!=s_nbr_words): raise Exception("Block_cipher: the number of input plaintext/key words does not match the number of plaintext/key words in state") 
        
        if len(p_input)!=s_nbr_words: raise Exception("Block_cipher: the number of plaintext words does not match the number of words in state") 
        for i in range(len(p_input)): self.inputs_constraints.append(op.Equal([p_input[i]], [self.states["STATE"].vars[1][0][i]], ID='IN_LINK_P_'+str(i)))
        
        if len(k_input)!=k_nbr_words: raise Exception("Block_cipher: the number of key words does not match the number of words in state") 
        for i in range(len(k_input)): self.inputs_constraints.append(op.Equal([k_input[i]], [self.states["KEY_STATE"].vars[1][0][i]], ID='IN_LINK_K_'+str(i)))
            
        if len(c_output)!=s_nbr_words: raise Exception("Block_cipher: the number of ciphertext words does not match the number of words in state") 
        for i in range(len(c_output)): self.outputs_constraints.append(op.Equal([self.states["STATE"].vars[nbr_rounds][s_nbr_layers][i]], [c_output[i]], ID='OUT_LINK_C_'+str(i)))
          
       
# The Skinny block cipher - NOT READY        
class Skinny_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
        
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(64,64) else 36 if (version[0],version[1])==(64,128) else 40 if (version[0],version[1])==(64,192)  else 40 if (version[0],version[1])==(128,128)  else 48 if (version[0],version[1])==(128,256)  else 56 if (version[0],version[1])==(128,384) else None
        if model_type==0:  (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, int(p_bitsize/16)), (4, int(16*k_bitsize / p_bitsize), 0, int(p_bitsize/16)), (4, 8, 0, int(p_bitsize/16))
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
                
        # create constraints
        if model_type==0:
            for i in range(1,nbr_rounds+1):              
                self.states["STATE"].SboxLayer("SB", i, 0, op.Skinny_4bit_Sbox)  # Sbox layer            
                self.states["STATE"].AddConstantLayer("C", i, 1, "xor", [0,0,0,0, 0,0,0,0, 2,0,0,0, 0,0,0,0])  # Constant layer      
                self.states["STATE"].PermutationLayer("SR", i, 2, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                self.states["STATE"].MatrixLayer("MC", i, 3, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer
            
            pass # TODO key schedule and subkey
                

# The AES block cipher
class AES_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
        
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=10 if version[1]==128 else 12 if version[1]==192 else 14 if version[1]==256  else None
        nbr_rounds += 1
        if model_type==0: 
            if k_bitsize==128:
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (7, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==192: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (9, int(16*k_bitsize / p_bitsize), 4, 8),  (1, 16, 0, 8) 
            if k_bitsize==256: 
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, 8),  (13, int(16*k_bitsize / p_bitsize), 8, 8),  (1, 16, 0, 8) 
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        self.p_bitsize, self.k_bitsize = version[0], version[1]
        
        perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        Rcon = [0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000]
        if k_bitsize==128: k_nbr_rounds, k_perm = nbr_rounds-1, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,13,14,15,12]
        elif k_bitsize==192: k_nbr_rounds, k_perm = int((nbr_rounds)/1.5)-1 if nbr_rounds % 3 == 0 else int((nbr_rounds)/1.5),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,21,22,23,20]
        elif k_bitsize==256: k_nbr_rounds, k_perm = int(nbr_rounds/2)-1 if nbr_rounds % 2 == 0 else int(nbr_rounds/2),  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,29,30,31,28]
        nk = int(k_bitsize/32)
        self.k_nbr_rounds = k_nbr_rounds
        # create constraints
        if model_type==0:
             # subkeys extraction
            if k_bitsize==128:
                for i in range(1,nbr_rounds+1): 
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[i][0])
            elif k_bitsize==192: 
                # test key = [0x8e, 0x73, 0xb0, 0xf7, 0xda,0x0e, 0x64, 0x52, 0xc8, 0x10, 0xf3, 0x2b,0x80,0x90,0x79,0xe5, 0x62,0xf8,0xea,0xd2,0x52,0x2c,0x6b,0x7b] 
                for i in range(1, nbr_rounds+1):
                    if i%3 == 1: 
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[1+int(i/3)*2][0][0:16])
                    elif i%3 == 2:
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[1+int(i/3)*2][0][16:24]+self.states["KEY_STATE"].vars[2+int(i/3)*2][0][0:8])
                    elif i%3 == 0:
                        self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [i for i in range(16)], self.states["KEY_STATE"].vars[2+int((i-1)/3)*2][0][8:24])
            elif k_bitsize==256: 
                for i in range(1, nbr_rounds+1):
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [j for j in range(16)], self.states["KEY_STATE"].vars[int((i+1)/2)][0] if i%2 != 0 else self.states["KEY_STATE"].vars[int((i+1)/2)][0][16:32])
                
            # key schedule
            if k_bitsize==128: 
                # test key = [0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c]
                for i in range(1,k_nbr_rounds+1): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for i in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for i in range(4*nk)]+[Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff])  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"] TO DO
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
            elif k_bitsize==192: 
                for i in range(1,k_nbr_rounds+1): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for i in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for i in range(4*nk)]+[Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff])  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"] TO DO
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0,1,2,3]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 7, op.bitwiseXOR, [[j, j+4] for j in range(12,16)],  [j for j in range(16,20)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 8, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
            elif k_bitsize==256:
                for i in range(1,k_nbr_rounds+1): 
                    self.states["KEY_STATE"].PermutationLayer("K_P", i, 0, k_perm) # Permutation layer
                    self.states["KEY_STATE"].SboxLayer("K_SB", i, 1, op.AES_Sbox, mask=([0 for i in range(4*nk)] + [1, 1, 1, 1])) # Sbox layer   
                    self.states["KEY_STATE"].AddConstantLayer("K_C", i, 2, "xor", [None for i in range(4*nk)]+[Rcon[i-1]>>24&0xff, Rcon[i-1]>>16&0xff, Rcon[i-1]>>8&0xff, Rcon[i-1]&0xff]+[None for i in range(4)])  # Constant layer, code_if_unrolled=[None for i in range(4*nk)]+["(Rcon[i]>>24&0xff)", "(Rcon[i]>>16&0xff)", "(Rcon[i]>>8&0xff)", "(Rcon[i]&0xff)"], TO DO
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[0,4*nk], [1,4*nk+1], [2,4*nk+2], [3,4*nk+3]], [0, 1, 2, 3]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j, j+4] for j in range(4)],  [j for j in range(4,8)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 5, op.bitwiseXOR, [[j, j+4] for j in range(4,8)],  [j for j in range(8,12)]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 6, op.bitwiseXOR, [[j, j+4] for j in range(8,12)],  [j for j in range(12,16)]) # XOR layer 
                    if i == k_nbr_rounds and nbr_rounds%2 == 1:
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 7)     # Identity layer 
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 8)     # Identity layer 
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 9)     # Identity layer 
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 10)     # Identity layer 
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 11)     # Identity layer 
                        self.states["KEY_STATE"].AddIndentityLayer("ID", i, 12)     # Identity layer 
                    else:
                        self.states["KEY_STATE"].PermutationLayer("K_P", i, 7, [i for i in range(36)]+[12,13,14,15]) # Permutation layer
                        self.states["KEY_STATE"].SboxLayer("K_SB", i, 8, op.AES_Sbox, mask=([0 for i in range(36)] + [1, 1, 1, 1])) # Sbox layer   
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 9, op.bitwiseXOR, [[j, j+20] for j in range(16, 20)],  [j for j in range(16, 20)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 10, op.bitwiseXOR, [[j, j+4] for j in range(16,20)],  [j for j in range(20,24)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 11, op.bitwiseXOR, [[j, j+4] for j in range(20,24)],  [j for j in range(24,28)]) # XOR layer 
                        self.states["KEY_STATE"].SingleOperatorLayer("K_XOR", i, 12, op.bitwiseXOR, [[j, j+4] for j in range(24,28)],  [j for j in range(28,32)]) # XOR layer 
                        
            

            # Internal permutation
            self.states["STATE"].AddRoundKeyLayer("ARK", 1, 0, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
            self.states["STATE"].AddIndentityLayer("ID", 1, 1)     # Identity layer 
            self.states["STATE"].AddIndentityLayer("ID", 1, 2)     # Identity layer 
            self.states["STATE"].AddIndentityLayer("ID", 1, 3)     # Identity layer 
            for i in range(2,nbr_rounds):     
                self.states["STATE"].SboxLayer("SB", i, 0, op.AES_Sbox) # Sbox layer   
                self.states["STATE"].PermutationLayer("SR", i, 1, perm_s) # Shiftrows layer
                self.states["STATE"].MatrixLayer("MC", i, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]], "0x1B")  #Mixcolumns layer
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer   
            
            self.states["STATE"].SboxLayer("SB", nbr_rounds, 0, op.AES_Sbox) # Sbox layer   
            self.states["STATE"].PermutationLayer("SR", nbr_rounds, 1, perm_s) # Shiftrows layer
            self.states["STATE"].AddRoundKeyLayer("ARK", nbr_rounds, 2, op.bitwiseXOR, self.states["SUBKEYS"], mask=[1 for i in range(16)])  # AddRoundKey layer            
            self.states["STATE"].AddIndentityLayer("ID", nbr_rounds, 3)     # Identity layer 
 
               
# The Speck block cipher        
class Speck_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
                
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=22 if (version[0],version[1])==(32,64) else 22 if (version[0],version[1])==(48,72) else 23 if (version[0],version[1])==(48,96)  else 26 if (version[0],version[1])==(64,96)  else 27 if (version[0],version[1])==(64,128)  else 28 if (version[0],version[1])==(96,96) else 29 if (version[0],version[1])==(96,144) else 32 if (version[0],version[1])==(128,128) else 33 if (version[0],version[1])==(128,192) else 34 if (version[0],version[1])==(128,256) else None
        if model_type==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 2, 0, p_bitsize>>1),  (6, int(2*k_bitsize / p_bitsize), 0, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
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
                
                # key schedule
                self.states["KEY_STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], left_k_index) # Rotation layer
                self.states["KEY_STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[left_k_index, right_k_index]], [left_k_index]) # Modular addition layer   
                self.states["KEY_STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], right_k_index) # Rotation layer 
                self.states["KEY_STATE"].AddConstantLayer("C", i, 3, "xor", [(i-1) if e==left_k_index else None for e in range(self.states["KEY_STATE"].nbr_words)], code_if_unrolled="i")  # Constant layer
                self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[left_k_index, right_k_index]], [right_k_index]) # XOR layer 
                self.states["KEY_STATE"].PermutationLayer("SHIFT", i, 5, perm) # key schedule word shift
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['r', rotr], 0) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("ADD", i, 1, op.ModAdd, [[0,1]], [0]) # Modular addition layer  
                self.states["STATE"].RotationLayer("ROT2", i, 2, ['l', rotl], 1) # Rotation layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 3, op.bitwiseXOR, self.states["SUBKEYS"], [1,0]) # Addroundkey layer 
                self.states["STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[0,1]], [1]) # XOR layer
                

# The Simon block cipher - NOT READY        
class Simon_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, model_type=0):
                
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(32,64) else 36 if (version[0],version[1])==(48,72) else 36 if (version[0],version[1])==(48,96)  else 42 if (version[0],version[1])==(64,96)  else 44 if (version[0],version[1])==(64,128)  else 52 if (version[0],version[1])==(96,96) else 54 if (version[0],version[1])==(96,144) else 68 if (version[0],version[1])==(128,128) else 69 if (version[0],version[1])==(128,192) else 72 if (version[0],version[1])==(128,256) else None
        # if model_type==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (8, 2, 2, p_bitsize>>1),  (?, int(2*k_bitsize / p_bitsize), 0, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        # create constraints
        if model_type==0:
            
            for i in range(1,nbr_rounds+1):    
                # subkeys extraction
                self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [right_k_index], self.states["KEY_STATE"].vars[i][0])
                
                # key schedule
                
                # TODO
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], 0, index_out=2) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], 0, index_out=3) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], 0, index_out=2) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 6, op.bitwiseXOR, self.states["SUBKEYS"], [0,1]) # Addroundkey layer 
                self.states["STATE"].PermutationLayer("PERM", i, 7, [1,0]) # Permutation layer



class Rocca_AD_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, model_type=0):
        
        if nbr_rounds==None: nbr_rounds=20
        if model_type==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 128+32*nbr_rounds, 32, 0
        perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        

        if model_type==0:
            for r in range(1, nbr_rounds+1):
                self.states["STATE"].PermutationLayer("P", r, 0, [i for i in range(16*7,16*8)] + perm_s + [i for i in range(16,16*2)] + [perm_s[i]+16*2 for i in range(16)] + [i for i in range(16*3,16*4)] + [perm_s[i]+16*4 for i in range(16)] + [perm_s[i]+16*5 for i in range(16)] + [i for i in range(16*6, 16*7)] + [i for i in range(16*8,16*(8+2*nbr_rounds))] + [i for i in range(16)] + [i for i in range(16*4, 16*5)])
                self.states["STATE"].SboxLayer("SB", r, 1, op.AES_Sbox, mask=[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[1 for i in range(16)]) # Sbox layer 
                self.states["STATE"].MatrixLayer("MC", r, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0+16,1+16,2+16,3+16], [4+16,5+16,6+16,7+16], [8+16,9+16,10+16,11+16], [12+16,13+16,14+16,15+16],[0+16*3,1+16*3,2+16*3,3+16*3], [4+16*3,5+16*3,6+16*3,7+16*3], [8+16*3,9+16*3,10+16*3,11+16*3], [12+16*3,13+16*3,14+16*3,15+16*3],[0+16*5,1+16*5,2+16*5,3+16*5], [4+16*5,5+16*5,6+16*5,7+16*5], [8+16*5,9+16*5,10+16*5,11+16*5], [12+16*5,13+16*5,14+16*5,15+16*5], [0+16*6,1+16*6,2+16*6,3+16*6], [4+16*6,5+16*6,6+16*6,7+16*6], [8+16*6,9+16*6,10+16*6,11+16*6], [12+16*6,13+16*6,14+16*6,15+16*6]], "0x1B")  #Mixcolumns layer
                self.states["STATE"].SingleOperatorLayer("XOR", r, 3, op.bitwiseXOR, [[i, i+16*(8+2*(r-1))] for i in range(16)] + [[i+16, i] for i in range(16)] + [[i+16*2, i+16*7] for i in range(16)] + [[i+16*3, i+16*2] for i in range(16)] + [[i+16*4, i+16*(8+2*(r-1)+1)] for i in range(16)] + [[i+16*5, i+16*4]  for i in range(16)] + [[i+16*6, i+16*(8+2*nbr_rounds+1)] for i in range(16)] + [[i+16*7, i+16*(8+2*nbr_rounds)] for i in range(16)],[i for i in range(16*8)]) # XOR layer 
               
            