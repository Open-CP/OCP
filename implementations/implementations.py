import os, os.path
import sys
import subprocess
import ctypes
import numpy as np
import importlib

# function that selects the variable bitsize when generating C code
def get_var_def_c(word_bitsize):   
    if word_bitsize <= 8: return 'uint8_t'
    elif word_bitsize <= 32: return 'uint32_t'
    elif word_bitsize <= 64: return 'uint64_t'
    else: return 'uint128_t'

# function that generates the implementation of the primitive
def generate_implementation(my_prim, filename, language = 'python', unroll = False):  
    
    nbr_rounds = my_prim.nbr_rounds
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as myfile:
        
        if language == 'c': myfile.write("#include <stdint.h>\n#include <stdio.h>\n\n")
        
        header_set = []
        matrix_seen = rot_seen = False
        nbr_rounds_table = [my_prim.states[s].nbr_rounds for s in my_prim.states]
        nbr_layers_table = [my_prim.states[s].nbr_layers for s in my_prim.states]
        constraints_table = [my_prim.states[s].constraints for s in my_prim.states]
        for i in range(len(my_prim.states)):
           for r in range(1,nbr_rounds_table[i]+1):
               for l in range(nbr_layers_table[i]+1):
                   for cons in constraints_table[i][r][l]:
                       # generate the unique header for certain types of operators
                       if cons.__class__.__name__ == 'Matrix' and not matrix_seen: 
                          header = cons.generate_implementation_header_unique(language)
                          for line in header: myfile.write(line + '\n')
                          myfile.write('\n')
                          matrix_seen = True
                       elif cons.__class__.__name__ == 'Rot' and not rot_seen: 
                          header = cons.generate_implementation_header_unique(language)
                          for line in header: myfile.write(line + '\n')
                          myfile.write('\n')
                          rot_seen = True                                              

                      # generate the header     
                       header_ID = cons.get_header_ID()
                       if header_ID not in header_set:
                           header_set.append(header_ID) 
                           header = cons.generate_implementation_header(language)
                           if header != None: 
                               for line in header: myfile.write(line + '\n')
                               myfile.write('\n')
                        
        if language == 'python':
                                           
            myfile.write("# Function implementing the " + my_prim.name + " function\n")
            myfile.write("# Input:\n")
            for my_input in my_prim.inputs: myfile.write("#   " + my_input + ": a list of " + str(len(my_prim.inputs[my_input])) + " words of " + str(my_prim.inputs[my_input][0].bitsize) + " bits \n")
            myfile.write("# Output:\n") 
            for my_output in my_prim.outputs: myfile.write("#   " + my_output + ": a list of " + str(len(my_prim.outputs[my_output])) + " words of " + str(my_prim.outputs[my_output][0].bitsize) + " bits \n") 
            myfile.write("def " + my_prim.name + "(" + ", ".join(my_prim.inputs) + ", " + ", ".join(my_prim.outputs) + "): \n")
            myfile.write("\n\t# Input \n")

            
            cpt, cptw = 0, 0
            my_input_name = sum([[i]*len(my_prim.inputs[i]) for i in my_prim.inputs], [])
            for s in my_prim.states: 
                for w in range(my_prim.states[s].nbr_words): 
                    if unroll: myfile.write("\t" + my_prim.states[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                    else: myfile.write("\t" + my_prim.states[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                    cptw = cptw+1
                    if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                    cpt = cpt+1
                    if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                myfile.write("\n")

            
            
            for s in my_prim.states: 
                if my_prim.states[s].nbr_temp_words!=0: myfile.write("\t")
                for w in range(my_prim.states[s].nbr_words, my_prim.states[s].nbr_words + my_prim.states[s].nbr_temp_words): 
                    if unroll: myfile.write(my_prim.states[s].vars[1][0][w].ID + " = ")
                    else: myfile.write(my_prim.states[s].vars[1][0][w].remove_round_from_ID() + " = ")
                if my_prim.states[s].nbr_temp_words!=0: myfile.write("0 \n")    
           
            
            if unroll: 
                for r in range(1,max(nbr_rounds_table)+1):
                    myfile.write("\n\t# Round " + str(r) + "\n")
                    for s in my_prim.states_implementation_order: 
                        if r <= my_prim.states[s].nbr_rounds:
                            for l in range(my_prim.states[s].nbr_layers+1):                        
                                for cons in my_prim.states[s].constraints[r][l]: 
                                    for line in cons.generate_implementation("python", unroll=True): myfile.write("\t" + line + "\n")      
                            myfile.write("\n")
            else: 
                myfile.write("\n\t# Round function \n")
                myfile.write("\tfor i in range(" + str(nbr_rounds) + "):\n")  
                for s in my_prim.states_implementation_order: 
                    for l in range(my_prim.states[s].nbr_layers+1):                        
                        for cons in my_prim.states[s].constraints[1][l]: 
                            for line in cons.generate_implementation("python"): myfile.write("\t\t" + line + "\n")      
                    myfile.write("\n")
                    
            myfile.write("\t# Output \n")
            cpt, cptw = 0, 0
            my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
            for s in my_prim.states: 
                for w in range(my_prim.states[s].nbr_words):
                    if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].ID + "\n")
                    else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].remove_round_from_ID() + "\n")
                    cptw = cptw+1
                    if cptw>=len(my_prim.outputs[my_output_name[cpt]]): cptw=0
                    cpt = cpt+1
                    if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break                           
                if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                myfile.write("\n")
            
            myfile.write("\n# test implementation\n")
            for my_input in my_prim.inputs: myfile.write(my_input + " = [" + ", ".join(["0x0"]*len(my_prim.inputs[my_input])) + "] \n")
            for my_output in my_prim.outputs: myfile.write(my_output + " = [" + ", ".join(["0x0"]*len(my_prim.outputs[my_output])) + "] \n")
            myfile.write(my_prim.name + "(" + ", ".join(my_prim.inputs) + ", " + ", ".join(my_prim.outputs) + ")\n")
            for my_input in my_prim.inputs: myfile.write("print('" + my_input + "', str([hex(i) for i in " + my_input + "]))\n") 
            for my_output in my_prim.outputs: myfile.write("print('" + my_output + "', str([hex(i) for i in " + my_output + "]))\n")         
           
          
        elif language == 'c':
                                 
             myfile.write("// Function implementing the " + my_prim.name + " function\n")
             myfile.write("// Input:\n")
             for my_input in my_prim.inputs: myfile.write("//   " + my_input + ": an array of " + str(len(my_prim.inputs[my_input])) + " words of " + str(my_prim.inputs[my_input][0].bitsize) + " bits \n")
             myfile.write("// Output:\n") 
             for my_output in my_prim.outputs: myfile.write("//   " + my_output + ": an array of " + str(len(my_prim.outputs[my_output])) + " words of " + str(my_prim.outputs[my_output][0].bitsize) + " bits \n") 
             myfile.write("void " + my_prim.name + "(" + ", ".join([get_var_def_c(my_prim.inputs[i][0].bitsize) + "* " + i for i in my_prim.inputs]) + ", " +  ", ".join([get_var_def_c(my_prim.outputs[i][0].bitsize) + "* " + i for i in my_prim.outputs]) + "){ \n")
             
             
             for s in my_prim.states_implementation_order: 
                 if unroll:  myfile.write("\t" + get_var_def_c(my_prim.states[s].word_bitsize) + " " + ', '.join([my_prim.states[s].vars[i][j][k].ID for i in range(my_prim.states[s].nbr_rounds+1) for j in range(my_prim.states[s].nbr_layers+1) for k in range(my_prim.states[s].nbr_words + + my_prim.states[s].nbr_temp_words)]  ) + ";\n")
                 else: myfile.write("\t" + get_var_def_c(my_prim.states[s].word_bitsize) + " " + ', '.join([my_prim.states[s].vars[1][j][k].remove_round_from_ID() for j in range(my_prim.states[s].nbr_layers+1) for k in range(my_prim.states[s].nbr_words + + my_prim.states[s].nbr_temp_words)]  ) + ";\n")
             myfile.write("\n\t// Input \n")
             
             cpt, cptw = 0, 0
             my_input_name = sum([[i]*len(my_prim.inputs[i]) for i in my_prim.inputs], [])
             for s in my_prim.states: 
                 for w in range(my_prim.states[s].nbr_words): 
                     if unroll: myfile.write("\t" + my_prim.states[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                     else: myfile.write("\t" + my_prim.states[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                     cpt = cpt+1
                     if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 myfile.write("\n")
                   
             if unroll:  
                for r in range(1,max(nbr_rounds_table)+1):
                     myfile.write("\n\t// Round " + str(r) + "\n")
                     for s in my_prim.states_implementation_order:
                         if  r <= my_prim.states[s].nbr_rounds:
                            for l in range(my_prim.states[s].nbr_layers+1):
                                for cons in my_prim.states[s].constraints[r][l]: 
                                    for line in cons.generate_implementation('c', unroll=True): myfile.write("\t" + line + "\n")
                            myfile.write("\n")
             else:
                 myfile.write("\n\t// Round function \n")
                 myfile.write("\tfor (int i=0; i<" + str(nbr_rounds) + "; i++) {\n")                     
                 for s in my_prim.states_implementation_order:
                    for l in range(my_prim.states[s].nbr_layers+1):
                        for cons in my_prim.states[s].constraints[1][l]: 
                            for line in cons.generate_implementation('c'): myfile.write("\t\t" + line + "\n")
                    myfile.write("\n")
                 myfile.write("\t}\n")     
                 
             myfile.write("\n\t// Output \n")
             cpt, cptw = 0, 0
             my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
             for s in my_prim.states: 
                 for w in range(my_prim.states[s].nbr_words): 
                     if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].ID + "; \n")
                     else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].remove_round_from_ID() + "; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.outputs[my_output_name[cpt]]): cptw=0
                     cpt = cpt + 1
                     if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 myfile.write("\n")
                     
             myfile.write("} \n")
             
             myfile.write("\n// test implementation\n")
             myfile.write("int main() {\n")
             for my_input in my_prim.inputs: myfile.write("\t" + get_var_def_c(my_prim.inputs[my_input][0].bitsize) + " " + my_input + "[" + str(len(my_prim.inputs[my_input])) + "] = {" + ", ".join(["0x0"]*len(my_prim.inputs[my_input])) + "}; \n") 
             for my_output in my_prim.outputs: myfile.write("\t" + get_var_def_c(my_prim.outputs[my_output][0].bitsize) + " " + my_output + "[" + str(len(my_prim.outputs[my_output])) + "] = {" + ", ".join(["0x0"]*len(my_prim.outputs[my_output])) + "}; \n") 
             myfile.write("\t" + my_prim.name + "(" + ", ".join(my_prim.inputs) + ", " + ", ".join(my_prim.outputs) + ");\n")
             for my_input in my_prim.inputs: 
                 myfile.write('\tprintf("' + my_input + ': ");') 
                 if my_prim.inputs[my_input][0].bitsize <= 32: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.inputs[my_input])) + ';i++){ printf("0x%x, ", ' + my_input + '[i]);} printf("\\n");\n')                       
                 else: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.inputs[my_input])) + ';i++){ printf("0x%llx, ", ' + my_input + '[i]);} printf("\\n");\n')                    
             for my_output in my_prim.outputs: 
                 myfile.write('\tprintf("' + my_output + ': ");') 
                 if my_prim.inputs[my_input][0].bitsize <= 32: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.outputs[my_output])) + ';i++){ printf("0x%x, ", ' + my_output + '[i]);} printf("\\n");\n')     
                 else:
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.outputs[my_output])) + ';i++){ printf("0x%llx, ", ' + my_output + '[i]);} printf("\\n");\n')     
             myfile.write('}\n')


        elif language == 'verilog':
                                 
             myfile.write("// Function implementing the " + my_prim.name + " function\n")
             myfile.write("// Input:\n")
             for my_input in my_prim.inputs: myfile.write("//   " + my_input + ": an array of " + str(len(my_prim.inputs[my_input])) + " words of " + str(my_prim.inputs[my_input][0].bitsize) + " bits \n")
             myfile.write("// Output:\n") 
             for my_output in my_prim.outputs: myfile.write("//   " + my_output + ": an array of " + str(len(my_prim.outputs[my_output])) + " words of " + str(my_prim.outputs[my_output][0].bitsize) + " bits \n") 
             myfile.write("module " + my_prim.name + "(" + ", ".join([i for i in my_prim.inputs]) + ", " +  ", ".join([i for i in my_prim.outputs]) + "); \n")
             
             for s in my_prim.inputs:  myfile.write("\n\tinput[" + str(len(s)*my_prim.inputs[s][0].bitsize-1) + ":0] " + s + "; \n")
             for s in my_prim.outputs: myfile.write("\toutput[" + str(len(my_prim.outputs[s])*my_prim.outputs[s][0].bitsize-1) + ":0] " + s + "; \n")

             for s in my_prim.states_implementation_order: 
                 if unroll:  myfile.write("\tlogic [" + str(my_prim.states[s].word_bitsize-1) + ":0] " + ', '.join([my_prim.states[s].vars[i][j][k].ID for i in range(my_prim.states[s].nbr_rounds+1) for j in range(my_prim.states[s].nbr_layers+1) for k in range(my_prim.states[s].nbr_words + + my_prim.states[s].nbr_temp_words)]  ) + ";")
                 else: myfile.write("\tlogic [" + str(my_prim.states[s].word_bitsize-1) + ":0] " + ', '.join([my_prim.states[s].vars[1][j][k].remove_round_from_ID() for j in range(my_prim.states[s].nbr_layers+1) for k in range(my_prim.states[s].nbr_words + + my_prim.states[s].nbr_temp_words)]  ) + ";\n")
             myfile.write("\n\n\t// Input \n")
             
             cpt, cptw = 0, 0
             my_input_name = sum([[i]*len(my_prim.inputs[i]) for i in my_prim.inputs], [])
             for s in my_prim.states: 
                 for w in range(my_prim.states[s].nbr_words): 
                     if unroll: myfile.write("\tassign " + my_prim.states[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(my_prim.states[s].word_bitsize-1 + my_prim.states[s].word_bitsize*cptw) + ":" + str(my_prim.states[s].word_bitsize*cptw) + "]; \n")
                     else: myfile.write("\tassign " + my_prim.states[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(my_prim.states[s].word_bitsize-1 + my_prim.states[s].word_bitsize*cptw) + ":" + str(my_prim.states[s].word_bitsize*cptw) + "]; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                     cpt = cpt+1
                     if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 myfile.write("\n")
                   
             if unroll:  
                for r in range(1,max(nbr_rounds_table)+1):
                     myfile.write("\n\t// Round " + str(r) + "\n")
                     for s in my_prim.states_implementation_order:
                         if  r <= my_prim.states[s].nbr_rounds:
                            for l in range(my_prim.states[s].nbr_layers+1):
                                for cons in my_prim.states[s].constraints[r][l]: 
                                    for line in cons.generate_implementation('verilog', unroll=True): myfile.write("\t" + line + "\n")
                            myfile.write("\n")
             else:
                 myfile.write("\n\t// Round function \n")
                 myfile.write("\tfor (int i=0; i<" + str(nbr_rounds) + "; i++) {\n")                     
                 for s in my_prim.states_implementation_order:
                    for l in range(my_prim.states[s].nbr_layers+1):
                        for cons in my_prim.states[s].constraints[1][l]: 
                            for line in cons.generate_implementation('verilog'): myfile.write("\t\t" + line + "\n")
                    myfile.write("\n")
                 myfile.write("\t}\n")     
                 
             myfile.write("\n\t// Output \n")
             cpt, cptw = 0, 0
             my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
             for s in my_prim.states: 
                 for w in range(my_prim.states[s].nbr_words): 
                     if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(my_prim.states[s].word_bitsize-1 + my_prim.states[s].word_bitsize*cptw) + ":" + str(my_prim.states[s].word_bitsize*cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].ID + "; \n")
                     else: myfile.write("\t" + my_output_name[cpt] + "[" + str(my_prim.states[s].word_bitsize-1 + my_prim.states[s].word_bitsize*cptw) + ":" + str(my_prim.states[s].word_bitsize*cptw) + "] = " + my_prim.states[s].vars[nbr_rounds][my_prim.states[s].nbr_layers][w].remove_round_from_ID() + "; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.outputs[my_output_name[cpt]]): cptw=0
                     cpt = cpt + 1
                     if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 myfile.write("\n")
                     
             myfile.write("endmodule \n")
             
             myfile.write("\n// test implementation\n")
             myfile.write("int main() {\n")
             for my_input in my_prim.inputs: myfile.write("\t" + get_var_def_c(my_prim.inputs[my_input][0].bitsize) + " " + my_input + "[" + str(len(my_prim.inputs[my_input])) + "] = {" + ", ".join(["0x0"]*len(my_prim.inputs[my_input])) + "}; \n") 
             for my_output in my_prim.outputs: myfile.write("\t" + get_var_def_c(my_prim.outputs[my_output][0].bitsize) + " " + my_output + "[" + str(len(my_prim.outputs[my_output])) + "] = {" + ", ".join(["0x0"]*len(my_prim.outputs[my_output])) + "}; \n") 
             myfile.write("\t" + my_prim.name + "(" + ", ".join(my_prim.inputs) + ", " + ", ".join(my_prim.outputs) + ");\n")
             for my_input in my_prim.inputs: 
                 myfile.write('\tprintf("' + my_input + ': ");') 
                 if my_prim.inputs[my_input][0].bitsize <= 32: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.inputs[my_input])) + ';i++){ printf("0x%x, ", ' + my_input + '[i]);} printf("\\n");\n')                       
                 else: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.inputs[my_input])) + ';i++){ printf("0x%llx, ", ' + my_input + '[i]);} printf("\\n");\n')                    
             for my_output in my_prim.outputs: 
                 myfile.write('\tprintf("' + my_output + ': ");') 
                 if my_prim.inputs[my_input][0].bitsize <= 32: 
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.outputs[my_output])) + ';i++){ printf("0x%x, ", ' + my_output + '[i]);} printf("\\n");\n')     
                 else:
                    myfile.write('\tfor (int i=0;i<' + str(len(my_prim.outputs[my_output])) + ';i++){ printf("0x%llx, ", ' + my_output + '[i]);} printf("\\n");\n')     
             myfile.write('}\n')


def test_implementation_python(cipher, cipher_name, input, output):
    print(f"****************TEST PYTHON IMPLEMENTATION of {cipher_name}****************")
    try:
        imp_cipher = importlib.import_module(f"files.{cipher_name}")
        importlib.reload(imp_cipher)
        func = getattr(imp_cipher, f"{cipher.name}")
        result = [0 for _ in range(len(output))]

        func(*input, result)

        if result == output:
            print("Test passed.")
        else:
            print(f'!!!!!!!!!!!!!!!!!Wrong!!!!!!!!!!!!!!!!!\nresult = {[hex(i) for i in result]}\nexpected output = {output}')
            return False
    except ImportError:
        print(f"Implementation module files.{cipher_name} version cannot be loaded.\n")
    except AttributeError as e:
        print(f"Function {cipher.name} not found in module files.{cipher_name}: {e}\n")
    except Exception as e:
        print(f"Function {cipher.name}: {e}.\n")    


def test_implementation_c(cipher, cipher_name, input, output):
    print(f"****************TEST C IMPLEMENTATION of {cipher_name}****************")
    first_var = next(iter(cipher.inputs.values()))[0]
    if first_var.bitsize <= 8:
        dtype_np = np.uint8
        dtype_ct = ctypes.c_uint8
    elif first_var.bitsize <= 32:
        dtype_np = np.uint32
        dtype_ct = ctypes.c_uint32
    elif first_var.bitsize <= 64:
        dtype_np = np.uint64
        dtype_ct = ctypes.c_uint64
    else:
        dtype_np = np.uint128
        dtype_ct = ctypes.c_uint128

    args_np = [np.array(arg, dtype=dtype_np) for arg in input]
    result = np.zeros(len(output), dtype=dtype_np)
    output = np.array(output, dtype=dtype_np)

    compile_command = f"gcc files/{cipher_name}.c -o files/{cipher_name}.out"
    compile_process = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if compile_process.returncode != 0:
        print(f"[ERROR] Compilation failed for {cipher_name}.c")
        return False

    try:
        func = getattr(ctypes.CDLL(f"files/{cipher_name}.out"), cipher.name)
        func.argtypes = [ctypes.POINTER(dtype_ct)] * (len(args_np) + 1)
        func_args = [arr.ctypes.data_as(ctypes.POINTER(dtype_ct)) for arr in args_np]
        func_args.append(result.ctypes.data_as(ctypes.POINTER(dtype_ct)))

        func(*func_args)

        if np.array_equal(result, output):
            print("Test passed.")
        else:
            print(f'Wrong! result = {[hex(i) for i in result]}, expected = {[hex(i) for i in output]}')
            return False
    except Exception as e:
        print(f"Failed to load or execute the C function: {e}")
        return False