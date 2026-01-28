import os, os.path
import subprocess
import ctypes
import traceback
import numpy as np
import importlib
from contextlib import redirect_stdout
import shutil
from implementations.t_table.convertion import TTable_Conversion

# function to check if a C compiler is available
def is_c_compiler_available():
    """Check if gcc or any C compiler is available in the system."""
    # Check for common C compilers
    compilers = ['gcc', 'clang', 'cl', 'cc']
    for compiler in compilers:
        if shutil.which(compiler) is not None:
            return True, compiler
    return False, None

# function to check if a Verilog compiler/simulator is available
def is_verilog_compiler_available():
    """Check if iverilog or another Verilog compiler is available in the system."""
    # Check for common Verilog compilers/simulators
    compilers = ['iverilog', 'verilog', 'vlog']
    for compiler in compilers:
        if shutil.which(compiler) is not None:
            return True, compiler
    return False, None

# function to check if a Rust compiler is available
def is_rust_compiler_available():
    """Check if rustc compiler is available in the system."""
    if shutil.which('rustc') is not None:
        return True
    return False

# function that selects the variable bitsize when generating C code
def get_var_def_c(word_bitsize):   
    if word_bitsize <= 8: return 'uint8_t'
    elif word_bitsize <= 32: return 'uint32_t'
    elif word_bitsize <= 64: return 'uint64_t'
    else: return 'uint128_t'

# function that generates the implementation of the primitive
def generate_implementation(my_prim, filename, language = 'python', unroll = False, ttable=False):  
    
    nbr_rounds = my_prim.nbr_rounds
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as myfile:
        
        if language == 'c': myfile.write("#include <stdint.h>\n#include <stdio.h>\n\n")
        
        header_set = []
        matrix_seen = rot_seen = False
        nbr_rounds_table = [my_prim.functions[s].nbr_rounds for s in my_prim.functions]
        nbr_layers_table = [my_prim.functions[s].nbr_layers for s in my_prim.functions]
        constraints_table = [my_prim.functions[s].constraints for s in my_prim.functions]
        for i in range(len(my_prim.functions)):
           for r in range(1,nbr_rounds_table[i]+1):
               for l in range(nbr_layers_table[i]+1):
                   for cons in constraints_table[i][r][l]:
                       # generate the unique header for certain types of operators
                       if cons.__class__.__name__ in ['Matrix', 'AESround'] and not matrix_seen: 
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
        #initialization of ttable conversion object 
        if ttable and ("PERMUTATION" in my_prim.functions): obj = TTable_Conversion(my_prim.functions["PERMUTATION"])
        else:ttable=False 
        #generate headers
        if ttable: 
            for line in obj.generate_headers(language): myfile.write(line+'\n')
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
            for s in my_prim.functions: 
                for w in range(my_prim.functions[s].nbr_words): 
                    if unroll: myfile.write("\t" + my_prim.functions[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                    else: myfile.write("\t" + my_prim.functions[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "] \n")
                    cptw = cptw+1
                    if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                    cpt = cpt+1
                    if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                myfile.write("\n")
            myfile.write("\n")
            
            for s in my_prim.functions: 
                if my_prim.functions[s].nbr_temp_words!=0: myfile.write("\t")
                for w in range(my_prim.functions[s].nbr_words, my_prim.functions[s].nbr_words + my_prim.functions[s].nbr_temp_words): 
                    if unroll: myfile.write(my_prim.functions[s].vars[1][0][w].ID + " = ")
                    else: myfile.write(my_prim.functions[s].vars[1][0][w].remove_round_from_ID() + " = ")
                if my_prim.functions[s].nbr_temp_words!=0: myfile.write("0 \n")    
            
            
            if unroll: 
                for r in range(1,max(nbr_rounds_table)+1):
                    myfile.write("\t# Round " + str(r) + "\n")
                    for s in my_prim.functions_implementation_order: 
                        if r <= my_prim.functions[s].nbr_rounds:
                            for l in range(my_prim.functions[s].nbr_layers+1):                        
                                for c,cons in enumerate(my_prim.functions[s].constraints[r][l]):
                                    if s == "PERMUTATION" and ttable:
                                        c_flag = obj.states[r][l][c]
                                        if c_flag==0:
                                            for line in cons.generate_implementation("python", unroll=True): myfile.write("\t" + line + "\n")
                                        elif c_flag==1: continue 
                                        else: 
                                            myfile.write("\t" + obj.con_list[r][l][c] + "\n")
                                    else:
                                        for line in cons.generate_implementation("python", unroll=True): myfile.write("\t" + line + "\n")      
                            myfile.write("\n")
            else: 
                myfile.write("\t# Round function \n")
                myfile.write("\tfor i in range(" + str(nbr_rounds) + "):\n")  
                for s in my_prim.functions_implementation_order: 
                    for l in range(my_prim.functions[s].nbr_layers+1):                        
                        for cons in my_prim.functions[s].constraints[1][l]:
                            for line in cons.generate_implementation("python"): myfile.write("\t\t" + line + "\n")      
                    myfile.write("\n")
                    
            myfile.write("\t# Output \n")
            cpt, cptw = 0, 0
            my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
            for s in my_prim.functions: 
                for w in range(my_prim.functions[s].nbr_words):
                    if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].ID + "\n")
                    else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].remove_round_from_ID() + "\n")
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
             
             
             for s in my_prim.functions_implementation_order: 
                 if unroll:  myfile.write("\t" + get_var_def_c(my_prim.functions[s].word_bitsize) + " " + ', '.join([my_prim.functions[s].vars[i][j][k].ID for i in range(my_prim.functions[s].nbr_rounds+1) for j in range(my_prim.functions[s].nbr_layers+1) for k in range(my_prim.functions[s].nbr_words + + my_prim.functions[s].nbr_temp_words)]  ) + ";\n")
                 else: myfile.write("\t" + get_var_def_c(my_prim.functions[s].word_bitsize) + " " + ', '.join([my_prim.functions[s].vars[1][j][k].remove_round_from_ID() for j in range(my_prim.functions[s].nbr_layers+1) for k in range(my_prim.functions[s].nbr_words + + my_prim.functions[s].nbr_temp_words)]  ) + ";\n")
             myfile.write("\n\t// Input \n")
             
             cpt, cptw = 0, 0
             my_input_name = sum([[i]*len(my_prim.inputs[i]) for i in my_prim.inputs], [])
             for s in my_prim.functions: 
                 for w in range(my_prim.functions[s].nbr_words): 
                     if unroll: myfile.write("\t" + my_prim.functions[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                     else: myfile.write("\t" + my_prim.functions[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(cptw) + "]; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                     cpt = cpt+1
                     if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 myfile.write("\n")
             myfile.write("\n")

             if unroll:  
                for r in range(1,max(nbr_rounds_table)+1):
                     myfile.write("\t// Round " + str(r) + "\n")
                     for s in my_prim.functions_implementation_order:
                         if  r <= my_prim.functions[s].nbr_rounds:
                            for l in range(my_prim.functions[s].nbr_layers+1):
                                for cons in my_prim.functions[s].constraints[r][l]:
                                    for line in cons.generate_implementation('c', unroll=True): myfile.write("\t" + line + "\n")
                            myfile.write("\n")
             else:
                 myfile.write("\t// Round function \n")
                 myfile.write("\tfor (int i=0; i<" + str(nbr_rounds) + "; i++) {\n")                     
                 for s in my_prim.functions_implementation_order:
                    for l in range(my_prim.functions[s].nbr_layers+1):
                        for cons in my_prim.functions[s].constraints[1][l]: 
                            for line in cons.generate_implementation('c'): myfile.write("\t\t" + line + "\n")
                    myfile.write("\n")
                 myfile.write("\t}\n")     
                 
             myfile.write("\t// Output \n")
             cpt, cptw = 0, 0
             my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
             for s in my_prim.functions: 
                 for w in range(my_prim.functions[s].nbr_words): 
                     if unroll: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].ID + "; \n")
                     else: myfile.write("\t" + my_output_name[cpt] + "[" + str(cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].remove_round_from_ID() + "; \n")
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
             
             for s in my_prim.inputs:  myfile.write("\n\tinput[" + str(len(my_prim.inputs[s])*my_prim.inputs[s][0].bitsize-1) + ":0] " + s + "; \n")
             for s in my_prim.outputs: myfile.write("\toutput[" + str(len(my_prim.outputs[s])*my_prim.outputs[s][0].bitsize-1) + ":0] " + s + "; \n")

             for s in my_prim.functions_implementation_order: 
                 if unroll:  myfile.write("\tlogic [" + str(my_prim.functions[s].word_bitsize-1) + ":0] " + ', '.join([my_prim.functions[s].vars[i][j][k].ID for i in range(my_prim.functions[s].nbr_rounds+1) for j in range(my_prim.functions[s].nbr_layers+1) for k in range(my_prim.functions[s].nbr_words + + my_prim.functions[s].nbr_temp_words)]  ) + ";")
                 else: myfile.write("\tlogic [" + str(my_prim.functions[s].word_bitsize-1) + ":0] " + ', '.join([my_prim.functions[s].vars[1][j][k].remove_round_from_ID() for j in range(my_prim.functions[s].nbr_layers+1) for k in range(my_prim.functions[s].nbr_words + + my_prim.functions[s].nbr_temp_words)]  ) + ";\n")
             myfile.write("\n\n\t// Input \n")
             
             cpt, cptw = 0, 0
             my_input_name = sum([[i]*len(my_prim.inputs[i]) for i in my_prim.inputs], [])
             for s in my_prim.functions: 
                 for w in range(my_prim.functions[s].nbr_words): 
                     if unroll: myfile.write("\tassign " + my_prim.functions[s].vars[1][0][w].ID + " = " + my_input_name[cpt] + "[" + str(my_prim.functions[s].word_bitsize-1 + my_prim.functions[s].word_bitsize*cptw) + ":" + str(my_prim.functions[s].word_bitsize*cptw) + "]; \n")
                     else: myfile.write("\tassign " + my_prim.functions[s].vars[1][0][w].remove_round_from_ID() + " = " + my_input_name[cpt] + "[" + str(my_prim.functions[s].word_bitsize-1 + my_prim.functions[s].word_bitsize*cptw) + ":" + str(my_prim.functions[s].word_bitsize*cptw) + "]; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.inputs[my_input_name[cpt]]): cptw=0
                     cpt = cpt+1
                     if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 if cpt>=sum(len(my_prim.inputs[a]) for a in my_prim.inputs): break
                 myfile.write("\n")
             myfile.write("\n")
                  
             if unroll:  
                for r in range(1,max(nbr_rounds_table)+1):
                     myfile.write("\t// Round " + str(r) + "\n")
                     for s in my_prim.functions_implementation_order:
                         if  r <= my_prim.functions[s].nbr_rounds:
                            for l in range(my_prim.functions[s].nbr_layers+1):
                                for cons in my_prim.functions[s].constraints[r][l]: 
                                    for line in cons.generate_implementation('verilog', unroll=True): myfile.write("\t" + line + "\n")
                            myfile.write("\n")
             else:
                 myfile.write("\t// Round function \n")
                 myfile.write("\tfor (int i=0; i<" + str(nbr_rounds) + "; i++) {\n")                     
                 for s in my_prim.functions_implementation_order:
                    for l in range(my_prim.functions[s].nbr_layers+1):
                        for cons in my_prim.functions[s].constraints[1][l]: 
                            for line in cons.generate_implementation('verilog'): myfile.write("\t\t" + line + "\n")
                    myfile.write("\n")
                 myfile.write("\t}\n")     
                 
             myfile.write("\t// Output \n")
             cpt, cptw = 0, 0
             my_output_name = sum([[i]*len(my_prim.outputs[i]) for i in my_prim.outputs], [])
             for s in my_prim.functions: 
                 for w in range(my_prim.functions[s].nbr_words): 
                     if unroll: myfile.write("\tassign " + my_output_name[cpt] + "[" + str(my_prim.functions[s].word_bitsize-1 + my_prim.functions[s].word_bitsize*cptw) + ":" + str(my_prim.functions[s].word_bitsize*cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].ID + "; \n")
                     else: myfile.write("\tassign " + my_output_name[cpt] + "[" + str(my_prim.functions[s].word_bitsize-1 + my_prim.functions[s].word_bitsize*cptw) + ":" + str(my_prim.functions[s].word_bitsize*cptw) + "] = " + my_prim.functions[s].vars[nbr_rounds][my_prim.functions[s].nbr_layers][w].remove_round_from_ID() + "; \n")
                     cptw = cptw+1
                     if cptw>=len(my_prim.outputs[my_output_name[cpt]]): cptw=0
                     cpt = cpt + 1
                     if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 if cpt>=sum(len(my_prim.outputs[a]) for a in my_prim.outputs): break
                 myfile.write("\n")
                     
             myfile.write("endmodule \n")
             
             myfile.write("\n// test implementation\n")
             myfile.write("module test_implementation();")
             for s in my_prim.inputs:  myfile.write("\n\tlogic[" + str(len(my_prim.inputs[s])*my_prim.inputs[s][0].bitsize-1) + ":0] " + s + "; \n")
             for s in my_prim.outputs: myfile.write("\tlogic[" + str(len(my_prim.outputs[s])*my_prim.outputs[s][0].bitsize-1) + ":0] " + s + "; \n")
             myfile.write('\tinteger i;\n')
             
             myfile.write("\n\t// Enter here your input test values (0 by default)\n")
             for s in my_prim.inputs:
                for i in range(len(my_prim.inputs[s])):
                    myfile.write('\tinitial ' + s + '[' + str(my_prim.inputs[s][0].bitsize*(i+1)-1) + ':' + str(my_prim.inputs[s][0].bitsize*i) + '] = ' + str(my_prim.inputs[s][0].bitsize) + "'h0; \n")
             myfile.write("\n")

             myfile.write("\t" + my_prim.name + " UUT (" + ", ".join(my_prim.inputs) + ", " + ", ".join(my_prim.outputs) + ");\n")
             myfile.write("\tinitial begin\n \t#1;\n")
             for my_input in my_prim.inputs: 
                 myfile.write('\t$display("' + my_input + ': ");')
                 myfile.write('\tfor (i=0;i<' + str(len(my_prim.inputs[my_input])) + ';i=i+1) begin $display("0x%h, ", ' + my_input + '[' + str(my_prim.inputs[my_input][0].bitsize) + '*(i+1)-1-:' + str(my_prim.inputs[my_input][0].bitsize) + ']); end\n')                       
             for my_output in my_prim.outputs: 
                 myfile.write('\t$display("' + my_output + ': ");') 
                 myfile.write('\tfor (i=0;i<' + str(len(my_prim.outputs[my_output])) + ';i=i+1) begin $display("0x%h, ", ' + my_output + '[' + str(my_prim.outputs[my_output][0].bitsize) + '*(i+1)-1-:' + str(my_prim.outputs[my_output][0].bitsize) + ']); end \n')     
             myfile.write("\tend\n")
             myfile.write('endmodule\n')

        elif language == 'rust':
            pass  # To be implemented in the future


def test_implementation_python(cipher, cipher_name, input, output):
    print(f"\n****************TEST PYTHON IMPLEMENTATION of {cipher_name}****************")
    
    # Check if Python implementation file exists
    py_file = f"files/{cipher_name}.py"
    if not os.path.exists(py_file):
        print(f"[INFO] Python implementation file '{py_file}' not found. Skipping Python test.")
        print(f"       The implementation should be generated first using generate_implementation().")
        return None
    
    print("Test input = ", [hex(i2) for i1 in input for i2 in i1])
    print("Test output = ", [hex(i) for i in output])
    try:
        with open(os.devnull, "w") as f, redirect_stdout(f):
            imp_cipher = importlib.import_module(f"files.{cipher_name}")
            importlib.reload(imp_cipher)
        func = getattr(imp_cipher, f"{cipher.name}")
        result = [0 for _ in range(len(output))]

        func(*input, result)
        print("Test result = ", [hex(i) for i in result])

        if result == output:
            print("Test passed.")
            return True
        else:
            print(f'!!!!!!!!!!!!!!!!!Wrong!!!!!!!!!!!!!!!!!\nTest result is not equal to expected Test output')
            return False
    except ImportError as e:
        print(f"[ERROR] Implementation module files.{cipher_name} cannot be loaded: {e}\n")
        return False
    except AttributeError as e:
        print(f"[ERROR] Function {cipher.name} not found in module files.{cipher_name}: {e}\n")
        return False
    except Exception as e:
        print(f"[ERROR] Function {cipher.name} failed: {e}.\n")
        traceback.print_exc() 
        return False    


def test_implementation_c(cipher, cipher_name, input, output):
    print(f"\n****************TEST C IMPLEMENTATION of {cipher_name}****************")
    
    # Check if C compiler is available
    compiler_available, compiler = is_c_compiler_available()
    if not compiler_available:
        print("[INFO] There is no C compiler available. Skipping C compilation test.")
        print("       To run C tests, please install gcc, clang, or another C compiler.")
        return None
    
    # Check if C implementation file exists
    c_file = f"files/{cipher_name}.c"
    if not os.path.exists(c_file):
        print(f"[INFO] C implementation file '{c_file}' not found. Skipping C test.")
        print(f"       The implementation should be generated first using generate_implementation().")
        return None
    
    print("Test input = ", [hex(i2) for i1 in input for i2 in i1])
    print("Test output = ", [hex(i) for i in output])
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

    compile_command = f"{compiler} files/{cipher_name}.c -o files/{cipher_name}.out"
    compile_process = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if compile_process.returncode != 0:
        print(f"[ERROR] Compilation failed for {cipher_name}.c")
        print(f"        Compiler: {compiler}")
        print(f"        Error output: {compile_process.stderr.decode()}")
        return False

    try:
        func = getattr(ctypes.CDLL(f"files/{cipher_name}.out"), cipher.name)
        func.argtypes = [ctypes.POINTER(dtype_ct)] * (len(args_np) + 1)
        func_args = [arr.ctypes.data_as(ctypes.POINTER(dtype_ct)) for arr in args_np]
        func_args.append(result.ctypes.data_as(ctypes.POINTER(dtype_ct)))

        func(*func_args)
        print("Test result = ", [hex(i) for i in result])

        if np.array_equal(result, output):
            print("Test passed.")
            return True
        else:
            print(f'!!!!!!!!!!!!!!!!!Wrong!!!!!!!!!!!!!!!!!\nTest result is not equal to expected Test output')
            return False
    except Exception as e:
        print(f"Failed to load or execute the C function: {e}")
        return False

def test_implementation_verilog(cipher, cipher_name, input, output):
    print(f"\n****************TEST VERILOG IMPLEMENTATION of {cipher_name}****************")

    # Check if Verilog compiler is available
    compiler_available, compiler = is_verilog_compiler_available()
    if not compiler_available:
        print("[INFO] There is no Verilog compiler/simulator available. Skipping Verilog test.")
        print("       To run Verilog tests, please install iverilog or another Verilog simulator.")
        print("       Install iverilog: https://bleyer.org/icarus/ (Windows) or 'apt install iverilog' (Linux)")
        return None

    # Check if Verilog implementation file exists
    sv_file = f"files/{cipher_name}.sv"
    if not os.path.exists(sv_file):
        print(f"[INFO] Verilog implementation file '{sv_file}' not found. Skipping Verilog test.")
        print(f"       The implementation should be generated first using generate_implementation().")
        return None
    
    print("Test input = ", [hex(i2) for i1 in input for i2 in i1])
    print("Test output = ", [hex(i) for i in output])

    print(f"[INFO] Verilog compiler detected: {compiler}")
    print("       Use a verilog compiler/simulator to test the generated verilog implementation.")
    print(f"       Example: {compiler} -g2012 -o files/{cipher_name}.out files/{cipher_name}.sv")
    print(f"                .\\files/{cipher_name}.out")
    print("[INFO] Automated Verilog testing is not yet implemented in this function.")
    return None

def test_implementation_rust(cipher, cipher_name, input, output):
    print(f"\n****************TEST RUST IMPLEMENTATION of {cipher_name}****************")
    
    # Check if Rust compiler is available
    if not is_rust_compiler_available():
        print("[INFO] There is no Rust compiler (rustc) available. Skipping Rust compilation test.")
        print("       To run Rust tests, please install Rust from https://rustup.rs/")
        return None
    
    # Check if Rust implementation file exists
    rust_file = f"files/{cipher_name}.rs"
    if not os.path.exists(rust_file):
        print(f"[INFO] Rust implementation file '{rust_file}' not found. Skipping Rust test.")
        print(f"       The implementation should be generated first using generate_implementation().")
        return None
    
    print("Test input = ", [hex(i2) for i1 in input for i2 in i1])
    print("Test output = ", [hex(i) for i in output])
    
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

    compile_command = f"rustc --crate-type cdylib -o files/{cipher_name}.dll files/{cipher_name}.rs"
    compile_process = subprocess.run(compile_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if compile_process.returncode != 0:
        print(f"[ERROR] Compilation failed for {cipher_name}.rs")
        print(f"        Error output: {compile_process.stderr.decode()}")
        return False

    try:
        func = getattr(ctypes.CDLL(f"files/{cipher_name}.dll"), cipher.name)
        func.argtypes = [ctypes.POINTER(dtype_ct)] * (len(args_np) + 1)
        func_args = [arr.ctypes.data_as(ctypes.POINTER(dtype_ct)) for arr in args_np]
        func_args.append(result.ctypes.data_as(ctypes.POINTER(dtype_ct)))

        func(*func_args)
        print("Test result = ", [hex(i) for i in result])

        if np.array_equal(result, output):
            print("Test passed.")
            return True
        else:
            print(f'!!!!!!!!!!!!!!!!!Wrong!!!!!!!!!!!!!!!!!\nTest result is not equal to expected Test output')
            return False
    except Exception as e:
        print(f"Failed to load or execute the Rust function: {e}")
        return False
