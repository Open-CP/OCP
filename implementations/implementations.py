import os, os.path

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
        nbr_rounds_table = [my_prim.states[s].nbr_rounds for s in my_prim.states]
        nbr_layers_table = [my_prim.states[s].nbr_layers for s in my_prim.states]
        constraints_table = [my_prim.states[s].constraints for s in my_prim.states]
        for i in range(len(my_prim.states)):
           for r in range(1,nbr_rounds_table[i]+1):
               for l in range(nbr_layers_table[i]+1):
                   for cons in constraints_table[i][r][l]:
                       if [cons.__class__.__name__, cons.model_version] not in header_set:
                           header_set.append([cons.__class__.__name__, cons.model_version]) 
                           if cons.generate_implementation_header(language) != None: 
                               for line in cons.generate_implementation_header(language): myfile.write(line + '\n')
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
                        if  r <= my_prim.states[s].nbr_rounds:
                            for l in range(my_prim.states[s].nbr_layers+1):                        
                                for cons in my_prim.states[s].constraints[r][l]: 
                                    for line in cons.generate_implementation("python", unroll=True): myfile.write("\t" + line + "\n")      
                            myfile.write("\n")
            else: 
                myfile.write("\n\t# Round function \n")
                myfile.write("\tfor i in range(" + str(nbr_rounds) + "):\n")  
                for s in my_prim.states_implementation_order: 
                    if s in my_prim.rounds_python_code_if_unrolled:
                        for code in my_prim.rounds_python_code_if_unrolled[s]:
                            myfile.write(f"\n\t\t{code[1]}\n")
                            for l in range(my_prim.states[s].nbr_layers+1):                        
                                for cons in my_prim.states[s].constraints[code[0]][l]: 
                                    for line in cons.generate_implementation("python"): myfile.write("\t\t\t" + line + "\n")                      
                    else:
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
                    if s in my_prim.rounds_c_code_if_unrolled:
                        for code in my_prim.rounds_c_code_if_unrolled[s]:
                            myfile.write(f"\n\t\t{code[1]}\n")
                            for l in range(my_prim.states[s].nbr_layers+1):                        
                                for cons in my_prim.states[s].constraints[code[0]][l]: 
                                    for line in cons.generate_implementation("c"): myfile.write("\t\t\t" + line + "\n")    
                            myfile.write("\t\t}\n")                  
                    else:
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
