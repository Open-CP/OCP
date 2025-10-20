import os, os.path
from numpy import linspace
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

# function that generates the figure describing the primitive  
def generate_figure(my_prim, filename): 

    os.makedirs(os.path.dirname(filename), exist_ok=True)      
    
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
    
    nbr_rounds_table = [my_prim.functions[s].nbr_rounds for s in my_prim.functions_display_order]
    nbr_layers_table = [my_prim.functions[s].nbr_layers for s in my_prim.functions_display_order]
    nbr_words_table = [my_prim.functions[s].nbr_words for s in my_prim.functions_display_order]
    constraints_table = [my_prim.functions[s].constraints for s in my_prim.functions_display_order]
    vars_table = [my_prim.functions[s].vars for s in my_prim.functions_display_order]
    vars_coord = []
    
    ax = plt.gca()
    
    ax.annotate(my_prim.name, xy=(-op_length, 4*elements_height), fontsize=3*var_font_size, ha="center")
    
    # computation of the maximum x-lenghth for each function
    max_length = [0]*len(my_prim.functions)
    for i in range(len(my_prim.functions)):
       for r in range(1,nbr_rounds_table[i]+1):
           for l in range(nbr_layers_table[i]+1):
               temp = x_space*(len(vars_table[i][r][l])-1) + var_length*len(vars_table[i][r][l])
               if temp > max_length[i]: max_length[i] = temp
               temp = x_space*(len(constraints_table[i][r][l])-1) + op_length*len(constraints_table[i][r][l])
               if temp > max_length[i]: max_length[i] = temp
     
    # display of the round delimitation lines
    for r in range(0,max(nbr_rounds_table)): 
        y_coord = 2*elements_height-r*(y_space_rounds + 2*(max(nbr_layers_table)+1)*(y_space_layer + elements_height))
        plt.plot([-op_length-40, sum(max_length)+x_space_state*(len(my_prim.functions)-1)], [y_coord, y_coord], linewidth=0.1, linestyle='dashed', color='gray')
        ax.annotate("Round " + str(r+1), xy=(-op_length,y_coord-8), fontsize=2*var_font_size, ha="center")
                
    # display the input variables
    y_coord = -y_space_in_out
    x_shift_input = 0
    cpt = 0
    for my_input in my_prim.inputs:
        for w in range(len(my_prim.inputs[my_input])):
            x_coord = x_shift_input + w*(x_space + op_length)
            ax.add_patch(Ellipse((x_coord,-y_coord), var_length, elements_height, facecolor=in_color))
            ax.annotate(my_prim.inputs[my_input][w].ID, xy=(x_coord,-y_coord), fontsize=var_font_size, ha="center")
            vars_coord.append((my_prim.inputs[my_input][w].ID,(x_coord,-y_coord)))
        x_shift_input = x_shift_input + x_space_state + max_length[cpt]
        cpt = cpt + 1
                
    # diplay the variables  
    max_y_space = 0
    x_shift_state = 0
    for i in range(len(my_prim.functions)):
       y_shift_round = 0
       for r in range(1,nbr_rounds_table[i]+1):
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
    for my_output in my_prim.outputs:
        for w in range(len(my_prim.outputs[my_output])):
            x_coord = x_shift_input + w*(x_space + op_length)
            ax.add_patch(Ellipse((x_coord,-y_coord), var_length, elements_height, facecolor=out_color))
            ax.annotate(my_prim.outputs[my_output][w].ID, xy=(x_coord,-y_coord), fontsize=var_font_size, ha="center")
            vars_coord.append((my_prim.outputs[my_output][w].ID,(x_coord,-y_coord)))
        x_shift_input = x_shift_input + x_space_state + max_length[cpt]
        cpt = cpt + 1  
        
    # diplay the operators and links to the variables 
    vars_coord = dict(vars_coord)
    x_shift_state = 0
    for i in range(len(my_prim.functions)):
       y_shift_round = 0
       for r in range(1,nbr_rounds_table[i]+1):
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
                       if constraints_table[i][r][l][w].__class__.__name__ == "Rot": ax.annotate(str(constraints_table[i][r][l][w].direction) + " - " + str(constraints_table[i][r][l][w].amount), xy=(x_coord+op_length/2,-y_coord-elements_height/4), fontsize=op_font_size, ha="center")
                       
                       # display the links with the variables
                       my_inputs = constraints_table[i][r][l][w].input_vars
                       my_outputs = constraints_table[i][r][l][w].output_vars
                       in_x_coord, in_y_coord = list(linspace(x_coord,x_coord+op_length,len(my_inputs)+2)[1:-1]), -y_coord+elements_height/2
                       out_x_coord, out_y_coord = list(linspace(x_coord,x_coord+op_length,len(my_outputs)+2)[1:-1]), -y_coord-elements_height/2
                       for j in range(len(my_inputs)):
                            if isinstance(my_inputs[j], list):
                                for jj in range(len(my_inputs[j])):
                                    (var_x_coord, var_y_coord) = vars_coord[my_inputs[j][jj].ID]
                                    ax.arrow(var_x_coord, var_y_coord, in_x_coord[j]-var_x_coord, in_y_coord-var_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                            else:
                                (var_x_coord, var_y_coord) = vars_coord[my_inputs[j].ID]
                                ax.arrow(var_x_coord, var_y_coord, in_x_coord[j]-var_x_coord, in_y_coord-var_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                       for j in range(len(my_outputs)):
                            if isinstance(my_outputs[j], list):
                                for jj in range(len(my_outputs[j])):
                                    (var_x_coord, var_y_coord) = vars_coord[my_outputs[j][jj].ID]
                                    ax.arrow(out_x_coord[j], out_y_coord, var_x_coord-out_x_coord[j], var_y_coord-out_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                            else:
                                (var_x_coord, var_y_coord) = vars_coord[my_outputs[j].ID]
                                ax.arrow(out_x_coord[j], out_y_coord, var_x_coord-out_x_coord[j], var_y_coord-out_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                   else:
                       (var_in_x_coord, var_in_y_coord) = vars_coord[constraints_table[i][r][l][w].input_vars[0].ID]
                       (var_out_x_coord, var_out_y_coord) = vars_coord[constraints_table[i][r][l][w].output_vars[0].ID]
                       ax.arrow(var_in_x_coord, var_in_y_coord, var_out_x_coord-var_in_x_coord, var_out_y_coord-var_in_y_coord, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
                       
           y_shift_round = y_shift_round + y_space_rounds + 2*(max(nbr_layers_table)+1)*(y_space_layer + elements_height)
           
       x_shift_state = x_shift_state + x_space_state + max_length[i]
       
    # display the input and output links   
    for j in range(len(my_prim.inputs_constraints)):
        (x_in, y_in) = vars_coord[my_prim.inputs_constraints[j].input_vars[0].ID]
        (x_state_in_coord, y_state_in_coord) = vars_coord[my_prim.inputs_constraints[j].output_vars[0].ID]
        ax.arrow(x_in, y_in, x_state_in_coord-x_in, y_state_in_coord-y_in, linewidth=0.3, length_includes_head=True, width= 0.15, head_width= 1 , zorder=0.5)
    for j in range(len(my_prim.outputs_constraints)):
        (x_state_out_coord, y_state_out_coord) = vars_coord[my_prim.outputs_constraints[j].input_vars[0].ID]
        (x_out, y_out) = vars_coord[my_prim.outputs_constraints[j].output_vars[0].ID]
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