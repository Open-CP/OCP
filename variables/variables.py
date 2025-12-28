# ********************* VARIABLES ********************* #
# Class that represents a variable object, i.e. a type of node in our graph modeling (the other type being the operators/contraints)
# A Variable node can only be linked to a Constraint node in the graph representation 

class Variable:
    def __init__(self, bitsize, value = None, ID = None):
        self.bitsize = bitsize    # bitsize of that variable
        self.value = value        # value of that variable (not necessarily set)
        self.ID = ID              # ID of that variable
        self.connected_vars = []  # list of variables connected, with corresponding operator each time (not counting the copy operators) and the input/output role
        self.copied_vars = []     # list of variables that are copies of that variable 

    def display_value(self, representation='binary'):   # method that displays the value of that variable, depending on the representation requested
        if representation == 'binary' and self.value:
            return bin(self.value)[2:].zfill(self.bitsize)
        elif representation == 'hexadecimal' and self.value:
            return hex(self.value)[2:].zfill((self.bitsize + 3) // 4)
        elif representation == 'integer':
            return str(self.value)
        else:
            return "Invalid representation"
        
    def display(self, representation='binary'):   # method that displays some information for that variable
        print("ID: " + self.ID + " / bitsize: " + str(self.bitsize) + " / value: ", end='') 
        print(self.display_value(representation)) 
            
    def remove_round_from_ID(self):   # method that removes the round number from the ID of that variable (used when unroll mode if off)
        return '_'.join(part for i, part in enumerate(self.ID.split("_")) if i != 1)
        