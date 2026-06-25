# ********************* VARIABLES ********************* #
# Class that represents a variable object, i.e. a type of node in our graph modeling (the other type being the operators/contraints)
# A Variable node can only be linked to a Constraint node in the graph representation 

class Variable:
    def __init__(self, bitsize, value = None, ID = None, copyorigin = None):
        if not isinstance(bitsize, int) or bitsize <= 0:
            raise ValueError("Variable bitsize must be a positive integer.")
        if ID is not None and not isinstance(ID, str):
            raise ValueError("Variable ID must be None or a string.")
        if value is not None and (not isinstance(value, int) or value < 0 or value >= 2**bitsize):
            raise ValueError("Variable value must be an integer fitting in bitsize.")

        self.bitsize = bitsize    # bitsize of that variable
        self.value = value        # value of that variable (not necessarily set)
        self.ID = ID              # ID of that variable
        self.connected_vars = []  # list of variables connected, with corresponding operator each time and the input/output role
        self.copied_vars = []     # list of variables that are copies of that variable - stored as tuples (variable, target operator for that variable, copy operator used to link that variable)
        self.copyorigin = copyorigin    # variable that is the origin of the copy (if this variable is a copy, None otherwise)
        
    def display_value(self, representation='binary'):   # method that displays the value of that variable, depending on the representation requested
        if self.value is None:
            return "None"
        if representation == 'binary':
            return bin(self.value)[2:].zfill(self.bitsize)
        elif representation == 'hexadecimal':
            return hex(self.value)[2:].zfill((self.bitsize + 3) // 4)
        elif representation == 'integer':
            return str(self.value)
        else:
            return "Invalid representation"
        
    def display(self, representation='binary'):   # method that displays some information for that variable
        display_id = "" if self.ID is None else self.ID
        print("ID: " + display_id + " / bitsize: " + str(self.bitsize) + " / value: " + self.display_value(representation))
    
    def remove_round_from_ID(self): # Remove the round number (ID format 'name_round_layer_pos'); IDs without a round are returned unchanged. Used when unroll mode is off.
        if self.ID is None:
            return ""
        parts = self.ID.split("_")
        if len(parts) >= 4 and parts[1].isdigit():
            return '_'.join(part for i, part in enumerate(parts) if i != 1)
        return self.ID
