import copy
from operators.operators import Operator, Rot, Shift, RaiseExceptionVersionNotExisting
from operators.boolean_operators import NOT, AND, N_XOR, XOR

class SHACAL2_Sigma0(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_vars = copy.deepcopy(input_vars) + copy.deepcopy(input_vars) + copy.deepcopy(input_vars)
        for i in range(3):
            temp_vars[i].ID += '_' + str(i)
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        if keysize==512:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 7, "SIGMA0_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 18, "SIGMA0_ROT_2"]]
            shift = [self.vars[0][0], self.vars[1][2], 'r', 3, "SIGMA0_SHR_1"]
        elif keysize==1024:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 1, "SIGMA0_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 8, "SIGMA0_ROT_2"]]
            shift = [self.vars[0][0], self.vars[1][2], 'r', 7, "SIGMA0_SHR_1"]
        
        self.layers.append([Rot([rotation[0]], [rotation[1]], rotation[2], rotation[3], rotation[4]) for rotation in rotations] + [Shift([shift[0]], [shift[1]], shift[2], shift[3], shift[4])])
        self.layers.append([N_XOR([self.vars[1][0], self.vars[1][1], self.vars[1][2]], [self.vars[2][0]], "SUM0_NXOR1")])

    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
                    if cons.__class__.__name__ == 'Rot': 
                        code_list += cons.generate_implementation_header_unique(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class SHACAL2_Sigma1(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_vars = copy.deepcopy(input_vars) + copy.deepcopy(input_vars) + copy.deepcopy(input_vars)
        for i in range(3):
            temp_vars[i].ID += '_' + str(i)
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        if keysize==512:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 17, "SIGMA1_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 19, "SIGMA1_ROT_2"]]
            shift = [self.vars[0][0], self.vars[1][2], 'r', 10, "SIGMA1_SHR_1"]
        elif keysize==1024:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 19, "SIGMA1_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 61, "SIGMA1_ROT_2"]]
            shift = [self.vars[0][0], self.vars[1][2], 'r', 6, "SIGMA1_SHR_1"]
        
        self.layers.append([Rot([rotation[0]], [rotation[1]], rotation[2], rotation[3], rotation[4]) for rotation in rotations] + [Shift([shift[0]], [shift[1]], shift[2], shift[3], shift[4])])
        self.layers.append([N_XOR([self.vars[1][0], self.vars[1][1], self.vars[1][2]], [self.vars[2][0]], "SUM0_NXOR1")])

    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
                    if cons.__class__.__name__ == 'Rot':
                        code_list += cons.generate_implementation_header_unique(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")



class SHACAL2_Sum0(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_vars = copy.deepcopy(input_vars) + copy.deepcopy(input_vars) + copy.deepcopy(input_vars)
        for i in range(3):
            temp_vars[i].ID += '_' + str(i)
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        if keysize==512:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 2, "SUM0_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 13, "SUM0_ROT_2"], [self.vars[0][0], self.vars[1][2], 'r', 22, "SUM0_ROT_3"]]
        elif keysize==1024:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 28, "SUM0_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 34, "SUM0_ROT_2"], [self.vars[0][0], self.vars[1][2], 'r', 39, "SUM0_ROT_3"]]
        
        self.layers.append([Rot([rotation[0]], [rotation[1]], rotation[2], rotation[3], rotation[4]) for rotation in rotations])
        self.layers.append([N_XOR([self.vars[1][0], self.vars[1][1], self.vars[1][2]], [self.vars[2][0]], "SUM0_NXOR1")])

        
    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
                    if cons.__class__.__name__ == 'Rot': 
                        code_list += cons.generate_implementation_header_unique(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class SHACAL2_Sum1(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_vars = copy.deepcopy(input_vars) + copy.deepcopy(input_vars) + copy.deepcopy(input_vars)
        for i in range(3):
            temp_vars[i].ID += '_' + str(i)
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        if keysize==512:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 6, "SUM1_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 11, "SUM1_ROT_2"], [self.vars[0][0], self.vars[1][2], 'r', 25, "SUM1_ROT_3"]]
        elif keysize==1024:
            rotations = [[self.vars[0][0], self.vars[1][0], 'r', 14, "SUM1_ROT_1"], [self.vars[0][0], self.vars[1][1], 'r', 18, "SUM1_ROT_2"], [self.vars[0][0], self.vars[1][2], 'r', 41, "SUM1_ROT_3"]]
        
        self.layers.append([Rot([rotation[0]], [rotation[1]], rotation[2], rotation[3], rotation[4]) for rotation in rotations])
        self.layers.append([N_XOR([self.vars[1][0], self.vars[1][1], self.vars[1][2]], [self.vars[2][0]], "SUM0_NXOR1")])

        
    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
                    if cons.__class__.__name__ == 'Rot': 
                        code_list += cons.generate_implementation_header_unique(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")




class SHACAL2_Maj(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_vars = copy.deepcopy(input_vars)
        for i in range(3):
            temp_vars[i].ID += '_' + str(i)
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        ANDOperations = [[self.vars[0][0], self.vars[0][1], self.vars[1][0], "Maj_AND_1"], [self.vars[0][0], self.vars[0][2], self.vars[1][1], "Maj_AND_2"], [self.vars[0][1], self.vars[0][2], self.vars[1][2], "Maj_AND_3"]]

        self.layers.append([AND([ANDOperation[0], ANDOperation[1]], [ANDOperation[2]], ANDOperation[3]) for ANDOperation in ANDOperations])
        self.layers.append([N_XOR([self.vars[1][0], self.vars[1][1], self.vars[1][2]], [self.vars[2][0]], "Maj_NXOR1")])
        
        
    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")


class SHACAL2_Ch(Operator):
    def __init__(self, input_vars, output_vars, keysize=512, ID = None):
        super().__init__(input_vars, output_vars, ID = ID)
        self.layers = []
        self.vars = []

        self.vars.append(input_vars)

        temp_var = [copy.deepcopy(input_vars[0])]
        temp_var[0].ID += '_0_0'
        self.vars.append(temp_var)

        
        temp_vars = [copy.deepcopy(input_vars[0])] + [copy.deepcopy(input_vars[1])]
        for i in range(2):
            temp_vars[i].ID += '_1_' + str(i) # 0: e ^ f; 1: NOT e ^ g
        self.vars.append(temp_vars)

        self.vars.append(output_vars)

        NOTOperation = [self.vars[0][0], self.vars[1][0], "Ch_NOT_1"]
        ANDOperations = [[self.vars[0][0], self.vars[0][1], self.vars[2][0], "Ch_AND_1"], 
                         [self.vars[1][0], self.vars[0][2], self.vars[2][1], "Ch_AND_2"]]

        self.layers.append([NOT([NOTOperation[0]], [NOTOperation[1]], NOTOperation[2])])
        self.layers.append([AND([ANDOperation[0], ANDOperation[1]], [ANDOperation[2]], ANDOperation[3]) for ANDOperation in ANDOperations])
        self.layers.append([XOR([self.vars[2][0], self.vars[2][1]], [self.vars[3][0]], "Ch_XOR1")])
      

        
    def generate_implementation_header(self, implementation_type='python'):
        header_set = []
        code_list = []
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                cons = self.layers[i][j]
                if [cons.__class__.__name__] not in header_set:
                    header_set.append([cons.__class__.__name__]) 
                    if cons.generate_implementation_header(implementation_type) != None: 
                        code_list += cons.generate_implementation_header(implementation_type)
        return code_list 
    
    def generate_implementation(self, implementation_type='python', unroll=False):
        if implementation_type == 'python' or implementation_type == 'c': 
            code_list = []
            if implementation_type == 'c':
                var_ids = [var.ID if unroll else var.remove_round_from_ID() for i in range(1, len(self.vars)-1) for var in self.vars[i]]
                claim_var_c = "uint8_t " + ", ".join(var_ids) + ";"
                code_list += [claim_var_c]
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    code_list += self.layers[i][j].generate_implementation(implementation_type, unroll=unroll)
            return code_list
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + implementation_type + "'")
        
    def generate_model(self, model_type='sat', unroll=True):
        if model_type == 'sat' or model_type == 'milp': 
            model_list = []
            for i in range(len(self.layers)):
                for j in range(len(self.layers[i])):
                    cons = self.layers[i][j]
                    cons.model_version = self.model_version.replace(self.__class__.__name__, cons.__class__.__name__)
                    model_list += cons.generate_model(model_type, unroll=unroll)
            return model_list
        elif model_type == 'cp': RaiseExceptionVersionNotExisting(str(self.__class__.__name__), self.model_version, model_type)
        else: raise Exception(str(self.__class__.__name__) + ": unknown model type '" + model_type + "'")
