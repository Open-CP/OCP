from abc import ABC, abstractmethod
from collections import defaultdict
import re


# Class that represents a differential/linear trail derived from the solution.
class Trail(ABC):
    def __init__(self, primitive, goal, start_round, end_round, solution):
        self.primitive = primitive  # The cipher primitive
        self.goal = goal # the attack goal
        self.start_round = start_round  # The start round of the trail
        self.end_round = end_round  # The end round of the trail
        self.solution = solution  # The solution dictionary mapping variable names to values
        self.trail_vars = self.gen_trail_vars()
        self.trail = self.extract_trail_from_solution()  # Extract the trail from the solution dictionary

    @abstractmethod
    def gen_trail_vars(self): # Generate variables representing the trail.
        pass

    @abstractmethod
    def extract_trail_from_solution(self):   # Extract the trail from the solution dictionary.
        pass
    
    @abstractmethod
    def print_trail(self, mode=2, hex_format=True, filename=None): # Print the trail in a human-readable format.
        pass  

    # @abstractmethod
    # def visualize_trail(self): # Generate a visual representation of the trail.
    #     pass  


class DifferentialTrail(Trail):
    def __init__(self, primitive, goal, start_round, end_round, solution):
        super().__init__(primitive, goal, start_round, end_round, solution)

    def gen_trail_vars(self): # Generate variables that represent the differential trail based on the cipher structure.       
        bitwise = "TRUNCATEDDIFF" not in self.goal
        trail_vars = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for s in self.primitive.states:
            for r in range(self.start_round[s], self.end_round[s]+1):
                for l in range(self.primitive.states[s].nbr_layers+1):
                    for var in self.primitive.states[s].vars[r][l]:
                        if bitwise and var.bitsize > 1:
                            trail_vars[s][r][l].extend([f"{var.ID}_{n}" for n in range(var.bitsize)]) # Append all bit-level variable names
                        else:
                            trail_vars[s][r][l].append(var.ID) # Append word-level variable name
        return trail_vars
    
    def extract_trail_from_solution(self):
        trail_values = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
        for s in self.primitive.states:
            for r in range(self.start_round[s], self.end_round[s] + 1):
                for l in range(self.primitive.states[s].nbr_layers + 1):
                    value = ""
                    for var in self.trail_vars[s][r][l]:
                        value += str(self.solution.get(var, "0"))  # Get the value of the variable from the solution
                    trail_values[s][r][l] = value
        return trail_values

    def print_trail(self, show_mode=2, hex_format=True):
        """
        Print the trail in a human-readable format.
        
        Parameters:
        - mode: 
            0 - Print only the first and last round (layer 0).
            1 - Print all rounds (layer 0).
            2 - Print all rounds and all layers.
        - hex_format: If True, convert values to hexadecimal strings.
        """       

        if "obj_fun_value" in self.solution:
            print(f"******** objective value of the solution: {self.solution['obj_fun_value']} ********")
            
        for s in self.primitive.states:
            print(f"========= State {s}: =========")

            if show_mode == 0:
                for r in [self.start_round[s], self.end_round[s]]:
                    print(f"Round {r}:")  
                    layers = [0 if s in ["STATE", "KEY_STATE"] else 1]
                    for l in layers:
                        value_str = self.trail[s][r][l]
                        if hex_format:
                            value_str = "0x" + hex(int(value_str, 2))[2:].zfill(len(value_str) // 4)
                        print(f"(Layer {l}): {value_str}")

            elif show_mode == 1:
                for r in list(range(self.start_round[s], self.end_round[s]+1)):
                    print(f"Round {r}:")
                    self._print_round_objective_function(r)
                    layers = [0 if s in ["STATE", "KEY_STATE"] else 1]
                    for l in layers:
                        value_str = self.trail[s][r][l]
                        if hex_format:
                            value_str = "0x" + hex(int(value_str, 2))[2:].zfill(len(value_str) // 4)
                        print(f"(Layer {l}): {value_str}")
    
            elif show_mode == 2:
                for r in list(range(self.start_round[s], self.end_round[s]+1)):
                    print(f"Round {r}:")
                    self._print_round_objective_function(r)
                    layers = list(range(self.primitive.states[s].nbr_layers + 1))
                    for l in layers:                        
                        value_str = self.trail[s][r][l]
                        if hex_format:
                            value_str = "0x" + hex(int(value_str, 2))[2:].zfill(len(value_str) // 4)
                        print(f"(Layer {l}): {value_str}")
                

    def _print_round_objective_function(self, r):
        obj_fun = self.solution.get("obj_fun", None)
        if obj_fun is not None and r <= len(obj_fun): # Print the objective value
            obj_fun_r = obj_fun[r-1]
            if isinstance(obj_fun_r, list) and len(obj_fun_r) == 1 and '+' in obj_fun_r[0]:
                obj_fun_r = obj_fun_r[0].split('+')
            w = 0
            for obj in obj_fun_r:
                match = re.match(r'(\d*\.*\d+|\d*)\s*(\w+)', obj.strip())
                if match:
                    coefficient = float(match.group(1)) if match.group(1) != '' else 1
                    variable = match.group(2)                
                    if variable in self.solution:
                        w += coefficient * self.solution[variable]
                else:
                    print(f"Warning: Unable to parse '{obj}'")
                    w = "-"
                    break
            print(f"Objective Function Value: {w}")