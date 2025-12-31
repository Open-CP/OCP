from abc import ABC, abstractmethod
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent.parent # this file -> attacks -> <ROOT>
sys.path.append(str(ROOT))

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


def bin_to_hex(bits): # Format bits as hex (with "-" for unknown nibbles).
    if len(bits) % 4 != 0:
        pad = 4 - len(bits) % 4
        bits += "0" * pad  # Pad with zeros to make length a multiple of 4
        print(f"[WARNING] Padded {pad} trailing '0'(s) to align to 4-bit nibbles for hex formatting.")
    hex_digits = []
    # Convert each 4-bit group to hex, but keep "-" when any bit is unknown.
    for i in range(0, len(bits), 4):
        chunk = bits[i:i + 4]
        if "-" in chunk:
            if chunk != "----":
                print(f"[WARNING] Nibble '{chunk}' contains mixed unknown bits; using '-' as a lossy representation.")
            hex_digits.append("-")
        else:
            hex_digits.append(hex(int(chunk, 2))[2:])
    return "".join(hex_digits)


# Class that represents a trail derived from the solution.
class Trail(ABC):
    def __init__(self, type, data, solution_trace=None):
        """
        Initialize the Trail object.

        Parameters:
        - type: The type of the trail (e.g., "differential", "linear")
        - data: A dictionary containing:
            "cipher": str, The name of the cipher (e.g., "AES")
            "rounds": List[int] | int, The number of rounds or a list of round indices (e.g., 3 or [1, 2, 3])
            ...
        - solution_trace: # Optional mapping from variable name to its value, for example, the solution returned from MILP/SAT solver.
        """
        assert "cipher" in data, "[WARNING] data must contain 'cipher'"
        assert "rounds" in data, "[WARNING] data must contain 'rounds'"
        self.type = type
        self.data = data
        self.solution_trace = solution_trace or {}
        self.json_filename = str(FILES_DIR / f"{self.data['cipher']}_{self.type}_trail.json")
        self.txt_filename = str(FILES_DIR / f"{self.data['cipher']}_{self.type}_trail.txt")


    def save_json(self): # Save the trail information into a .json file.
        trail_dict = {
            "type": str(self.type).upper(),
            "data": dict(self.data),
            "solution_trace": dict(self.solution_trace),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool": "OCP",
            }
        with open(self.json_filename, "w") as f:
            json.dump(trail_dict, f, ensure_ascii=False, indent='\t')

    def save_trail_txt(self, show_mode=2): # Save the trail in a human-readable format into a .txt file.
        lines = self.print_trail(show_mode)
        with open(self.txt_filename, "w") as f:
            f.write(lines)
        return lines

    def save_trail_tex(self, filename=None): # TO DO
        pass

    def save_trail_pdf(self, filename=None): # TO DO
        pass

    @abstractmethod
    def print_trail(self, show_mode):
        lines = "========== Trail ==========\n"
        lines += f"Type: {self.type}\n"
        lines += f"Cipher: {self.data['cipher']}\n"
        return lines


class DifferentialTrail(Trail):
    def __init__(self, data, solution_trace=None):
        """
        Parameters:
        - data: A dictionary containing:
            "cipher": str, The name of the cipher (e.g., "AES")
            "functions": List[str], The list of functions involved in the cipher (e.g., ["PERMUTATION", "KEY_SCHEDULE"])
            "rounds": Dict[str, List[int] | int], For each function, the number of rounds or a list of round indices (e.g., {"PERMUTATION": 3} or {"PERMUTATION": [1, 2, 3]})
            "diff_weight": float | int | None, The weight (defined as the negetive of logarithm base 2 of the differential probability) of the differential trail (e.g., 2)
            "rounds_diff_weight": List[float] | None, The list of weigts of each round (e.g., [0, 1, 1])
            "trail_struct": Dict, The structure of the trail
        """
        super().__init__("differential", data, solution_trace=solution_trace)


    def print_trail(self, show_mode=2, hex_format=True):
        """
        Print the trail in a human-readable format.

        Parameters:
        - mode:
            0 - Print only the first and last round (first layer) states excluding temporary variables.
            1 - Print all rounds (first layer) excluding temporary variables.
            2 - Print all rounds and all layers excluding temporary variables.
            3 - Print all rounds and all layers including temporary variables.
        - hex_format: If True, print the values in hexadecimal format; otherwise, print in binary format.
        """
        lines = super().print_trail(show_mode)
        lines += f"Print the differential trail in {'hexadecimal' if hex_format else 'binary'} format.\n"
        if show_mode == 0: lines += "Show Mode: First Layer of First and Last Round.\n"
        elif show_mode == 1: lines += "Show Mode: First Layer of All Rounds (layer 0)\n"
        elif show_mode == 2: lines += "Show Mode: All Layers of All Rounds\n"
        elif show_mode == 3: lines += "Show Mode: All Layers of All Rounds (Including Temporary Words)\n"

        trail_struct = self.data['trail_struct']

        # Print inputs
        if "inputs" in trail_struct and isinstance(trail_struct["inputs"], dict):
            lines += f"######## Input: ########\n"
            for name, node_list in trail_struct["inputs"].items():
                diff = ""
                for node in node_list:
                    diff += f"{node['bin_values']}"
                lines += f"{name}: " + (f"{bin_to_hex(diff)}" if hex_format else diff)
                lines += "\n"

        # Print functions
        if "functions" in trail_struct and isinstance(trail_struct["functions"], dict):
            for fun in self.data["functions"]:
                lines += f"######## Function: {fun} ########\n"
                if show_mode == 0:
                    show_rounds = [self.data["rounds"][fun][0], self.data["rounds"][fun][-1]] if len(self.data["rounds"][fun]) > 1 else [self.data["rounds"][fun][0]]
                elif show_mode == 1 or show_mode == 2 or show_mode == 3:
                    show_rounds = list(range(self.data["rounds"][fun][0], self.data["rounds"][fun][-1] + 1))
                for r in show_rounds:
                    lines += f"Round {r}:\n"
                    for l in trail_struct["functions"][fun][r]:
                        # if fun == "SUBKEYS" and l == 0: # For SUBKEYS, layer 0 is meaningless, so skip it
                        #     continue
                        if (show_mode == 0 or show_mode == 1) and l != 0 and fun != "SUBKEYS": # For show_mode 0 and 1, only layer 0 is printed
                            continue
                        lines += f"Layer {l}: "
                        diff = ""
                        for i in range(trail_struct["functions"][fun]["nbr_words"]):
                            node = trail_struct["functions"][fun][r][l][i]
                            diff += f"{node['bin_values']}"
                        lines += f"{bin_to_hex(diff)}" if hex_format else diff
                        if show_mode == 3:
                            diff = ""
                            for i in range(trail_struct["functions"][fun]["nbr_temp_words"]):
                                node = trail_struct["functions"][fun][r][l][trail_struct["functions"][fun]["nbr_words"] + i]
                                diff += f"{node['bin_values']}"
                            lines += f"{bin_to_hex(diff)}" if hex_format else diff
                        lines += "\n"

        # Print outputs
        if "outputs" in trail_struct and isinstance(trail_struct["outputs"], dict):
            lines += f"######## Output: ########\n"
            for name, node_list in trail_struct["outputs"].items():
                diff = ""
                for node in node_list:
                    diff += f"{node['bin_values']}"
                lines += f"{name}: " + (f"{bin_to_hex(diff)}" if hex_format else diff)
                lines += "\n"

        if "diff_weight" in self.data and self.data["diff_weight"] is not None:
            lines += f"\nTotal Weight: {self.data['diff_weight']}\n"
        if "rounds_diff_weight" in self.data and self.data["rounds_diff_weight"] is not None:
            lines += f"rounds_diff_weight: {self.data['rounds_diff_weight']}\n"
        print(lines)
        return lines
