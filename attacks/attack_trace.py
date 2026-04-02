from abc import ABC, abstractmethod
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1] # this file -> attacks -> <ROOT>
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


class AttackTrace(ABC):
    def __init__(self, attack_type, data, solution_trace=None):
        """
        Initialize the AttackTrace object.

        Parameters:
        - attack_type: The type of the trail (e.g., "differential", "linear")
        - data: A dictionary, must contain:
            "cipher": str, The name of the cipher (e.g., "AES")
            "rounds": List[int], The number of rounds, a list of round indices (e.g. [1, 2, 3])
        - solution_trace: Optional mapping from variable name to its value, for example, the solution returned from MILP/SAT solver.
        """

        assert "cipher" in data, "[WARNING] data must contain 'cipher'"
        assert "rounds" in data, "[WARNING] data must contain 'rounds'"

        self.type = attack_type
        self.data = data
        self.solution_trace = solution_trace or {}

    def to_dict(self):
        return {
            "type": str(self.type).upper(),
            "data": dict(self.data),
            "solution_trace": dict(self.solution_trace),
            "tool": "OCP1.0",
        }

    @abstractmethod
    def save_json(self, **kwargs):  # Save the trail information into a .json file.
        pass

    @abstractmethod
    def save_txt(self, **kwargs): # Save the trail in a human-readable format into a .txt file.
        pass

    @abstractmethod
    def save_tex(self, **kwargs):
        pass

    @abstractmethod
    def save_pdf(self, **kwargs):
        pass


# Abstract base class for trail-type attack results
class Trail(AttackTrace):
    def __init__(self, attack_type, data, solution_trace=None):
        """
        Parameters:
        - data: A dictionary containing:
            "functions": List[str], The list of functions involved in the cipher (e.g., ["PERMUTATION", "KEY_SCHEDULE"])
            "config_model": Configuration for the model.
            "config_solver": Configuration for the solver.
        """
        super().__init__(attack_type, data, solution_trace=solution_trace)
        config_model = data.get("config_model", {})
        config_solver = data.get("config_solver", {})
        solver_name = config_solver.get("solver", "DEFAULT")

        if "filename" in config_model:
            model_path = Path(config_model["filename"])
            base_name = model_path.stem.removesuffix("_model")
            base_path = model_path.with_name(base_name)
            self.json_filename = str(base_path) + f"_{self.type}_{solver_name}_trail.json"
            self.txt_filename = str(base_path) + f"_{self.type}_{solver_name}_trail.txt"
        else:
            base_path = FILES_DIR / f"{self.data['cipher']}_{self.type}_{solver_name}_trail"
            self.json_filename = str(base_path) + f".json"
            self.txt_filename = str(base_path) + f".txt"

    def save_json(self):
        trail_dict = self.to_dict()
        with open(self.json_filename, "w") as f:
            json.dump(trail_dict, f, ensure_ascii=False, indent='\t')

    def save_txt(self, show_mode=2, hex_format=True):
        lines = self.print_trail(show_mode, hex_format=hex_format)
        with open(self.txt_filename, "w") as f:
            f.write(lines)
        return lines

    def save_tex(self): # TO DO
        raise NotImplementedError("LaTeX export is not implemented yet.")

    def save_pdf(self): # TO DO
        raise NotImplementedError("PDF export is not implemented yet.")

    @abstractmethod
    def print_trail(self, show_mode, hex_format=True):
        """
        Print the trail in a human-readable format.

        Parameters:
        - show_mode:
            0 - Print only the first and last round (first layer) states excluding temporary variables.
            1 - Print all rounds (first layer) excluding temporary variables.
            2 - Print all rounds and all layers excluding temporary variables.
            3 - Print all rounds and all layers including temporary variables.
        - hex_format: If True, print the values in hexadecimal format; otherwise, print in binary format.
        """
        lines = "========== Trail ==========\n"
        lines += f"Type: {self.type}\n"
        lines += f"Cipher: {self.data['cipher']}\n"
        lines += f"Print {self.type} trail in {'hexadecimal' if hex_format else 'binary'} format.\n"

        if show_mode == 0:
            lines += "Show Mode: First Layer of First and Last Round.\n"
        elif show_mode == 1:
            lines += "Show Mode: First Layer of All Rounds (layer 0)\n"
        elif show_mode == 2:
            lines += "Show Mode: All Layers of All Rounds\n"
        elif show_mode == 3:
            lines += "Show Mode: All Layers of All Rounds (Including Temporary Words)\n"
        else:
            lines += f"[WARNING] Invalid show_mode {show_mode}. Cannot print the trail.\n"
            return lines

        def _validate_trail_struct(trail_struct):
            """
            Validate the basic structure of trail_struct. For example:
            trail_struct = {
                            "inputs": {...},
                            "outputs": {...},
                            "functions": {
                                "PERMUTATION": {
                                    "rounds": [],
                                    "nbr_words": ...,
                                    "nbr_temp_words": ...,
                                    1: {...},
                                    2: {...},
                                    3: {...},
                                },
                                ...
                            }
                        }
            """
            if not isinstance(trail_struct, dict):
                return "[WARNING] trail_struct is not a dictionary. Cannot print the trail structure.\n"

            for key in ("inputs", "functions", "outputs"):
                if key in trail_struct and not isinstance(trail_struct[key], dict):
                    return f"[WARNING] trail_struct['{key}'] is not a dictionary.\n"

            if "functions" not in trail_struct:
                return "[WARNING] trail_struct does not contain 'functions'. Cannot print the trail structure.\n"

            for fun, fun_struct in trail_struct["functions"].items():
                if not isinstance(fun_struct, dict):
                    return f"[WARNING] trail_struct['functions']['{fun}'] is not a dictionary.\n"

                if "rounds" not in fun_struct or not isinstance(fun_struct["rounds"], list) or len(fun_struct["rounds"]) == 0:
                    return f"[WARNING] 'rounds' is missing or invalid for function '{fun}'.\n"

                if "nbr_words" not in fun_struct or not isinstance(fun_struct["nbr_words"], int):
                    return f"[WARNING] 'nbr_words' is missing or invalid for function '{fun}'.\n"

                if "nbr_temp_words" not in fun_struct or not isinstance(fun_struct["nbr_temp_words"], int):
                    return f"[WARNING] 'nbr_temp_words' is missing or invalid for function '{fun}'.\n"

                for r in fun_struct["rounds"]:
                    if r not in fun_struct:
                        return f"[WARNING] Round {r} is missing for function '{fun}'.\n"
                    if not isinstance(fun_struct[r], dict):
                        return f"[WARNING] trail_struct['functions']['{fun}'][{r}] is not a dictionary.\n"

            return None

        trail_struct = self.data.get("trail_struct", None)
        warning = _validate_trail_struct(trail_struct)
        if warning is not None:
            lines += warning
            return lines

        # Print inputs
        if "inputs" in trail_struct:
            lines += "######## Input: ########\n"
            for name, node_list in trail_struct["inputs"].items():
                state = "".join(node["bin_values"] for node in node_list)
                lines += f"{name}: " + (bin_to_hex(state) if hex_format else state) + "\n"

        # Print functions
        for fun, fun_struct in trail_struct["functions"].items():
            lines += f"######## Function: {fun} ########\n"

            rounds = fun_struct["rounds"]
            if show_mode == 0:
                show_rounds = [rounds[0], rounds[-1]] if len(rounds) > 1 else [rounds[0]]
            else:
                show_rounds = rounds

            for r in show_rounds:
                lines += f"Round {r}:\n"
                for l in fun_struct[r]:
                    if show_mode in {0, 1} and l != 0 and fun != "SUBKEYS":
                        continue

                    lines += f"Layer {l}: "

                    nbr_words = fun_struct["nbr_words"]
                    nbr_temp_words = fun_struct["nbr_temp_words"]
                    layer_nodes = fun_struct[r][l]

                    state = "".join(layer_nodes[i]["bin_values"] for i in range(nbr_words))
                    lines += bin_to_hex(state) if hex_format else state

                    if show_mode == 3 and nbr_temp_words > 0:
                        temp_state = "".join(layer_nodes[nbr_words + i]["bin_values"] for i in range(nbr_temp_words))
                        lines += bin_to_hex(temp_state) if hex_format else temp_state
                    lines += "\n"

        # Print outputs
        if "outputs" in trail_struct:
            lines += "######## Output: ########\n"
            for name, node_list in trail_struct["outputs"].items():
                state = "".join(node["bin_values"] for node in node_list)
                lines += f"{name}: " + (bin_to_hex(state) if hex_format else state) + "\n"

        return lines


class DifferentialTrail(Trail):
    def __init__(self, data, solution_trace=None):
        """
        Parameters:
        - data: A dictionary containing:
            "diff_weight": float | int | None, The weight (defined as the negative of logarithm base 2 of the differential probability) of the differential trail (e.g., 2)
            "rounds_diff_weight": List[float] | None, The list of weights of each round (e.g., [0, 1, 1])
            "trail_struct": Dict, The structure of the trail
        """
        super().__init__("differential", data, solution_trace=solution_trace)


    def print_trail(self, show_mode=2, hex_format=True):
        lines = super().print_trail(show_mode, hex_format=hex_format)

        if "diff_weight" in self.data and self.data["diff_weight"] is not None:
            lines += f"\nTotal Weight: {self.data['diff_weight']}\n"
        if "rounds_diff_weight" in self.data and self.data["rounds_diff_weight"] is not None:
            lines += f"rounds_diff_weight: {self.data['rounds_diff_weight']}\n"
        print(lines)
        return lines


class LinearTrail(Trail):
    def __init__(self, data, solution_trace=None):
        """
        Parameters:
        - data: A dictionary containing:
            "linear_weight": float | int | None, The weight (defined as the negative of logarithm base 2 of the linear correlation) of the linear trail (e.g., 2)
            "rounds_linear_weight": List[float] | None, The list of weights of each round (e.g., [0, 1, 1])
            "trail_struct": Dict, The structure of the trail
        """
        super().__init__("linear", data, solution_trace=solution_trace)


    def print_trail(self, show_mode=2, hex_format=True):
        lines = super().print_trail(show_mode, hex_format=hex_format)

        if "linear_weight" in self.data and self.data["linear_weight"] is not None:
            lines += f"\nTotal Weight: {self.data['linear_weight']}\n"
        if "rounds_linear_weight" in self.data and self.data["rounds_linear_weight"] is not None:
            lines += f"rounds_linear_weight: {self.data['rounds_linear_weight']}\n"
        print(lines)
        return lines
