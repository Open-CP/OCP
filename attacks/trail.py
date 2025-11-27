from abc import ABC, abstractmethod
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parent.parent # this file -> attacks -> <ROOT>
sys.path.append(str(ROOT))

FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(parents=True, exist_ok=True)


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
            "trail_values": List[str], The values of the trail
        """
        data["functions"] = data.get("functions", ["PERMUTATION"])
        if isinstance(data['rounds'], int):
            data['rounds'] = {s: list(range(1, data['rounds'] + 1)) for s in data['functions']}
        if isinstance(data['trail_values'], list):
            data['trail_values'] = {"PERMUTATION": data['trail_values']}
        super().__init__("differential", data, solution_trace=solution_trace)


    def print_trail(self, show_mode=2):
        """
        Print the trail in a human-readable format.

        Parameters:
        - mode:
            0 - Print only the first and last round (layer 0) in hexadecimal strings.
            1 - Print all rounds (layer 0) in hexadecimal strings.
            2 - Print all rounds and all layers in hexadecimal strings.
        """
        lines = super().print_trail(show_mode)
        if "diff_weight" in self.data and self.data["diff_weight"] is not None:
            lines += f"Total Weight: {self.data['diff_weight']}\n"
        if "rounds_diff_weight" in self.data and self.data["rounds_diff_weight"] is not None:
            lines += f"rounds_diff_weight: {self.data['rounds_diff_weight']}\n"

        trail_values = self.data['trail_values']
        for fun in trail_values:
            print(f"Printing trail for function: {fun}...")
            if show_mode == 0:
                show_rounds = [1, len(trail_values[fun])] if len(trail_values[fun]) > 1 else [1]
                show_layers = list(range(len(trail_values[fun][0])))
            elif show_mode == 1:
                show_rounds = list(range(1, len(trail_values[fun]) + 1))
                show_layers = [0]
            elif show_mode == 2:
                show_rounds = list(range(1, len(trail_values[fun]) + 1))
                show_layers = list(range(len(trail_values[fun][0])))
            else:
                raise ValueError(f"[WARNING] show_mode {show_mode} should be 0, 1, or 2.")

            lines += f"-------- {fun}: --------\n"
            for r in show_rounds:
                lines += f"Round {r}:\n"
                for l in show_layers:
                    value_str = self.data['trail_values'][fun][r-1][l]
                    lines += f"{value_str}\n"
        print(lines)
        return lines
