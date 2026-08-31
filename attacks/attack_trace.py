"""Attack results: representation, formatting, and persistence.

Defines the ``AttackTrace`` abstract base and its concrete trails (``DifferentialTrail`` /
``LinearTrail``) built from a solver solution, plus the shared helpers used to turn a solution
into per-round trail structures and render/save them.
"""

from abc import ABC, abstractmethod
from pathlib import Path
import json

from tools.paths import get_files_dir
from tools.model_constraints import expand_var_ids


def bin_to_hex(bits):
    """Format a bit string as hexadecimal, keeping ``"-"`` for unknown nibbles.

    ``bits`` is read most-significant-bit first. Each 4-bit group (nibble) becomes one
    hex digit. A nibble is rendered as ``"-"`` whenever it contains any unknown bit
    (``"-"``); this is lossy, since a partially unknown nibble such as ``"1-01"`` cannot
    be represented exactly. When the length is not a multiple of 4, the input is padded
    with leading ``"0"`` (on the high/MSB side) so that the represented value is
    preserved. A warning is printed when padding is added or when a mixed nibble is
    collapsed to ``"-"``.

    Args:
        bits (str): A string of ``"0"`` / ``"1"`` / ``"-"`` characters, MSB first.

    Returns:
        str: The hex string, one character per nibble.

    Examples:
        >>> bin_to_hex("1010")
        'a'
        >>> bin_to_hex("----")    # fully-unknown nibble
        '-'
        >>> bin_to_hex("1-01")    # mixed unknown bits -> lossy '-' (warns)
        '-'
        >>> bin_to_hex("101")     # left-padded to "0101" (warns)
        '5'
    """
    if len(bits) % 4 != 0:
        pad = 4 - len(bits) % 4
        bits = "0" * pad + bits  # Left-pad with zeros (high/MSB side) to align nibbles, preserving the value
        print(f"[WARNING] Padded {pad} leading '0'(s) to align to 4-bit nibbles for hex formatting.")
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
    """Abstract base for an attack result.

    Args:
        attack_type (str): The type of the trail (e.g. ``"differential"``, ``"linear"``, ``"integral"``).
        data (dict): Attack data. Must contain ``"cipher"`` (str, cipher name,
            e.g. ``"AES"``); other keys are read on demand by the concrete subclasses.
        solution_trace (dict, optional): Mapping from variable name to its value,
            e.g. the solution returned by a MILP/SAT solver. Defaults to None.
    """

    def __init__(self, attack_type, data, solution_trace=None):
        if "cipher" not in data:
            raise ValueError("data must contain 'cipher'")

        self.type = attack_type
        self.data = data
        self.solution_trace = solution_trace or {}

    def to_dict(self):
        """Return the attack result as a JSON-serializable dictionary.

        Returns:
            dict: A mapping with the keys ``"type"`` (uppercased attack type),
            ``"data"``, ``"solution_trace"``, and ``"tool"`` (the OCP version tag).
        """
        data = dict(self.data)
        config_model = data.get("config_model")
        if isinstance(config_model, dict) and "decimal_objective_function" in config_model:
            config_model = dict(config_model)
            config_model.pop("decimal_objective_function", None)
            data["config_model"] = config_model
        return {
            "type": str(self.type).upper(),
            "data": data,
            "solution_trace": dict(self.solution_trace),
            "tool": "OCP1.0",
        }

    def _set_output_filenames(self, suffix):
        # Set output filenames to "<name>_<type>_<solver>_<suffix>.{json,txt,pdf,tex}":
        config_model = self.data.get("config_model", {})
        solver_name = self.data.get("config_solver", {}).get("solver", "DEFAULT")
        if "filename" in config_model:
            model_path = Path(config_model["filename"])
            stem = model_path.stem
            base_name = stem[:-len("_model")] if stem.endswith("_model") else stem
            base_path = model_path.with_name(f"{base_name}_{self.type}_{solver_name}_{suffix}")
        else:
            base_path = get_files_dir() / f"{self.data['cipher']}_{self.type}_{solver_name}_{suffix}"
        self.json_filename = f"{base_path}.json"
        self.txt_filename = f"{base_path}.txt"
        self.pdf_filename = f"{base_path}.pdf"
        self.tex_filename = f"{base_path}.tex"

    @abstractmethod
    def save_json(self, **kwargs):
        """Save the attack result to a ``.json`` file."""
        pass

    @abstractmethod
    def save_txt(self, **kwargs):
        """Save the attack result as human-readable text to a ``.txt`` file."""
        pass

    @abstractmethod
    def save_tex(self, **kwargs):
        """Save the attack result as a LaTeX ``.tex`` file."""
        pass

    @abstractmethod
    def save_pdf(self, **kwargs):
        """Save the attack result to a ``.pdf`` file."""
        pass


class Trail(AttackTrace):
    """Abstract base for trail-type attack results (differential, linear, ...).

    Args:
        attack_type (str): See :class:`AttackTrace`.
        data (dict): In addition to the base keys, may contain ``"functions"`` 
            (list of str, e.g. ``["PERMUTATION", "KEY_SCHEDULE"]``), ``"config_model"`` 
            (model configuration), and ``"config_solver"`` (solver configuration).
        solution_trace (dict, optional): See :class:`AttackTrace`. Defaults to None.
    """

    def __init__(self, attack_type, data, solution_trace=None):
        super().__init__(attack_type, data, solution_trace=solution_trace)
        self._set_output_filenames("trail")

    def print_trail(self, show_mode=2, hex_format=True):
        """Format the trail and print it to stdout.

        Args:
            show_mode (int): Level of detail, see :meth:`format_trail`.
            hex_format (bool): If True, format the values in hexadecimal; otherwise, in binary.
        """
        print(self.format_trail(show_mode, hex_format=hex_format))
    
    def save_json(self):
        trail_dict = self.to_dict()
        Path(self.json_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(trail_dict, f, ensure_ascii=False, indent='\t',
                      default=lambda o: f"<{type(o).__name__}>")

    def save_txt(self, show_mode=2, hex_format=True):
        """Save the trail as human-readable text to a ``.txt`` file.

        Args:
            show_mode (int): Level of detail, see :meth:`format_trail`.
            hex_format (bool): If True, format the values in hexadecimal; otherwise, in binary.
        """
        lines = self.format_trail(show_mode, hex_format=hex_format)
        Path(self.txt_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.txt_filename, "w", encoding="utf-8") as f:
            f.write(lines)
        return lines

    def save_tex(self): # TO DO
        raise NotImplementedError("LaTeX export is not implemented yet.")

    def save_pdf(self): # TO DO
        raise NotImplementedError("PDF export is not implemented yet.")

    @abstractmethod
    def format_trail(self, show_mode=2, hex_format=True):
        """Return the trail as a human-readable string.

        Args:
            show_mode (int): Level of detail:

                * ``0`` - first and last round only (first layer), excluding temporary variables.
                * ``1`` - all rounds (first layer only), excluding temporary variables.
                * ``2`` - all rounds and all layers, excluding temporary variables.
                * ``3`` - all rounds and all layers, including temporary variables.

            hex_format (bool): If True, format the values in hexadecimal; otherwise, in binary.

        Returns:
            str: The formatted trail.
        """
        lines = "========== Trail ==========\n"
        lines += f"Type: {self.type} ({'hexadecimal' if hex_format else 'binary'})\n"
        lines += f"Cipher: {self.data['cipher']}\n"

        if show_mode == 0:
            lines += "Show Mode: First Layer of First and Last Round.\n"
        elif show_mode == 1:
            lines += "Show Mode: First Layer of All Rounds (layer 0)\n"
        elif show_mode == 2:
            lines += "Show Mode: All Layers of All Rounds\n"
        elif show_mode == 3:
            lines += "Show Mode: All Layers of All Rounds (Including Temporary Words)\n"
        else:
            lines += f"[ERROR] Invalid show_mode {show_mode}. Cannot format the trail.\n"
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
                return "[WARNING] trail_struct is not a dictionary. Cannot format the trail structure.\n"

            for key in ("inputs", "functions", "outputs"):
                if key in trail_struct and not isinstance(trail_struct[key], dict):
                    return f"[WARNING] trail_struct['{key}'] is not a dictionary.\n"

            if "functions" not in trail_struct:
                return "[WARNING] trail_struct does not contain 'functions'. Cannot format the trail structure.\n"

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
    """A differential trail.

    Args:
        data (dict): In addition to the base keys, may contain ``"diff_weight"``
            (float, int, or None; the trail weight, i.e. the negative base-2
            logarithm of the differential probability, e.g. ``2``),
            ``"rounds_diff_weight"`` (list of float or None; per-round weights,
            e.g. ``[0, 1, 1]``), and ``"trail_struct"`` (dict; the trail structure).
        solution_trace (dict, optional): See :class:`AttackTrace`. Defaults to None.
    """

    def __init__(self, data, solution_trace=None):
        super().__init__("differential", data, solution_trace=solution_trace)


    def format_trail(self, show_mode=2, hex_format=True):
        lines = super().format_trail(show_mode, hex_format=hex_format)

        if "diff_weight" in self.data and self.data["diff_weight"] is not None:
            lines += f"\nTotal Weight: {self.data['diff_weight']}\n"
        if "rounds_diff_weight" in self.data and self.data["rounds_diff_weight"] is not None:
            lines += f"rounds_diff_weight: {self.data['rounds_diff_weight']}\n"
        return lines


class LinearTrail(Trail):
    """A linear trail.

    Args:
        data (dict): In addition to the base keys, may contain ``"linear_weight"``
            (float, int, or None; the trail weight, i.e. the negative base-2
            logarithm of the linear correlation, e.g. ``2``),
            ``"rounds_linear_weight"`` (list of float or None; per-round weights,
            e.g. ``[0, 1, 1]``), and ``"trail_struct"`` (dict; the trail structure).
        solution_trace (dict, optional): See :class:`AttackTrace`. Defaults to None.
    """

    def __init__(self, data, solution_trace=None):
        super().__init__("linear", data, solution_trace=solution_trace)


    def format_trail(self, show_mode=2, hex_format=True):
        lines = super().format_trail(show_mode, hex_format=hex_format)

        if "linear_weight" in self.data and self.data["linear_weight"] is not None:
            lines += f"\nTotal Weight: {self.data['linear_weight']}\n"
        if "rounds_linear_weight" in self.data and self.data["rounds_linear_weight"] is not None:
            lines += f"rounds_linear_weight: {self.data['rounds_linear_weight']}\n"
        return lines


class IntegralDistinguisher(AttackTrace):
    """An integral (division-property) distinguisher result.

    Args:
        data (dict): In addition to the base keys, may contain ``"goal"``,
            ``"status"``, ``"balanced_bits"`` (list), ``"config_model"``, and
            ``"config_solver"``.
        solution_trace (dict, optional): See :class:`AttackTrace`. Defaults to None.
    """

    def __init__(self, data, solution_trace=None):
        super().__init__("integral", data, solution_trace=solution_trace)
        self._set_output_filenames("distinguisher")

    def format_distinguisher(self):
        """Return the distinguisher as a human-readable string.

        Returns:
            str: The formatted distinguisher.
        """
        lines = []
        lines.append("========== Integral Distinguisher ==========")
        lines.append(f"Cipher: {self.data['cipher']}")
        lines.append(f"Goal: {self.data.get('goal')}")
        lines.append(f"Status: {self.data.get('status')}")
        lines.append(f"Balanced bits: {self.data.get('balanced_bits', [])}")
        lines.append(f"Model file: {self.data.get('config_model', {}).get('filename')}")
        lines.append("")
        return "\n".join(lines)

    def print_distinguisher(self):
        """Format the distinguisher and print it to stdout."""
        print(self.format_distinguisher())

    def save_json(self):
        Path(self.json_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.json_filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent='\t')
    
    def save_txt(self):
        text = self.format_distinguisher()
        Path(self.txt_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(self.txt_filename, "w", encoding="utf-8") as f:
            f.write(text)

    def save_tex(self): # TO DO
        raise NotImplementedError("LaTeX export is not implemented yet.")

    def save_pdf(self): # TO DO
        raise NotImplementedError("PDF export is not implemented yet.")


# -------------------- Trail extraction from solver assignments -------------------- #
def solution_bit(solution, var_id):
    """Map a solver value to '0', '1', or '-'."""

    value = solution.get(var_id, None)
    if value is None:
        return "-"
    try:
        return "1" if int(round(value)) == 1 else "0"
    except (TypeError, ValueError, OverflowError):
        return "-"


def extract_trail_structures(cipher, goal, solution, truncated_marker):
    """Extract a structured trail from a solver assignment."""

    bitwise = truncated_marker not in goal

    def node(var):
        ids = expand_var_ids(var, bitwise=bitwise)
        bits = "".join(solution_bit(solution, var_id) for var_id in ids)
        return {
            "var_ID": getattr(var, "ID", str(var)),
            "variables": ids,
            "bin_values": bits,
        }

    trail_struct = {
        "bitwise": bitwise,
        "inputs": {},
        "outputs": {},
        "functions": {},
    }

    if hasattr(cipher, "inputs") and isinstance(cipher.inputs, dict):
        for name, var_list in cipher.inputs.items():
            trail_struct["inputs"][name] = [node(v) for v in var_list]
    if hasattr(cipher, "outputs") and isinstance(cipher.outputs, dict):
        for name, var_list in cipher.outputs.items():
            trail_struct["outputs"][name] = [node(v) for v in var_list]

    for fun in cipher.functions:
        cipher_function = cipher.functions[fun]
        fun_store = {
            "rounds": list(range(1, cipher_function.nbr_rounds + 1)),
            "nbr_words": cipher_function.nbr_words if hasattr(cipher_function, "nbr_words") else None,
            "nbr_temp_words": cipher_function.nbr_temp_words if hasattr(cipher_function, "nbr_temp_words") else None,
        }
        for round_index in range(1, cipher_function.nbr_rounds + 1):
            round_store = {}
            for layer_index in range(cipher_function.nbr_layers + 1):
                round_store[layer_index] = [node(v) for v in cipher_function.vars[round_index][layer_index]]
            fun_store[round_index] = round_store
        trail_struct["functions"][fun] = fun_store
    return trail_struct


def extract_and_format_trails(
    cipher,
    goal,
    config_model,
    config_solver,
    show_mode,
    solutions,
    trail_class,
    truncated_marker,
    weight_key,
    rounds_weight_key,
):
    """Build each distinct trail and immediately save it; return the deduplicated list.

    The goal-specific aggregate (differential total probability, linear ELP, ...) is
    computed by the caller from the returned trails.
    """

    trails = []
    trail_structs = []
    for i, solution in enumerate(solutions):
        trail_struct = extract_trail_structures(cipher, goal, solution, truncated_marker)
        if trail_struct in trail_structs:
            continue
        trail_structs.append(trail_struct)
        data = {
            "cipher": f"{cipher.nbr_rounds}_round_{cipher.name}",
            "functions": config_model["functions"],
            "rounds": config_model["rounds"],
            "config_model": config_model,
            "config_solver": config_solver,
            "trail_struct": trail_struct,
            weight_key: solution.get("obj_fun_value"),
            rounds_weight_key: solution.get("rounds_obj_fun_values"),
            "integer_obj_fun_value": solution.get("integer_obj_fun_value"),
        }
        trail = trail_class(data, solution_trace=solution)
        if i > 0:
            print(f"[INFO] Saving the {i+1}-th Trail.")
            trail.json_filename = (
                trail.json_filename.replace(".json", f"_{i}.json")
                if trail.json_filename
                else str(get_files_dir() / f"{trail.data['cipher']}_trail_{i}.json")
            )
            trail.txt_filename = (
                trail.txt_filename.replace(".txt", f"_{i}.txt")
                if trail.txt_filename
                else str(get_files_dir() / f"{trail.data['cipher']}_trail_{i}.txt")
            )
        trail.print_trail(show_mode=show_mode)
        trail.save_json()
        trail.save_txt(show_mode=show_mode)
        trails.append(trail)
    return trails
