"""KNOT: the KNOT permutation."""

from primitives.primitives import Permutation
from operators.Sbox import KNOT_Sbox
import variables.variables as var


# Per-version parameters keyed by (state size, mode). nbr_rounds is the design's full rounds.
KNOT_PERMUTATION_VERSIONS = {
    (256, 'AEAD'): {'rows': 4, 'cols': 64, 'nbr_rounds': 52, 'd': 6, 'lfsr': {'width': 6, 'taps': [5, 4], 'init': 1}, 'offsets': [0, 1, 8, 25], 'direction': 'l'},
    (384, 'AEAD'): {'rows': 4, 'cols': 96, 'nbr_rounds': 76, 'd': 7, 'lfsr': {'width': 7, 'taps': [6, 5], 'init': 1}, 'offsets': [0, 1, 8, 55], 'direction': 'l'},
    (512, 'AEAD'): {'rows': 4, 'cols': 128, 'nbr_rounds': 100, 'd': 7, 'lfsr': {'width': 7, 'taps': [6, 5], 'init': 1}, 'offsets': [0, 1, 16, 25], 'direction': 'l'},
    (256, 'Hash'): {'rows': 4, 'cols': 64, 'nbr_rounds': 68, 'd': 7, 'lfsr': {'width': 7, 'taps': [6, 5], 'init': 1}, 'offsets': [0, 1, 8, 25], 'direction': 'l'},
    (384, 'Hash'): {'rows': 4, 'cols': 96, 'nbr_rounds': 104, 'd': 7, 'lfsr': {'width': 7, 'taps': [6, 5], 'init': 1}, 'offsets': [0, 1, 8, 55], 'direction': 'l'},
    (512, 'Hash'): {'rows': 4, 'cols': 128, 'nbr_rounds': 140, 'd': 8, 'lfsr': {'width': 8, 'taps': [7, 5, 4, 3], 'init': 1}, 'offsets': [0, 1, 16, 25], 'direction': 'l'},
}
KNOT_PERMUTATION_DEFAULT_VERSION = (256, 'AEAD')


# The KNOT internal permutation
def knot_version_params(version):
    """The table entry for ``version``, normalising a list to a tuple; raises if there is none."""
    version = tuple(version) if isinstance(version, list) else version
    if version not in KNOT_PERMUTATION_VERSIONS:
        raise ValueError(f"[ERROR] Unknown version {version} for KNOT_permutation. Expected one of "
                         f"{[list(v) for v in sorted(KNOT_PERMUTATION_VERSIONS)]}.")
    return version, KNOT_PERMUTATION_VERSIONS[version]


class KNOT_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the KNOT internal permutation.

        Args:
            name: Name of the permutation
            version: (width, mode) pair, e.g. (256, 'AEAD')
            s_input: Input state
            s_output: Output state
            nbr_rounds: Number of rounds
            represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """
        if represent_mode != 0: raise ValueError(f"[ERROR] The represent_mode {represent_mode} is not supported for KNOT_permutation.")
        version, params = knot_version_params(version)
        rows, cols, d, lfsr, offsets, direction = params["rows"], params["cols"], params["d"], params["lfsr"], params["offsets"], params["direction"]
        total = rows * cols
        if nbr_rounds is None: nbr_rounds = params["nbr_rounds"]
        super().__init__(name, s_input, s_output, nbr_rounds, [3, total, 0, 1])
        S = self.functions["PERMUTATION"]

        sbox_index = [[cols*r + c for r in reversed(range(rows))] for c in range(cols)]
        shift_table = [0]*total
        for r in range(rows):
            off = offsets[r] % cols
            for c in range(cols):
                src = (c - off) % cols if direction == "l" else (c + off) % cols
                shift_table[cols*r + c] = cols*r + src
        rc_mask = [1]*d + [None]*(total - d)
        constant_table = self.gen_rounds_constant_table(d, lfsr, nbr_rounds)

        # create constraints
        for i in range(1, nbr_rounds + 1):
            S.AddConstantLayer("C", i, 0, "xor", rc_mask, constant_table) # AddConstant layer
            S.SboxLayer("SB", i, 1, KNOT_Sbox, index=sbox_index) # Sbox layer
            S.PermutationLayer("P", i, 2, shift_table) # ShiftRow layer

    def gen_test_vectors(self, rounds=None, version=None):
        version = tuple(version) if isinstance(version, list) else version
        if version == (256, 'AEAD') and (rounds == 52 or rounds is None):
            plaintext = [0] * 256
            ciphertext = [int(b) for row in [
                "0100000010010010110010110000000010001101111111111011001011011001",
                "0010011010010011110110010111100011101111101011111111110000110010",
                "1010111111010010101011011000100011100000111001010011110011100000",
                "1010100111011000000101010111011101111010111011010010001000111101",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])
        elif version == (384, 'AEAD') and (rounds == 76 or rounds is None):
            plaintext = [0] * 384
            ciphertext = [int(b) for row in [
                "111101001011001000110101001010010100010111011111011000000100111101100010110110110010110110000111",
                "010011110110111100101110011110110010010000100000110110101001000001110100100101000000111000100101",
                "111101001100110100110101100001000011010100110100100011101010011101111100000100001001101011110111",
                "000001111110101011000001001101111101011101010000101000001011101000100011000010011111111011011001",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])
        elif version == (512, 'AEAD') and (rounds == 100 or rounds is None):
            plaintext = [0] * 512
            ciphertext = [int(b) for row in [
                "11010011010000101000100111101010110100100101101101111110011011010000001101011011100111100010010011010001110010101001111101111101",
                "10100110011101011000001110000000110100101011111110111100011110010110101110100100010111011111100101111001110101100011110010011110",
                "10101111101111010000000011011010000101000100001101001111110110101010001010100111000111001101001000001100000001101000001110100100",
                "10110100010010100101010101111100110000011011101100111011101100001100010011001100011100110100101101101110011111000110000110101100",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])
        elif version == (256, 'Hash') and (rounds == 68 or rounds is None):
            plaintext = [0] * 256
            ciphertext = [int(b) for row in [
                "0001111011011010111110101010101000001110000110110001000101100001",
                "1101001101010100111110100110101111100010010111001101001010100010",
                "0011011110001010100000100011011001111101001011100110110110110010",
                "1000101101101000011101011111001010101101010101001000101110001110",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])
        elif version == (384, 'Hash') and (rounds == 104 or rounds is None):
            plaintext = [0] * 384
            ciphertext = [int(b) for row in [
                "111101000001101010000100000101010001110110100000000001011111101110001001010110111101101000100011",
                "001111001010100000011001110101110010111101111010010110100001111101111101000100011000111110001101",
                "111110110000100101101101101011010010001000111111110010100011111111111110100001111110101101110001",
                "001011100100110100100011101000000011011000001001010000000011001101010111011000111000001111011101",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])
        elif version == (512, 'Hash') and (rounds == 140 or rounds is None):
            plaintext = [0] * 512
            ciphertext = [int(b) for row in [
                "11110100101111000000011100011001100011010100010111011111111000000111101011110110100010011011000100110010110110010000011100101011",
                "10110010101111001101010111000010001101010010100111101011001011010111101001101110110100111100101100011001110001001010110101011110",
                "00110100001101001011100101000000101111111010101011111001111111001110100000001011011100010111100011001010110111000010110000100101",
                "01101100001111101000000101001101111110100111111110110110001001001011011011000101011100000111000100101101011001010000101001100010",
            ] for b in row]
            self.test_vectors.append([[plaintext], ciphertext])

    def gen_rounds_constant_table(self, d, lfsr, nbr_rounds):
        # Fibonacci LFSR {width, taps, init}: each step left-shifts and feeds back the XOR of the tap bits.
        width, taps, state = lfsr["width"], lfsr["taps"], lfsr.get("init", 1)
        mask, constant_table = (1 << width) - 1, []
        for _ in range(nbr_rounds):
            constant_table.append([(state >> k) & 1 for k in range(d)])
            feedback = 0
            for tap in taps:
                feedback ^= (state >> tap) & 1
            state = ((state << 1) | feedback) & mask
        return constant_table


def KNOT_PERMUTATION(r=None, version=None, represent_mode=0, copy_operator=False):
    if version is None: version = KNOT_PERMUTATION_DEFAULT_VERSION
    version, params = knot_version_params(version)
    n = params["rows"] * params["cols"]
    my_input, my_output = [var.Variable(1, ID="in" + str(i)) for i in range(n)], [var.Variable(1, ID="out" + str(i)) for i in range(n)]
    my_permutation = KNOT_permutation(f"KNOT{version[0]}_{version[1]}_PERM", version, my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors(version=version, rounds=r)
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation
