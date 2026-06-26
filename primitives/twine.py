from primitives.primitives import Permutation, Block_cipher, generateID
from operators.Sbox import TWINE_Sbox
from operators.boolean_operators import XOR
import operators.operators as op
import variables.variables as var



# The TWINE internal permutation
class TWINE_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the TWINE internal permutation.
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        if nbr_rounds==None: nbr_rounds=36
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (4, 16, 8, 4)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]

        pi = [5, 0, 1, 4, 7, 12, 3, 8, 13, 6, 9, 2, 15, 10, 11, 14]
        even_nibbles = [2*j for j in range(8)]
        copy_even = [j for j in range(16)] + even_nibbles
        shuffle = [pi.index(j) for j in range(16)] + [j for j in range(16, 24)]

        # create constraints
        for i in range(1,nbr_rounds+1):
            S.PermutationLayer("COPY_EVEN", i, 0, copy_even)
            for j in range(8):
                S.constraints[i][1].append(TWINE_Sbox([S.vars[i][1][16+j]], [S.vars[i][2][16+j]], ID=generateID("SB", i, 2, j)))
            for j in range(16):
                S.constraints[i][1].append(op.Equal([S.vars[i][1][j]], [S.vars[i][2][j]], ID=generateID("SB_EQ", i, 2, j)))
            S.SingleOperatorLayer("XOR", i, 2, XOR, [[2*j+1, 16+j] for j in range(8)], [2*j+1 for j in range(8)])
            S.PermutationLayer("SHUFFLE", i, 3, shuffle)

    def gen_test_vectors(self):
        if self.nbr_rounds == 36:
            plaintext = [0] * 16
            ciphertext = [0xC, 0x0, 0xC, 0x0, 0xC, 0x0, 0xC, 0x0, 0xC, 0x0, 0xC, 0x0, 0xC, 0x0, 0xC, 0x0]
            self.test_vectors.append([[plaintext], ciphertext])


def TWINE_PERMUTATION(r=None, represent_mode=0, copy_operator=False):
    my_input, my_output = [var.Variable(4,ID="in"+str(i)) for i in range(16)], [var.Variable(4,ID="out"+str(i)) for i in range(16)]
    my_permutation = TWINE_permutation("TWINE_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors()
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


# The TWINE block cipher
class TWINE_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the TWINE block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        assert version in [[64,80], [64,128]], f"Unsupported version: {version}."
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=36
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 8, 4),  (5, k_bitsize//4, 2 if k_bitsize == 80 else 3, 4),  (1, 8, 0, 4)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        pi = [5, 0, 1, 4, 7, 12, 3, 8, 13, 6, 9, 2, 15, 10, 11, 14]
        even_nibbles = [2*j for j in range(8)]
        odd_nibbles = [2*j+1 for j in range(8)]
        copy_even = [j for j in range(16)] + even_nibbles
        shuffle = [pi.index(j) for j in range(16)] + [j for j in range(16, 24)]
        constant_table = self.gen_rounds_constant_table()
        subkey_indexes = [1, 3, 4, 6, 13, 14, 15, 16] if k_bitsize == 80 else [2, 3, 12, 15, 17, 18, 28, 31]
        key_shuffle = [j+4 for j in range(k_bitsize//4-4)] + [1, 2, 3, 0]

        # subkeys extraction
        for i in range(1,SK.nbr_rounds+1):
            SK.ExtractionLayer("SK_EX", i, 0, subkey_indexes, KS.vars[i][0])

        # key schedule
        for i in range(1,KS.nbr_rounds):
            pairs = [(0, 1), (16, 4)] if k_bitsize == 80 else [(0, 1), (16, 4), (30, 23)]
            output_indexes = {out_index for _, out_index in pairs}
            temp_base = KS.nbr_words
            for temp_index, (in_index, out_index) in enumerate(pairs):
                KS.constraints[i][0].append(TWINE_Sbox([KS.vars[i][0][in_index]], [KS.vars[i][1][temp_base+temp_index]], ID=generateID("K_SB", i, 1, out_index)))
                KS.constraints[i][0].append(XOR([KS.vars[i][0][out_index], KS.vars[i][1][temp_base+temp_index]], [KS.vars[i][1][out_index]], ID=generateID("K_XOR", i, 1, out_index)))
            for j in range(KS.nbr_words):
                if j not in output_indexes:
                    KS.constraints[i][0].append(op.Equal([KS.vars[i][0][j]], [KS.vars[i][1][j]], ID=generateID("K_NL_EQ", i, 1, j)))
            KS.AddConstantLayer("K_C", i, 1, "xor", [True if j in [7, 19] else None for j in range(KS.nbr_words + KS.nbr_temp_words)], constant_table)
            KS.PermutationLayer("K_SHUFFLE", i, 2, key_shuffle)
            KS.AddIdentityLayer("K_ID1", i, 3)
            KS.AddIdentityLayer("K_ID2", i, 4)

        # Internal permutation
        for i in range(1,S.nbr_rounds+1):
            S.PermutationLayer("COPY_EVEN", i, 0, copy_even)
            S.AddRoundKeyLayer("ARK", i, 1, XOR, SK, [0]*16+[1]*8)
            for j in range(8):
                S.constraints[i][2].append(TWINE_Sbox([S.vars[i][2][16+j]], [S.vars[i][3][16+j]], ID=generateID("SB", i, 3, j)))
            for j in range(16):
                S.constraints[i][2].append(op.Equal([S.vars[i][2][j]], [S.vars[i][3][j]], ID=generateID("SB_EQ", i, 3, j)))
            S.SingleOperatorLayer("XOR", i, 3, XOR, [[odd_nibbles[j], 16+j] for j in range(8)], odd_nibbles)
            if i < S.nbr_rounds:
                S.PermutationLayer("SHUFFLE", i, 4, shuffle)
            else:
                S.AddIdentityLayer("OUT_ID", i, 4)

    def gen_rounds_constant_table(self):
        con_table = [
            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x03, 0x06, 0x0C, 0x18, 0x30, 0x23,
            0x05, 0x0A, 0x14, 0x28, 0x13, 0x26, 0x0F, 0x1E, 0x3C, 0x3B, 0x35, 0x29,
            0x11, 0x22, 0x07, 0x0E, 0x1C, 0x38, 0x33, 0x25, 0x09, 0x12, 0x24, 0x0B,
        ]
        return [[con >> 3, con & 0x7] for con in con_table]

    def gen_test_vectors(self, version): # Test vectors from Table 11 of the TWINE specification.
        if version == [64, 80]:
            plaintext = [int(nibble, 16) for nibble in "0123456789ABCDEF"]
            key = [int(nibble, 16) for nibble in "00112233445566778899"]
            ciphertext = [int(nibble, 16) for nibble in "7C1F0F80B1DF9C28"]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [64, 128]:
            plaintext = [int(nibble, 16) for nibble in "0123456789ABCDEF"]
            key = [int(nibble, 16) for nibble in "00112233445566778899AABBCCDDEEFF"]
            ciphertext = [int(nibble, 16) for nibble in "979FF9B379B5A9B8"]
            self.test_vectors.append([[plaintext, key], ciphertext])


def TWINE_BLOCKCIPHER(r=None, version = [64,80], represent_mode=0, copy_operator=False):
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(4,ID="in"+str(i)) for i in range(p_bitsize//4)], [var.Variable(4,ID="k"+str(i)) for i in range(k_bitsize//4)], [var.Variable(4,ID="out"+str(i)) for i in range(p_bitsize//4)]
    my_cipher = TWINE_block_cipher(f"TWINE{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher
