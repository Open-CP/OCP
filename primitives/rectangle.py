from primitives.primitives import Permutation, Block_cipher, generateID
from operators.Sbox import RECTANGLE_Sbox
from operators.boolean_operators import XOR
import operators.operators as op
import variables.variables as var


# The RECTANGLE internal permutation
class RECTANGLE_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the RECTANGLE internal permutation.
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        if nbr_rounds==None: nbr_rounds=25
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (2, 64, 0, 1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]
        sbox_indexes = [[48+j, 32+j, 16+j, j] for j in range(16)]
        shift_rows = [j for j in range(16)] + [16 + ((j-1)%16) for j in range(16)] + [32 + ((j-12)%16) for j in range(16)] + [48 + ((j-13)%16) for j in range(16)]

        # create constraints
        for i in range(1,nbr_rounds+1):
            S.SboxLayer("SB", i, 0, RECTANGLE_Sbox, index=sbox_indexes)  # Sbox layer
            S.PermutationLayer("SR", i, 1, shift_rows) # ShiftRow layer

    def gen_test_vectors(self):
        if self.nbr_rounds == 25:
            plaintext = [0] * 64
            ciphertext = [int(bit) for row in [
                "0000000000000000",
                "1111111111111111",
                "1111111111111111",
                "0000000000000000",
            ] for bit in reversed(row)]
            self.test_vectors.append([[plaintext], ciphertext])


def RECTANGLE_PERMUTATION(r=None, represent_mode=0, copy_operator=False):
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(64)], [var.Variable(1,ID="out"+str(i)) for i in range(64)]
    my_permutation = RECTANGLE_permutation("RECTANGLE_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors()
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


# The RECTANGLE block cipher
class RECTANGLE_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the RECTANGLE block cipher.
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
        if nbr_rounds==None or nbr_rounds==25:
            nbr_rounds=26
            print(f"[INFO] For RECTANGLE, after 25 round transformations, there is still a final AddRoundKey layer. Hence, the internal modeling round number is set to {nbr_rounds}. Please keep this in mind when interpreting subsequent relative files.")
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (3, p_bitsize, 0, 1),  (3, k_bitsize, 0, 1),  (1, p_bitsize, 0, 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        constant_table = self.gen_rounds_constant_table()
        shift_rows = [j for j in range(16)] + [16 + ((j-1)%16) for j in range(16)] + [32 + ((j-12)%16) for j in range(16)] + [48 + ((j-13)%16) for j in range(16)]
        state_sbox_indexes = [[48+j, 32+j, 16+j, j] for j in range(16)]

        # subkeys extraction
        for i in range(1,SK.nbr_rounds+1):
            if k_bitsize == 80:
                SK.ExtractionLayer("SK_EX", i, 0, [j for j in range(64)], KS.vars[i][0])
            elif k_bitsize == 128:
                SK.ExtractionLayer("SK_EX", i, 0, list(range(16)) + list(range(32,48)) + list(range(64,80)) + list(range(96,112)), KS.vars[i][0])

        # key schedule
        if k_bitsize == 80:
            key_sbox_indexes = [[48+j, 32+j, 16+j, j] for j in range(4)]
            key_sbox_bits = {index for indexes in key_sbox_indexes for index in indexes}
            for i in range(1,KS.nbr_rounds):
                for sbox_pos, indexes in enumerate(key_sbox_indexes):
                    in_vars = [KS.vars[i][0][index] for index in indexes]
                    out_vars = [KS.vars[i][1][index] for index in indexes]
                    KS.constraints[i][0].append(RECTANGLE_Sbox(in_vars, out_vars, ID=generateID("K_SB", i, 1, sbox_pos)))
                for j in range(KS.nbr_words + KS.nbr_temp_words):
                    if j not in key_sbox_bits:
                        KS.constraints[i][0].append(op.Equal([KS.vars[i][0][j]], [KS.vars[i][1][j]], ID=generateID("K_SB_EQ", i, 1, j)))
                for j in range(16):
                    KS.constraints[i][1].append(XOR([KS.vars[i][1][(j-8)%16], KS.vars[i][1][16+j]], [KS.vars[i][2][j]], ID=generateID("K_F_ROW0", i, 2, j)))
                    KS.constraints[i][1].append(op.Equal([KS.vars[i][1][32+j]], [KS.vars[i][2][16+j]], ID=generateID("K_F_ROW1_EQ", i, 2, j)))
                    KS.constraints[i][1].append(op.Equal([KS.vars[i][1][48+j]], [KS.vars[i][2][32+j]], ID=generateID("K_F_ROW2_EQ", i, 2, j)))
                    KS.constraints[i][1].append(XOR([KS.vars[i][1][48+((j-12)%16)], KS.vars[i][1][64+j]], [KS.vars[i][2][48+j]], ID=generateID("K_F_ROW3", i, 2, j)))
                    KS.constraints[i][1].append(op.Equal([KS.vars[i][1][j]], [KS.vars[i][2][64+j]], ID=generateID("K_F_ROW4_EQ", i, 2, j)))
                KS.AddConstantLayer("K_C", i, 2, "xor", [True]*5, constant_table)
        elif k_bitsize == 128:
            key_sbox_indexes = [[96+j, 64+j, 32+j, j] for j in range(8)]
            key_sbox_bits = {index for indexes in key_sbox_indexes for index in indexes}
            for i in range(1,KS.nbr_rounds):
                for sbox_pos, indexes in enumerate(key_sbox_indexes):
                    in_vars = [KS.vars[i][0][index] for index in indexes]
                    out_vars = [KS.vars[i][1][index] for index in indexes]
                    KS.constraints[i][0].append(RECTANGLE_Sbox(in_vars, out_vars, ID=generateID("K_SB", i, 1, sbox_pos)))
                for j in range(KS.nbr_words + KS.nbr_temp_words):
                    if j not in key_sbox_bits:
                        KS.constraints[i][0].append(op.Equal([KS.vars[i][0][j]], [KS.vars[i][1][j]], ID=generateID("K_SB_EQ", i, 1, j)))
                for j in range(32):
                    KS.constraints[i][1].append(XOR([KS.vars[i][1][(j-8)%32], KS.vars[i][1][32+j]], [KS.vars[i][2][j]], ID=generateID("K_F_ROW0", i, 2, j)))
                    KS.constraints[i][1].append(op.Equal([KS.vars[i][1][64+j]], [KS.vars[i][2][32+j]], ID=generateID("K_F_ROW1_EQ", i, 2, j)))
                    KS.constraints[i][1].append(XOR([KS.vars[i][1][64+((j-16)%32)], KS.vars[i][1][96+j]], [KS.vars[i][2][64+j]], ID=generateID("K_F_ROW2", i, 2, j)))
                    KS.constraints[i][1].append(op.Equal([KS.vars[i][1][j]], [KS.vars[i][2][96+j]], ID=generateID("K_F_ROW3_EQ", i, 2, j)))
                KS.AddConstantLayer("K_C", i, 2, "xor", [True]*5, constant_table)

        # Internal permutation
        for i in range(1,S.nbr_rounds+1):
            S.AddRoundKeyLayer("ARK", i, 0, XOR, SK, [1]*64) # Addroundkey layer
            if i < 26:
                S.SboxLayer("SB", i, 1, RECTANGLE_Sbox, index=state_sbox_indexes)  # Sbox layer
                S.PermutationLayer("SR", i, 2, shift_rows) # ShiftRow layer
            elif i == 26:
                S.AddIdentityLayer("ID", i, 1)
                S.AddIdentityLayer("ID", i, 2)

    def gen_rounds_constant_table(self):
        constant_table = []
        rc = 0x01
        for _ in range(25):
            constant_table.append([(rc >> i) & 1 for i in range(5)])
            rc_bits = [(rc >> i) & 1 for i in range(5)]
            rc = ((rc << 1) & 0x1E) | (rc_bits[4] ^ rc_bits[2])
        return constant_table

    def gen_test_vectors(self, version): # Test vectors from Table 10 of the RECTANGLE specification.
        if version == [64, 80]:
            plaintext = [0] * 64
            key = [0] * 80
            ciphertext = [int(bit) for row in [
                "0010110110010110",
                "1110001101010100",
                "1110100010110001",
                "0000100001110100",
            ] for bit in reversed(row)]
            self.test_vectors.append([[plaintext, key], ciphertext])

            plaintext = [1] * 64
            key = [1] * 80
            ciphertext = [int(bit) for row in [
                "1001100101000101",
                "1010101000110100",
                "1010111000111101",
                "0000000100010010",
            ] for bit in reversed(row)]
            self.test_vectors.append([[plaintext, key], ciphertext])

        elif version == [64, 128]:
            plaintext = [0] * 64
            key = [0] * 128
            ciphertext = [int(bit) for row in [
                "1010111011100110",
                "0011011000010011",
                "0100010010100100",
                "1001100111101110",
            ] for bit in reversed(row)]
            self.test_vectors.append([[plaintext, key], ciphertext])

            plaintext = [1] * 64
            key = [1] * 128
            ciphertext = [int(bit) for row in [
                "1110100000111110",
                "1110111111101110",
                "0100101000010101",
                "0111101001000110",
            ] for bit in reversed(row)]
            self.test_vectors.append([[plaintext, key], ciphertext])


def RECTANGLE_BLOCKCIPHER(r=None, version = [64, 80], represent_mode=0, copy_operator=False):
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = RECTANGLE_block_cipher(f"RECTANGLE{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher
