from primitives.primitives import Permutation, Block_cipher, generateID
from operators.Sbox import LBlock_Sbox0, LBlock_Sbox1, LBlock_Sbox2, LBlock_Sbox3, LBlock_Sbox4, LBlock_Sbox5, LBlock_Sbox6, LBlock_Sbox7, LBlock_Sbox8, LBlock_Sbox9
from operators.boolean_operators import XOR
import operators.operators as op
import variables.variables as var


# The LBlock internal permutation
class LBlock_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the LBlock internal permutation.
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        if nbr_rounds==None: nbr_rounds=32
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (6, 64, 32, 1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]

        sboxes = [LBlock_Sbox7, LBlock_Sbox6, LBlock_Sbox5, LBlock_Sbox4, LBlock_Sbox3, LBlock_Sbox2, LBlock_Sbox1, LBlock_Sbox0]
        copy_x = [i for i in range(64)] + [i for i in range(32)]
        nibble_perm = [1, 3, 0, 2, 5, 7, 4, 6]
        p_layer = [4*nibble_perm[i//4] + i%4 if i < 32 else i for i in range(96)]
        rotate_y = [i for i in range(32)] + [32 + 4*((i//4 + 2)%8) + i%4 for i in range(32)] + [i for i in range(64, 96)]
        feistel_swap = [i for i in range(32)] + [64+i for i in range(32)] + [i for i in range(64, 96)]

        # create constraints
        for i in range(1,nbr_rounds+1):
            S.PermutationLayer("COPY_X", i, 0, copy_x)
            for nibble, sbox in enumerate(sboxes):
                indexes = [4*nibble+j for j in range(4)]
                in_vars = [S.vars[i][1][index] for index in indexes]
                out_vars = [S.vars[i][2][index] for index in indexes]
                S.constraints[i][1].append(sbox(in_vars, out_vars, ID=generateID("SB", i, 2, nibble)))
            for j in range(32, S.nbr_words + S.nbr_temp_words):
                S.constraints[i][1].append(op.Equal([S.vars[i][1][j]], [S.vars[i][2][j]], ID=generateID("SB_EQ", i, 2, j)))
            S.PermutationLayer("P", i, 2, p_layer)
            S.PermutationLayer("ROT", i, 3, rotate_y)
            S.SingleOperatorLayer("XOR", i, 4, XOR, [[j, 32+j] for j in range(32)], [j for j in range(32)])
            S.PermutationLayer("SWAP", i, 5, feistel_swap)

    def gen_test_vectors(self):
        if self.nbr_rounds == 32:
            plaintext = [0] * 64
            ciphertext = [int(bit) for bit in f"{0x405ffcf045c8c91a:064b}"]
            self.test_vectors.append([[plaintext], ciphertext])


def LBLOCK_PERMUTATION(r=None, represent_mode=0, copy_operator=False):
    my_input, my_output = [var.Variable(1,ID="in"+str(i)) for i in range(64)], [var.Variable(1,ID="out"+str(i)) for i in range(64)]
    my_permutation = LBlock_permutation("LBLOCK_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors()
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


# The LBlock block cipher
class LBlock_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the LBlock block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        assert version == [64,80], f"Unsupported version: {version}."
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (8, p_bitsize, 32, 1),  (3, k_bitsize, 0, 1),  (1, 32, 0, 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        constant_table = self.gen_rounds_constant_table()
        sboxes = [LBlock_Sbox7, LBlock_Sbox6, LBlock_Sbox5, LBlock_Sbox4, LBlock_Sbox3, LBlock_Sbox2, LBlock_Sbox1, LBlock_Sbox0]
        copy_x = [i for i in range(64)] + [i for i in range(32)]
        nibble_perm = [1, 3, 0, 2, 5, 7, 4, 6]
        p_layer = [4*nibble_perm[i//4] + i%4 if i < 32 else i for i in range(96)]
        rotate_y = [i for i in range(32)] + [32 + 4*((i//4 + 2)%8) + i%4 for i in range(32)] + [i for i in range(64, 96)]
        feistel_swap = [i for i in range(32)] + [64+i for i in range(32)] + [i for i in range(64, 96)]
        output_swap = [32+i for i in range(32)] + [i for i in range(32)] + [i for i in range(64, 96)]
        key_perm = [(29+j)%80 for j in range(80)]
        key_constant_mask = [True if 29 <= j <= 33 else None for j in range(80)]

        # subkeys extraction
        for i in range(1,SK.nbr_rounds+1):
            SK.ExtractionLayer("SK_EX", i, 0, [j for j in range(32)], KS.vars[i][0])

        # key schedule
        for i in range(1,KS.nbr_rounds):
            KS.PermutationLayer("K_PERM", i, 0, key_perm)
            KS.constraints[i][1].append(LBlock_Sbox9([KS.vars[i][1][j] for j in range(4)], [KS.vars[i][2][j] for j in range(4)], ID=generateID("K_SB", i, 2, 0)))
            KS.constraints[i][1].append(LBlock_Sbox8([KS.vars[i][1][4+j] for j in range(4)], [KS.vars[i][2][4+j] for j in range(4)], ID=generateID("K_SB", i, 2, 1)))
            for j in range(8, KS.nbr_words + KS.nbr_temp_words):
                KS.constraints[i][1].append(op.Equal([KS.vars[i][1][j]], [KS.vars[i][2][j]], ID=generateID("K_SB_EQ", i, 2, j)))
            KS.AddConstantLayer("K_C", i, 2, "xor", key_constant_mask, constant_table)

        # Internal permutation
        for i in range(1,S.nbr_rounds+1):
            S.PermutationLayer("COPY_X", i, 0, copy_x)
            S.AddRoundKeyLayer("ARK", i, 1, XOR, SK, [1]*32)
            for nibble, sbox in enumerate(sboxes):
                indexes = [4*nibble+j for j in range(4)]
                in_vars = [S.vars[i][2][index] for index in indexes]
                out_vars = [S.vars[i][3][index] for index in indexes]
                S.constraints[i][2].append(sbox(in_vars, out_vars, ID=generateID("SB", i, 3, nibble)))
            for j in range(32, S.nbr_words + S.nbr_temp_words):
                S.constraints[i][2].append(op.Equal([S.vars[i][2][j]], [S.vars[i][3][j]], ID=generateID("SB_EQ", i, 3, j)))
            S.PermutationLayer("P", i, 3, p_layer)
            S.PermutationLayer("ROT", i, 4, rotate_y)
            S.SingleOperatorLayer("XOR", i, 5, XOR, [[j, 32+j] for j in range(32)], [j for j in range(32)])
            S.PermutationLayer("SWAP", i, 6, feistel_swap)
            if i < S.nbr_rounds:
                S.AddIdentityLayer("OUT_ID", i, 7)
            else:
                S.PermutationLayer("OUT_SWAP", i, 7, output_swap)

    def gen_rounds_constant_table(self):
        return [[(i >> j) & 1 for j in reversed(range(5))] for i in range(1, self.functions["KEY_SCHEDULE"].nbr_rounds)]

    def gen_test_vectors(self, version): # Test vectors from Appendix I of the LBlock specification.
        if version == [64, 80]:
            plaintext = [int(bit) for bit in f"{0x0000000000000000:064b}"]
            key = [int(bit) for bit in f"{0x00000000000000000000:080b}"]
            ciphertext = [int(bit) for bit in f"{0xc218185308e75bcd:064b}"]
            self.test_vectors.append([[plaintext, key], ciphertext])

            plaintext = [int(bit) for bit in f"{0x0123456789abcdef:064b}"]
            key = [int(bit) for bit in f"{0x0123456789abcdeffedc:080b}"]
            ciphertext = [int(bit) for bit in f"{0x4b7179d8ebee0c26:064b}"]
            self.test_vectors.append([[plaintext, key], ciphertext])


def LBLOCK_BLOCKCIPHER(r=None, version = [64,80], represent_mode=0, copy_operator=False):
    p_bitsize, k_bitsize = version[0], version[1]
    my_plaintext, my_key, my_ciphertext = [var.Variable(1,ID="in"+str(i)) for i in range(p_bitsize)], [var.Variable(1,ID="k"+str(i)) for i in range(k_bitsize)], [var.Variable(1,ID="out"+str(i)) for i in range(p_bitsize)]
    my_cipher = LBlock_block_cipher(f"LBLOCK{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher
