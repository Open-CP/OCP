from primitives.primitives import Permutation, Block_cipher
from operators.Sbox import Skinny_4bit_Sbox, Skinny_8bit_Sbox, SKINNY_TTable
from operators.boolean_operators import XOR
import variables.variables as var

# The Skinny internal permutation
class Skinny_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Skinny internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation (e.g., 64 or 128)
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        # define the parameters
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==64 else 40 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (4, 16, 0, int(p_bitsize/16))
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        self.test_vectors = self.gen_test_vectors(version)
        round_constants = self.gen_rounds_constant_table()
        sbox = Skinny_4bit_Sbox if word_bitsize==4 else Skinny_8bit_Sbox

        S = self.functions["PERMUTATION"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):
                S.SboxLayer("SB", i, 0, sbox)
                S.AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer
                S.PermutationLayer("SR", i, 2, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                S.MatrixLayer("MC", i, 3, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer

    def gen_rounds_constant_table(self):
        constant_table = []
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                    0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                    0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                    0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                    0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]

        for i in range(1,self.functions["PERMUTATION"].nbr_rounds+1):
            rc = round_constants[i-1]
            c0, c1, c2 = rc & 0xF, rc >> 4, 0x2
            constant_table.append([c0,c1,c2])
        return constant_table

    def gen_test_vectors(self, version):
        if version == 64:
            IN = [0x0, 0x6, 0x0, 0x3, 0x4, 0xf, 0x9, 0x5, 0x7, 0x7, 0x2, 0x4, 0xd, 0x1, 0x9, 0xd]
            OUT = [0x2, 0xe, 0x4, 0xc, 0xa, 0xa, 0x8, 0x1, 0xa, 0xc, 0xc, 0x4, 0x8, 0x7, 0x5, 0x6]
        elif version == 128:
            IN = [0x3a, 0x0c, 0x47, 0x76, 0x7a, 0x26, 0xa6, 0x8d, 0xd3, 0x82, 0xa6, 0x95, 0xe7, 0x02, 0x2e, 0x25]
            OUT = [0x34, 0x45, 0x55, 0x36, 0x13, 0xf2, 0xae, 0x5, 0x72, 0xa4, 0x9f, 0xe3, 0x3c, 0x90, 0xbe, 0xf9]
        return [[IN], OUT]


def SKINNY_PERMUTATION(r=None, version=64, represent_mode=0):
    my_input, my_output = [var.Variable(int(version/16),ID="in"+str(i)) for i in range(16)], [var.Variable(int(version/16),ID="out"+str(i)) for i in range(16)]
    my_permutation = Skinny_permutation(f"SKINNY{version}_PERM", version, my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.clean_graph()
    return my_permutation


# The Skinny block cipher
class Skinny_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Skinny block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize), e.g., (64, 128)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        # define the parameters
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(64,64) else 36 if (version[0],version[1])==(64,128) else 40 if (version[0],version[1])==(64,192)  else 40 if (version[0],version[1])==(128,128)  else 48 if (version[0],version[1])==(128,256)  else 56 if (version[0],version[1])==(128,384) else None
        self.tweak_size = int(k_bitsize/p_bitsize)
        k_nbr_rounds = nbr_rounds if self.tweak_size == 1 else nbr_rounds + 1
        if represent_mode==0:
            if self.tweak_size ==1: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, int(p_bitsize/16)), (1, int(16*k_bitsize / p_bitsize), 0, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif self.tweak_size == 2: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, int(p_bitsize/16)), (3, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif self.tweak_size ==3: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (4, 16, 0, int(p_bitsize/16)), (5, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            k_perm_T = [i + 16 * j for j in range(self.tweak_size) for i in [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]]
            if s_word_bitsize == 4:
                mat1 = [[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,0,0]]
                mat2 = [[1,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0]]
            elif s_word_bitsize == 8:
                mat1 = [[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1], [1,0,1,0,0,0,0,0]]
                mat2 = [[0,1,0,0,0,0,0,1],[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0]]
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        self.test_vectors = self.gen_test_vectors(version)
        round_constants = self.gen_rounds_constant_table()
        sbox = Skinny_4bit_Sbox if s_word_bitsize == 4 else Skinny_8bit_Sbox
        if self.tweak_size >= 2: self.functions_implementation_order = ["KEY_SCHEDULE", "SUBKEYS", "PERMUTATION"]

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        # create constraints
        if represent_mode==0:

            # Subkey extraction
            for i in range(1,nbr_rounds+1):
                if self.tweak_size == 1:
                    SK.ExtractionLayer("SK_EX", i, 0, [i for i in range(8)], KS.vars[i][0])
                elif self.tweak_size >= 2:
                    SK.ExtractionLayer("SK_EX", i, 0, [i for i in range(self.tweak_size*16,self.tweak_size*16+8)], KS.vars[i+1][0])

            # Key schedule
            if self.tweak_size == 1:
                for i in range(1, k_nbr_rounds):
                    KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
            elif self.tweak_size == 2:
                for i in range(1, k_nbr_rounds):
                    if i == 1:
                        KS.AddIdentityLayer("K_ID", i, 0)     # Identity layer
                        KS.AddIdentityLayer("K_ID", i, 1)     # Identity layer
                    else:
                        KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        KS.GF2Linear_TransLayer("K_LFSR", i, 1, [j for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                    KS.SingleOperatorLayer("K_XOR", i, 2, XOR, [[j,16+j] for j in range(8)], [j for j in range(32,40)]) # XOR layer
            elif self.tweak_size == 3:
                for i in range(1, k_nbr_rounds):
                    if i == 1:
                        KS.AddIdentityLayer("K_ID", i, 0)     # Identity layer
                        KS.AddIdentityLayer("K_ID", i, 1)     # Identity layer
                        KS.AddIdentityLayer("K_ID", i, 2)     # Identity layer
                    else:
                        KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        KS.GF2Linear_TransLayer("K_LFSR1", i, 1, [j for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                        KS.GF2Linear_TransLayer("K_LFSR2", i, 2, [j for j in range(32,40)], [j for j in range(32,40)], mat=mat2)
                    KS.SingleOperatorLayer("K_XOR", i, 3, XOR, [[j,16+j] for j in range(8)], [j for j in range(48,56)]) # XOR layer
                    KS.SingleOperatorLayer("K_XOR", i, 4, XOR, [[j,16+j] for j in range(32,40)], [j for j in range(48,56)]) # XOR layer

            # Internal permutation
            for i in range(1,nbr_rounds+1):
                #i can remove hte sbox layer but that means 2 ttables and in the end still 2 lookups so im going to keep this
                S.SboxLayer("SB", i, 0, sbox) # Sbox layer
                S.AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer
                S.AddRoundKeyLayer("ARK", i, 2, XOR, SK, mask=[1 for i in range(8)])  # AddRoundKey layer
                S.TTableLayer("TT", i, 3, SKINNY_TTable, [0, 7, 10, 13, 1, 4, 11, 14, 2, 5, 8, 15, 3, 6, 9, 12], [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15])

    def gen_rounds_constant_table(self):
        constant_table = []
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                    0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                    0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                    0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                    0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]

        for i in range(1,self.functions["PERMUTATION"].nbr_rounds+1):
            rc = round_constants[i-1]
            c0, c1, c2 = rc & 0xF, rc >> 4, 0x2
            constant_table.append([c0,c1,c2])
        return constant_table

    def gen_test_vectors(self, version):
        # test vectors from https://eprint.iacr.org/2016/660.pdf and https://github.com/inmcm/skinny_cipher/blob/master/Python/skinny.py
        if version == [64, 64]:
            plaintext = [0x0, 0x6, 0x0, 0x3, 0x4, 0xf, 0x9, 0x5, 0x7, 0x7, 0x2, 0x4, 0xd, 0x1, 0x9, 0xd]
            key = [0xf, 0x5, 0x2, 0x6, 0x9, 0x8, 0x2, 0x6, 0xf, 0xc, 0x6, 0x8, 0x1, 0x2, 0x3, 0x8]
            ciphertext = [0xb, 0xb, 0x3, 0x9, 0xd, 0xf, 0xb, 0x2, 0x4, 0x2, 0x9, 0xb, 0x8, 0xa, 0xc, 0x7]
        elif version == [128, 128]:
            plaintext = [0xf2, 0x0a, 0xdb, 0x0e, 0xb0, 0x8b, 0x64, 0x8a, 0x3b, 0x2e, 0xee, 0xd1, 0xf0, 0xad, 0xda, 0x14]
            key = [0x4f, 0x55, 0xcf, 0xb0, 0x52, 0x0c, 0xac, 0x52, 0xfd, 0x92, 0xc1, 0x5f, 0x37, 0x07, 0x3e, 0x93]
            ciphertext = [0x22, 0xff, 0x30, 0xd4, 0x98, 0xea, 0x62, 0xd7, 0xe4, 0x5b, 0x47, 0x6e, 0x33, 0x67, 0x5b, 0x74]
        elif version == [64, 128]:
            plaintext = [0xc, 0xf, 0x1, 0x6, 0xc, 0xf, 0xe, 0x8, 0xf, 0xd, 0x0, 0xf, 0x9, 0x8, 0xa, 0xa]
            key = [0x9, 0xe, 0xb, 0x9, 0x3, 0x6, 0x4, 0x0, 0xd, 0x0, 0x8, 0x8, 0xd, 0xa, 0x6, 0x3, 0x7, 0x6, 0xa, 0x3, 0x9, 0xd, 0x1, 0xc, 0x8, 0xb, 0xe, 0xa, 0x7, 0x1, 0xe, 0x1]
            ciphertext = [0x6, 0xc, 0xe, 0xd, 0xa, 0x1, 0xf, 0x4, 0x3, 0xd, 0xe, 0x9, 0x2, 0xb, 0x9, 0xe]
        elif version == [64, 192]:
            plaintext = [0x5, 0x3, 0x0, 0xc, 0x6, 0x1, 0xd, 0x3, 0x5, 0xe, 0x8, 0x6, 0x6, 0x3, 0xc, 0x3]
            key = [0xe, 0xd, 0x0, 0x0, 0xc, 0x8, 0x5, 0xb, 0x1, 0x2, 0x0, 0xd, 0x6, 0x8, 0x6, 0x1, 0x8, 0x7, 0x5, 0x3, 0xe, 0x2, 0x4, 0xb, 0xf, 0xd, 0x9, 0x0, 0x8, 0xf, 0x6, 0x0, 0xb, 0x2, 0xd, 0xb, 0xb, 0x4, 0x1, 0xb, 0x4, 0x2, 0x2, 0xd, 0xf, 0xc, 0xd, 0x0]
            ciphertext = [0xd, 0xd, 0x2, 0xc, 0xf, 0x1, 0xa, 0x8, 0xf, 0x3, 0x3, 0x0, 0x3, 0x0, 0x3, 0xc]
        elif version == [128, 256]:
            plaintext = [0x3a, 0x0c, 0x47, 0x76, 0x7a, 0x26, 0xa6, 0x8d, 0xd3, 0x82, 0xa6, 0x95, 0xe7, 0x02, 0x2e, 0x25]
            key = [0x00, 0x9c, 0xec, 0x81, 0x60, 0x5d, 0x4a, 0xc1, 0xd2, 0xae, 0x9e, 0x30, 0x85, 0xd7, 0xa1, 0xf3, 0x1a, 0xc1, 0x23, 0xeb, 0xfc, 0x00, 0xfd, 0xdc, 0xf0, 0x10, 0x46, 0xce, 0xed, 0xdf, 0xca, 0xb3]
            ciphertext = [0xb7, 0x31, 0xd9, 0x8a, 0x4b, 0xde, 0x14, 0x7a, 0x7e, 0xd4, 0xa6, 0xf1, 0x6b, 0x9b, 0x58, 0x7f]
        elif version == [128, 384]:
            plaintext = [0xa3,0x99,0x4b,0x66,0xad,0x85,0xa3,0x45,0x9f,0x44,0xe9,0x2b,0x08,0xf5,0x50,0xcb]
            key = [0xdf,0x88,0x95,0x48,0xcf,0xc7,0xea,0x52,0xd2,0x96,0x33,0x93,0x01,0x79,0x74,0x49, 0xab,0x58,0x8a,0x34,0xa4,0x7f,0x1a,0xb2,0xdf,0xe9,0xc8,0x29,0x3f,0xbe,0xa9,0xa5, 0xab,0x1a,0xfa,0xc2,0x61,0x10,0x12,0xcd,0x8c,0xef,0x95,0x26,0x18,0xc3,0xeb,0xe8]
            ciphertext = [0x94, 0xec, 0xf5, 0x89, 0xe2, 0x1, 0x7c, 0x60, 0x1b, 0x38, 0xc6, 0x34, 0x6a, 0x10, 0xdc, 0xfa]
        return [[plaintext, key], ciphertext]

def SKINNY_BLOCKCIPHER(r=None, version=[64, 64], represent_mode=0):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/16), int(version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(16)], [var.Variable(word_size,ID="k"+str(i)) for i in range(16*m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(16)]
    my_cipher = Skinny_block_cipher(f"SKINNY{p_bitsize}_{k_bitsize}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.clean_graph()
    return my_cipher
