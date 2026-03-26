from typing import List

from primitives.primitives import Permutation, Block_cipher
from operators.Sbox import MIDORI_Sbox, MIDORI_Sbox_0, MIDORI_Sbox_1, MIDORI_Sbox_2, MIDORI_Sbox_3
from operators.boolean_operators import XOR
import variables.variables as var

def gen_rounds_constant_table():
    BETA_MATS: List[List[List[int]]] = [
        [[0,0,1,0],[0,1,0,0],[0,0,1,1],[1,1,1,1]],  # 0
        [[0,1,1,0],[1,0,1,0],[1,0,0,0],[1,0,0,0]],  # 1
        [[1,0,0,0],[0,1,0,1],[1,0,1,0],[0,0,1,1]],  # 2
        [[0,0,0,0],[1,0,0,0],[1,1,0,1],[0,0,1,1]],  # 3
        [[0,0,0,1],[0,0,1,1],[0,0,0,1],[1,0,0,1]],  # 4
        [[1,0,0,0],[1,0,1,0],[0,0,1,0],[1,1,1,0]],  # 5
        [[0,0,0,0],[0,0,1,1],[0,1,1,1],[0,0,0,0]],  # 6
        [[0,1,1,1],[0,0,1,1],[0,1,0,0],[0,1,0,0]],  # 7
        [[1,0,1,0],[0,1,0,0],[0,0,0,0],[1,0,0,1]],  # 8
        [[0,0,1,1],[1,0,0,0],[0,0,1,0],[0,0,1,0]],  # 9
        [[0,0,1,0],[1,0,0,1],[1,0,0,1],[1,1,1,1]],  # 10
        [[0,0,1,1],[0,0,0,1],[1,1,0,1],[0,0,0,0]],  # 11
        [[0,0,0,0],[1,0,0,0],[0,0,1,0],[1,1,1,0]],  # 12
        [[1,1,1,1],[1,0,1,0],[1,0,0,1],[1,0,0,0]],  # 13
        [[1,1,1,0],[1,1,0,0],[0,1,0,0],[1,1,1,0]],  # 14
        [[0,1,1,0],[1,1,0,0],[1,0,0,0],[1,0,0,1]],  # 15
        [[0,1,0,0],[0,1,0,1],[0,0,1,0],[1,0,0,0]],  # 16
        [[0,0,1,0],[0,0,0,1],[1,1,1,0],[0,1,1,0]],  # 17
        [[0,0,1,1],[1,0,0,0],[1,1,0,1],[0,0,0,0]],  # 18
    ]

    def _beta_bits_by_state_index(beta_mat_4x4: List[List[int]]) -> List[int]:
        bits = [0] * 16
        for r in range(4):
            for c in range(4):
                bits[r + 4 * c] = beta_mat_4x4[r][c] & 1
        return bits

    return [_beta_bits_by_state_index(BETA_MATS[0])]+[_beta_bits_by_state_index(m) for m in BETA_MATS]


# The FUTURE block cipher
class MIDORI_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the MIDORI block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize) - (64, 128)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        assert version in [[64, 128], [128, 128]], f"FUTURE only supports (64, 128), (128, 128) versions, got {version}"
        p_bitsize, k_bitsize = version[0], version[1]
        if represent_mode==0:
            if p_bitsize==64:
                nbr_rounds = 17
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize) = (4, 16, 0, 4)
                (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize) = (2, 32, 0, 4)
                (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (1, 16, 0, 4)
            else:
                nbr_rounds = 21
                (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize) = (7, 16, 0, 8)
                (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize) = (2, 16, 0, 8)
                (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (1, 16, 0, 8)
            k_nbr_rounds = nbr_rounds

        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds,
                        [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize],
                        [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize],
                        [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        self.functions_implementation_order = ["KEY_SCHEDULE", "SUBKEYS", "PERMUTATION"]
        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]
        constant_table = gen_rounds_constant_table()
        shift_rows = [0, 10, 5, 15, 14, 4, 11, 1, 9, 3, 12, 6, 7, 13, 2, 8]
        mix_columns_matrix = [[0,1,1,1],[1,0,1,1],[1,1,0,1],[1,1,1,0]]
        mix_columns_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

        # create constraints
        if represent_mode==0:
            # key schedule (alternate between K1 and K2 for LED-128)
            #hard set the first two layers 
            #this one each time there is at mask whic his the first 16 words only 
            if p_bitsize==128:
                for rd in range(1, k_nbr_rounds+1):
                    if rd ==1:
                        KS.AddIdentityLayer("KS_ID",rd, 0)
                        KS.AddIdentityLayer("KS_ID",rd, 1)
                    elif rd == k_nbr_rounds:#take the output fromthe 1 layer 
                        KS.ExtractionLayer("KS_EX",rd, 0, list(range(16)), KS.vars[1][-1])
                        KS.AddIdentityLayer("KS_ID",rd, 1)
                    else:
                        KS.ExtractionLayer("KS_EX",rd, 0, list(range(16)), KS.vars[1][-1])
                        KS.AddConstantLayer("KS_ADD_CONST", rd, 1, "xor", [True]*16, constant_table)
            else:#64 bit plain text size 
                for rd in range(1, k_nbr_rounds+1):
                    if rd ==1:
                        KS.SingleOperatorLayer("KS_XOR", rd, 0, XOR, [[i,i+16] for i in range(16)],list(range(16)))
                        KS.AddIdentityLayer("KS_ID",rd, 1)
                    elif rd == k_nbr_rounds:#take the output fromthe 1 layer 
                        KS.ExtractionLayer("KS_EX",rd, 0, list(range(32)), KS.vars[1][-1])
                        KS.AddIdentityLayer("KS_ID",rd, 1)
                    else:
                        KS.ExtractionLayer("KS_EX", rd, 0, list(range(16))+list(range(16,32)) if rd%2==0 else list(range(16, 32))+list(range(16)), KS.vars[1][0])#extract the output of layer 1
                        KS.AddConstantLayer("KS_ADD_CONST", rd, 1, "xor", [True]*16 + [None]*16, constant_table)

            for i in range(1, k_nbr_rounds+1):
                SK.ExtractionLayer("SK_EX", i, 0, list(range(16)), KS.vars[i][-1])#extract from the input    
            #for the 128 bit plaintext 
            #   do fow differint SboxLayers
            # Internal permutation
            for i in range(1, nbr_rounds+1):
                if i ==nbr_rounds-1:
                    S.AddRoundKeyLayer("ARK", i, 0, XOR, SK, [1]*16)
                    for j in range(1, s_nbr_layers):
                        S.AddIdentityLayer("ID", i, j)
                elif i==nbr_rounds:
                    nlyr = self.midori_sbox_layers(i, 0, p_bitsize, S)
                    S.AddRoundKeyLayer("ARK", i, nlyr, XOR, SK, [1]*16)
                    for j in range(nlyr+1, s_nbr_layers):
                        S.AddIdentityLayer("ID", i, j)
                else:
                    S.AddRoundKeyLayer("ARK", i, 0, XOR, SK, [1]*16) 
                    nlyr = self.midori_sbox_layers(i, 1, p_bitsize, S)
                    S.PermutationLayer("SR", i, nlyr, shift_rows)  # permute
                    S.MatrixLayer("S_MAT", i, nlyr+1, mix_columns_matrix, mix_columns_index)
    def midori_sbox_layers(self, rd, lyr, p_bit,S):
        if p_bit==64:
            S.SboxLayer("SB", rd, lyr, MIDORI_Sbox)
            return lyr+1
        else:
            sbox = [MIDORI_Sbox_0, MIDORI_Sbox_1, MIDORI_Sbox_2, MIDORI_Sbox_3]
            for j in range(lyr, lyr+4):
                S.SboxLayer(f"SB_{(j-lyr)%4}", rd, j, sbox[(j-lyr)%4], [1 if a%4==(j-lyr)%4 else 0 for a in range(16)])
            return lyr+4
    def gen_test_vectors(self, version=None):
        if version == [64, 128]:
            P = [ 0x4, 0x2, 0xC, 0x2, 0x0, 0xF, 0xD, 0x3, 0xB, 0x5, 0x8, 0x6, 0x8, 0x7, 0x9, 0xE]
            K = [ 0x6, 0x8, 0x7, 0xD, 0xE, 0xD, 0x3, 0xB, 0x3, 0xC, 0x8, 0x5, 0xB, 0x3, 0xF, 0x3,0x5, 0xB, 0x1, 0x0, 0x0, 0x9, 0x8, 0x6,0x3, 0xE, 0x2, 0xA, 0x8, 0xC, 0xB, 0xF]
            C = [0x6, 0x6, 0xB, 0xC, 0xD, 0xC, 0x6, 0x2, 0x7, 0x0, 0xD, 0x9, 0x0, 0x1, 0xC, 0xD]
            self.test_vectors.append([[P, K], C])
            P = [0]*16
            K = [0]*32
            C = [0x3, 0xc, 0x9, 0xc, 0xc, 0xe, 0xd, 0xa, 0x2, 0xb, 0xb, 0xd, 0x4, 0x4, 0x9, 0xa]
            self.test_vectors.append([[P, K], C])
        elif version == [128,128]:
            P = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            K = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
            C = [0xc0, 0x55, 0xcb, 0xb9, 0x59, 0x96, 0xd1, 0x49, 0x02, 0xb6, 0x05, 0x74, 0xd5, 0xe7, 0x28, 0xd6]
            self.test_vectors.append([[P, K], C])
            P = [0x51, 0x08, 0x4c, 0xe6, 0xe7, 0x3a, 0x5c, 0xa2, 0xec, 0x87, 0xd7, 0xba, 0xbc, 0x29, 0x75, 0x43]
            K = [0x68, 0x7d, 0xed, 0x3b, 0x3c, 0x85, 0xb3, 0xf3, 0x5b, 0x10, 0x09, 0x86, 0x3e, 0x2a, 0x8c, 0xbf]
            C = [0x1e, 0x0a, 0xc4, 0xfd, 0xdf, 0xf7, 0x1b, 0x4c, 0x18, 0x01, 0xb7, 0x3e, 0xe4, 0xaf, 0xc8, 0x3d]
            self.test_vectors.append([[P, K], C]) 



def MIDORI_BLOCKCIPHER(r=None, version=[64, 128], represent_mode=0, copy_operator=False):
    """
    MIDORI block cipher
    :param r: Number of rounds (optional)
    :param version: (p_bitsize, k_bitsize) - [64, 64] for LED-64 or [64, 128] for LED-128
    :param represent_mode: Representation mode
    """
    bit_size = 4 if version[0]==64 else 8
    key_size = 32 if len(set(version))==2 else 16
    my_plaintext = [var.Variable(bit_size, ID="p"+str(i)) for i in range(16)]
    my_key = [var.Variable(bit_size, ID="k"+str(i)) for i in range(key_size)]
    my_ciphertext = [var.Variable(bit_size, ID="c"+str(i)) for i in range(16)]
    my_cipher = MIDORI_block_cipher(f"MIDORI{version[0]}_{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher


