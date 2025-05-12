from primitives.primitives import Permutation, Block_cipher
from operators.Sbox import PRESENT_Sbox
from operators.boolean_operators import XOR


# The PRESENT internal permutation               
class PRESENT_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the PRESENT internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if nbr_rounds==None: nbr_rounds=31
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 2, version, 0, 1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        s = self.states["STATE"]
        perm = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):              
                s.SboxLayer("SB", i, 0, PRESENT_Sbox,index=[list(range(i, i + 4)) for i in range(0, nbr_words, 4)])  # Sbox layer            
                s.PermutationLayer("P", i, 1, perm) # Permutation layer


# The PRESENT block cipher      
# Test vectors for PRESENT-80
# plaintext = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ciphertext = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1]          
# Test vectors for PRESENT-128
# plaintext = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# key = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ciphertext = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1]
# https://www.iacr.org/archive/ches2007/47270450/47270450.pdf


class PRESENT_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the PRESENT block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None or nbr_rounds==31: nbr_rounds=32
        if represent_mode==0: 
            (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (3, p_bitsize, 0, 1),  (3, k_bitsize, 0, 1),  (1, p_bitsize, 0, 1)
            perm_ks = [(61+i)%k_bitsize for i in range(k_bitsize)]
            if k_bitsize == 80:
                sbox_mask_ks, sbox_index_ks, cons_mask_ks = [1], [[0, 1, 2, 3]], [None]*60 + [True]*5
            elif k_bitsize == 128:
                sbox_mask_ks, sbox_index_ks, cons_mask_ks = [1, 1], [[0, 1, 2, 3], [4, 5, 6, 7]], [None]*61 + [True]*5
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        S = self.states["STATE"]
        KS = self.states["KEY_STATE"]
        SK = self.states["SUBKEYS"]
        
        constant_table = self.gen_rounds_constant_table()
        perm = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]

        # create constraints
        if represent_mode==0:
            # subkeys extraction
            for i in range(1,SK.nbr_rounds+1):    
                SK.ExtractionLayer("SK_EX", i, 0, [i for i in range(64)], self.states["KEY_STATE"].vars[i][0])

            # key schedule
            for i in range(1,KS.nbr_rounds):    
                KS.PermutationLayer("PERM", i, 0, perm_ks) # Permutation layer 
                KS.SboxLayer("SB", i, 1, PRESENT_Sbox, sbox_mask_ks, sbox_index_ks)  # Sbox layer            
                KS.AddConstantLayer("C", i, 2, "xor", cons_mask_ks, constant_table)# Constant layer                      
                
            # Internal permutation          
            for i in range(1,S.nbr_rounds+1): 
                S.AddRoundKeyLayer("ARK", i, 0, XOR, SK, [1]*64) # Addroundkey layer 
                if i < 32: 
                    S.SboxLayer("SB", i, 1, PRESENT_Sbox,index=[list(range(i, i + 4)) for i in range(0, s_nbr_words, 4)])  # Sbox layer            
                    S.PermutationLayer("P", i, 2, perm) # Permutation layer
                elif i == 32:
                    S.AddIdentityLayer("ID", i, 1)
                    S.AddIdentityLayer("ID", i, 2)

    def gen_rounds_constant_table(self):
        return [[(i >> j) & 1 for j in reversed(range(5))] for i in range(1, self.states["KEY_STATE"].nbr_rounds)]