from primitives.primitives import Block_cipher
from operators.modular_operators import ModAdd
from operators.SHACAL2BooleanFunctions import SHACAL2_Sigma0, SHACAL2_Sigma1, SHACAL2_Sum0, SHACAL2_Sum1, SHACAL2_Maj, SHACAL2_Ch
import variables.variables as var

class SHACAL2_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Simon block cipher.
        :param name: Cipher name
        :param version: p_bitsize
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        wordSize = 32 if version == 256 else 64
        if nbr_rounds==None: (nbr_rounds, k_nbr_rounds) = (64, 49) if version==256 else (80, 65)
        if represent_mode==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (10, 8, 4, wordSize),  (6, 16, 2, wordSize),  (1, 1, 0, wordSize)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        self.test_vectors = self.gen_test_vectors()
        
        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        constant_table = self.gen_rounds_constant_table(version)

        if represent_mode==0:
            for i in range(1,nbr_rounds+1):
                if i<=16:
                    SK.ExtractionLayer("SK_EX", i, 0, [i-1], KS.vars[1][0])
                else:
                    SK.ExtractionLayer("SK_EX", i, 0, [15], KS.vars[i-15][0])
                   
                if i<k_nbr_rounds:
                    KS.SingleOperatorLayer("Sigma0", i, 0, SHACAL2_Sigma0, [[1]], [16])
                    KS.SingleOperatorLayer("Sigma1", i, 1, SHACAL2_Sigma1, [[14]], [17])
                    KS.SingleOperatorLayer("Add1", i, 2, ModAdd, [[17, 9]], [17])
                    KS.SingleOperatorLayer("Add2", i, 3, ModAdd, [[17, 16]], [17])
                    KS.SingleOperatorLayer("Add3", i, 4, ModAdd, [[17, 0]], [17])
                    KS.PermutationLayer("Key_Perm", i, 5, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17])
                    
                S.SingleOperatorLayer("Sum0", i, 0, SHACAL2_Sum0, [[0]], [8]) # SUM0 operator
                S.SingleOperatorLayer("Sum1", i, 1, SHACAL2_Sum1, [[4]], [9]) # SUM1 operator
                S.SingleOperatorLayer("Maj", i, 2, SHACAL2_Maj, [[0, 1, 2]], [10]) # Maj operator
                S.SingleOperatorLayer("Ch", i, 3, SHACAL2_Ch, [[4, 5, 6]], [11]) # Ch operator
                
                S.SingleOperatorLayer("ADD1", i, 4, ModAdd, [[8, 10], [7, 11]], [8, 11])
                S.SingleOperatorLayer("ADD2", i, 5, ModAdd, [[11, 9]], [11])
                S.AddRoundKeyLayer("ARK", i, 6, ModAdd, SK, [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
                S.AddConstantLayer("K_C", i, 7, 'modadd', [None,None,None,None,None,None,None,None,None,None,None,True], constant_table)

                S.SingleOperatorLayer("ADD3", i, 8, ModAdd, [[3, 11], [8, 11]], [3, 8]) #Add to d
                S.PermutationLayer("Perm", i, 9, [8, 0, 1, 2, 3, 4, 5, 6]) #Final permutation
        

    def gen_rounds_constant_table(self, version):
        if version == 256:
            constant_table = [[0x428a2f98], [0x71374491], [0xb5c0fbcf], [0xe9b5dba5], [0x3956c25b], [0x59f111f1], [0x923f82a4], [0xab1c5ed5],
                            [0xd807aa98], [0x12835b01], [0x243185be], [0x550c7dc3], [0x72be5d74], [0x80deb1fe], [0x9bdc06a7], [0xc19bf174],
                            [0xe49b69c1], [0xefbe4786], [0x0fc19dc6], [0x240ca1cc], [0x2de92c6f], [0x4a7484aa], [0x5cb0a9dc], [0x76f988da],
                            [0x983e5152], [0xa831c66d], [0xb00327c8], [0xbf597fc7], [0xc6e00bf3], [0xd5a79147], [0x06ca6351], [0x14292967],
                            [0x27b70a85], [0x2e1b2138], [0x4d2c6dfc], [0x53380d13], [0x650a7354], [0x766a0abb], [0x81c2c92e], [0x92722c85],
                            [0xa2bfe8a1], [0xa81a664b], [0xc24b8b70], [0xc76c51a3], [0xd192e819], [0xd6990624], [0xf40e3585], [0x106aa070],
                            [0x19a4c116], [0x1e376c08], [0x2748774c], [0x34b0bcb5], [0x391c0cb3], [0x4ed8aa4a], [0x5b9cca4f], [0x682e6ff3],
                            [0x748f82ee], [0x78a5636f], [0x84c87814], [0x8cc70208], [0x90befffa], [0xa4506ceb], [0xbef9a3f7], [0xc67178f2]]
        elif version == 512:
            #TODO ADD table
            pass
        return constant_table
    
    def gen_test_vectors(self, version=256):   
        # TODO: add test vectors that are non-all-zero and give link to them 
        if version == 256:
            KEY = [0x80000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]
            IN = [0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000]
            OUT = [0x361AB632, 0x2FA9E7A7, 0xBB23818D, 0x839E01BD, 0xDAFDF473, 0x05426EDD, 0x297AEDB9, 0xF6202BAE]
        elif version == 512:
            pass
        return [[IN, KEY], OUT]
    

def SHACAL2_BLOCKCIPHER(r=None, version=256, represent_mode=0):
    wordSize = 32 if version == 256 else 64
    my_plaintext, my_key, my_ciphertext = [var.Variable(wordSize,ID="in"+str(i)) for i in range(8)], [var.Variable(wordSize,ID="k"+str(i)) for i in range(16)], [var.Variable(wordSize,ID="out"+str(i)) for i in range(8)]
    my_cipher = SHACAL2_block_cipher(f"SHACAL2_{wordSize*8}_{wordSize*16}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.clean_graph()
    return my_cipher