from primitives.primitives import Permutation, Block_cipher
import operators.operators as op

#  The Rocca_AD internal permutation  
class Rocca_AD_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Rocca_AD internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if nbr_rounds==None: nbr_rounds=20
        if represent_mode==0: 
            nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 128+32*nbr_rounds, 32, 0
            perm_s = [0,5,10,15, 4,9,14,3, 8,13,2,7, 12,1,6,11]
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if represent_mode==0:
            for r in range(1, nbr_rounds+1):
                self.states["STATE"].PermutationLayer("P", r, 0, [i for i in range(16*7,16*8)] + perm_s + [i for i in range(16,16*2)] + [perm_s[i]+16*2 for i in range(16)] + [i for i in range(16*3,16*4)] + [perm_s[i]+16*4 for i in range(16)] + [perm_s[i]+16*5 for i in range(16)] + [i for i in range(16*6, 16*7)] + [i for i in range(16*8,16*(8+2*nbr_rounds))] + [i for i in range(16)] + [i for i in range(16*4, 16*5)])
                self.states["STATE"].SboxLayer("SB", r, 1, op.AES_Sbox, mask=[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[0 for i in range(16)]+[1 for i in range(16)]+[1 for i in range(16)]) # Sbox layer 
                self.states["STATE"].MatrixLayer("MC", r, 2, [[2,3,1,1], [1,2,3,1], [1,1,2,3], [3,1,1,2]], [[0+16,1+16,2+16,3+16], [4+16,5+16,6+16,7+16], [8+16,9+16,10+16,11+16], [12+16,13+16,14+16,15+16],[0+16*3,1+16*3,2+16*3,3+16*3], [4+16*3,5+16*3,6+16*3,7+16*3], [8+16*3,9+16*3,10+16*3,11+16*3], [12+16*3,13+16*3,14+16*3,15+16*3],[0+16*5,1+16*5,2+16*5,3+16*5], [4+16*5,5+16*5,6+16*5,7+16*5], [8+16*5,9+16*5,10+16*5,11+16*5], [12+16*5,13+16*5,14+16*5,15+16*5], [0+16*6,1+16*6,2+16*6,3+16*6], [4+16*6,5+16*6,6+16*6,7+16*6], [8+16*6,9+16*6,10+16*6,11+16*6], [12+16*6,13+16*6,14+16*6,15+16*6]], "0x1B")  #Mixcolumns layer
                self.states["STATE"].SingleOperatorLayer("XOR", r, 3, op.bitwiseXOR, [[i, i+16*(8+2*(r-1))] for i in range(16)] + [[i+16, i] for i in range(16)] + [[i+16*2, i+16*7] for i in range(16)] + [[i+16*3, i+16*2] for i in range(16)] + [[i+16*4, i+16*(8+2*(r-1)+1)] for i in range(16)] + [[i+16*5, i+16*4]  for i in range(16)] + [[i+16*6, i+16*(8+2*nbr_rounds+1)] for i in range(16)] + [[i+16*7, i+16*(8+2*nbr_rounds)] for i in range(16)],[i for i in range(16*8)]) # XOR layer 
               
        