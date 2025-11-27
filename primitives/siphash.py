from primitives.primitives import Permutation
from operators.boolean_operators import XOR
from operators.modular_operators import ModAdd
import variables.variables as var


# The SipHash internal permutation
class SipHash_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the SipHash internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """
        nbr_layers = 10
        nbr_words = 4 
        nbr_temp_words = 0
        word_bitsize = 64
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):  
                S.SingleOperatorLayer("ADD1", i, 0, ModAdd, [[0,1], [2,3]], [0, 2]) # Modular addition layer
                S.RotationLayer("ROT1", i, 1, [['l', 13, 1], ['l', 16, 3]]) # Rotation layer
                S.SingleOperatorLayer("XOR1", i, 2, XOR, [[0,1], [2,3]], [1, 3]) # XOR layer
                S.RotationLayer("ROT2", i, 3, [['l', 32, 0]]) # Rotation layer
                S.PermutationLayer("PERM1", i, 4, [2,1,0,3]) # Permutation layer
                S.SingleOperatorLayer("ADD2", i, 5, ModAdd, [[0,1], [2,3]], [0, 2]) # Modular addition layer
                S.RotationLayer("ROT3", i, 6, [['l', 17, 1], ['l', 21, 3]]) # Rotation layer
                S.SingleOperatorLayer("XOR2", i, 7, XOR, [[0,1], [2,3]], [1, 3]) # XOR layer
                S.RotationLayer("ROT4", i, 8, [['l', 32, 0]]) # Rotation layer
                S.PermutationLayer("PERM2", i, 9, [2,1,0,3]) # Permutation layer

        self.test_vectors = self.gen_test_vectors()
    
    def gen_test_vectors(self):
        # Test vector from  https://www.aumasson.jp/siphash/siphash.pdf
        IN =[0x7469686173716475, 0x6b617f6d656e6665, 0x6b7f62616d677361, 0x7c6d6c6a717c6d7b]
        OUT = [0x4d07749cdd0858e0, 0x0d52f6f62a4f59a4, 0x634cb3577b01fd3d, 0xa5224d6f55c7d9c8] 
        return [[IN], OUT]
    
    
def SIPHASH_PERMUTATION(r=None, represent_mode=0): 
    my_input, my_output = [var.Variable(64,ID="in"+str(i)) for i in range(4)], [var.Variable(64,ID="out"+str(i)) for i in range(4)]
    my_permutation = SipHash_permutation("SipHash_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    return my_permutation