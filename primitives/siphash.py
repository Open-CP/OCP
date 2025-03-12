from primitives.primitives import Permutation, Block_cipher
import operators.operators as op


# The SipHash internal permutation  
class SipHash_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the SipHash internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        nbr_layers = 10
        nbr_words = 4 
        nbr_temp_words = 0
        word_bitsize = 64
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.states["STATE"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):  
                S.SingleOperatorLayer("ADD1", i, 0, op.ModAdd, [[0,1], [2,3]], [0, 2]) # Modular addition layer
                S.RotationLayer("ROT1", i, 1, [['l', 13, 1], ['l', 16, 3]]) # Rotation layer
                S.SingleOperatorLayer("XOR1", i, 2, op.bitwiseXOR, [[0,1], [2,3]], [1, 3]) # XOR layer
                S.RotationLayer("ROT2", i, 3, [['l', 32, 0]]) # Rotation layer
                S.PermutationLayer("PERM1", i, 4, [2,1,0,3]) # Permutation layer
                S.SingleOperatorLayer("ADD2", i, 5, op.ModAdd, [[0,1], [2,3]], [0, 2]) # Modular addition layer
                S.RotationLayer("ROT3", i, 6, [['l', 17, 1], ['l', 21, 3]]) # Rotation layer
                S.SingleOperatorLayer("XOR2", i, 7, op.bitwiseXOR, [[0,1], [2,3]], [1, 3]) # XOR layer
                S.RotationLayer("ROT4", i, 8, [['l', 32, 0]]) # Rotation layer
                S.PermutationLayer("PERM2", i, 9, [2,1,0,3]) # Permutation layer
                