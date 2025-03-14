from primitives.primitives import Permutation
import operators.operators as op


# The ChaCha internal permutation  
class ChaCha_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the ChaCha internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        nbr_layers = 12
        nbr_words = 16 
        nbr_temp_words = 4
        word_bitsize = 64
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.states["STATE"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):  
                pass

