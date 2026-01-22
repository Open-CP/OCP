from primitives.primitives import Permutation
from operators.modular_operators import ModAdd
from operators.boolean_operators import XOR
from operators.operators import Equal
import variables.variables as var


# The ChaCha internal permutation
class ChaCha_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the ChaCha internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """
        
        if represent_mode==0:
            if nbr_rounds ==None: nbr_rounds = 20
            nbr_layers = 12 # 1 for each of the 12 operations in 1 quarter round
            nbr_words = 16 # Words in the state of Chacha
            nbr_temp_words = 0
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["PERMUTATION"]
        
            for i in range(1,nbr_rounds+1):  
                if i%2 == 0:
                    # Even rounds are diagonal rounds
                    QR1 = [0, 5, 10, 15]
                    QR2 = [1, 6, 11, 12]
                    QR3 = [2, 7, 8, 13]
                    QR4 = [3, 4, 9, 14]
                else:
                    # Odd rounds are column rounds
                    QR1 = [0, 4, 8, 12]
                    QR2 = [1, 5, 9, 13]
                    QR3 = [2, 6, 10, 14]
                    QR4 = [3, 7, 11, 15]
                
                # Compute a', d', c', b' for all 4 quater rounds in parallel
                S.SingleOperatorLayer("Add1", i, 0, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [QR1[0], QR2[0], QR3[0], QR4[0]])
                S.SingleOperatorLayer("XOR1", i, 1, XOR, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])
                S.RotationLayer("Rot1", i, 2, [['l', 16, QR1[3], QR1[3]], ['l', 16, QR2[3], QR2[3]], ['l', 16, QR3[3], QR3[3]], ['l', 16, QR4[3], QR4[3]]])
                S.SingleOperatorLayer("Add2", i , 3, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [QR1[2], QR2[2], QR3[2], QR4[2]])
                S.SingleOperatorLayer("XOR2", i, 4, XOR, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [QR1[1], QR2[1], QR3[1], QR4[1]])
                S.RotationLayer("Rot2", i, 5, [['l', 12, QR1[1], QR1[1]], ['l', 12, QR2[1], QR2[1]], ['l', 12, QR3[1], QR3[1]], ['l', 12, QR4[1], QR4[1]]])

                # Compute a'', d'', c'', b'' for all 4 quater rounds in parallel
                S.SingleOperatorLayer("Add3", i, 6, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [QR1[0], QR2[0], QR3[0], QR4[0]])
                S.SingleOperatorLayer("XOR3", i, 7, XOR, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])
                S.RotationLayer("Rot3", i, 8, [['l', 8, QR1[3], QR1[3]], ['l', 8, QR2[3], QR2[3]], ['l', 8, QR3[3], QR3[3]], ['l', 8, QR4[3], QR4[3]]])
                S.SingleOperatorLayer("Add4", i , 9, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [QR1[2], QR2[2], QR3[2], QR4[2]])
                S.SingleOperatorLayer("XOR4", i, 10, XOR, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [QR1[1], QR2[1], QR3[1], QR4[1]])
                S.RotationLayer("Rot4", i, 11, [['l', 7, QR1[1], QR1[1]], ['l', 7, QR2[1], QR2[1]], ['l', 7, QR3[1], QR3[1]], ['l', 7, QR4[1], QR4[1]]])
        
    def gen_test_vectors(self):
        # Test vectors from https://datatracker.ietf.org/doc/html/rfc8439
        IN = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, 0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c, 0x00000001, 0x09000000, 0x4a000000, 0x00000000]
        OUT = [0x837778ab, 0xe238d763, 0xa67ae21e, 0x5950bb2f, 0xc4f2d0c7, 0xfc62bb2f, 0x8fa018fc, 0x3f5ec7b7, 0x335271c2, 0xf29489f3, 0xeabda8fc, 0x82e46ebd, 0xd19c12b4, 0xb04e16de, 0x9e83d0cb, 0x4e3c50a2]
        self.test_vectors.append([[IN], OUT])


def CHACHA_PERMUTATION(r=None, represent_mode=0, copy_operator=False): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_permutation = ChaCha_permutation("ChaCha_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors()
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


# The ChaCha permutation to generate the key stream
class ChaCha_keypermutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the ChaCha internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """
        
        if represent_mode==0:
            if nbr_rounds ==None: nbr_rounds = 21 # 21st round is used add the initial state to obtain the final key stream
            nbr_layers = 13 # 1 for each of the 12 operations in 1 quarter round
            nbr_words = 16 # Words in the state of Chacha
            nbr_temp_words = 16 # To retain the initial input for adding with final state to obtain the key stream
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["PERMUTATION"]
        
            for i in range(1,nbr_rounds+1):  
                if i%2 == 0:
                    # Even rounds are diagonal rounds
                    QR1 = [0, 5, 10, 15]
                    QR2 = [1, 6, 11, 12]
                    QR3 = [2, 7, 8, 13]
                    QR4 = [3, 4, 9, 14]
                else:
                    # Odd rounds are column rounds
                    QR1 = [0, 4, 8, 12]
                    QR2 = [1, 5, 9, 13]
                    QR3 = [2, 6, 10, 14]
                    QR4 = [3, 7, 11, 15]
                
                # In the first round copy the initial word to temporary words
                if i == 1:
                    InIndex = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
                    OutIndex = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                    S.SingleOperatorLayer("Equal", i, 0, Equal, InIndex, OutIndex)
                else:
                    S.AddIdentityLayer("Identity", i, 0)


                if i == 21:
                    InIndex = [[0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23], [8, 24], [9, 25], [10, 26], [11, 27], [12, 28], [13, 29], [14, 30], [15, 31]]
                    OutIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, InIndex, OutIndex)
                    for j in range(2, nbr_layers):
                        name = 'Identity' + str(j)
                        S.AddIdentityLayer(name, i, j)
                else:
                    # Compute a', d', c', b' for all 4 quater rounds in parallel
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [QR1[0], QR2[0], QR3[0], QR4[0]])
                    S.SingleOperatorLayer("XOR1", i, 2, XOR, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])
                    S.RotationLayer("Rot1", i, 3, [['l', 16, QR1[3], QR1[3]], ['l', 16, QR2[3], QR2[3]], ['l', 16, QR3[3], QR3[3]], ['l', 16, QR4[3], QR4[3]]])
                    S.SingleOperatorLayer("Add2", i , 4, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [QR1[2], QR2[2], QR3[2], QR4[2]])
                    S.SingleOperatorLayer("XOR2", i, 5, XOR, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [QR1[1], QR2[1], QR3[1], QR4[1]])
                    S.RotationLayer("Rot2", i, 6, [['l', 12, QR1[1], QR1[1]], ['l', 12, QR2[1], QR2[1]], ['l', 12, QR3[1], QR3[1]], ['l', 12, QR4[1], QR4[1]]])

                    # Compute a'', d'', c'', b'' for all 4 quater rounds in parallel
                    S.SingleOperatorLayer("Add3", i, 7, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [QR1[0], QR2[0], QR3[0], QR4[0]])
                    S.SingleOperatorLayer("XOR3", i, 8, XOR, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])
                    S.RotationLayer("Rot3", i, 9, [['l', 8, QR1[3], QR1[3]], ['l', 8, QR2[3], QR2[3]], ['l', 8, QR3[3], QR3[3]], ['l', 8, QR4[3], QR4[3]]])
                    S.SingleOperatorLayer("Add4", i , 10, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [QR1[2], QR2[2], QR3[2], QR4[2]])
                    S.SingleOperatorLayer("XOR4", i, 11, XOR, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [QR1[1], QR2[1], QR3[1], QR4[1]])
                    S.RotationLayer("Rot4", i, 12, [['l', 7, QR1[1], QR1[1]], ['l', 7, QR2[1], QR2[1]], ['l', 7, QR3[1], QR3[1]], ['l', 7, QR4[1], QR4[1]]])

    def gen_test_vectors(self):
        # Test vectors from https://datatracker.ietf.org/doc/html/rfc8439
        IN = [0x61707865, 0x3320646e, 0x79622d32, 0x6b206574, 0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c, 0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c, 0x00000001, 0x09000000, 0x4a000000, 0x00000000]
        OUT = [0xe4e7f110, 0x15593bd1, 0x1fdd0f50, 0xc47120a3, 0xc7f4d1c7, 0x0368c033, 0x9aaa2204, 0x4e6cd4c3, 0x466482d2, 0x09aa9f07, 0x05d7c214, 0xa2028bd9, 0xd19c12b5, 0xb94e16de, 0xe883d0cb, 0x4e3c50a2]
        self.test_vectors.append([[IN], OUT])
    

def CHACHA_KEYPERMUTATION(r=None, represent_mode=0, copy_operator=False): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_permutation = ChaCha_keypermutation("ChaCha_KEYPERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors()
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation