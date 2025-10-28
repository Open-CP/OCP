from primitives.primitives import Permutation
from operators.modular_operators import ModAdd
from operators.boolean_operators import XOR
from operators.operators import Equal
import variables.variables as var


# The Salsa internal permutation
class Salsa_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Salsa internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if represent_mode==0:
            if nbr_rounds ==None: nbr_rounds = 20
            nbr_layers = 12 # 1 for each of the 12 operations in 1 quarter round
            nbr_words = 16 # Words in the state of Salsa
            nbr_temp_words = 4 # Temporary words to store the internal states
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["FUNCTION"]

            TW  = [16,17,18,19]
 
            for i in range(1,nbr_rounds+1):  
                if i%2 == 0:
                    # Even rounds are row rounds
                    QR1 = [0, 1, 2, 3]
                    QR2 = [5, 6, 7, 4]
                    QR3 = [10, 11, 8,9]
                    QR4 = [15, 12, 13, 14]

                else:
                    # Odd rounds are column rounds
                    QR1 = [0, 4, 8, 12]
                    QR2 = [5, 9, 13,1]
                    QR3 = [10, 14, 2,6]
                    QR4 = [15, 3, 7, 11]

                # 4 quarter rounds in parallel 
                S.SingleOperatorLayer("Add1", i, 0, ModAdd, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [TW[0], TW[1], TW[2], TW[3]])
                S.RotationLayer("Rot1", i, 1, [['l', 7, TW[0], TW[0]], ['l', 7, TW[1], TW[1]], ['l', 7, TW[2], TW[2]], ['l', 7, TW[3], TW[3]]])
                S.SingleOperatorLayer("XOR1", i, 2, XOR, [[TW[0], QR1[1]], [TW[1], QR2[1]], [TW[2], QR3[1]], [TW[3], QR4[1]]], [QR1[1], QR2[1], QR3[1], QR4[1]])

                S.SingleOperatorLayer("Add2", i, 3, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [TW[0], TW[1], TW[2], TW[3]])
                S.RotationLayer("Rot2", i, 4, [['l', 9, TW[0], TW[0]], ['l', 9, TW[1], TW[1]], ['l', 9, TW[2], TW[2]], ['l', 9, TW[3], TW[3]]])
                S.SingleOperatorLayer("XOR2", i, 5, XOR, [[TW[0], QR1[2]], [TW[1], QR2[2]], [TW[2], QR3[2]], [TW[3], QR4[2]]], [QR1[2], QR2[2], QR3[2], QR4[2]])

                S.SingleOperatorLayer("Add3", i, 6, ModAdd, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [TW[0], TW[1], TW[2], TW[3]])
                S.RotationLayer("Rot3", i, 7, [['l', 13, TW[0], TW[0]], ['l', 13, TW[1], TW[1]], ['l', 13, TW[2], TW[2]], ['l', 13, TW[3], TW[3]]])
                S.SingleOperatorLayer("XOR3", i, 8, XOR, [[TW[0], QR1[3]], [TW[1], QR2[3]], [TW[2], QR3[3]], [TW[3], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])

                S.SingleOperatorLayer("Add4", i, 9, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [TW[0], TW[1], TW[2], TW[3]])
                S.RotationLayer("Rot4", i, 10, [['l', 18, TW[0], TW[0]], ['l', 18, TW[1], TW[1]], ['l', 18, TW[2], TW[2]], ['l', 18, TW[3], TW[3]]])
                S.SingleOperatorLayer("XOR4", i, 11, XOR, [[TW[0], QR1[0]], [TW[1], QR2[0]], [TW[2], QR3[0]], [TW[3], QR4[0]]], [QR1[0], QR2[0], QR3[0], QR4[0]])
                                      
        self.test_vectors = self.gen_test_vectors()

    def gen_test_vectors(self):
        # Test vectors from https://cr.yp.to/snuffle/salsafamily-20071225.pdf

        IN = [  0x61707865, 0x04030201, 0x08070605, 0x0c0b0a09,
                0x100f0e0d, 0x3320646e, 0x01040103, 0x06020905,
                0x00000007, 0x00000000, 0x79622d32, 0x14131211,
                0x18171615, 0x1c1b1a19, 0x201f1e1d, 0x6b206574]
        
        OUT = [ 0x58318d3e, 0x0292df4f, 0xa28d8215, 0xa1aca723,
                0x697a34c7, 0xf2f00ba8, 0x63e9b0a1, 0x27250e3a,
                0xb1c7f1f3, 0x62066edc, 0x66d3ccf1, 0xb0365cf3,
                0x091ad09e, 0x64f0c40f, 0xd60d95ea, 0x00be78c9]
        
        return [[IN], OUT]
    

def SALSA_PERMUTATION(r=None): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_cipher = Salsa_permutation("SALSA_PERM", my_input, my_output, nbr_rounds=r)
    return my_cipher    
    



# The Salsa permutation to generate the key stream
class Salsa_keypermutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Salsa internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if represent_mode==0:
            if nbr_rounds ==None: nbr_rounds = 21 # 21st round is used add the initial state to obtain the final key stream
            nbr_layers = 13 # 1 for each of the 12 operations in 1 quarter round
            nbr_words = 16 # Words in the state of Salsa
            nbr_temp_words = 20 # To retain the initial input for adding with final state to obtain the key stream
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["FUNCTION"]

            TW  = [16,17,18,19] # Temporary words to store the internal states
        
            for i in range(1,nbr_rounds+1):  
                if i%2 == 0:
                    # Even rounds are row rounds
                    QR1 = [0, 1, 2, 3]
                    QR2 = [5, 6, 7, 4]
                    QR3 = [10, 11, 8,9]
                    QR4 = [15, 12, 13, 14]
                else:
                    # Odd rounds are column rounds
                    QR1 = [0, 4, 8, 12]
                    QR2 = [5, 9, 13,1]
                    QR3 = [10, 14, 2,6]
                    QR4 = [15, 3, 7, 11]
                
                # In the first round copy the initial word to temporary words
                if i == 1:
                    InIndex = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
                    OutIndex = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
                    S.SingleOperatorLayer("Equal", i, 0, Equal, InIndex, OutIndex)
                else:
                    S.AddIdentityLayer("Identity", i, 0)


                if i == 21:
                    InIndex = [[0, 20], [1, 21], [2, 22], [3, 23], [4, 24], [5, 25], [6, 26], [7, 27], [8, 28], [9, 29], [10, 30], [11, 31], [12, 32], [13, 33], [14, 34], [15, 35]]
                    OutIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, InIndex, OutIndex)
                    for j in range(2, nbr_layers):
                        name = 'Identity' + str(j)
                        S.AddIdentityLayer(name, i, j)
                else:
                    # 4 quarter rounds in parallel
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, [[QR1[0], QR1[3]], [QR2[0], QR2[3]], [QR3[0], QR3[3]], [QR4[0], QR4[3]]], [TW[0], TW[1], TW[2], TW[3]])
                    S.RotationLayer("Rot1", i, 2, [['l', 7, TW[0], TW[0]], ['l', 7, TW[1], TW[1]], ['l', 7, TW[2], TW[2]], ['l', 7, TW[3], TW[3]]])
                    S.SingleOperatorLayer("XOR1", i, 3, XOR, [[TW[0], QR1[1]], [TW[1], QR2[1]], [TW[2], QR3[1]], [TW[3], QR4[1]]], [QR1[1], QR2[1], QR3[1], QR4[1]])

                    S.SingleOperatorLayer("Add2", i, 4, ModAdd, [[QR1[0], QR1[1]], [QR2[0], QR2[1]], [QR3[0], QR3[1]], [QR4[0], QR4[1]]], [TW[0], TW[1], TW[2], TW[3]])
                    S.RotationLayer("Rot2", i, 5, [['l', 9, TW[0], TW[0]], ['l', 9, TW[1], TW[1]], ['l', 9, TW[2], TW[2]], ['l', 9, TW[3], TW[3]]])
                    S.SingleOperatorLayer("XOR2", i, 6, XOR, [[TW[0], QR1[2]], [TW[1], QR2[2]], [TW[2], QR3[2]], [TW[3], QR4[2]]], [QR1[2], QR2[2], QR3[2], QR4[2]])

                    S.SingleOperatorLayer("Add3", i, 7, ModAdd, [[QR1[1], QR1[2]], [QR2[1], QR2[2]], [QR3[1], QR3[2]], [QR4[1], QR4[2]]], [TW[0], TW[1], TW[2], TW[3]])
                    S.RotationLayer("Rot3", i, 8, [['l', 13, TW[0], TW[0]], ['l', 13, TW[1], TW[1]], ['l', 13, TW[2], TW[2]], ['l', 13, TW[3], TW[3]]])
                    S.SingleOperatorLayer("XOR3", i, 9, XOR, [[TW[0], QR1[3]], [TW[1], QR2[3]], [TW[2], QR3[3]], [TW[3], QR4[3]]], [QR1[3], QR2[3], QR3[3], QR4[3]])

                    S.SingleOperatorLayer("Add4", i, 10, ModAdd, [[QR1[2], QR1[3]], [QR2[2], QR2[3]], [QR3[2], QR3[3]], [QR4[2], QR4[3]]], [TW[0], TW[1], TW[2], TW[3]])
                    S.RotationLayer("Rot4", i, 11, [['l', 18, TW[0], TW[0]], ['l', 18, TW[1], TW[1]], ['l', 18, TW[2], TW[2]], ['l', 18, TW[3], TW[3]]])
                    S.SingleOperatorLayer("XOR4", i, 12, XOR, [[TW[0], QR1[0]], [TW[1], QR2[0]], [TW[2], QR3[0]], [TW[3], QR4[0]]], [QR1[0], QR2[0], QR3[0], QR4[0]])

        self.test_vectors = self.gen_test_vectors()

    def gen_test_vectors(self):
        # Test vectors from https://cr.yp.to/snuffle/salsafamily-20071225.pdf

        IN = [  0x61707865, 0x04030201, 0x08070605, 0x0c0b0a09,
                0x100f0e0d, 0x3320646e, 0x01040103, 0x06020905,
                0x00000007, 0x00000000, 0x79622d32, 0x14131211,
                0x18171615, 0x1c1b1a19, 0x201f1e1d, 0x6b206574]
        
        OUT = [ 0xb9a205a3, 0x0695e150, 0xaa94881a, 0xadb7b12c,
                0x798942d4, 0x26107016, 0x64edb1a4, 0x2d27173f,
                0xb1c7f1fa, 0x62066edc, 0xe035fa23, 0xc4496f04,
                0x2131e6b3, 0x810bde28, 0xf62cb407, 0x6bdede3d]
        
        return [[IN], OUT]
    
def SALSA_KEYPERMUTATION(r=None): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_cipher = Salsa_keypermutation("SALSA_KEYPERM", my_input, my_output, nbr_rounds=r)
    return my_cipher    


