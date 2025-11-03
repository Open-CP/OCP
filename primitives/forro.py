from primitives.primitives import Permutation
from operators.modular_operators import ModAdd
from operators.boolean_operators import XOR
from operators.operators import Equal
import variables.variables as var


# The Forro internal permutation
class Forro_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_subrounds=None, represent_mode=0):
        """
        Initialize the Forro internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_subrounds: Number of subrounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if represent_mode==0:
            if nbr_subrounds ==None: nbr_subrounds = 14*4
            nbr_layers = 12 # 1 for each of the 12 operations in 1 subround round
            nbr_words = 16 # Words in the state of Forro
            nbr_temp_words = 0
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_subrounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["FUNCTION"]
        
            for i in range(1,nbr_subrounds +1):  
                if i%8 == 1:
                    SR = [0, 4, 8, 12, 3]
                elif i%8 ==2:
                    SR = [1, 5, 9, 13, 0]
                elif i%8 ==3:
                    SR = [2, 6, 10, 14, 1]
                elif i%8 ==4:
                    SR = [3, 7, 11, 15, 2]
                elif i%8 ==5:
                    SR = [0, 5, 10, 15, 3]
                elif i%8 ==6:
                    SR = [1, 6, 11, 12, 0]
                elif i%8 ==7:
                    SR = [2, 7, 8, 13, 1]
                else:
                    SR = [3, 4, 9, 14, 2]
                
                
                S.SingleOperatorLayer("Add1", i, 0, ModAdd, [[SR[3], SR[4]]], [SR[3]])
                S.SingleOperatorLayer("XOR1", i, 1, XOR, [[SR[2], SR[3]]], [SR[2]] )
                S.SingleOperatorLayer("Add2", i, 2, ModAdd, [[SR[1], SR[2]]], [SR[1]])
                S.RotationLayer("Rot1", i, 3, [['l', 10, SR[1], SR[1]]])
                S.SingleOperatorLayer("Add3", i, 4, ModAdd, [[SR[1], SR[0]]], [SR[0]])
                S.SingleOperatorLayer("XOR2", i, 5, XOR, [[SR[0], SR[4]]], [SR[4]] )

                S.SingleOperatorLayer("Add4", i, 6, ModAdd, [[SR[4], SR[3]]], [SR[3]])
                S.RotationLayer("Rot2", i, 7, [['l', 27, SR[3], SR[3]]])
                S.SingleOperatorLayer("Add5", i, 8, ModAdd, [[SR[3], SR[2]]], [SR[2]])
                S.SingleOperatorLayer("XOR3", i, 9, XOR, [[SR[2], SR[1]]], [SR[1]] )
                S.SingleOperatorLayer("Add6", i, 10, ModAdd, [[SR[1], SR[0]]], [SR[0]])
                S.RotationLayer("Rot3", i, 11, [['l', 8, SR[0], SR[0]]])



        
        self.test_vectors = self.gen_test_vectors()


    def gen_test_vectors(self):
        # Test vectors from https://github.com/murcoutinho/forro_cipher/blob/main/test/test_ref.c

        IN = [  0x686e696d, 0x69762061, 0x65206164, 0x646e6120,
                0x00000000, 0x00000000, 0x746c6f76, 0x61616461,
                0x70207261, 0x6520726f, 0x20657473, 0x73696170,
                0x74736f6d, 0x61206f72, 0x72626173, 0x61636e61]
        
        OUT = [ 0xf9fe4058, 0x45dc7391, 0x5075018e, 0xf7eb3f6d,
                0x25821062, 0x11334ef1, 0x06d33da0, 0x9c9f3bed,
                0x1e167e5f, 0x4d289ed3, 0x77dd96f8, 0x47d21a6b,
                0x6382742c, 0xc7cfac37, 0xd42a0926, 0x901b01f0]
        
        return [[IN], OUT]
    

def FORRO_PERMUTATION(r=None): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_cipher = Forro_permutation("Forro_PERM", my_input, my_output, nbr_subrounds=r)
    return my_cipher    
    





# The Forro permutation to generate the key stream
class Forro_keypermutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_subrounds=None, represent_mode=0):
        """
        Initialize the Forro internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_subrounds: Number of subrounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        if represent_mode==0:
            if nbr_subrounds ==None: nbr_subrounds = 14*4+1 # the last round is used to add the initial state to obtain the final key stream
            nbr_layers = 13 # 1 for each of the 12 operations in 1 subround round
            nbr_words = 16 # Words in the state of forro
            nbr_temp_words = 16 # To retain the initial input for adding with final state to obtain the key stream
            word_bitsize = 32
            super().__init__(name, s_input, s_output, nbr_subrounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
            S = self.functions["FUNCTION"]
        
            for i in range(1,nbr_subrounds +1):  
                if i%8 == 1:
                    SR = [0, 4, 8, 12, 3]
                elif i%8 ==2:
                    SR = [1, 5, 9, 13, 0]
                elif i%8 ==3:
                    SR = [2, 6, 10, 14, 1]
                elif i%8 ==4:
                    SR = [3, 7, 11, 15, 2]
                elif i%8 ==5:
                    SR = [0, 5, 10, 15, 3]
                elif i%8 ==6:
                    SR = [1, 6, 11, 12, 0]
                elif i%8 ==7:
                    SR = [2, 7, 8, 13, 1]
                else:
                    SR = [3, 4, 9, 14, 2]
                
                # In the first round copy the initial word to temporary words
                if i == 1:
                    InIndex = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
                    OutIndex = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
                    S.SingleOperatorLayer("Equal", i, 0, Equal, InIndex, OutIndex)
                else:
                    S.AddIdentityLayer("Identity", i, 0)


                if i == 14*4+1:
                    InIndex = [[0, 16], [1, 17], [2, 18], [3, 19], [4, 20], [5, 21], [6, 22], [7, 23], [8, 24], [9, 25], [10, 26], [11, 27], [12, 28], [13, 29], [14, 30], [15, 31]]
                    OutIndex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, InIndex, OutIndex)
                    for j in range(2, nbr_layers):
                        name = 'Identity' + str(j)
                        S.AddIdentityLayer(name, i, j)
                else:
                    S.SingleOperatorLayer("Add1", i, 1, ModAdd, [[SR[3], SR[4]]], [SR[3]])
                    S.SingleOperatorLayer("XOR1", i, 2, XOR, [[SR[2], SR[3]]], [SR[2]] )
                    S.SingleOperatorLayer("Add2", i, 3, ModAdd, [[SR[1], SR[2]]], [SR[1]])
                    S.RotationLayer("Rot1", i, 4, [['l', 10, SR[1], SR[1]]])
                    S.SingleOperatorLayer("Add3", i, 5, ModAdd, [[SR[1], SR[0]]], [SR[0]])
                    S.SingleOperatorLayer("XOR2", i, 6, XOR, [[SR[0], SR[4]]], [SR[4]] )

                    S.SingleOperatorLayer("Add4", i, 7, ModAdd, [[SR[4], SR[3]]], [SR[3]])
                    S.RotationLayer("Rot2", i, 8, [['l', 27, SR[3], SR[3]]])
                    S.SingleOperatorLayer("Add5", i, 9, ModAdd, [[SR[3], SR[2]]], [SR[2]])
                    S.SingleOperatorLayer("XOR3", i, 10, XOR, [[SR[2], SR[1]]], [SR[1]] )
                    S.SingleOperatorLayer("Add6", i, 11, ModAdd, [[SR[1], SR[0]]], [SR[0]])
                    S.RotationLayer("Rot3", i, 12, [['l', 8, SR[0], SR[0]]])

        self.test_vectors = self.gen_test_vectors()

    def gen_test_vectors(self):
        # Test vectors from https://github.com/murcoutinho/forro_cipher/blob/main/test/test_ref.c
        IN = [  0x686e696d, 0x69762061, 0x65206164, 0x646e6120,
                0x00000000, 0x00000000, 0x746c6f76, 0x61616461,
                0x70207261, 0x6520726f, 0x20657473, 0x73696170,
                0x74736f6d, 0x61206f72, 0x72626173, 0x61636e61]
        
        OUT = [0x626ca9c5, 0xaf5293f2, 0xb59562f2, 0x5c59a08d,
               0x25821062, 0x11334ef1, 0x7b3fad16, 0xfe00a04e,
               0x8e36f0c0, 0xb2491142, 0x98430b6b, 0xbb3b7bdb,
               0xd7f5e399, 0x28f01ba9, 0x468c6a99, 0xf17e7051]
        
        return [[IN], OUT]   


def FORRO_KEYPERMUTATION(r=None): 
    my_input, my_output = [var.Variable(32,ID="in"+str(i)) for i in range(16)], [var.Variable(32,ID="out"+str(i)) for i in range(16)]
    my_cipher = Forro_keypermutation("Forro_KEYPERM", my_input, my_output, nbr_subrounds=r)
    return my_cipher     





