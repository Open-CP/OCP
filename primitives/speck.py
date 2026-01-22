from primitives.primitives import Permutation, Block_cipher
from operators.boolean_operators import XOR
from operators.modular_operators import ModAdd
import variables.variables as var


# The Speck internal permutation
class Speck_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Speck internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        # define the parameters
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=22 if version==32 else 22 if version==48 else 26 if version==64 else 28 if version==96 else 32 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (4, 2, 0, p_bitsize>>1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]
        rotr, rotl = (7, 2) if version == 32 else (8, 3)

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT1", i, 0, ['r', rotr, 0]) # Rotation layer
                S.SingleOperatorLayer("ADD", i, 1, ModAdd, [[0,1]], [0]) # Modular addition layer
                S.RotationLayer("ROT2", i, 2, ['l', rotl, 1]) # Rotation layer
                S.SingleOperatorLayer("XOR", i, 3, XOR, [[0,1]], [1]) # XOR layer

    def gen_test_vectors(self, version): # Generated from the block cipher with all-zero subkeys
        if version == 32:
            IN = [0x6574, 0x694c]
            OUT = [0x689d, 0x44b7]
            self.test_vectors.append([[IN], OUT])
        elif version == 48:
            IN = [0x20796c, 0x6c6172]
            OUT = [0x83900c, 0xf96ec6]
            self.test_vectors.append([[IN], OUT])
        elif version == 64:
            IN = [0x74614620, 0x736e6165]
            OUT = [0x9940d993, 0x9fbef5e]
            self.test_vectors.append([[IN], OUT])
        elif version == 96:
            IN = [0x65776f68202c, 0x656761737520]
            OUT = [0x3de90d079e18, 0x6da23cb0723e]
            self.test_vectors.append([[IN], OUT])
        elif version == 128:
            IN = [0x6c61766975716520, 0x7469206564616d20]
            OUT = [0xce1e9101126bf885, 0x1b30637b2ae10e96]
            self.test_vectors.append([[IN], OUT])   

def SPECK_PERMUTATION(r=None, version=32, represent_mode=0, copy_operator=False):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_permutation = Speck_permutation(f"SPECK{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors(version=version)
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


class Speck_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Speck block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        # define the parameters
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=22 if (version[0],version[1])==(32,64) else 22 if (version[0],version[1])==(48,72) else 23 if (version[0],version[1])==(48,96)  else 26 if (version[0],version[1])==(64,96)  else 27 if (version[0],version[1])==(64,128)  else 28 if (version[0],version[1])==(96,96) else 29 if (version[0],version[1])==(96,144) else 32 if (version[0],version[1])==(128,128) else 33 if (version[0],version[1])==(128,192) else 34 if (version[0],version[1])==(128,256) else None
        if represent_mode==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 2, 0, p_bitsize>>1),  (6, int(2*k_bitsize / p_bitsize), 0, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        round_constants = self.gen_rounds_constant_table()
        rotr, rotl = (7, 2) if version[0] == 32 else (8, 3)
        if k_bitsize==p_bitsize: perm, right_k_index, left_k_index = ([0,1], 0, 1)
        elif k_bitsize*2==p_bitsize*3: perm, right_k_index, left_k_index = ([1,0,2], 1, 2)
        elif k_bitsize==2*p_bitsize: perm, right_k_index, left_k_index = ([2,0,1,3], 2, 3)

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        # create constraints
        if represent_mode==0:

            for i in range(1,nbr_rounds+1):
                # subkeys extraction
                SK.ExtractionLayer("SK_EX", i, 0, [left_k_index], KS.vars[i][0])

            for i in range(1,nbr_rounds):
                # key schedule
                KS.RotationLayer("K_ROT1", i, 0, ['r', rotr, right_k_index]) # Rotation layer
                KS.SingleOperatorLayer("K_ADD", i, 1, ModAdd, [[right_k_index, left_k_index]], [right_k_index]) # Modular addition layer
                KS.RotationLayer("K_ROT2", i, 2, ['l', rotl, left_k_index]) # Rotation layer
                KS.AddConstantLayer("K_C", i, 3, "xor", [True if e==right_k_index else None for e in range(KS.nbr_words)], round_constants)  # Constant layer
                KS.SingleOperatorLayer("K_XOR", i, 4, XOR, [[right_k_index, left_k_index]], [left_k_index]) # XOR layer
                KS.PermutationLayer("K_SHIFT", i, 5, perm) # key schedule word shift

            for i in range(1,nbr_rounds+1):
                # Internal permutation
                S.RotationLayer("ROT1", i, 0, ['r', rotr, 0]) # Rotation layer
                S.SingleOperatorLayer("ADD", i, 1, ModAdd, [[0,1]], [0]) # Modular addition layer
                S.RotationLayer("ROT2", i, 2, ['l', rotl, 1]) # Rotation layer
                S.AddRoundKeyLayer("ARK", i, 3, XOR, SK, [1,0]) # Addroundkey layer
                S.SingleOperatorLayer("XOR", i, 4, XOR, [[0,1]], [1]) # XOR layer

    def gen_rounds_constant_table(self):
        constant_table = []
        for i in range(1,self.functions["PERMUTATION"].nbr_rounds+1):
            constant_table.append([i-1])
        return constant_table

    def gen_test_vectors(self, version):
        # test vectors from https://github.com/inmcm/Simon_Speck_Ciphers/blob/master/Python/simonspeckciphers/tests/test_simonspeck.py
        if version == [32, 64]:
            plaintext = [0x6574, 0x694c]
            key = [0x1918, 0x1110, 0x0908, 0x0100]
            ciphertext = [0xa868, 0x42f2]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [48, 72]:
            plaintext = [0x20796c, 0x6c6172]
            key = [0x121110, 0x0a0908, 0x020100]
            ciphertext = [0xc049a5, 0x385adc]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [48, 96]:
            plaintext = [0x6d2073, 0x696874]
            key = [0x1a1918, 0x121110, 0x0a0908, 0x020100]
            ciphertext = [0x735e10, 0xb6445d]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [64, 96]:
            plaintext = [0x74614620, 0x736e6165]
            key = [0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x9f7952ec, 0x4175946c]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [64, 128]:
            plaintext = [0x3b726574, 0x7475432d]
            key = [0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x8c6fa548, 0x454e028b]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [96, 96]:
            plaintext = [0x65776f68202c, 0x656761737520]
            key = [0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x9e4d09ab7178, 0x62bdde8f79aa]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [96, 144]:
            plaintext = [0x656d6974206e, 0x69202c726576]
            key = [0x151413121110, 0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x2bf31072228a, 0x7ae440252ee6]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [128, 128]:
            plaintext = [0x6c61766975716520, 0x7469206564616d20]
            key = [0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0xa65d985179783265, 0x7860fedf5c570d18]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [128, 192]:
            plaintext = [0x7261482066656968, 0x43206f7420746e65]
            key = [0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x1be4cf3a13135566, 0xf9bc185de03c1886]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [128, 256]:
            plaintext = [0x65736f6874206e49, 0x202e72656e6f6f70]
            key = [0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x4109010405c0f53e, 0x4eeeb48d9c188f43]
            self.test_vectors.append([[plaintext, key], ciphertext])
        
def SPECK_BLOCKCIPHER(r=None, version = [32, 64], represent_mode=0, copy_operator=False):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="p"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="c"+str(i)) for i in range(2)]
    my_cipher = Speck_block_cipher(f"SPECK{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher
