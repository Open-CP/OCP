from primitives.primitives import Permutation, Block_cipher
from operators.boolean_operators import XOR, AND, ANDXOR
import variables.variables as var


# The Simon internal permutation
class Simon_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Simon internal permutation
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==32 else 36 if version==48 else 42 if version==64 else 52 if version==96 else 68 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (5, 2, 3, p_bitsize>>1)
        elif represent_mode==1: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (4, 2, 3, p_bitsize>>1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        self.test_vectors = self.gen_test_vectors(version)

        S = self.functions["PERMUTATION"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer
                S.SingleOperatorLayer("AND", i, 1, AND, [[2, 3]], [2]) # bitwise AND layer
                S.SingleOperatorLayer("XOR1", i, 2, XOR, [[1, 2]], [1]) # XOR layer
                S.SingleOperatorLayer("XOR2", i, 3, XOR, [[1, 4]], [1]) # XOR layer
                S.PermutationLayer("PERM", i, 4, [1,0]) # Permutation layer


        elif represent_mode==1:
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer
                S.SingleOperatorLayer("ANDXOR", i, 1, ANDXOR, [[2, 3, 1]], [1]) # bitwise AND-XOR layer
                S.SingleOperatorLayer("XOR", i, 2, XOR, [[1, 4]], [1]) # XOR layer
                S.PermutationLayer("PERM", i, 3, [1,0]) # Permutation layer


    def gen_test_vectors(self, version):
        if version == 32:
            IN = [0x6565, 0x6877]
            OUT = [0xdb3c, 0x569b]
        elif version == 48:
            IN = [0x612067, 0x6e696c]
            OUT = [0x911b84, 0x11c29c]
        elif version == 64:
            IN = [0x6f722067, 0x6e696c63]
            OUT = [0xb3dbca80, 0x840afe75]
        elif version == 96:
            IN = [0x2072616c6c69, 0x702065687420]
            OUT = [0x6b6ccce37858, 0x41aa41637590]
        elif version == 128:
            IN = [0x6373656420737265, 0x6c6c657661727420]
            OUT = [0x10b9a695d8bb2564, 0xcf80a07ebfa62541]
        return [[IN], OUT]


def SIMON_PERMUTATION(r=None, version=32, represent_mode=0):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_permutation = Simon_permutation(f"SIMON{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.clean_graph()
    return my_permutation


# The Simon block cipher
class Simon_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Simon block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(32,64) else 36 if (version[0],version[1])==(48,72) else 36 if (version[0],version[1])==(48,96)  else 42 if (version[0],version[1])==(64,96)  else 44 if (version[0],version[1])==(64,128)  else 52 if (version[0],version[1])==(96,96) else 54 if (version[0],version[1])==(96,144) else 68 if (version[0],version[1])==(128,128) else 69 if (version[0],version[1])==(128,192) else 72 if (version[0],version[1])==(128,256) else None
        if represent_mode==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (6, 2, 3, p_bitsize>>1),  (6, int(2*k_bitsize/p_bitsize), 2, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        if k_nbr_words == 4: k_nbr_layers += 1
        k_nbr_rounds = max(1, nbr_rounds - k_nbr_words + 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        self.test_vectors = self.gen_test_vectors(version)

        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        constant_table = self.gen_rounds_constant_table(version)

        # create constraints
        if represent_mode==0:

            for i in range(1,nbr_rounds+1):
                # subkeys extraction
                if i <= k_nbr_words:
                    SK.ExtractionLayer("SK_EX", i, 0, [(k_nbr_words-i%k_nbr_words)%k_nbr_words], KS.vars[1][0])
                else:
                    SK.ExtractionLayer("SK_EX", i, 0, [0], KS.vars[i-k_nbr_words+1][0])

            for i in range(1,k_nbr_rounds):
                # key schedule
                KS.RotationLayer("K_ROT1", i, 0, ['r', 3, 0, k_nbr_words]) # Rotation layer
                if k_nbr_words == 2 or k_nbr_words == 3:
                    KS.SingleOperatorLayer("K_XOR1", i, 1, XOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words-1]) # XOR layer
                    KS.RotationLayer("K_ROT2", i, 2, ['r', 1, k_nbr_words]) # Rotation layer
                    KS.SingleOperatorLayer("K_XOR2", i, 3, XOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words]) # XOR layer
                    KS.AddConstantLayer("K_C", i, 4, "xor", [True if e==k_nbr_words else None for e in range(KS.nbr_words+KS.nbr_temp_words)], constant_table)  # Constant layer
                    KS.PermutationLayer("K_PERM", i, 5, [k_nbr_words]+[i for i in range(k_nbr_words)]) # Shiftrows layer
                elif k_nbr_words == 4:
                    KS.SingleOperatorLayer("K_XOR1", i, 1, XOR, [[2, 4]], [4]) # XOR layer
                    KS.SingleOperatorLayer("K_XOR2", i, 2, XOR, [[3, 4]], [5]) # XOR layer
                    KS.RotationLayer("K_ROT2", i, 3, ['r', 1, 4]) # Rotation layer
                    KS.SingleOperatorLayer("K_XOR3", i, 4, XOR, [[4, 5]], [4]) # XOR layer
                    KS.AddConstantLayer("K_C", i, 5, "xor", [True if e==k_nbr_words else None for e in range(KS.nbr_words+KS.nbr_temp_words)], constant_table)  # Constant layer
                    KS.PermutationLayer("K_PERM", i, 6, [4,0,1,2]) # Shiftrows layer

            # Internal permutation
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer
                S.SingleOperatorLayer("AND", i, 1, AND, [[2, 3]], [2]) # bitwise AND layer
                S.SingleOperatorLayer("XOR1", i, 2, XOR, [[1, 2]], [1]) # XOR layer
                S.SingleOperatorLayer("XOR2", i, 3, XOR, [[1, 4]], [1]) # XOR layer
                S.AddRoundKeyLayer("ARK", i, 4, XOR, SK, [0,1]) # Addroundkey layer
                S.PermutationLayer("PERM", i, 5, [1,0]) # Permutation layer


    def gen_rounds_constant_table(self, version):
        constant_table = []
        # Z Arrays (stored bit reversed for easier usage)
        z0 = 0b01100111000011010100100010111110110011100001101010010001011111
        z1 = 0b01011010000110010011111011100010101101000011001001111101110001
        z2 = 0b11001101101001111110001000010100011001001011000000111011110101
        z3 = 0b11110000101100111001010001001000000111101001100011010111011011
        z4 = 0b11110111001001010011000011101000000100011011010110011110001011
        z=z0 if (version[0],version[1])==(32,64) else z0 if (version[0],version[1])==(48,72) else z1 if (version[0],version[1])==(48,96)  else z2 if (version[0],version[1])==(64,96)  else z3 if (version[0],version[1])==(64,128)  else z2 if (version[0],version[1])==(96,96) else z3 if (version[0],version[1])==(96,144) else z2 if (version[0],version[1])==(128,128) else z3 if (version[0],version[1])==(128,192) else z4 if (version[0],version[1])==(128,256) else None
        round_constant = (2 ** (version[0] >> 1) - 1) ^ 3
        for i in range(1,self.functions["PERMUTATION"].nbr_rounds+1):
            constant_table.append([round_constant ^ ((z >> ((i-1) % 62)) & 1)])
        return constant_table

    def gen_test_vectors(self, version):
        # test vectors from https://github.com/inmcm/Simon_Speck_Ciphers/blob/master/Python/simonspeckciphers/tests/test_simonspeck.py
        if version == [32, 64]:
            plaintext = [0x6565, 0x6877]
            key = [0x1918, 0x1110, 0x0908, 0x0100]
            ciphertext = [0xc69b, 0xe9bb]
        elif version == [48, 72]:
            plaintext = [0x612067, 0x6e696c]
            key = [0x121110, 0x0a0908, 0x020100]
            ciphertext = [0xdae5ac, 0x292cac]
        elif version == [48, 96]:
            plaintext = [0x726963, 0x20646e]
            key = [0x1a1918, 0x121110, 0x0a0908, 0x020100]
            ciphertext = [0x6e06a5, 0xacf156]
        elif version == [64, 96]:
            plaintext = [0x6f722067, 0x6e696c63]
            key = [0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x5ca2e27f, 0x111a8fc8]
        elif version == [64, 128]:
            plaintext = [0x656b696c, 0x20646e75]
            key = [0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100]
            ciphertext = [0x44c8fc20, 0xb9dfa07a]
        elif version == [96, 96]:
            plaintext = [0x2072616c6c69, 0x702065687420]
            key = [0x0d0c0b0a0908, 0x050403020100]
            ciphertext = [0x602807a462b4, 0x69063d8ff082]
        elif version == [96, 144]:
            plaintext = [0x746168742074, 0x73756420666f]
            key = [0x151413121110, 0x0d0c0b0a0908, 0x050403020100]
            ciphertext =  [0xecad1c6c451e, 0x3f59c5db1ae9]
        elif version == [128, 128]:
            plaintext = [0x6373656420737265, 0x6c6c657661727420]
            key = [0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x49681b1e1e54fe3f, 0x65aa832af84e0bbc]
        elif version == [128, 192]:
            plaintext = [0x206572656874206e, 0x6568772065626972]
            key = [0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0xc4ac61effcdc0d4f, 0x6c9c8d6e2597b85b]
        elif version == [128, 256]:
            plaintext = [0x74206e69206d6f6f, 0x6d69732061207369]
            key = [0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100]
            ciphertext = [0x8d2b5579afc8a3a0, 0x3bf72a87efe7b868]
        return [[plaintext, key], ciphertext]


def SIMON_BLOCKCIPHER(r=None, version=[32,64], represent_mode=0):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = Simon_block_cipher(f"SIMON{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.clean_graph()
    return my_cipher
