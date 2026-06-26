from primitives.primitives import Permutation, Block_cipher
from operators.boolean_operators import XOR, AND
import variables.variables as var


# The Simeck internal permutation
class Simeck_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Simeck internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==32 else 36 if version==48 else 44 if version==64 else None
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (5, 2, 2, p_bitsize>>1)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])

        S = self.functions["PERMUTATION"]

        # create constraints
        for i in range(1,nbr_rounds+1):
            S.RotationLayer("ROT", i, 0, [['l', 5, 0, 2], ['l', 1, 0, 3]]) # Rotation layer
            S.SingleOperatorLayer("AND", i, 1, AND, [[0, 2]], [2]) # bitwise AND layer
            S.SingleOperatorLayer("XOR1", i, 2, XOR, [[1, 2]], [1]) # XOR layer
            S.SingleOperatorLayer("XOR2", i, 3, XOR, [[1, 3]], [1]) # XOR layer
            S.PermutationLayer("PERM", i, 4, [1,0]) # Permutation layer

    def gen_test_vectors(self, version):
        if version == 32:
            IN = [0x6565, 0x6877]
            OUT = [0xda9d, 0xac1f]
            self.test_vectors.append([[IN], OUT])
        elif version == 48:
            IN = [0x726963, 0x20646e]
            OUT = [0x16854b, 0x3c42ee]
            self.test_vectors.append([[IN], OUT])
        elif version == 64:
            IN = [0x656b696c, 0x20646e75]
            OUT = [0x7468a906, 0xfe16a88c]
            self.test_vectors.append([[IN], OUT])


def SIMECK_PERMUTATION(r=None, version=32, represent_mode=0, copy_operator=False):
    p_bitsize, word_size = version, int(version/2)
    my_input, my_output = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_permutation = Simeck_permutation(f"SIMECK{p_bitsize}_PERM", p_bitsize, my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.gen_test_vectors(version=version)
    my_permutation.post_initialization(copy_operator=copy_operator)
    return my_permutation


# The Simeck block cipher
class Simeck_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Simeck block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """

        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(32,64) else 36 if (version[0],version[1])==(48,96) else None
        if represent_mode!=0: raise Exception(f"{self.__class__.__name__}: represent_mode {represent_mode} not existing")
        (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (6, 2, 2, p_bitsize>>1),  (6, int(2*k_bitsize/p_bitsize), 2, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        k_nbr_rounds = max(1, nbr_rounds - k_nbr_words + 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])


        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]

        constant_table = self.gen_rounds_constant_table(version)

        # create constraints
        for i in range(1,nbr_rounds+1):
            # subkeys extraction
            if i <= k_nbr_words:
                SK.ExtractionLayer("SK_EX", i, 0, [(k_nbr_words-i%k_nbr_words)%k_nbr_words], KS.vars[1][0])
            else:
                SK.ExtractionLayer("SK_EX", i, 0, [0], KS.vars[i-k_nbr_words+1][0])

        for i in range(1,k_nbr_rounds):
            # key schedule: new = k3 xor F(k2) xor c_i, then shift to [new, k0, k1, k2]
            KS.RotationLayer("K_ROT", i, 0, [['l', 5, 2, k_nbr_words], ['l', 1, 2, k_nbr_words+1]]) # Rotation layer
            KS.SingleOperatorLayer("K_AND", i, 1, AND, [[2, k_nbr_words]], [k_nbr_words]) # bitwise AND layer
            KS.SingleOperatorLayer("K_XOR1", i, 2, XOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words]) # XOR layer
            KS.SingleOperatorLayer("K_XOR2", i, 3, XOR, [[k_nbr_words, k_nbr_words+1]], [k_nbr_words]) # XOR layer
            KS.AddConstantLayer("K_C", i, 4, "xor", [True if e==k_nbr_words else None for e in range(KS.nbr_words+KS.nbr_temp_words)], constant_table)  # Constant layer
            KS.PermutationLayer("K_PERM", i, 5, [k_nbr_words]+[j for j in range(k_nbr_words-1)]) # key schedule word shift

        # Internal permutation
        for i in range(1,nbr_rounds+1):
            S.RotationLayer("ROT", i, 0, [['l', 5, 0, 2], ['l', 1, 0, 3]]) # Rotation layer
            S.SingleOperatorLayer("AND", i, 1, AND, [[0, 2]], [2]) # bitwise AND layer
            S.SingleOperatorLayer("XOR1", i, 2, XOR, [[1, 2]], [1]) # XOR layer
            S.SingleOperatorLayer("XOR2", i, 3, XOR, [[1, 3]], [1]) # XOR layer
            S.AddRoundKeyLayer("ARK", i, 4, XOR, SK, [0,1]) # Addroundkey layer
            S.PermutationLayer("PERM", i, 5, [1,0]) # Permutation layer

    def gen_rounds_constant_table(self, version):
        constant_table = []
        word_bitsize = version[0] >> 1
        round_constant = (2 ** word_bitsize) - 4
        lfsr = [1, 1, 1, 1, 1]
        for i in range(1,self.functions["PERMUTATION"].nbr_rounds+1):
            constant_table.append([round_constant ^ lfsr[0]])
            new_bit = lfsr[0] ^ lfsr[2]
            lfsr = lfsr[1:] + [new_bit]
        return constant_table

    def gen_test_vectors(self, version):
        if version == [32, 64]:
            plaintext = [0x6565, 0x6877]
            key = [0x1918, 0x1110, 0x0908, 0x0100]
            ciphertext = [0x770d, 0x2c76]
            self.test_vectors.append([[plaintext, key], ciphertext])
        elif version == [48, 96]:
            plaintext = [0x726963, 0x20646e]
            key = [0x1a1918, 0x121110, 0x0a0908, 0x020100]
            ciphertext = [0xf3cf25, 0xe33b36]
            self.test_vectors.append([[plaintext, key], ciphertext])


def SIMECK_BLOCKCIPHER(r=None, version=[32,64], represent_mode=0, copy_operator=False):
    p_bitsize, k_bitsize, word_size, m = version[0], version[1], int(version[0]/2), int(2*version[1]/version[0])
    my_plaintext, my_key, my_ciphertext = [var.Variable(word_size,ID="in"+str(i)) for i in range(2)], [var.Variable(word_size,ID="k"+str(i)) for i in range(m)], [var.Variable(word_size,ID="out"+str(i)) for i in range(2)]
    my_cipher = Simeck_block_cipher(f"SIMECK{p_bitsize}_{k_bitsize}", [p_bitsize, k_bitsize], my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.gen_test_vectors(version=version)
    my_cipher.post_initialization(copy_operator=copy_operator)
    return my_cipher
