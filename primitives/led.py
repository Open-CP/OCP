from primitives.primitives import Permutation, Block_cipher
from operators.Sbox import PRESENT_Sbox
from operators.boolean_operators import XOR
import variables.variables as var

def gen_rounds_constant_table(nbr_rounds, key_size=64):
    rc_values = [
        0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B,
        0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33, 0x27, 0x0E,
        0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30,
        0x21, 0x02, 0x05, 0x0B, 0x17, 0x2E, 0x1C, 0x38,
        0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A,
        0x34, 0x29, 0x12, 0x24, 0x08, 0x11, 0x22, 0x04
    ]
    
    # Split key_size byte into upper and lower nibbles
    upper_ks = (key_size >> 4) & 0xF  # ks7||ks6||ks5||ks4
    lower_ks = key_size & 0xF         # ks3||ks2||ks1||ks0
    
    constant_table = []
    for i in range(1, nbr_rounds + 1):
        rc = rc_values[i-1]
        const = [0] * 8        
        const[0], const[1], const[2], const[3] = 0 ^ upper_ks, 1 ^ upper_ks, 2 ^ lower_ks, 3 ^ lower_ks
        rc0, rc1, rc2, rc3, rc4, rc5 = (rc >> 0) & 0x1, (rc >> 1) & 0x1, (rc >> 2) & 0x1, (rc >> 3) & 0x1, (rc >> 4) & 0x1, (rc >> 5) & 0x1
        const[4] = (rc5 << 2) | (rc4 << 1) | rc3   # 0||rc5||rc4||rc3
        const[5] = (rc2 << 2) | (rc1 << 1) | rc0   # 0||rc2||rc1||rc0
        const[6] = (rc5 << 2) | (rc4 << 1) | rc3   # 0||rc5||rc4||rc3
        const[7] = (rc2 << 2) | (rc1 << 1) | rc0   # 0||rc2||rc1||rc0
        constant_table.append(const)
    
    return constant_table


# The LED internal permutation
class LED_permutation(Permutation):
    def __init__(self, name, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the LED internal permutation
        :param name: Name of the permutation
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds
        :param represent_mode: Integer specifying the mode of representation used for encoding the permutation.
        """

        if nbr_rounds==None: nbr_rounds=32  
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = (4, 16, 0, 4)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        S = self.functions["PERMUTATION"]
        
        constant_table = gen_rounds_constant_table(nbr_rounds, key_size=64)
        shift_rows = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
        mix_columns_matrix = [[4, 1, 2, 2], [8, 6, 5, 6], [11, 14, 10, 9], [2, 2, 15, 11]]
        mix_columns_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        
        # create constraints
        if represent_mode==0:
            for i in range(1, nbr_rounds+1):
                S.AddConstantLayer("AC", i, 0, "xor", [True]*8, constant_table)  # AddConstants layer
                S.SboxLayer("SB", i, 1, PRESENT_Sbox)  # SubCells layer (uses PRESENT S-box)
                S.PermutationLayer("SR", i, 2, shift_rows)  # ShiftRows layer
                S.MatrixLayer("MC", i, 3, mix_columns_matrix, mix_columns_index, "0x3")  # MixColumnsSerial layer (over GF(2^4) with irreducible polynomial x^4 + x + 1)
                

    def gen_test_vectors(self, version = None):
        # Test vectors for LED permutation (same as LED64 with all-zero key)
        IN = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
        OUT = [0x3, 0x4, 0x0, 0xC, 0x9, 0x0, 0x3, 0x7, 0xC, 0x1, 0xA, 0x9, 0x2, 0x0, 0x0, 0x8]
        self.test_vectors.append([[IN], OUT])


def LED_PERMUTATION(r=None, represent_mode=0, copy_operator=False):
    my_input = [var.Variable(4, ID="in"+str(i)) for i in range(16)]
    my_output = [var.Variable(4, ID="out"+str(i)) for i in range(16)]
    my_permutation = LED_permutation("LED_PERM", my_input, my_output, nbr_rounds=r, represent_mode=represent_mode)
    my_permutation.clean_graph()
    if copy_operator: my_permutation.add_copy_operators()
    my_permutation.build_dictionaries()
    my_permutation.gen_test_vectors()
    return my_permutation


# The LED block cipher
class LED_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the LED block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize) - (64, 64) for LED-64 or (64, 128) for LED-128
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        assert version in [[64, 64], [64, 128]], f"LED only supports (64, 64) and (64, 128) versions, got {version}"
        p_bitsize, k_bitsize = version[0], version[1]
        
        # LED-64: 8 steps × 4 rounds = 32 rounds, 9 key additions (8 before each step + 1 final)
        # LED-128: 12 steps × 4 rounds = 48 rounds, 13 key additions (12 before each step + 1 final)
        if nbr_rounds == None: nbr_rounds = 32 if k_bitsize == 64 else 48    
        
        if represent_mode==0:
            (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize) = (5, 16, 0, 4)
            (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize) = (1, k_bitsize // 4, 0, 4)
            (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (1, 16, 0, 4)
            k_nbr_rounds = (nbr_rounds//4)+1 
        
        super().__init__(name, p_input, k_input, c_output, nbr_rounds+1, k_nbr_rounds, 
                        [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], 
                        [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], 
                        [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        S = self.functions["PERMUTATION"]
        KS = self.functions["KEY_SCHEDULE"]
        SK = self.functions["SUBKEYS"]
        
        constant_table = gen_rounds_constant_table(nbr_rounds, key_size=k_bitsize)
        shift_rows = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
        mix_columns_matrix = [[4, 1, 2, 2], [8, 6, 5, 6], [11, 14, 10, 9], [2, 2, 15, 11]]
        mix_columns_index = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        
        # create constraints
        if represent_mode==0:
            # No key schedule, just reuse the key
            for i in range(1, k_nbr_rounds+1): KS.AddIdentityLayer("ID", i, 0)
            
            # Subkeys extraction (alternate between K1 and K2 for LED-128)
            for i in range(1, nbr_rounds+2):
                if i % 4 == 1: 
                    if k_bitsize == 64: SK.ExtractionLayer("SK_EX", i, 0, list(range(16)), KS.vars[((i-1)//4)+1][0])
                    elif k_bitsize == 128:
                        if i % 8 == 1: SK.ExtractionLayer("SK_EX", i, 0, list(range(16)), KS.vars[((i-1)//4)+1][0])
                        elif i % 8 == 5: SK.ExtractionLayer("SK_EX", i, 0, list(range(16, 32)), KS.vars[((i-1)//4)+1][0])
                    
            # Internal permutation
            for i in range(1, nbr_rounds+1):
                # Key addition at the beginning of each step (every 4 rounds, starting with round 1)
                if i % 4 == 1: S.AddRoundKeyLayer("ARK", i, 0, XOR, SK)  # AddRoundKey layer
                else: S.AddIdentityLayer("ID", i, 0)
                S.AddConstantLayer("AC", i, 1, "xor", [True]*8, constant_table)  # AddConstants layer
                S.SboxLayer("SB", i, 2, PRESENT_Sbox)  # SubCells layer
                S.PermutationLayer("SR", i, 3, shift_rows)  # ShiftRows layer
                S.MatrixLayer("MC", i, 4, mix_columns_matrix, mix_columns_index, "0x3")  # MixColumnsSerial layer
                
            # Final key addition
            if (nbr_rounds+1) % 4 == 1: S.AddRoundKeyLayer("ARK", nbr_rounds+1, 0, XOR, SK)  # AddRoundKey layer
            else: S.AddIdentityLayer("ID", nbr_rounds+1, 0)
            S.AddIdentityLayer("ID", nbr_rounds+1, 1)
            S.AddIdentityLayer("ID", nbr_rounds+1, 2)
            S.AddIdentityLayer("ID", nbr_rounds+1, 3)
            S.AddIdentityLayer("ID", nbr_rounds+1, 4)

    def gen_test_vectors(self, version=None):
        # Test vectors from LED paper (https://eprint.iacr.org/2012/600.pdf)
        if version == [64, 64]:
            # LED-64 test vector (all zeros)
            P = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
            K = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
            C = [0x3, 0x4, 0x0, 0xC, 0x9, 0x0, 0x3, 0x7, 0xC, 0x1, 0xA, 0x9, 0x2, 0x0, 0x0, 0x8]
            self.test_vectors.append([[P, K], C])
            
            # Additional LED-64 test vector from paper:
            P = [0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF]
            K = [0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF]
            C = [0xA, 0x5, 0x3, 0xF, 0x0, 0x5, 0x8, 0xC, 0x0, 0x1, 0x9, 0x5, 0x3, 0xE, 0x3, 0x8]
            self.test_vectors.append([[P, K], C])
        elif version == [64, 128]:
            # LED-128 test vector (all zeros)
            P = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
            K = [0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
                 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0]
            C = [0x3, 0xB, 0x8, 0xD, 0xD, 0x2, 0x5, 0xB, 0xE, 0xA, 0x0, 0xA, 0xC, 0x0, 0xC, 0x1]
            self.test_vectors.append([[P, K], C])
            
            # Additional LED-128 test vector from paper:
            P = [0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF]
            K = [0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
                 0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF]
            C = [0xD, 0x2, 0x7, 0x4, 0x6, 0x4, 0xF, 0xF, 0xB, 0x5, 0x0, 0xC, 0x8, 0x8, 0x1, 0x2]
            self.test_vectors.append([[P, K], C])   


def LED_BLOCKCIPHER(r=None, version=[64, 64], represent_mode=0, copy_operator=False):
    """
    LED block cipher
    :param r: Number of rounds (optional)
    :param version: (p_bitsize, k_bitsize) - [64, 64] for LED-64 or [64, 128] for LED-128
    :param represent_mode: Representation mode
    """
    my_plaintext = [var.Variable(4, ID="p"+str(i)) for i in range(16)]
    my_key = [var.Variable(4, ID="k"+str(i)) for i in range(16 if version[1] == 64 else 32)]
    my_ciphertext = [var.Variable(4, ID="c"+str(i)) for i in range(16)]
    my_cipher = LED_block_cipher(f"LED{version[1]}", version, my_plaintext, my_key, my_ciphertext, nbr_rounds=r, represent_mode=represent_mode)
    my_cipher.clean_graph()
    if copy_operator: my_cipher.add_copy_operators()
    my_cipher.build_dictionaries()
    my_cipher.gen_test_vectors(version=version)
    return my_cipher
