from primitives.primitives import Permutation, Block_cipher
import operators.operators as op


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
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
                
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==32 else 36 if version==48 else 42 if version==64 else 52 if version==96 else 68 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 5, 2, 3, p_bitsize>>1
        elif represent_mode==1: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 2, 3, p_bitsize>>1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        S = self.states["STATE"]

        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):                
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer 
                S.SingleOperatorLayer("AND", i, 1, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                S.SingleOperatorLayer("XOR1", i, 2, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                S.SingleOperatorLayer("XOR2", i, 3, op.bitwiseXOR, [[1, 4]], [1]) # XOR layer 
                S.PermutationLayer("PERM", i, 4, [1,0]) # Permutation layer
       

        elif represent_mode==1:
            for i in range(1,nbr_rounds+1):         
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer 
                S.SingleOperatorLayer("ANDXOR", i, 1, op.bitwiseANDXOR, [[2, 3, 1]], [1]) # bitwise AND-XOR layer
                S.SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[1, 4]], [1]) # XOR layer 
                S.PermutationLayer("PERM", i, 3, [1,0]) # Permutation layer

                
# The Simon block cipher 
# Test vector for simon32_64: plaintext = [0x6565, 0x6877], key = [0x1918, 0x1110, 0x0908, 0x0100], ciphertext = ['0xc69b', '0xe9bb']
# Test vector for simon48_72: plaintext = [0x612067, 0x6e696c], key = [0x121110, 0x0a0908, 0x020100], ciphertext = ['0xdae5ac', '0x292cac']
# Test vector for simon48_96:# plaintext = [0x726963, 0x20646e], key = [0x1a1918, 0x121110, 0x0a0908, 0x020100], ciphertext = ['0x6e06a5', '0xacf156']
# Test vector for simon64_96: plaintext = [0x6f722067, 0x6e696c63], key = [0x13121110, 0x0b0a0908, 0x03020100], ciphertext = ['0x5ca2e27f', '0x111a8fc8']
# Test vector for simon64_128: plaintext = [0x656b696c, 0x20646e75], key = [0x1b1a1918, 0x13121110, 0x0b0a0908, 0x03020100], ciphertext = ['0x44c8fc20', '0xb9dfa07a']
# Test vectors for simon96_96: plaintext = [0x2072616c6c69, 0x702065687420], key = [0x0d0c0b0a0908, 0x050403020100], ciphertext = ['0x602807a462b4', '0x69063d8ff082']
# Test vectors for simon96_144: plaintext = [0x746168742074, 0x73756420666f], key = [0x151413121110, 0x0d0c0b0a0908, 0x050403020100], ciphertext =  ['0xecad1c6c451e', '0x3f59c5db1ae9']
# Test vectors for simon128_128: plaintext = [0x6373656420737265, 0x6c6c657661727420], key = [0x0f0e0d0c0b0a0908, 0x0706050403020100], ciphertext = ['0x49681b1e1e54fe3f', '0x65aa832af84e0bbc']
# Test vectors for simon128_192: plaintext = [0x206572656874206e, 0x6568772065626972], key = [0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100], ciphertext = ['0xc4ac61effcdc0d4f', '0x6c9c8d6e2597b85b']
# Test vectors for simon128_256: plaintext = [0x74206e69206d6f6f, 0x6d69732061207369], key = [0x1f1e1d1c1b1a1918, 0x1716151413121110, 0x0f0e0d0c0b0a0908, 0x0706050403020100], ciphertext = ['0x8d2b5579afc8a3a0', '0x3bf72a87efe7b868']
# https://github.com/inmcm/Simon_Speck_Ciphers/blob/master/Python/simonspeckciphers/tests/test_simonspeck.py

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
        
        S = self.states["STATE"]
        KS = self.states["KEY_STATE"]
        SK = self.states["SUBKEYS"]

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
                KS.RotationLayer("ROT1", i, 0, ['r', 3, 0, k_nbr_words]) # Rotation layer
                if k_nbr_words == 2 or k_nbr_words == 3:
                    KS.SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words-1]) # XOR layer 
                    KS.RotationLayer("ROT2", i, 2, ['r', 1, k_nbr_words]) # Rotation layer 
                    KS.SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words]) # XOR layer 
                    KS.AddConstantLayer("C", i, 4, "xor", [True if e==k_nbr_words else None for e in range(KS.nbr_words+KS.nbr_temp_words)], constant_table)  # Constant layer
                    KS.PermutationLayer("PERM", i, 5, [k_nbr_words]+[i for i in range(k_nbr_words)]) # Shiftrows layer
                elif k_nbr_words == 4:
                    KS.SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[2, 4]], [4]) # XOR layer 
                    KS.SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[3, 4]], [5]) # XOR layer 
                    KS.RotationLayer("ROT2", i, 3, ['r', 1, 4]) # Rotation layer 
                    KS.SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[4, 5]], [4]) # XOR layer 
                    KS.AddConstantLayer("C", i, 5, "xor", [True if e==k_nbr_words else None for e in range(KS.nbr_words+KS.nbr_temp_words)], constant_table)  # Constant layer
                    KS.PermutationLayer("PERM", i, 6, [4,0,1,2]) # Shiftrows layer           
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                S.RotationLayer("ROT", i, 0, [['l', 1, 0, 2], ['l', 8, 0, 3], ['l', 2, 0, 4]]) # Rotation layer 
                S.SingleOperatorLayer("AND", i, 1, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                S.SingleOperatorLayer("XOR1", i, 2, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                S.SingleOperatorLayer("XOR2", i, 3, op.bitwiseXOR, [[1, 4]], [1]) # XOR layer 
                S.AddRoundKeyLayer("ARK", i, 4, op.bitwiseXOR, SK, [0,1]) # Addroundkey layer 
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
        for i in range(1,self.states["STATE"].nbr_rounds+1):    
            constant_table.append([round_constant ^ ((z >> ((i-1) % 62)) & 1)])
        return constant_table

