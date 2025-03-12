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
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 7, 2, 2, p_bitsize>>1
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        
        # create constraints
        if represent_mode==0:
            for i in range(1,nbr_rounds+1):                
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], [0], index_out=[2]) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], [0], index_out=[3]) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], [0], index_out=[2]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].PermutationLayer("PERM", i, 6, [1,0]) # Permutation layer
       

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
        if represent_mode==0: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (8, 2, 2, p_bitsize>>1),  (6, int(2*k_bitsize/p_bitsize), 2, p_bitsize>>1),  (1, 1, 0, p_bitsize>>1)
        if k_nbr_words == 4: k_nbr_layers += 1
        k_nbr_rounds = max(1, nbr_rounds - k_nbr_words + 1)
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        
        constant_table = self.gen_rounds_constant_table(version)

        # create constraints
        if represent_mode==0:
            
            for i in range(1,nbr_rounds+1):    
                # subkeys extraction
                if i <= k_nbr_words:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [(k_nbr_words-i%k_nbr_words)%k_nbr_words], self.states["KEY_STATE"].vars[1][0])
                else:
                    self.states["SUBKEYS"].ExtractionLayer("SK_EX", i, 0, [0], self.states["KEY_STATE"].vars[i-k_nbr_words+1][0])
                                
            for i in range(1,k_nbr_rounds): 
                # key schedule
                self.states["KEY_STATE"].RotationLayer("ROT1", i, 0, ['r', 3], [0], [k_nbr_words]) # Rotation layer
                if k_nbr_words == 2 or k_nbr_words == 3:
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words-1]) # XOR layer 
                    self.states["KEY_STATE"].RotationLayer("ROT2", i, 2, ['r', 1], [k_nbr_words], [k_nbr_words]) # Rotation layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 3, op.bitwiseXOR, [[k_nbr_words-1, k_nbr_words]], [k_nbr_words]) # XOR layer 
                    self.states["KEY_STATE"].AddConstantLayer("C", i, 4, "xor", [True if e==k_nbr_words else None for e in range(self.states["KEY_STATE"].nbr_words+self.states["KEY_STATE"].nbr_temp_words)], constant_table)  # Constant layer
                    self.states["KEY_STATE"].PermutationLayer("PERM", i, 5, [k_nbr_words]+[i for i in range(k_nbr_words)]) # Shiftrows layer
                elif k_nbr_words == 4:
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 1, op.bitwiseXOR, [[2, 4]], [4]) # XOR layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 2, op.bitwiseXOR, [[3, 4]], [5]) # XOR layer 
                    self.states["KEY_STATE"].RotationLayer("ROT2", i, 3, ['r', 1], [4], [4]) # Rotation layer 
                    self.states["KEY_STATE"].SingleOperatorLayer("XOR", i, 4, op.bitwiseXOR, [[4, 5]], [4]) # XOR layer 
                    self.states["KEY_STATE"].AddConstantLayer("C", i, 5, "xor", [True if e==k_nbr_words else None for e in range(self.states["KEY_STATE"].nbr_words+self.states["KEY_STATE"].nbr_temp_words)], constant_table)  # Constant layer
                    self.states["KEY_STATE"].PermutationLayer("PERM", i, 6, [4,0,1,2]) # Shiftrows layer           
            
            # Internal permutation
            for i in range(1,nbr_rounds+1):
                self.states["STATE"].RotationLayer("ROT1", i, 0, ['l', 1], [0], [2]) # Rotation layer 
                self.states["STATE"].RotationLayer("ROT2", i, 1, ['l', 8], [0], [3]) # Rotation layer 
                self.states["STATE"].SingleOperatorLayer("AND", i, 2, op.bitwiseAND, [[2, 3]], [2]) # bitwise AND layer   
                self.states["STATE"].SingleOperatorLayer("XOR1", i, 3, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].RotationLayer("ROT3", i, 4, ['l', 2], [0], [2]) # Rotation layer
                self.states["STATE"].SingleOperatorLayer("XOR2", i, 5, op.bitwiseXOR, [[1, 2]], [1]) # XOR layer 
                self.states["STATE"].AddRoundKeyLayer("ARK", i, 6, op.bitwiseXOR, self.states["SUBKEYS"], [0,1]) # Addroundkey layer 
                self.states["STATE"].PermutationLayer("PERM", i, 7, [1,0]) # Permutation layer

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

