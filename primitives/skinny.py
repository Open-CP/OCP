from primitives.primitives import Permutation, Block_cipher
import operators.operators as op

# The Skinny internal permutation       
class Skinny_permutation(Permutation):
    def __init__(self, name, version, s_input, s_output, nbr_rounds=None, represent_mode=0):
        """
        Initialize the Skinny internal permutation.
        :param name: Name of the permutation
        :param version: Bit size of the permutation (e.g., 64 or 128)
        :param s_input: Input state
        :param s_output: Output state
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        # define the parameters
        p_bitsize = version
        if nbr_rounds==None: nbr_rounds=32 if version==64 else 40 if version==128 else None
        if represent_mode==0: nbr_layers, nbr_words, nbr_temp_words, word_bitsize = 4, 16, 0, int(p_bitsize/16)
        super().__init__(name, s_input, s_output, nbr_rounds, [nbr_layers, nbr_words, nbr_temp_words, word_bitsize])
        round_constants = self.gen_rounds_constant_table()
        sbox = op.Skinny_4bit_Sbox if word_bitsize==4 else op.Skinny_8bit_Sbox 

        S = self.states["STATE"]

        # create constraints
        if represent_mode==0: 
            for i in range(1,nbr_rounds+1):              
                S.SboxLayer("SB", i, 0, sbox)           
                S.AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer            
                S.PermutationLayer("SR", i, 2, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                S.MatrixLayer("MC", i, 3, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer
    
    def gen_rounds_constant_table(self):
        constant_table = []
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                    0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                    0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                    0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                    0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
            
        for i in range(1,self.states["STATE"].nbr_rounds+1):              
            rc = round_constants[i-1]
            c0, c1, c2 = rc & 0xF, rc >> 4, 0x2     
            constant_table.append([c0,c1,c2])
        return constant_table


# The Skinny block cipher        
class Skinny_block_cipher(Block_cipher):
    def __init__(self, name, version, p_input, k_input, c_output, nbr_rounds=None, represent_mode=0):
        """
        Initializes the Skinny block cipher.
        :param name: Cipher name
        :param version: (p_bitsize, k_bitsize), e.g., (64, 128)
        :param p_input: Plaintext input
        :param k_input: Key input
        :param c_output: Ciphertext output
        :param nbr_rounds: Number of rounds (optional)
        :param represent_mode: Integer specifying the mode of representation used for encoding the cipher.
        """
        
        # define the parameters
        p_bitsize, k_bitsize = version[0], version[1]
        if nbr_rounds==None: nbr_rounds=32 if (version[0],version[1])==(64,64) else 36 if (version[0],version[1])==(64,128) else 40 if (version[0],version[1])==(64,192)  else 40 if (version[0],version[1])==(128,128)  else 48 if (version[0],version[1])==(128,256)  else 56 if (version[0],version[1])==(128,384) else None
        self.tweak_size = int(k_bitsize/p_bitsize)
        k_nbr_rounds = nbr_rounds if self.tweak_size == 1 else nbr_rounds + 1
        if represent_mode==0: 
            if self.tweak_size ==1: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (1, int(16*k_bitsize / p_bitsize), 0, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif self.tweak_size == 2: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (3, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            elif self.tweak_size ==3: (s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize), (k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize), (sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize) = (5, 16, 0, int(p_bitsize/16)), (5, int(16*k_bitsize / p_bitsize), 8, int(p_bitsize/16)), (1, 8, 0, int(p_bitsize/16))
            k_perm_T = [i + 16 * j for j in range(self.tweak_size) for i in [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]]    
            if s_word_bitsize == 4:
                mat1 = [[1,None],[2,None],[3,None],[0,1]]
                mat2 = [0,3],[0,None],[1,None],[2,None]
            elif s_word_bitsize == 8:
                mat1 = [[1,None],[2,None],[3,None],[4,None],[5,None],[6,None],[7,None],[0,2]]
                mat2 = [[1,7],[0,None],[1,None],[2,None],[3,None],[4,None],[5,None],[6,None]]                
        super().__init__(name, p_input, k_input, c_output, nbr_rounds, k_nbr_rounds, [s_nbr_layers, s_nbr_words, s_nbr_temp_words, s_word_bitsize], [k_nbr_layers, k_nbr_words, k_nbr_temp_words, k_word_bitsize], [sk_nbr_layers, sk_nbr_words, sk_nbr_temp_words, sk_word_bitsize])
        round_constants = self.gen_rounds_constant_table()
        sbox = op.Skinny_4bit_Sbox if s_word_bitsize == 4 else op.Skinny_8bit_Sbox
        if self.tweak_size >= 2: self.states_implementation_order = ["KEY_STATE", "SUBKEYS", "STATE"] 
        
        # create constraints
        if represent_mode==0: 

            # Subkey extraction
            for i in range(1,nbr_rounds+1): 
                if self.tweak_size == 1:
                    SK.ExtractionLayer("SK_EX", i, 0, [i for i in range(8)], KS.vars[i][0])
                elif self.tweak_size >= 2:
                    SK.ExtractionLayer("SK_EX", i, 0, [i for i in range(self.tweak_size*16,self.tweak_size*16+8)], KS.vars[i+1][0])
                
            # Key schedule
            if self.tweak_size == 1:
                for i in range(1, k_nbr_rounds): 
                    KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
            elif self.tweak_size == 2: 
                for i in range(1, k_nbr_rounds): 
                    if i == 1: 
                        KS.AddIdentityLayer("ID", i, 1)     # Identity layer 
                    else:
                        KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        KS.SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                    KS.SingleOperatorLayer("K_XOR", i, 2, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(32,40)]) # XOR layer 
            elif self.tweak_size == 3:
                for i in range(1, k_nbr_rounds): 
                    if i == 1: 
                        KS.AddIdentityLayer("ID", i, 0)     # Identity layer 
                        KS.AddIdentityLayer("ID", i, 1)     # Identity layer
                        KS.AddIdentityLayer("ID", i, 2)     # Identity layer 
                    else:
                        KS.PermutationLayer("K_P", i, 0, k_perm_T) # Permutation layer
                        KS.SingleOperatorLayer("K_XOR1", i, 1, op.bitwiseXOR, [[j,j] for j in range(16,24)], [j for j in range(16,24)], mat=mat1)
                        KS.SingleOperatorLayer("K_XOR2", i, 2, op.bitwiseXOR, [[j,j] for j in range(32,40)], [j for j in range(32,40)], mat=mat2)    
                    KS.SingleOperatorLayer("K_XOR", i, 3, op.bitwiseXOR, [[j,16+j] for j in range(8)], [j for j in range(48,56)]) # XOR layer 
                    KS.SingleOperatorLayer("K_XOR", i, 4, op.bitwiseXOR, [[j,16+j] for j in range(32,40)], [j for j in range(48,56)]) # XOR layer 
                    
            # Internal permutation
            for i in range(1,nbr_rounds+1):  
                S.SboxLayer("SB", i, 0, sbox) # Sbox layer                        
                S.AddConstantLayer("C", i, 1, "xor", [True,None,None,None, True,None,None,None, True], round_constants)  # Constant layer            
                S.AddRoundKeyLayer("ARK", i, 2, op.bitwiseXOR, SK, mask=[1 for i in range(8)])  # AddRoundKey layer   
                S.PermutationLayer("SR", i, 3, [0,1,2,3, 7,4,5,6, 10,11,8,9, 13,14,15,12]) # Shiftrows layer
                S.MatrixLayer("MC", i, 4, [[1,0,1,1], [1,0,0,0], [0,1,1,0], [1,0,1,0]], [[0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15]])  #Mixcolumns layer

        # Generate python and c code for round function if unrolled
        self.rounds_code_if_unrolled()

    def rounds_code_if_unrolled(self):
        if (self.tweak_size == 2 or self.tweak_size == 3) and self.states["KEY_STATE"].nbr_rounds >= 2: 
            self.rounds_python_code_if_unrolled = {"KEY_STATE": [[1, "if i == 0:"], [2, "else:"]]}
            self.rounds_c_code_if_unrolled = {"KEY_STATE": [[1, "if (i == 0)"+"{"], [2, "else{"]]}
    
    def gen_rounds_constant_table(self):
        constant_table = []
        round_constants = [0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3E, 0x3D, 0x3B, 0x37, 0x2F, 0x1E, 0x3C, 0x39, 0x33,
                                    0x27, 0x0E, 0x1D, 0x3A, 0x35, 0x2B, 0x16, 0x2C, 0x18, 0x30, 0x21, 0x02, 0x05, 0x0B,
                                    0x17, 0x2E, 0x1C, 0x38, 0x31, 0x23, 0x06, 0x0D, 0x1B, 0x36, 0x2D, 0x1A, 0x34, 0x29,
                                    0x12, 0x24, 0x08, 0x11, 0x22, 0x04, 0x09, 0x13, 0x26, 0x0c, 0x19, 0x32, 0x25, 0x0a,
                                    0x15, 0x2a, 0x14, 0x28, 0x10, 0x20]
            
        for i in range(1,self.states["STATE"].nbr_rounds+1):              
            rc = round_constants[i-1]
            c0, c1, c2 = rc & 0xF, rc >> 4, 0x2     
            constant_table.append([c0,c1,c2])
        return constant_table